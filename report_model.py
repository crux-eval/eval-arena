import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go

from arena import model_table, pass1_to_battle, battle_summary

def fig_diff_vs_sum(bmname: str, diffvsum: pd.DataFrame):
    figs = px.scatter(diffvsum, x=diffvsum['diff'].abs(), y='sum',
                      custom_data=['model_a', 'model_b', 'sum', 'diff', 'pvalue', 'std_count', 'accA', 'accB'])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.3f})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.3f})", 
        "A + B: %{customdata[2]}", 
        "A - B: %{customdata[3]}", 
        "p-value: %{customdata[4]:.4f}", 
        "std(A-B): %{customdata[5]:.2f}", 
        ])  + '<extra></extra>')

    maxy = diffvsum['sum'].max()
    refs = []
    data_sz = diffvsum.iloc[0]['total']
    x = np.linspace(0, data_sz / 2, 100)
    refs.append(pd.DataFrame({'x': x, 'y': x, 'type': 'x=y'}))
    for alpha in [0.05, 0.1, 0.2]:
        thres = stats.chi2.ppf(1-alpha, 1)
        y = np.linspace(1, maxy, 200)
        refs.append(pd.DataFrame({'x': 1 + np.sqrt(y * thres), 'y': y, 'type': f'pvalue={alpha}'}))
    
    df_ref = pd.concat(refs, axis=0)
    figl = px.line(df_ref, x='x', y='y', color='type')
    figl.update_layout(hovermode=False)

    fig = go.Figure(data=figl.data + figs.data)
    fig.update_layout(
        width=800, height=600, title=bmname,
        xaxis_title="|#A_win - #B_win|",
        yaxis_title="#A_win + #B_win"
    )
    return fig

def fig_pvalue_vs_diff(bmname: str, diffvsum: pd.DataFrame):
    figs = px.scatter(diffvsum, x=(diffvsum['accA'] - diffvsum['accB']).abs(), y='pvalue',
            custom_data=['model_a', 'model_b', 'accA', 'accB', 'pvalue', 'std_acc'])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]}",
        "Model B: %{customdata[1]}", 
        "acc(A): %{customdata[2]:.3f}", 
        "acc(B): %{customdata[3]:.3f}", 
        "p-value: %{customdata[4]:.4f}", 
        "std(acc(A)-acc(B)): %{customdata[5]:.4f}", 
        ])  + '<extra></extra>')
    
    figs.update_layout(
        width=800, height=800,
        title=bmname,
        xaxis_title="acc(Model A) - acc(Model B)|",
        yaxis_title="p-value",
    )
    return figs

def fig_accs_and_pvalues(bmname, diffvsum):
    figs = px.scatter(diffvsum, x='accA', y='accB',
            color='pvalue', range_color=[0, 0.2],
            custom_data=['model_a', 'model_b', 'accA', 'accB', 'pvalue', 'std_acc'])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]}",
        "Model B: %{customdata[1]}", 
        "acc(A): %{customdata[2]:.3f}", 
        "acc(B): %{customdata[3]:.3f}", 
        "p-value: %{customdata[4]:.4f}", 
        "std(acc(A)-acc(B)): %{customdata[5]:.4f}", 
        ])  + '<extra></extra>')
    
    figs.update_layout(
        width=800, height=800,
        title=bmname,
        xaxis_title="acc(Model A)",
        yaxis_title="acc(Model B)",
        legend_title='p_value',
    )
    return figs


def get_sections(result: pd.DataFrame, benchmark_id):
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]
    summary = battle_summary(battles)
    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(benchmark_id, summary).to_html(full_html=False),
        "fig_pvalue_vs_diff": fig_pvalue_vs_diff(benchmark_id, summary).to_html(full_html=False),
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(full_html=False),
        "model_table": model_table(battles_no_ties, result).to_html(
            index=False,
            formatters={
                'pass1': '{:.1%}'.format,
                'std': '{:.2%}'.format,
                'win_rate': '{:.1%}'.format,
                'elo': '{:.1f}'.format
        }),
    }
    return sections


def gen_model_report(benchmark_id: str, benchmark_results, OUTPUT_PATH):
    sections = get_sections(benchmark_results, benchmark_id)
    from jinja2 import Template
    template_path=r"templates/template_model.html"
    output_path = f"{OUTPUT_PATH}/model_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'sections': sections}))

