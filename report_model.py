from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template

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

def fig_noise_character(bmname: str, diffvsum: pd.DataFrame):
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
        "|acc(A)-acc(B)|: %{x:.3f}", 
        "p-value: %{customdata[4]:.4f}", 
        ])  + '<extra></extra>')
    
    figs.update_layout(
        width=800, height=600,
        title=bmname,
        xaxis_title="|acc(Model A) - acc(Model B)|",
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

def fig_cov_baseline(bmname: str, diffvsum: pd.DataFrame):
    df = diffvsum
    # df["is_close"] = np.where(df["diff"].abs() < df["total"] / 20, "close", "not_close")
    # df = df[df["accA"] >= df["accB"]]
    df["is_close"] = np.where(np.abs(df["accA"] - df["accB"]) / df["std(A-B)"] <= 3, "close: ≤3σ", ">3σ")
    
    figs = px.scatter(df,
                    x=0.5*(df["accB"] + df["accA"]), y='std_acc',
                    # x=df["accB"], y='std_acc',
                    color="is_close",
                    # error_x=df["accA"] - df["accB"],
                    custom_data=[df['model_a'], 'model_b', 'sum', 'diff', 'pvalue', 'std(A-B)', 'accA', 'accB'])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.3f})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.3f})", 
        # "A + B: %{customdata[2]}", 
        # "A - B: %{customdata[3]}", 
        "p-value: %{customdata[4]:.4f}", 
        "std(A-B): %{customdata[5]:.2f}", 
        ])  + '<extra></extra>')

    figs.update_traces(
        marker=dict(
            size=3,
            opacity=0.5, 
        )
    )

    data_sz = diffvsum.iloc[0]['total']
    x = np.linspace(0, 1, 100)
    y = np.sqrt(x*(1-x) / data_sz)

    figl = go.Figure()

    figl.add_trace(go.Scatter(
        x=x, y=y, name="σ(acc)",
        # hoverinfo="skip",
        line=dict(color='lightgreen')
    ))

    figl.add_trace(go.Scatter(
        x=x, y=np.sqrt(2)*y, name="sqrt(2) σ(acc)",
        # hoverinfo="skip",
        line=dict(color='darkgreen')
    ))

    fig = go.Figure(data=figl.data + figs.data)
    fig.update_layout(
        width=800, height=600, title=bmname,
        xaxis_title="mean(acc(A), acc(B))",
        yaxis_title="σ(A-B)"
    )
    return fig

def get_sections(result: pd.DataFrame, benchmark_id):
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]
    summary = battle_summary(battles)
    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(benchmark_id, summary).to_html(full_html=False),
        "fig_pvalue_vs_diff": fig_pvalue_vs_diff(benchmark_id, summary).to_html(full_html=False),
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(full_html=False),
        "fig_cov_baseline": fig_cov_baseline(benchmark_id, summary).to_html(full_html=False),
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


def summary_stats(s, f=2, percent=True):
    return f"{s['mean']:.2g}±{s['std']:.2g} | [{s['min']:.2g}--{s['max']:.2g}] | n={int(s['count'])} "

def format_stats_badge(s):
    s_percent = dict(s)
    for st in ["mean", "std", "min", "max"]:
        s_percent[st] = 100 * s[st]
    summary = summary_stats(s_percent)
    mean = 100*s["mean"]
    return f'<span class="tooltip" data-tooltip="{summary}">{mean:.2g}</span>'

def write_summary_table(summary_count: pd.DataFrame, output_path: Path):
    summary_count = summary_count.sort_values(by='benchmark_id')

    def link_detail(bid):
        l1 = f"""<a href="model_{bid}.html">models </a> """
        l2 = f"""<a href="ex_{bid}.html"> examples </a>"""
        l3 = f"""<a href="ex_v_model_{bid}.html"> data </a>"""
        return l1 + '|' + l2 + '|' + l3
    summary_count['link to details'] = summary_count['benchmark_id'].apply(link_detail)

    def normalize(counts, includes):
        percent = counts.copy(deep=True)
        for c in includes:
            percent[c] = percent[c] / percent['size']
        return percent

    includes_cols = ['benchmark_id', 'size',  'std(A-B)', 'corr(A,B)', 'no_solve', 'tau-', 'sig_noise','link to details']
    percent_cols = ['p5_min', 'p5_max', 'no_solve', 'tau-']
    summary_percent = normalize(summary_count, percent_cols)

    print(summary_percent)
    template_path = r"templates/summary.html"

    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({
                'count_table': summary_count[includes_cols].to_html(escape=False, index=False),
                'percent_table': summary_percent[includes_cols].to_html(
                    escape=False,
                    classes="number-table",
                    index=False,
                    formatters={
                        "std(A-B)": lambda x: format_stats_badge(x),
                        "corr(A,B)": lambda x: format_stats_badge(x),
                        'p5_min': lambda x: f'{x*100:.2g}',
                        'p5_max': lambda x: f'{x*100:.2g}',
                        'no_solve': lambda x: f'{x*100:.2g}',
                        'tau-': lambda x: f'{x*100:.2g}',
                        'sig_noise': '{:.1g}'.format,
                    }),
            }))
            

def gen_model_report(benchmark_id: str, benchmark_results, OUTPUT_PATH):
    sections = get_sections(benchmark_results, benchmark_id)
    template_path=r"templates/template_model.html"
    output_path = f"{OUTPUT_PATH}/model_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'sections': sections}))

