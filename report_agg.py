import json, math, glob
from collections import Counter

import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go

from arena import result_table, pass1_to_battle, battle_summary

def fig_diff_vs_sum(bmname: str, diffvsum: pd.DataFrame):
    figs = px.scatter(diffvsum, x=diffvsum['diff'].abs(), y='sum',
                      custom_data=['model_a', 'model_b', 'sum', 'diff', 'pvalue', 'std_count', 'accA', 'accB'])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.2f})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.2f})", 
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
    
    # fig = go.Figure(data=figs.data)
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
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(full_html=False),
        "result_table (no ties)": result_table(battles_no_ties, result).to_html(
            index=False,
            formatters={
                'pass1': '{:.1%}'.format,
                'win_rate': '{:.1%}'.format,
                'elo': '{:.1f}'.format
        }),
        # "result_table": result_table(battles, result).style \
        #     .format(precision=3).to_html()
    }
    return sections


def gen_benchmark_report(benchmark_id: str, benchmark_results, output_path):
    sections = get_sections(benchmark_results, benchmark_id)
    from jinja2 import Template
    template_path=r"agg_template_description.html"
    output_path = f"{output_path}/agg_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'sections': sections}))

OUTPUT_PATH = 'crux-eval.github.io/eval-arena'
if __name__ == '__main__':
    records = []
    for fname in glob.glob(f"data/*.jsonl"):
        with open(fname, 'rt') as f:
            records.extend([json.loads(l) for l in f.readlines()])
    eval_results = pd.DataFrame(records)
    benchmarks = set(eval_results['benchmark_id'])
    for b in benchmarks:
        benchmark_result = eval_results[eval_results['benchmark_id'] == b]
        gen_benchmark_report(b, benchmark_result, OUTPUT_PATH)
# pushd .; cd crux-eval.github.io/; git commit -am 'report'; git push; popd
