import json, math, glob
from collections import Counter

import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go

from arena import result_table, pass1_to_battle

def fig_diff_vs_sum(battles):
    data_sz = len(set(battles['example_id']))
    bmname = set(battles['benchmark_id_a']).pop()

    print(data_sz)
    def aggfunc(input: pd.Series):
        sufs = Counter(input.values) # model_a, model_b, neither, both
        res = {} 
        res['diff'] = sufs['model_a'] - sufs['model_b']
        res['sum'] = sufs['model_a'] + sufs['model_b'] 
        # res['pvalue-chi2'] = 1 if res['diff'] == 0 else (1 - stats.chi2.cdf( (np.abs(res['diff']) - 1)**2 / res['sum'], 1))
        res['pvalue'] = stats.binomtest(sufs['model_a'], res['sum'], p=0.5).pvalue
        total = sufs.total()
        pa = sufs['model_a'] / total
        pb = sufs['model_b'] / total 
        res['std'] = np.sqrt(total * (pa*(1-pa) +  pb*(1-pb) + 2*pa*pb))
        return res

    diffvsum = battles[['model_a', 'model_b', 'winner']]\
        .groupby(['model_a', 'model_b'])\
        .aggregate(aggfunc)\
        ['winner'].apply(pd.Series)\
        .reset_index(drop=False)
    figs = px.scatter(diffvsum, x=diffvsum['diff'].abs(), y='sum', custom_data=['model_a', 'model_b', 'sum', 'diff', 'pvalue', 'std'])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]}",
        "Model B: %{customdata[1]}", 
        "|A - B|: %{customdata[3]}", 
        "A + B: %{customdata[2]}", 
        "p-value: %{customdata[4]:.4f}", 
        "std(A-B): %{customdata[5]:.4f}", 
        ])  + '<extra></extra>')

    min_p5 = diffvsum[diffvsum['pvalue'] < 0.05]['diff'].abs().min() / data_sz
    max_p5 = diffvsum[diffvsum['pvalue'] > 0.05]['diff'].abs().max() / data_sz
    print(f'{bmname}\t N={data_sz},\t diff_min/max={min_p5}/{max_p5}')
    maxy = diffvsum['sum'].max()
    refs = []
    for alpha in [0.05, 0.1]:
        thres = stats.chi2.ppf(1-alpha, 1)
        print('thres', thres)
        y = np.linspace(1, maxy, 200)
        refs.append(pd.DataFrame({'x': 1 + np.sqrt(y * thres), 'y': y, 'type': f'pvalue={alpha}'}))
    
    x = np.linspace(0, data_sz / 2, 100)
    refs.append(pd.DataFrame({'x': x, 'y': x, 'type': 'x=y'}))
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

def fig_accs_and_pvalues(battles):
    def aggfunc(input: pd.Series):
        sufs = Counter(input.values) # model_a, model_b, neither, both
        res = {} 
        total = sufs.total()
        res['diff'] = sufs['model_a'] - sufs['model_b']
        res['sum'] = sufs['model_a'] + sufs['model_b'] 
        res['accA'] = (sufs['model_a'] + sufs['both']) / total
        res['accB'] = (sufs['model_b'] + sufs['both']) / total
        pv = stats.binomtest(sufs['model_a'], res['sum'], p=0.5).pvalue
        res['p_value'] = pv
        pa = sufs['model_a'] / total
        pb = sufs['model_b'] / total
        res['std'] = np.sqrt(1 / total * (pa*(1-pa) +  pb*(1-pb) + 2*pa*pb))
        return res
    
    diffvsum = battles[['model_a', 'model_b', 'winner']]\
        .groupby(['model_a', 'model_b'])\
        .aggregate(aggfunc)\
        ['winner'].apply(pd.Series)\
        .reset_index(drop=False)
    figs = px.scatter(diffvsum, x='accA', y='accB',
            color='p_value', range_color=[0, 0.2],
            custom_data=['model_a', 'model_b', 'accA', 'accB', 'p_value', 'std'])
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
    bmname = set(battles['benchmark_id_a']).pop()
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

    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(battles).to_html(full_html=False),
        "fig_diff_vs_sum": fig_diff_vs_sum(battles).to_html(full_html=False),
        "result_table (no ties)": result_table(battles_no_ties, result)\
        .to_html(formatters={
            'pass1': '{:.1%}'.format,
            'win_rate': '{:.1%}'.format,
            'elo': '{:.1f}'.format
        }),
        # "result_table": result_table(battles, result).style \
        #     .format(precision=3).to_html()
    }
    return sections

def gen_benchmark_report(benchmark_id: str, eval_results):
    sections = get_sections(eval_results[eval_results['benchmark_id'] == benchmark_id], benchmark_id)
    from jinja2 import Template
    template_path=r"agg_template_description.html"
    output_path = rf"crux-eval.github.io/eval-arena/agg_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'sections': sections}))


if __name__ == '__main__':
    records = []
    for fname in glob.glob(f"data/*.jsonl"):
        with open(fname, 'rt') as f:
            records.extend([json.loads(l) for l in f.readlines()])
    eval_results = pd.DataFrame(records)
    print(set(eval_results['benchmark_id']))
    for b in set(eval_results['benchmark_id']):
        gen_benchmark_report(b, eval_results)
# pushd .; cd crux-eval.github.io/; git commit -am 'report'; git push; popd
