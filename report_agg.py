import json, math, glob

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

import plotly.express as px

from arena import result_table, pass1_to_battle, compute_pairwise_win_fraction, compute_pvalues 

def visualize_pairwise_win_fraction(battles, title, max_num_models=100):
    a_win, b_win, neither, both = compute_pairwise_win_fraction(battles, max_num_models)

    fig = px.imshow(a_win, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser",
                  yaxis_title="Model A: Winner",
                  xaxis_side="top", height=900, width=900,
                  title_y=0.07, title_x=0.5)

    sort_keys = a_win.keys()  
    extra_info = (pd.concat([a_win, b_win, neither, both])
    .stack(dropna=False)
    .groupby(level=[0,1])
    .apply(tuple).apply(lambda t: tuple(s if not math.isnan(s) else 'nan' for s in t))
    .unstack()
    ).loc[sort_keys, sort_keys]
    fig.update_traces(customdata=extra_info, hovertemplate="<br>".join(
        [
            "A Wins: %{customdata[0]:.3f}",
            "B Wins: %{customdata[1]:.3f}",
            "Neither: %{customdata[2]:.3f}",
            "Both: %{customdata[3]:.3f}",
        ])
    )
    return fig


def visualize_pvalues(battles, title, max_num_models=100):
    row_beats_col, pvalue, diffs, sums, chi2 = compute_pvalues(battles, max_num_models)
    fig = px.imshow(pvalue,
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B",
                  yaxis_title="Model A",
                  xaxis_side="top", height=900, width=900,
                  title_y=0.07, title_x=0.5)

    sort_keys = row_beats_col.keys() 
    extra_info = (pd.concat([row_beats_col, pvalue, diffs, sums, chi2])
        .stack(dropna=False)
        .groupby(level=[0,1])
        .apply(tuple).apply(lambda t: tuple(s if not math.isnan(s) else 'nan' for s in t))
        .unstack()
    ).loc[sort_keys, sort_keys]
    
    # display(row_beats_col)
    # display(extra_info)
    fig.update_traces(customdata=extra_info, hovertemplate=
        "<br>".join([
        "Model A: %{y}",
        "Model B: %{x}", 
        "binom p-value: %{customdata[1]:.4f}",
        "chi2 p-value: %{customdata[4]:.4f}",
        "A_win frac.: %{customdata[0]:.2f}",
        "A_win - B_win: %{customdata[2]}",
        "A_win + B_win: %{customdata[3]}",
        ])  + '<extra></extra>')

    return fig
#consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

def fig_delta_vs_pvalues(battles: pd.DataFrame, result: pd.DataFrame):
    a_win, pvalue, diffs, sums, chi2 = compute_pvalues(battles, 100)
    df = pvalue.reset_index()
    df_pval = df.melt(id_vars='model_a', value_vars=list(df.columns[1:]), var_name='model_b', value_name='p_value')
    df = diffs.reset_index()
    df_diffs = df.melt(id_vars='model_a', value_vars=list(df.columns[1:]), var_name='model_b', value_name='diff')
    benchmark_size = len(set(result['example_id']))
    df_diffs['|acc(A) - acc(B)|'] = df_diffs['diff'].abs() / benchmark_size
    df_diffs['models'] = df_diffs['model_a'] + ' vs. ' + df_diffs['model_b']

    df = df_pval.merge(df_diffs, on=['model_a', 'model_b'])
    return px.scatter(df, x='|acc(A) - acc(B)|', y='p_value', hover_data='models')


def get_sections(result: pd.DataFrame, benchmark_id):
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]

    fig_pvalues = visualize_pvalues(battles_no_ties, f'p-values {benchmark_id}', max_num_models=60)
    # fig_pairwin = visualize_pairwise_win_fraction(battles, f'win_rates {benchmark_id}', max_num_models=60)

    sections = {
        "p-values": fig_pvalues.to_html(full_html=False),
        "delta vs. p-values": fig_delta_vs_pvalues(battles, result).to_html(full_html=False),
        # "pairwise wins (including ties)": fig_pairwin.to_html(full_html=False),
        "result table": result_table(battles_no_ties, result).style \
                .format(precision=3).to_html()
    }
    return sections


def gen_benchmark_report(benchmark_id: str, eval_results):
    sections = get_sections(eval_results[eval_results['benchmark_id'] == benchmark_id], benchmark_id)
    from jinja2 import Template
    template_path=r"report_template.html"
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
