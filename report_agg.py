# %%
from collections import defaultdict
import json, math, glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
from tqdm import tqdm
import math
import os

# %%


# figs = basic_analysis(result, benchmark_id)
def pass1_to_battle(result: pd.DataFrame):
    pa = pd.merge(result, result, on=['example_id'], suffixes=["_a", "_b"], how='outer')

    awins = (pa['pass1_a'] > 0) & (pa['pass1_b'] == 0)
    bwins = (pa['pass1_a'] == 0) & (pa['pass1_b'] > 0)
    ties_neither = (pa['pass1_a'] == 0) & (pa['pass1_b'] == 0)
    ties_both = (pa['pass1_a'] > 0) & (pa['pass1_b'] > 0)
    # pa[['winner']][awins] = 'model_a' 
    pa['winner'] = 'a'
    pa.loc[awins, 'winner'] = 'model_a'
    pa.loc[bwins, 'winner'] = 'model_b'
    pa.loc[ties_neither, 'winner'] = 'neither'
    pa.loc[ties_both, 'winner'] = 'both'
    return pa 

def compute_pairwise_win_fraction(battles, max_num_models=30):
    num_battles_ptbl = pd.pivot_table(battles,
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)
    
    a_win = pd.pivot_table(
        battles[battles['winner'] == "model_a"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0) / num_battles_ptbl  
    
    b_win = pd.pivot_table(
        battles[battles['winner'] == "model_b"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0) / num_battles_ptbl

    neither = pd.pivot_table(
        battles[battles['winner'] == "neither"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0) / num_battles_ptbl
    
    both = pd.pivot_table(
        battles[battles['winner'] == "both"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0) / num_battles_ptbl 

    # 
    prop_wins = a_win.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    sort_keys = list(prop_wins.keys())
    return tuple(x.loc[sort_keys, sort_keys] for x in [a_win, b_win, neither, both])

def compute_pvalues(battles, max_num_models=100):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles['winner'] == "model_a"],
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(battles,
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (
        (a_win_ptbl) /
        (num_battles_ptbl)
    )
    # display(mcnemar)
    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]

    wins = a_win_ptbl.loc[model_names, model_names]
    diffs = (wins - wins.T)
    sums = (wins + wins.T)
    suf_stats = (pd.concat([wins, wins - wins.T, wins + wins.T])
    .stack(dropna=False)
    .groupby(level=[0,1])
    .apply(tuple)
    .unstack()
    ).loc[model_names, model_names]
    chi2 = suf_stats.applymap(lambda x: 1 if x[2] == 0 else 1 - stats.chi2.cdf( (np.abs(x[1]) - 1)**2 / (x[2]), 1))
    binom = suf_stats.applymap(lambda x: stats.binomtest(x[0], x[2], p=0.5).pvalue if x[2] > 0 else 1)
    return row_beats_col, binom, diffs, sums, chi2 

def visualize_pairwise_win_fraction(battles, title, max_num_models=30):
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

def visualize_pvalues(battles, title, max_num_models=30):
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

def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000
):
    from sklearn.linear_model import LogisticRegression
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["both", "both"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["both", "neither"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie
    # display(ptbl_win)
    # display(ptbl_tie)

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]
    # fig = px.imshow(X)
    # display(fig)
    # print(Y, Y.shape)

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "gpt-3.5-turbo-0613" in models.index:
        elo_scores += 1000 - elo_scores[models["gpt-3.5-turbo-0613"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def result_table(battles_no_ties, result):
    model_elos = compute_mle_elo(battles_no_ties)
    pair_stats = compute_pairwise_win_fraction(battles_no_ties)
    a_win = pair_stats[0]
    win_rates = a_win.mean(axis=1).sort_values(ascending=False)
    win_elo = win_rates.to_frame(name='win_rate').join(model_elos.to_frame(name='elo')).reset_index()
    accs = result.groupby('model').agg('mean', numeric_only=True).reset_index().sort_values(by='pass1', ascending=False)
    all_stats = win_elo.merge(accs, left_on='model_a', right_on='model')[['model', 'pass1', 'win_rate', 'elo']]
    return all_stats

def get_sections(result: pd.DataFrame, benchmark_id):
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]

    fig_pvalues = visualize_pvalues(battles_no_ties, f'p-values {benchmark_id}', max_num_models=60)
    fig_pairwin = visualize_pairwise_win_fraction(battles, f'win_rates {benchmark_id}', max_num_models=60)

    sections = {
        "p-values": fig_pvalues.to_html(full_html=False),
        "delta vs. p-values": fig_delta_vs_pvalues(battles, result).to_html(full_html=False),
        "pairwise wins (including ties)": fig_pairwin.to_html(full_html=False),
        "result table": result_table(battles_no_ties, result).to_html(float_format='%10.3f')
    }

    return sections


def gen_benchmark_report(benchmark_id: str):
    sections = get_sections(eval_results[eval_results['benchmark_id'] == benchmark_id], benchmark_id)
    from jinja2 import Template
    template_path=r"report_template.html"
    output_path = rf"crux-eval.github.io/reports/agg_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'sections': sections}))

records = []
for fname in glob.glob(f"data/*.jsonl"):
    with open(fname, 'rt') as f:
        records.extend([json.loads(l) for l in f.readlines()])
eval_results = pd.DataFrame(records)
print(set(eval_results['benchmark_id']))
gen_benchmark_report('mbpp+')
gen_benchmark_report('humaneval+')
gen_benchmark_report('CRUXEval-input')
gen_benchmark_report('CRUXEval-output')

# pushd .; cd crux-eval.github.io/; git commit -am 'report'; git push; popd
