from typing import List, Optional
import math
from collections import Counter

import numpy as np
import numpy.random as rng
import scipy.stats as stats
import pandas as pd

def pass1_to_battle(result: pd.DataFrame):
    pa = pd.merge(result, result, on=['example_id'], suffixes=["_a", "_b"], how='outer')
    pa = pa[pa['model_a'] != pa['model_b']]

    awins = (pa['pass1_a'] > 0) & (pa['pass1_b'] == 0)
    bwins = (pa['pass1_a'] == 0) & (pa['pass1_b'] > 0)
    ties_neither = (pa['pass1_a'] == 0) & (pa['pass1_b'] == 0)
    ties_both = (pa['pass1_a'] > 0) & (pa['pass1_b'] > 0)
    pa['winner'] = 'a'
    pa.loc[awins, 'winner'] = 'model_a'
    pa.loc[bwins, 'winner'] = 'model_b'
    pa.loc[ties_neither, 'winner'] = 'neither'
    pa.loc[ties_both, 'winner'] = 'both'
    return pa

def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000, ref_model="gpt-3.5-turbo-0613",
):
    """
    calculate Elo based on winrate, code from chatbot arena with minor changes.
    """
    from sklearn.linear_model import LogisticRegression
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["both", "neither"])) == 0:
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

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if ref_model in models.index:
        elo_scores += 1000 - elo_scores[models[ref_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def result_table(battles, result):
    win_rates = battles[['model_a', 'model_b', 'winner']]\
        .groupby(['model_a'])\
        .aggregate({'winner': lambda x: Counter(x)['model_a'] / Counter(x).total()})\
        .reset_index().rename(columns={'winner': 'win_rate'})

    model_elos = compute_mle_elo(battles).to_frame('elo').reset_index()
    win_elo = win_rates.merge(model_elos, on='model_a')
    accs = result.groupby('model').agg('mean', numeric_only=True).reset_index()
    all_stats = win_elo.merge(accs, left_on='model_a', right_on='model')[['model', 'pass1', 'win_rate', 'elo']].sort_values(by='pass1', ascending=False)
    return all_stats

def null_samples(weights, tie_prob, num_samples = 100000):
    not_ties = rng.rand(num_samples, weights.size) > tie_prob
    not_tie_bernoulis = not_ties * np.sign(rng.randn(*not_ties.shape))
    scores = not_tie_bernoulis @ weights
    return scores

def sign_test_niid(response_a: List, response_b: List, tie_probs: Optional[List[float]], weights: Optional[List[float]], sample_all=False) -> float:
    if weights is None:
        weights = np.ones(len(response_a))
    if tie_probs is None:
        tie_probs = np.zeros(len(response_a))
    assert all(weights >= 0)
    weights = weights / weights.sum()

    comparisons = np.where(response_a > response_b, 1, 0) + np.where(response_b > response_a, -1, 0)
    score_thres = np.abs(np.sum(comparisons * weights))

    # then answer the question, how many under the null hypothesis is more extreme than this one
    # given no ties, or should ties be considered too
    not_ties = comparisons != 0
    print(f'k / n = {comparisons.sum()}/{not_ties.sum()}\t thres: {score_thres}', )
    if sample_all:
        samps = null_samples(weights, tie_probs)
    else:
        samps = null_samples(weights[not_ties], tie_probs[not_ties])
    cdf = stats.ecdf(samps).cdf

    pvalue = (1 - cdf.evaluate(score_thres - 1e-10)) + cdf.evaluate(-score_thres + 1e-10)
    print('pvalue', 1 - np.mean(np.abs(samps) <= score_thres - 1e-10))
    # print(pd.Series(samps).describe())
    return cdf, pvalue
