from dataclasses import dataclass
import math
from collections import Counter

import numpy as np
import scipy.stats as stats
import pandas as pd


def _hard_outcome(pa: pd.DataFrame, thres: float = 0.5):
    a_pass = pa["pass1_a"] > thres
    b_pass = pa["pass1_b"] > thres
    awins = a_pass & ~b_pass
    bwins = ~a_pass & b_pass
    neither = ~a_pass & ~b_pass
    both = a_pass & b_pass 

    assert all(awins | bwins | neither | both) \
        and sum(awins) + sum(bwins) + sum(both) + sum(neither) == len(pa), "outcomes should be unique and complete"
    pa.loc[awins, "winner"] = "model_a"
    pa.loc[bwins, "winner"] = "model_b"
    pa.loc[neither, "winner"] = "neither"
    pa.loc[both, "winner"] = "both"
    return pa

def _prob_outcome(pa: pd.DataFrame):
    a_pass = pa["pass1_a"]
    b_pass = pa["pass1_b"]
    awins = a_pass * (1 - b_pass)
    bwins = (1 - a_pass) * b_pass
    neither = (1 - a_pass) * (1 - b_pass)
    both = a_pass * b_pass

    assert np.allclose(awins + bwins + both + neither, 1), "sum of probs should be 1"
    pa["awins"] = awins
    pa["bwins"] = bwins
    pa["neither"] = neither
    pa["both"] = both
    return pa

def _diff_outcome(pa: pd.DataFrame, thres: float = 0.5):
    soft_diff = pa["pass1_a"] - pa["pass1_b"]
    hard_diff = np.where(pa["pass1_a"] > thres, 1, 0) - np.where(pa["pass1_b"] > thres, 1, 0)
    pa["A-B"] = soft_diff 
    pa["A-B_hard"] = hard_diff
    return pa

def pass1_to_battle(result: pd.DataFrame, thres=0.5):
    """
    generates a pairwise comparison table from pass1 information using 3 ways to summarize the outcome
        - 1: using a threshold to decide the winner and store one of 4 outcomes in the winner column
        - 2: use the difference in scores, which can generalize to 
    """
    pa = pd.merge(result, result, on=["example_id"], suffixes=["_a", "_b"], how="outer")
    print(pa)
    pa = pa[pa["model_a"] != pa["model_b"]]
    
    pa = _hard_outcome(pa)
    pa = _prob_outcome(pa)
    pa = _diff_outcome(pa, thres)
    return pa

def _comp_stats(outcomes: pd.Series):
    sufs = Counter(outcomes.values) # model_a, model_b, neither, both are the possible outcomes
    total = sufs.total()
    model_a, model_b, both, neither = sufs["model_a"], sufs["model_b"], sufs["both"], sufs["neither"]
    assert model_a + model_b + both + neither == total
    pa = model_a / total
    pb = model_b / total
    diff = model_a - model_b
    sum = model_a + model_b
    std_count = np.sqrt(total * (pa*(1-pa) +  pb*(1-pb) + 2*pa*pb))
    pvalue = stats.binomtest(model_a, sum, p=0.5).pvalue if sum != 0 else 1
    res = dict(
        sum = sum,
        diff = diff,
        total = total,
        accA = (model_a + both) / total,
        accB = (model_b + both) / total,
        pvalue = pvalue,
        std_count = std_count,
        std_acc = std_count / total,
        covAB = (both / total - (model_a + both) / total*(model_b + both) / total),
    )
    return res

def _pair_summary(df: pd.DataFrame):
    N = len(df)
    sufs = Counter(df["winner"])
    awin, bwin, both, neither = sufs["model_a"], sufs["model_b"], sufs["both"], sufs["neither"]
    assert awin + bwin + both + neither == N
    pawin = awin / N
    pbwin = bwin / N

    corr_ab = df["pass1_a"].corr(df["pass1_b"], method="pearson")
    r = { 
        "corr(A,B)": corr_ab,
        "std(A-B)": df["A-B"].std(ddof=0) / np.sqrt(N),
        "std_signtest": 1/N * np.sqrt((df["awins"].sum() + df["bwins"].sum())),
        "std_bootstrap": np.sqrt(1/N * (pawin * (1 - pawin) + pbwin * (1 - pbwin) + 2 * pawin * pbwin)),
        "std(A-B)_hard": df["A-B_hard"].std(ddof=0) / np.sqrt(N),
        "unpaired_std": 1/np.sqrt(N) * np.sqrt(df["pass1_a"].var(ddof=0) + df["pass1_b"].var(ddof=0)),
    }

    assert np.allclose(r["std(A-B)_hard"], r["std_bootstrap"])
    return pd.Series({**r, **_comp_stats(df["winner"])})

def battle_summary(battles):
    diffvsum = battles.groupby(["model_a", "model_b"])\
        .apply(_pair_summary)\
        .reset_index(drop=False)
    return diffvsum
    
def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000, ref_model="gpt-3.5-turbo-0613",
):
    """
    calculate Elo based on winrate, code from chatbot arena (https://chat.lmsys.org/)
    https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH
    with a bugfix for when a model never wins, and add reference model as an argument
    """
    from sklearn.linear_model import LogisticRegression
    
    def ties_plus_two_wins(outcomes: pd.Series):
        sufs = Counter(outcomes.values) # model_a, model_b, neither, both are the possible outcomes
        # print(sufs)
        return 2*sufs["model_a"] + sufs["both"] + sufs["neither"]

    ptbl_win = pd.pivot_table(
        df,
        values=["winner"],
        index="model_a",
        columns="model_b",
        aggfunc=ties_plus_two_wins,
    ).reset_index().set_index("model_a").droplevel(axis=1, level=0)

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

def model_table(battles, result):
    win_rates = battles[["model_a", "model_b", "winner"]]\
        .groupby(["model_a"])\
        .aggregate({"winner": lambda x: Counter(x)["model_a"] / Counter(x).total()})\
        .reset_index().rename(columns={"winner": "win_rate"})

    model_elos = compute_mle_elo(battles).to_frame("elo").reset_index()
    win_elo = win_rates.merge(model_elos, on="model_a")

    def _stds(g: pd.Series):
        pass1s = g["pass1"]
        Ns= g["N"]
        data_sz = len(pass1s)
        p = pass1s.to_numpy()
        return pd.Series({
            "std_i": np.sqrt(1 / data_sz * np.mean(p*(1-p))),
            "std": np.sqrt(1 / data_sz) * np.std(p),
            "pass1": np.mean(pass1s),
            "N": np.mean(Ns),
        })

    # add std if pass1 is not just 0 or 1 
    basic_stats = result[["model", "pass1", "N"]].groupby("model").apply(_stds).reset_index()
    table_inds = ["model", "pass1", "std", "std_i", "N", "win_rate", "elo"]

    all_stats = win_elo.merge(basic_stats, left_on="model_a", right_on="model")[table_inds].sort_values(by="pass1", ascending=False)
    return all_stats

def example_table(result, all_stats):
    records = []
    ids = set(result["example_id"]) 
    for current_id in list(ids):
        example_data = result[result["example_id"] == current_id][["model", "pass1"]]
        example_data["correct"] = np.where(example_data["pass1"] > 0, 1, 0)
        ex = example_data[["model", "correct"]].merge(all_stats[["model", "elo", "pass1"]], left_on = "model", right_on = "model")
        r = {}
        r["example_id"] = current_id
        solved_ex = ex[ex["correct"] == 1]
        r["min_elo"] = solved_ex["elo"].min()
        r["num_solved"] = len(solved_ex)
        r["models"] = solved_ex["model"].to_numpy()
        r["acc"] = len(solved_ex) / len(ex)
        r["tau"] = stats.kendalltau(ex["correct"], ex["pass1"]).statistic
        # r['corr'] = stats.pearsonr(ex['correct'], ex['pass1']).statistic
        records.append(r)

    return pd.DataFrame(records)

@dataclass
class ArenaResult:
    battles: pd.DataFrame
    summary: pd.DataFrame
    model_table: pd.DataFrame
    example_table: pd.DataFrame
    input_table: pd.DataFrame
    summary_stats: dict


def summarize_benchmark(result: pd.DataFrame) -> ArenaResult:
    benchmarks = set(result['benchmark_id'])
    assert len(benchmarks) == 1
    bid = benchmarks.pop()
    battles = pass1_to_battle(result)
    summary = battle_summary(battles)
    agg_results = model_table(battles, result)
    ex = example_table(result, agg_results)
    close_pairs = summary[summary["pvalue"] > 2.7e-3] # 3 sigma

    summary_stats = {
        'benchmark_id': bid,
        'size': int(summary.iloc[0]['total']),
        'models': len(set(summary["model_a"])),
        'total_pairs': len(summary),
        'close_pairs': len(close_pairs),

        'std(A)': agg_results["std"].describe().to_dict(),
        'std_i': agg_results["std_i"].describe().to_dict(),

        'std(A-B)': close_pairs["std(A-B)"].describe().to_dict(),
        'std_signtest': close_pairs["std_signtest"].describe().to_dict(),
        'corr(A,B)': close_pairs["corr(A,B)"].describe().to_dict(),
        
        'p5_min': summary[summary['pvalue'] < 0.05]['diff'].abs().min(),
        'p5_max': summary[summary['pvalue'] > 0.05]['diff'].abs().max(),
        'min_dist': int(summary['sum'].abs().min()),
        'no_solve': (ex['acc'] == 0).to_numpy().sum(),
        'tau-': (ex['tau'] < 0).to_numpy().sum(),
    }

    return ArenaResult(input_table=result, battles=battles,
                       summary=summary, model_table=agg_results,
                       example_table=ex, summary_stats=summary_stats) 
