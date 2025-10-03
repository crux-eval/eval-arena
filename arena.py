from dataclasses import dataclass
import math
from collections import Counter

import numpy as np
from numpy import mean, var, std
import scipy.stats as stats
import pandas as pd


@dataclass
class ArenaResult:
    summary: pd.DataFrame
    model_table: pd.DataFrame
    example_table: pd.DataFrame
    input_table: pd.DataFrame
    summary_stats: dict


@dataclass
class ReportArgs:
    out_dir: str = "gh-pages/"
    data: str = "data/*.jsonl"
    recompute: bool = True # generate results for all data and summary line
    write_summary: bool = True # use results in out_dir/tmp to generate the summary table
    
    max_diff: float = 0.1 # skip models that are more than max_diff in performance
    sigma_thres: float = 5.0 # how many std to consider as not close
    min_perf: float = 0.05 # too bad for inconlusion, including near 0 models does give some extreme results


def cov(A, B, ddof=0):
    return np.cov(A, B, ddof=ddof)[0, 1]
    
class Paired:
    @staticmethod
    def sample_vars(A: np.ndarray, B: np.ndarray, dof=0) -> dict:
        assert A.shape[0] == B.shape[0], "should be paired" 
        return {
            "var(E(A-B))": var(mean(A-B, axis=1)),
            "E(var(A-B))": mean(var(A, axis=1) + var(B, axis=1)),
            "var(A-B)": var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            "cov(A,B)": cov(mean(A, axis=1), mean(B, axis=1)),
            "var(A) + var(B)": var(A) + var(B),
            # "_var(A-B)": mean(A**2 + B**2) - 2 * mean(mean(A, axis=1) * mean(B, axis=1)) - mean(A-B)**2,
        }
    
    @staticmethod
    def sample_vars_unbiased(A: np.ndarray, B: np.ndarray, dof=0) -> dict:
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = A.shape[1]
        return {
            "var(E(A-B))": var(mean(A-B, axis=1)) - mean(var(A, axis=1)/(kA-1) + var(B, axis=1)/(kA-1)) ,
            "E(var(A-B))": mean(var(A, axis=1)* (1 + 1/(kA-1)) + var(B, axis=1) * (1 + 1/(kB-1))),
            "var(A-B)": var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)), # actually this is slightly biased too, but we ignore it
            # "_var(A-B)": mean(A**2 + B**2) - 2 * mean(mean(A, axis=1) * mean(B, axis=1)) - mean(A-B)**2,
        }
    
    @staticmethod
    def bernoulli_sample_vars(A: np.ndarray, B: np.ndarray, dof=0) -> dict:
        ...

    @staticmethod
    def bernoulli_p_vars(pA: np.ndarray, pB: np.ndarray) -> dict:
        assert pA.shape[0] == pB.shape[0]
        assert pA.shape[1] == pB.shape[1] == 1
        pA = pA.flatten()
        pB = pB.flatten()
        return {
            "var(E(A-B))": var(pA - pB),
            "E(var(A-B))": mean(pA*(1-pA) + pB*(1-pB)),
            "var(A-B)": np.clip(mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*cov(pA, pB), a_min=0, a_max=None),
            "cov(A,B)": cov(pA, pB),
            "_var(A-B)": mean(pA + pB - 2*pA*pB) - mean(pA - pB)**2,
        }
    
    @staticmethod
    def bernoulli_self(ph: np.ndarray, K: np.ndarray) -> dict:
        ph = ph.flatten()
        pB = ph
        mu = mean(ph)
        covAA = mean((ph*ph - 1/K*ph)*(1+1/(K-1)))  - mu * mu
        return {
            "var(E(A-B))": var(ph - pB),
            "E(var(A-B))": mean(ph*(1-ph) + pB*(1-pB)),
            "var(A-B)": mean(ph)*(1-mean(ph)) + mean(pB)*(1-mean(pB)) - 2*covAA,
            "cov(A,B)": covAA,
            "var(A) + var(B)": mean(ph)*(1-mean(ph)) + mean(pB)*(1-mean(pB)),
            "_var(A-B)": mean(ph + pB - 2*ph*pB) - mean(ph - pB)**2,
        }
    

class BattleSummary:
    @staticmethod
    def _prob_outcome(pa: pd.DataFrame) -> pd.DataFrame:
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
    
    @staticmethod
    def pass1_to_battle(df_input: pd.DataFrame) -> pd.DataFrame:
        """
        generates a pairwise comparison table from pass1 information using 
        """
        pa = pd.merge(df_input, df_input, on=["example_id"], suffixes=["_a", "_b"], how="outer")
        pa = BattleSummary._prob_outcome(pa)
        return pa
    
    @staticmethod
    def filter_battles(battles: pd.DataFrame, df_model, max_diff: float = 1) -> pd.DataFrame:
        df_pass = df_model[["model", "pass1"]]
        df_pairs = pd.merge(df_pass, df_pass, suffixes=["_a", "_b"], how="cross")
        df_pairs = df_pairs[(df_pairs["pass1_a"] - df_pairs["pass1_b"]).abs() <= max_diff]
        df_pairs = df_pairs[["model_a", "model_b"]]
        print(f"{len(battles)=}", f"{len(df_pairs)=}")
        battles = df_pairs.merge(battles, on=["model_a", "model_b"], how="inner")
        return battles

    @staticmethod
    def _pair_summary(df: pd.DataFrame):
        N = len(df)
        awin, bwin, both, neither = df["awins"], df["bwins"], df["both"], df["neither"]
        assert np.allclose(awin.sum() + bwin.sum() + both.sum() + neither.sum(), N)
        assert np.allclose(awin + bwin + both + neither, np.ones((N, 1)))
        assert np.allclose(awin + both, df["pass1_a"])
        assert np.allclose(bwin + both, df["pass1_b"])
        pawin = awin.sum() / N
        pbwin = bwin.sum() / N
        pA = df["pass1_a"].to_numpy().reshape(N, 1)
        pB = df["pass1_b"].to_numpy().reshape(N, 1)
        
        assert np.allclose(df["pass1_a"] - df["pass1_b"], awin - bwin)
        r = {
            "sum(A!=B)": awin.sum() + bwin.sum(),
            "sum(A-B)": awin.sum() - bwin.sum(),
            "total": N,
            "accA": df["pass1_a"].mean(), 
            "accB": df["pass1_b"].mean(),

            "corr(A,B)": df["pass1_a"].corr(df["pass1_b"], method="pearson"),
            "std_signtest": np.sqrt(1/N * (awin + bwin).mean()),
            "std(E(A-B))": np.sqrt(1/N * (awin - bwin).var(ddof=0)),
            "E(std(A-B))": np.sqrt(1/N * np.mean(awin + bwin - (awin - bwin)**2)),
            "std(A-B)": np.sqrt(1/N * (pawin * (1 - pawin) + pbwin * (1 - pbwin) + 2 * pawin * pbwin)),
            "std(A-B)_2": np.sqrt(1/N * Paired.bernoulli_p_vars(pA, pB)["var(A-B)"]),

            "pvalue": stats.binomtest(int(awin.sum()), int(awin.sum() + bwin.sum()), p=0.5).pvalue if int(awin.sum() + bwin.sum()) != 0 else 1,
        }

        assert np.allclose(r["std(A-B)"], r["std(A-B)_2"]), r
        assert np.allclose(r["std(E(A-B))"]**2 + r["E(std(A-B))"]**2, r["std(A-B)"]**2)
        assertcond = r["std(A-B)"] <= r["std_signtest"] or np.allclose(r["std(A-B)"], r["std_signtest"])
        if not assertcond:
            print("signtest not as big, strange", r)

        return pd.Series(r)

    @staticmethod
    def battle_summary(battles: pd.DataFrame) -> pd.DataFrame:
        diffvsum = battles.groupby(["model_a", "model_b"])\
            .apply(BattleSummary._pair_summary)\
            .reset_index(drop=False)
        return diffvsum


def model_table(df_input, battles: pd.DataFrame | None = None):
    def _stds(g: pd.Series):
        pass1s = g["pass1"]
        data_sz = len(pass1s)
        p = pass1s.to_numpy()
        vars = {
            "E(std(A))": np.sqrt(1 / data_sz * np.mean(p*(1-p))),
            "std(E(A))": 1 / np.sqrt(data_sz) * np.std(p),
            "std(A)": np.sqrt(1 / data_sz * p.mean()* (1-p.mean())),
            "pass1": np.mean(pass1s),
            "N": np.mean(g["N"]),
        }
        assert np.allclose(vars["E(std(A))"]**2 + vars["std(E(A))"]**2, vars["std(A)"]**2)
        return pd.Series(vars)

    # add std if pass1 is not just 0 or 1 
    model_stats = df_input[["model", "pass1", "N"]].groupby("model").apply(_stds).reset_index()
    table_inds = ["model", "pass1", "std(E(A))", "E(std(A))", "std(A)", "N"]

    if battles:
        table_inds += ["win_rate"]
        win_rates = battles[["model_a", "model_b", "awins"]]\
            .groupby(["model_a"])\
            .aggregate({"awins": "mean"})\
            .reset_index().rename(columns={"awins": "win_rate"})
        model_stats = win_rates.merge(model_stats, left_on="model_a", right_on="model")[table_inds].sort_values(by="pass1", ascending=False)
    return model_stats

def example_table(df_input, df_model_table):
    model_table = df_model_table[["model", "pass1"]].rename(columns={"pass1": "pass1_of_model"})
    
    df_work = df_input[["example_id", "model", "pass1"]].copy()
    df_work["pass@any_of_ex"] = np.where(df_work["pass1"] > 0, 1, 0)
    df_work["pass1_of_ex"] = df_work["pass1"]
    
    df_merged = df_work.merge(model_table, on="model")
    
    def agg_func(g: pd.DataFrame):
        solved = g[g["pass@any_of_ex"] == 1]
        return pd.Series({
            "min_pass1_of_model": solved["pass1_of_model"].min() if len(solved) > 0 else np.nan,
            "num_solved": len(solved),
            "models": solved["model"].to_numpy(),
            "pass1_of_ex": g["pass1_of_ex"].mean(),
            "pass@any_of_ex": g["pass@any_of_ex"].mean(),
            "tau": stats.kendalltau(g["pass1_of_ex"], g["pass1_of_model"]).statistic
        })
    
    result = df_merged.groupby("example_id").apply(agg_func).reset_index()
    return result

def summarize_benchmark(df_input: pd.DataFrame, args: ReportArgs) -> ArenaResult:
    benchmarks = set(df_input["benchmark_id"])
    assert len(benchmarks) == 1
    bid = benchmarks.pop()

    if "N" not in df_input.columns:
        df_input["N"] = 1
        print(f"assuming N=1 on {bid}")
    df_input["N"] = df_input["N"].fillna(1, inplace=False)

    df_model = model_table(df_input)
    df_example = example_table(df_input, df_model)

    battles = BattleSummary.pass1_to_battle(df_input)
    battles = BattleSummary.filter_battles(battles, df_model, args.max_diff)
    summary = BattleSummary.battle_summary(battles)

    close_pairs = summary[
        (summary["pvalue"] > 2 * stats.norm.sf(abs(args.sigma_thres))) &  # p=0.05, 0.0027, 6e-7  for 1.96, 3, 5 sigma
        (summary["accA"] > args.min_perf) & (summary["accB"] > args.min_perf)
    ]

    summary_stats = {
        "benchmark_id": bid,
        "size": len(df_example),
        "models": len(df_model),
        "total_pairs": len(summary),
        "close_pairs": len(close_pairs),
        "no_solve": (df_example["pass1_of_ex"] == 0).to_numpy().sum(),
        "tau-": (df_example["tau"] < 0).to_numpy().sum(),
    }

    model_stats_keys = ["std(A)", "std(E(A))", "E(std(A))"]
    for key in model_stats_keys:
        summary_stats[key] = df_model[key].describe().to_dict()

    close_pair_stats_keys = ["std(A-B)", "E(std(A-B))", "std(E(A-B))", "std_signtest", "corr(A,B)", "sum(A!=B)"]
    for key in close_pair_stats_keys:
        summary_stats[key] = close_pairs[key].describe().to_dict()

    return ArenaResult(input_table=df_input,
                       model_table=df_model,
                       example_table=df_example,
                       summary=summary,
                       summary_stats=summary_stats,
                    ) 
