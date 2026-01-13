from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats

from estimators import Paired, Unpaired

logger = logging.getLogger(__name__)


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
    recompute: bool = True # generate results for all data and summary 
    write_summary: bool = True # use results in out_dir/tmp to generate the summary table
    
    max_diff: float = 0.1 # skip models that are more than max_diff in performance
    sigma_thres: float = 5.0 # how many std to consider as not close

    min_perf: float = 0.1 # near 0 models behave differently
    include_var_components: bool = True # use when there are multiple samples to report SE_x

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
        pa = pa[pa["model_a"] != pa["model_b"]]
        pa = BattleSummary._prob_outcome(pa)
        return pa
    
    @staticmethod
    def filter_battles(battles: pd.DataFrame, df_model, max_diff: float = 1) -> pd.DataFrame:
        df_pass = df_model[["model", "pass1"]]
        df_pairs = pd.merge(df_pass, df_pass, suffixes=["_a", "_b"], how="cross")
        df_pairs = df_pairs[(df_pairs["pass1_a"] - df_pairs["pass1_b"]).abs() <= max_diff]
        df_pairs = df_pairs[["model_a", "model_b"]]
        logger.info(f"Battles: {len(battles)}, filtered pairs: {len(df_pairs)}")
        battles = df_pairs.merge(battles, on=["model_a", "model_b"], how="inner")
        return battles

    @staticmethod
    def _pair_summary(df: pd.DataFrame):
        N = len(df)
        pA = df["pass1_a"].to_numpy().reshape(N, 1)
        pB = df["pass1_b"].to_numpy().reshape(N, 1)
        
        kA = df["count_a"].to_list()[0]
        kB = df["count_b"].to_list()[0]
        
        # note this is incorrect if model_a == model_b
        vars = Paired.from_bernoulli_prob_unbiasedK(pA, pB, kA, kB).clipped()
        # vars = Paired.from_bernoulli_prob(pA, pB)
        awin, bwin = df["awins"], df["bwins"]
        assert np.allclose(df["pass1_a"] - df["pass1_b"], awin - bwin)
        r = {
            "total": N,
            "accA": df["pass1_a"].mean(),
            "accB": df["pass1_b"].mean(),
            "sum(A!=B)": awin.sum() + bwin.sum(),
            "sum(A-B)": awin.sum() - bwin.sum(),
            "SE_signtest": np.sqrt(1/N * (awin + bwin).mean()),
            "pvalue": stats.binomtest(int(awin.sum()), int(awin.sum() + bwin.sum()), p=0.5).pvalue if int(awin.sum() + bwin.sum()) != 0 else 1,

            "SE_pred(A-B)": np.sqrt(1/N * vars["E(var(A-B))"]),
            "SE_x(A-B)": np.sqrt(1/N * vars["var(E(A-B))"]),
            "SE(A-B)": np.sqrt(1/N * vars["var(A-B)"]),

            "corr(A,B)": df["pass1_a"].corr(df["pass1_b"], method="pearson"),
        }
        assert r["SE(A-B)"] <= r["SE_signtest"] or np.allclose(r["SE(A-B)"], r["SE_signtest"])
        return pd.Series(r)

    @staticmethod
    def battle_summary(battles: pd.DataFrame) -> pd.DataFrame:
        diffvsum = battles.groupby(["model_a", "model_b"], group_keys=True, as_index=False)\
            .apply(BattleSummary._pair_summary)
        return diffvsum


def model_table(df_input, battles: pd.DataFrame | None = None):
    def _stds(g: pd.Series):
        pA = g["pass1"].to_numpy()
        kA = g["count"].to_numpy()
        if len(set(kA)) == 1:
            kA = kA[0]
        else:
            kA = 1
        vars = Unpaired.from_bernoulli_prob_unbiasedK(pA, kA).clipped()
        data_sz = len(pA)
        vars = {
            "SE_pred(A)": np.sqrt(1 / data_sz * vars.E_var),
            "SE_x(A)": np.sqrt(1 / data_sz * vars.var_E),
            "SE(A)": np.sqrt(1 / data_sz * vars.total_var),
            "pass1": np.mean(pA),
            "pass@count": np.mean(pA > 0),
            "count": np.mean(g["count"]),
        }
        return pd.Series(vars)

    model_stats = df_input[["model", "pass1", "count"]].groupby("model").apply(_stds).reset_index()

    if battles is not None:
        win_rates = battles[["model_a", "model_b", "awins"]]\
            .groupby(["model_a"])\
            .aggregate({"awins": "mean"})\
            .reset_index().rename(columns={"awins": "win_rate"})
        org_cols = ["model", "pass1", "pass@count", "win_rate", "count", "SE(A)", "SE_x(A)", "SE_pred(A)"]
        model_stats = win_rates.merge(model_stats, left_on="model_a", right_on="model")[org_cols].sort_values(by="pass1", ascending=False)
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
    # df_input["count"] = df_input["count"].fillna(1, inplace=False)

    battles = BattleSummary.pass1_to_battle(df_input)
    df_model = model_table(df_input, battles)
    battles = BattleSummary.filter_battles(battles, df_model, args.max_diff)
    summary = BattleSummary.battle_summary(battles)
    df_example = example_table(df_input, df_model)

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

    model_stats_keys = ["SE(A)", "SE_x(A)", "SE_pred(A)"]
    for key in model_stats_keys:
        df_model_above_min = df_model[df_model["pass1"] > args.min_perf]
        summary_stats[key] = df_model_above_min[key].describe().to_dict()

    close_pair_stats_keys = ["SE(A-B)", "SE_x(A-B)", "SE_pred(A-B)", "SE_signtest", "corr(A,B)", "sum(A!=B)"]
    for key in close_pair_stats_keys:
        summary_stats[key] = close_pairs[key].describe().to_dict()

    return ArenaResult(input_table=df_input,
                       model_table=df_model,
                       example_table=df_example,
                       summary=summary,
                       summary_stats=summary_stats,
                    ) 
