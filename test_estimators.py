from abc import ABC, abstractmethod
from collections import defaultdict
import random

import numpy as np
from numpy import mean, var, std
import pandas as pd
import scipy.stats as stats

import pytest

from estimators import Paired, Single, SingleTest, VarComps

class GenerativeModel(ABC):
    idx: np.ndarray # data index of the model, designed for paired
    Npop: int # population size
    N: int # sample size

    @abstractmethod
    def sample_preds(self) -> np.ndarray:
        """
        returns: array of sampled predictions given the same idx
        """
        pass

    @abstractmethod
    def true_vars(self) -> VarComps:
        """
        returns: true variance components for comparisons
        """
        pass

    def sample_idx(self):
        return np.random.choice(self.Npop, size=self.N, replace=True)

class BernoulliModel(GenerativeModel):
    def __init__(self, pA: np.ndarray, N: int, K: int, resample_method=False):
        self.pA = pA
        self.Npop = self.pA.shape[0]
        self.N = N
        self.idx = self.sample_idx()
        self.K = K
        self.resample_method = resample_method


    def sample_preds(self) -> np.ndarray:
        p = self.pA[self.idx]
        A = np.random.rand(self.N, self.K)
        A = np.where(A < p, 1, 0)
        return A

    def true_vars(self) -> dict[str]:
        return Single.from_bernoulli_prob(self.pA)

    
class TestEstimator(ABC):
    def tscores(self, star: float, estimates: np.ndarray):
        errors = estimates - star
        se_mean = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        tstat = mean(errors) / se_mean
        # p_value_t = 2 * stats.t.sf(abs(tstat), df=len(errors)-1)
        return tstat
    
    @staticmethod
    def plot_distribution_vs_star(estimates, star):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=estimates,  name="hist"))
        fig.add_trace(go.Scatter(y=list(range(len(estimates))), x=estimates, mode="markers", marker=dict(size=3, opacity=0.25), name="empirical"))
        fig.add_vline(x=star, line_dash="solid", line_color="green")

        se_mean = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        fig.add_vline(x=np.mean(estimates), line_dash="dash", line_color="blue", annotation_text=f"star={star:.3f} est={np.mean(estimates):.4f}+-{se_mean:.4f}, ", name="mean pred")
        fig.add_vline(x=np.mean(estimates) + se_mean, line_dash="dot", line_color="yellow", name=f"mean + se(mean)")
        fig.add_vline(x=np.mean(estimates) - se_mean, line_dash="dot", line_color="yellow", name=f"mean + se(mean)")
        display(fig)
    
    def evaluate_estimator(self, truth: VarComps, ests: list[VarComps], estimator, resample_method: bool, verbose=False):
        if verbose:
            print(f"{truth=}")
        rel_error_stats = []
        for comp in [key for key in truth.to_dict()]:
            star = truth[comp]
            ests_comp = np.array([e[comp] for e in ests])

            t = self.tscores(star, ests_comp)
            rel_errors = (ests_comp - star) / star
            rel_error_stats.append({
                "estimator": estimator.__name__,
                "resample": resample_method,
                "comp": comp,
                "t_score": t,
                "mean_rel_error": mean(rel_errors),
                "mean_abs": np.mean(np.abs(rel_errors)),
                "rms": np.sqrt(np.mean(rel_errors**2)),
                "star": star,
                "mean": np.mean(ests_comp),
                "std": np.std(ests_comp),
                "unbiased": ests[0].unbiased,
            })
            if verbose:
                self.plot_distribution_vs_star(ests_comp, star)

        results = pd.DataFrame(rel_error_stats)
        if verbose:
            pd.options.display.float_format = '{:.3g}'.format
            display(results)

        return results



class TestPairEstimators(TestEstimator):
    @staticmethod
    def total_variance_test(v: VarComps):
        assert np.allclose(v["var(A-B)"], v["E(var(A-B))"] + v["var(E(A-B))"]), v.to_dict()

    def get_estimates(self, modelA: BernoulliModel, modelB: BernoulliModel, estimator, attempts=20) -> list[VarComps]:
        """Get estimates from paired models with synchronized indices"""
        ests = []
        for i in range(attempts):
            # Synchronize indices for paired sampling
            if modelA.resample_method:
                shared_idx = modelA.sample_idx()
                modelA.idx = shared_idx
                modelB.idx = shared_idx

            A = modelA.sample_preds()
            B = modelB.sample_preds()
            vhat = estimator(A, B)
            ests.append(vhat)
        return ests

    def test_estimators(self, pA=None, pB=None, K=5, N=100, verbose=False):
        results_table = pd.DataFrame()
        for resample_method in [True, False]:
            modelA = BernoulliModel(pA, K=K, N=N, resample_method=resample_method)
            modelB = BernoulliModel(pB, K=K, N=N, resample_method=resample_method)
            modelB.idx = modelA.idx
            truth = Paired.from_bernoulli_prob(modelA.pA, modelB.pA)
            for estimator in [
                Paired.from_samples,
                Paired.from_samples_unbiased,
            ]:
                ests: list[VarComps] = self.get_estimates(modelA, modelB, estimator, attempts=1000)
                results = self.evaluate_estimator(truth, ests, estimator, modelA.resample_method, verbose)
                results_table = pd.concat([results_table, results], ignore_index=True)

        return results_table


class TestSingleEstimators(TestEstimator):
    def get_estimates(self, model: GenerativeModel, estimator, attempts: int) -> list[VarComps]:
        ests = []
        for i in range(attempts):
            if model.resample_method:
                model.idx = model.sample_idx()
            A = model.sample_preds()
            vhat = estimator(A)
            ests.append(vhat)
        return ests

 
    def test_estimators(self, pA=None, K=5, N=100, verbose=False):
        results_table = pd.DataFrame()
        for resample_method in [True, False]:
            model = BernoulliModel(pA, K=K, N=N, resample_method=resample_method)
            truth = model.true_vars()
            for estimator in [
                Single.from_samples,
                SingleTest.from_samples_naive,
                SingleTest.from_samples_unbiased,
                SingleTest.from_samples_unbiasedNK,
                Single.from_samples_unbiasedK,
            ]:
                ests: list[VarComps] = self.get_estimates(model, estimator, attempts=1000)
                results = self.evaluate_estimator(truth, ests, estimator, model.resample_method, verbose)
                results_table = pd.concat([results_table, results], ignore_index=True)

        return results_table


if __name__ == "__main__":
    # np.random.seed(42)
    # for _ in range(10):
    TestSingleEstimators().test_estimator()
    # TestPairEstimators().test_self_uniform()
    # TestPairEstimators().test_uniform()
