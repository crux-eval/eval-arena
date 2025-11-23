from abc import ABC, abstractmethod
from collections import defaultdict
import random

import numpy as np
from numpy import mean, var, std
import pandas as pd
import scipy.stats as stats

import pytest

from estimators import Paired, Single, SingleTest, VarComps

class TestEstimator(ABC):
    def tscores(self, star: float, estimates: np.ndarray):
        errors = estimates - star
        se_mean = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        tstat = mean(errors) / se_mean
        # p_value_t = 2 * stats.t.sf(abs(tstat), df=len(errors)-1)
        # p_value_N = 2 * stats.norm.sf(abs(tstat))
        # print(f"{rms=:.2e}, {bias=:.2e}, {abs(bias)/se_mean=:.2f}, {p_value_t=:.2e}, {p_value_N=:.2e}")
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


class TestPairEstimators(TestEstimator):
    @staticmethod
    def total_variance_test(v: dict):
        assert np.allclose(v["var(A-B)"], v["E(var(A-B))"] + v["var(E(A-B))"]), v

    def paired_sample_vs_truth(self, pA, pB, K, niters=100):
        N = pA.shape[0]
        vstar = Paired.from_bernoulli_prob(pA, pB)
        self.total_variance_test(vstar)
        
        res = []
        for i in range(niters):
            A = np.random.rand(N, K)
            A = np.where(A < pA, 1, 0)

            B = np.random.rand(N, K)
            B = np.where(B < pB, 1, 0)

            vhat = Paired.from_samples(A, B)
            self.total_variance_test(vhat) 

            res.append({
                "var(A-B)": vhat["var(A-B)"],
            })
        return vstar["var(A-B)"], pd.DataFrame(res)

    def self_sample_vs_truth(self, pA, K=10, niters=100):
        vstar = Paired.from_bernoulli_prob(pA, pA)
        N = pA.shape[0]
        res = []
        for i in range(niters):
            A = np.random.rand(N, K)
            A = np.where(A < pA, 1, 0)
            pA_hat = A.mean(axis=1, keepdims=True)
            var_bernoulli_self = Paired.from_bernoulli_prob_self(pA_hat, K*np.ones_like(pA_hat))
            self.total_variance_test(var_bernoulli_self)
            var_bernoulli = Paired.from_bernoulli_prob(pA_hat, pA_hat)
            self.total_variance_test(var_bernoulli)

            res.append({
                "bernouli_hat": var_bernoulli["var(A-B)"],
                "bernoulli_self_hat": var_bernoulli_self["var(A-B)"],
            })
        return vstar["var(A-B)"], pd.DataFrame(res)

    def test_uniform(self, niter=2, N=164):
        pA = np.random.rand(N, 1)
        pB = (pA + 1*(np.random.rand(N, 1)-0.5)).clip(0, 1)
        for _ in range(niter):
            star, df_hats = self.paired_sample_vs_truth(pA, pB, 10)
            self.assert_inside_n_sigma(star, df_hats["var(A-B)"])

    def test_self_uniform(self, niter=2, N=164):
        pA = np.random.rand(N, 1)
        for _ in range(niter):
            star, df_hats = self.self_sample_vs_truth(pA, 10)
            print(df_hats)
            self.assert_inside_n_sigma(star, df_hats["bernoulli_self_hat"].to_numpy(), n_sigma=5)

            with pytest.raises(AssertionError, match="Bias"):
                self.assert_inside_n_sigma(star, df_hats["bernouli_hat"].to_numpy(), n_sigma=3)


class GenerativeModel(ABC):
    @abstractmethod
    def sample(self, K: int, seed: None = None) -> np.ndarray:
        """
        Args:
            K: Number of predictions to sample per question
            idx: Optional array of indices for paired samples. If None, then use all indices.
        Returns:
            Array of sampled predictions N x K
        """
        pass

    @abstractmethod
    def true_vars(self) -> dict[str]:
        pass

    def get_data_seed(self):
        pass

    @staticmethod
    def sample_indices(N: int):
        return np.random.choice(N, size=N, replace=True)

class BernoulliModel(GenerativeModel):
    def __init__(self, pA: np.ndarray, N: int, K: int, resample_method=False):
        self.pA = pA
        self.Npop = self.pA.shape[0]
        self.N = N
        self.idx = self.sample_idx()

        self.K = K
        self.resample_method = resample_method

    
    def sample_idx(self):
        return np.random.choice(self.Npop, size=self.N, replace=True)

    def sample(self) -> np.ndarray:
        Npop = self.pA.shape[0]
        # defaults to using the population
        if self.resample_method:
            self.idx = self.sample_idx()
        
        p = self.pA[self.idx]
        A = np.random.rand(self.N, self.K)
        A = np.where(A < p, 1, 0)
        return A

    def true_vars(self) -> dict[str]:
        return Single.from_bernoulli_prob(self.pA)


class TestSingleEstimators(TestEstimator):
    @staticmethod
    def total_variance_test(v: dict):
        assert np.allclose(v["var(A)"], v["E(var(A))"] + v["var(E(A))"]), v

    def get_estimates(self, model: GenerativeModel, estimator, attempts=20) -> list[VarComps]:
        ests = []
        for i in range(attempts):
            A = model.sample()
            vhat = estimator(A)
            # self.total_variance_test(vhat)
            ests.append(vhat)
        return ests

    def evaluate_estimator(self, model, estimator, verbose=False):
        """
        Get samples estimates, 
        """
        truth = model.true_vars()
        print(f"{truth=}")
        rel_error_stats = []
        for i in range(1):
            np.random.seed(42)
            ests: list[VarComps] = self.get_estimates(model, estimator, attempts=1000)
            for comp in ["var(A)", "E(var(A))", "var(E(A))"]:
            # for comp in ["var(A)"]:
                star = truth[comp]
                ests_comp = np.array([e[comp] for e in ests])
                
                t = self.tscores(star, ests_comp)
                rel_errors = (ests_comp - star) / star
                rel_error_stats.append({
                    "iter": i,
                    "estimator": estimator.__name__,
                    "resample": model.resample_method,
                    "K": model.K,
                    "comp": comp,
                    "t_score": t,
                    "mean_abs_mean": mean(rel_errors),
                    "mean_abs": np.mean(np.abs(rel_errors)),
                    # "median_abs": np.median(np.abs(rel_errors)),
                    "rms": np.sqrt(np.mean(rel_errors**2)),
                    "star": star,
                    "unbiased": ests[0].unbiased,
                })
                if verbose:
                    display(TestEstimator.plot_distribution_vs_star(ests_comp, star))

        results = pd.DataFrame(rel_error_stats)
        if verbose:
            pd.options.display.float_format = '{:.3g}'.format
            display(results)

        return results
 
    def test_estimators(self, pA, K, N, verbose=False):
        def result_is_biased(results: pd.DataFrame, comp: str = "E(var(A))"):
            biased = results[results["comp"] == comp]["t_score"].abs() > 5
            all_biased = all(biased)
            any_biased = any(biased)
            if any_biased and not all_biased:
                print("some but not appear biased")
            return any_biased 
        

        results_table = pd.DataFrame()
        for method in [True, False]:
            model = BernoulliModel(pA, K=K, N=N, resample_method=method)
            results = self.evaluate_estimator(model, estimator=Single.from_samples, verbose=verbose)
            results_table = pd.concat([results_table, results], ignore_index=True)
            # assert result_is_biased(results, "E(var(A))") == True
            # assert result_is_biased(results, "var(E(A))") == True
            results = self.evaluate_estimator(model, estimator=SingleTest.from_samples_naive, verbose=verbose)
            results_table = pd.concat([results_table, results], ignore_index=True)

            results = self.evaluate_estimator(model, estimator=SingleTest.from_samples_unbiased, verbose=verbose)
            results_table = pd.concat([results_table, results], ignore_index=True)

            results = self.evaluate_estimator(model, estimator=SingleTest.from_samples_unbiasedNK, verbose=verbose)
            results_table = pd.concat([results_table, results], ignore_index=True)
            
            results = self.evaluate_estimator(model, estimator=Single.from_samples_unbiasedK, verbose=verbose)
            results_table = pd.concat([results_table, results], ignore_index=True)
            # assert result_is_biased(results, "E(var(A))") == False
            # assert result_is_biased(results, "var(E(A))") == False
            # assert result_is_biased(results, "var(A)") == False
        # self.is_unbiased(model, K=5, estimator=Single.from_sample_unbiased)
        return results_table
        # pA = np.random.beta(3, 1, (30, 1))
        # model = BernoulliModel(pA)
        # self.is_close_and_unbiased(model, K=10)

        # pA = 0.3 + 0.3 * np.random.rand(10, 1)
        # model = BernoulliModel(pA)
        # self.is_close_and_unbiased(model, K=3)
        # self.is_biased(model, K=3)

        # this uniform can fail when the true variance is near the maximum possible 0.25
        # pA = np.random.rand(10, 1)
        # self.is_close_and_unbiased(pA, K=10)
        # self.is_biased(pA, K=10)


if __name__ == "__main__":
    # np.random.seed(42)
    # for _ in range(10):
    TestSingleEstimators().test_estimator()
    # TestPairEstimators().test_self_uniform()
    # TestPairEstimators().test_uniform()
