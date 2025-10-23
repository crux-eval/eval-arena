from abc import ABC, abstractmethod

import numpy as np
from numpy import mean, var, std
import pandas as pd
import scipy.stats as stats

import pytest

from estimators import Paired, Single


class TestEstimator(ABC):
    def assert_inside_n_sigma(self, star: float, estimates: np.ndarray, n_sigma=5):
        def rmse(diff):
            return np.sqrt(np.mean(diff**2))

        errors = estimates - star
        rms = rmse(errors)
        bias = mean(errors)
        
        se_mean = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        tstat = mean(errors) / se_mean
        p_value_t = 2 * stats.t.sf(abs(tstat), df=len(errors)-1)
        p_value_N = 2 * stats.norm.sf(abs(tstat))
        print(f"{rms=:.2e}, {bias=:.2e}, {abs(bias)/se_mean=:.2f}, {p_value_t=:.2e}, {p_value_N=:.2e}")
        # self.plot_distribution_vs_star(estimates, star, se_mean)

        assert abs(bias) < n_sigma * se_mean, (
            f"Bias/se_mean ({bias / se_mean:.3f}) exceeds {n_sigma}σ threshold:\n"        
        )
    
    @staticmethod
    def plot_distribution_vs_star(estimates, star):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=estimates))
        fig.add_vline(x=star, line_dash="solid", line_color="red", name=f"star = {star}")
        fig.add_vline(x=np.mean(estimates), line_dash="dash", line_color="blue", name=f"mean pred")

        se_mean = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        fig.add_vline(x=np.mean(estimates) + se_mean, line_dash="dot", line_color="yellow", name=f"mean + se(mean)")
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
            var_bernoulli_self = Paired.from_bernoulli_prob_self(pA_hat, K)
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


class TestSingleEstimators(TestEstimator):
    @staticmethod
    def total_variance_test(v: dict):
        assert np.allclose(v["var(A)"], v["E(var(A))"] + v["var(E(A))"]), v

    def sample_vs_truth(self, pA, K, niters=20, key="var(A)"):
        N = pA.shape[0]
        vstar = Single.from_bernoulli_prob(pA)
        self.total_variance_test(vstar)
        
        res = []
        for i in range(niters):
            A = np.random.rand(N, K)
            A = np.where(A < pA, 1, 0)

            vhat = Single.from_samples(A)
            self.total_variance_test(vhat)
            vhat_unbiased = Single.from_sample_unbiased(A)
            self.total_variance_test(vhat_unbiased)

            res.append({
                "biased": vhat[key],
                "unbiased": vhat_unbiased[key],
            })
        return vstar[key], pd.DataFrame(res)

    def is_close_and_unbiased(self, pA, K):
        for key in ["var(A)", "E(var(A))", "var(E(A))"]:
            print(f"testing {key} is unbiased using {K} samples")
            star, df_hats = self.sample_vs_truth(pA, K=K, niters=100, key=key)
            self.assert_inside_n_sigma(star, df_hats["unbiased"].to_numpy(), n_sigma=5)
            # self.assert_inside_boostrap(star, df_hats[key + "_hat_unbiased"])

    def is_biased(self, pA, K):
        for key in ["var(A)"]:
            print(f"testing {key} is biased using {K} samples")
            with pytest.raises(AssertionError, match="Bias"):
                for _ in range(100):
                    star, df_hats = self.sample_vs_truth(pA, K=K, niters=100, key=key)
                    self.assert_inside_n_sigma(star, df_hats["biased"].to_numpy(), n_sigma=3)
    
    def test_estimator(self):
        pA = np.random.beta(0.2, 0.8, (50, 1))
        self.is_close_and_unbiased(pA, K=5)

        pA = np.random.beta(3, 1, (30, 1))
        self.is_close_and_unbiased(pA, K=10)

        pA = 0.1 + 0.4 * np.random.rand(10, 1)
        self.is_close_and_unbiased(pA, K=3)
        self.is_biased(pA, K=3)

        # this uniform can fail when the true variance is near the maximum possible 0.25
        # pA = np.random.rand(10, 1)
        # self.is_close_and_unbiased(pA, K=10)
        # self.is_biased(pA, K=10)


if __name__ == "__main__":
    np.random.seed(42)
    # for _ in range(10):
    TestSingleEstimators().test_estimator()
    TestPairEstimators().test_self_uniform()
    TestPairEstimators().test_uniform()
