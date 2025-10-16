import numpy as np
from numpy import mean, var, std
import pandas as pd

import pytest

def cov(A, B, ddof=0):
    return np.sum((A - np.mean(A)) * (B - np.mean(B))) / (len(A) - ddof)

class Paired:
    @staticmethod
    def from_samples(A: np.ndarray, B: np.ndarray, dof=0) -> dict[str, float]:
        assert A.shape[0] == B.shape[0], "should be paired" 
        return {
            "var(E(A-B))": var(mean(A, axis=1) - mean(B, axis=1)),
            "E(var(A-B))": mean(var(A, axis=1) + var(B, axis=1)),
            "var(A-B)": var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            "cov(A,B)": cov(mean(A, axis=1), mean(B, axis=1)),
            "var(A) + var(B)": var(A) + var(B),
            # "_var(A-B)": mean(A**2 + B**2) - 2 * mean(mean(A, axis=1) * mean(B, axis=1)) - mean(A-B)**2,
        }
    
    @staticmethod
    def from_samples_unbiased(A: np.ndarray, B: np.ndarray, dof=0) -> dict[str, float]:
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = A.shape[1]
        return {
            "var(E(A-B))": var(mean(A-B, axis=1)) - mean(var(A, axis=1)/(kA-1) + var(B, axis=1)/(kA-1)) ,
            "E(var(A-B))": mean(var(A, axis=1)* (1 + 1/(kA-1)) + var(B, axis=1) * (1 + 1/(kB-1))),
            "var(A-B)": var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)), # actually this is slightly biased too, but we ignore it

            "_var(A-B)": mean(A**2 + B**2) - 2 * mean(mean(A, axis=1) * mean(B, axis=1)) - mean(A-B)**2,
        }

    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray, pB: np.ndarray) -> dict[str, float]:
        """Calculate variance for Bernoulli random variables
            Args:
                pA, pB: Success probabilities of shape (n_samples, 1)
        """
        assert pA.shape[0] == pB.shape[0]
        assert pA.shape[1] == pB.shape[1] == 1
        pA = pA.flatten()
        pB = pB.flatten()
        return {
            "var(E(A-B))": var(pA - pB),
            "E(var(A-B))": mean(pA*(1-pA) + pB*(1-pB)),
            "var(A-B)": np.clip(mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*cov(pA, pB), a_min=0, a_max=None),

            "_var(A-B)": mean(pA + pB - 2*pA*pB) - mean(pA - pB)**2,
        }
    
    @staticmethod
    def from_bernoulli_prob_self(pA: np.ndarray, K: np.ndarray) -> dict[str, float]:
        """Calculate variance for Bernoulli random variables
            Args:
                pA: Success probabilities of shape (n_samples, 1)
                K: number of samples of pA, needed for bias correction
        """
        pA = pA.flatten()
        pB = pA
        mu = mean(pA)
        covAA = mean((pA*pA - 1/K*pA)*(1+1/(K-1)))  - mu * mu
        return {
            "var(E(A-B))": var(pA - pB),
            "E(var(A-B))": mean(pA*(1-pA) + pB*(1-pB)),
            "var(A-B)": mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*covAA,
            "cov(A,B)": covAA,
            "var(A) + var(B)": mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)),
            "_var(A-B)": mean(pA + pB - 2*pA*pB) - mean(pA - pB)**2,
        }


class TestBias():
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

            pA_hat = A.mean(axis=1, keepdims=True)
            var_bernoulli_self = Paired.from_bernoulli_prob_self(pA_hat, K)
            var_bernoulli = Paired.from_bernoulli_prob(pA_hat, pA_hat)
            self.total_variance_test(var_bernoulli)

            res.append({
                "v_hat": vhat["var(A-B)"],
                # "bernouli_hat": var_bernoulli["var(A-B)"],
                # "var_bernoulli_self_hat": var_bernoulli_self["var(A-B)"],
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
            var_bernoulli = Paired.from_bernoulli_prob(pA_hat, pA_hat)
            self.total_variance_test(var_bernoulli)

            res.append({
                "bernouli_hat": var_bernoulli["var(A-B)"],
                "bernoulli_self_hat": var_bernoulli_self["var(A-B)"],
            })
        return vstar["var(A-B)"], pd.DataFrame(res)

    def check_results(self, star: float, df_hats: pd.DataFrame):
        def rmse(diff):
            return np.sqrt(np.mean(diff**2))

        for method in df_hats.columns:
            col = f'{method}'
            relative_error = df_hats[col] / star - 1
            rms = rmse(relative_error)
            bias = mean(relative_error)
            std_error = np.std(relative_error, ddof=1) / np.sqrt(len(df_hats))
        
            # Bias should be within n_sigma standard errors
            n_sigma = 5 # to avoid failing by accident
            print(f"{method:12s}: RMSE = {rms:8.6f}, Bias = {bias:8.6f}, Bias/SE ratio = {abs(bias)/std_error:.2f}")
            assert abs(bias) < n_sigma * std_error, (
                f"Bias ({bias:.6f}) exceeds {n_sigma}σ threshold:\n"        
            )
    
    def test_uniform(self, niter=2, N=164):
        pA = np.random.rand(N, 1)
        pB = (pA + 1*(np.random.rand(N, 1)-0.5)).clip(0, 1)
        for _ in range(niter):
            star, df_hats = self.paired_sample_vs_truth(pA, pB, 10)
            self.check_results(star, df_hats)

    def test_self_uniform(self, niter=2, N=164):
        pA = np.random.rand(N, 1)
        for _ in range(niter):
            star, df_hats = self.self_sample_vs_truth(pA, 10)
            self.check_results(star, df_hats[["bernoulli_self_hat"]])

            with pytest.raises(AssertionError, match="Bias"):
                self.check_results(star, df_hats[["bernouli_hat"]])

# if __name__ == "__main__":
#     # np.random.seed(42)
