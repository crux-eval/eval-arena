import numpy as np
from numpy import mean, var

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
            
            "_var(A-B)": mean(A**2 + B**2) - 2 * mean(mean(A, axis=1) * mean(B, axis=1)) - mean(A-B)**2,
        }
    
    @staticmethod
    def from_samples_unbiased(A: np.ndarray, B: np.ndarray, dof=0) -> dict[str, float]:
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = A.shape[1]
        return {
            "var(E(A-B))": var(mean(A-B, axis=1)) - mean(var(A, axis=1)/(kA-1) + var(B, axis=1)/(kA-1)) ,
            "E(var(A-B))": mean(var(A, axis=1)* (1 + 1/(kA-1)) + var(B, axis=1) * (1 + 1/(kB-1))),
            "var(A-B)": var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),

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
                K: number of samples for bias correction
        """
        assert all(K > 1), "need more than 1 sample per problem"
        pA = pA.flatten()
        pB = pA
        # use all 
        # pA * (pA * K - 1)/(K-1)
        covAA = mean((pA*pA - 1/K*pA)*(K/(K-1))) - mean(pA)**2
        var_A_minus_B = mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*covAA
        E_var_A_minus_B = mean((pA*(1-pA) + pB*(1-pB)) * (K/(K-1)))
        assert var_A_minus_B > 0 or np.allclose(var_A_minus_B, 0), f"{var_A_minus_B=}"
        assert np.allclose(var_A_minus_B, E_var_A_minus_B)
        return {
            "var(E(A-B))": 0,
            # by indepedence of the noise conditioned on a prompt
            "E(var(A-B))": E_var_A_minus_B,
            # "E(var(A-B))": var_diff,
            "var(A-B)": var_A_minus_B,
        }


class Single:
    @staticmethod
    def from_samples(A: np.ndarray, dof=0) -> dict:
        return {
            "var(E(A))": var(mean(A, axis=1)),
            "E(var(A))": mean(var(A, axis=1)),
            "var(A)": var(A),
        } 
    
    @staticmethod
    def from_sample_unbiased(A: np.ndarray) -> dict:
        kA = A.shape[1]
        N = A.shape[0]
        return {
            "var(E(A))": var(mean(A, axis=1)) - 1/(kA-1) * mean(var(A, axis=1)) + var(A)*1/(N*kA - 1), 
            "E(var(A))": mean(var(A, axis=1)) * (1 + 1/(kA-1)), 
            "var(A)": var(A) * (1 + 1/(N*kA - 1)), 
        }

    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray) -> dict:
        return {
            "var(E(A))": var(pA),
            "E(var(A))": mean(pA*(1-pA)),
            "var(A)": mean(pA)*(1-mean(pA))
        }
    