import numpy as np
from dataclasses import dataclass

from abc import ABC, abstractmethod
from numpy import mean, var

def cov(A, B, ddof=0):
    return np.sum((A - np.mean(A)) * (B - np.mean(B))) / (len(A) - ddof)

@dataclass
class EstimatorProperty:
    total_biased: bool = True
    component_biased: bool = 1e-8
    total_variance_tolerance = 1e-8
    total_var_accuracy = (100, 0.2)
    pred_var_accuracy = (100, 0.2)
    data_var_accuracy = (100, 0.2)


@dataclass
class VarComps:
    total_var: float  # var(A-B) or var(A)
    var_E: float  # var(E(A-B)) or var(E(A))
    E_var: float  # E(var(A-B)) or E(var(A))
    paired: bool
    unbiased: bool = False
    satisfy_total_variance: bool = True

    def __post_init__(self):
        total = self.var_E + self.E_var
        if not np.isclose(total, self.total_var, rtol=self.satisfy_total_variance):
            rtol = (total - self.total_var) / self.total_var
            raise ValueError(f"Total variance did not hold. components {self.to_dict()}, rtol={rtol}")

    def to_dict(self) -> dict[str, float]:
        key = "A-B" if self.paired else "A"
        return {
            f"var({key})": self.total_var,
            f"var(E({key}))": self.var_E,
            f"E(var({key}))": self.E_var,
        }

    def __getitem__(self, key: str) -> float:
        return self.to_dict()[key]


class Paired:
    @staticmethod
    def from_samples(A: np.ndarray, B: np.ndarray) -> VarComps:
        """Calculate variance from
            Args:
                A, B: Success probabilities of shape (N questions, by K predictions for each question)
        """
        assert A.shape[0] == B.shape[0], "should be paired"
        return VarComps(
            total_var=var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            var_E=var(mean(A, axis=1) - mean(B, axis=1)),
            E_var=mean(var(A, axis=1) + var(B, axis=1)),
            paired=True,
            unbiased=False
        )
    
    @staticmethod
    def from_samples_unbiased(A: np.ndarray, B: np.ndarray) -> VarComps:
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = B.shape[1]
        return VarComps(
            total_var=var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            var_E=var(mean(A-B, axis=1)) - mean(var(A, axis=1)/(kA-1) + var(B, axis=1)/(kA-1)),
            E_var=mean(var(A, axis=1)* (1 + 1/(kA-1)) + var(B, axis=1) * (1 + 1/(kB-1))),
            paired=True,
            unbiased=True
        )

    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray, pB: np.ndarray) -> VarComps:
        """Calculate variance from the probability of Bernoulli
            Args:
                pA, pB: Success probabilities of shape (n_samples, 1)
        """
        assert pA.shape[0] == pB.shape[0]
        assert pA.shape[1] == pB.shape[1] == 1
        pA = pA.flatten()
        pB = pB.flatten()
        return VarComps(
            total_var=np.clip(mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*cov(pA, pB), a_min=0, a_max=None),
            var_E=var(pA - pB),
            E_var=mean(pA*(1-pA) + pB*(1-pB)),
            paired=True,
            unbiased=False
        )


class PairedExperimental:
    @staticmethod
    def from_bernoulli_prob_self(pA: np.ndarray, K: np.ndarray) -> VarComps:
        """Calculate variance for Bernoulli random variables
            Args:
                pA: Success probabilities of shape (n_samples, 1)
                K: number of samples for bias correction
        """
        assert pA.shape == K.shape
        assert all(K > 1), "need more than 1 sample per problem"
        pA = pA.flatten()
        # use all
        # pA * (pA * K - 1)/(K-1)
        # a direct computation using leave one out sampling
        covAA = mean((pA*pA - 1/K*pA)*(K/(K-1))) - mean(pA)**2
        var_A_minus_B = 2*mean(pA)*(1-mean(pA)) - 2*covAA

        E_var_A_minus_B = 2*mean(pA*(1-pA) * K/(K-1))
        assert var_A_minus_B > 0 or np.allclose(var_A_minus_B, 0), f"{var_A_minus_B=}"
        assert np.allclose(var_A_minus_B, E_var_A_minus_B)
        return VarComps(
            total_var=var_A_minus_B,
            var_E=0.0,  # by definition
            E_var=E_var_A_minus_B,  # by independence of the noise conditioned on a prompt
            paired=True,
            unbiased=False
        )
    
    @staticmethod
    def from_samples_random_diffs(A: np.ndarray, B: np.ndarray, M=1000) -> VarComps:
        assert A.shape[0] == B.shape[0], "should be paired"
        # For each row i, generate M samples of A_ij - B_ik where j and k are randomly drawn
        AB_diff_samples = np.array([
            np.random.choice(A[i], size=M, replace=True) - np.random.choice(B[i], size=M, replace=True) 
            for i in range(A.shape[0])
        ])
        return Single.from_samples(AB_diff_samples)
    
    @staticmethod
    def from_samples_balanced_diff(A: np.ndarray, B: np.ndarray) -> VarComps:
        """
        For each row i, generate A_ij - B_ik where j and k covers all columns of A and B respectively
        """
        assert A.shape[0] == B.shape[0], "should be paired"
        N = A.shape[0]
        kA = A.shape[1]
        kB = B.shape[1]
        AB = np.zeros((N, kA*kB)) 
        for i in range(N):
            diffs = A[i][:, np.newaxis] - B[i][np.newaxis, :]
            AB[i, :] = diffs.flatten()
        comps = Single.from_samples(AB)
        comps.paired = True
        return comps
    


class Single:
    @staticmethod
    def from_samples(A: np.ndarray) -> VarComps:
        return VarComps(
            total_var=var(A),
            var_E=var(mean(A, axis=1)),
            E_var=mean(var(A, axis=1)),
            paired=False,
            unbiased=False
        )
    
  
    @staticmethod
    def from_samples_unbiasedK(A: np.ndarray) -> VarComps:
        """
        only be unbiased in K, assumes N is large enough
        """
        kA = A.shape[1]
        N = A.shape[0]
        return VarComps(
            total_var=var(A),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1)) - 1/(kA-1) * mean(var(A, axis=1)),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1)) * (1 + 1/(kA-1)),
            paired=False,
            unbiased=True
        )

    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray) -> VarComps:
        return VarComps(
            total_var=mean(pA)*(1-mean(pA)),
            var_E=var(pA),
            E_var=mean(pA*(1-pA)),
            paired=False,
            unbiased=False
        )
    
    
class SingleExperimental:
    @staticmethod
    def from_samples_naive(A: np.ndarray, M=100) -> VarComps:
        # draw M independent samples from the ith row of A, expanding kA to M for accurate direct estimations
        N, kA = A.shape
        A_expand_rows = np.array([np.random.choice(A[i], size=M, replace=True) for i in range(N)])
        # A_expand = np.array([np.random.choice(A.flatten, size=M, replace=True) for i in range(N)])
        return VarComps(
            total_var=np.mean(var(A_expand_rows, axis=0)),
            var_E=float("nan") if kA == 1 else var(mean(A_expand_rows, axis=1)),
            # E_var is estimated directly
            E_var=float("nan") if kA == 1 else N * var(mean(A_expand_rows, axis=0)),
            paired=False,
            unbiased=False,
            satisfy_total_variance=False, # total variance is not expected to hold
        )
    
    @staticmethod
    def from_samples_unbiased(A: np.ndarray) -> VarComps:
        kA = A.shape[1]
        N = A.shape[0]
        return VarComps(
            total_var=var(A) * (1 + 1/(N*kA - 1)),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1)) - 1/(kA-1) * mean(var(A, axis=1)) + var(A)*1/(N*kA - 1),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1)* (1 + 1/(kA-1))),
            paired=False,
            unbiased=True
        )
    
    @staticmethod
    def from_samples_unbiasedNK(A: np.ndarray) -> VarComps:
        """
        unbiased when drawing repeated samples
        """
        kA = A.shape[1]
        N = A.shape[0]
        return VarComps(
            total_var=mean(var(A, axis=1, ddof=0)) + var(mean(A, axis=1), ddof=1),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1), ddof=1) - mean(var(A, axis=1, ddof=1)) + mean(var(A, axis=1, ddof=0)),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1, ddof=1)),
            paired=False,
            unbiased=True
        )