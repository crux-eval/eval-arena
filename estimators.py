import numpy as np
from dataclasses import dataclass

from abc import ABC, abstractmethod
from numpy import mean, var

def cov(A: np.ndarray, B: np.ndarray, ddof=0) -> float:
    return np.sum((A - np.mean(A)) * (B - np.mean(B))) / (len(A) - ddof)

@dataclass
class VarComps(ABC):
    total_var: float  # var(A-B) or var(A)
    var_E: float  # var(E(A-B)) or var(E(A))
    E_var: float  # E(var(A-B)) or E(var(A))
    unbiased: bool = False
    satisfy_total_variance: bool = True

    def __post_init__(self):
        if not self.satisfy_total_variance:
            return
        total = self.var_E + self.E_var
        if not np.isclose(total, self.total_var):
            rtol = (total - self.total_var) / self.total_var
            raise ValueError(f"Total variance did not hold. components {self.to_dict()}, rtol={rtol}")

    @abstractmethod
    def to_dict(self) -> dict[str, float]:
        pass

    def __getitem__(self, key: str) -> float:
        return self.to_dict()[key]


@dataclass
class PairedVarComps(VarComps):
    def to_dict(self) -> dict[str, float]:
        return {
            "var(A-B)": self.total_var,
            "var(E(A-B))": self.var_E,
            "E(var(A-B))": self.E_var,
        }


@dataclass
class SingleVarComps(VarComps):
    def to_dict(self) -> dict[str, float]:
        return {
            "var(A)": self.total_var,
            "var(E(A))": self.var_E,
            "E(var(A))": self.E_var,
        }


class Paired:
    @staticmethod
    def from_samples(A: np.ndarray, B: np.ndarray) -> VarComps:
        """Calculate variance from
            Args:
                A, B: Success probabilities of shape (N questions, by K predictions for each question)
        """
        assert A.shape[0] == B.shape[0], "should be paired"
        return PairedVarComps(
            total_var=var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            var_E=var(mean(A, axis=1) - mean(B, axis=1)),
            E_var=mean(var(A, axis=1) + var(B, axis=1)),
            unbiased=False
        )
    
    @staticmethod
    def from_samples_unbiasedK(A: np.ndarray, B: np.ndarray) -> VarComps:
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = B.shape[1]
        return PairedVarComps(
            total_var=var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            var_E=var(mean(A-B, axis=1)) - mean(var(A, axis=1)/(kA-1) + var(B, axis=1)/(kA-1)),
            E_var=mean(var(A, axis=1)* (1 + 1/(kA-1)) + var(B, axis=1) * (1 + 1/(kB-1))),
            unbiased=False
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
        return PairedVarComps(
            total_var=np.clip(mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*cov(pA, pB), a_min=0, a_max=None),
            var_E=var(pA - pB),
            E_var=mean(pA*(1-pA) + pB*(1-pB)),
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
        return PairedVarComps(
            total_var=var_A_minus_B,
            var_E=0.0,  # by definition
            E_var=E_var_A_minus_B,  # by independence of the noise conditioned on a prompt
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
        return PairedVarComps(
            total_var=comps.total_var,
            var_E=comps.var_E,
            E_var=comps.E_var,
        )
    

class Single:
    @staticmethod
    def from_samples(A: np.ndarray) -> VarComps:
        return SingleVarComps(
            total_var=var(A),
            var_E=var(mean(A, axis=1)),
            E_var=mean(var(A, axis=1)),
            unbiased=False
        )
    
  
    @staticmethod
    def from_samples_unbiasedK(A: np.ndarray) -> VarComps:
        """
        only be unbiased in K, assumes N is large enough
        """
        kA = A.shape[1]
        N = A.shape[0]
        return SingleVarComps(
            total_var=var(A),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1)) - 1/(kA-1) * mean(var(A, axis=1)),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1)) * (1 + 1/(kA-1)),
            unbiased=False
        )

    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray) -> VarComps:
        return SingleVarComps(
            total_var=mean(pA)*(1-mean(pA)),
            var_E=var(pA),
            E_var=mean(pA*(1-pA)),
            unbiased=False
        )   
    
class SingleExperimental:
    @staticmethod
    def from_samples_naive(A: np.ndarray, M=100) -> VarComps:
        # draw M independent samples from the ith row of A, expanding kA to M for accurate direct estimations
        N, kA = A.shape
        A_expand_rows = np.array([np.random.choice(A[i], size=M, replace=True) for i in range(N)])
        # A_expand = np.array([np.random.choice(A.flatten, size=M, replace=True) for i in range(N)])
        return SingleVarComps(
            total_var=np.mean(var(A_expand_rows, axis=0)),
            var_E=float("nan") if kA == 1 else var(mean(A_expand_rows, axis=1)),
            # E_var is estimated directly
            E_var=float("nan") if kA == 1 else N * var(mean(A_expand_rows, axis=0)),
            unbiased=False,
            satisfy_total_variance=False, # total variance is not expected to hold
        )
    
    @staticmethod
    def from_samples_unbiased(A: np.ndarray) -> VarComps:
        kA = A.shape[1]
        N = A.shape[0]
        return SingleVarComps(
            total_var=var(A) * (1 + 1/(N*kA - 1)),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1)) - 1/(kA-1) * mean(var(A, axis=1)) + var(A)*1/(N*kA - 1),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1)* (1 + 1/(kA-1))),
            unbiased=True
        )
    
    @staticmethod
    def from_samples_unbiasedNK(A: np.ndarray) -> VarComps:
        """
        unbiased estimator in both N and K
        """
        kA = A.shape[1]
        N = A.shape[0]
        return SingleVarComps(
            total_var=mean(var(A, axis=1, ddof=0)) + var(mean(A, axis=1), ddof=1),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1), ddof=1) - mean(var(A, axis=1, ddof=1)) + mean(var(A, axis=1, ddof=0)),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1, ddof=1)),
            unbiased=True
        )
    
    @staticmethod
    def from_samples_unbiased_stratified(A: np.ndarray) -> VarComps:
        """
        Test our understanding of stratified sampling, where the basic estimator is too big by O(1/NK) since the sample is less random
        """
        kA = A.shape[1]
        N = A.shape[0]

        return SingleVarComps(
            total_var=float("nan") if kA == 1 else var(A) + 1/(N*kA) * mean(var(A, axis=1) * (1 + 1/(kA-1))),
            var_E=float("nan") if kA == 1 else var(A) + (1/(N*kA) - 1) * mean(var(A, axis=1) * (1 + 1/(kA-1))),
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1) * (1 + 1/(kA-1))),
            unbiased=True,
            satisfy_total_variance=True
        )