import numpy as np
from dataclasses import dataclass, replace

from abc import ABC, abstractmethod
from numpy import mean, var

def cov(A: np.ndarray, B: np.ndarray, ddof=0) -> float:
    return np.sum((A - np.mean(A)) * (B - np.mean(B))) / (len(A) - ddof)

@dataclass(frozen=True)
class VarComps(ABC):
    total_var: float
    """The total variance, shown as var(A-B) or var(A)"""
    var_E: float
    """The data variance, shown as var(E(A-B)) or var(E(A))"""
    E_var: float
    """The prediction variance, shown as E(var(A-B)) or E(var(A))"""
    unbiased: bool = False
    satisfy_total_var: bool = True

    def __post_init__(self):
        if self.satisfy_total_var and not np.isnan(self.var_E) and not np.isnan(self.E_var):
            total = self.var_E + self.E_var
            if not np.isclose(total, self.total_var):
                rtol = (total - self.total_var) / self.total_var
                raise ValueError(f"Total variance did not hold. components {self.to_dict()}, rtol={rtol}")

    def clipped(self):
        """Return a clipped version of this dataclass with values in valid range"""

        def clip(x):
            return x if np.isnan(x) else max(0, x)
        
        has_nan = any(np.isnan([self.total_var, self.var_E, self.E_var]))
        need_clip = any([
            self.total_var < 0,
            self.var_E < 0,
            self.E_var < 0
        ])
        
        return replace(
            self,
            total_var=clip(self.total_var),
            var_E=clip(self.var_E),
            E_var=clip(self.E_var),
            satisfy_total_var=not (has_nan or need_clip),
        )

    @abstractmethod
    def to_dict(self) -> dict[str, float]:
        pass

    def __getitem__(self, key: str) -> float:
        return self.to_dict()[key]


class PairedVarComps(VarComps):
    def to_dict(self) -> dict[str, float]:
        return {
            "var(A-B)": self.total_var,
            "var(E(A-B))": self.var_E,
            "E(var(A-B))": self.E_var,
        }


class SingleVarComps(VarComps):
    def to_dict(self) -> dict[str, float]:
        return {
            "var(A)": self.total_var,
            "var(E(A))": self.var_E,
            "E(var(A))": self.E_var,
        }


class Paired:
    """
    Namespace containing the recommended estimators
    """
    @staticmethod
    def from_samples(A: np.ndarray, B: np.ndarray) -> VarComps:
        """Calculate variance from samples
            Args:
                A, B: metric results (N questions, by K predictions for each question)
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
        """Like from_samples, but correcting for small K. When the estimator is unbiased in K, large N can average out the error
        this is especially important when the data noise var_E is smaller than E_var
        """
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = B.shape[1]
        bias = 1/(kA-1) * mean(var(A, axis=1)) + 1/(kB-1) * mean(var(B, axis=1))
        return PairedVarComps(
            total_var=var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            var_E=var(mean(A, axis=1) - mean(B, axis=1)) - bias,
            E_var=mean(var(A, axis=1)) + mean(var(B, axis=1)) + bias,
            unbiased=False
        )
    
    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray, pB: np.ndarray) -> VarComps:
        """
        same as from_samples, but useful on correctness evals that's already aggregated into a probability of correct rather than 0,1 samples.
        i.e. from_bernoulli_prob(pA, pA) == from_samples(A, B)
        if pA == A.sum(axis=1) and pB == B.sum(axis=1)

        Args:
                pA, pB: probability of correct on each of N questions
        """
        assert pA.size == pB.size
        pA = pA.flatten()
        pB = pB.flatten()
        return PairedVarComps(
            total_var=mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*cov(pA, pB),
            var_E=var(pA - pB),
            E_var=mean(pA*(1-pA)) + mean(pB*(1-pB)),
            unbiased=False
        )
    
    @staticmethod
    def from_bernoulli_prob_unbiasedK(pA: np.ndarray, pB: np.ndarray, kA: int, kB: int) -> VarComps:
        """Like from_bernoulli_prob, but correcting for small K.
        Args:
                pA, pB: metric results N questions
                kA, kB: the number of actual samples used to calculate pA and pB
        """
        assert pA.size == pB.size
        pA = pA.flatten()
        pB = pB.flatten()
        if kA == 1 or kB == 1:
            bias = np.nan
        else:
            bias = 1/(kA-1) * mean(pA*(1-pA)) + 1/(kB-1) * mean(pB*(1-pB)) 
        return PairedVarComps(
            total_var=mean(pA)*(1-mean(pA)) + mean(pB)*(1-mean(pB)) - 2*cov(pA, pB),
            var_E=var(pA - pB) - bias,
            E_var=mean(pA*(1-pA)) + mean(pB*(1-pB)) + bias,
            unbiased=False
        )


class Unpaired:
    """
    Namespace containing the same methods as Paired.
    These can be useful as a baseline, but has less power compared to Paired
    """
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
        kA = A.shape[1]
        N = A.shape[0]
        bias = 1/(kA-1) * mean(var(A, axis=1)) 
        return SingleVarComps(
            total_var=var(A),
            var_E=float("nan") if kA == 1 else var(mean(A, axis=1)) - bias,
            E_var=float("nan") if kA == 1 else mean(var(A, axis=1)) + bias,
            unbiased=False
        )

    @staticmethod
    def from_bernoulli_prob(pA: np.ndarray) -> VarComps:
        pA = pA.flatten()
        return SingleVarComps(
            total_var=mean(pA)*(1-mean(pA)),
            var_E=var(pA),
            E_var=mean(pA*(1-pA)),
            unbiased=False
        )   
    
    @staticmethod
    def from_bernoulli_prob_unbiasedK(pA: np.ndarray, kA: int) -> VarComps:
        pA = pA.flatten()
        if kA == 1:
            bias = np.nan
        else:
            bias = 1/(kA-1) * mean(pA*(1-pA)) 
        return SingleVarComps(
            total_var=mean(pA)*(1-mean(pA)),
            var_E=var(pA) - bias,
            E_var=mean(pA*(1-pA))  + bias,
            unbiased=False
        )


class PairedExperimental:
    """
    Namespace containing experimental estimators. Useful for testing, and for understanding the setup and bias, but not recommended for most.
    """
    @staticmethod
    def from_bernoulli_prob_self(pA: np.ndarray, K: np.ndarray) -> VarComps:
        """Calculate the variance components of A-A', A' contains the exact same set of samples as A
        Used to check that the two ways to compute this agrees with each other
            Args:
                pA: Success probabilities of shape (n_samples, 1)
                K: number of samples for bias correction
        """
        assert pA.shape == K.shape
        assert all(K > 1), "need more than 1 sample per problem"
        pA = pA.flatten()
        # a direct computation using leave one out sampling
        covAA = mean((pA*pA - 1/K*pA)*(K/(K-1))) - mean(pA)**2
        var_A_minus_B = 2*mean(pA)*(1-mean(pA)) - 2*covAA

        # using 0
        E_var_A_minus_B = 2*mean(pA*(1-pA) * (1+1/(K-1)))
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
        """
        Actually resamples M random pairs, used to test when we reach the same RMS as the better estimator
        """
        assert A.shape[0] == B.shape[0]
        # For each row i, generate M samples of A_ij - B_ik where j and k are randomly drawn
        AB_diff_samples = np.array([
            np.random.choice(A[i], size=M, replace=True) - np.random.choice(B[i], size=M, replace=True) 
            for i in range(A.shape[0])
        ])
        return Unpaired.from_samples(AB_diff_samples)
    
    @staticmethod
    def from_samples_balanced_diff(A: np.ndarray, B: np.ndarray) -> VarComps:
        """
        For each row i, generate A_ij - B_ik where j and k covers all columns of A and B respectively
        should agree with the basic from_samples estimator
        """
        assert A.shape[0] == B.shape[0]
        N = A.shape[0]
        kA = A.shape[1]
        kB = B.shape[1]
        A_minus_B = np.zeros((N, kA*kB)) 
        for i in range(N):
            diffs = A[i][:, np.newaxis] - B[i][np.newaxis, :]
            A_minus_B[i, :] = diffs.flatten()
        comps = Unpaired.from_samples(A_minus_B)
        return PairedVarComps(
            total_var=comps.total_var,
            var_E=comps.var_E,
            E_var=comps.E_var,
        )

    @staticmethod
    def from_samples_unbiasedK_off1(A: np.ndarray, B: np.ndarray) -> VarComps:
        """
        Used to check if the slightly biased correction 1/kA (instead of 1/(kA-1)) is observably worse -- it is for small kA. So not recommended.
        """
        assert A.shape[0] == B.shape[0] # paired data
        kA = A.shape[1]
        kB = B.shape[1]
        bias = 1/(kA) * mean(var(A, axis=1)) + 1/(kB) * mean(var(B, axis=1))
        return PairedVarComps(
            total_var=var(A) + var(B) - 2 * cov(mean(A, axis=1), mean(B, axis=1)),
            var_E=var(mean(A, axis=1) - mean(B, axis=1)) - bias,
            E_var=mean(var(A, axis=1)) + mean(var(B, axis=1)) + bias,
            unbiased=False
        )


class UnpairedExperimental:
    """
    Namespace containing experimental estimators. Useful for testing, and for understanding the setup and bias, but not recommended for most.
    """
    @staticmethod
    def from_samples_naive(A: np.ndarray, M=100) -> VarComps:
        # draw M independent samples from the ith row of A, expanding kA to M for accurate direct estimations
        N, kA = A.shape
        A_expand_rows = np.array([np.random.choice(A[i], size=M, replace=True) for i in range(N)])
        return SingleVarComps(
            total_var=np.mean(var(A_expand_rows, axis=0)),
            var_E=float("nan") if kA == 1 else var(mean(A_expand_rows, axis=1)),
            E_var=float("nan") if kA == 1 else N * var(mean(A_expand_rows, axis=0)),
            unbiased=False,
            satisfy_total_variance=False, # total variance is not expected to hold
        )
    
    @staticmethod
    def from_samples_unbiasedNK(A: np.ndarray) -> VarComps:
        """
        unbiased estimator in both N and K. The challenge is to have an estimator that passes the unbiasedness test even on small N
        This is not recommended, since the unbiased one has higher rms.
        When N is too small such that the bias matters, all the estimators are too inaccurate to be useful.
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
        Test our understanding of stratified sampling where K samples are draw from each question.
        Here the basic estimator is too small by O(1/NK) since the sample is less random
        For example, if p_1=0.9, p_2=0.1, the true total var is still 0.25, but a sample estimator
        will return (0.81+0.01)*0.25 instead and there is nothing we can do if K=1.
        For K > 1, the stratified sample [~1, ~1, ..; ~0, ~0, ..] needs this estimator to get the right answer
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