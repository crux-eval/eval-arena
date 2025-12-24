from abc import ABC, abstractmethod

import numpy as np
from numpy import mean
import pandas as pd

from estimators import Paired, Unpaired, UnpairedExperimental, VarComps 


class GenerativeModel(ABC):
    idx: np.ndarray # data index of the model to support pairing 
    N_pop: int # population size
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
        return np.random.choice(self.N_pop, size=self.N, replace=True)


class BernoulliModel(GenerativeModel):
    def __init__(self, pA: np.ndarray, N: int, K: int):
        self.pA = pA
        self.N_pop = self.pA.shape[0]
        self.N = N
        self.idx = self.sample_idx()
        self.K = K

    def sample_preds(self) -> np.ndarray:
        p = self.pA[self.idx]
        A = np.random.rand(self.N, self.K)
        A = np.where(A < p, 1, 0)
        return A

    def true_vars(self) -> dict[str]:
        return Unpaired.from_bernoulli_prob(self.pA)


class BernoulliModelStratified(BernoulliModel):
    """
    Not resampling the data, for measuring prediction variance directly
    """
    def __init__(self, pA: np.ndarray, N: int, K: int):
        assert N == pA.shape[0], f"For stratified sampling, N must equal Npop (got N={N}, Npop={pA.shape[0]})"
        super().__init__(pA, N, K)

    def sample_idx(self):
        return np.arange(self.N)


class BootstrapModel(GenerativeModel):
    """
    given A with size N_pop by K_pop. select rows of A iid with replacement (boostrap) N times,
    then for each selected row,  draw K samples from that row, also iid with replacement.
    useful to evaluating the SE of variance.
    """
    def __init__(self, A: np.ndarray, N: int, K: int):
        self.A = A
        self.N_pop = self.A.shape[0]
        self.N = N
        self.idx = self.sample_idx()
        self.K = K

    def sample_preds(self) -> np.ndarray:
        N_pop, K_pop = self.A.shape
        col_idx = np.random.choice(K_pop, size=(self.N, self.K), replace=True)
        return self.A[self.idx[:, None], col_idx]

    def true_vars(self) -> dict[str]:
        return Unpaired.from_samples(self.A)

    
class TestEstimator(ABC):
    @staticmethod
    def tscores(star: float, estimates: np.ndarray):
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
    
    @staticmethod
    def evaluate_estimator(truth: VarComps, ests: list[VarComps], verbose=False):
        if verbose:
            print(f"{truth=}")
        rel_error_stats = []
        for comp in [key for key in truth.to_dict()]:
            star = truth[comp]
            ests_comp = np.array([e[comp] for e in ests])

            t = TestEstimator.tscores(star, ests_comp)
            rel_errors = (ests_comp - star) / star
            rel_error_stats.append({
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
                TestEstimator.plot_distribution_vs_star(ests_comp, star)

        results = pd.DataFrame(rel_error_stats)
        if verbose:
            pd.options.display.float_format = '{:.3g}'.format
            display(results)

        return results


class TestPairedEstimators(TestEstimator):
    def get_estimates(self, modelA: BernoulliModel, modelB: BernoulliModel, estimator, attempts=20) -> list[VarComps]:
        """Get estimates from paired models with synchronized indices"""
        ests = []
        for i in range(attempts):
            # Synchronize indices for paired sampling
            shared_idx = modelA.sample_idx()
            modelA.idx = shared_idx
            modelB.idx = shared_idx

            A = modelA.sample_preds()
            B = modelB.sample_preds()
            vhat = estimator(A, B)
            ests.append(vhat)
        return ests

    def estimator_results(self, truth, modelA, modelB, estimators, verbose=False, attempts=1000):
        results_table = pd.DataFrame()
        for estimator in estimators:
            ests: list[VarComps] = self.get_estimates(modelA, modelB, estimator, attempts=attempts)
            results = TestEstimator.evaluate_estimator(truth, ests, verbose)
            results["estimator"] = estimator.__name__
            results_table = pd.concat([results_table, results], ignore_index=True)

        return results_table


class TestSingleEstimators(TestEstimator):
    def get_estimates(self, model: GenerativeModel, estimator, attempts: int) -> list[VarComps]:
        ests = []
        for i in range(attempts):
            model.idx = model.sample_idx()
            A = model.sample_preds()
            vhat = estimator(A)
            ests.append(vhat)
        return ests
    
    def estimator_results(self, truth, model, estimators, verbose=False, attempts=1000):
        results_table = pd.DataFrame()
        for estimator in estimators:
            ests: list[VarComps] = self.get_estimates(model, estimator, attempts=attempts)
            results = TestEstimator.evaluate_estimator(truth, ests, verbose)
            results["estimator"] = estimator.__name__
            results_table = pd.concat([results_table, results], ignore_index=True)
        return results_table


# ============================================================================
# actual test functions
# ============================================================================

def _table_single(pA, K, N):
    model = BernoulliModel(pA, K=K, N=N)
    truth = model.true_vars()
    estimators = [
        Unpaired.from_samples,
        Unpaired.from_samples_unbiasedK,
        UnpairedExperimental.from_samples_unbiasedNK,
    ]
    t = TestSingleEstimators().estimator_results(truth, model, estimators, verbose=False)
    return t


def _table_paired(pA, pB, K, N):
    modelA = BernoulliModel(pA, K=K, N=N)
    modelB = BernoulliModel(pB, K=K, N=N)
    truth = Paired.from_bernoulli_prob(pA, pB)
    estimators_to_test = [
        Paired.from_samples,
        Paired.from_samples_unbiasedK,
    ]
    return TestPairedEstimators().estimator_results(truth, modelA, modelB, estimators_to_test, verbose=False)


def _subtest_stratified(pA, K):
    model = BernoulliModelStratified(pA, K=K, N=pA.shape[0])
    truth = model.true_vars()
    estimators_to_test = [
        UnpairedExperimental.from_samples_unbiased_stratified
    ]
    t = TestSingleEstimators().estimator_results(truth, model, estimators_to_test, verbose=False)
    t_unbiased = t[t["unbiased"] == True]
    if not all(t_unbiased["t_score"].abs() < 4):
        raise ValueError("some unbiased estimator appears to be biased")

    if any(t[t["comp"] == "var(A)"]["rms"] > 0.25):
        raise ValueError("some total variance has unacceptable relative error")


def test_single():
    pA = np.array([[0.3], [0.1]])
    t = _table_single(pA, K=10, N=2)
    t_unbiased = t[t["unbiased"] == True]
    assert all(t_unbiased["t_score"].abs() < 4)

    pA = np.array([[0.1], [0.9]])
    t = _table_single(pA, K=10, N=2)
    t_unbiased = t[t["unbiased"] == True]
    assert all(t_unbiased["t_score"].abs() < 4)

    pA = np.random.beta(0.2, 0.8, (200, 1))
    t = _table_single(pA, 10, N=100)
    assert all(t["rms"] < 0.26)

    pA = np.random.beta(0.2, 0.8, (400, 1))
    t = _table_single(pA, 20, N=400)
    assert all(t["rms"] < 0.13)

    pA = 0.2 + 0.4 * np.random.rand(1000, 1)
    t = _table_single(pA, K=10, N=500)
    assert all(t[t["estimator"] != "from_samples"]["rms"] < 0.25)
    assert all(t[t["estimator"] != "from_samples"]["t_score"] < 4)
    assert any(t[t["estimator"] == "from_samples"]["rms"] > 0.25)

    pA = np.random.beta(2, 1, (200, 1))
    t = _table_single(pA, 10, N=100)
    assert all(t[t["estimator"] != "from_samples"]["rms"] < 0.25)


def test_stratified():
    pA = np.array([[0.9], [0.1]])
    _subtest_stratified(pA, 2)
    _subtest_stratified(pA, 10)

    pA = np.array([[0.4], [0.1]])
    _subtest_stratified(pA, 20)


def test_paired():
    """Test paired model estimators."""
    # unrelated Beta distributions should have reasonable error
    pA = np.random.beta(0.3, 0.7, (100, 1))
    pB = np.random.beta(0.2, 0.8, (100, 1))
    t = _table_paired(pA, pB, K=10, N=100)
    print(t)
    assert all(t["rms"] < 0.25)
    assert any(t["rms"] > 0.1)

    # Larger sample size - should have smaller error
    pA = np.random.beta(0.3, 0.7, (500, 1))
    pB = np.random.beta(0.2, 0.8, (500, 1))
    t = _table_paired(pA, pB, K=40, N=400)
    assert all(t["rms"] < 0.13)
    assert all(t[t["comp"] == "var(A-B)"]["rms"] < 0.05)

    # Correlated models - unbiased should outperform biased for var(E(A-B))
    pA = np.random.beta(0.3, 0.7, (500, 1))
    pB = np.clip(pA + 0.1 * (np.random.randn(*pA.shape) - 0), 0, 1)
    t = _table_paired(pA, pB, K=200, N=500)

    # In this test, the naive estimator should have higher error for var(E(A-B))
    est_varE = t[(t["comp"] == "var(E(A-B))") & (t["estimator"] == Paired.from_samples.__name__)]["rms"].values[0]
    est2_varE = t[(t["comp"] == "var(E(A-B))") & (t["estimator"] == Paired.from_samples_unbiasedK.__name__)]["rms"].values[0]
    assert est_varE > est2_varE


if __name__ == "__main__":
    # np.random.seed(42)
    test_single()
    test_stratified()
    test_paired()