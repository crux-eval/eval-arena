import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def gaussian_tail_exact(z: np.ndarray) -> np.ndarray:
    """
    Gaussian tail (CLT approximation): P(Z > z) for standard normal.
    """
    return 1 - stats.norm.cdf(z)


def gaussian_chernoff_bound(z: np.ndarray) -> np.ndarray:
    """
    Chernoff/Gaussian upper bound: \exp(-z^2/2)
    """
    return np.exp(-z**2 / 2)


def _hoeffding_bound(t: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Wainwright (2.10)
    \Pr[\sum_{i=1}^n X_i > t] ≤ \exp(-\frac{t^2}{2\sum_{i=1}^n \sigma_i^2})
    """
    return np.exp(-t**2 / (2*np.sum(sigma2)))


def hoeffding_bound_z(z, SE, N, sigma2) -> np.ndarray:
    """
    substitute t = z SE N
    """
    return _hoeffding_bound(z * SE * N, sigma2)


def _bernstein_bound(delta: np.ndarray, X2: np.ndarray, N: int, b: float = 1.0) -> np.ndarray:
    """
    Wainwright 2.22b
    \Pr[\sum_{i=1}^n X_i >= n \delta ] ≤ \exp(- n \frac{\delta^2}{2(1/n \sum_i \E[X_i^2] + b\delta /3)})
    """
    delta = np.asarray(delta)
    X2_mean = np.mean(X2)
    denom = 2 * (X2_mean + (b * delta) / 3.0)
    exponent = -N * (delta ** 2) / denom
    return np.exp(exponent)


def bernstein_bound(z, SE, N, X2, b=1.0) -> np.ndarray:
    """
    substitute delta = z SE
    """
    return _bernstein_bound(z * SE, X2, N, b)


def bennett_bound(z, SE, N, X2, b=1.0) -> np.ndarray:
    """
    Bennett's inequality (tighter than Bernstein for some regimes)
    """
    delta = z * SE
    delta = np.asarray(delta)
    X2_mean = np.mean(X2)
    v = X2_mean

    # Bennett: exp(-n * v / b^2 * h(b * delta / v))
    # where h(u) = (1+u) log(1+u) - u
    u = b * delta / v
    h = (1 + u) * np.log(1 + u + 1e-10) - u
    return np.exp(-N * v / (b**2) * h)

def binomaial_exact(t: np.ndarray, N: int, p: float):
    """
    Pr[Binom(N, p) >= t]
    """
    return 1 - stats.binom.cdf(t - 1, N, p)


def _binomial_bound(delta: np.ndarray, N: int, p: float):
    """
    Wainwright Exercise 2.9
    P[\sum_i X_i <= delta N] ≤ \exp(-N D(delta||p))
    """
    kl = delta * np.log(delta / p) + (1 - delta) * np.log((1 - delta) / (1 - p))
    return np.exp(-N * kl)

def binomial_bound(t: np.ndarray, N: int, p: float):
    """
    Wainwright Exercise 2.9
    P[\sum_i X_i >= t] ≤ \exp(-N D((N-t)/N||p))
    """
    delta = 1 - t / N
    return _binomial_bound(delta, N, p)

def plot_tail_bounds(sigma: float = 0.5, N: int = 30, M: float = 1.0,
                     z_max: float = 5.0, n_points: int = 500):
    """
    Create both regular and log-scale plots of tail bounds.

    Args:
        sigma: standard deviation of each RV (max 0.5 for Bernoulli)
        N: number of samples
        M: bound on |X - μ|
        z_max: maximum z-score to plot
        n_points: number of points for plotting
    """
    z = np.linspace(0.01, z_max, n_points)

    # Standard error
    p = 0.5
    sigma = np.sqrt(p*(1-p))
    SE = sigma / np.sqrt(N)

    # Compute bounds
    gaussian_exact = gaussian_tail_exact(z)
    gaussian_chernoff = gaussian_chernoff_bound(z)

    # Hoeffding with sigma^2 = 0.25 (max variance for Bernoulli)
    hoeffdingz = hoeffding_bound_z(z, SE, N, np.ones(N)*sigma*sigma)

    # Bernstein and Bennett bounds
    X2 = np.ones(N) * sigma**2  # E[X_i^2] for centered RVs
    bernstein = bernstein_bound(z, SE, N, X2, b=M)
    
    binom = binomaial_exact(z*SE*N + N*p, N, p=p)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Regular Scale", "Log Scale"),
        horizontal_spacing=0.1
    )

    print(binomial_bound(z*SE*N + N*p, N, p=p))
    traces = [
        ('Gaussian (exact CLT)', gaussian_exact, "red", 'solid'),
        ('Gaussian Chernoff exp(-z²/2)', gaussian_chernoff, "red", 'dash'),
        (f'Hoeffding', hoeffdingz, "blue", 'solid'),
        (f'Bernstein', bernstein, "green", 'solid'),
        (f'Binom exact', binomaial_exact(z*SE*N + N*p, N, p=p), "green", 'solid'),
        (f'Binom bound', binomial_bound(z*SE*N + N*p, N, p=p), "green", 'solid'),
    ]

    for name, y_data, color, dash in traces:
        # Regular scale
        fig.add_trace(go.Scatter(
            x=z, y=y_data, name=name,
            line=dict(color=color, width=2, dash=dash),
            legendgroup=name
        ), row=1, col=1)

        # Log scale
        fig.add_trace(go.Scatter(
            x=z, y=y_data, name=name,
            line=dict(color=color, width=2, dash=dash),
            legendgroup=name, showlegend=False
        ), row=1, col=2)

    # Update y-axis to log scale for second plot
    fig.update_yaxes(type="log", row=1, col=2)

    # Add axis labels
    fig.update_xaxes(title_text="z-score", row=1, col=1)
    fig.update_xaxes(title_text="z-score", row=1, col=2)
    fig.update_yaxes(title_text="P(Z > z)", row=1, col=1)
    fig.update_yaxes(title_text="P(Z > z)", row=1, col=2)

    fig.update_layout(
        width=1200,
        height=500,
        legend=dict(x=1.02, y=1, xanchor='left'),
        title_text=f"Tail Bounds Comparison (σ={sigma}, N={N}, M={M})"
    )

    return fig


if __name__ == "__main__":
    # Default parameters
    sigma = 0.5  # max variance for Bernoulli (p=0.5)
    N = 100      # number of samples
    M = 1.0      # bound on |X - μ| for [0,1] RVs

    # Combined plot with default params
    fig = plot_tail_bounds(sigma=sigma, N=N, M=M)

    # Save to OUTPUT directory
    import os
    os.makedirs('OUTPUT', exist_ok=True)
    fig.write_html('OUTPUT/tailbounds.html')
    print(f"Generated plots with σ={sigma}, N={N}, M={M}")
    print(f"Plot saved to OUTPUT/tailbounds.html")
