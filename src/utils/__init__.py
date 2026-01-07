import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Define runtime working environment variables
if "PROJECT_ROOT_PATH" not in os.environ:
    raise RuntimeError(
        "Working environment not activated. Please load .envrc file first"
    )


def __projectdir__(s=""):
    return pathlib.Path(os.environ["PROJECT_ROOT_PATH"]) / s


def __figuresdir__(s=""):
    return pathlib.Path(os.environ["PROJECT_FIGURES_PATH"]) / s


def __datadir__(s=""):
    return pathlib.Path(os.environ["PROJECT_DATA_PATH"]) / s


def __cachedir__(s=""):
    return pathlib.Path(os.environ["JOBLIB_CACHE_DIR"]) / s


def __experimentsdir__(s=""):
    return pathlib.Path(os.environ["PROJECT_EXPERIMENTS_PATH"]) / s


# Configure joblib's caching mechanism
# NOTE: joblib cannot cache arbitrary functions, because it cannot
# hash/pickle all possible input/output values. In particular, it
# isn't able to memoize functions that return `tfb.Bijector`s or
# or `tfd.Distribution`s. For these functions we use @__cache__
# which calculates the return value of the function once when it
# is called the first time and then caches it.
import joblib

__memory__ = joblib.Memory(__cachedir__("joblib"), verbose=2)


def __cache__(func):
    def cached_func():
        if not hasattr(cached_func, "_cache"):
            cached_func._cache = func()
        return cached_func._cache

    return cached_func


def batch_generator(generator, batch_size: int):
    """Convert a generator into batches"""
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # yield remaining items
        yield batch


def dyplot(
    results, names=None, runplot=False, traceplot=False, cornerplot=True
):
    display(results.summary())
    display("Information (bans)", nats_to_ban(results.information[-1]))

    if runplot:
        try:
            fig, axes = dynesty.plotting.runplot(results)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not make run plot: {e}")

    if traceplot:
        try:
            fig, axes = dynesty.plotting.traceplot(
                results, show_titles=True, labels=names, verbose=True
            )
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not make trace plot: {e}")

    if cornerplot:
        try:
            fg, ax = dynesty.plotting.cornerplot(results, labels=names)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not make corner plot: {e}")


def plot_power_spectrum_db(x, fs=16000, ax=None, label=None, **kwargs):
    """
    Compute and plot the power spectrum (in dB) onto a given matplotlib axis.

    Parameters:
        x : np.ndarray, shape (N,)
            Input signal.
        fs : int
            Sampling rate in Hz.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure is created.
        label : str, optional
            Label for legend.
        **kwargs : passed to `ax.plot()`

    Returns:
        freqs : np.ndarray
        power_db : np.ndarray
        line : matplotlib.lines.Line2D
    """
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    power = np.abs(X) ** 2 / N
    power_db = 10 * np.log10(power + 1e-6)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Power Spectrum (dB)")

    (line,) = ax.plot(freqs, power_db, label=label, **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True)

    if label:
        ax.legend()

    return freqs, power_db, line


def top_p_indices(prob, p=0.95):
    """
    Return the smallest set of indices whose sorted values sum to at least p.

    Parameters:
        prob : 1D array-like
            Non-negative values (e.g. probabilities or weights).
        p : float
            Cumulative threshold between 0 and 1.

    Returns:
        indices : np.ndarray
            Indices of top elements (in descending order of value).
    """
    prob = np.array(prob)
    if prob.ndim != 1:
        raise ValueError("Input must be a 1D array")
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")

    prob = prob / np.sum(prob)  # normalize
    sorted_idx = np.argsort(prob)[::-1]
    cumsum = np.cumsum(prob[sorted_idx])
    cutoff = np.searchsorted(cumsum, p)
    return sorted_idx[: cutoff + 1]


class time_this:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.walltime = self.end - self.start
        print(f"Walltime: {self.walltime:.3f} s")

import dynesty
import scipy.linalg
from dynesty import plotting


def normalize_power(d, return_multiplier=False):
    # If d contains multiple channels (columns), normalize each channel with respect to channel 1
    if len(d.shape) == 1:
        multiplier = np.sqrt(len(d) / np.dot(d, d))
    else:
        assert len(d.shape) == 2
        _, multiplier = normalize_power(d[:, 0], return_multiplier=True)
    normalized = multiplier * d
    return (normalized, multiplier) if return_multiplier else normalized


def nats_to_ban(x):
    return x * np.log10(np.exp(1))


def importance_weights(results):
    weights = np.exp(results.logwt - results.logz[-1])
    return weights


def get_posterior_moments(results):
    mean, cov = dynesty.utils.mean_and_cov(
        results.samples, importance_weights(results)
    )
    return mean, cov


def resample_equal(results, n):
    samples = dynesty.utils.resample_equal(
        results.samples, importance_weights(results)
    )
    i = np.random.choice(len(samples), size=n, replace=False)
    return samples[i, :]


def correlationmatrix(cov):
    """https://en.wikipedia.org/wiki/Correlation#Correlation_matrices"""
    sigma = np.sqrt(np.diag(cov))
    corr = np.diag(1 / sigma) @ cov @ np.diag(1 / sigma)
    return corr


def kl_mvn(to, fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    m_to, S_to = to
    m_fr, S_fr = fr

    d = m_fr - m_to

    c, lower = scipy.linalg.cho_factor(S_fr)

    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)

    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d)) / 2.0