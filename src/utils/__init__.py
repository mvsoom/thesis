import matplotlib.pyplot as plt
import numpy as np


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
