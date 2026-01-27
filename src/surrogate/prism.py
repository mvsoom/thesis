# %%
import numpy as np
from tqdm import tqdm

from prism.svi import pad_waveforms
from surrogate import source
from utils import __memory__


def warp_time(t_ms, period_ms):
    return t_ms / period_ms


def dewarp_time(tau, period_ms):
    return tau * period_ms


def extract_train_data(lf_samples):
    for s in tqdm(lf_samples):
        tau = warp_time(s["t"], s["p"]["T0"])
        du = s["u"]
        oq = s["p"]["Oq"]
        yield tau, du, oq


def extract_test_data(lf_samples):
    for s in tqdm(lf_samples):
        tau = warp_time(s["t"], s["p"]["T0"])
        du = s["u"]
        log_prob_u = s["log_prob_u"]
        if np.isfinite(log_prob_u):
            yield tau, du, log_prob_u


@__memory__.cache
def get_train_data(n=None, width=None, offset=0):
    lf_samples = source.get_lf_samples(10_000)[offset : offset + n]

    triples = list(extract_train_data(lf_samples))

    waveforms = [(tau, du) for tau, du, oq in triples]
    oq = np.array([oq for _, _, oq in triples])

    X, y = pad_waveforms(waveforms, width=width)

    return X, y, oq


@__memory__.cache
def get_test_data(n=None, width=None, offset=0):
    lf_samples = source.get_lf_samples(10_000)[offset : offset + n]

    triples = list(extract_test_data(lf_samples))

    waveforms = [(tau, du) for tau, du, _ in triples]
    log_prob_u = np.array([oq for _, _, oq in triples])

    X, y = pad_waveforms(waveforms, width=width)

    return np.array(X), np.array(y), log_prob_u
