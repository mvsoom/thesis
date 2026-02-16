import numpy as np


def kl_div(p, q, eps=1e-16):
    """Compute the Kullback-Leibler divergence D_KL(p || q)"""
    p = normalize_weights(p, eps)
    q = normalize_weights(q, eps)
    return np.sum(p * (np.log(p) - np.log(q)))


def normalize_weights(weights, eps=1e-16):
    logt = np.log(weights + eps)
    logt -= logt.max()
    exp_l = np.exp(logt)
    w = exp_l / (exp_l.sum() + eps)
    return w


def average_list_of_dicts(dicts, keys=None):
    """Average a list of dictionaries with numeric values."""
    if keys is None:
        keys = set().union(*(d.keys() for d in dicts))

    avg_dict = {}
    for key in keys:
        try:
            avg_dict[key] = np.mean([d[key] for d in dicts if key in d]).item()
        except:
            pass

    return avg_dict


def weighted_pitch_error(f0, weights, true_pitch, eps=1e-16):
    """Weighted absolute and root-mean-square error between estimated and true pitch

    Each element in `f0` is weighted by the corresponding element in `weights`.
    """
    w = normalize_weights(weights)

    err = f0 - true_pitch
    est = (w * f0).sum()
    wmae = (w * np.abs(err)).sum()
    wrmse = np.sqrt((w * err**2).sum())

    return est, wmae, wrmse
