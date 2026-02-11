from warnings import warn

import numpy as np

from utils import praat, pyglottal
from utils.reskew import polarity_reskew


def gci_estimates_from_quickgci(waveform, fs):
    """Run QuickGCI on speech waveform

    WARNING: requires positive polarity! This is checked heuristically and imperfectly.

    Parameters are according to Serwy (2017).

    Most important ones:
    - Rotation of analytic signal by -90 degrees (theta=-pi/2) to get positive peaks at GCIs [theta]
    - Bandpass filter [fmin, fmax]
    """
    polarity = polarity_reskew(waveform, fs)

    if polarity < 0:
        warn(
            "Speech waveform thought to have negative polarity. QuickGCI might fail to find GCIs."
        )

    return pyglottal.quick_gci(
        waveform,
        fs=fs,
        fmin=20,
        fmax=400,
        theta=-np.pi / 2,
        reps=2,
    )


def gci_estimates_from_praat(waveform, fs):
    """Run Praat's pulse extraction algorithm on speech waveform"""
    return praat.get_pulses(waveform, fs)
