# %%
import matlab.engine
import numpy as np

from utils import __datadir__

# start MATLAB once
MATLAB = matlab.engine.start_matlab()

# add EGIFA toolbox recursively
MATLAB.addpath(MATLAB.genpath(str(__datadir__("EGIFA"))), nargout=0)

# print("SEDREAMS path:", MATLAB.which("SEDREAMS_GCIDetection"))
# print("DYPSA path:", MATLAB.which("dypsagoi"))


def _as_matlab_vector(x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)  # column vector
    return matlab.double(x.tolist())


def _as_numpy_vector(x):
    # MATLAB engine returns nested lists
    arr = np.array(x).ravel()
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr


def _as_numpy_gci(x):
    # Convert array of MATLAB indices to numpy indices
    gci = _as_numpy_vector(x)
    if gci.size == 0:
        return np.asarray([], dtype=int)

    # convert MATLAB indexing -> Python indexing
    return gci.astype(np.int64) - 1


def gci_estimates_from_dypsagoi(waveform, fs):
    """
    Run DYPSA (dypsagoi) on a speech waveform.

    MATLAB signature:
        [gci,goi,gcic,goic,gdwav,udash,crnmp] = dypsagoi(s,fs,opt)
    """

    gci, *_ = MATLAB.dypsagoi(
        _as_matlab_vector(waveform),
        float(fs),
        nargout=7,  # safer: request all outputs
    )

    return _as_numpy_gci(gci)


def gci_estimates_from_sedreams(waveform, fs, f0mean=150.0):
    """
    Run SEDREAMS on a speech waveform.

    MATLAB signature:
        [gci,MeanBasedSignal] = SEDREAMS_GCIDetection(wave,Fs,F0mean)
    """

    gci, _ = MATLAB.SEDREAMS_GCIDetection(
        _as_matlab_vector(waveform),
        float(fs),
        float(f0mean),
        nargout=2,
    )

    return _as_numpy_gci(gci)
