# %%
from utils import __datadir__
from utils.matlab import (
    add_path_recursive,
    matlab_col,
    matlab_engine,
    matlab_idx_to_numpy0,
)

# start MATLAB once
MATLAB = matlab_engine()

# add EGIFA toolbox recursively
add_path_recursive(__datadir__("EGIFA"))

# print("SEDREAMS path:", MATLAB.which("SEDREAMS_GCIDetection"))
# print("DYPSA path:", MATLAB.which("dypsagoi"))


def gci_estimates_from_dypsagoi(waveform, fs):
    """
    Run DYPSA (dypsagoi) on a speech waveform.

    MATLAB signature:
        [gci,goi,gcic,goic,gdwav,udash,crnmp] = dypsagoi(s,fs,opt)
    """

    gci, *_ = MATLAB.dypsagoi(
        matlab_col(waveform),
        float(fs),
        nargout=7,  # safer: request all outputs
    )

    return matlab_idx_to_numpy0(gci)


def gci_estimates_from_sedreams(waveform, fs, f0mean=150.0):
    """
    Run SEDREAMS on a speech waveform.

    MATLAB signature:
        [gci,MeanBasedSignal] = SEDREAMS_GCIDetection(wave,Fs,F0mean)
    """

    gci, _ = MATLAB.SEDREAMS_GCIDetection(
        matlab_col(waveform),
        float(fs),
        float(f0mean),
        nargout=2,
    )

    return matlab_idx_to_numpy0(gci)
