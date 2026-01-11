"""Physical constants used in the program"""
import numpy as np

_ZERO = 1e-3

# We work up until 5 kHz
FS_KHZ = 10.
FS_HZ = 1000*FS_KHZ

# Sampling power spectra
DF = 1. # Hz
TIMIT_FS_HZ = 16000
def spectrum_frequencies(fs=FS_HZ):
    return np.arange(0, fs/2, DF)


# The boundary factor `c` (Riutort-Mayol 2020)
BOUNDARY_FACTOR = 2.0

# Determined empirically from Serwy (2017)
_MIN_FUNDAMENTAL_FREQUENCY_HZ = 50  # Hz
_MAX_FUNDAMENTAL_FREQUENCY_HZ = 500  # Hz

MIN_PERIOD_LENGTH_MSEC = 1000 / _MAX_FUNDAMENTAL_FREQUENCY_HZ
MAX_PERIOD_LENGTH_MSEC = 1000 / _MIN_FUNDAMENTAL_FREQUENCY_HZ

MEAN_PERIOD_LENGTH_MSEC = 7.141548  # From APLAWD

# From Hawks+ (1995) Fig. 1
MIN_FORMANT_FREQ_HZ = 100.0
MAX_FORMANT_FREQ_HZ = 5000.0

# Bounds for VTR frequency
MIN_X_HZ = 100.0
MAX_X_HZ = FS_HZ / 2

# More precise bounds are in `vtr/band_bounds.py`
MIN_FORMANT_BAND_HZ = 10.0
MAX_FORMANT_BAND_HZ = 500.0

# Bounds for VTR bandwidth
MIN_Y_HZ = 20.0
MAX_Y_HZ = FS_HZ / 2

# The reference formant F and B tolerances for loosely based on Pinson (1963)
SIGMA_FB_REFERENCE_HZ = np.array([5.0, 10.0, 15.0])

# Reference spectral tilt for the filter envelope and sensitivity
FILTER_SPECTRAL_TILT_DB = -2.0
SIGMA_TILT_DB = (
    2.0 + 6.0
)  # Take into account the sloppiness (+6 dB) of our estimator

# Expected value of the VT impulse response energy
IMPULSE_RESPONSE_ENERGY_MSEC = 1.0

# In APLAWD and VTRFormants, reject a voiced group or vowel segment
# if it has less than `MIN_NUM_PERIODS` pitch periods
MIN_NUM_PERIODS = 3

# Determined empirically from Holmberg (1988)
MIN_DECLINATION_TIME_MSEC = 0.2
MAX_DECLINATION_TIME_MSEC = 4.5

# Lower bounds for the open quotient are based on Drugman (2019, Table 1) and Henrich (2005)
MIN_OQ = 0.30
MAX_OQ = 1 - _ZERO

# Bounds for the asymmetry coefficient are based on Doval (2006, p. 5)
MIN_AM = 0.5
MAX_AM = 1 - _ZERO

# Bounds for the return phase quotient are based on Doval (2006, p. 5)
MIN_QA = _ZERO
MAX_QA = 1 - _ZERO

# Bounds for the generic LF model parameters assuming `Ee == 1`
LF_GENERIC_PARAMS = ("T0", "Oq", "am", "Qa")
LF_GENERIC_BOUNDS = {
    "T0": [MIN_PERIOD_LENGTH_MSEC, MAX_PERIOD_LENGTH_MSEC],
    "Oq": [MIN_OQ, MAX_OQ],
    "am": [MIN_AM, MAX_AM],
    "Qa": [MIN_QA, MAX_QA],
}

LF_T_PARAMS = ("T0", "Te", "Tp", "Ta")


# Noise floor
def db_to_power(x, ref=1.):
    return ref*10**(x/10)

NOISE_FLOOR_DB = (
    -60.0
)  # Assuming power normalized data, i.e., the power is unity
NOISE_FLOOR_POWER = db_to_power(NOISE_FLOOR_DB)