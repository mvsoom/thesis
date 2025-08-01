import os

import joblib

memory = joblib.Memory(
    os.environ["JOBLIB_CACHE_DIR"],
)
