import os

import jax
import numpy as np
import soundfile as sf

from utils.audio import resample


class OpenGlotI:
    # From OPENGLOT paper Table 1
    true_resonance_frequencies = {
        "a": [730, 1090, 2440, 3500],
        "e": [530, 1840, 2480, 3500],
        "i": [390, 1990, 2550, 3500],
        "o": [570, 840, 2410, 3500],
        "u": [440, 1020, 2240, 3500],
        "ae": [660, 1720, 2410, 3500],
    }

    @staticmethod
    def parse_wav(wav_file):
        vowel, modality, true_pitch = (
            wav_file.split("/")[-1].split(".")[-2].split("_")[:3]
        )
        vowel = vowel.lower()
        modality = modality.lower()
        true_pitch = int(true_pitch.lower()[:-2])  # strip 'Hz'
        return vowel, modality, true_pitch

    @staticmethod
    def read_wav(wav_file, target_fs, verbose=True):
        audio, original_fs = sf.read(
            wav_file,
            always_2d=False,
            dtype="float64" if jax.config.jax_enable_x64 else "float32",
        )

        # Split channels
        x = audio[:, 0]
        gf = audio[:, 1]

        x = resample(x, original_fs, target_fs)
        gf = resample(gf, original_fs, target_fs)

        # Normalize data to unit power
        scale = 1 / np.sqrt(np.mean(x**2))
        x = x * scale
        gf = gf * scale

        if verbose:
            print(
                f"Loaded {wav_file} ({len(x)} samples, {original_fs} Hz resampled to {target_fs} Hz); rescaled to unit power by {scale:.3f}"
            )

        return x, gf, original_fs


class OpenGlotII:
    true_resonance_frequencies = {
        "male": {
            "a": [752, 1095, 2616, 3169],
            "i": [340, 2237, 2439, 3668],
            "u": [367, 1180, 2395, 3945],
            "ae": [693, 1521, 2435, 3252],
        },
        "female": {
            "a": [848, 1210, 2923, 3637],
            "i": [379, 2634, 4256, 5395],
            "u": [420, 1264, 2714, 4532],
            "ae": [795, 1700, 2692, 3740],
        },
    }

    @staticmethod
    def parse_wav(wav_file):
        base = os.path.splitext(os.path.basename(wav_file))[0]

        gender, vowel, p, adduction = base.split("_")
        gender, vowel, true_pitch, adduction = (
            gender.lower(),
            vowel.lower(),
            int(p.rstrip("Hz").lower()),
            adduction.lower(),
        )

        return gender, vowel, true_pitch, adduction

    @staticmethod
    def read_wav(wav_file, target_fs, verbose=True):
        return OpenGlotI.read_wav(wav_file, target_fs, verbose)  # same