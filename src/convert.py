"""Unit conversions for WESAD raw signals.

Source: WESAD readme (wesad_readme.pdf, section II.1 and II.2). Constants and
formulas are copied from there; nothing is filtered, resampled, or filled.
"""
import numpy as np

VCC = 3.0
CHAN_BIT = 2 ** 16

# Native sampling rates (Hz), WESAD readme sections II.1-II.3.
FS_CHEST = 700
FS_WRIST = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}


def ecg_mv(signal):
    return (signal / CHAN_BIT - 0.5) * VCC


def eda_us(signal):
    return ((signal / CHAN_BIT) * VCC) / 0.12


def emg_mv(signal):
    return (signal / CHAN_BIT - 0.5) * VCC


def temp_c(signal):
    vout = (signal * VCC) / (CHAN_BIT - 1.0)
    rntc = ((10 ** 4) * vout) / (VCC - vout)
    lg = np.log(rntc)
    return -273.15 + 1.0 / (1.12764514e-3 + 2.34282709e-4 * lg + 8.77303013e-8 * lg ** 3)


def acc_g(signal, cmin=28000.0, cmax=38000.0):
    return (signal - cmin) / (cmax - cmin) * 2 - 1


def resp_pct(signal):
    return (signal / CHAN_BIT - 0.5) * 100


def wrist_acc_g(signal):
    """E4 ACC is in units of 1/64 g."""
    return signal / 64.0
