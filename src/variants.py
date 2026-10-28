"""Sensitivity-test variants of the practice world (one labeled design choice changed at a time).

VARIANTS[name] -> keyword arguments for WesadStressEnv. "base" is the world used everywhere else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effects as fx
from effects import Effect


def _with(action, signal, effect):
    e = {a: dict(sigs) for a, sigs in fx.EFFECTS.items()}
    e[action] = dict(e[action])
    e[action][signal] = effect
    if action == "paced_breathing":                       # the sigh is a labeled copy of paced breathing
        e["physiological_sigh"] = e[action]
    return e


VARIANTS = {
    "base": {},
    "equal_weights": {"weights": {"hrv": 0.25, "hr": 0.25, "eda": 0.25, "resp": 0.25}},
    "heart_heavier": {"weights": {"hrv": 0.40, "hr": 0.25, "eda": 0.20, "resp": 0.15}},
    "zcap3": {"z_cap": 3.0},
    "zcap10": {"z_cap": 10.0},
    "pmr_half": {"effects": _with("pmr", "eda", Effect("spread", 0.51))},
    "meditation_half": {"effects": _with("guided_meditation", "eda", Effect("share", 0.115))},
    "no_twins": {"mirror_twins": False},
    "breathing_during_counted": {"effects": _with("paced_breathing", "resp", Effect("raw", 10.20, 2.60))},
    "cap2": {"cap_n": 2},
    "cap6": {"cap_n": 6},
    "audio_flat17": {"effects": _with("calming_audio", "eda", Effect("share", 1 - 101.3 / 121.3))},
}
