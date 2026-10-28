"""Action effects for the WESAD stress environment.

Every value comes from docs/simulator_spec.md (see "Effects by action" and the environment
decisions there). Several are labeled estimates, extrapolations or stand-ins; the labels live in
the spec and must go into the README.

Effects move a signal toward its subject baseline only: never past it and never away from it.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

BLOCK_MIN = 5                         # effect runs 5 minutes
FADE_MIN = 5                          # then the "after" window, 5 more minutes
LOCKOUT_MIN = BLOCK_MIN + FADE_MIN    # no new action until the effect is over (no overlaps)

ACTIONS = ["do_nothing", "paced_breathing", "calming_audio", "pmr",
           "haptic_pulse", "guided_meditation", "physiological_sigh"]

# The four reward signals and the features each effect is applied to ("twin" features included).
SIGNAL_FEATURES = {
    "hrv": ["chest_hrv_rmssd"],
    "hr": ["chest_hr", "wrist_hr"],
    "resp": ["chest_resp_rate"],
    "eda": ["wrist_eda_level", "chest_eda_level"],
}
REWARD_FEATURE = {"hrv": "chest_hrv_rmssd", "hr": "chest_hr", "resp": "chest_resp_rate", "eda": "wrist_eda_level"}
CALM_DIR = {"hrv": +1, "hr": -1, "resp": -1, "eda": -1}   # +1: calm = higher HRV; -1: calm = lower value
WEIGHTS = {"hrv": 0.35, "hr": 0.25, "eda": 0.20, "resp": 0.20}   # labeled design choice


@dataclass(frozen=True)
class Effect:
    kind: str                                   # "raw" (feature units), "spread" (baseline SDs), "share" (share of the excess over baseline)
    during: Union[float, Callable[[float], float]]   # value during the block; callable takes seconds since the action started
    after: Optional[float] = None               # value held for the after window; None = fade linearly to zero


def _share_from_half_lives(hl_action_s, hl_ref_s):
    """Share of the excess removed after tau seconds when the half-life is hl_action instead of hl_ref."""
    k = 1.0 / hl_action_s - 1.0 / hl_ref_s
    return lambda tau: 1.0 - 2.0 ** (-tau * k)


_BREATHING = {
    "hrv": Effect("raw", 25.86, 3.82),    # You et al. 2021 Table 1: +25.86 ms during, +3.82 ms after
    # breaths/min lower. During the block the drop (10.20 in the paper) is the instructed 6/min pacing itself, not a
    # benefit, so it is NOT counted (Lakshita's decision B); only the lasting drop after the block (2.60) counts.
    "resp": Effect("raw", 0.0, 2.60),
}

EFFECTS = {
    "paced_breathing": _BREATHING,
    "calming_audio": {
        "hrv": Effect("raw", 2.1),        # S07 Kumpulainen 2025 (at rest vs coffee-shop sound)
        "hr": Effect("raw", 1.7),
        "resp": Effect("raw", 1.2),
        "eda": Effect("share", _share_from_half_lives(101.3, 121.3)),   # S06 half-lives
    },
    "pmr": {"eda": Effect("spread", 1.02)},                # S14 eta^2 .51 -> Cohen f 1.02 (rough, labeled)
    "haptic_pulse": {"eda": Effect("spread", 0.54)},       # S15 Hedges' g (response counts, mismatch labeled)
    "guided_meditation": {"eda": Effect("share", 0.23)},   # S20 figure-read estimate (labeled)
    "physiological_sigh": _BREATHING,                      # labeled stand-in: borrowed from paced breathing
}


def effect_value(effect, minute_in_action):
    """Effect size for the 1-minute window that starts `minute_in_action` minutes after the action began.

    Minutes 0-4 are the block (the callable, if any, is evaluated at the window midpoint). Minutes 5-9
    are the after window: the effect's `after` value if it has one, otherwise a linear fade from the
    end-of-block value to zero. Anything else is 0.
    """
    m = minute_in_action
    if m < 0 or m >= LOCKOUT_MIN:
        return 0.0
    end_value = effect.during(BLOCK_MIN * 60.0) if callable(effect.during) else effect.during
    if m < BLOCK_MIN:
        return effect.during((m + 0.5) * 60.0) if callable(effect.during) else effect.during
    if effect.after is not None:
        return effect.after
    return end_value * max(0.0, 1.0 - ((m + 0.5) - BLOCK_MIN) / FADE_MIN)


def shift_toward_baseline(x, base_mean, base_sd, calm_dir, kind, magnitude):
    """Simulated value of x after an effect. Moves x toward base_mean, only from the stressed side."""
    if not np.isfinite(x):
        return x
    gap = (base_mean - x) * calm_dir          # > 0 when x is on the stressed side of baseline
    if gap <= 0:
        return x
    if kind == "raw":
        size = magnitude
    elif kind == "spread":
        size = magnitude * base_sd
    elif kind == "share":
        size = magnitude * gap
    else:
        raise ValueError(f"unknown effect kind {kind!r}")
    return x + calm_dir * min(size, gap)
