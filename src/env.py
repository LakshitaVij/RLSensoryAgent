"""WESAD stress-regulation environment (Gymnasium), replay-based. Design and sources: docs/simulator_spec.md.

An episode replays one subject's recording from minute 20 (after the fixed 20-min baseline). Every
2 minutes the agent picks one of 7 actions. An action's effect is a temporary nudge toward baseline on
top of the real recording; the real recording itself is never changed. The reward is the weighted gain
over "do nothing" in distance from baseline. Reads only data/features/ and val_split.json.
"""
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effects as fx

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
START_MIN = 20          # first decision at minute 20 (end of the fixed baseline)
STEP_MIN = 2            # decision every 2 minutes; each step covers two 60 s windows
Z_CAP = 5.0             # labeled design choice (Step A)
CAP_N = 4               # max actions started in any rolling 60 min (labeled design choice)
CAP_WINDOW_MIN = 60
N_FEATURES = 16


def usable_subjects():
    """Train and validation subjects only (held-out and excluded subjects are never listed)."""
    v = json.loads((ROOT / "val_split.json").read_text())
    return {"train": list(v["train"]), "validation": list(v["validation"])}


class WesadStressEnv(gym.Env):
    def __init__(self, split="train", subjects=None, features_dir=FEATURES_DIR, weights=None, random_start=False, features=None,
                 z_cap=Z_CAP, effects=None, mirror_twins=True, held_out_ok=False, cap_n=CAP_N):
        super().__init__()
        allowed = usable_subjects()
        pool = allowed["train"] + allowed["validation"]
        if held_out_ok:                                      # ONLY src/final_eval.py; the final one-time held-out run
            pool = pool + list(json.loads((ROOT / "splits.json").read_text())["held_out"])
        self.subjects = list(subjects) if subjects is not None else list(allowed[split])
        bad = [s for s in self.subjects if s not in pool]
        if bad:
            raise ValueError(f"not a usable train/validation subject: {bad}")
        self.random_start = random_start    # training only: start each episode at a random decision time (the run still ends at the end of the recording)
        self.weights = dict(fx.WEIGHTS if weights is None else weights)
        self.z_cap = z_cap                                   # sensitivity-test switches (defaults = the base world)
        self.effects = fx.EFFECTS if effects is None else effects
        self.cap_n = cap_n                                   # actions allowed per rolling 60 min (sensitivity switch)
        self.mirror_twins = mirror_twins                     # False: an effect applies only to its primary feature, not the twin
        self.data = {s: self._load(s, Path(features_dir)) for s in self.subjects}
        self.names = self.data[self.subjects[0]]["names"]
        self.idx = {n: i for i, n in enumerate(self.names)}
        # which readings the agent SEES (default all 16). The reward and effects always use the full data; only the observation shrinks.
        if features is not None:
            unknown = [f for f in features if f not in self.idx]
            if unknown:
                raise ValueError(f"unknown features: {unknown}")
        self.obs_idx = [self.idx[f] for f in (features if features is not None else self.names)]
        self.action_space = spaces.Discrete(len(fx.ACTIONS))
        n_obs = len(self.obs_idx)
        low = np.array([-self.z_cap] * n_obs + [0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([self.z_cap] * n_obs + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.sid = None

    @staticmethod
    def _load(sid, d):
        f = pd.read_csv(d / f"{sid}_features.csv")
        b = pd.read_csv(d / f"{sid}_baseline.csv")
        names = list(b["feature"])
        assert len(names) == N_FEATURES, f"{sid}: expected {N_FEATURES} features"
        return {
            "names": names,
            "raw": f[[f"raw_{n}" for n in names]].to_numpy(float),
            "mean": b["baseline_mean"].to_numpy(float),
            "sd": b["baseline_sd"].to_numpy(float),
        }

    # ---- simulation of one window ----
    def _sim_row(self, w):
        """Raw feature values of window w with the active action's effect applied (if any)."""
        d = self.data[self.sid]
        row = d["raw"][w].copy()
        if self.active is not None:
            action, start = self.active
            m = w - start                                     # window index == minute
            for sig, eff in self.effects.get(action, {}).items():
                value = fx.effect_value(eff, m)
                if value <= 0:
                    continue
                for feat in self.signal_features(sig):
                    j = self.idx[feat]
                    row[j] = fx.shift_toward_baseline(row[j], d["mean"][j], d["sd"][j],
                                                      fx.CALM_DIR[sig], eff.kind, value)
        return row

    def signal_features(self, sig):
        feats = fx.SIGNAL_FEATURES[sig]
        return feats if self.mirror_twins else feats[:1]

    def _z(self, row):
        d = self.data[self.sid]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (row - d["mean"]) / d["sd"]
        z[~np.isfinite(d["sd"]) | (d["sd"] == 0)] = np.nan
        return np.clip(z, -self.z_cap, self.z_cap)

    # ---- bookkeeping ----
    def _lock_left(self):
        if self.active is None:
            return 0
        return max(0, self.active[1] + fx.LOCKOUT_MIN - self.minute)

    def _actions_last_hour(self):
        return sum(1 for s in self.starts if self.minute - s < CAP_WINDOW_MIN)

    def _mask(self):
        mask = np.ones(len(fx.ACTIONS), dtype=bool)
        if self._lock_left() > 0 or self._actions_last_hour() >= self.cap_n:
            mask[1:] = False                                   # only "do nothing"
        return mask

    def _obs(self):
        z = self._z(self._sim_row(self.minute - 1))[self.obs_idx]   # latest completed window, only the readings the agent sees
        nan_count = int(np.isnan(z).sum())
        z = np.nan_to_num(z, nan=0.0)
        n = self._actions_last_hour()
        book = [self._lock_left() / fx.LOCKOUT_MIN, min(n, self.cap_n) / self.cap_n, float(n >= self.cap_n)]
        return np.concatenate([z, book]).astype(np.float32), nan_count

    def _info(self, **extra):
        info = {"action_mask": self._mask(), "subject": self.sid, "minute": self.minute,
                "lockout_left_min": self._lock_left(), "actions_last_hour": self._actions_last_hour()}
        info.update(extra)
        return info

    # ---- Gymnasium API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "subject" in options:
            sid = options["subject"]
            if sid not in self.data:
                raise ValueError(f"subject {sid} is not in this environment")
        else:
            sid = self.subjects[int(self.np_random.integers(len(self.subjects)))]
        self.sid = sid
        self.minute = START_MIN
        if self.random_start:
            last_k = (self.data[sid]["raw"].shape[0] - (START_MIN + STEP_MIN)) // STEP_MIN
            self.minute = START_MIN + STEP_MIN * int(self.np_random.integers(last_k + 1))
        self.starts = []
        self.active = None
        obs, nan_count = self._obs()
        return obs, self._info(obs_nan=nan_count)

    def step(self, action):
        action = int(action)
        if not self._mask()[action]:
            raise ValueError(f"action {action} ({fx.ACTIONS[action]}) is masked at minute {self.minute}")
        if self.active is not None and self.minute >= self.active[1] + fx.LOCKOUT_MIN:
            self.active = None
        started = action != 0
        if started:
            self.starts.append(self.minute)
            self.active = (fx.ACTIONS[action], self.minute)

        d = self.data[self.sid]
        gain = {sig: 0.0 for sig in fx.WEIGHTS}
        skipped = 0
        windows = [self.minute + k for k in range(STEP_MIN)]
        for w in windows:
            z_real = self._z(d["raw"][w])
            z_sim = self._z(self._sim_row(w))
            for sig in fx.WEIGHTS:
                j = self.idx[fx.REWARD_FEATURE[sig]]
                if not (np.isfinite(z_real[j]) and np.isfinite(z_sim[j])):
                    skipped += 1                                # missing value: zero gain, no filling
                    continue
                gain[sig] += (abs(z_real[j]) - abs(z_sim[j])) / len(windows)   # average over the step's windows
        reward = float(sum(self.weights[s] * gain[s] for s in gain))

        self.minute += STEP_MIN
        if self.active is not None and self.minute >= self.active[1] + fx.LOCKOUT_MIN:
            self.active = None
        truncated = self.minute + STEP_MIN > d["raw"].shape[0]  # out of recording (time limit, not a real end)
        obs, nan_count = self._obs()
        return obs, reward, False, truncated, self._info(gain=gain, skipped_missing=skipped,
                                                          started=started, obs_nan=nan_count)
