"""Check 3: is the fixed first 20 minutes a sensible 'baseline'? (evaluation only; reads the label files.)

For each usable subject (train + validation; never the held-out 5), compare the first 20 one-minute
windows -- what the agent treats as 'normal' -- with the windows the dataset labels as baseline (label 1):
  - overlap: share of the first 20 windows whose majority label is baseline, and where the labeled baseline sits;
  - signal distance: for each reward signal, (mean of the first 20 windows - mean of the labeled-baseline windows)
    divided by that subject's baseline spread (from the first 20 windows). Small = our 'normal' matches the dataset's.
The labels are used here only for checking; the environment never reads them.

Usage: .venv/bin/python -I src/check_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from env import usable_subjects

SIGNALS = {"HRV": "chest_hrv_rmssd", "HR": "chest_hr", "breath": "chest_resp_rate", "sweat": "wrist_eda_level"}
LABEL_NAMES = {0: "none", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation", 5: "x", 6: "x", 7: "x"}


def main():
    allowed = usable_subjects()
    subjects = allowed["train"] + allowed["validation"]
    print(f"{'subj':5s}{'first-20 share labeled':>24s}{'labeled baseline sits at (min)':>34s}   distance in baseline spreads: "
          + "  ".join(f"{k:>6s}" for k in SIGNALS))
    rows = []
    for sid in sorted(subjects, key=lambda s: int(s[1:])):
        f = pd.read_csv(ROOT / "data" / "features" / f"{sid}_features.csv")
        b = pd.read_csv(ROOT / "data" / "features" / f"{sid}_baseline.csv").set_index("feature")
        lab = pd.read_csv(ROOT / "data" / "eval_only" / f"{sid}_window_labels.csv")["label_mode"].to_numpy()
        first = lab[:20]
        share = {LABEL_NAMES[k]: float(np.mean(first == k)) for k in np.unique(first)}
        main_label = max(share, key=share.get)
        is_base = np.flatnonzero(lab == 1)
        where = f"{is_base.min()}-{is_base.max() + 1}" if len(is_base) else "none"
        dist = {}
        for name, feat in SIGNALS.items():
            raw = f[f"raw_{feat}"].to_numpy()
            dist[name] = (raw[:20].mean() - raw[is_base].mean()) / b.loc[feat, "baseline_sd"] if len(is_base) else np.nan
        rows.append((sid, share.get("baseline", 0.0), dist))
        print(f"{sid:5s}{share.get('baseline', 0.0):>16.2f} (mostly {main_label}){where:>20s}      "
              + "  ".join(f"{dist[k]:6.2f}" for k in SIGNALS))
    m = np.mean([r[1] for r in rows])
    print(f"\nmean share of the first 20 min that is labeled baseline: {m:.2f}")
    print("mean |distance| per signal:", {k: round(float(np.nanmean([abs(r[2][k]) for r in rows])), 2) for k in SIGNALS})


if __name__ == "__main__":
    main()
