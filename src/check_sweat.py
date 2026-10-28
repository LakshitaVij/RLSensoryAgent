"""Check 2: do the wrist sweat level and the cleaner chest sweat level agree? (evaluation only; no labels used)

Per usable subject, over the windows the agent sees (minute 20 onward): Pearson and Spearman correlation between
the baseline-relative wrist and chest sweat levels (uncapped z). Spearman is shown because a few extreme values
(tiny baseline spread for some subjects) can dominate Pearson.
Usage: .venv/bin/python -I src/check_sweat.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from env import usable_subjects


def main():
    allowed = usable_subjects()
    print(f"{'subj':5s}{'Pearson':>9s}{'Spearman':>10s}")
    P, S = [], []
    for sid in sorted(allowed["train"] + allowed["validation"], key=lambda s: int(s[1:])):
        f = pd.read_csv(ROOT / "data" / "features" / f"{sid}_features.csv").iloc[20:]
        w, c = f["z_wrist_eda_level"], f["z_chest_eda_level"]
        p, s = w.corr(c), w.corr(c, method="spearman")
        P.append(p); S.append(s)
        print(f"{sid:5s}{p:9.2f}{s:10.2f}")
    print(f"\nmedian Pearson {np.nanmedian(P):.2f} | median Spearman {np.nanmedian(S):.2f} | min Spearman {np.nanmin(S):.2f}")


if __name__ == "__main__":
    main()
