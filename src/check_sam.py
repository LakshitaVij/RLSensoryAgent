"""Check 1: does our reward-distance agree with how people felt? (evaluation only; uses labels + questionnaires)

For each usable subject (train + validation; never the held-out 5) and each of the 5 rated conditions
(baseline, amusement, stress/TSST, meditation 1, meditation 2): average over that condition's labeled windows the
weighted distance from baseline used by the reward, D = sum_w weight * |z_signal| (capped z, the 4 reward signals), and
compare it with that condition's SAM arousal rating (1 = low, 9 = high). SX_quest.csv lists the conditions in order and
one 'DIM' line (valence;arousal) per condition in the same order. Meditation 1/2 are the first/second run of label 4.
Usage: .venv/bin/python -I src/check_sam.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import effects as fx
from env import usable_subjects

LABEL = {"Base": 1, "TSST": 2, "Fun": 3, "Medi 1": 4, "Medi 2": 4}
KIND = {"Base": "baseline", "TSST": "stress", "Fun": "amusement", "Medi 1": "meditation", "Medi 2": "meditation"}


def read_quest(sid):
    order, dim = [], []
    with open(ROOT / sid / f"{sid}_quest.csv", newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if not row:
                continue
            tag = row[0].strip()
            if tag == "# ORDER":
                order = [c.strip() for c in row[1:] if c.strip() and c.strip() not in ("bRead", "fRead", "sRead")]
            elif tag == "# DIM":
                dim.append((float(row[1]), float(row[2])))
    return order, dim


def runs_of(idx, gap=5):
    out, cur = [], [idx[0]] if len(idx) else []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > gap:
            out.append(cur); cur = []
        cur.append(b)
    if cur:
        out.append(cur)
    return out


def main():
    allowed = usable_subjects()
    rows = []
    for sid in sorted(allowed["train"] + allowed["validation"], key=lambda s: int(s[1:])):
        order, dim = read_quest(sid)
        f = pd.read_csv(ROOT / "data" / "features" / f"{sid}_features.csv")
        lab = pd.read_csv(ROOT / "data" / "eval_only" / f"{sid}_window_labels.csv")["label_mode"].to_numpy()
        dist = sum(w * f[f"zc_{fx.REWARD_FEATURE[s]}"].abs() for s, w in fx.WEIGHTS.items()).to_numpy()
        # stressed-side only: HRV below baseline, heart rate / breathing / sweat above baseline (calm-side values count 0)
        dist_s = sum(w * (-fx.CALM_DIR[s] * f[f"zc_{fx.REWARD_FEATURE[s]}"]).clip(lower=0) for s, w in fx.WEIGHTS.items()).to_numpy()
        med_runs = runs_of(np.flatnonzero(lab == 4))
        if len(order) != len(dim) or len(med_runs) != 2:
            print(f"{sid}: skipped (conditions {len(order)}, SAM lines {len(dim)}, meditation runs {len(med_runs)})")
            continue
        for name, (val, aro) in zip(order, dim):
            idx = np.array(med_runs[0] if name == "Medi 1" else med_runs[1] if name == "Medi 2" else np.flatnonzero(lab == LABEL[name]))
            rows.append({"subject": sid, "condition": KIND[name], "name": name, "arousal": aro, "valence": val, "D": float(dist[idx].mean()), "Dstress": float(dist_s[idx].mean()), "windows": len(idx)})
    df = pd.DataFrame(rows)
    print("average by condition type  (D = weighted distance from baseline either side; Dstress = stressed side only; SAM arousal 1-9):")
    print(df.groupby("condition")[["D", "Dstress", "arousal"]].mean().round(2).to_string())
    rho = df["D"].corr(df["arousal"], method="spearman")
    per = df.groupby("subject").apply(lambda g: g["D"].corr(g["arousal"], method="spearman"), include_groups=False)
    print(f"\nSpearman(D, arousal) over all {len(df)} subject-conditions: {rho:.2f}   |   Spearman(Dstress, arousal): {df['Dstress'].corr(df['arousal'], method='spearman'):.2f}")
    pers = df.groupby("subject").apply(lambda g: g["Dstress"].corr(g["arousal"], method="spearman"), include_groups=False)
    print(f"per-subject Spearman with Dstress: median {pers.median():.2f}, positive in {(pers > 0).sum()} of {pers.notna().sum()} subjects")
    print(f"per-subject Spearman (5 points each): median {per.median():.2f}, positive in {(per > 0).sum()} of {per.notna().sum()} subjects")
    stress_gt_base = np.mean([g.loc[g.condition == 'stress', 'D'].iloc[0] > g.loc[g.condition == 'baseline', 'D'].iloc[0] for _, g in df.groupby('subject')])
    print(f"stress D higher than baseline D in {stress_gt_base:.0%} of subjects")
    df.to_csv(ROOT / "results" / "check_sam_rows.csv", index=False)


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
