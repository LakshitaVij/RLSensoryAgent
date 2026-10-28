"""Report-only sanity checks on exported train subjects: durations, NaNs, ranges, label counts.

Usage: python -I src/check_conversions.py S3 [S4 ...]
Reads only data/clean/<train subject>/. Does not analyze signal content.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILES = ["chest_700Hz", "wrist_acc_32Hz", "wrist_bvp_64Hz", "wrist_eda_4Hz", "wrist_temp_4Hz"]


def check(sid):
    if sid not in json.loads((ROOT / "splits.json").read_text())["train"] or sid in json.loads((ROOT / "excluded.json").read_text())["excluded_from_use"]:
        sys.exit(f"{sid} is not a train subject; refusing.")
    print(f"=== {sid} ===")
    for name in FILES:
        df = pd.read_csv(ROOT / "data" / "clean" / sid / f"{name}.csv")
        print(f"{name}: rows={len(df)} duration_s={df['t_s'].iloc[-1]:.1f} NaN={int(df.isna().sum().sum())}")
        for col in df.columns.drop(["t_s", "label"], errors="ignore"):
            s = df[col]
            print(f"    {col:10s} min={s.min():10.4f} max={s.max():10.4f} mean={s.mean():10.4f}")
        if "label" in df:
            print("    label counts:", df["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    for s in sys.argv[1:]:
        check(s)
