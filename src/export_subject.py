"""Convert one TRAIN subject's SX.pkl to clean CSVs, one per native sampling rate.

Usage: python -I src/export_subject.py S3 [S4 ...]
Refuses any subject not listed under "train" in splits.json (checked BEFORE the
pickle is opened). Loads only the official WESAD pickles. No resampling,
filtering, or gap-filling.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert as cv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "clean"


def export(sid, allow_heldout=False, out_root=None):
    """allow_heldout/out_root are used ONLY by src/final_eval.py (the one-time held-out run)."""
    train = json.loads((ROOT / "splits.json").read_text())["train"]
    held = json.loads((ROOT / "splits.json").read_text())["held_out"]
    if allow_heldout:
        if sid not in held:
            sys.exit(f"{sid} is not a held-out subject.")
    elif sid not in train or sid in json.loads((ROOT / "excluded.json").read_text())["excluded_from_use"]:
        sys.exit(f"{sid} is not a train subject; refusing to open it.")
    with open(ROOT / sid / f"{sid}.pkl", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest, wrist, label = d["signal"]["chest"], d["signal"]["wrist"], np.asarray(d["label"]).ravel()
    out = (Path(out_root) if out_root else OUT) / sid
    out.mkdir(parents=True, exist_ok=True)

    # Chest values in the .pkl are already in SI units (checked on S3 ranges), so they are written as stored.
    acc = np.asarray(chest["ACC"])
    c = pd.DataFrame({
        "t_s": np.arange(len(label)) / cv.FS_CHEST,
        "ecg_mV": np.asarray(chest["ECG"]).ravel(),
        "eda_uS": np.asarray(chest["EDA"]).ravel(),
        "emg_mV": np.asarray(chest["EMG"]).ravel(),
        "temp_C": np.asarray(chest["Temp"]).ravel(),
        "resp_pct": np.asarray(chest["Resp"]).ravel(),
        "acc_x_g": acc[:, 0], "acc_y_g": acc[:, 1], "acc_z_g": acc[:, 2],
        "label": label,
    })
    c.to_csv(out / "chest_700Hz.csv", index=False)

    def wrist_df(name, fs, cols):
        a = np.asarray(wrist[name], dtype=float)
        a = a.reshape(len(a), -1)
        df = pd.DataFrame(a, columns=cols)
        df.insert(0, "t_s", np.arange(len(df)) / fs)
        return df

    acc_w = wrist_df("ACC", cv.FS_WRIST["ACC"], ["acc_x_g", "acc_y_g", "acc_z_g"])
    acc_w[["acc_x_g", "acc_y_g", "acc_z_g"]] = cv.wrist_acc_g(acc_w[["acc_x_g", "acc_y_g", "acc_z_g"]].to_numpy())
    acc_w.to_csv(out / "wrist_acc_32Hz.csv", index=False)
    wrist_df("BVP", cv.FS_WRIST["BVP"], ["bvp_raw"]).to_csv(out / "wrist_bvp_64Hz.csv", index=False)
    wrist_df("EDA", cv.FS_WRIST["EDA"], ["eda_uS"]).to_csv(out / "wrist_eda_4Hz.csv", index=False)
    wrist_df("TEMP", cv.FS_WRIST["TEMP"], ["temp_C"]).to_csv(out / "wrist_temp_4Hz.csv", index=False)
    print(f"{sid}: wrote {out}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        export(s)
