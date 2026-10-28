"""Step A: per-window state features for TRAIN subjects.

Usage: .venv/bin/python -I src/build_features.py S4 [S5 ...]

Inputs : data/clean/<SX>/*.csv (written by export_subject.py)
Outputs: data/features/<SX>_baseline.csv     baseline mean and spread (ddof=1) per raw feature
         data/features/<SX>_features.csv     16 features: raw (raw_), baseline-normalised (z_) and capped at +/-Z_CAP (zc_), NO labels
         data/eval_only/<SX>_window_labels.csv   window labels, for evaluation only (rule 9)

Definitions and sources: docs/simulator_spec.md ("Step A feature definitions").
Refuses any subject that is not a train subject or is excluded. No gap-filling:
windows with too few beats give NaN, and beat counts are reported in quality columns.
"""
import json
import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "features"
EVAL = ROOT / "data" / "eval_only"

WIN_S = 60          # body-signal window (WESAD paper Sec. 4.1)
SUB_S = 5           # movement / muscle-level sub-window (WESAD paper Sec. 4.1)
N_SUB = WIN_S // SUB_S
BASELINE_WINDOWS = 20   # fixed first 20 min (CLAUDE.md / spec)
FS_CHEST, FS_BVP, FS_EDA, FS_TEMP, FS_ACC = 700, 64, 4, 4, 32
Z_CAP = 5.0   # LABELED DESIGN CHOICE (Lakshita): cap on |baseline-normalised value|; not from a paper; test 3 and 10 later


def check_subject(sid, allow_heldout=False):
    split = json.loads((ROOT / "splits.json").read_text())
    if allow_heldout:                                   # only src/final_eval.py passes this
        if sid not in split["held_out"]:
            sys.exit(f"{sid} is not a held-out subject.")
        return
    excl = json.loads((ROOT / "excluded.json").read_text())["excluded_from_use"]
    if sid not in split["train"] or sid in excl:
        sys.exit(f"{sid} is not a usable train subject; refusing.")


def windows(a, fs, n_win, per=WIN_S):
    """First n_win full windows of `per` seconds -> shape (n_win, per*fs)."""
    n = int(per * fs)
    return np.asarray(a[: n_win * n], dtype=float).reshape(n_win, n)


def slope(y2d, fs):
    """Least-squares slope per row (units per second) of an array of shape (rows, samples)."""
    t = np.arange(y2d.shape[1]) / fs
    t = t - t.mean()
    return (y2d - y2d.mean(axis=1, keepdims=True)) @ t / np.sum(t ** 2)


def beat_stats(peak_idx, fs, n_win):
    """Heart rate (bpm), RMSSD (ms) and beat count per window from peak sample indices.

    Each NN interval is assigned to the window containing its second beat.
    RMSSD uses successive differences of NN intervals that both fall in the window.
    Beat errors are corrected upstream by NeuroKit2's published method (Lipponen & Tarvainen 2019, `correct_artifacts=True`); no thresholds of our own, no filling.
    """
    pk = np.asarray(peak_idx, dtype=float) / fs
    hr = np.full(n_win, np.nan)
    rmssd = np.full(n_win, np.nan)
    nb = np.zeros(n_win, dtype=int)
    if len(pk) < 3:
        return hr, rmssd, nb
    nn = np.diff(pk)                       # seconds
    w_of_interval = (pk[1:] // WIN_S).astype(int)
    for w in range(n_win):
        in_w = nn[w_of_interval == w]
        nb[w] = int(np.sum((pk // WIN_S).astype(int) == w))
        if len(in_w) >= 1:
            hr[w] = 60.0 / in_w.mean()
        if len(in_w) >= 2:
            rmssd[w] = np.sqrt(np.mean(np.diff(in_w * 1000.0) ** 2))
    return hr, rmssd, nb


def acc_movement(ax, ay, az, fs, n_win):
    mag = np.sqrt(np.asarray(ax, float) ** 2 + np.asarray(ay, float) ** 2 + np.asarray(az, float) ** 2)
    sub = windows(mag, fs, n_win).reshape(n_win, N_SUB, int(SUB_S * fs))
    return sub.std(axis=2).mean(axis=1)    # std of 3D magnitude per 5 s, averaged over the window


def build(sid, allow_heldout=False, clean_root=None, out_root=None, eval_root=None):
    """The three *_root arguments and allow_heldout are used ONLY by src/final_eval.py."""
    check_subject(sid, allow_heldout)
    OUT_ = Path(out_root) if out_root else OUT
    EVAL_ = Path(eval_root) if eval_root else EVAL
    d = (Path(clean_root) if clean_root else CLEAN) / sid
    chest = pd.read_csv(d / "chest_700Hz.csv")
    bvp = pd.read_csv(d / "wrist_bvp_64Hz.csv")
    eda_w = pd.read_csv(d / "wrist_eda_4Hz.csv")
    temp_w = pd.read_csv(d / "wrist_temp_4Hz.csv")
    acc_w = pd.read_csv(d / "wrist_acc_32Hz.csv")

    n_win = int(min(len(chest) / FS_CHEST, len(bvp) / FS_BVP, len(eda_w) / FS_EDA,
                    len(temp_w) / FS_TEMP, len(acc_w) / FS_ACC) // WIN_S)

    f = {}
    # ---- wrist ----
    e = windows(eda_w["eda_uS"].to_numpy(), FS_EDA, n_win)
    f["wrist_eda_level"] = e.mean(axis=1)
    f["wrist_eda_trend"] = slope(e, FS_EDA)
    ppg = nk.ppg_clean(bvp["bvp_raw"].to_numpy(), sampling_rate=FS_BVP, method="elgendi")
    _, ppg_info = nk.ppg_peaks(ppg, sampling_rate=FS_BVP, method="elgendi", correct_artifacts=True)
    f["wrist_hr"], _wrist_rmssd_unused, nb_w = beat_stats(ppg_info["PPG_Peaks"], FS_BVP, n_win)
    # wrist RMSSD is deliberately NOT a feature: at 64 Hz the pulse timing is too coarse (values of hundreds of ms); see docs/simulator_spec.md
    f["wrist_move"] = acc_movement(acc_w["acc_x_g"], acc_w["acc_y_g"], acc_w["acc_z_g"], FS_ACC, n_win)
    f["wrist_temp"] = windows(temp_w["temp_C"].to_numpy(), FS_TEMP, n_win).mean(axis=1)

    # ---- chest ----
    ecg = nk.ecg_clean(chest["ecg_mV"].to_numpy(), sampling_rate=FS_CHEST, method="pantompkins1985")
    _, ecg_info = nk.ecg_peaks(ecg, sampling_rate=FS_CHEST, method="pantompkins1985", correct_artifacts=True)
    f["chest_hr"], f["chest_hrv_rmssd"], nb_c = beat_stats(ecg_info["ECG_R_Peaks"], FS_CHEST, n_win)

    ce = windows(chest["eda_uS"].to_numpy(), FS_CHEST, n_win)
    f["chest_eda_level"] = ce.mean(axis=1)
    f["chest_eda_trend"] = slope(ce, FS_CHEST)

    emg_abs = np.abs(chest["emg_mV"].to_numpy())
    sub = windows(emg_abs, FS_CHEST, n_win).reshape(n_win, N_SUB, int(SUB_S * FS_CHEST)).mean(axis=2)
    f["chest_emg_level"] = sub.mean(axis=1)                     # mean |EMG| over the window (absolute integral / length)
    f["chest_emg_trend"] = slope(sub, 1.0 / SUB_S)              # slope of the activity level within the window

    rsp = nk.rsp_process(chest["resp_pct"].to_numpy(), sampling_rate=FS_CHEST, method="khodadad2018")[0]
    rate = windows(rsp["RSP_Rate"].to_numpy(), FS_CHEST, n_win)
    clean = windows(rsp["RSP_Clean"].to_numpy(), FS_CHEST, n_win)
    f["chest_resp_rate"] = np.nanmean(rate, axis=1)
    f["chest_resp_depth"] = clean.max(axis=1) - clean.min(axis=1)
    f["chest_resp_trend"] = np.array([np.nan if np.isnan(r).all() else
                                      np.polyfit(np.arange(len(r))[~np.isnan(r)] / FS_CHEST, r[~np.isnan(r)], 1)[0]
                                      for r in rate])

    f["chest_temp"] = windows(chest["temp_C"].to_numpy(), FS_CHEST, n_win).mean(axis=1)
    f["chest_move"] = acc_movement(chest["acc_x_g"], chest["acc_y_g"], chest["acc_z_g"], FS_CHEST, n_win)

    raw = pd.DataFrame(f)
    base = raw.iloc[:BASELINE_WINDOWS]
    mu, sd = base.mean(), base.std(ddof=1)
    z = (raw - mu) / sd.replace(0, np.nan)

    feats = pd.concat([
        pd.DataFrame({"window_start_s": np.arange(n_win) * WIN_S}),
        raw.add_prefix("raw_"), z.add_prefix("z_"), z.clip(-Z_CAP, Z_CAP).add_prefix("zc_"),
        pd.DataFrame({"n_beats_chest": nb_c, "n_beats_wrist": nb_w}),
    ], axis=1)
    OUT_.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_ / f"{sid}_features.csv", index=False)
    # baseline mean and spread per raw feature (first 20 windows), so an environment can recompute z after applying an effect
    pd.DataFrame({"feature": raw.columns, "baseline_mean": mu.to_numpy(), "baseline_sd": sd.to_numpy()}) \
        .to_csv(OUT_ / f"{sid}_baseline.csv", index=False)

    # labels: evaluation only, written to a separate folder (rule 9)
    lab = windows(chest["label"].to_numpy(), FS_CHEST, n_win).astype(int)
    mode = np.array([np.bincount(r).argmax() for r in lab])
    frac = np.array([np.bincount(r).max() / r.size for r in lab])
    EVAL_.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"window_start_s": np.arange(n_win) * WIN_S, "label_mode": mode, "label_mode_frac": frac}) \
        .to_csv(EVAL_ / f"{sid}_window_labels.csv", index=False)

    n_clip = int((z.abs() > Z_CAP).sum().sum())
    print(f"{sid}: {n_win} windows -> {OUT_ / (sid + '_features.csv')}  ({n_clip} of {z.size} values capped at |z|={Z_CAP})")
    print("  NaN per raw feature:", {k: int(v) for k, v in raw.isna().sum().items() if v})
    print("  baseline windows with NaN spread (z undefined):", [k for k, v in sd.items() if not np.isfinite(v) or v == 0])


if __name__ == "__main__":
    for s in sys.argv[1:]:
        build(s)
