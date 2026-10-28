"""Score every checkpoint of a wesad_ppo.py run against the best-possible score (early-stopping tool).

Usage (project root):  python3.10 src/score_checkpoints.py models/<run_name>
                       python3.10 src/score_checkpoints.py models/<run_name> --plot

Scores each iter_XXXX.pt (and the final models/<run_name>.pt if present) greedily on the validation
and train subjects with oracle.evaluate, and prints mean model/best (total reward divided by the
best-possible total) per checkpoint. Note: once a checkpoint is picked on the validation subjects,
they are no longer a clean test set; the 5 held-out subjects stay the real test.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import oracle
from env import WesadStressEnv, usable_subjects
from eval_policy import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = sorted(run_dir.glob("iter_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    final = run_dir.parent / f"{run_dir.name}.pt"
    items = [(int(re.findall(r"\d+", p.stem)[0]), p) for p in ckpts]
    if final.exists():
        items.append((items[-1][0] + 1 if items else 1, final))
    if not items:
        sys.exit(f"no checkpoints found in {run_dir}")

    allowed = usable_subjects()
    env = WesadStressEnv(subjects=allowed["train"] + allowed["validation"])
    rows = []
    for it, path in items:
        policy = load_policy(str(path))
        out = {}
        for sp in ("validation", "train"):
            r = oracle.evaluate(env, policy, allowed[sp], "ckpt")
            out[sp] = float(np.nanmean([x["fraction_of_best"] for x in r]))
            out[sp + "_match"] = float(np.nanmean([x["match_rate"] for x in r]))
        rows.append((it, path.name, out))
        print(f"{path.name:18s} iter {it:4d}  validation {out['validation']:.2f} (match {out['validation_match']:.2f})"
              f"   train {out['train']:.2f} (match {out['train_match']:.2f})")
    best = max(rows, key=lambda r: r[2]["validation"])
    print(f"\nbest on validation: {best[1]} (iter {best[0]}) model/best = {best[2]['validation']:.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        its = [r[0] for r in rows]
        plt.plot(its, [r[2]["train"] for r in rows], label="train subjects")
        plt.plot(its, [r[2]["validation"] for r in rows], label="validation subjects")
        plt.xlabel("iteration"); plt.ylabel("mean model / best possible"); plt.legend()
        out = run_dir.parent / f"{run_dir.name}_curve.png"
        plt.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
