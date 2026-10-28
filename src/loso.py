"""Leave-one-subject-out scoring for wesad_ppo.py (evaluation tooling, not the PPO).

For each of the 9 usable real subjects: train on the other 8 (a short run), then score the saved
model on the left-out subject with oracle.evaluate. Reports model/best per left-out subject, with
the greedy oracle and a random masked policy for comparison, and the spread across subjects/seeds.
Reads only train/validation subjects; the 5 held-out subjects are never touched.

Usage (project root):  python3.10 src/loso.py [--steps 100000] [--seeds 42 43 44]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import oracle
from env import WesadStressEnv, usable_subjects
from variants import VARIANTS
from eval_policy import load_policy


def train_fold(train_ids, steps, seed, extra=""):
    cmd = [sys.executable, str(ROOT / "wesad_ppo.py"), "--subjects", ",".join(train_ids),
           "--total-timesteps", str(steps), "--seed", str(seed), "--checkpoint-every", "0"] + extra.split()
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"saved model to (\S+\.pt)", out.stdout)
    if out.returncode != 0 or not m:
        sys.exit(f"training failed for {train_ids} (seed {seed}):\n{out.stderr[-1500:]}")
    return ROOT / m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS), help="sensitivity-test variant of the practice world")
    ap.add_argument("--features", default="", help="comma-separated readings the agent sees (default all 16)")
    ap.add_argument("--extra", default="", help='extra wesad_ppo.py settings, e.g. "--ent-coef 0.05"')
    args = ap.parse_args()

    allowed = usable_subjects()
    everyone = allowed["train"] + allowed["validation"]
    feats = args.features.split(",") if args.features else None
    env = WesadStressEnv(subjects=everyone, features=feats, **VARIANTS[args.variant])
    rng = np.random.default_rng(0)
    rand = lambda obs, m: int(rng.choice(np.flatnonzero(m)))

    rows = []
    t0 = time.time()
    for seed in args.seeds:
        for held in everyone:
            model = train_fold([s for s in everyone if s != held], args.steps, seed, args.extra + (f" --features {args.features}" if args.features else "") + f" --variant {args.variant}")
            m = oracle.evaluate(env, load_policy(str(model), features=feats), [held], "model")[0]
            g = oracle.evaluate(env, None, [held], "greedy")[0]
            r = oracle.evaluate(env, rand, [held], "random")[0]
            pe = oracle.evaluate(env, oracle.persistence_policy(env), [held], "persistence")[0] if feats is None else dict(total=float("nan"), best=m["best"])   # the fair rule needs the full 16 readings
            rows.append({"seed": seed, "held_out": held, "best": m["best"], "model": m["total"],
                         "greedy": g["total"], "persistence": pe["total"], "random": r["total"],
                         "match": m["match_rate"], "mix": m["mix"]})
            print(f"seed {seed} left out {held:4s} best {m['best']:6.2f} model {m['total']:6.2f} "
                  f"({m['total'] / m['best']:.2f})  fair-rule {pe['total'] / pe['best']:.2f}  hindsight {g['total'] / g['best']:.2f}  random {r['total'] / r['best']:.2f}", flush=True)

    frac = lambda k: np.array([r[k] / r["best"] for r in rows])
    print(f"\nmean over {len(rows)} folds ({time.time() - t0:.0f}s):  model/best {frac('model').mean():.2f} "
          f"(sd {frac('model').std():.2f})  fair-rule {frac('persistence').mean():.2f}  hindsight {frac('greedy').mean():.2f}  random {frac('random').mean():.2f}")
    for seed in args.seeds:
        sel = [r for r in rows if r["seed"] == seed]
        print(f"  seed {seed}: model/best {np.mean([r['model'] / r['best'] for r in sel]):.2f}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / f"loso_{int(time.time())}.json"
    path.write_text(json.dumps({"steps": args.steps, "seeds": args.seeds, "extra": args.extra, "features": args.features, "variant": args.variant, "rows": rows}, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
