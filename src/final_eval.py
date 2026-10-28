"""THE ONE-TIME HELD-OUT RUN (Version 1). Read docs/simulator_spec.md "Freeze list" before running.

What it does, in order:
  1. exports the 5 held-out subjects (splits.json) and builds their features with the SAME code and settings;
  2. trains the final models on all 9 usable subjects with the FROZEN settings, for each seed: the 16-feature model
     and the watch-only model (sees only wrist heart rate and wrist movement; added 2026-10-23 before any held-out run);
  3. scores each model once, greedily, on the held-out subjects and compares with the fair rule (16-feature only),
     the hindsight yardstick and a random masked policy;
  4. writes results/final_eval_<timestamp>.json. It refuses to run a second time unless --second-run is given.

It only runs with:  python3.10 src/final_eval.py --confirm OPEN-HELD-OUT-ONCE
Nothing here is tuned after seeing the held-out score. Description of results: a simulation result, see the README rules.
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
CONFIRM = "OPEN-HELD-OUT-ONCE"

# ---- FROZEN SETTINGS (docs/simulator_spec.md, Freeze list) ----
CONFIGS = {"all16": None, "watch_only": ["wrist_hr", "wrist_move"]}      # the readings each model sees (None = all 16)
FROZEN = {"entropy_coef": 0.1, "total_timesteps": 100000, "features": {"all16": "all 16", "watch_only": "wrist_hr, wrist_move"}, "variant": "base",
          "random_start": False, "training_subjects": "all 9 usable (train + validation)", "policy": "greedy argmax over the masked scores"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--second-run", action="store_true", help="allow a second held-out run (it will be logged as a second run)")
    args = ap.parse_args()
    if args.confirm != CONFIRM:
        sys.exit(f"refusing: this opens the 5 held-out subjects once. Re-run with --confirm {CONFIRM} when you are sure.")
    prior = sorted((ROOT / "results").glob("final_eval_*.json")) if (ROOT / "results").exists() else []
    if prior and not args.second_run:
        sys.exit(f"refusing: a held-out run already exists ({prior[-1].name}). Not rerunning to improve a score.")

    print("1/4 preparing held-out data ...", flush=True)
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-I", str(ROOT / "src" / "_prepare_heldout.py"), "--confirm", CONFIRM],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("held-out preparation failed:\n" + (r.stderr or r.stdout)[-1500:])

    import oracle
    from env import WesadStressEnv, usable_subjects
    from eval_policy import load_policy
    held = json.loads((ROOT / "splits.json").read_text())["held_out"]
    allowed = usable_subjects()
    nine = allowed["train"] + allowed["validation"]
    envs = {name: WesadStressEnv(subjects=held, features_dir=ROOT / "data" / "heldout" / "features", held_out_ok=True, features=feats)
            for name, feats in CONFIGS.items()}
    env = envs["all16"]
    rng = np.random.default_rng(0)
    rand = lambda obs, m: int(rng.choice(np.flatnonzero(m)))

    rows = []
    for seed in args.seeds:
        for name, feats in CONFIGS.items():
            print(f"2/4 training final model '{name}', seed {seed} ...", flush=True)
            cmd = [sys.executable, str(ROOT / "wesad_ppo.py"), "--subjects", ",".join(nine), "--total-timesteps", str(FROZEN["total_timesteps"]),
                   "--seed", str(seed), "--ent-coef", str(FROZEN["entropy_coef"]), "--checkpoint-every", "0"]
            if feats:
                cmd += ["--features", ",".join(feats)]
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            m = re.search(r"saved model to (\S+\.pt)", out.stdout)
            if out.returncode != 0 or not m:
                sys.exit("final training failed:\n" + out.stderr[-1500:])
            policy = load_policy(str(ROOT / m.group(1)), features=feats)
            print(f"3/4 scoring '{name}', seed {seed} on the held-out subjects ...", flush=True)
            for sid in held:
                mo = oracle.evaluate(envs[name], policy, [sid], "model")[0]
                fr = oracle.evaluate(env, oracle.persistence_policy(env), [sid], "fair_rule")[0] if feats is None else None   # the fair rule needs all 16 readings
                hs = oracle.evaluate(env, None, [sid], "hindsight")[0]
                rd = oracle.evaluate(env, rand, [sid], "random")[0]
                rows.append({"config": name, "seed": seed, "subject": sid, "best": mo["best"], "model": mo["total"],
                             "fair_rule": fr["total"] if fr else None, "hindsight": hs["total"], "random": rd["total"],
                             "match_rate": mo["match_rate"], "mix": mo["mix"], "model_file": m.group(1)})

    frac = lambda k, rs: np.array([r[k] / r["best"] for r in rs])
    print("\n4/4 FINAL HELD-OUT RESULT (mean fraction of the best possible score):")
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        fair = f"{frac('fair_rule', rs).mean():.2f}" if name == "all16" else "n/a"
        print(f"  [{name}] model {frac('model', rs).mean():.2f}   fair rule {fair}   hindsight {frac('hindsight', rs).mean():.2f}   random {frac('random', rs).mean():.2f}")
        for seed in args.seeds:
            print(f"      seed {seed}: model {frac('model', [r for r in rs if r['seed'] == seed]).mean():.2f}")
        for sid in held:
            print(f"      {sid}: model {frac('model', [r for r in rs if r['subject'] == sid]).mean():.2f}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / f"final_eval_{int(time.time())}.json"
    path.write_text(json.dumps({"frozen_settings": FROZEN, "seeds": args.seeds, "second_run": bool(prior), "rows": rows}, indent=2))
    print("saved", path, "(a simulation result; do not rerun to improve it)")


if __name__ == "__main__":
    main()
