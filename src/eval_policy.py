"""Score a trained wesad_ppo.py model against the best-possible score (evaluation tooling, not the PPO).

Usage (project root):  python3.10 src/eval_policy.py models/<run>.pt [--sample] [--splits validation train]

By default the policy is greedy (argmax over the masked action scores); --sample draws from the
masked distribution instead. Compares the model with the greedy oracle, a random masked policy and
do-nothing, per subject, using oracle.evaluate. Reads only train/validation subjects.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import gymnasium as gym
import oracle
import wesad_ppo
from env import WesadStressEnv, usable_subjects


def load_policy(model_path, sample=False, features=None):
    envs = gym.vector.SyncVectorEnv([lambda: WesadStressEnv(split="validation", features=features)])   # only to size the network
    agent = wesad_ppo.Agent(envs)
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()

    def policy(obs, mask):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            if sample:
                return int(agent.get_action_and_value(x, m)[0])
            logits = torch.where(m, agent.actor(x), torch.tensor(-1e8))
            return int(logits.argmax(dim=-1))
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["validation", "train"])
    args = ap.parse_args()

    allowed = usable_subjects()
    subjects = [s for sp in args.splits for s in allowed[sp]]
    env = WesadStressEnv(subjects=subjects)
    policy = load_policy(args.model, args.sample)
    rng = np.random.default_rng(0)
    rand = lambda obs, m: int(rng.choice(np.flatnonzero(m)))

    print(f"model: {args.model} ({'sampled' if args.sample else 'greedy'})")
    print(f"{'subject':8s}{'split':>11s}{'best':>8s}{'model':>8s}{'greedy':>8s}{'random':>8s}{'model/best':>11s}{'match':>7s}   action mix")
    fracs = {sp: [] for sp in args.splits}
    for sp in args.splits:
        for sid in allowed[sp]:
            m = oracle.evaluate(env, policy, [sid], "model")[0]
            g = oracle.evaluate(env, None, [sid], "greedy")[0]
            r = oracle.evaluate(env, rand, [sid], "random")[0]
            fracs[sp].append(m["fraction_of_best"])
            mix = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(m["mix"].items(), key=lambda kv: -kv[1]))
            print(f"{sid:8s}{sp:>11s}{m['best']:8.2f}{m['total']:8.2f}{g['total']:8.2f}{r['total']:8.2f}"
                  f"{m['fraction_of_best']:11.2f}{m['match_rate']:7.2f}   {mix}")
    for sp, v in fracs.items():
        print(f"mean model/best on {sp}: {np.nanmean(v):.2f}")


if __name__ == "__main__":
    main()
