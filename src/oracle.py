"""Best-possible score and a greedy oracle for the WESAD stress environment (evaluation tooling, not a PPO).

The replay is deterministic and an action's effect depends only on when it starts, so each action
started at decision i has a fixed value V[i, a] (its weighted gain over its 10 windows). With the
10-minute lockout there are no overlaps, so the total reward of an episode is the sum of the values of
the actions started. The best-possible plan is found by dynamic programming under the lockout and
the rolling-hour cap. Uses the environment's own functions, so the numbers match what the env returns.

Usage (from the project root):  .venv/bin/python -I src/oracle.py
"""
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effects as fx
from env import CAP_N, CAP_WINDOW_MIN, START_MIN, STEP_MIN, WesadStressEnv, usable_subjects

LOCK_STEPS = fx.LOCKOUT_MIN // STEP_MIN        # 5 decision steps
CAP_STEPS = CAP_WINDOW_MIN // STEP_MIN         # 30 decision steps


def n_steps(env, sid):
    return (env.data[sid]["raw"].shape[0] - (START_MIN + STEP_MIN)) // STEP_MIN + 1


def action_values(env, sid):
    """V[i, a]: weighted gain over the action's 10 windows if action a starts at decision i (0 for a = 0)."""
    env.reset(options={"subject": sid})
    d = env.data[sid]
    K = n_steps(env, sid)
    last_w = START_MIN + STEP_MIN * K - 1
    V = np.zeros((K, len(fx.ACTIONS)))
    for i in range(K):
        s = START_MIN + STEP_MIN * i
        for a in range(1, len(fx.ACTIONS)):
            env.active = (fx.ACTIONS[a], s)
            total = 0.0
            for m in range(fx.LOCKOUT_MIN):
                w = s + m
                if w > last_w:
                    break
                z_real, z_sim = env._z(d["raw"][w]), env._z(env._sim_row(w))
                for sig, wt in env.weights.items():
                    j = env.idx[fx.REWARD_FEATURE[sig]]
                    if np.isfinite(z_real[j]) and np.isfinite(z_sim[j]):
                        total += wt * (abs(z_real[j]) - abs(z_sim[j])) / STEP_MIN
            V[i, a] = total
    env.active = None
    return V


def optimal_plan(V, cap_n=CAP_N):
    """Best total reward and the actions to start, as {decision index: action}, by dynamic programming."""
    K = V.shape[0]

    def keep(i, recent):
        return tuple(x for x in recent if i - x < CAP_STEPS)

    @lru_cache(maxsize=None)
    def best(i, recent):
        if i >= K:
            return 0.0, None
        value, choice = best(i + 1, keep(i + 1, recent))[0], 0
        if len(recent) < cap_n and (not recent or i - recent[-1] >= LOCK_STEPS):
            for a in range(1, V.shape[1]):
                v = V[i, a] + best(i + 1, keep(i + 1, recent + (i,)))[0]
                if v > value + 1e-15:
                    value, choice = v, a
        return value, choice

    plan, recent = {}, ()
    for i in range(K):
        recent = keep(i, recent)
        _, a = best(i, recent)
        if a:
            plan[i] = a
            recent = recent + (i,)
    return best(0, ())[0], plan


def greedy_action(V, i, mask):
    """Allowed action with the highest value if started now; do nothing when none helps."""
    allowed = np.flatnonzero(mask)
    a = int(allowed[np.argmax(V[i, allowed])])
    return a if V[i, a] > 1e-12 else 0


def persistence_policy(env):
    """A FAIR baseline that sees only what the agent sees: the 19-number observation (plus the effects table
    and the subject's baseline mean/spread, which a deployed system also has). It assumes the current window's
    condition persists for the 10 minutes an action covers, scores each allowed action by its weighted gain
    under that assumption, and picks the best one (do nothing when no action would help). No future data.
    """
    def policy(obs, mask):
        d = env.data[env.sid]
        raw_now = d["mean"] + np.asarray(obs[:16], dtype=float) * d["sd"]   # (z_cap-limited observation)      # current window, rebuilt from the observation
        z_real = env._z(raw_now)
        best_a, best_v = 0, 1e-12
        for a in range(1, len(fx.ACTIONS)):
            if not mask[a]:
                continue
            total = 0.0
            for m in range(fx.LOCKOUT_MIN):
                row = raw_now.copy()
                for sig, eff in env.effects.get(fx.ACTIONS[a], {}).items():
                    value = fx.effect_value(eff, m)
                    if value <= 0:
                        continue
                    for feat in env.signal_features(sig):
                        j = env.idx[feat]
                        row[j] = fx.shift_toward_baseline(row[j], d["mean"][j], d["sd"][j], fx.CALM_DIR[sig], eff.kind, value)
                z_sim = env._z(row)
                for sig, wt in env.weights.items():
                    j = env.idx[fx.REWARD_FEATURE[sig]]
                    if np.isfinite(z_real[j]) and np.isfinite(z_sim[j]):
                        total += wt * (abs(z_real[j]) - abs(z_sim[j])) / STEP_MIN
            if total > best_v:
                best_a, best_v = a, total
        return best_a
    return policy


def evaluate(env, policy, subjects, name="policy"):
    """Run policy(obs, mask) -> action on each subject. Returns one row per subject:
    total reward, best possible, fraction of best, match rate vs the greedy oracle, action mix."""
    rows = []
    for sid in subjects:
        V = action_values(env, sid)
        best, _ = optimal_plan(V, env.cap_n)
        obs, info = env.reset(options={"subject": sid})
        total, i, free, match, mix = 0.0, 0, 0, 0, Counter()
        while True:
            mask = info["action_mask"]
            a = int(policy(obs, mask)) if policy is not None else greedy_action(V, i, mask)
            if mask[1:].any():                      # a decision where an action was possible
                free += 1
                match += int(a == greedy_action(V, i, mask))
            mix[fx.ACTIONS[a]] += 1
            obs, r, term, trunc, info = env.step(a)
            total += r
            i += 1
            if term or trunc:
                break
        rows.append({"subject": sid, "policy": name, "total": total, "best": best,
                     "fraction_of_best": total / best if best > 1e-12 else float("nan"),
                     "match_rate": match / free if free else float("nan"), "mix": dict(mix)})
    return rows


if __name__ == "__main__":
    env = WesadStressEnv(subjects=sum(usable_subjects().values(), []))
    rng = np.random.default_rng(0)
    pols = {"greedy oracle": None,
            "random (masked)": lambda obs, m: int(rng.choice(np.flatnonzero(m))),
            "do nothing": lambda obs, m: 0}
    print(f"{'subject':8s} {'best':>8s} {'greedy':>8s} {'random':>8s} {'nothing':>8s}   greedy/best")
    for sid in env.subjects:
        out = {n: evaluate(env, p, [sid], n)[0] for n, p in pols.items()}
        g = out["greedy oracle"]
        print(f"{sid:8s} {g['best']:8.3f} {g['total']:8.3f} {out['random (masked)']['total']:8.3f} "
              f"{out['do nothing']['total']:8.3f}   {g['fraction_of_best']:.2f}")
