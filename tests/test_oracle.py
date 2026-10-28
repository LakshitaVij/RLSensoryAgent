"""Tests for src/oracle.py on real train/validation files (no synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import effects as fx
import oracle
from env import WesadStressEnv, usable_subjects

ALL = sum(usable_subjects().values(), [])


@pytest.fixture(scope="module")
def env():
    return WesadStressEnv(subjects=ALL)


def test_dp_plan_replayed_in_env_gives_exactly_the_best_score(env):
    for sid in ALL:
        V = oracle.action_values(env, sid)
        best, plan = oracle.optimal_plan(V)
        obs, info = env.reset(options={"subject": sid})
        total, i = 0.0, 0
        while True:
            a = plan.get(i, 0)
            obs, r, term, trunc, info = env.step(a)      # raises if the plan breaks the lockout or cap
            total += r
            i += 1
            if term or trunc:
                break
        assert total == pytest.approx(best, abs=1e-9)


def test_best_beats_greedy_beats_random_beats_nothing(env):
    rng = np.random.default_rng(1)
    for sid in ALL:
        greedy = oracle.evaluate(env, None, [sid], "greedy")[0]
        rand = oracle.evaluate(env, lambda o, m: int(rng.choice(np.flatnonzero(m))), [sid], "random")[0]
        assert greedy["best"] + 1e-9 >= greedy["total"] >= -1e-12
        assert greedy["best"] + 1e-9 >= rand["total"] >= -1e-12


def test_greedy_does_nothing_when_no_action_helps():
    mask = np.ones(7, dtype=bool)
    V = np.zeros((1, 7))
    assert oracle.greedy_action(V, 0, mask) == 0
    V[0, 3] = 0.5
    assert oracle.greedy_action(V, 0, mask) == 3
    mask[3] = False
    assert oracle.greedy_action(V, 0, mask) == 0
