"""Tests for src/effects.py and src/env.py.

Real train/validation feature files plus hand-written single-number arithmetic checks.
No synthetic recordings; held-out and eval-only files are never read.
Run: .venv/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import effects as fx
from env import WesadStressEnv, usable_subjects


# ---------- hand-written arithmetic checks (single numbers) ----------
def test_raw_shift_lowers_stressed_heart_rate():
    assert fx.shift_toward_baseline(80, 70, 5, -1, "raw", 3) == 77

def test_shift_never_passes_baseline():
    assert fx.shift_toward_baseline(72, 70, 5, -1, "raw", 3) == 70

def test_shift_never_pushes_away_from_baseline():
    assert fx.shift_toward_baseline(65, 70, 5, -1, "raw", 3) == 65      # already on the calm side

def test_hrv_is_pushed_up_toward_baseline():
    assert fx.shift_toward_baseline(40, 50, 10, +1, "raw", 5) == 45
    assert fx.shift_toward_baseline(48, 50, 10, +1, "raw", 5) == 50
    assert fx.shift_toward_baseline(55, 50, 10, +1, "raw", 5) == 55

def test_spread_and_share_kinds():
    assert fx.shift_toward_baseline(80, 70, 5, -1, "spread", 0.54) == pytest.approx(77.3)
    assert fx.shift_toward_baseline(80, 70, 5, -1, "share", 0.23) == pytest.approx(77.7)

def test_nan_stays_nan():
    assert np.isnan(fx.shift_toward_baseline(np.nan, 70, 5, -1, "raw", 3))

def test_effect_timeline_pmr():
    e = fx.EFFECTS["pmr"]["eda"]
    assert [fx.effect_value(e, m) for m in range(5)] == [1.02] * 5       # full during the block
    assert fx.effect_value(e, 5) == pytest.approx(1.02 * 0.9)            # linear fade over the after window
    assert fx.effect_value(e, 9) == pytest.approx(1.02 * 0.1)
    assert fx.effect_value(e, 10) == 0 and fx.effect_value(e, -1) == 0

def test_effect_timeline_uses_cited_after_value():
    e = fx.EFFECTS["paced_breathing"]["hrv"]
    assert fx.effect_value(e, 0) == 25.86
    assert [fx.effect_value(e, m) for m in range(5, 10)] == [3.82] * 5

def test_paced_breathing_does_not_count_the_instructed_breathing_rate_drop():
    e = fx.EFFECTS["paced_breathing"]["resp"]
    assert [fx.effect_value(e, m) for m in range(5)] == [0.0] * 5        # during: pacing itself, not counted
    assert [fx.effect_value(e, m) for m in range(5, 10)] == [2.60] * 5   # after: the lasting drop counts

def test_audio_sweat_share_at_end_of_block():
    share = fx.EFFECTS["calming_audio"]["eda"].during
    assert share(300.0) == pytest.approx(0.2871, abs=0.001)             # about 29% by minute 5
    assert share(0.0) == 0.0

def test_sigh_borrows_paced_breathing():
    assert fx.EFFECTS["physiological_sigh"] is fx.EFFECTS["paced_breathing"]

def test_weights_sum_to_one():
    assert sum(fx.WEIGHTS.values()) == pytest.approx(1.0)


# ---------- environment checks on real train / validation files ----------
SUBJECTS = usable_subjects()
ALL = SUBJECTS["train"] + SUBJECTS["validation"]

@pytest.fixture(scope="module")
def env():
    return WesadStressEnv(subjects=ALL)

def run_episode(env, sid, policy, seed=0):
    obs, info = env.reset(seed=seed, options={"subject": sid})
    steps, total, infos = 0, 0.0, []
    while True:
        a = policy(info["action_mask"], steps)
        obs, r, term, trunc, info = env.step(a)
        steps += 1; total += r; infos.append((r, info))
        assert obs.shape == (19,) and np.all(np.isfinite(obs))
        assert env.observation_space.contains(obs)
        if term or trunc:
            return steps, total, infos

def test_do_nothing_scores_exactly_zero_and_ends_by_truncation(env):
    for sid in ALL:
        n_win = env.data[sid]["raw"].shape[0]
        steps, total, infos = run_episode(env, sid, lambda m, t: 0)
        assert total == 0.0 and all(r == 0.0 for r, _ in infos)
        assert steps == (n_win - 22) // 2 + 1                            # steps until fewer than 2 windows remain
        assert infos[-1][1]["minute"] + 2 > n_win

def test_observation_without_action_matches_precomputed_capped_features(env):
    sid = ALL[0]
    d = env.data[sid]
    obs, _ = env.reset(options={"subject": sid})
    z = np.clip((d["raw"][19] - d["mean"]) / d["sd"], -5, 5)
    assert np.allclose(obs[:16], z, atol=1e-5) and np.all(obs[16:] == 0)

def test_gains_never_negative_and_other_features_untouched(env):
    rng = np.random.default_rng(0)
    twin = {env.idx[f] for fs in fx.SIGNAL_FEATURES.values() for f in fs}
    total_reward = 0.0
    for sid in ALL:
        def policy(mask, t):
            allowed = np.flatnonzero(mask)
            return int(rng.choice(allowed))
        _, total, infos = run_episode(env, sid, policy)
        total_reward += total
        for r, info in infos:
            assert r >= -1e-12
            assert all(g >= -1e-12 for g in info["gain"].values())
    assert total_reward > 0.0                                            # the effects do something

def test_non_twin_features_equal_real_values_during_an_action(env):
    sid = ALL[1]
    d = env.data[sid]
    twin = {env.idx[f] for fs in fx.SIGNAL_FEATURES.values() for f in fs}
    obs, info = env.reset(options={"subject": sid})
    obs, *_ = env.step(1)                                                # paced breathing at minute 20
    z_real = np.clip((d["raw"][21] - d["mean"]) / d["sd"], -5, 5)
    for j in range(16):
        if j not in twin:
            assert obs[j] == pytest.approx(z_real[j], abs=1e-5)

def test_lockout_blocks_for_ten_minutes_then_frees(env):
    sid = ALL[0]
    obs, info = env.reset(options={"subject": sid})
    assert info["action_mask"].all()
    obs, r, term, trunc, info = env.step(2)                              # calming audio at minute 20
    for expected_left in (8, 6, 4, 2):                                   # minutes 22, 24, 26, 28
        assert not info["action_mask"][1:].any() and info["action_mask"][0]
        assert info["lockout_left_min"] == expected_left
        obs, r, term, trunc, info = env.step(0)
    assert info["minute"] == 30 and info["lockout_left_min"] == 0
    assert info["action_mask"].all()                                     # free again at minute 30

def test_forbidden_action_raises(env):
    env.reset(options={"subject": ALL[0]})
    env.step(1)
    with pytest.raises(ValueError):
        env.step(2)

def test_rolling_hour_cap(env):
    for sid in ALL:
        _, _, infos = run_episode(env, sid, lambda m, t: 1 if m[1] else 0)
        starts = [i["minute"] - 2 for r, i in infos if i["started"]]
        for s in starts:
            assert sum(1 for x in starts if s - 60 < x <= s) <= 4        # never more than 4 in any 60 minutes
    # the cap really binds: with 10-minute actions the agent would otherwise start 6 per hour
    _, _, infos = run_episode(env, ALL[0], lambda m, t: 1 if m[1] else 0)
    assert any(i["actions_last_hour"] == 4 and not i["action_mask"][1:].any() and i["lockout_left_min"] == 0
               for r, i in infos)

def test_held_out_excluded_and_unknown_subjects_are_refused():
    for bad in ("S2", "S8", "S10", "S11", "S17", "S3"):
        with pytest.raises(ValueError):
            WesadStressEnv(subjects=[bad])

def test_same_seed_same_trajectory(env):
    def go():
        rng = np.random.default_rng(7)
        out = []
        obs, info = env.reset(seed=3)
        done = False
        while not done:
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            obs, r, term, trunc, info = env.step(a)
            out.append((a, round(r, 12))); done = term or trunc
        return info["subject"], out
    assert go() == go()

def test_environment_code_never_references_label_files():
    for name in ("env.py", "effects.py"):
        text = (SRC / name).read_text()
        assert "eval_only" not in text and "label_mode" not in text


def test_random_start_varies_the_start_and_still_ends_cleanly():
    env = WesadStressEnv(subjects=ALL, random_start=True)
    starts = set()
    for seed in range(30):
        obs, info = env.reset(seed=seed)
        sid = info["subject"]
        n_win = env.data[sid]["raw"].shape[0]
        assert 20 <= info["minute"] <= n_win - 2 and info["minute"] % 2 == 0   # a step needs two windows
        starts.add(info["minute"])
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(0)
            total += r; done = term or trunc
        assert total == 0.0 and info["minute"] + 2 > n_win
    assert len(starts) > 5                                               # starts really vary

def test_default_env_still_starts_at_minute_20():
    env = WesadStressEnv(subjects=ALL)
    for seed in range(5):
        assert env.reset(seed=seed)[1]["minute"] == 20


def test_feature_subset_shrinks_only_the_observation():
    feats = ["wrist_hr", "wrist_move"]
    full, small = WesadStressEnv(subjects=ALL), WesadStressEnv(subjects=ALL, features=feats)
    assert small.observation_space.shape == (5,) and full.observation_space.shape == (19,)
    sid = ALL[0]
    o_full, _ = full.reset(options={"subject": sid}); o_small, _ = small.reset(options={"subject": sid})
    cols = [full.idx[f] for f in feats]
    assert np.allclose(o_small[:2], o_full[cols], atol=1e-6) and np.allclose(o_small[2:], o_full[16:])
    # rewards are identical: the reward uses the full data whatever the agent sees
    for a in (1, 0, 0, 0, 0, 0):
        _, r1, *_ = full.step(a); _, r2, *_ = small.step(a)
        assert r1 == pytest.approx(r2)

def test_unknown_feature_is_refused():
    with pytest.raises(ValueError):
        WesadStressEnv(subjects=ALL, features=["not_a_feature"])


def test_variants_build_and_base_is_unchanged():
    import oracle
    from variants import VARIANTS
    base = WesadStressEnv(subjects=ALL, **VARIANTS["base"])
    plain = WesadStressEnv(subjects=ALL)
    sid = ALL[0]
    assert oracle.optimal_plan(oracle.action_values(base, sid))[0] == pytest.approx(oracle.optimal_plan(oracle.action_values(plain, sid))[0])
    for name, kw in VARIANTS.items():
        env = WesadStressEnv(subjects=ALL, **kw)
        obs, info = env.reset(options={"subject": sid})
        assert env.observation_space.contains(obs)

def test_variant_switches_do_what_they_say():
    from variants import VARIANTS
    assert WesadStressEnv(subjects=ALL, **VARIANTS["zcap3"]).observation_space.high[0] == 3.0
    nt = WesadStressEnv(subjects=ALL, **VARIANTS["no_twins"])
    assert nt.signal_features("hr") == ["chest_hr"] and nt.signal_features("eda") == ["wrist_eda_level"]
    assert WesadStressEnv(subjects=ALL, **VARIANTS["breathing_during_counted"]).effects["paced_breathing"]["resp"].during == 10.20
    assert fx.EFFECTS["paced_breathing"]["resp"].during == 0.0          # the base table was not modified
    assert sum(WesadStressEnv(subjects=ALL, **VARIANTS["equal_weights"]).weights.values()) == pytest.approx(1.0)


def test_cap_switch_limits_actions_started_per_hour():
    """With cap_n=2, a third action inside 60 minutes is masked; cap_n=4 (default) still allows it."""
    sid = usable_subjects()["train"][0]
    for cap, expect_masked in [(2, True), (4, False)]:
        env = WesadStressEnv(subjects=[sid], cap_n=cap)
        env.reset(options={"subject": sid})
        for _ in range(2):                                 # start 2 actions, waiting out each 10-minute lockout
            env.step(1)
            for _ in range(4):
                env.step(0)
        assert (not env._mask()[1]) == expect_masked
