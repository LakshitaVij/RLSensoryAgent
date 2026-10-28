# ReadMe for PPO V1

## Overview

Here I have built a reinforcement learning agent that learns which stress regulation exercise to suggest and when, from wearable sensor readings.

 It is trained in a simulator built on the WESAD dataset.

This is just version 1, so it acts as a rehearsal inside a simulator. The practise world replays real WESAD recordings, and each action’s effect comes from published papers , so the agent can only learn what we have put into it. Version 1 tests whether the PPO learning machinery works , like can it find the best actions the simulator implies, measured against a computed best possible score?

Right now this project’s limitation is it cannot show whether the actions help real people or not, and that is what version 2 is for! In version 2 I aim to test out this trained RL agent on me. I will have wearable sensors read my heart, HRV and sweat, and when I drift from my own baseline, the agent plays a breathing exercise through my earbuds, intended to bring those signals back toward baseline.

I am building this system since stress is an unfortunate constant of our modern world, and makes us vulnerable to a whole array of diseases, unhealthy coping patterns, and just an overall decreased quality of life. I also wanted to teach myself hardware and build out my own RL agent project. So I thought to combine these two aspects and see if I could help build an external regulation system that actually works using hardware x RL, which just tells you to breathe when you forget to. 

## How it works

Every 2 minutes the agent looks at 16 readings from the body (heart, sweat, breathing, muscle tension, temperature, movement) and picks one of 7 actions: do nothing, paced breathing, calming audio, progressive muscle relaxation, haptic pulse, guided meditation, or physiological sigh.The simulator then nudges the person's real recorded signals toward their own baseline, using effect sizes taken from papers, and the agent is rewarded for how much closer to baseline the person gets.

We replay real recordings of people's heart, sweat and breathing, and when the agent picks an exercise, we temporarily nudge those readings toward the person's own baseline by the amount published papers report for that exercise, never changing the recording itself. The agent earns reward for how much closer to baseline the nudged readings are than the real ones, and it practices this over thousands of moments and people. So it learns when to pick which exercise, but because the effects come from papers, it can't tell us whether the exercises work on real people, which is what Version 2 on myself will test.

## Pipeline

1. **Data.** WESAD (Schmidt et al. 2018): 15 subjects, lab protocol with baseline, amusement, stress, meditation and recovery. Chest RespiBAN and wrist Empatica E4 signals. S3 is excluded; nine subjects are usable. Five are held out (S2, S8, S10, S11, S17), chosen once with a fixed random seed.
2. **State.** 16 readings per 60-second window (accelerometer in 5-second sub-windows), measured against each person's own baseline (their first 20 windows) in units of that person's baseline standard deviation, capped at plus or minus 5. 
    - **Wrist (Empatica E4), 5 features:** EDA level and trend (EDA is electrodermal activity, i.e. skin conductance, a proxy for sweat), heart rate from the BVP pulse signal, movement from the accelerometer, and skin temperature.
    - **Chest (RespiBAN), 11 features:** heart rate from ECG, HRV as RMSSD (the root mean square of successive differences between heartbeats, which tracks how variable the beat timing is), EDA level and trend, EMG level and trend (EMG is electromyography, i.e. muscle tension), breathing rate, depth and trend, skin temperature, and movement.
    
    **The condition labels (baseline, stress, amusement, meditation) are never part of the state or the reward.** The agent has to infer from the signals alone whether an action is worthwhile.
    
3. **Actions.** 7 actions, one fixed 5-minute duration each. A decision every 2 minutes. A running action locks out new choices for 10 minutes (5-minute effect plus 5-minute fade), at most 4 actions start in any rolling 60 minutes, and only "do nothing" is allowed during lockout or at the cap.
4. **Effects.** Temporary nudges on the real recording, only toward the person's baseline, never past it and never away. Sources and conversions are in `docs/simulator_spec.md`.
5. **Reward.** Per decision step, the average over two windows of the weighted gain in distance from baseline versus the real recording: HRV 35%, heart rate 25%, sweat 20%, breathing 20%. Never negative; "do nothing" is exactly 0.
6. **Agent.** A CleanRL-style PPO: categorical actor, invalid-action masking, `SyncVectorEnv`, GAE, clipping, entropy bonus, checkpointing. Final settings: entropy coefficient 0.1, 100,000 steps, 3 seeds (42, 43, 44), greedy policy at evaluation.
7. **Evaluation.** Leave-one-subject-out (LOSO) over the nine development subjects, then one final run on the five held-out subjects. The held-out subjects were opened exactly once.

## How scores are measured

**Score** = the reward the agent earned ÷ the best reward that was possible on that recording.  The best possible score is the highest total reward you could get by choosing the best set of actions, with full knowledge of the recording, under the lockout and cap. We work this out by using dynamic programming, which essentially chooses the best set of actions for a recording, to maximize total reward under the lockout and the cap. So 1.0 is perfect and 0 is nothing.

We have PPO and three baselines to compare against:

**PPO (Proximal Policy Optimization):**  It learns by trial and error which action to pick from the 16 current readings, earning reward in the simulator, and at test time it picks the highest-scoring allowed action.

- **Pro:** it needs no hand-written rules. It learns from reward alone, never sees the effects table, and still beat random clearly on people it had never seen (0.68 against 0.36 on the held-out five). The invalid-action mask lets it respect the lockout and the cap.
- **Con:** it didn't beat the simple fair rule (0.68 against 0.73), even though the rule is hand-written with the effects table. It also learns from only 9 recordings and can overfit them. Scores moved with settings and seeds (0.64 to 0.72 across seeds), and it was weak on two development people (S15, S16) with all 16 readings. It almost never picked PMR, the haptic pulse or meditation, whose cited effects are small in this world.

**Settings used:** entropy 0.1, 100,000 steps, 3 seeds, trained on all 9 usable people for the final run, picking the top-scoring allowed action at evaluation.

Why these settings:

- **Entropy 0.1:** it sits in the middle of the flat region (0.01 gave 0.52, 0.05 gave 0.56, 0.1 gave 0.57 and 0.2 gave 0.58). The differences between 0.05 and 0.2 are within noise, so we took the middle.
- **100,000 steps:** in one early run, training plateaued around 40,000 steps and early stopping didn't help, so 100,000 is comfortably past the plateau without extra tuning.
- **3 seeds:** the same setup moves about 0.02 between seeds, so three runs show whether a result is stable and not luck.
- **All 9 usable people for the final run:** we chose our settings with leave-one-subject-out, then trained the final model on everyone, because 9 recordings is already few. We did not test this choice separately.
- **Top-scoring allowed action at evaluation:** it gives one repeatable answer per situation instead of a random draw, and the mask keeps it from picking blocked actions.

A caveat: the PPO's score tells us how well it learned the simulator's rules. It says nothing about whether the actions help real people.

- **Random:** picks any allowed action at random.
    - **Pro:** it's the floor, showing what pure guessing earns, so any real learning has something to beat.
    - **Con:** it's a weak comparison. Beating random only shows the agent learned something, not that it learned much, and it doesn't score 0 because the reward is never negative.
    
- **Fair rule:**  hand-written baseline that, at each decision, picks the action whose cited effect would help the most if the person's current readings stayed the same for the next 10 minutes, using only what the agent sees and no future data.
    - **Pro:** it's the honest test of whether learning added anything beyond common sense, because it sees the same inputs as the agent and uses the same cited effects.
    - **Con:** it isn't a pure no-knowledge baseline. It already uses the effects table, which is much of what the simulator contains, so it can look strong.
- **Hindsight yardstick:** picks each action using the real next 10 minutes, which no live agent can know, so it's a ruler to measure against and not a goal.
    - **Pro:** it shows how much headroom exists, because it's the best you could do with perfect knowledge of the next 10 minutes. It shows whether the agent's score is close to the ceiling or far from it.
    - **Con:** no live agent can reach it, so it can make a decent score look weak, and a reader may mistake it for a target. We have to label it a yardstick, as the README text does.

The same setup moves by about 0.02 between seeds, so differences under 0.03 are called "no clear change".

## Reward Schematics

**What the agent is rewarded for:** getting the person's body closer to their own normal. Every 2 minutes, the simulator compares two versions of the same minutes: the real recording, and the copy nudged by the action the agent picked. The reward is how much closer to baseline the nudged copy is, averaged over those two one-minute windows. 

**Four signals count**, each measured as distance from that person's own baseline (their first 20 minutes) in units of their own baseline spread, capped at 5:

| Signal | Sensor | Weight |
| --- | --- | --- |
| HRV (RMSSD) | chest ECG | 35% |
| Heart rate | chest ECG | 25% |
| Sweat (EDA) | wrist | 20% |
| Breathing rate | chest | 20% |

The weights are our own design choice, not from a paper, and they were sensitivity-tested.

**Formula:** reward = Σ weight × ( |distance, real| − |distance, nudged| ), averaged over the two windows.

- **distance, real:** how far the person's real reading is from their own normal, in baseline spreads. For example, the real heart rate is 2 spreads above normal.
- **distance, nudged:** the same distance after the action's effect, say 1.5 spreads above normal.
- **real minus nudged:** how much closer the action got them. 2 − 1.5 = 0.5.
- **× weight:** that gain counts as much as its signal matters. Heart rate has a weight of 0.25, so 0.5 × 0.25 = 0.125.
- **Σ (sum):** do this for all four signals (HRV, heart rate, sweat, breathing) and add them up.
- **Averaged over the two windows:** a decision step covers 2 minutes, so the reward is the average of the two minutes' totals.

### Properties:

- **Do nothing earns exactly 0**, because then the nudged copy equals the real recording.
- **The reward is never negative.** Effects only move readings toward baseline, never past it and never away. That's why random scores about 0.4 and not 0, and why the cap on actions per hour keeps "do nothing" competitive.
- **Condition labels (baseline, stress, amusement, meditation) never enter the reward.** They're used only to build the simulator and check the results.

## Results (simulation results)

| Readings used | PPO | Fair rule | Hindsight | Random |
| --- | --- | --- | --- | --- |
| 16-readings used | 0.68 | 0.73 | 0.78 | 0.36 |

Seeds for the 16-reading model: 0.64 / 0.72 / 0.67. The 16-reading model mostly chose "do nothing", then calming audio, physiological sigh and paced breathing; guided meditation and haptic pulse were chosen once each across all runs, and progressive muscle relaxation never.

A 4-reading model that sees only what my hardware will measure (chest heart rate, chest HRV, and sweat level and trend) scored 0.66 in leave-one-subject-out on the nine development people. It was never scored on held-out people, so treat that number as a development result, not a final one. Aggregate numbers for every run are in `results/summary.md`.

## Where it fails

The held-out set is the 5 people (S2, S8, S10, S11, S17) we set aside and never opened during development, so we could test the final agent once on people it had never seen.

- **The PPO scored below the simple rule:** 0.68 versus 0.73 on held-out. The agent learned something real (it beats random and "do nothing" on all five held-out people), but a hand-written rule using only the current readings does better.
- **Large spread between people:** Five held-out people is a small sample.
- **Two development people were weak with the 16-reading model.** The simple rule did much better on them. Their wrist sweat sits at the cap for most of the recording. The 4-reading hardware-matched model recovers them. The cause is not proven.
- **Three actions are almost never chosen** (PMR never; haptic pulse and guided meditation once each on held-out). Their cited effects are small next to breathing and calming audio in this practice world. This says nothing about whether they help real people.
- **The held-out result cannot be reused as a fair test.** It was opened once; any later change can only be checked on the nine development people.
- **Training plateaus by about 40,000 steps** and the critic can memorize the few training recordings. A plausible but untested reason the 4-reading model scores higher is less overfitting.
- **Several effects are estimates**, borrowed or converted (next section).
- **No result here shows the actions help real people.**

## Citations vs Assumptions

**Cited (details in `docs/simulator_spec.md`):**

- **Effects:** paced breathing on HRV and breathing rate (You et al. 2021); calming audio on HRV, heart rate, breathing and sweat (Kumpulainen et al. 2025, plus a sweat-recovery study listed in the spec); PMR and haptic pulse on sweat (studies listed in the spec).
- **Methods:** beat detection by Pan and Tompkins 1985 through NeuroKit2 (v0.2.13), with artifact correction (Lipponen and Tarvainen 2019); breath detection (Khodadad et al. 2018, published for impedance tomography); HRV (Pham et al. 2021); window sizes (WESAD paper, Sec. 4.1).

**Our own choices, not from papers:**

- Reward weights.
- Cap of 4 actions per hour (idea from Liao et al. 2020, HeartSteps).
- Baseline readings capped at plus or minus 5.
- 20-minute baseline, 2-minute decision step, 10-minute lockout, one 5-minute duration for every action.
- Turning published effect sizes (eta-squared, Hedges' g) into shifts in a person's own baseline spreads.
- Stand-ins and estimates: the sigh borrows paced breathing's numbers; the meditation effect is read off a figure; the PMR conversion is rough; the haptic effect uses response counts.
- The paced-breathing effect we use was measured with a visual pacer; I'm delivering it by audio, so the effect may differ
- Blank effect cells mean "no effect, not tested."
- The breathing-rate drop counts only after the paced-breathing block, not during it, because the in-block drop is just the instructed pacing. Counting it makes an easier, breathing-dominated world, not a better agent.

## How to reproduce

Two environments are used:

- `.venv`: NeuroKit2 0.2.13, numpy 2.5.3, pandas 2.3.3, gymnasium 1.4.0 (data processing and tests).
- `python3.10`: torch 2.13.0, gymnasium 1.3.0, numpy 1.26.4 (training and evaluation).

`requirements-data.txt` and `requirements-train.txt` list the installed versions. The tests use the processed WESAD files, so they cannot run without the data download.

**WESAD data is not included.** Download it from the official source and only load `.pkl` files from that download (pickle safety). Models, run logs, audio and per-person result files are also not included.

Pipeline order: `src/export_subject.py` → `src/build_features.py` → `src/make_split.py` and `src/make_val_split.py` (already frozen) → training through `wesad_ppo.py` → evaluation through `src/loso.py`, `src/eval_policy.py`, `src/oracle.py`.

```
# tests (39 pass with the WESAD data processed locally)
.venv/bin/python -m pytest tests -q

# LOSO run
python3.10 src/loso.py --steps 100000 --seeds 42 43 44 --extra "--ent-coef 0.1"
```

The one-time held-out run is `python3.10 src/final_eval.py --confirm OPEN-HELD-OUT-ONCE`. It has already been used; a guard refuses a repeat unless `--second-run` is passed. It is documented for transparency and is not meant to be rerun to improve a score. `frozen_files.json` holds SHA-256 fingerprints of the files that were frozen before that run, and a test fails if any change.

## Hardware status

Version 2 is planned. The first build is a DIY chest ECG plus a sweat sensor on an ESP32, with audio-only delivery. **Nothing has been built or measured on a person yet.** This is not a medical device and not medical advice. The ECG board must be battery-powered, never on mains-connected USB with electrodes on the body.

## License

All rights reserved.

## Citations

Full list in `docs/simulator_spec.md`. Named in this README:

- Schmidt et al. 2018, WESAD, ICMI.
- Pan and Tompkins 1985.
- Lipponen and Tarvainen 2019.
- Khodadad et al. 2018.
- Pham et al. 2021.
- You et al. 2021.
- Kumpulainen et al. 2025.
- Borchardt and Zoccola 2018.
- Balban et al. 2023.
- Liao et al. 2020 (HeartSteps).