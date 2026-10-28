# Version 1 results summary (aggregate numbers only)

All numbers are **simulation results** from the practice world described in `docs/simulator_spec.md`. They do not show that the actions help real people.
Score = fraction of the best possible total reward on a recording (best possible found by dynamic programming under the 10-minute lockout and the cap of 4 actions per hour). Random scores about 0.4, not 0, because the reward is never negative. The same setup moves by about 0.02 between seeds, so differences under 0.03 are called "no clear change".
Per-person result files exist locally but are not published, because the WESAD dataset terms were not checked.

## Final held-out run (once; five held-out people; 3 seeds)
Command (it refuses a second run): `python3.10 src/final_eval.py --confirm OPEN-HELD-OUT-ONCE`

| Model | Score | Fair rule | Hindsight yardstick | Random |
|---|---|---|---|---|
| 16-reading PPO | 0.68 | 0.73 | 0.78 | 0.36 |
| Watch-only PPO (wrist heart rate + wrist movement; not pursued) | 0.49 | n/a | 0.78 | 0.37 |

Seeds: 16-reading 0.64 / 0.72 / 0.67; watch-only 0.48 / 0.48 / 0.53. Both models earned positive total reward on all five held-out people (do nothing = 0).

## Development: leave-one-subject-out (nine people, 100,000 steps, 3 seeds unless noted)
Command pattern: `python3.10 src/loso.py --steps 100000 --seeds 42 43 44 --extra "--ent-coef 0.1"` (add `--features a,b,c` or `--variant name`)

| Setup | Score |
|---|---|
| Random / fair rule / hindsight yardstick | 0.43 / 0.66 / 0.77 |
| 16-reading PPO, entropy 0.01 / 0.05 / **0.1 (frozen)** / 0.2 | 0.52 / 0.56 / **0.57** / 0.58 |
| 16-reading PPO, random-start episodes (entropy 0.05) | 0.55 |
| Wrist heart rate + wrist movement only (entropy 0.05) | 0.54 |
| Hardware-matched model: chest heart rate, chest HRV, sweat level and trend (development only, no held-out score) | 0.66 |

## Sensor-group ablations (entropy 0.1)
| Dropped group | Score | Verdict |
|---|---|---|
| Heart (heart rate, HRV), 3 seeds | 0.50 | agent near random |
| Sweat, 3 seeds | 0.57 | no clear change |
| Breathing, 1 seed | 0.60 | no clear change |
| Muscle, 1 seed | 0.59 | no clear change |
| Temperature, 1 seed | 0.58 | no clear change |
| Movement, 1 seed | 0.56 | no clear change |
Comparison base: 0.57 (3 seeds), 0.58 (1 seed).

## Sensitivity of the practice world (1 seed, base 0.58; entropy 0.1)
Equal weights 0.61; heart-heavier weights 0.56; z cap 3: 0.58; z cap 10: 0.61; PMR effect halved 0.57; meditation effect halved 0.60; no twin mirroring 0.56; audio sweat as flat 16.5%: 0.60. All within 0.03 of base.
Counting the breathing-rate drop during the paced-breathing block (a breathing-dominated, easier world): 0.65 over 3 seeds (fair rule 0.63 there).
Cap per hour (scores are not comparable across caps, because the best possible changes): cap 2: 0.58 with fair rule 0.57 (1 seed); cap 6: 0.67 with fair rule 0.75 (3 seeds).

## Where these numbers come from
Each row was produced by the command pattern shown, on the nine development people (leave-one-subject-out) or the five held-out people (final run). The code, settings and splits are frozen and fingerprinted in `frozen_files.json`. Reproducing the numbers needs the WESAD data from the official download, which is not included here.
