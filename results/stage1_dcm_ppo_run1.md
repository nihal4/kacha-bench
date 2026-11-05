# Stage 1, run 1: PPO on top of the DCM controller (no memory)

Raw results of the first full DCM+PPO run, copied from the training PC's `results/` output
(`--then-evaluate`) on 2026-11-05. Numbers are exactly as printed; interpretation is in
[`docs/status_and_plan.md`](../docs/status_and_plan.md).

## Run

| Item | Value |
|---|---|
| Run name | `g1-road-peak_monsoon-dcm-smooth-wet0.7-s0` |
| Job | `kacha job peak_monsoon 50000000 "--robot g1 --base dcm --road smooth --wet_start 0.7 --then-evaluate"`, later cut short by setting the job to 7,000,000 steps (the run resumed at 7,650,000 steps, trained 0 more and went straight to the evaluation) |
| Steps trained | 7,650,000 (of the 50M planned) |
| Seed | 0 (one seed) |
| Speed | about 370 environment steps/s on 4 cores |
| Evaluation | final `model.zip` (not `best_model.zip`), 20 episodes per condition, road seeds 70000+k, no assist, no observation noise |
| Curriculum | soil wetness (LI) started at 0.70, reached 1.00 by about 1.0M steps; each training road drawn between 0.70 and the current level |
| Training depth of mud | 15 cm (peak-monsoon depth) in every training episode; only LI varied |

Training-episode "solved" averaged about 35% over the whole run (range 14% to 80% between
100-episode readings), with no upward trend after about 1.5M steps. Those readings mix roads
of different wetness, so they are noisy.

## Season tables

Columns: Success, Fell, Left road, Distance (m), Time (s), fall causes (counts of falls).
20 episodes per row, 8.3 m road, 60 s limit.

### Smooth road

| Method | Season | Success | Fell | Left road | Distance | Time | Causes |
|---|---|---|---|---|---|---|---|
| scripted controller | dry | 0% | 100% | 0% | 0.61 | 2.2 | no_trigger 14, slipped 6 |
| scripted controller | early_monsoon | 0% | 100% | 0% | 0.59 | 2.2 | slipped 13, no_trigger 7 |
| scripted controller | peak_monsoon | 0% | 100% | 0% | 0.49 | 2.1 | sank 17, slipped 2, stuck 1 |
| DCM capture-point controller | dry | 65% | 35% | 0% | 6.01 | 33.1 | slipped 6, no_trigger 1 |
| DCM capture-point controller | early_monsoon | 100% | 0% | 0% | 8.30 | 45.4 | - |
| DCM capture-point controller | peak_monsoon | 0% | 100% | 0% | -0.20 | 1.8 | sank 17, slipped 3 |
| trained policy (run 1) | dry | 0% | 100% | 0% | 0.43 | 4.7 | slipped 18, no_trigger 2 |
| trained policy (run 1) | early_monsoon | 5% | 95% | 0% | 0.27 | 5.7 | slipped 16, no_trigger 2, stuck 1 |
| trained policy (run 1) | peak_monsoon | 0% | 100% | 0% | -0.14 | 1.8 | sank 20 |

### Rutted road

| Method | Season | Success | Fell | Left road | Distance | Time | Causes |
|---|---|---|---|---|---|---|---|
| scripted controller | dry | 0% | 100% | 0% | 0.26 | 1.6 | no_trigger 14, slipped 6 |
| scripted controller | early_monsoon | 0% | 100% | 0% | 0.31 | 1.8 | slipped 11, no_trigger 9 |
| scripted controller | peak_monsoon | 0% | 100% | 0% | 0.36 | 1.9 | sank 16, slipped 2, stuck 1, no_trigger 1 |
| DCM capture-point controller | dry | 0% | 100% | 0% | 0.28 | 3.8 | slipped 20 |
| DCM capture-point controller | early_monsoon | 0% | 100% | 0% | -0.03 | 3.1 | slipped 19, stuck 1 |
| DCM capture-point controller | peak_monsoon | 0% | 100% | 0% | -0.21 | 1.8 | sank 14, slipped 6 |
| trained policy (run 1) | dry | 0% | 100% | 0% | 0.06 | 2.5 | slipped 19, no_trigger 1 |
| trained policy (run 1) | early_monsoon | 0% | 100% | 0% | 0.09 | 2.9 | slipped 20 |
| trained policy (run 1) | peak_monsoon | 0% | 100% | 0% | -0.16 | 1.7 | sank 14, slipped 5, stuck 1 |

## Soil-wetness (LI) sweeps

Base season peak_monsoon (mud depth 15 cm), 20 episodes per value, road seeds 70000+k, no
assist, no observation noise. Success with 95% Wilson interval. Fall causes are shares of
falls. Full per-row output (gait metrics, cost of transport) is in the training PC's
`results/sweep_LI_*.csv`.

### Smooth road

| Method | LI | Success (95% CI) | Fell | Distance (m) | sank | slipped | stuck | no_trigger |
|---|---|---|---|---|---|---|---|---|
| DCM | 0.5 | 100% (84-100) | 0% | 8.30 | 0% | 0% | 0% | 0% |
| DCM | 0.6 | 100% (84-100) | 0% | 8.30 | 0% | 0% | 0% | 0% |
| DCM | 0.65 | 100% (84-100) | 0% | 8.30 | 0% | 0% | 0% | 0% |
| DCM | 0.7 | 100% (84-100) | 0% | 8.30 | 0% | 0% | 0% | 0% |
| DCM | 0.75 | 0% (0-16) | 25% | 6.55 | 0% | 100% | 0% | 0% |
| DCM | 0.8 | 0% (0-16) | 100% | -0.03 | 5% | 90% | 5% | 0% |
| DCM | 0.9 | 0% (0-16) | 100% | -0.15 | 60% | 40% | 0% | 0% |
| DCM | 1.0 | 0% (0-16) | 100% | -0.21 | 75% | 25% | 0% | 0% |
| trained (run 1) | 0.5 | 0% (0-16) | 100% | 0.06 | 0% | 85% | 0% | 15% |
| trained (run 1) | 0.6 | 0% (0-16) | 100% | 0.08 | 0% | 95% | 5% | 0% |
| trained (run 1) | 0.65 | 0% (0-16) | 100% | 0.04 | 0% | 100% | 0% | 0% |
| trained (run 1) | 0.7 | 0% (0-16) | 100% | -0.20 | 5% | 90% | 5% | 0% |
| trained (run 1) | 0.75 | 40% (22-61) | 60% | 3.24 | 8% | 92% | 0% | 0% |
| trained (run 1) | 0.8 | 100% (84-100) | 0% | 8.30 | 0% | 0% | 0% | 0% |
| trained (run 1) | 0.9 | 0% (0-16) | 95% | 0.28 | 5% | 95% | 0% | 0% |
| trained (run 1) | 1.0 | 0% (0-16) | 100% | -0.13 | 100% | 0% | 0% | 0% |

LI 0.85 was not in this sweep.

### Rutted road

Every row is 0% (0-16) with 100% falls, for both DCM and the trained policy, at every LI
from 0.5 to 1.0. Mean distance is between about -0.29 m and +0.10 m (the robot falls at the
start). Fall causes: slipped (90-100%) for LI 0.5 to 0.8, a mix of sank and slipped at 0.9,
mostly sank at 1.0 (DCM 95%, trained 70% plus 20% stuck).

## Earlier short check (800k steps, one seed, 10 test roads per value)

Same method before the "easier mud in the mix" fix; from `results/wet_check_LI_summary.md`.

| LI | DCM alone | PPO on DCM (curriculum, 800k steps) |
|---|---|---|
| 0.70 | 100% | 20% |
| 0.75 | 0% | 90% |
| 0.80 | 0% | 100% |
| 0.85 | 0% (all slip) | 0% (no falls; too slow, 7.5 of 8.3 m in 60 s) |
