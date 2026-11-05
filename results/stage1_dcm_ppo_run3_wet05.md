# Stage 1, run 3: PPO on top of DCM, memory, trained from LI 0.5 (`--wet_start 0.5 --history 5`)

Same as run 2 ([`stage1_dcm_ppo_run2_memory.md`](stage1_dcm_ppo_run2_memory.md)) except that the soil-wetness
curriculum starts at LI 0.5 instead of 0.7, so training roads include firm mud. Smooth-road LI
sweep copied from the training PC on 2026-11-06. Interpretation is in
[`docs/status_and_plan.md`](../docs/status_and_plan.md).

## Run

| Item | Value |
|---|---|
| Run name | `g1-road-peak_monsoon-dcm-smooth-wet0.5-h5-s0` |
| Job | `kacha job peak_monsoon 3000000 "--robot g1 --base dcm --road smooth --wet_start 0.5 --history 5 --then-evaluate"` (as recommended; step count to be confirmed from the run's log) |
| Seed | 0 (one seed) |
| Evaluation | final `model.zip`, 20 episodes per LI, road seeds 70000+k, no assist, no observation noise |

## Soil-wetness sweep, smooth road

Base season peak_monsoon (mud depth 15 cm). Success with 95% Wilson interval; fall causes are
shares of falls. DCM rows are the same as in the earlier files.

| Method | LI | Success (95% CI) | Fell | Left road | Distance (m) | sank | slipped | Step (m) | Foot lift (m) | Stance sinkage (m) | Cost of transport |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DCM | 0.5 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.14 | 0.099 | 0.000 | 0.88 |
| DCM | 0.6 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.14 | 0.099 | 0.000 | 1.02 |
| DCM | 0.65 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.14 | 0.099 | 0.001 | 1.06 |
| DCM | 0.7 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.13 | 0.097 | 0.002 | 1.12 |
| DCM | 0.75 | 0% (0-16) | 25% | 0% | 6.55 | 0% | 100% | 0.10 | 0.092 | 0.009 | 2.80 |
| DCM | 0.8 | 0% (0-16) | 100% | 0% | -0.03 | 5% | 90% | 0.11 | 0.082 | 0.019 | 7.32 |
| DCM | 0.9 | 0% (0-16) | 100% | 0% | -0.15 | 60% | 40% | 0.17 | 0.062 | 0.008 | - |
| DCM | 1.0 | 0% (0-16) | 100% | 0% | -0.21 | 75% | 25% | 0.16 | 0.024 | 0.048 | - |
| run 3 | 0.5 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.13 | 0.106 | 0.000 | 1.03 |
| run 3 | 0.6 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.13 | 0.106 | 0.000 | 1.14 |
| run 3 | 0.65 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.13 | 0.105 | 0.001 | 1.19 |
| run 3 | 0.7 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.13 | 0.102 | 0.002 | 1.21 |
| run 3 | 0.75 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.13 | 0.098 | 0.004 | 1.23 |
| run 3 | 0.8 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0.11 | 0.090 | 0.014 | 1.42 |
| run 3 | 0.9 | 0% (0-16) | 100% | 0% | -0.14 | 5% | 95% | 0.15 | 0.078 | 0.029 | - |
| run 3 | 1.0 | 0% (0-16) | 100% | 0% | -0.18 | 100% | 0% | 0.00 | 0.018 | 0.039 | - |

LI 0.85 and the firm side below 0.5 were not in this sweep.

## Comparison (smooth road, 20 roads per LI)

| LI | DCM | Run 2 (memory, from 0.7) | Run 3 (memory, from 0.5) |
|---|---|---|---|
| 0.5 | 100% | 0% | 100% |
| 0.6 | 100% | 100% | 100% |
| 0.65 | 100% | 100% | 100% |
| 0.7 | 100% | 70% | 100% |
| 0.75 | 0% | 100% | 100% |
| 0.8 | 0% | 100% | 100% |
| 0.9 | 0% | 0% | 0% |
| 1.0 | 0% | 0% | 0% |

Runs 2 and 3 differ only in the start of the training range (and any differences in step count
to be confirmed), so this comparison is closer to matched, but each has one seed.
Cost of transport is slightly higher than DCM where both succeed (for example 1.21 against 1.12
at LI 0.7).
