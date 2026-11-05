# Stage 1, run 2: PPO on top of DCM with memory (`--history 5`)

Same setup as run 1 ([`stage1_dcm_ppo_run1.md`](stage1_dcm_ppo_run1.md)) except that the policy's
input holds the last 5 observations. Smooth-road soil-wetness sweep copied from the training
PC's `results/sweep_LI_smooth_summary.md` on 2026-11-05. Interpretation is in
[`docs/status_and_plan.md`](../docs/status_and_plan.md).

## Run

| Item | Value |
|---|---|
| Run name | `g1-road-peak_monsoon-dcm-smooth-wet0.7-h5-s0` |
| Job | `kacha job peak_monsoon 3000000 "--robot g1 --base dcm --road smooth --wet_start 0.7 --history 5 --then-evaluate"` |
| Steps trained | 3,000,004 (a power cut at about 2.5M steps stopped training once; it resumed from the last checkpoint) |
| Seed | 0 (one seed) |
| Speed | about 350 to 370 environment steps/s on 4 cores |
| Evaluation | final `model.zip`, 20 episodes per LI, road seeds 70000+k, no assist, no observation noise |
| Curriculum | wetness 0.70 at the start, 0.95 at 1.2M steps, 1.00 by 2.0M steps; each training road drawn between 0.70 and the current level; training mud depth fixed at 15 cm |

Training-episode "solved" (noisy: about 100 episodes from a few roads of different wetness):
14% at 0.5M, 23% at 1.0M, 19% at 1.5M, 56% at 2.5M, 72% at 2.95M, 45% at 3.0M (the last two are
50k steps apart, which shows the reading noise). Run 1 averaged about 35% and did not trend
upward.

## Soil-wetness sweep, smooth road

Base season peak_monsoon (mud depth 15 cm). Success with 95% Wilson interval; fall causes
are shares of falls. DCM rows are the same as in run 1's file.

| Method | LI | Success (95% CI) | Fell | Left road | Distance (m) | sank | slipped | stuck | Step (m) | Foot lift (m) | Stance sinkage (m) | Cost of transport |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DCM | 0.5 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.14 | 0.099 | 0.000 | 0.88 |
| DCM | 0.6 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.14 | 0.099 | 0.000 | 1.02 |
| DCM | 0.65 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.14 | 0.099 | 0.001 | 1.06 |
| DCM | 0.7 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.13 | 0.097 | 0.002 | 1.12 |
| DCM | 0.75 | 0% (0-16) | 25% | 0% | 6.55 | 0% | 100% | 0% | 0.10 | 0.092 | 0.009 | 2.80 |
| DCM | 0.8 | 0% (0-16) | 100% | 0% | -0.03 | 5% | 90% | 5% | 0.11 | 0.082 | 0.019 | 7.32 |
| DCM | 0.9 | 0% (0-16) | 100% | 0% | -0.15 | 60% | 40% | 0% | 0.17 | 0.062 | 0.008 | - |
| DCM | 1.0 | 0% (0-16) | 100% | 0% | -0.21 | 75% | 25% | 0% | 0.16 | 0.024 | 0.048 | - |
| run 2 (memory) | 0.5 | 0% (0-16) | 90% | 20% | 1.76 | 0% | 100% | 0% | 0.12 | 0.082 | 0.000 | 3.33 |
| run 2 (memory) | 0.6 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.12 | 0.095 | 0.000 | 1.15 |
| run 2 (memory) | 0.65 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.12 | 0.096 | 0.000 | 1.23 |
| run 2 (memory) | 0.7 | 70% (48-85) | 20% | 10% | 7.27 | 0% | 100% | 0% | 0.12 | 0.094 | 0.001 | 1.46 |
| run 2 (memory) | 0.75 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.12 | 0.093 | 0.004 | 1.35 |
| run 2 (memory) | 0.8 | 100% (84-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0.12 | 0.087 | 0.010 | 1.41 |
| run 2 (memory) | 0.9 | 0% (0-16) | 100% | 0% | -0.13 | 50% | 45% | 5% | 0.36 | 0.107 | 0.016 | - |
| run 2 (memory) | 1.0 | 0% (0-16) | 100% | 0% | -0.20 | 95% | 5% | 0% | - | 0.016 | 0.045 | - |

LI 0.85 was not in the sweep. The rutted-road and season tables for run 2 were produced by the
same job; add them here when copied from the PC (`results/seasons_*.md`, `results/sweep_LI_rutted_summary.md`).

## Comparison with run 1 (same LI values, smooth road)

| LI | DCM | Run 1 (no memory, 7.65M steps) | Run 2 (memory, 3M steps) |
|---|---|---|---|
| 0.5 | 100% | 0% | 0% |
| 0.6 | 100% | 0% | 100% |
| 0.65 | 100% | 0% | 100% |
| 0.7 | 100% | 0% | 70% |
| 0.75 | 0% | 40% | 100% |
| 0.8 | 0% | 100% | 100% |
| 0.9 | 0% | 0% | 0% |
| 1.0 | 0% | 0% | 0% |

The two runs differ in more than the history length (steps trained, curriculum speed) and each
has one seed, so the difference is not yet attributable to memory.
