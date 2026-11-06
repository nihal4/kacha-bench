# Stage 1, run 4: PPO on DCM, memory length 10, trained from LI 0.5 (`--wet_start 0.5 --history 10`)

Same as run 3 ([`stage1_dcm_ppo_run3_wet05.md`](stage1_dcm_ppo_run3_wet05.md)) except that the policy sees the last 10
observations instead of 5. Results copied from the training PC on 2026-11-06. Interpretation is in
[`docs/status_and_plan.md`](../docs/status_and_plan.md).

| Item | Value |
|---|---|
| Run name | `g1-road-peak_monsoon-dcm-smooth-wet0.5-h10-s0` |
| Steps trained | 3,002,368 |
| Seed | 0 (one seed) |
| Evaluation | final `model.zip`, 20 episodes per condition, no assist, no observation noise |

## Soil-wetness sweep, smooth road (20 roads per LI)

| LI | DCM | Run 3 (history 5) | Run 4 (history 10) |
|---|---|---|---|
| 0.5 | 100% | 100% | 100% |
| 0.6 | 100% | 100% | 100% |
| 0.65 | 100% | 100% | 100% |
| 0.7 | 100% | 100% | 100% |
| 0.75 | 0% | 100% | 100% |
| 0.8 | 0% | 100% | 70% (48-85), 30% fell (slipped 83%, stuck 17%) |
| 0.9 | 0% | 0% | 0% (sank 90%) |
| 1.0 | 0% | 0% | 0% (sank 100%) |

Run 4 details (smooth road): distance 8.30 m for LI 0.5 to 0.75, 6.64 m at 0.8; cost of transport 1.08,
1.09, 1.15, 1.20, 1.18, 2.21 at LI 0.5, 0.6, 0.65, 0.7, 0.75, 0.8 (run 3: 1.03, 1.14, 1.19, 1.21, 1.23, 1.42).
LI 0.85 was not in this sweep.

## Season tables, smooth road (20 roads per season, road seeds 50000+k)

| Method | Season | Success | Fell | Left road | Distance | Time | Causes |
|---|---|---|---|---|---|---|---|
| run 4 | dry | 0% | 100% | 0% | 2.16 | 13.7 | slipped 18, no_trigger 2 |
| run 4 | early_monsoon | 100% | 0% | 0% | 8.30 | 45.8 | - |
| run 4 | peak_monsoon | 0% | 100% | 0% | -0.21 | 1.7 | sank 20 |

DCM and scripted rows are unchanged (see the run 3 file).

## Rutted road

Season tables: run 4 gets 0% in every season (dry 0.22 m, early monsoon 0.16 m, peak monsoon -0.16 m).
LI sweep: 0% at every LI from 0.5 to 1.0, distance between -0.27 m and 0.33 m, falls dominated by slipping
(LI 0.5 to 0.8) and sinking (0.9 and 1.0). DCM and run 3 behave the same way.

## Reading

History 10 is not better than history 5: it matches run 3 from LI 0.5 to 0.75, is lower at 0.8
(70% against 100%; the intervals overlap, so one seed cannot separate them), and still fails dry
ground. Longer memory does not fix the dry-season gap.
