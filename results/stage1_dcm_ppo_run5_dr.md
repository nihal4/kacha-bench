# Stage 1, run 5: PPO on DCM with domain randomisation (`--wet_start 0.4 --wet_max 0.85 --dr --history 5`)

Training roads get a random mud depth (4 to 15 cm) or no mud layer (25% dry roads), and the wetness curriculum
runs from LI 0.4 to a ceiling of 0.85. Results copied from the training PC on 2026-11-06. Interpretation is in
[`docs/status_and_plan.md`](../docs/status_and_plan.md).

| Item | Value |
|---|---|
| Run folder | `g1-road-peak_monsoon-dcm-smooth-wet0.4-h5-s0` (the folder name lacks the `-dr` and `-max0.85` tags: the `kacha` script built the name itself; fixed in commit 9cc8383; `config.json` has `g1_dr: true`, `g1_wet_max: 0.85`, `g1_dr_dry_share: 0.25`, depth range 0.04 to 0.15) |
| Job | `kacha job peak_monsoon 3000000 "--robot g1 --base dcm --road smooth --wet_start 0.4 --wet_max 0.85 --dr --history 5 --then-evaluate"` |
| Snapshots | `snapshots/step_000500000` to `step_003000000` (every 500k steps) |
| Seed | 0 (one seed) |
| Evaluation | final `model.zip`, 20 episodes per condition, no assist, no observation noise |

## Season tables (20 roads per season, road seeds 50000+k)

| Road | Season | Success | Fell | Distance (m) | Time (s) | Causes |
|---|---|---|---|---|---|---|
| smooth | dry | **50%** | 50% | 4.96 | 27.1 | slipped 6, no_trigger 4 |
| smooth | early_monsoon | 100% | 0% | 8.30 | 42.8 | - |
| smooth | peak_monsoon | 0% | 100% | -0.16 | 1.9 | sank 17, stuck 2, slipped 1 |
| rutted | dry | 0% | 100% | 0.02 | 2.9 | slipped 19, no_trigger 1 |
| rutted | early_monsoon | 0% | 100% | 0.05 | 3.1 | slipped 20 |
| rutted | peak_monsoon | 0% | 100% | -0.21 | 1.8 | sank 14, slipped 6 |

For reference, smooth road: DCM dry 65%, early monsoon 100% (45.4 s), peak monsoon 0%; run 3 (no domain randomisation) dry 0%.

## Soil-wetness sweep, smooth road (20 roads per LI)

| LI | DCM | Run 3 (no DR, from 0.5) | Run 5 (DR, from 0.4, cap 0.85) |
|---|---|---|---|
| 0.5 | 100% | 100% | 100% |
| 0.6 | 100% | 100% | 100% |
| 0.65 | 100% | 100% | 100% |
| 0.7 | 100% | 100% | 100% |
| 0.75 | 0% | 100% | 100% |
| 0.8 | 0% | **100%** | **0%** (distance 0.28 m, slipped 90%, sank 10%) |
| 0.9 | 0% | 0% | 0% (slipped 100%) |
| 1.0 | 0% | 0% | 0% (sank 70%, stuck 25%) |

Cost of transport of run 5 where it succeeds: 0.91, 0.99, 1.05, 1.09, 1.22 at LI 0.5, 0.6, 0.65, 0.7, 0.75 (DCM 0.88, 1.02, 1.06, 1.12 at 0.5 to 0.7).
LI 0.85 was not in this sweep. Rutted-road LI sweep not copied yet (expected 0% everywhere).

## Reading

- Domain randomisation recovered the dry season (0% to 50%; DCM 65%, intervals overlap) and kept early monsoon at 100%.
- It lost LI 0.8 (100% to 0%) on the 15 cm mud of the sweep. The runs differ in three ways (domain randomisation, start LI 0.4 against 0.5, ceiling 0.85 against 1.00) and each has one seed,
  so the cause is not isolated. **Hypothesis:** the hardest condition (deep mud at LI 0.8 and above) is a small share of the
  randomised training roads, so the policy practises it less.

## Snapshot sweep (smooth road, 20 roads per LI, run 5 snapshots)

Command: `g1_sweep.py --axis LI --values 0.75 0.8 0.85 --methods run:<RUN>/snapshots/step_N ...`, 2026-11-06.

| Snapshot | LI 0.75 | LI 0.8 | LI 0.85 |
|---|---|---|---|
| 1.5M steps | 0% | 0% | 0% |
| 2.0M | 0% | 0% | 0% |
| 2.5M | 0% (1.29 m) | 0% | 0% |
| 3.0M (final) | **100%** (8.30 m) | 0% (0.18 m) | 0% |

All failures are mostly slipping (80 to 100% of falls). LI 0.8 was never crossed by any snapshot, and LI 0.75 only at the
end of training. The run looks under-trained for the wider training distribution (progress still jumping between 2.5M
and 3.0M), not like a policy that learned LI 0.8 and then lost it. Whether more steps or a different depth range
is needed is the next experiment.

## Continued to 6M steps: season tables (smooth and rutted road, 20 episodes, road seeds 50000+k)

Run continued from 3.0M to 6.0M steps (same settings; resumed once at about 5.35M after a restart that also changed the
periodic eval, see `docs/training.md`). Final `model.zip` evaluated automatically; sweeps for this policy follow below
once copied. Dry LI -0.2 with no mud layer; early monsoon LI 0.70, 8 cm; peak monsoon LI 1.00, 15 cm.

| Road | Season | DCM | Run 5 at 6M | (Run 5 at 3M) |
|---|---|---|---|---|
| smooth | dry | 65% | 50% (6.35 m, slipped 8) | 50% |
| smooth | early monsoon | 100% | 100% (41.3 s) | 100% |
| smooth | peak monsoon (LI 1.00) | 0% | 0% (sank 18, slipped 2) | 0% |
| rutted | dry | 0% | 0% | 0% |
| rutted | early monsoon | 0% | 0% | 0% |
| rutted | peak monsoon | 0% | 0% | 0% |

- 3M more steps changed nothing in the season tables: dry stays at 50% (DCM 65%), peak (LI 1.00) at 0%.
- Training-mix check (20 smooth roads drawn from the training distribution, LI 0.4 to 0.85, 25% dry, depth 4 to 15 cm;
  `record.py --check 20`, no observation noise): 19 of 20 walked, all 7 dry roads walked; the one fall was at 3.9 kPa
  (about LI 0.82 to 0.85). Not a result: one seed, drawn from the training distribution.
- Fixed on 2026-11-06: progress videos and `record.py` filmed the rutted road (all 0%) instead of the run's road.

## Continued to 6M steps: smooth-road LI sweep (20 roads per LI, road seeds 70000+k, base peak_monsoon 15 cm)

Copied from the PC's `sweep_LI_smooth_summary.md`, 2026-11-06. DCM rows are the same as in the earlier files.

| LI | DCM | Run 5, 6M | Fell | Left road | Distance (m) | Main cause |
|---|---|---|---|---|---|---|
| 0.5 | 100% | 40% (22-61) | 0% | 60% | 8.18 | left the road (walked on, drifted off) |
| 0.6 | 100% | 100% (84-100) | 0% | 0% | 8.30 | - |
| 0.65 | 100% | 100% | 0% | 0% | 8.30 | - |
| 0.7 | 100% | 100% | 0% | 0% | 8.30 | - |
| 0.75 | 0% | 100% | 0% | 0% | 8.30 | - |
| 0.8 | 0% | 0% (0-16) | 100% | 0% | 1.78 | slipped 100% |
| 0.9 | 0% | 0% | 100% | 0% | 0.09 | slipped 80%, stuck 20% |
| 1.0 | 0% | 0% | 100% | 0% | -0.07 | sank 85% |

Reading:

- Works at LI 0.6 to 0.75 (100%, same as run 3 and run 2 there). LI 0.8 is still 0% after 6M steps, while run 2 (memory,
  wet 0.7 start, no DR) and run 3 (wet 0.5) got 100% at 0.8. So domain randomisation plus the wider range
  bought dry ground (0% in run 3, 50% here) and lost LI 0.8. One seed each, so this is a trade seen once, not a proven one.
- New weakness at LI 0.5 (40%): all 12 non-successes left the road after walking 8 m, which is steering drift on firm
  mud, not a fall. Run 3 got 100% at 0.5.
- Peak monsoon (LI 1.00) fails as before.

## Continued to 6M steps: rutted-road LI sweep (20 roads per LI)

DCM and run 5 are both 0% at every LI from 0.5 to 1.0. Falls happen in the first 0 to 0.35 m (distance 0.03 to 0.36 m at
LI 0.5 to 0.8) with 95 to 100% "slipped" even at LI 0.5, where the smooth road gives 100%. So the rutted road fails at the
start regardless of mud strength: the cause is the rut geometry (feet on rut walls), not wetness. This matches the plan to
build a geometry ladder (shallower and narrower ruts first) before testing mud on the rutted road.
