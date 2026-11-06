# Stage 1, run 6: PPO on DCM with policy-chosen walking speed and a 300 s limit

Same as run 5 ([`stage1_dcm_ppo_run5_dr.md`](stage1_dcm_ppo_run5_dr.md): history 5, wet_start 0.4, wet_max 0.85, domain
randomisation, 25% dry roads) from scratch, plus `--speed_action` (13th output: walking speed 0.05 to 0.20 m/s, applied at
foot touchdown) and `--max_seconds 300`. One seed. Results copied from the PC, 2026-11-07. Design and reasons:
[`../docs/next_experiments.md`](../docs/next_experiments.md) section A.

| Item | Value |
|---|---|
| Run name | `g1-road-peak_monsoon-dcm-smooth-wet0.4-dr-max0.85-spd-t300-h5-s0` |
| Job | `kacha job peak_monsoon 6000000 "--robot g1 --base dcm --road smooth --wet_start 0.4 --wet_max 0.85 --dr --history 5 --speed_action --max_seconds 300 --then-evaluate"` (steps to be confirmed from the run's log) |
| Evaluation | final `model.zip`, 20 roads per cell, no assist, no observation noise, **time limit 300 s for every method in this file's sweep; the season tables printed below still show the DCM and scripted rows from the 60 s runs** (see caveat) |

Caveat: the season-table rows for the scripted controller and DCM are identical to the earlier files (same numbers and times),
which suggests they were not re-run with the 300 s limit in that table; the LI-sweep DCM row at LI 0.75 differs (70% against 0%
before), so the sweep did use the new limit. To be confirmed on the PC (`KACHA_MAX_SECONDS` in the job's environment).

## Soil-wetness sweep, smooth road (base peak_monsoon, 15 cm)

| LI | DCM (300 s) | Run 6 | Fell | Distance (m) | Cause of falls |
|---|---|---|---|---|---|
| 0.5 | 100% | **100%** (84-100) | 0% | 8.30 | - |
| 0.6 | 100% | **100%** | 0% | 8.30 | - |
| 0.65 | 100% | **100%** | 0% | 8.30 | - |
| 0.7 | 100% | **100%** | 0% | 8.30 | - |
| 0.75 | 70% (48-85) | **100%** | 0% | 8.30 | - |
| 0.8 | 0% | **100%** (84-100) | 0% | 8.30 | - |
| 0.9 | 0% | 0% (0-16) | 100% | -0.12 | sank 65%, slipped 30%, stuck 5% |
| 1.0 | 0% | 0% | 100% | -0.31 | sank 100% |

Step length shrinks with wetness (0.12 m at LI 0.5 to 0.09 m at LI 0.8) and cost of transport stays near 1 up to LI 0.75
(1.58 at 0.8), so the policy slows and shortens its steps as the mud gets wetter, as intended. This is the first policy to cross
LI 0.8 on every road and also LI 0.5 (run 5 got 40% there).

## Season tables (20 roads per cell, road seeds 50000+k)

| Road | Season | Run 6 | Run 5 (6M) | DCM |
|---|---|---|---|---|
| smooth | dry | **0%** (95% fell, 5% left road; 0.72 m in 7 s; slipped 16) | 50% | 65% |
| smooth | early monsoon | 100% (51.7 s) | 100% | 100% |
| smooth | peak monsoon (LI 1.00) | 0% (sank 20) | 0% | 0% |
| rutted | dry / early / peak | 0% / 0% / 0% | 0% | 0% |

Rutted-road LI sweep (0.5 to 1.0): 0% for DCM and run 6 at every LI; falls within 0.3 m even at LI 0.5 (geometry at the start,
see next_experiments.md section E).

## What the progress video showed (3.4M steps, four fixed conditions)

- Dry ground: after about 1.5 s the robot swings a leg up and forward and falls backwards at 2.1 s (distance -0.43 m).
  The final model also fails dry (0%, falls at about 7 s on average).
- LI 0.82 (3.4M): walks about 0.2 m in 9 s with commanded speed 0.13 to 0.15 m/s but makes almost no progress (stepping in
  place and drifting back), then falls at about 10 s. The final model crosses LI 0.8 on 20 of 20 roads, so this is an
  early-checkpoint behaviour.
- LI 0.60 and 0.75 reached the end (about 52 s and 55 s).

## Reading and what is next

- LI 0.8 is solved by letting the policy choose its speed and giving it time: 100% on 20 roads (run 5: 0%). One seed; the
  gain is from the speed output and the 300 s limit together (not separated).
- LI 0.85 was not in the sweep (the earlier memory run timed out there); 0.9 and 1.0 fail by sinking, as before.
- New weak point: dry ground is 0% (DCM 65%, run 5 50%). Hypotheses (not tested): the policy learned on mud (leg lift, slow
  speeds) does not suit dry ground; 25% dry roads in training may be too few for the new action; dry may need its own speed range.
- Next: test dry (record `--check` on dry roads, speed trace), LI 0.85 and 0.88 sweep, and the matched ablation (speed output
  with the 60 s limit, or 300 s without the speed output) to separate the two changes.
