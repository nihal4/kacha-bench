# KachaBench: status, plan and results

Read this first (humans and AI assistants). It says what the project claims, what is done,
what the numbers are, what is known to be wrong, and what comes next.
Last updated: 2026-11-05. Maintainer: S. M. Nihal Ahmed.

Rule for editing this file: every number needs a source (a file in `results/` or a run
folder), and anything not measured is labelled **hypothesis**. Update the date and the
"Where we are" section whenever something finishes.

## 1. What the project is

A simulation study (MuJoCo) of when, how and why a walking humanoid (Unitree G1) fails on
rural Bangladeshi monsoon roads: wet clay, ruts and water-filled potholes. The mud model is
a reduced-order model from soil mechanics (`docs/mud_physics.md`). The study design is in
`docs/study.md`, the humanoid task in `docs/humanoid.md`, training in `docs/training.md`.

Target: a journal-quality robustness study (see the plan in section 6). The framing is
a robustness and failure-analysis study, not a leaderboard.

## 2. What we can claim today, and what we cannot

Supported by the data so far (smooth road, 20 test roads per point, one simulator):

- The model-based capture-point (DCM) controller crosses the 8.3 m smooth road 20/20 at soil
  liquidity index LI 0.5 to 0.7 and 0/20 from LI 0.75 up. Falls at 0.75 to 0.8 are slips;
  from 0.9 up they are mostly sinking.
- A PPO policy trained on top of DCM with a soil-wetness curriculum (one seed, 7.65M steps)
  crosses 40% at LI 0.75 and 100% at LI 0.80, where DCM alone gets 0%, but it gets 0% at LI
  0.5 to 0.7 where DCM gets 100%, and 0% at LI 0.9 and 1.0.

Not supported, do not claim:

- "PPO extends the breakdown point." It works in a narrow band and is worse elsewhere.
- Anything about the rutted road: DCM, the scripted gait and the trained policy all get 0%
  there at every wetness, even at LI 0.5, so no reference condition exists on that road.
- Anything about real mud. The season values and several geometry values are assumptions
  (section 8) and the model is not validated against measured data.

## 3. Where we are

Done:

- Terrain, mud and water models; Go1 validation runs; G1 humanoid scene and environment
  with failure-cause recording and gait metrics.
- Controllers: scripted gait, DCM capture-point controller, PPO on the scripted base, PPO on
  the DCM base (`--base dcm`), evaluation of Unitree's pretrained policy (code exists,
  not run yet).
- Training PC workflow (`kacha` command, dashboard on port 8081) and one-command
  train-then-evaluate (`--then-evaluate`).
- DCM breakdown curve on the smooth road (20 roads per point).
- First full PPO-on-DCM run (run 1) and its evaluation: `results/stage1_dcm_ppo_run1.md`.

Running now (as of about 18:00 Dhaka time, 2026-11-05):

- Run 2, memory test: `g1-road-peak_monsoon-dcm-smooth-wet0.7-h5-s0`, same setup as run 1 with
  `--history 5`, 3,000,000 steps, then the automatic evaluation (about 19:15 expected).
  Question: does a policy that sees its recent past (and can infer the soil) fix the narrow
  band? Judge it only by the LI sweep on the smooth road, not by training numbers.

Not done: see the plan, section 6.

## 4. Results so far (numbers)

All from `results/stage1_dcm_ppo_run1.md` (full tables there). Smooth road, base season
peak monsoon (mud depth 15 cm), 20 episodes per value, 95% Wilson intervals:

| LI | DCM alone | PPO on DCM, run 1 (no memory, 1 seed) |
|---|---|---|
| 0.5 | 100% (84-100) | 0% |
| 0.6 | 100% | 0% |
| 0.65 | 100% | 0% |
| 0.7 | 100% | 0% |
| 0.75 | 0% (0-16) | 40% (22-61) |
| 0.8 | 0% | 100% (84-100) |
| 0.9 | 0% (sank 60%, slipped 40%) | 0% (slipped 95%) |
| 1.0 | 0% (sank 75%) | 0% (sank 100%) |

Season tables, smooth road (20 episodes): DCM dry 65%, early monsoon 100%, peak monsoon
0%; scripted gait 0% in all three; trained policy 0% / 5% / 0%.

Rutted road: 0% for every method in every season and at every LI from 0.5 to 1.0.

Training facts for run 1: 7.65M steps at about 370 steps/s (4 cores); the curriculum
reached LI 1.00 by about 1.0M steps; training-episode "solved" averaged about 35% and did
not trend upward after about 1.5M steps.

Earlier 800k-step check (before the "easier mud in the mix" fix): policy 20% at LI 0.70,
90% at 0.75, 100% at 0.80, 0% at 0.85 (no falls but too slow). The fix did not restore LI 0.70
in run 1 (0%); note the depth differs between the check's "early monsoon" season rows and
the sweep rows, so compare only sweep rows with sweep rows.

## 5. Known problems and decisions

1. **Eval during training is uninformative.** Eval episodes use the season's full wetness
   (LI 1.00) while training uses a curriculum, so eval success stays 0% and "best model"
   (`best_model.zip`) is selected on noise. Use the final `model.zip` and the LI sweeps.
   Possible fix: evaluate at a chosen LI (for example the current curriculum level or 0.8).
2. **Only the latest checkpoint is kept** (`model.zip` plus one `.bak`). A good earlier
   policy cannot be recovered. Possible fix: keep periodic checkpoints.
3. **Training "solved" is noisy.** Each reading averages about 100 episodes from about 5
   roads, each with its own wetness. Do not make decisions on single readings.
4. **Training depth is fixed at 15 cm.** Only LI varies in the curriculum. The policy
   performs badly in dry (0%) and early-monsoon (5%) season tables, where depth is 0 and 8 cm.
   **Hypothesis:** it never saw those conditions.
5. **Policy has no wetness observation and no memory in run 1.** **Hypothesis:** this is why
   it works only in a narrow LI band. Run 2 (`--history 5`) tests this.
6. **No working baseline on the rutted road.** Failures there cannot be attributed to
   mud. Needs a geometry ladder (section 6, step 6).
7. **DCM dry season is 65% on the smooth road but 100% at LI 0.5 with mud.** Unexplained
   (the failures are slips). Use LI 0.5 as the reference and investigate the dry case.
8. **The 50M-step plan was too long.** Run 1 plateaued by about 1.5M steps. Use 3M steps to test
   a change.
9. **The training PC runs about 370 steps/s**, not the 500-700 the docs estimated.
10. **Progress videos can be slow or killed** (low priority film job, 1 h limit while the 4
    training processes use all cores).

## 6. Plan

Each step has a "done when" line. Do one at a time and record the result here.

1. Finish run 2 (memory). Done when its LI sweep on the smooth road is compared with run 1
   in `results/`, and the claim about learning is decided.
2. Freeze the scope in one paragraph (smooth road and soil wetness as the main study; rutted
   road as future work unless step 6 gives a baseline).
3. Assumption register: complete the table in section 8 with source and effect for every
   assumed value. Done when each is marked measured, published, estimated or guessed.
4. Sensitivity sweeps: use `scripts/g1_sweep.py --axis` for depth, friction and suction (and
   the soil strength constants), halving and doubling each. Done when the breakdown LI is
   reported as a range with the assumptions that move it most.
5. Unitree pretrained baseline: `unitree:PATH` in `g1_sweep.py` with `motion.pt` from
   `unitree_rl_gym` (`deploy/pre_train/g1/motion.pt`). Done when its curve is on the same plot.
6. Geometry ladder: smooth, 0.5 cm roughness, 1 cm, shallow ruts, full ruts, to find where
   DCM stops working. Done when each rung has a reference condition or the baseline's limit
   is stated. Run mud sweeps only where the reference succeeds.
7. Validate the mud model against the Godon et al. (2025) Go1 foot-force data
   (`scripts/mud_probe.py`). Done when at least one measured curve is reproduced, or the gap is
   written down.
8. Seeds and statistics: train seeds 1 and 2 of the final learned configuration and use 30
   to 50 test roads per condition. Done when results have intervals across seeds.
9. Figures and failure analysis: breakdown curves with intervals, failure-cause mix against
   wetness, gait metrics. Done when every figure is regenerated by one script.
10. Reproducibility: pinned versions (done), committed configs and seeds, README that
    reproduces the main figure.
11. Write the paper (with limitations), check novelty against recent work, choose the venue
    last. Rough estimate: two to three months of part-time work.

Other things that can come later: stage 2 (box in hands) and stage 3 (head load), potholes
with hidden water in the evaluation, the Kaggle/MJX port, uphill and muddy steps.

## 7. How to run things

Training PC (always-on Linux, 4 cores): `bash ~/kacha/repo/scripts/desktop/setup.sh` once,
then:

```
kacha update                              # pull the latest code
kacha job SEASON STEPS "EXTRA FLAGS"      # set the job
kacha start | stop | restart | status | log
kacha export [RUN]                        # zip a run
```

Commands used so far:

```
# run 1 (no memory); the job was set to 7000000 afterwards to jump to the evaluation
kacha job peak_monsoon 50000000 "--robot g1 --base dcm --road smooth --wet_start 0.7 --then-evaluate"
# run 2 (memory)
kacha job peak_monsoon 3000000 "--robot g1 --base dcm --road smooth --wet_start 0.7 --history 5 --then-evaluate"
```

- The season only sets the curriculum ceiling (peak_monsoon = LI 1.00, depth 15 cm). The
  curriculum starts at `--wet_start` and rises 0.05 whenever 60% of recent training episodes
  succeed.
- Run folders are `~/kacha/runs/<name>`; names include the options and seed. `kacha job`
  always resumes an existing run with the same name.
- `--then-evaluate` writes `RUN/results/` (season tables and LI sweeps on both roads) and a
  `DONE` marker; it takes under 10 minutes. Setting the job to fewer steps than the run
  already has makes it skip training and only evaluate.
- The dashboard is at `http://<pc>:8081`.

## 8. Assumption register (first version, to be completed)

| Assumption | Where | Source | Status |
|---|---|---|---|
| Season LI and mud depth (dry -0.2/0, early monsoon 0.70/8 cm, peak 1.00/15 cm, extreme 1.20/20 cm) | `kacha/mud.py` `SEASONS` | marked ASSUMPTION in the code until field moisture is measured | assumed |
| Rut depth, meander, puddle counts on the muddy rutted road | `docs/humanoid.md` | visual estimates from field photos | estimated |
| Mud as a reduced-order model (bearing, sideways resistance, suction on each foot) | `kacha/mud.py` | soil mechanics, Bangladeshi clay data; `docs/mud_physics.md` | partly grounded, not validated |
| Suction share vs wetness (breakout ratio) | `kacha/mud.py` | trend from Godon et al. (2025) | published trend, not validated here |
| Pothole sizes | `docs/water_physics.md` | LGED (2025) rural-road guideline | published |
| Flat-foot mud contact for the G1 | `docs/humanoid.md` | modelling choice | assumed |
| DCM controller uses the true simulator state | `kacha/dcm.py` | modelling choice | idealised (upper bound) |
| Training mud depth fixed at 15 cm | `scripts/train.py` / `kacha/g1_env.py` | design choice | known limitation |
| Success = reach the end of the 8.3 m road within 60 s with no fall | `kacha/g1_env.py` | design choice | slow walkers fail (see LI 0.85 check) |

## 9. Notes for AI assistants working on this repo

- Do not state results that are not in `results/` or a run folder. Mark guesses as hypotheses.
- Do not start a long training run before checking the plan above. Test changes with about 3M
  steps. Change one thing per run.
- Judge learned policies by the fixed-road LI sweeps, never by training numbers or the
  training-time eval.
- Keep every result file with the exact command and run name that produced it.
- Update sections 3 to 6 of this file when a step finishes.
- The user runs training on their own PC; you generally cannot push to GitHub. Hand changes
  over as a patch or a zip.
