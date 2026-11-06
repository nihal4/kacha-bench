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
  0.5 to 0.7 where DCM gets 100%, and 0% at LI 0.9 and 1.0 (run 1, no memory).
- With a 5-step observation history (run 2, one seed, 3M steps) the policy crosses 100% at LI
  0.6, 0.65, 0.75 and 0.8 and 70% at 0.7, so the breakdown moves from LI 0.70 to somewhere
  between 0.82 and 0.87: 100% at 0.82; at 0.85 nobody falls but the robot is too slow to finish
  in 60 s (7.81 of 8.3 m); falls start at 0.87 (60%) and are universal at 0.9 and 1.0. Its
  lower limit is sharp: 0% at LI 0.4, 0.45, 0.5 and 0.55, 100% at 0.6, just below its training
  range (which starts at 0.7). DCM alone is 100% at 0.4 to 0.7 and 0% from 0.75.
- Run 3 (memory, trained from LI 0.5 with `--wet_start 0.5`, one seed): 100% at LI 0.5, 0.6,
  0.65, 0.7, 0.75 and 0.8 and 0% at 0.9 and 1.0 (0.85 not tested yet). It covers all of
  DCM's range and extends beyond it to at least 0.8, and it removed run 2's LI 0.5 failure.
  Runs 2 and 3 differ mainly in the start of the training range, so this is close to a
  matched comparison, but each has one seed. On the season tables (20 roads each, smooth road)
  run 3 gets 100% in early monsoon (DCM 100%, run 1 5%), 0% in dry (DCM 65%) and 0% in peak
  monsoon (every method sinks at LI 1.00); on the rutted road every method gets 0% in every season.
- Run 4 (memory length 10, from LI 0.5, one seed): 100% at LI 0.5 to 0.75, 70% at 0.8, 0% at 0.9 and
  1.0; dry 0%, early monsoon 100%. Not better than history 5 (run 3: 100% up to 0.8), so history 5
  is kept. Longer memory does not fix the dry-season gap.
- Run 5 (domain randomisation: random mud depth with 25% dry roads, curriculum from LI 0.4 to 0.85,
  history 5, one seed): dry 50% (DCM 65%, run 3 0%), early monsoon 100%; LI sweep 100% at 0.5 to 0.75
  but **0% at 0.8** (run 3 had 100%). So the dry gap is closed at the cost of the wettest range; the
  cause is not isolated (three differences from run 3, one seed). Snapshot sweep: LI 0.8 is 0% at
  every snapshot (1.5M to 3M) and LI 0.75 only reached 100% at 3M, so the run looks under-trained for
  the wider training distribution, not regressed.

Not supported, do not claim:

- "Memory is what fixed it." Run 2 differs from run 1 in more than the history (3M against
  7.65M steps, curriculum speed) and each has one seed. Needs a matched no-memory run at 3M
  steps and more seeds of both.
- "PPO extends the breakdown point" as a general statement: it is one seed in one condition
  (smooth road, 15 cm mud), and run 1 showed a narrow-band policy that was worse than DCM
  outside the band.
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

Run 2 (memory test) finished on 2026-11-05: `g1-road-peak_monsoon-dcm-smooth-wet0.7-h5-s0`, same
setup as run 1 with `--history 5`, 3M steps. Smooth-road LI sweep: 100% at LI 0.6, 0.65, 0.75,
0.8; 70% at 0.7; 0% at 0.5, 0.9 and 1.0. See `results/stage1_dcm_ppo_run2_memory.md`. Its
rutted-road and season tables are not copied into the repo yet.

Run 3 finished (2026-11-06): `g1-road-peak_monsoon-dcm-smooth-wet0.5-h5-s0`, run 2 with the
curriculum starting at LI 0.5. Smooth-road sweep: 100% at LI 0.5, 0.6, 0.65, 0.7, 0.75, 0.8;
0% at 0.9 and 1.0. See `results/stage1_dcm_ppo_run3_wet05.md`.

Run 4 finished (2026-11-06): history 10 from LI 0.5, see `results/stage1_dcm_ppo_run4_h10.md`.

Run 5 finished (2026-11-06): domain-randomised run, see `results/stage1_dcm_ppo_run5_dr.md` (folder
`g1-road-peak_monsoon-dcm-smooth-wet0.4-h5-s0`).

Nothing was running when this was written.

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

Run 2 (memory, `--history 5`, 3M steps, one seed), same sweep (full table in
`results/stage1_dcm_ppo_run2_memory.md`): LI 0.5 0%, 0.6 100%, 0.65 100%, 0.7 70% (48-85),
0.75 100%, 0.8 100%, 0.9 0%, 1.0 0%. The upper limit moved from LI 0.70 (DCM) to between 0.8
and 0.9; the narrow band of run 1 is gone; LI 0.5 fails (slips; outside the training range).
Finer sweep (same file): 0% at 0.4, 0.45, 0.55; 100% at 0.82; 0.85 has no falls but only
7.81 of 8.3 m in 60 s (a timeout); 60% falls at 0.87; DCM is 100% at 0.4 to 0.55 and 0% at 0.82 to 0.9.

Season tables, smooth road (20 episodes), run 1: DCM dry 65%, early monsoon 100%, peak monsoon
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
   **Hypothesis:** it never saw those conditions. Run 3 (trained from LI 0.5) fixed early monsoon
   (100%) but dry is still 0% against DCM's 65%: dry has LI -0.2 and no mud layer, outside the
   training range.
5. **Policy has no wetness observation and no memory in run 1.** **Hypothesis:** this is why
   it works only in a narrow LI band. Run 2 (`--history 5`) widened the band to LI 0.6-0.8, which
   fits the hypothesis, but it is one seed and not a matched comparison (section 2).
5b. **LI 0.5 fails for the memory policy** (slips, 20% leave the road) though DCM crosses it.
   Run 3 (`--wet_start 0.5`) removed it (100% at LI 0.5 to 0.8, one seed), so this was a
   training-range effect.
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

1. DONE: run 2 (memory) finished and compared with run 1 (`results/stage1_dcm_ppo_run2_memory.md`).
1b. Make the memory result trustworthy. Done when there are at least 3 seeds of the memory
   policy (`--seed 1`, `--seed 2`, 3M steps each, about 2.5 h each) and a matched no-memory
   run at 3M steps with the same seeds, compared with intervals across seeds.
1c. DONE for run 2: finer LI sweep (0% at 0.4 to 0.55, 100% at 0.82, 0.85 is a timeout, falls
   from 0.87). DONE: `--wet_start 0.5` (run 3) fixed the LI 0.5 failure.
1d. Better baseline first, before the robustness tests. Target: one controller with at least
   90% success at every LI from 0.5 to 0.85 on the smooth road, over 3 seeds. Run 3 is at 100%
   from 0.5 to 0.8 (one seed). Remaining: fine sweep of run 3 at 0.4, 0.45, 0.82, 0.85, 0.87
   (the 0.85 point was a 60 s timeout for run 2, so consider reporting distance and falls
   separately from success); then seeds 1 and 2 of the run 3 configuration.
1e. Attribute the gain to memory: a matched no-memory run (`--history 1 --wet_start 0.5`, 3M
   steps, same seeds). History 10 (run 4, one seed) was not better than 5 (100% to LI 0.75, 70% at 0.8, dry 0%); keep 5. Implemented 2026-11-06 (tested by a short
   smoke run, not yet by a full run): `--wet_max LI` (curriculum ceiling), `--dr` with
   `--dry_share` and `--depth_range` (per-road mud depth, including none), and
   `--snapshot_every` (policy copies in `RUN/snapshots/`, usable as `run:PATH`). Not yet
   implemented: evaluating at mixed wetness during training.
1g. Design change (2026-11-06, user's idea, supported by results/dcm_slow_walking_check.md): the policy chooses its walking speed (`--speed_action`) and the crossing time limit is raised (`--max_seconds 300`, same for every method). New run from scratch with run 5's other settings; arms for balance are a later, separate change. The old 60 s results stay as the earlier baseline.
1f. (Run 5 at 6M: dry 50%, LI 0.6-0.75 100%, LI 0.8 0%, LI 0.5 40%; see results/stage1_dcm_ppo_run5_dr.md; next: retry with --depth_range 0.08 0.15 --dry_share 0.2.) Dry ground: run 3 fails the dry season (0%, DCM 65%). The baseline should also work
   there (it is the reference for normal conditions). Next run: domain-randomised training
   `--wet_start 0.4 --wet_max 0.85 --dr --history 5` (3M steps), then the season tables
   (dry should recover toward DCM's 65% or better) and the LI sweep (100% from 0.5 to 0.8
   should be kept). Also investigate why DCM itself is only 65% in dry, and re-run the dry
   comparison with 100 roads. Rough estimate (not validated): the sole's bearing capacity
   falls below body weight at about LI 0.86, so LI 1.00 (peak monsoon as defined) is probably
   a physical limit for this foot; decide whether to redefine peak monsoon (for example
   LI 0.85) or report LI 1.00 as that limit.
   Run 5 result: dry 50% and LI 0.5-0.75 at 100%, but LI 0.8 at 0%. The snapshot sweep shows it was
   never learned (0% at 1.5M to 3M) and LI 0.75 appeared only at 3M: under-trained. Next: continue the
   same run to 6M steps (copy the folder to the `-dr-max0.85` name, delete `results/DONE`, set the
   job to 6000000). If LI 0.8 is still 0%, retry with depths concentrated on the season depths
   (`--depth_range 0.08 0.15`, `--dry_share 0.2`).
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
# run 3 (memory, training range from LI 0.5)
kacha job peak_monsoon 3000000 "--robot g1 --base dcm --road smooth --wet_start 0.5 --history 5 --then-evaluate"
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
