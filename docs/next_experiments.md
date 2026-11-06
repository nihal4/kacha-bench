# Next experiments and open questions (written 2026-11-06)

Where the study stands: the DCM walking controller crosses the smooth road up to LI 0.70 (60 s limit); PPO on top of it
(run 5, 6M steps) crosses LI 0.6 to 0.75 and early monsoon, gets 50% on dry ground, 0% at LI 0.8 and above, and nothing crosses
the rutted road. Numbers: [`status_and_plan.md`](status_and_plan.md) and [`../results/`](../results/).
Everything below is a plan or an idea, not a result, unless it says DONE. Ideas marked (author) came from the author.

Ground rules: one change per training run; the old runs stay valid as the earlier baseline; any change to the rules of the
test (time limit, conditions) applies to every method in the comparison and is decided before looking at results.

## A. Policy-chosen walking speed and a longer time limit (author) - DONE as run 6, see results/stage1_dcm_ppo_run6_speed.md

Result: LI 0.5 to 0.8 all 100% (20 roads each) on the smooth road; LI 0.9 and 1.0 still 0%; dry ground 0% (new weak point). The text below is the plan as written before the run.

- Why: the DCM base walks at a fixed 0.2 m/s and the policy cannot change it, and the limit is 60 s. At LI 0.8 the plain
  DCM falls at 0.2 m/s but crosses all 6 roads at 0.1 m/s with 300 s (`results/dcm_slow_walking_check.md`). People walk
  slowly and carefully on slippery ground; the robot should be able to.
- What exists: `--speed_action` (13th policy output, 0.05 to 0.20 m/s, applied at foot touchdown) and `--max_seconds`
  (`docs/training.md`).
- Planned run: run 5's settings plus `--speed_action --max_seconds 300`, from scratch, 6M steps:
  `kacha job peak_monsoon 6000000 "--robot g1 --base dcm --road smooth --wet_start 0.4 --wet_max 0.85 --dr --history 5 --speed_action --max_seconds 300 --then-evaluate"`.
  Two things change at once (speed output and time limit) because slow walking needs time; the old DCM and run 5 are re-scored
  with 300 s for a fair comparison (`KACHA_MAX_SECONDS=300`).
- Done when: LI sweep 0.5 to 0.9 and season tables for this run and for DCM and run 5 at 300 s; decision rule: crossing
  LI 0.8 on at least 90% of 20 roads counts as solving it.

## A2. Why does run 6 fall on dry ground? (0%, DCM 65%) - diagnostic built, run on the PC pending

`scripts/g1_trace.py --run RUN --ground dry --episodes 10` runs the same dry roads six ways: the full policy; its speed
choice with the leg corrections zeroed; its leg corrections with the speed fixed at 0.15 or 0.20 m/s; and plain DCM at
0.15 and 0.20. The table says whether the leg corrections, the chosen speed, or the base itself is what falls. Evaluation
only (a few minutes). `--ground 0.8` runs the same split on LI 0.8 mud for comparison.

## B. Is run 5's loss of LI 0.8 a training-mix gap? - evaluation only, pending

- Calculation (not a measurement): only about 3% of run 5's training roads are in the hard cell (LI >= 0.78 and depth >= 12 cm).
- Test: `g1_sweep.py --axis depth --values 0.04 0.08 0.12 0.15 --fix LI=0.8` for run 5 and run 3. Passes at shallow depth but
  not at 15 cm: training-mix gap (then use a mix weighted to the hard cell). Fails at every depth: an LI problem, so rely on A.

## C. Mud-model review (author): heavy monsoon should be sticky and deep, not just slippery - proposed, not decided

Author's point: in heavy monsoon the mud gets sticky, the leg sinks and gets stuck, water fills holes and a leg can fall in and
stick, thicker mud needs more pull force to free the leg. It is not like a wet tile floor. What the model does today is in
[`mud_physics.md`](mud_physics.md) section 8: no sinkage until about LI 0.88, grip falling from about 1.0 to the 0.3 floor
between LI 0.5 and 0.85 (this is where "slipped" comes from), suction proportional to su so it weakens as mud gets wetter,
"stuck" 0 to 5% of falls, water-filled holes implemented but absent from the smooth-road baseline.

Steps, in the order proposed:

1. Sensitivity sweep (evaluation only): suction peak `beta_max` 0.6 to 1.0, foot size, `alpha`, on DCM and the best policy.
   Done when we know whether the conclusions (where the policies break down, the mix of causes) move.
2. Split the failure label "slipped" into surface slide and soil shear failure; rename to "traction loss" in tables.
3. Two regimes: early monsoon = slippery film on firm clay (low friction, small sinkage); peak monsoon = deep, soft, sticky mud,
   with water-filled holes on the road used for the peak-monsoon tests. Needs the author's decision on the definitions (see questions).
4. Time-dependent (rate-dependent) suction: real breakout force depends on how fast the foot is pulled; the model has none.
5. Validate against Godon et al. (2025) foot-force data, and plan a fall-cone test of real village-road soil (gives su directly).

Where the author's picture and the literature differ: Godon et al. (2025) measured suction on a Go1 foot peaking at
intermediate wetness and vanishing in nearly liquid mud, so "wetter = stickier" holds up to slurry and then reverses.

## D. Arms for balance (author) - idea, after A

- Idea: on slippery ground people use their arms for balance. Today the arms are held in a fixed pose by the DCM
  (`kacha/dcm.py`, upper-body joints `u0[12:]`) and the policy only corrects the 12 leg joints, so it cannot use them.
- Plan: add arm-joint corrections as extra policy outputs around the DCM's nominal arm pose (`--arm_action`), small range at first.
  One change per run: train it with the same settings as run A (speed output on), so run A is the control.
- Hypothesis: arm motion can absorb angular momentum and body tilt when a foot slides, reducing slip-induced falls at LI 0.75 to 0.85.
- Test: LI sweep 0.5 to 0.9 and fall-cause shares for run A versus run D; also run D with the arm outputs frozen at evaluation
  to show the arms are what matters. Risks: more outputs to learn (slower training), arms hitting the body, gain only on some causes.
- Decision rule: counts as an improvement only if success rises or falls drop at LI >= 0.75 on 20 roads per LI with
  intervals that do not overlap, or on 2 seeds.

## E. Rutted road

- Every method falls within the first 0.4 m at every LI, even at LI 0.5 (`results/stage1_dcm_ppo_run5_dr.md`), so the failure
  is the rut geometry at the start, not mud. Plan: a geometry ladder (shallower, narrower, then full ruts; DCM first, a
  scripted standing-start check) before any mud test on the rutted road.

## F. Smaller items

- LI 0.5 weakness of run 5 (40%: walks 8 m then leaves the road): check in run A.
- Dry ground: n = 100 roads, DCM against the best policy (`g1_evaluate.py --run RUN --dcm --road smooth --seasons dry --episodes 100`).
- Seeds 1 and 2 of the best configuration; matched no-memory run (`--history 1 --wet_start 0.5`).
- Failure-label check (do the labels match what happens), observation-noise tests, Unitree baseline, Godon validation.

## Questions for the author (decisions needed)

1. Peak monsoon: redefine it at about LI 0.85 (the controller's limit), or keep LI 1.00 and report it as a bearing limit?
2. Crossing time limit: is 300 s for every method acceptable, with crossing time reported next to success?
3. Mud model: go with C1 (sensitivity first) before changing the design, or change to two regimes (C3) now?
4. Arms: after run A only, or earlier at the cost of not knowing which change helped?
5. Is there any measurement of a real village road (soil wetness, mud depth, foot sinkage) we can get access to for C5?
