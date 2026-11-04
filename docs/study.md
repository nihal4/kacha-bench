# Study plan: how and why humanoid walking fails on monsoon mud

KachaBench is used as a **robustness study**, not a leaderboard: a small number of
controllers, each differing in one way, tested in depth across controlled mud
conditions, with the cause of every failure recorded. The environment, physics and
tasks are described in `docs/humanoid.md`, `docs/mud_physics.md` and
`docs/water_physics.md`.

## Research questions

1. **Breakdown.** At what mud conditions does humanoid walking break down, and how
   sharply? (Success against soil strength, mud depth, friction, suction and water.)
2. **Mechanism.** Why does it fail? What share of falls are slips, sinking, suction
   trapping a foot, stepping where water hid a hole, or plain balance loss, and how
   does that mix change with the mud?
3. **Training conditions.** Does a policy trained on rigid ground (Unitree's) or on
   one season hold up in another? (Train-on / test-on matrix.)
4. **Load.** Does carrying a load (hand box, head load) bring the breakdown point
   earlier, and by how much? *(Needs stages 2-3.)*
5. **Gait adaptation.** Do trained policies change their gait as the mud gets worse
   the way people do on slippery or unstable ground (shorter steps, flatter landing,
   higher foot lift, longer double support)?

## Factors (what is varied)

One property at a time, everything else at the base season (`scripts/g1_sweep.py`):

| Axis | Meaning | Suggested values |
|---|---|---|
| `LI` | liquidity index: soil strength and suction together | -0.2, 0.2, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2 |
| `depth` | mud depth above the firm base (m) | 0.02, 0.05, 0.08, 0.12, 0.15, 0.20 |
| `mu` | surface friction of wet clay | 0.15, 0.2, 0.3, 0.45, 0.6 |
| `beta` | suction ratio alone | 0, 0.2, 0.4, 0.6 |
| `fill` | water in potholes and rut dips (hides depth) | 0, 0.5, 1.0 |
| road | `--road rutted` (the monsoon road) or `--road smooth` (no ruts, dips or roughness) | both |
| load | none, hand box, head load (and masses) | *(stages 2-3)* |

Because the season values in `docs/mud_physics.md` are assumptions, results are
reported **across the whole plausible range** of each property rather than for one
guessed value per season. A finding that holds over the range does not depend on the
assumption.

## Controllers (what is compared)

Each differs from the main one in one respect. Train each with 3 seeds (`--seed 0/1/2`).

| Controller | How | Purpose |
|---|---|---|
| Unitree pretrained | `unitree:motion.pt` | standard rigid-ground controller |
| Scripted gait alone | `scripted` | the reference without learning |
| DCM capture-point controller | `dcm` | classical model-based walking (`docs/humanoid.md`) |
| PPO, dry road | `kacha job dry 50000000 "--robot g1"` | trained without mud |
| PPO, early monsoon | `kacha job early_monsoon ...` | trained on slippery mud |
| PPO, mixed seasons | `kacha job mix ...` | trained on variety |
| PPO + memory | `... "--robot g1 --history 5"` | can it infer sinkage and slip? |

Optional: `--no_reference` and `--assist_frac 0` as checks that the scripted gait and
the assist are needed (`docs/humanoid.md`, "Baselines").

Compute on the 4-core PC: about 10 h per 50M-step run, so 4 trained controllers x 3
seeds is about 5 days. Sweeps are cheaper (episodes run on all 4 cores with
`--workers 4`).

## Measurements (recorded for every episode)

`G1MudRoadEnv` records, in `info["metrics"]` at the end of each episode:

- **Outcome**: success, off_road, too_slow, or the trigger of a fall.
- **Fall trigger** (what went wrong first in the 0.5 s before the collapse; the last
  0.1 s is ignored because the collapse itself slides and sinks the feet):
  - `hidden_water`: a foot sank where water covered a hole (water at the foot and
    yield over 4 cm)
  - `stuck`: suction over 60 N pulling down a lifting foot
  - `sank`: a loaded foot's mud yield over 8 cm
  - `slipped`: a loaded foot slid more than 4 cm in total (the slowest sole corner is
    used, so a foot rolling about a corner does not count; brief slides at touchdown
    are normal and much shorter)
  - `no_trigger`: none of these, i.e. a balance failure
- **Fall mode**: `tipped` (trunk went over) or `collapsed` (pelvis dropped below
  0.45 m with the trunk fairly upright: legs gave way or splits).
- **Gait** (walking part only, after the stand and speed ramp): step length, cadence,
  double-support share, peak foot lift above the mud surface in swing (0 = the foot
  never left the mud), mean stance sinkage, total loaded-foot sliding, cost of
  transport (joint mechanical work / (m g distance)), mean speed.
- **Conditions**: soil strength, mud depth, suction ratio, friction, water fill.

A foot counts as loaded when it carries more than 15% of body weight; contacts are
debounced over 3 control steps (60 ms) before counting a touchdown.

All thresholds are heuristics chosen by inspecting traces of the scripted controller;
they should be reported, and a sensitivity check (e.g. halving and doubling each)
belongs in the paper.

First results with the DCM controller (smooth road, 15 cm mud): success 100% at LI
0.6 and 0% from LI 0.75, falls shifting from slips (LI 0.75-0.85) to sinking (LI
0.9-1.0). With the scripted controller (no assist), the triggers already separate the
conditions as expected: on the dry road falls are balance failures, in early-monsoon
mud mostly slips, in peak-monsoon mud sinking (with an occasional suction trap).

## Analyses

1. **Robustness curves**: success (95% Wilson intervals) against each axis, one line
   per controller (`g1_sweep.py` writes the plot). Report the breakdown point: the
   value where success falls below 50%.
2. **Failure mix**: stacked shares of fall triggers against each axis.
3. **Train/test matrix**: rows = training condition, columns = test season, cells =
   success.
4. **Gait trends**: step length, foot lift, double support and cost of transport
   against `LI`, compared with the direction of the human adaptations in
   `docs/humanoid.md` (Cham & Redfern 2002; Gates et al. 2012).
5. **Statistics**: 3 training seeds per controller, 30-50 test roads per condition;
   compare controllers with intervals across seeds, not single runs.

## Commands

```
# baseline table per season
python scripts/g1_evaluate.py --scripted --unitree ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
    --run ~/kacha/runs/g1-road-early_monsoon-s0 --seasons dry early_monsoon peak_monsoon \
    --episodes 30 --out results/seasons.md

# robustness curve over soil wetness (add --road smooth to separate mud from ruts)
python scripts/g1_sweep.py --axis LI --values -0.2 0.2 0.5 0.7 0.9 1.0 1.1 1.2 \
    --methods dcm scripted run:~/kacha/runs/g1-road-early_monsoon-s0 \
              unitree:~/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
    --base peak_monsoon --episodes 30 --workers 4 --out results/sweep_LI
```

Run sweeps when no training is running, or with fewer workers.

## Scope and limits

- Stage 1 (walking) first; stages 2-3 add the load factor.
- The mud and water models are reduced-order and not yet validated against measured
  foot forces (`docs/mud_physics.md`, section 8). Findings are about the simulated
  physics; the sweeps over each property limit how much they rest on any one value.
- Uphill roads and muddy steps are optional extensions.
