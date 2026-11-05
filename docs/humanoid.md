# Humanoid tasks (Unitree G1)

KachaBench's humanoid track tests one task at a time, in order of difficulty:

| Stage | Task | Fails if | Status |
|---|---|---|---|
| 1 | Walk the length of a muddy, rutted village road | falls, or leaves the road | **built** (`kacha/g1_env.py`) |
| 2 | Same road, carrying a box with both hands | also: box touches mud, falls away, or is hit too hard | planned |
| 3 | Box on the head steadied with one hand, a second box in the other hand (as railway-station porters carry loads) | same, for both boxes | planned |
| later | Raised earthen road going uphill; muddy steps | | planned |

Code: `kacha/g1_scene.py` (robot on the road, contacts, gains), `kacha/gait.py`
(scripted mud-walking controller), `kacha/g1_env.py` (stage 1), `kacha/mud.py`
(`FlatFootMud`), `kacha/look.py` (scene appearance), `scripts/g1_preview.py`.

## Robot

MuJoCo Menagerie `unitree_g1` (29 actuators: 6 per leg, 3 waist, 7 per arm; about 35 kg,
1.32 m). The head is part of the torso link, so a box on the head moves rigidly with the
trunk (stage 3). In stage 1 the policy controls the 12 leg joints; waist and arms
follow the scripted controller (arms slightly out, small counter-swing).

Joint PD gains follow Unitree's G1 RL setup (unitree_rl_gym, `g1_config`): kp hip 100,
knee 150, ankle 40; kd 2, 4, 2. Waist 150/3, shoulders and elbows 40/1, wrists 20/0.5.
Only the soles (four corner points per foot), the pelvis and the torso touch the world;
the robot does not collide with itself. Physics step 5 ms, control 50 Hz.

With these real gains the G1 cannot stand by holding a pose: at an ankle stiffness of
40 N m/rad a 35 kg body tips faster than the ankle resists (gravity stiffness about
m g h, roughly 240 N m/rad). It has to balance actively, which is why the scripted
controller below includes balance feedback.

## Mud under flat feet

The quadruped mud model (`docs/mud_physics.md`) assumed a ball-shaped foot. G1 soles are
flat rectangles, about 17 x 6 cm (from the corner contact points plus their radius), so
the full area bears from the first millimetre. `FlatFootMud` keeps the same scheme (a
yielding support plate per foot, plate friction for sideways resistance, explicit
suction and swing drag) and the same soil strength `su` per season, with rectangular
footing formulas:

| Quantity | Formula | Source |
|---|---|---|
| Bearing | `Nc su B L f`, `Nc = 5 (1 + 0.2 D/B)(1 + 0.2 B/L)`, `D/B <= 2.5`; `f` = share of the sole touching (heel or toe during roll-off, at least 1/4) | Skempton (1951), rectangular footing on clay |
| Sole edges and shin | `alpha su 2 (B + L) min(d, 4 cm)` + `alpha su pi d_leg (d - 4 cm)` | adhesion, `alpha = 1` |
| Sideways | `alpha su B L` (sole adhesion) + `Np su B d` (front face), `Np` as in Matlock (1970) | |
| Suction on lift-off | `beta` x bearing x `d / zp` + edge adhesion | as for the quadruped |
| Plate stiffness | rigid punch with the equal-area circle | |

The plate turns with the foot's heading. What this gives for the G1 (343 N on one foot
while walking, 172 N per foot standing):

| Season | su (kPa) | Capacity at the surface | Result |
|---|---|---|---|
| Early monsoon | 6.8 | 460 N | feet stay on top: slippery (friction 0.3) with suction |
| Peak monsoon | 1.7 | 115 N | feet sink to the firm base, 15 cm (ankle deep) |
| Extreme | 0.7 | 46 N | sink to the firm base, 20 cm |

The same limitations as the quadruped model apply (`docs/mud_physics.md`, section 8),
plus: the sole is treated as flat on the mud even when the foot is tilted, and the
sideways resistance uses the narrow face for every direction.

## Reference gait (scripted, no motion capture)

No motion-capture data of people crossing muddy roads exists that we know of, so the
reference is generated: foot trajectories solved to joint angles by analytic leg inverse
kinematics (`leg_ik`, checked against MuJoCo's forward kinematics to 0.01 mm, sole kept
level). Its style follows measured human adaptations to slippery or unstable ground:

| Feature | Value | Basis |
|---|---|---|
| Shorter strides | steps of 0.22 m at 0.4 m/s | Cham & Redfern (2002): shorter normalised stride when a floor is expected to be slippery |
| Flat-footed landing | sole level throughout | Cham & Redfern (2002): smaller foot angle at heel contact; Gates et al. (2012): flatter foot at contact (about 11 vs 22 deg) |
| Longer double support | stance 65% of the cycle | Cham & Redfern (2002): slower loading of the foot |
| High foot lift | 10 cm | Gates et al. (2012): minimum toe clearance about 3.8 times higher, about 10 deg more knee flexion in swing |
| Lift before swinging forward | first 20% of the swing | assumption: breaking suction in mud |
| Arms slightly out | shoulder roll 0.3 rad | assumption |
| Stance knee | 0.29 rad | Unitree's G1 default; deeper crouches sagged under the real knee gain |
| Cycle | 1.1 s | assumption |

Gates et al. found no change in step width at matched speed, so the stance width is not
widened.

**Balance feedback** (`ScriptedWalker`): ankle torque from the centre-of-mass error over
the feet on the ground (gain 20 rad/m, 2 rad per m/s of velocity error), hip feedback on
trunk tilt (1 rad/rad), and step placement from the forward-speed error (Raibert 1986).
The gains were found by search in simulation: the robot stands indefinitely at Unitree's
gains, and steps for 2-4 s on flat ground before falling. Making it walk unaided was not
pursued further; the learned policy supplies the rest.

## Model-based baseline: capture-point (DCM) walking

`kacha/dcm.py` is a classical humanoid walking controller, the strong non-learned
baseline in the study (`--dcm` in `scripts/g1_evaluate.py`, `dcm` in
`scripts/g1_sweep.py`, `--controller dcm` in `scripts/g1_preview.py`).

| Layer | What it does | Source |
|---|---|---|
| ZMP plan | constant under the stance foot in single support (0.50 s), a straight move to the next foot in double support (0.20 s); footsteps 0.27 m apart at the commanded speed, ramped up over 3 steps | Englsberger et al. (2015) |
| DCM reference | exact solution of the divergent component of motion for each ZMP piece, solved backward from the end of the plan | Englsberger et al. (2015) |
| Balance | ZMP command r = r_ref + (1 + k/omega)(xi - xi_ref), k = 5, clipped to the sole (single support) or the hull of both soles | Kajita et al. (2003); Englsberger et al. (2015) |
| Step adjustment | the swing foot is sent to where the DCM will be at touchdown, minus the planned offset, within step limits | Khadiv et al. (2016) |
| Footholds | the landing point is moved up to 10 cm toward flatter ground; each sole is tilted to the ground slope under it (within the ankle range) | |
| Forces | the ground force m(g + c_ddot) acting at r is split between the feet; joint torques tau = bias(q, q_dot) - J_c^T lambda from the full rigid-body model | |
| Posture | IK puts the pelvis at 0.72 m and upright; trunk pitch and roll (300 N m/rad) and heading (100 N m/rad) are held through the stance hips | |
| Output | the G1's position servos carry the torque as a target offset, u = q_des + tau / kp | |

It uses the true simulator state, so it is an idealised, fully informed controller.
It sets its own servo gains (hips and knees kp 250, ankles 60) rather than Unitree's
RL gains; motor torque limits are unchanged. It walks at 0.2 m/s (about 45 s for the
road), so every method is evaluated with a 60 s limit.

What was needed to make it walk, found by tracing failures in simulation: planning
around the ankle rather than the sole centre; a double-support phase; starting from a
settled crouch; stiffer joint tracking than the RL gains; swing-foot position feedback
solved against the pelvis's real pose; trunk and heading feedback at the stance hips;
and asking the feet for no twisting moment (asking for any made the stance foot spin
loose). Gains were then chosen by grid and random search over perturbed starts.

**Results so far** (8.3 m road, 60 s, no assist):

| Road | Season | Crossed |
|---|---|---|
| smooth (no ruts or roughness) | dry | 14/16, 7/8, 6/6 in three batches |
| smooth | early monsoon | 8/8 |
| smooth | peak monsoon | 0/8 (6 sank, 2 slipped) |
| smooth, 0.5 cm surface roughness | dry | 0/6 |
| rutted (the monsoon road) | dry | 0/8 |

Soil-wetness sweep on the smooth road with 15 cm of mud (6 episodes per value):
success 100% at LI 0.6, 0% from LI 0.75; falls are slips at LI 0.75-0.85 and mostly
sinking at LI 0.9-1.0; cost of transport rises from 1.0 to 2.7 and steps shorten
from 0.14 to 0.09 m before the breakdown.

With 20 episodes per value (PC run): 100% for LI 0.5-0.7, 0% from 0.75 (70% too
slow to finish, the rest slip), slips at 0.8 and mostly sinking at 0.9-1.0; cost of
transport 0.88 at LI 0.5, 1.12 at 0.7, 2.8 at 0.75 and 7.3 at 0.8.

**Limits.** It is sensitive to surface roughness: even 0.5 cm stops it, because each
step loses a little forward momentum and the lag builds up. Re-planning the speed to
catch up made the plan jump and was turned off. Its success on long walks is also
chaotic: tiny numerical differences change single outcomes, so rates need many
episodes. Contact-based support areas (only the corners touching) were tried and
removed: contacts flicker and the support area jumped.

## Training on the DCM base (`--base dcm`)

The scripted gait proved too weak a base: two runs (1.5M steps here, 6.8M on the PC)
never learned to walk without the balance assist. With `--base dcm` the policy instead
corrects the DCM controller, which already crosses the smooth road. Every physics step
(200 Hz) the DCM computes its joint targets and the policy's correction is added to the
leg targets (half the scripted-base action scale, since the DCM servos are stiffer).
The policy acts at 50 Hz; its observation adds the DCM's tracking error (52 inputs),
the imitation reward compares the legs with the DCM's joint plan, and the commanded
speed is the DCM's 0.2 m/s. No balance assist by default (the DCM balances).

With zero corrections the environment is exactly the DCM controller: on the rutted
early-monsoon road it falls within 0.3 m (return -15) and on the smooth road it
crosses (return 92), so there is a large, learnable gap.

Cost: the DCM runs in Python at the physics rate, about 170 control steps/s per core
(it fails if run at 100 or 50 Hz), so roughly 500-700 steps/s on 4 cores: 15M steps
take about 6-8 hours.

One command trains and then evaluates (`--then-evaluate`): season tables on the smooth
and rutted roads and soil-wetness sweeps (trained policy, DCM, scripted gait; 20
episodes each), written to `runs/<run>/results/` and shown in the dashboard's Results
section:

```
kacha job early_monsoon 15000000 "--robot g1 --base dcm --then-evaluate"
kacha start
```

## Stage 1 task

The G1 starts at x = -4 m on a 10 m `muddy_rut_road` (deep, meandering wheel ruts with
standing water, plus two potholes) and must reach x = 4.3 m within 40 s at a commanded
speed of 0.25-0.4 m/s. A new road is generated every 20 episodes. The rut depth,
meander and puddle counts in the preset are visual estimates from field photos, not
measurements.

**Action** (12): corrections added to the scripted controller's leg targets, scaled
0.25 rad (hip pitch, knee), 0.20 (ankle pitch), 0.15 (hip roll), 0.10 (hip yaw, ankle roll).

**Observation** (50): gyro, gravity direction, body velocity, commanded speed, gait
phase (sin, cos), leg joint angles relative to the reference, leg joint velocities,
last action, which feet the reference has down. Uniform noise as in the Go1 task.

**Reward** (task-specific: cross the road, do not fall). Per control step, times 0.02 s:

| Term | Weight | |
|---|---|---|
| progress `clip(v_x, -0.3, cmd) / cmd` | +1.0 | only once walking starts; standing still earns 0 |
| alive | +0.2 | |
| upright (trunk z-axis up) | +0.3 | |
| imitation `exp(-|q - q_ref|^2 / 0.25)` | +0.3 | style guide, not exact copying |
| lateral offset `y^2` | -0.5 | |
| action rate | -0.02 | |
| energy `sum |q_dot tau|` | -0.0005 | |
| foot slip (sole speed while on the ground) | -0.1 | |

One-off: **-20** on a fall (trunk tilted past ~45 deg, pelvis below 0.45 m, or trunk on
the ground) or on leaving the road; **+20** on reaching the end. Falling and leaving the
road end the episode. Success = reached the end without failing.

## Training (4-core CPU, no GPU)

Measures that keep training from falling at every step, as asked, and avoid the late
collapse of the 50M-step Go1 run (best at 12.9M steps, then success fell from 42% to 6%
as the action noise shrank from 0.6 to 0.12):

- **Residual on a scripted controller**: exploration happens around a gait that already
  has the style and a balance reflex, not around random joint motion.
- **Small exploration**: initial action std 0.3 on the scaled corrections (about
  0.03-0.08 rad), no entropy bonus.
- **Balance assist curriculum** (Yu, Turk & Liu 2018): a virtual torque on the pelvis
  toward upright (400 N m/rad, 40 N m s/rad), starting at full strength. It only resists
  tilt; it does not push the robot forward or hold its weight. **Evaluation always runs
  without it.** Two schedules (`--assist_mode`):
  - `adaptive` (default): every 50k steps the assist drops by 0.05 if recent training
    episodes covered at least 3 m on average, i.e. it is reduced as the policy copes, as
    in Yu et al. A deadline stops it staying on for good: after `--assist_frac` (default
    40%) of training it fades linearly to zero over a further 10%.
  - `linear`: fades from 1 to 0 over the first `--assist_frac` of training.
  A 1.5M-step check run with the linear schedule (fade over 600k steps) showed why the
  adaptive one is the default: training distance per episode rose to 1.9 m at 200k
  steps, then fell as the assist dropped faster than the policy improved.
  The level is saved in `assist_state.json`, so `--resume` continues from it.
- **Stable updates**: KL target 0.02 per update, learning rate decaying linearly from
  3e-4 to 1e-5.

Networks 256-256-128 (ELU). Other PPO settings as for the Go1 (`docs/training.md`).
Speed: about 300-550 steps/s per core, so about 1,400 steps/s on 4 cores and about
10 hours for 50M steps.

Commands on the training PC:

```
kacha job early_monsoon 50000000 "--robot g1"    # -> runs/g1-road-early_monsoon-s0
kacha start
```

Suggested order: early monsoon (slippery, feet stay up) first, then peak monsoon (feet
sink to the firm base).

## Baselines

All baselines use the same reward, roads and evaluation. Train each with the flags
below (on the PC: `kacha job SEASON STEPS "--robot g1 FLAGS"`), then compare with
`scripts/g1_evaluate.py`.

| Baseline | Flags | Run name | What it tests |
|---|---|---|---|
| Scripted controller alone | (no training) `--scripted` in the evaluator | | the reference by itself: the floor |
| Unitree pretrained policy | (no training) `--unitree motion.pt` in the evaluator | | the manufacturer's rigid-ground walking policy on muddy roads |
| PPO, no reference | `--no_reference` | `g1-road-S-noref-s0` | is the scripted gait needed? |
| PPO + reference, no assist | `--assist_frac 0` | `g1-road-S-noassist-s0` | is the assist curriculum needed? |
| **PPO + reference + assist** | (default) | `g1-road-S-s0` | the full method |
| + height scan | `--scan` | `g1-road-S-scan-s0` | does seeing the ground help? (water still hides pothole depth) |
| + memory | `--history 5` | `g1-road-S-h5-s0` | does a short history help infer sinkage and slip? |

Run each over several seeds (`--seed 1`, `--seed 2`, ...) before comparing.

**No reference** uses Unitree's convention: leg targets are the default standing angles
(hip pitch -0.1, knee 0.3, ankle -0.2 rad) plus 0.25 rad x action, with the usual
exploration noise (std 0.6) since there is no gait to explore around. The imitation
reward is off; the gait clock (phase) stays in the observation.

**Unitree's policy** (`deploy/pre_train/g1/motion.pt` in
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)) was trained in
Isaac Gym on rigid ground for the 12-joint G1. The evaluator rebuilds its 47 inputs
exactly as Unitree's own MuJoCo deployment script does (gyro x 0.25, gravity direction,
command x (2, 2, 0.25), joint angles from the default, joint velocities x 0.05, last
action, 0.8 s gait clock) and applies its actions in the no-reference mode, which uses
the same gains, 50 Hz control and action scale. Differences to keep in mind: our model
is Menagerie's 29-joint G1 with the waist and arms held by PD rather than welded, and
our physics step is 5 ms (Unitree's deployment uses 2 ms). It is commanded the same
speeds as the other methods. A TorchScript file can run code when loaded, so load
only files from a source you trust.

```
git clone https://github.com/unitreerobotics/unitree_rl_gym ~/unitree_rl_gym
python scripts/g1_evaluate.py --scripted \
    --unitree ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt \
    --run ~/kacha/runs/g1-road-early_monsoon-s0 ~/kacha/runs/g1-road-early_monsoon-noref-s0 \
    --seasons dry early_monsoon peak_monsoon --episodes 10 --out results/stage1.md
```

The evaluator runs 10 episodes per season on road seeds 50000 + k (the same roads and
speed commands for every method), without assist or observation noise, and writes a
markdown table plus a JSON file with every episode.

Preview without training: `MUJOCO_GL=egl python scripts/g1_preview.py --season peak_monsoon --assist 1`.

## Open questions

1. The scripted controller alone falls after 2-4 s; the DCM controller is the strong
   non-learned baseline. Using the DCM controller as the base for the learned policy
   (instead of the scripted gait) is the obvious next step for training.
2. Box mass for stages 2-3 must respect the G1 arm rating (about 2 kg basic, 3 kg EDU,
   per arm, Unitree specifications); a head load goes through the legs and can be heavier.
3. Hands for stages 2-3: palms only (29 actuators) or three-finger hands (43).

## References

- Cham, R. & Redfern, M. S. (2002). Changes in gait when anticipating slippery floors. *Gait & Posture* 15(2).
- Gates, D. H., Wilken, J. M., Scott, S. J., Sinitski, E. H. & Dingwell, J. B. (2012). Kinematic strategies for walking across a destabilizing rock surface. *Gait & Posture* 35(1).
- Englsberger, J., Ott, C. & Albu-Schäffer, A. (2015). Three-dimensional bipedal walking control based on divergent component of motion. *IEEE Transactions on Robotics* 31(2).
- Kajita, S. et al. (2003). Biped walking pattern generation by using preview control of zero-moment point. *IEEE ICRA*.
- Khadiv, M., Herzog, A., Moosavian, S. A. A. & Righetti, L. (2016). Step timing adjustment: a step toward generating robust gaits. *IEEE-RAS Humanoids*.
- Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press.
- Skempton, A. W. (1951). The bearing capacity of clays. *Building Research Congress*.
- Matlock, H. (1970). Correlations for design of laterally loaded piles in soft clay. *OTC*.
- Yu, W., Turk, G. & Liu, C. K. (2018). Learning symmetric and low-energy locomotion. *ACM TOG (SIGGRAPH)* 37(4).
- Unitree Robotics, unitree_rl_gym (G1 configuration); Unitree G1 specifications.
