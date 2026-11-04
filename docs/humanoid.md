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

1. The scripted controller alone falls after 2-4 s; a stronger scripted baseline would
   make the benchmark's baseline table more informative.
2. Box mass for stages 2-3 must respect the G1 arm rating (about 2 kg basic, 3 kg EDU,
   per arm, Unitree specifications); a head load goes through the legs and can be heavier.
3. Hands for stages 2-3: palms only (29 actuators) or three-finger hands (43).

## References

- Cham, R. & Redfern, M. S. (2002). Changes in gait when anticipating slippery floors. *Gait & Posture* 15(2).
- Gates, D. H., Wilken, J. M., Scott, S. J., Sinitski, E. H. & Dingwell, J. B. (2012). Kinematic strategies for walking across a destabilizing rock surface. *Gait & Posture* 35(1).
- Raibert, M. H. (1986). *Legged Robots That Balance*. MIT Press.
- Skempton, A. W. (1951). The bearing capacity of clays. *Building Research Congress*.
- Matlock, H. (1970). Correlations for design of laterally loaded piles in soft clay. *OTC*.
- Yu, W., Turk, G. & Liu, C. K. (2018). Learning symmetric and low-energy locomotion. *ACM TOG (SIGGRAPH)* 37(4).
- Unitree Robotics, unitree_rl_gym (G1 configuration); Unitree G1 specifications.
