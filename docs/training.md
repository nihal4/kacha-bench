# Training the Go1 on the village road

Code: `kacha/walk_env.py` (environment), `scripts/train.py` (PPO), `scripts/record.py`
(videos), `scripts/desktop/` (always-on PC: services, `kacha` command, dashboard).

## Environment

The Go1 starts at one end of a generated 10 m road and must walk forward at a commanded
speed (0.3–0.8 m/s, new each episode) for up to 20 s. A new road is generated every 20
episodes. With `--season`, mud and water-filled potholes are switched on (`mix` draws dry,
early or peak monsoon per road).

Settings follow MuJoCo Playground's Go1 joystick task
(`mujoco_playground/_src/locomotion/go1/joystick.py`):

| Setting | Value |
|---|---|
| Control / physics step | 0.02 s / 0.004 s |
| Joint servos | Kp 35, Kd 0.5, action scale 0.5 rad around the standing pose |
| Observation (46) | body velocity, gyro, gravity direction, command, joint angles and velocities, last action; uniform noise as in Playground |
| `--scan` adds (45) | heights at 9 x 5 points ahead of the robot; water hides pothole depth |
| Termination | trunk turned past horizontal, or trunk touching the ground |

Reward (each term multiplied by the 0.02 s control step), Playground scales:

| Term | Scale | Term | Scale |
|---|---|---|---|
| tracking_lin_vel `exp(-|v - cmd|^2 / 0.25)` | +1.0 | dof_pos_limits | -1.0 |
| tracking_ang_vel `exp(-w_z^2 / 0.25)` | +0.5 | termination | -1.0 |
| pose `exp(-sum w (q - q0)^2)` | +0.5 | torques | -0.0002 |
| feet_air_time (threshold 0.1 s) | +0.1 | action_rate | -0.01 |
| lin_vel_z | -0.5 | energy | -0.001 |
| ang_vel_xy | -0.05 | feet_slip | -0.1 |
| orientation | -5.0 | | |

Not used: Playground's foot-clearance and foot-height terms, and lateral and yaw commands.

### Reward versions

`Go1WalkEnv(reward=...)` and `train.py --reward` select one. `v2` is the default; a run
started without the flag before v2 existed is resumed with `playground` (read from its
`config.json`).

**`playground`** is the port above, with the summed step reward clipped at zero (legged_gym
"only positive rewards"). Three problems, measured in the environment (`seed 0`, 300
steps):

1. *Standing still pays almost as much as walking.* With the fixed sigma 0.25, `v = 0`
   earns `exp(-cmd^2/0.25)` of the tracking reward: 0.70 at 0.3 m/s, 0.30 at 0.55 m/s.
   `pose` and `tracking_ang_vel` pay up to 1.0/s more for not moving. Zero actions scored
   0.033 per step against a ceiling of about 0.04 for perfect walking (before penalties).
   Playground avoids this with lateral/yaw commands and a stand-still cost; with forward
   commands only, nothing does.
2. *The torque cost was `sum(tau^2)`, not Playground's `||tau||_2 + ||tau||_1`.* Under PPO's
   initial exploration noise (std 0.6) it cost 0.197 per step against 0.024 for
   Playground's form, more than the whole standing reward. This is the likely cause of the
   earlier drift to standing still that the clip was added to fix.
3. *The zero clip hides penalties in mud.* In peak monsoon with exploration noise, 60% of
   steps scored exactly 0, so tilting, slipping and sinking gave no learning signal.

**`v2`** changes only these three things:

| Change | v2 | Effect (zero actions / noise std 0.6, rigid road) |
|---|---|---|
| Forward tracking | `exp(-(v_x - cmd)^2 / (0.5 cmd)^2 - v_y^2 / 0.25)`; `pose` and `tracking_ang_vel` multiplied by `clip(v_x / cmd, 0, 1)` | standing: 0.033 → 0.0003 per step; walking at the command still earns ~0.04 |
| Torque cost | `||tau||_2 + ||tau||_1` (Playground) | 0.197 → 0.024 per step under noise |
| Clip | summed reward floored at `-1.0/s` (−0.02 per step) instead of 0; one-off fall penalty 3.0 outside the clip | penalised steps visible again; see below |

Choosing the floor: under exploration noise the summed terms have a median of −0.44/s on
the rigid road and −1.3/s in peak-monsoon mud (orientation and body rotation dominate),
so −1.0/s keeps most of that signal (14% of noisy rigid-road steps reach the floor, against
69% with a −0.25/s floor). A negative reward invites falling on purpose to end the episode;
the most that saves is `1.0/s × dt / (1 − gamma) = 2.0`, so the fall penalty is 3.0.
These two constants are design choices and belong in the sensitivity study.

**Success ("walked")**: the episode ends without a fall AND the robot reached the end of
the road or covered at least half the commanded distance. Standing still is not success.

### Speed choices

- Only feet and trunk collide with the ground, like Playground's "feet only" Go1:
  leg capsules on the height field created dozens of contacts and made each physics step
  ~25 times slower.
- Ground roughness is smooth over ~10 cm (cell-by-cell noise made tiny spikes and 10+
  contacts per foot).
- Grid 4 cm: the smallest LGED pothole (15 cm) still spans about 4 cells.

Result: about 500–650 environment steps per second per CPU core with random actions.

## PPO (Stable-Baselines3, CPU)

Networks 256-128-64 (ELU) for policy and value, learning rate 3e-4, 1024 steps per
environment per update, batch 1024, 5 epochs, gamma 0.99, GAE 0.95, clip 0.2, initial
action std about 0.6, observation normalisation. Profiles: `desktop` (4 environments,
one per core) and `smoke` (2 environments, ~1 minute).

Checkpoints are written every 50k steps, power-cut safe (temporary file, then rename;
the previous one kept as `.bak`). `--resume` continues from the last checkpoint and
keeps the evaluation history. Evaluation every 100k steps on 5 fixed episodes; the best
model is kept with its input normalisation.

## On the training PC

One-time setup: `bash ~/kacha/repo/scripts/desktop/setup.sh`. Then:

```
kacha job none 20000000     # season and steps
kacha start                 # training service (never starts at boot by itself)
kacha status                # progress, evaluation, speed, CPU temperature
kacha log                   # follow the log
kacha video                 # film the best policy now
kacha export                # zip a run
kacha update                # pull new code
```

Dashboard: `http://<pc>:8081` (curves, live numbers, progress videos every 500k steps).
It runs next to the fly-brain dashboard on 8080 without conflict.

## G1 training options (stage 1)

- `--base dcm`: PPO corrects the DCM capture-point controller. `--history N`: the policy sees
  the last N observations (a simple memory).
- `--wet_start LI`: soil-wetness curriculum. Training starts at this liquidity index and rises by 0.05
  whenever 60% of recent training episodes succeed; each road's wetness is drawn between
  `--wet_start` and the current level. `--wet_max LI` caps the curriculum (default: the season's LI,
  1.00 for peak monsoon, which nothing crosses; 0.85 is a reasonable ceiling).
- `--dr`: domain randomisation of the mud layer. Each training road gets a mud depth drawn from
  `--depth_range LO HI` (default 0.04 0.15 m), or no mud layer (dry ground) with probability
  `--dry_share` (default 0.25). Needs `--wet_start`. Run names get `-dr` and `-max<LI>` tags.
- `--snapshot_every STEPS` (default 500000, 0 = off): copies the policy to `RUN/snapshots/step_N/`
  (model, normalisation, config). A snapshot folder works as `--methods run:PATH` in `g1_sweep.py`
  and `g1_evaluate.py`, so an earlier policy can be recovered if later training gets worse.

Example (the next baseline run, see `docs/status_and_plan.md`):

```
kacha job peak_monsoon 3000000 "--robot g1 --base dcm --road smooth --wet_start 0.4 --wet_max 0.85 --dr --history 5 --then-evaluate"
```

## Dashboard eval and progress videos (changed 2026-11-06)

- The periodic evaluation during training (dashboard "Walked (eval)", `best_model.zip`) now uses the
  wetness range being trained (LI from `--wet_start` to `--wet_max`/season, with the `--dr` depth
  distribution), one fresh road per episode. Before, it ran at the season's LI 1.00, which no policy
  passes, so it always read 0%. It is still only a progress signal: judge policies by the LI sweeps.
- Progress videos (`scripts/record.py`) are now cheap: no shadows, reflections, sky or plants,
  360x640, 12 fps, filmed at the training wetness range. A video takes about 35 s. The watcher writes
  `runs/<run>/progress/step_<N>k.log` with the recorder's output, so a failure can be diagnosed.

## Policy-chosen walking speed and time limit (added 2026-11-06)

- `--speed_action` (needs `--base dcm`): the policy gets a 13th output that sets the DCM's walking speed between
  0.05 and 0.20 m/s (starts at 0.15). The speed is taken up only at foot touchdown and changes by at most 0.03 m/s per
  step (`kacha/dcm.py`, `V_STEP`), so the footstep plan never jumps mid-swing. The observation already contains the
  speed (now the planned speed). Checked on early-monsoon smooth road: constant 0.15 crosses, constant 0.20 is the
  controller's reliable speed, 0.25 falls (so the range stops at 0.20), 0.05 drifts off the road after about 130 s, and
  speeding up or slowing down mid-walk did not cause falls on its own. Changes the action size: always a new run.
- `--max_seconds S`: time allowed to cross the road, in training and in the automatic evaluation (which passes it to
  `g1_evaluate.py` and `g1_sweep.py` through the `KACHA_MAX_SECONDS` environment variable). The limit must be the same
  for every method in a comparison. Reason: at LI 0.8 the DCM crosses at 0.1 m/s but needs about 200 s
  (`results/dcm_slow_walking_check.md`).
- Run names get `-spd` and `-t<S>` tags (`kacha job ... "--speed_action --max_seconds 300"`).
- Progress videos are lit (soft light from the camera, no shadows, sky background) and carry a text overlay: run name,
  steps trained, clip and road seed, the road's mud (wetness LI, strength, depth, or "dry ground"), live time, distance
  and walking speed, and at the end of each clip FELL / LEFT THE ROAD / OUT OF TIME / REACHED THE END.
- Videos from runs trained with a wetness curriculum film four fixed conditions, the same in every video so checkpoints can
  be compared: dry ground, firm mud LI 0.60, LI 0.75 and the hard case LI 0.82 (15 cm mud). Outcomes are whatever they
  are (no balancing of successes and failures). Videos play in real time (12.5 fps, one frame per 0.08 s of simulation). A clip is cut after 120 s of
  simulated time and then says STILL WALKING, so a slow walker's video is up to 4 x 2 minutes long and takes some minutes to render at low priority.
