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

As in Playground, the summed step reward (times the control step) is **clipped at zero**
(legged_gym calls this "only positive rewards"). Without the clip, a first learning check
drifted to standing still: with exploration noise, the energy and torque penalties per
step outweighed the tracking reward, so moving less was the easiest way to score more.

Not used: Playground's foot-clearance and foot-height terms, and lateral and yaw commands.

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
