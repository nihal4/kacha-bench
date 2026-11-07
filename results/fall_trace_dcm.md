# Fall trace, plain DCM (no policy), smooth road, 15 cm mud

Tool: `scripts/g1_fall_trace.py --ground LI --episodes N` (prints the last steps before each fall).
Run here with mujoco 3.17.0, roads seed 900-903, speed 0.15 m/s, 60 s limit.

| LI | walked | falls | falls with a foot thrown up (>0.12 m) |
|---|---|---|---|
| 0.78 | 0/4 (one timed out at 3.45 m) | 3 | 3 of 3 |
| 0.85 | 0/4 | 4 | 1 of 4 |

What the traces show
- Suction is small in the DCM-only runs: at most ~80 N, mostly 0-50 N. The "stuck by suction" idea
  is not what holds the foot here.
- At LI 0.85 the robot slides backwards from the start (distance -0.1 to -0.3 m, ends in 3-4 s).
- At LI 0.78 it stands nearly still for many seconds, then at a lift-off one foot goes from the
  ground to 0.5-0.65 m in about 0.1 s (about 7 m/s, joint speeds 20-40 rad/s). The DCM swing
  target at that moment is only 0.08-0.10 m above the ground, so the foot is far from where it
  was commanded.
- Not explained yet: what drives that leg so far from its target. To do next: log the servo error
  and torque on the swing leg, and run with `--run` on the arms and run 6 policies.
- Policy runs not checked here (no trained run in this container).
