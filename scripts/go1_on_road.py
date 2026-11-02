"""Put the Unitree Go1 on the village road, hold its standing pose, record a video.

Step 1 of the plan: check that the robot and terrain work together (rigid ground,
no mud or water yet). The robot does not walk; its motors just hold the "home" pose.

Usage (headless PC):
    MUJOCO_GL=egl python scripts/go1_on_road.py
Writes videos/go1_on_road.mp4 and prints whether the robot stayed standing.
"""

import argparse
import os

import imageio
import mujoco
import numpy as np

from kacha import RoadConfig, generate_village_road
from kacha.robot_scene import build_robot_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--out", default="videos/go1_on_road.mp4")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    cfg = RoadConfig()
    heights, _, _, _ = generate_village_road(cfg, seed=args.seed)
    model, home = build_robot_scene(heights, cfg.length, cfg.width)
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)

    targets = home[7:].copy()          # joint angles of the standing pose
    fps = 30
    steps_per_frame = int(round(1 / (fps * model.opt.timestep)))
    n_frames = int(args.seconds * fps)

    renderer = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 2.2, -20, 135

    start_z = data.qpos[2]
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            data.ctrl[:] = targets
            mujoco.mj_step(model, data)
        if renderer is not None:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    trunk_z = data.qpos[2]
    up = data.xmat[model.body("trunk").id].reshape(3, 3)[2, 2]  # 1 = upright
    feet = {"FR", "FL", "RR", "RL"}
    terrain_id = model.geom("terrain").id
    touching = set()
    for c in data.contact[: data.ncon]:
        for g in (c.geom1, c.geom2):
            name = model.body(model.geom_bodyid[g]).name
            other = c.geom2 if g == c.geom1 else c.geom1
            if other == terrain_id and name.split("_")[0] in feet and "calf" in name:
                touching.add(name.split("_")[0])

    print(f"Trunk height: start {start_z:.3f} m, end {trunk_z:.3f} m")
    print(f"Upright (1 = level): {up:.3f}")
    print(f"Feet touching the road: {sorted(touching)}")
    ok = np.isfinite(trunk_z) and up > 0.9 and len(touching) == 4
    print("Go1 is standing on the road." if ok else "Problem: robot is not standing on four feet.")

    if frames:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        imageio.mimsave(args.out, frames, fps=fps, macro_block_size=8)
        print(f"Video saved: {args.out}")


if __name__ == "__main__":
    main()
