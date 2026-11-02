"""Put the Unitree Go1 on the village road, hold its standing pose, record a video.

No walking: the motors just hold the "home" standing pose. Use it to check the robot
and terrain work together, on rigid ground or on mud for a given season.

Usage (headless PC):
    MUJOCO_GL=egl python scripts/go1_on_road.py                       # rigid road
    MUJOCO_GL=egl python scripts/go1_on_road.py --season peak_monsoon # mud
Seasons: dry, early_monsoon, peak_monsoon, extreme (see kacha/mud.py).
"""

import argparse
import os

import imageio
import mujoco
import numpy as np

from kacha import RoadConfig, generate_village_road
from kacha.mud import SEASONS, MudContact, MudParams
from kacha.robot_scene import build_robot_scene, terrain_height

FEET = ["FR", "FL", "RR", "RL"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--season", default="none", choices=["none"] + list(SEASONS))
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--out", default=None)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    out = args.out or f"videos/go1_{args.season}.mp4"

    cfg = RoadConfig()
    heights, _, _, _ = generate_village_road(cfg, seed=args.seed)
    mud_p = MudParams.from_season(args.season) if args.season != "none" else None
    depth = mud_p.depth if mud_p else 0.0
    model, home = build_robot_scene(heights, cfg.length, cfg.width, mud_depth=depth)
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)

    mud = None
    if mud_p and depth > 0:
        surface = lambda x, y: terrain_height(heights, cfg.length, cfg.width, x, y)
        mud = MudContact(model, mud_p, surface, FEET)
        print(f"Season {args.season}: su {mud_p.su/1e3:.2f} kPa, mud depth {depth*100:.0f} cm, "
              f"suction ratio {mud_p.beta:.2f}")

    targets = home[7:].copy()          # joint angles of the standing pose
    fps = 30
    steps_per_frame = int(round(1 / (fps * model.opt.timestep)))
    n_frames = int(args.seconds * fps)

    renderer = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 1.6, -12, 120

    start_z = data.qpos[2]
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            data.ctrl[:] = targets
            if mud:
                mud.apply(model, data)
            mujoco.mj_step(model, data)
        if renderer is not None:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    trunk_z = data.qpos[2]
    up = data.xmat[model.body("trunk").id].reshape(3, 3)[2, 2]  # 1 = level
    ground = terrain_height(heights, cfg.length, cfg.width, data.qpos[0], data.qpos[1])
    clearance = trunk_z - ground
    terrain_id = model.geom("terrain").id
    on_base = set()
    for c in data.contact[: data.ncon]:
        for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if other == terrain_id and model.geom(g).name in FEET:
                on_base.add(model.geom(g).name)

    print(f"Trunk height: start {start_z:.3f} m, end {trunk_z:.3f} m "
          f"({clearance:.3f} m above the {'mud' if mud else 'road'} surface)")
    print(f"Upright (1 = level): {up:.3f}")
    if mud:
        print("Foot sinkage (cm): " + ", ".join(
            f"{n} {s*100:.1f}" for n, s in zip(FEET, mud.sinkage)))
        print(f"Feet down to the firm base: {sorted(on_base) or 'none'}")
    else:
        print(f"Feet touching the road: {sorted(on_base)}")
    ok = np.isfinite(trunk_z) and up > 0.9 and (mud is not None or len(on_base) == 4)
    print("Go1 is standing." if ok else "Problem: robot is not standing properly.")

    if frames:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        imageio.mimsave(out, frames, fps=fps, macro_block_size=8)
        print(f"Video saved: {out}")


if __name__ == "__main__":
    main()
