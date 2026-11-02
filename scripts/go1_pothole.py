"""Go1 standing with its front-right foot over a water-filled pothole.

Shows the hidden-depth hazard: the water surface looks like ground to a height
scanner, but the foot drops to the pothole floor (plus any mud). The motors only hold
the standing pose; there is no walking yet.

Usage (headless PC):
    MUJOCO_GL=egl python scripts/go1_pothole.py --season peak_monsoon
"""

import argparse
import os

import imageio
import mujoco
import numpy as np

from kacha import RoadConfig, generate_village_road
from kacha.mud import SEASONS, MudContact, MudParams
from kacha.robot_scene import GO1_FEET, build_robot_scene, terrain_height
from kacha.water import SEASON_FILL, Water, WaterForces

FR_OFFSET = np.array([0.188, -0.127])     # FR foot relative to the trunk (home pose)


def roll_pitch(xmat):
    R = xmat.reshape(3, 3)
    return np.degrees(np.arctan2(R[2, 1], R[2, 2])), np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="peak_monsoon", choices=list(SEASONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    cfg = RoadConfig()
    heights, _, _, info = generate_village_road(cfg, seed=args.seed)
    hole = max(info["potholes"], key=lambda p: p["depth"])
    water = Water(info["potholes"], fill=SEASON_FILL[args.season])
    mud_p = MudParams.from_season(args.season)

    base_xy = np.array(hole["center"]) - FR_OFFSET
    model, home = build_robot_scene(heights, cfg.length, cfg.width, mud_depth=mud_p.depth,
                                    water=water, base_xy=base_xy)
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)

    surface = lambda x, y: terrain_height(heights, cfg.length, cfg.width, x, y)
    mud = MudContact(model, mud_p, surface, GO1_FEET) if mud_p.depth > 0 else None
    wforces = WaterForces(model, water)

    cx, cy = hole["center"]
    floor = surface(cx, cy)
    lvl = water.level_at(cx, cy)
    seen = water.perceived_height(cx, cy, floor)
    print(f"Season {args.season}: pothole {hole['severity']}, {hole['depth']*100:.1f} cm deep, "
          f"{hole['radius']*200:.0f} cm across")
    if lvl is not None:
        print(f"Water depth in the pothole: {(lvl - floor)*100:.1f} cm (fill {SEASON_FILL[args.season]:.0%})")
    print(f"Height scanner sees {seen:.3f} m; real floor {floor:.3f} m "
          f"(hidden drop {(seen - floor)*100:.1f} cm, plus {mud_p.depth*100:.0f} cm mud)")

    fr = model.geom("FR").id
    fr_start = data.geom_xpos[fr][2]
    fps = 30
    spf = int(round(1 / (fps * model.opt.timestep)))
    renderer, frames = None, []
    if not args.no_video:
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 1.3, -15, 60

    max_b = max_d = 0.0
    for _ in range(int(args.seconds * fps)):
        for _ in range(spf):
            data.ctrl[:] = home[7:]
            if mud:
                mud.apply(model, data)
            else:
                data.qfrc_applied[:] = 0
            wforces.apply(model, data)
            max_b, max_d = max(max_b, wforces.last_buoyancy), max(max_d, wforces.last_drag)
            mujoco.mj_step(model, data)
        if renderer:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    roll, pitch = roll_pitch(data.xmat[model.body("trunk").id])
    fr_drop = fr_start - data.geom_xpos[fr][2]
    print(f"FR foot dropped {fr_drop*100:.1f} cm; trunk roll {roll:+.1f} deg, pitch {pitch:+.1f} deg")
    if mud:
        print("Foot sinkage below mud/ground surface (cm): " + ", ".join(
            f"{n} {s*100:.1f}" for n, s in zip(GO1_FEET, mud.sinkage)))
    print(f"Largest water forces on the robot: buoyancy {max_b:.2f} N, drag {max_d:.2f} N "
          f"(robot weight 125 N)")
    up = data.xmat[model.body("trunk").id].reshape(3, 3)[2, 2]
    print("Go1 is still upright." if up > 0.9 else "Go1 tipped over.")

    if frames:
        out = f"videos/go1_pothole_{args.season}.mp4"
        os.makedirs("videos", exist_ok=True)
        imageio.mimsave(out, frames, fps=fps, macro_block_size=8)
        print(f"Video saved: {out}")


if __name__ == "__main__":
    main()
