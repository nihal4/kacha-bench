"""Film the G1 on the muddy road with the scripted controller only (no learned policy).

Shows the reference mud-walking gait, the mud physics and the scene before training.
With --assist the balance assist used early in training is on.

Usage (headless PC):
    MUJOCO_GL=egl python scripts/g1_preview.py --season peak_monsoon
    MUJOCO_GL=egl python scripts/g1_preview.py --season early_monsoon --assist 1.0 --seconds 15
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from kacha.g1_env import G1MudRoadEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="peak_monsoon",
                    choices=["dry", "early_monsoon", "peak_monsoon", "extreme"])
    ap.add_argument("--assist", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or f"videos/g1_{a.season}{'_assist' if a.assist else ''}.mp4"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    env = G1MudRoadEnv(season=a.season, seed=a.seed, look=True, obs_noise=False,
                       assist=a.assist, max_seconds=a.seconds)
    env.reset(seed=a.seed)
    writer = imageio.get_writer(out, fps=25, macro_block_size=8)
    info, done = {}, False
    while not done:
        _, _, te, tr, info = env.step(np.zeros(12))
        done = te or tr
        if env.steps % 2 == 0 or done:
            writer.append_data(env.render())
    for _ in range(25):
        writer.append_data(env.render())
    writer.close()
    env.close()
    why = "reached the end" if info.get("is_success") else \
          "fell" if info.get("fell") else "left the road" if info.get("off_road") else "time up"
    print(f"{out}: {env.t:.1f} s, {info['distance']:.2f} m, {why}, "
          f"sinkage {np.round(100 * info['sinkage'], 1)} cm")


if __name__ == "__main__":
    main()
