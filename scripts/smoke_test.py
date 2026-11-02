"""Build a village-road terrain, drop a test body on it and check it settles.

Usage:
    python scripts/smoke_test.py          # headless
    python scripts/smoke_test.py --view   # open the MuJoCo viewer
"""

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from kacha import RoadConfig, build_model, generate_village_road


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = RoadConfig()
    heights, _, _, info = generate_village_road(cfg, seed=args.seed)
    model = build_model(heights, cfg.length, cfg.width)
    data = mujoco.MjData(model)

    print(f"Terrain grid: {heights.shape}, heights {heights.min():.3f} to {heights.max():.3f} m")
    print(f"Potholes: {len(info['potholes'])}")

    if args.view:
        mujoco.viewer.launch(model, data)
        return

    for _ in range(2000):  # 4 seconds of simulated time
        mujoco.mj_step(model, data)

    z = data.qpos[2]
    speed = np.linalg.norm(data.qvel[:3])
    print(f"Probe height after 4 s: {z:.3f} m, speed {speed:.4f} m/s")
    assert np.isfinite(z), "simulation produced NaN"
    assert heights.min() - 0.5 < z < 1.0, "probe fell through or did not land"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
