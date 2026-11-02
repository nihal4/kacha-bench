"""Film a trained policy on the same three test roads (used for progress videos).

Usage:
    MUJOCO_GL=egl python scripts/record.py --run runs/go1-none-s0 [--best] [--out video.mp4]
Each clip ends with a border: green = walked (success), red = fell or stood still.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

TEST_SEEDS = (7, 8, 9)
MAX_STEPS = 500          # 10 s of walking per clip


def record(run, out, best=False, seeds=TEST_SEEDS):
    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from kacha.walk_env import Go1WalkEnv

    cfg = json.load(open(os.path.join(run, "config.json")))
    mk = lambda: Go1WalkEnv(season=cfg["season"], height_scan=cfg["scan"], seed=0)
    m_path = os.path.join(run, "best_model.zip" if best else "model.zip")
    v_path = os.path.join(run, "best_vecnormalize.pkl" if best else "vecnormalize.pkl")
    norm = VecNormalize.load(v_path, DummyVecEnv([mk]))
    norm.training = False
    model = PPO.load(m_path, device="cpu")
    tmp = out[:-4] + ".part.mp4"
    writer = imageio.get_writer(tmp, fps=25, macro_block_size=8)
    results = []
    for seed in seeds:
        env = mk()
        obs, _ = env.reset(seed=seed)
        info, frame, done = {}, None, False
        for t in range(MAX_STEPS):
            a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            done = te or tr
            if t % 2 == 0 or done:
                frame = env.render()
                writer.append_data(frame)
            if done:
                break
        if not done:                                  # clip ended before the episode did
            walked = info["distance"] >= 0.5 * env.cmd * (t + 1) * env.dt
            info["is_success"] = walked
        ok = bool(info.get("is_success", False))
        results.append(ok)
        mark = frame.copy()
        mark[:10] = mark[-10:] = mark[:, :10] = mark[:, -10:] = (40, 180, 60) if ok else (200, 50, 50)
        for _ in range(20):
            writer.append_data(mark)
        for _ in range(6):
            writer.append_data(mark // 4)
        env.close()
    writer.close()
    os.replace(tmp, out)
    print(f"{out}: walked {sum(results)}/{len(results)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    record(a.run, a.out or os.path.join(a.run, "best.mp4" if a.best else "latest.mp4"), best=a.best)
