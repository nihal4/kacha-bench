"""Film a trained policy on the same three test roads (used for progress videos).

Usage:
    MUJOCO_GL=egl python scripts/record.py --run runs/go1-none-s0 [--best] [--out video.mp4]
Each clip ends with a border: green = success (Go1: walked; G1: reached the end of the
road), red = fell, left the road or stood still.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

TEST_SEEDS = (7, 8, 9)
MAX_STEPS = 500          # 10 s of walking per clip
G1_MAX_STEPS = 3000      # g1: the whole road (60 s); success = reached the end


def record(run, out, best=False, seeds=TEST_SEEDS, check=0, noise=False):
    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    cfg = json.load(open(os.path.join(run, "config.json")))
    g1 = cfg.get("robot") == "g1"
    if g1:
        from kacha.g1_env import G1MudRoadEnv, env_kwargs_from_config
        kw = env_kwargs_from_config(cfg)
        ws = cfg.get("g1_wet_start")
        if ws is not None:                       # film at the wetness range being trained
            from kacha.mud import SEASONS
            top = cfg.get("g1_wet_max") or SEASONS.get(cfg.get("season"), {}).get("LI", ws)
            kw["mud"] = dict(LI=top, LI_lo=ws)
            if cfg.get("g1_dr"):
                kw["mud"].update(depth_lo=cfg["g1_dr_depth_range"][0],
                                 depth_hi=cfg["g1_dr_depth_range"][1], p_dry=cfg["g1_dr_dry_share"])
        def mk():
            e = G1MudRoadEnv(seed=0, look=False, obs_noise=noise, max_seconds=60.0, **kw)
            e.render_size, e.cheap_render = (360, 640), True    # low graphics: fast
            return e
    else:
        from kacha.walk_env import Go1WalkEnv
        mk = lambda: Go1WalkEnv(season=cfg["season"], height_scan=cfg["scan"], seed=0,
                                reward=cfg.get("reward", "playground"))
    m_path = os.path.join(run, "best_model.zip" if best else "model.zip")
    v_path = os.path.join(run, "best_vecnormalize.pkl" if best else "vecnormalize.pkl")
    norm = VecNormalize.load(v_path, DummyVecEnv([mk]))
    norm.training = False
    model = PPO.load(m_path, device="cpu")
    if check:                      # no video: success of this exact setup on many roads
        res = []
        for sd in range(1000, 1000 + check):
            env = mk(); obs, _ = env.reset(seed=sd); done = False
            while not done:
                a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                obs, _, te, tr, info = env.step(a[0]); done = te or tr
            res.append(bool(info.get("is_success", False)))
            su = env.mud_p.su / 1e3 if env.mud_p is not None else None
            print(f"seed {sd}: {'ok' if res[-1] else 'FELL'}  su={su and round(su, 1)} kPa  dist={info.get('distance', 0):.2f}", flush=True)
            env.close()
        print(f"check: {sum(res)}/{len(res)} walked (noise={'on' if noise else 'off'})")
        return
    tmp = out[:-4] + ".part.mp4"
    writer = imageio.get_writer(tmp, fps=12, macro_block_size=8)
    results = []
    for seed in seeds:
        env = mk()
        obs, _ = env.reset(seed=seed)
        info, frame, done = {}, None, False
        for t in range(G1_MAX_STEPS if g1 else MAX_STEPS):
            a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            done = te or tr
            if t % 4 == 0 or done:
                frame = env.render()
                writer.append_data(frame)
            if done:
                break
        if not done and not g1:                       # clip ended before the episode did
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
    ap.add_argument("--check", type=int, default=0, help="no video: run N roads, print success")
    ap.add_argument("--noise", action="store_true", help="observation noise on (as in training)")
    a = ap.parse_args()
    if a.check:
        record(a.run, "", best=a.best, check=a.check, noise=a.noise)
        raise SystemExit
    record(a.run, a.out or os.path.join(a.run, "best.mp4" if a.best else "latest.mp4"), best=a.best)
