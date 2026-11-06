"""Why does a speed-action policy fall on a given ground? Runs the same roads in several modes:
  full         the policy as trained (leg corrections + its own speed)
  speed_only   its speed choice, leg corrections set to zero
  legs_only    its leg corrections, speed fixed (0.15 and 0.20 m/s)
  dcm_only     plain DCM at a fixed speed (0.15 and 0.20 m/s), no policy
Prints success, mean time of the end of the episode, mean commanded speed, and how big the leg corrections are.
Evaluation only. Example (dry ground):
  python scripts/g1_trace.py --run RUN --ground dry --episodes 10
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ground", default="dry", help="dry, or a soil wetness as LI (e.g. 0.8, 15 cm mud)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--noise", action="store_true")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from kacha.g1_env import G1MudRoadEnv, env_kwargs_from_config, SPEED_MIN, SPEED_MAX
    from kacha.terrain import ROADS
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    assert cfg.get("g1_speed_action"), "this tool is for runs trained with --speed_action"
    kw = env_kwargs_from_config(cfg)
    kw["road"] = ROADS[cfg.get("g1_road") or "smooth"]
    kw["mud"] = dict(depth=0.0) if a.ground == "dry" else dict(LI=float(a.ground), depth=0.15)
    T = cfg.get("g1_max_seconds") or 60.0
    mk = lambda: G1MudRoadEnv(seed=0, look=False, obs_noise=a.noise, max_seconds=T, **kw)
    norm = VecNormalize.load(os.path.join(a.run, "vecnormalize.pkl"), DummyVecEnv([mk]))
    norm.training = False
    model = PPO.load(os.path.join(a.run, "model.zip"), device="cpu")
    code = lambda v: 2.0 * (v - SPEED_MIN) / (SPEED_MAX - SPEED_MIN) - 1.0
    modes = [("full", None, None), ("speed_only", 0.0, None),
             ("legs_only @0.15", None, 0.15), ("legs_only @0.20", None, 0.20),
             ("dcm_only @0.15", 0.0, 0.15), ("dcm_only @0.20", 0.0, 0.20)]
    print(f"ground {a.ground}, {a.episodes} roads, time limit {T:g} s, noise {'on' if a.noise else 'off'}")
    print(f"{'mode':18s} {'success':>8s} {'ended at (s)':>13s} {'distance (m)':>13s} {'mean speed':>11s} {'mean |leg corr|':>16s}  causes")
    for name, leg_scale, fixed_speed in modes:
        ok, tend, dist, spd, mag, causes = 0, [], [], [], [], {}
        for ep in range(a.episodes):
            env = mk(); obs, _ = env.reset(seed=900 + ep); done = False
            while not done:
                act, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                act = act[0].astype(np.float64).copy()
                mag.append(float(np.abs(act[:12]).mean()))
                if leg_scale is not None:
                    act[:12] *= leg_scale
                if fixed_speed is not None:
                    act[12] = code(fixed_speed)
                obs, _, te, tr, info = env.step(act); done = te or tr
                spd.append(float(env.speed))
            ok += bool(info.get("is_success", False)); tend.append(env.t); dist.append(info.get("distance", 0.0))
            if not info.get("is_success", False):
                c = info.get("cause") or ("left road" if info.get("off_road") else "timeout")
                causes[c] = causes.get(c, 0) + 1
            env.close()
        print(f"{name:18s} {100 * ok / a.episodes:7.0f}% {np.mean(tend):13.1f} {np.mean(dist):13.2f} {np.mean(spd):11.3f} {np.mean(mag):16.3f}  {causes or '-'}", flush=True)


if __name__ == "__main__":
    main()
