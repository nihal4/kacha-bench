"""What happens just before a fall? Runs plain DCM (no policy) or a policy on mud of a given
wetness and prints, for the last steps before each fall: pelvis x, foot height above the ground,
foot sinkage, suction force and contact force on each foot, and the largest joint speed.
A 'kick' is reported when a foot rises more than KICK_H above the ground within 0.3 s.
Evaluation only. Example:
  python scripts/g1_fall_trace.py --ground 0.82 --episodes 6
  python scripts/g1_fall_trace.py --run RUN --ground 0.75 --episodes 10
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KICK_H = 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="trained run (default: plain DCM at --speed)")
    ap.add_argument("--ground", default="0.82")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--last", type=int, default=12, help="steps printed before each fall")
    a = ap.parse_args()
    from kacha.g1_env import G1MudRoadEnv, env_kwargs_from_config, SPEED_MIN, SPEED_MAX
    from kacha.terrain import ROADS
    if a.run:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        cfg = json.load(open(os.path.join(a.run, "config.json")))
        T = cfg.get("g1_max_seconds") or a.seconds
    else:
        cfg = dict(g1_reference=True, g1_history=1, g1_base="dcm", g1_road="smooth",
                   g1_speed_action=True, g1_wet_start=None)
        T = a.seconds
    kw = env_kwargs_from_config(cfg)
    kw["road"] = ROADS[cfg.get("g1_road") or "smooth"]
    kw["mud"] = dict(depth=0.0) if a.ground == "dry" else dict(LI=float(a.ground), depth=0.15)
    mk = lambda: G1MudRoadEnv(seed=0, look=False, max_seconds=T, **kw)
    if a.run:
        norm = VecNormalize.load(os.path.join(a.run, "vecnormalize.pkl"), DummyVecEnv([mk]))
        norm.training = False
        model = PPO.load(os.path.join(a.run, "model.zip"), device="cpu")
    code = 2.0 * (a.speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN) - 1.0
    ok = kicks = falls = 0
    for ep in range(a.episodes):
        env = mk(); obs, _ = env.reset(seed=900 + ep); done = False; rows = []
        while not done:
            if a.run:
                act, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
                act = act[0].astype(np.float64)
            else:
                act = np.zeros(env.action_space.shape[0]); act[12] = code
            obs, _, te, tr, info = env.step(act); done = te or tr
            d, m = env.data, env.model
            fz = []
            for k, gs in enumerate(env.mud.geoms if env.mud else []):
                pts = d.geom_xpos[gs]
                c = pts.mean(axis=0)
                fz.append(float(pts[:, 2].min() - env.mud.r_pt[k] - env.surface(c[0], c[1])))
            rows.append(dict(t=env.t, x=float(d.qpos[0]), fz=fz,
                             sink=env.mud.zp.copy() if env.mud else np.zeros(2),
                             suc=env.mud.suction.copy() if env.mud else np.zeros(2),
                             N=env.mud.normal.copy() if env.mud else np.zeros(2),
                             vmax=float(np.abs(d.qvel[6:18]).max())))
        succ = bool(info.get("is_success", False)); ok += succ
        if not succ:
            falls += 1
            tail = rows[-a.last:]
            kick = False
            for i in range(len(rows)):
                for k in range(2):
                    j = [r for r in rows[max(0, i - 15):i + 1]]
                    if j and max(r["fz"][k] for r in j) - min(r["fz"][k] for r in j) > KICK_H and \
                            rows[i]["fz"][k] > KICK_H and i >= len(rows) - 40:
                        kick = True
            kicks += kick
            print(f"\nroad {900+ep}: {info.get('outcome')}/{info.get('fall_mode')}  ended {env.t:.1f} s  "
                  f"distance {info['distance']:+.2f} m  kick={'yes' if kick else 'no'}")
            print(f"  {'t(s)':>6s} {'x(m)':>6s} {'footL z':>8s} {'footR z':>8s} {'sinkL':>6s} {'sinkR':>6s} "
                  f"{'sucL N':>7s} {'sucR N':>7s} {'N_L':>6s} {'N_R':>6s} {'max joint rad/s':>16s}")
            for r in tail:
                print(f"  {r['t']:6.2f} {r['x']:6.2f} {r['fz'][0]:8.3f} {r['fz'][1]:8.3f} {r['sink'][0]:6.3f} "
                      f"{r['sink'][1]:6.3f} {r['suc'][0]:7.0f} {r['suc'][1]:7.0f} {r['N'][0]:6.0f} {r['N'][1]:6.0f} {r['vmax']:16.1f}")
        env.close()
    print(f"\nground {a.ground}: {ok}/{a.episodes} walked, {falls} fell, {kicks} with a foot kicked up")


if __name__ == "__main__":
    main()
