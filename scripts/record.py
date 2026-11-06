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
VIDEO_MAX_S = 120.0      # a clip is cut after this much simulated time (videos play in real time)
# G1 videos: four fixed, informative conditions (the same in every video, so checkpoints can be compared);
# smooth-road training runs only. Mud depth 15 cm = the peak-monsoon layer used in the LI sweeps.
G1_CLIPS = (("dry ground", dict(depth=0.0)),
            ("firm mud, LI 0.60", dict(LI=0.60, depth=0.15)),
            ("edge of proven band, LI 0.75", dict(LI=0.75, depth=0.15)),
            ("hard case, LI 0.82", dict(LI=0.82, depth=0.15)))


def _font(size):
    try:
        import matplotlib
        from PIL import ImageFont
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def overlay(frame, lines, big=None):
    """Draw text lines on a dark translucent box at the top-left of the frame (copy)."""
    import numpy as np
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).convert("RGBA")
    f = _font(max(12, frame.shape[0] // 28))
    pad, lh = 8, f.size + 4
    box = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(box)
    w = max(d.textlength(t, font=f) for t in lines) + 2 * pad
    d.rectangle([0, 0, w, lh * len(lines) + 2 * pad], fill=(0, 0, 0, 150))
    for i, t in enumerate(lines):
        d.text((pad, pad + i * lh), t, font=f, fill=(255, 255, 255, 255))
    if big:
        text, col = big
        fb = _font(frame.shape[0] // 9)
        tw = d.textlength(text, font=fb)
        d.rectangle([(img.width - tw) / 2 - 16, img.height * 0.40, (img.width + tw) / 2 + 16, img.height * 0.40 + fb.size + 24],
                    fill=(0, 0, 0, 170))
        d.text(((img.width - tw) / 2, img.height * 0.40 + 8), text, font=fb, fill=col + (255,))
    return np.array(Image.alpha_composite(img, box).convert("RGB"))


def _li(su_kpa):
    import math
    return 1.0 - math.log10(su_kpa / 1.7) / 2.0      # inverse of su = 1.7 kPa * 100^(1 - LI)
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
        from kacha.terrain import ROADS            # the road the run trained on (smooth/rutted)
        kw["road"] = ROADS[cfg.get("g1_road") or "rutted"]
        ws = cfg.get("g1_wet_start")
        if ws is not None:                       # film at the wetness range being trained
            from kacha.mud import SEASONS
            top = cfg.get("g1_wet_max") or SEASONS.get(cfg.get("season"), {}).get("LI", ws)
            kw["mud"] = dict(LI=top, LI_lo=ws)
            if cfg.get("g1_dr"):
                kw["mud"].update(depth_lo=cfg["g1_dr_depth_range"][0],
                                 depth_hi=cfg["g1_dr_depth_range"][1], p_dry=cfg["g1_dr_dry_share"])
        def mk(mud=None):
            k = dict(kw, mud=mud) if mud is not None else kw
            e = G1MudRoadEnv(seed=0, look=False, obs_noise=noise, max_seconds=cfg.get("g1_max_seconds") or 60.0, **k)
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
    writer = imageio.get_writer(tmp, fps=12.5, macro_block_size=8)
    results = []
    clips = G1_CLIPS if (g1 and cfg.get("g1_wet_start") is not None) else None
    if clips:
        seeds = (7, 8, 9, 10)[:len(clips)]
    stride = 4                                    # every 4th control step = 0.08 s: plays in real time at 12.5 fps
    video_steps = int(min(cfg.get("g1_max_seconds") or 60.0, VIDEO_MAX_S) / 0.02)
    run_name = os.path.basename(os.path.normpath(run)).replace("g1-road-", "")
    steps_k = f"{model.num_timesteps / 1000:,.0f}k steps"
    for ci, seed in enumerate(seeds, 1):
        label = clips[ci - 1][0] if clips else ""
        env = mk(clips[ci - 1][1]) if clips else mk()
        obs, _ = env.reset(seed=seed)
        info, frame, done = {}, None, False
        for t in range(video_steps if g1 else MAX_STEPS):
            a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            done = te or tr
            if t % stride == 0 or done:
                frame = raw = env.render()
                if g1:
                    ms = env.mud_summary()
                    cond = ("dry ground, no mud" if ms["su_kpa"] is None else
                            f"mud: wetness LI {_li(ms['su_kpa']):.2f} | strength {ms['su_kpa']:.1f} kPa | depth {ms['depth'] * 100:.0f} cm")
                    frame = overlay(frame, [f"{run_name}  |  {steps_k}",
                                            f"clip {ci}/{len(seeds)}: {label}  (road seed {seed})" if label else f"clip {ci}/{len(seeds)}  road seed {seed}",
                                            cond,
                                            f"time {env.t:5.1f} s | distance {info.get('distance', 0.0):+.2f} m of 8.3 | speed {env.speed:.2f} m/s"])
                writer.append_data(frame)
            if done:
                break
        if not done and not g1:                       # clip ended before the episode did
            walked = info["distance"] >= 0.5 * env.cmd * (t + 1) * env.dt
            info["is_success"] = walked
        ok = bool(info.get("is_success", False))
        results.append(ok)
        if g1:
            why = "REACHED THE END" if ok else ("FELL" if info.get("fell") else
                  "LEFT THE ROAD" if info.get("off_road") else
                  "STILL WALKING" if not done else "OUT OF TIME")
            frame = overlay(raw, [f"{run_name}  |  {steps_k}", f"clip {ci}/{len(seeds)}: {label}  (road seed {seed})" if label else f"clip {ci}/{len(seeds)}  road seed {seed}", cond,
                                    f"distance {info.get('distance', 0.0):+.2f} m | time {env.t:.1f} s"],
                            big=(why, (60, 200, 80) if ok else (230, 70, 60)))
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
