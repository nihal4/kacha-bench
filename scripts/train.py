"""Train a Go1 walking policy on the village road with PPO (Stable-Baselines3, CPU).

Examples:
    python scripts/train.py --profile smoke                          # 1-2 min pipeline check
    python scripts/train.py --season none --profile desktop          # rigid road
    python scripts/train.py --season peak_monsoon --scan --profile desktop
    python scripts/train.py --season none --profile desktop --resume # continue a run

Outputs in runs/<name>/: model.zip (+.bak), vecnormalize.pkl (+.bak), best_model.zip,
best_vecnormalize.pkl, evaluations.npz, progress.csv, config.json.
Log lines "[progress] N steps | training episodes: return R | solved S%" are read by the
dashboard ("solved" = survived the episode without falling).
"""
import argparse
import json
import os
import sys

PROFILES = {
    # Always-on Linux PC (i3-8100: 4 cores, 12 GB): one environment per core.
    "desktop": dict(n_envs=4, threads=1, total_steps=20_000_000, n_steps=1024, batch_size=1024,
                    n_epochs=5, eval_every=100_000, eval_episodes=5, max_minutes=0, nice=10),
    # Quick end-to-end check (~1-2 minutes).
    "smoke": dict(n_envs=2, threads=1, total_steps=8192, n_steps=512, batch_size=512,
                  n_epochs=2, eval_every=4096, eval_episodes=2, max_minutes=10, nice=10),
}
SEASON_CHOICES = ["none", "dry", "early_monsoon", "peak_monsoon", "extreme", "mix"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="none", choices=SEASON_CHOICES)
    ap.add_argument("--scan", action="store_true", help="add a 9x5 height scan (water hides pothole depth)")
    ap.add_argument("--profile", default="desktop", choices=list(PROFILES))
    ap.add_argument("--total_steps", type=int)
    ap.add_argument("--n_envs", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reward", default="v2", choices=["v2", "playground"],
                    help="Go1 reward version (docs/training.md); 'playground' reproduces older runs")
    ap.add_argument("--robot", default="go1", choices=["go1", "g1"],
                    help="go1 (quadruped walking) or g1 (humanoid, stage 1 muddy road)")
    ap.add_argument("--assist_frac", type=float, default=0.4,
                    help="g1: share of training over which the balance assist fades from 1 to 0 "
                         "(0 = no assist)")
    ap.add_argument("--assist_mode", default="adaptive", choices=["adaptive", "linear"],
                    help="g1: adaptive = lower the assist only when training episodes cover "
                         "enough distance (with a deadline at assist_frac); linear = fixed fade")
    ap.add_argument("--no_reference", action="store_true",
                    help="g1 baseline: no scripted controller, joints from Unitree's default pose")
    ap.add_argument("--history", type=int, default=1,
                    help="g1: number of past observations in the input (simple memory)")
    args = ap.parse_args()
    cfg = dict(PROFILES[args.profile])
    for k in ("total_steps", "n_envs"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)
    if args.robot == "g1":
        tags = ("-noref" if args.no_reference else "") + ("-noassist" if args.assist_frac <= 0 else "") \
            + ("-scan" if args.scan else "") + (f"-h{args.history}" if args.history > 1 else "")
        args.name = args.name or f"g1-road-{args.season}{tags}-s{args.seed}"
    else:
        tag = "" if args.reward == "playground" else f"-{args.reward}"
        args.name = args.name or f"go1-{args.season}{'-scan' if args.scan else ''}{tag}-s{args.seed}"
    return args, cfg


def make_env(season, scan, seed, reward="v2", robot="go1", assist=0.0, reference=True,
             history=1):
    def _f():
        if robot == "g1":
            from kacha.g1_env import G1MudRoadEnv
            return G1MudRoadEnv(season="dry" if season == "none" else season, seed=seed,
                                assist=assist, reference=reference, height_scan=scan,
                                history=history)
        from kacha.walk_env import Go1WalkEnv
        return Go1WalkEnv(season=season, seed=seed, height_scan=scan, reward=reward)
    return _f


def main():
    args, cfg = parse()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kacha.safety import SafetyCallback, limit_threads
    limit_threads(cfg["threads"])
    try:
        os.nice(cfg["nice"])
    except OSError:
        pass

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, EvalCallback,
                                                    EveryNTimesteps)
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecMonitor,
                                                  VecNormalize)

    out = os.path.join(args.runs_dir, args.name)
    os.makedirs(out, exist_ok=True)
    model_path, vn_path = os.path.join(out, "model.zip"), os.path.join(out, "vecnormalize.pkl")
    cfg_file = os.path.join(out, "config.json")
    if args.resume and os.path.exists(cfg_file):
        # a resumed run keeps the reward it was started with (runs before v2 have no
        # "reward" key and used the Playground port)
        old = json.load(open(cfg_file))
        args.reward = old.get("reward", "playground")
        args.robot = old.get("robot", "go1")
        args.no_reference = not old.get("g1_reference", True)
        args.history = old.get("g1_history", 1)
        args.assist_frac = old.get("assist_frac", args.assist_frac)
        args.assist_mode = old.get("assist_mode", "linear")
        print(f"[train] resuming: robot {args.robot}, reward version {args.reward}", flush=True)
    g1 = args.robot == "g1"
    config = {"env": "g1_mud_road" if g1 else "go1_walk", "robot": args.robot,
              "season": args.season, "scan": args.scan, "seed": args.seed,
              "profile": args.profile, "reward": "g1_stage1" if g1 else args.reward,
              "assist_frac": args.assist_frac if g1 else 0.0,
              "assist_mode": args.assist_mode if g1 else "",
              "g1_reference": not args.no_reference, "g1_history": args.history, **cfg}
    if not (args.resume and os.path.exists(cfg_file)):
        json.dump(config, open(cfg_file, "w"), indent=2)

    Vec = SubprocVecEnv if cfg["n_envs"] > 1 else DummyVecEnv
    g1_kw = dict(reference=not args.no_reference, history=args.history)
    use_assist = g1 and args.assist_frac > 0
    venv = Vec([make_env(args.season, args.scan, args.seed * 1000 + i, args.reward, args.robot,
                         assist=1.0 if use_assist else 0.0, **g1_kw) for i in range(cfg["n_envs"])])
    venv = VecMonitor(venv, info_keywords=("is_success", "distance"))
    # evaluation never uses the balance assist
    eval_env = VecMonitor(DummyVecEnv([make_env(args.season, args.scan, 10_000 + args.seed,
                                                args.reward, args.robot, assist=0.0, **g1_kw)]),
                          info_keywords=("is_success", "distance"))

    resumed = False
    if args.resume and os.path.exists(model_path):
        for suffix in ("", ".bak"):
            try:
                norm = VecNormalize.load(vn_path + suffix, venv)
                model = PPO.load(model_path + suffix, env=norm, device="cpu")
                resumed = True
                print(f"[train] resumed from {model_path + suffix} at {model.num_timesteps:,} steps", flush=True)
                break
            except Exception as e:
                print(f"[train] checkpoint {model_path + suffix} unreadable ({e})", flush=True)
    if not resumed:
        norm = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if g1:
            # The policy corrects a scripted controller, so exploration is kept small
            # (std 0.3 on actions scaled to 0.1-0.25 rad) and every update is limited by a
            # KL target; the learning rate decays to avoid the late collapse seen in the
            # 50M-step Go1 run (docs/humanoid.md, "Training").
            lr = lambda f: 1e-5 + (3e-4 - 1e-5) * f
            model = PPO("MlpPolicy", norm, learning_rate=lr, n_steps=cfg["n_steps"],
                        batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"], gamma=0.99,
                        gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0,
                        target_kl=0.02,
                        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                                           activation_fn=torch.nn.ELU,
                                           # without the scripted controller there is no
                                           # gait to explore around: usual noise (std 0.6)
                                           log_std_init=-0.5 if args.no_reference else -1.2),
                        seed=args.seed, device="cpu", verbose=1)
        else:
            model = PPO("MlpPolicy", norm, learning_rate=3e-4, n_steps=cfg["n_steps"],
                        batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"], gamma=0.99,
                        gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0,
                        policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                                           activation_fn=torch.nn.ELU, log_std_init=-0.5),
                        seed=args.seed, device="cpu", verbose=1)
    model.set_logger(configure(out, ["stdout", "csv"]))

    eval_norm = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    def save():
        """Power-cut safe: write temporary files, keep the previous checkpoint as .bak,
        then swap the new files in (renames are atomic)."""
        import shutil

        def flush(path):
            with open(path, "rb+") as fh:
                os.fsync(fh.fileno())
        tmp_m, tmp_v = os.path.join(out, "model_tmp.zip"), vn_path + ".tmp"
        model.save(tmp_m)
        model.get_vec_normalize_env().save(tmp_v)
        flush(tmp_m)
        flush(tmp_v)
        for f in (model_path, vn_path):
            if os.path.exists(f) and os.path.getsize(f) > 0:
                shutil.copy2(f, f + ".bak")
                flush(f + ".bak")
        os.replace(tmp_m, model_path)
        os.replace(tmp_v, vn_path)
        fd = os.open(out, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)

    class SyncNorm(BaseCallback):
        """Evaluate with the current input normalisation; save it with every new best model."""
        def _on_step(self):
            src = self.model.get_vec_normalize_env()
            eval_norm.obs_rms = src.obs_rms.copy()
            return True

    class SaveBestNorm(BaseCallback):
        def _on_step(self):
            self.model.get_vec_normalize_env().save(os.path.join(out, "best_vecnormalize.pkl"))
            return True

    eval_cb = EvalCallback(eval_norm, n_eval_episodes=cfg["eval_episodes"],
                           eval_freq=max(cfg["eval_every"] // cfg["n_envs"], 1), log_path=out,
                           best_model_save_path=out, deterministic=True,
                           callback_on_new_best=SaveBestNorm(), verbose=1)
    ev_file = os.path.join(out, "evaluations.npz")
    if resumed and os.path.exists(ev_file):            # keep the earlier evaluation history
        old = np.load(ev_file)
        keep = old["timesteps"] <= model.num_timesteps
        eval_cb.evaluations_timesteps = list(old["timesteps"][keep])
        eval_cb.evaluations_results = [list(r) for r in old["results"][keep]]
        eval_cb.evaluations_length = [list(r) for r in old["ep_lengths"][keep]]
        if "successes" in old.files:
            eval_cb.evaluations_successes = [list(r) for r in old["successes"][keep]]
        if len(eval_cb.evaluations_results):
            eval_cb.best_mean_reward = float(np.max(np.mean(old["results"][keep], axis=1)))

    class Save(BaseCallback):
        def _on_step(self):
            save()
            return True

    ASSIST_STEP, ASSIST_GOAL_M, ASSIST_EVERY = 0.05, 3.0, 50_000

    class Assist(BaseCallback):
        """g1 balance assist, from 1 down to 0.
        linear:   fade over the first assist_frac of training.
        adaptive: every 50k steps, lower it by 0.05 if recent training episodes covered at
                  least 3 m on average (as in Yu, Turk & Liu 2018, assistance is reduced as
                  the policy copes); after the assist_frac deadline it fades linearly to 0
                  over a further 10% of training, so it can never stay on for good."""
        STATE = os.path.join(out, "assist_state.json")

        def __init__(self):
            super().__init__()
            self.level, self.last_check, self.sent = 1.0, 0, None
            if resumed and os.path.exists(self.STATE):       # keep the level after --resume
                st = json.load(open(self.STATE))
                self.level, self.last_check = float(st["level"]), int(st["last_check"])
                print(f"[train] assist resumed at {self.level:.2f}", flush=True)

        def _deadline_cap(self, t):
            T = cfg["total_steps"]
            start = args.assist_frac * T
            return 1.0 if t <= start else max(0.0, 1.0 - (t - start) / (0.1 * T))

        def _on_step(self):
            t = self.model.num_timesteps
            if args.assist_mode == "linear":
                self.level = float(max(0.0, 1.0 - t / max(args.assist_frac * cfg["total_steps"], 1)))
            elif t - self.last_check >= ASSIST_EVERY:
                self.last_check = t
                dist = [e["distance"] for e in self.model.ep_info_buffer if "distance" in e]
                if dist and np.mean(dist) >= ASSIST_GOAL_M:
                    self.level = max(0.0, self.level - ASSIST_STEP)
            a = min(self.level, self._deadline_cap(t)) if args.assist_mode == "adaptive" else self.level
            if self.sent is None or abs(a - self.sent) >= 0.01 or (a == 0.0 and self.sent != 0.0):
                self.model.get_env().env_method("set_assist", a)
                self.sent = a
                tmp = self.STATE + ".tmp"
                json.dump({"level": self.level, "last_check": self.last_check, "applied": a,
                           "steps": t}, open(tmp, "w"))
                os.replace(tmp, self.STATE)
            assist_now[0] = a
            return True

    assist_now = [1.0 if use_assist else 0.0]

    class Progress(BaseCallback):
        def _on_step(self):
            m = self.model
            rews = [e["r"] for e in m.ep_info_buffer]
            succ = list(m.ep_success_buffer)
            dist = [e["distance"] for e in m.ep_info_buffer if "distance" in e]
            line = f"[progress] {m.num_timesteps:,} steps"
            if rews:
                line += f" | training episodes: return {np.mean(rews):.2f}"
            if succ:
                line += f" | solved {100 * np.mean(succ):.0f}%"
            if dist:
                line += f" | distance {np.mean(dist):.2f} m"
            if use_assist:
                line += f" | assist {assist_now[0]:.2f}"
            print(line, flush=True)
            return True

    safety = SafetyCallback(save, max_minutes=cfg["max_minutes"])
    every = 50_000 if cfg["total_steps"] > 100_000 else cfg["n_steps"] * cfg["n_envs"]
    callbacks = CallbackList([SyncNorm(), eval_cb] + ([Assist()] if use_assist else []) + [
                              EveryNTimesteps(n_steps=every, callback=Save()),
                              EveryNTimesteps(n_steps=every, callback=Progress()), safety])

    remaining = max(cfg["total_steps"] - model.num_timesteps, 0)
    print(f"[train] {args.name}: robot {args.robot}, season {args.season}, height scan {args.scan}, "
          f"reward {config['reward']}, {cfg['n_envs']} envs, training for {remaining:,} steps -> {out}", flush=True)
    try:
        model.learn(remaining, callback=callbacks, reset_num_timesteps=not resumed, progress_bar=False)
    except KeyboardInterrupt:
        print("\n[train] interrupted, saving...", flush=True)
    save()
    venv.close()
    eval_env.close()
    why = safety.stop_reason or "finished"
    print(f"[train] {why}. Saved to {out} at {model.num_timesteps:,} steps. Use --resume to continue.",
          flush=True)


if __name__ == "__main__":
    main()
