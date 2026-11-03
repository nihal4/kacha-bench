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
                    help="reward version (docs/training.md); 'playground' reproduces older runs")
    args = ap.parse_args()
    cfg = dict(PROFILES[args.profile])
    for k in ("total_steps", "n_envs"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)
    tag = "" if args.reward == "playground" else f"-{args.reward}"
    args.name = args.name or f"go1-{args.season}{'-scan' if args.scan else ''}{tag}-s{args.seed}"
    return args, cfg


def make_env(season, scan, seed, reward="v2"):
    def _f():
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
        args.reward = json.load(open(cfg_file)).get("reward", "playground")
        print(f"[train] resuming with reward version {args.reward}", flush=True)
    config = {"env": "go1_walk", "season": args.season, "scan": args.scan, "seed": args.seed,
              "profile": args.profile, "reward": args.reward, **cfg}
    if not (args.resume and os.path.exists(cfg_file)):
        json.dump(config, open(cfg_file, "w"), indent=2)

    Vec = SubprocVecEnv if cfg["n_envs"] > 1 else DummyVecEnv
    venv = Vec([make_env(args.season, args.scan, args.seed * 1000 + i, args.reward) for i in range(cfg["n_envs"])])
    venv = VecMonitor(venv, info_keywords=("is_success", "distance"))
    eval_env = VecMonitor(DummyVecEnv([make_env(args.season, args.scan, 10_000 + args.seed, args.reward)]),
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
            print(line, flush=True)
            return True

    safety = SafetyCallback(save, max_minutes=cfg["max_minutes"])
    every = 50_000 if cfg["total_steps"] > 100_000 else cfg["n_steps"] * cfg["n_envs"]
    callbacks = CallbackList([SyncNorm(), eval_cb,
                              EveryNTimesteps(n_steps=every, callback=Save()),
                              EveryNTimesteps(n_steps=every, callback=Progress()), safety])

    remaining = max(cfg["total_steps"] - model.num_timesteps, 0)
    print(f"[train] {args.name}: season {args.season}, height scan {args.scan}, "
          f"reward {args.reward}, {cfg['n_envs']} envs, training for {remaining:,} steps -> {out}", flush=True)
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
