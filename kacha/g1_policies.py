"""Controllers for G1 stage-1 evaluation and robustness sweeps.

Scripted     the scripted mud-walking controller alone (zero corrections)
Trained      a PPO run from scripts/train.py (final or best checkpoint)
DCM          the model-based capture-point walking controller (kacha/dcm.py)
Unitree      Unitree's pretrained G1 walking policy (unitree_rl_gym motion.pt), trained
             on rigid ground. A TorchScript file can run code when loaded: only load
             files from a source you trust.

make_method(spec) builds one from a string: "scripted", "run:PATH", "run-best:PATH",
"unitree:PATH", "dcm". run_episode(method, env, seed) plays one episode and returns its
metrics.
"""
import json
import os

import numpy as np

from .g1_env import env_kwargs_from_config


class Scripted:
    name = "scripted controller"
    env_kw = {}

    def reset(self, env):
        pass

    def act(self, env, obs):
        return np.zeros(12)


class Trained:
    def __init__(self, run, best=False):
        from stable_baselines3 import PPO
        import pickle
        cfg = json.load(open(os.path.join(run, "config.json")))
        self.env_kw = env_kwargs_from_config(cfg)
        self.env_kw.pop("season")
        tag = "best_" if best else ""
        self.model = PPO.load(os.path.join(run, f"{tag}model.zip" if best else "model.zip"),
                              device="cpu")
        with open(os.path.join(run, f"{tag}vecnormalize.pkl"), "rb") as f:
            vn = pickle.load(f)
        self.mean, self.var = vn.obs_rms.mean, vn.obs_rms.var
        self.clip, self.eps = vn.clip_obs, vn.epsilon
        self.name = os.path.basename(os.path.normpath(run)) + (" (best)" if best else "")

    def reset(self, env):
        pass

    def act(self, env, obs):
        o = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        a, _ = self.model.predict(o[None].astype(np.float32), deterministic=True)
        return a[0]


class Unitree:
    """unitree_rl_gym G1 policy: 47 inputs (deploy/deploy_mujoco/deploy_mujoco.py), legs
    only, targets = default + 0.25 x action, gait clock with a 0.8 s period. Run in the
    environment's reference=False mode, which uses exactly this action convention."""
    name = "Unitree pretrained (rigid ground)"
    env_kw = {"reference": False}

    def __init__(self, path):
        import torch
        self.torch = torch
        self.policy = torch.jit.load(os.path.expanduser(path), map_location="cpu")
        self.policy.eval()

    def reset(self, env):
        self.action = np.zeros(12, dtype=np.float32)

    @staticmethod
    def _gravity(q):
        w, x, y, z = q
        return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])

    def act(self, env, obs_unused):
        from .g1_env import UNITREE_DEFAULT
        d = env.data
        obs = np.zeros(47, dtype=np.float32)
        phase = (env.t % 0.8) / 0.8
        cmd = np.array([env.speed, 0.0, 0.0])
        obs[0:3] = d.qvel[3:6] * 0.25
        obs[3:6] = self._gravity(d.qpos[3:7])
        obs[6:9] = cmd * np.array([2.0, 2.0, 0.25])
        obs[9:21] = d.qpos[7:19] - UNITREE_DEFAULT
        obs[21:33] = d.qvel[6:18] * 0.05
        obs[33:45] = self.action
        obs[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        with self.torch.no_grad():
            a = self.policy(self.torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
        self.action = a.astype(np.float32)
        return a                 # the environment applies default + 0.25 x action


class DCM:
    """Capture-point walking (kacha/dcm.py) at the physics rate, at its own reliable
    speed (default 0.2 m/s) rather than the episode's command. It starts from its own
    settled walking crouch, and sets its own joint servo gains (motor torque limits
    unchanged)."""
    name = "DCM capture-point controller"
    env_kw = {}

    def __init__(self, speed=0.2, **params):
        from .dcm import DCMParams
        self.speed, self.params = speed, DCMParams(**params)

    def reset(self, env):
        from .dcm import DCMWalker
        self.w = DCMWalker(env.model, self.params, surface=env.surface)
        upper = env.u_ref[12:].copy()
        self.w.settle(env.data, upper)
        self.w.reset(env.data, self.speed, upper)
        self.ctrl_fn = self.w

    def act(self, env, obs):
        return np.zeros(12)


def make_method(spec):
    kind, _, path = spec.partition(":")
    if kind == "scripted":
        return Scripted()
    if kind in ("run", "run-best"):
        return Trained(path, best=kind == "run-best")
    if kind == "unitree":
        return Unitree(path)
    if kind == "dcm":
        return DCM(**({"speed": float(path)} if path else {}))
    raise ValueError(f"unknown method {spec!r} (scripted, dcm[:SPEED], run:PATH, run-best:PATH, unitree:PATH)")


def run_episode(method, env, seed):
    """One episode on road `seed`, no assist; returns env.episode_metrics() plus flags."""
    env.set_assist(0.0)
    obs, _ = env.reset(seed=seed)
    method.reset(env)
    info, done = {}, False
    fn = getattr(method, "ctrl_fn", None)
    while not done:
        obs, _, te, tr, info = env.step(method.act(env, obs), ctrl_fn=fn)
        done = te or tr
    m = dict(info["metrics"])
    m.update(success=bool(info.get("is_success")), fell=bool(info["fell"]),
             off_road=bool(info["off_road"]), road_seed=seed)
    return m
