"""Go1 walking environment on the village road (Gymnasium API).

The Go1 starts at one end of a generated road and must walk forward along it at a
commanded speed. Seasons switch on mud and water-filled potholes.

Controller settings, reward terms and weights follow MuJoCo Playground's Go1 joystick
task (google-deepmind/mujoco_playground, locomotion/go1/joystick.py), which builds on
Rudin et al. (2022), "Learning to walk in minutes using massively parallel deep RL":
Kp 35, Kd 0.5, action scale 0.5, 50 Hz control on a 4 ms physics step, uniform
observation noise, and the reward terms below. Not used here: Playground's foot
clearance / foot height terms and its lateral and yaw commands (we command forward
speed only).
"""

import gymnasium as gym
import mujoco
import numpy as np

from .mud import SEASONS, MudContact, MudParams
from .robot_scene import GO1_FEET, build_robot_scene, terrain_height
from .terrain import RoadConfig, generate_village_road
from .water import SEASON_FILL, Water, WaterForces

ALL_SEASONS = ["none"] + list(SEASONS)
# Playground Go1 joystick reward scales
WEIGHTS = dict(tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-0.5, ang_vel_xy=-0.05,
               orientation=-5.0, dof_pos_limits=-1.0, pose=0.5, termination=-1.0,
               torques=-0.0002, action_rate=-0.01, energy=-0.001, feet_slip=-0.1,
               feet_air_time=0.1)
TRACKING_SIGMA = 0.25
POSE_W = np.tile([1.0, 1.0, 0.1], 4)            # hip, thigh, calf per leg
# Reward versions (docs/training.md, "Reward versions"):
#   "playground" - the original port (kept to reproduce earlier runs)
#   "v2"         - fixes: standing still no longer pays, Playground's torque cost,
#                  soft clip floor with a one-off fall penalty
REWARD_VERSIONS = ("playground", "v2")
V2_SIGMA_FRAC = 0.5      # v2 forward-speed tolerance: sigma_x = (0.5 * cmd)^2
V2_CLIP_LO = -1.0        # v2 floor of the summed reward, per second (times dt per step)
V2_FALL_PENALTY = 3.0    # v2 one-off penalty on a fall, applied after the clip
NOISE = dict(linvel=0.1, gyro=0.2, gravity=0.05, joint_pos=0.03, joint_vel=1.5)
KP, KD = 35.0, 0.5


class Go1WalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, season="none", seed=None, max_steps=1000, cmd_range=(0.3, 0.8),
                 regen_every=20, height_scan=False, frame_skip=5, timestep=0.004,
                 action_scale=0.5, obs_noise=True, render_mode=None, reward="v2"):
        if reward not in REWARD_VERSIONS:
            raise ValueError(f"reward must be one of {REWARD_VERSIONS}, got {reward!r}")
        self.reward_version = reward
        self.obs_noise = obs_noise
        self.timestep = timestep
        self.season = season
        self.rng = np.random.default_rng(seed)
        self.max_steps, self.cmd_range = max_steps, cmd_range
        self.regen_every, self.height_scan = regen_every, height_scan
        self.frame_skip, self.action_scale = frame_skip, action_scale
        self.render_mode = render_mode
        self.cfg = RoadConfig()
        self._resets = 0
        self._renderer = None
        self.model = None
        self._build()
        n_obs = 46 + (45 if height_scan else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

    # ------------------------------------------------------------------ world
    def _pick_season(self):
        if self.season == "mix":
            return str(self.rng.choice(["dry", "early_monsoon", "peak_monsoon"]))
        return self.season

    def _build(self):
        for _ in range(20):                       # a road whose start is free of potholes
            heights, _, _, info = generate_village_road(self.cfg, seed=int(self.rng.integers(1 << 30)))
            if all(p["center"][0] - p["radius"] > -4.4 + 0.5 for p in info["potholes"]):
                break
        self.heights, self.potholes = heights, info["potholes"]
        self.cur_season = self._pick_season()
        wet = self.cur_season in SEASONS
        self.mud_p = MudParams.from_season(self.cur_season) if wet else None
        depth = self.mud_p.depth if wet else 0.0
        self.water = Water(self.potholes, fill=SEASON_FILL[self.cur_season] if wet else 0.0)
        self.spawn = np.array([-4.4, 0.0])
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model, self.home = build_robot_scene(heights, self.cfg.length, self.cfg.width,
                                                  mud_depth=depth, water=self.water,
                                                  base_xy=self.spawn, feet_trunk_only=True,
                                                  timestep=self.timestep)
        self.data = mujoco.MjData(self.model)
        surf = lambda x, y: terrain_height(self.heights, self.cfg.length, self.cfg.width, x, y)
        self.surface = surf
        self.mud = MudContact(self.model, self.mud_p, surf, GO1_FEET) if depth > 0 else None
        self.wforces = WaterForces(self.model, self.water) if self.water.pools else None
        m = self.model
        self.trunk = m.body("trunk").id
        self.feet = [m.geom(n).id for n in GO1_FEET]
        self.terrain = m.geom("terrain").id
        self.trunk_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.trunk}
        self.plates = {m.geom(f"plate_{n}").id for n in GO1_FEET} if self.mud else set()
        self.dt = m.opt.timestep * self.frame_skip
        m.actuator_gainprm[:, 0] = KP                 # position servos: Kp 35, Kd 0.5
        m.actuator_biasprm[:, 1] = -KP
        m.actuator_biasprm[:, 2] = -KD
        lo, hi = m.jnt_range[1:13, 0], m.jnt_range[1:13, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.95
        self.soft_lo, self.soft_hi = mid - half, mid + half

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._build()
        elif self._resets and self._resets % self.regen_every == 0:
            self._build()
        self._resets += 1
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.home
        d.qpos[7:] += self.rng.uniform(-0.05, 0.05, 12)
        mujoco.mj_forward(m, d)
        if self.mud:
            self.mud.zp[:] = 0.0
        self.cmd = float(self.rng.uniform(*self.cmd_range))
        self.last_action = np.zeros(12)
        self._v6 = np.zeros(6)
        self.air_time = np.zeros(4)
        self.steps = 0
        self.x0 = float(d.qpos[0])
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        m, d = self.model, self.data
        d.ctrl[:] = self.home[7:] + self.action_scale * a
        if self.mud is None and self.wforces is None:
            mujoco.mj_step(m, d, nstep=self.frame_skip)
        else:
            for _ in range(self.frame_skip):
                if self.mud:
                    self.mud.apply(m, d)
                else:
                    d.qfrc_applied[:] = 0.0
                if self.wforces:
                    self.wforces.apply(m, d)
                mujoco.mj_step(m, d)
        self.steps += 1

        R = d.xmat[self.trunk].reshape(3, 3)
        v = R.T @ d.qvel[0:3]                     # body-frame linear velocity
        w = d.qvel[3:6]                           # body-frame angular velocity
        g = R.T @ np.array([0.0, 0.0, -1.0])      # gravity seen from the body

        contact = self._foot_contacts()
        first = (self.air_time > 0) & contact
        self.air_time += self.dt
        air_rew = float(np.sum((self.air_time - 0.1) * first))
        self.air_time[contact] = 0.0
        slip = 0.0
        for k in np.flatnonzero(contact):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, self.feet[k], self._v6, 0)
            slip += float(self._v6[3] ** 2 + self._v6[4] ** 2)

        q, qd, tau = d.qpos[7:19], d.qvel[6:18], d.actuator_force
        fell = g[2] > 0.0 or self._trunk_hit()      # Playground: up-vector below horizontal
        terms = self._terms(v, w, g, q, qd, tau, a, fell, slip, air_rew)
        reward = self._total(terms, fell)
        self.last_terms = terms
        self.last_action = a
        reached = d.qpos[0] > self.cfg.length / 2 - 0.5
        terminated = bool(fell)
        truncated = bool(not fell and (reached or self.steps >= self.max_steps))
        info = {"distance": float(d.qpos[0] - self.x0), "season": self.cur_season,
                "speed_cmd": self.cmd, "speed": float(v[0])}
        if terminated or truncated:
            # success = did not fall AND walked: reached the road end, or covered at least
            # half the commanded distance (standing still for the whole episode is not success)
            walked = info["distance"] >= 0.5 * self.cmd * self.steps * self.dt
            info["is_success"] = bool(not fell and (reached or walked))
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ reward
    def _terms(self, v, w, g, q, qd, tau, a, fell, slip, air_rew):
        """Unweighted reward terms. See docs/training.md for both versions."""
        pose = np.exp(-np.sum((q - self.home[7:]) ** 2 * POSE_W))
        yaw = np.exp(-(w[2] ** 2) / TRACKING_SIGMA)
        if self.reward_version == "playground":
            track = np.exp(-((v[0] - self.cmd) ** 2 + v[1] ** 2) / TRACKING_SIGMA)
            torques = float(np.sum(tau ** 2))
        else:
            # Fix 1: with the fixed sigma 0.25, standing still (v = 0) earned exp(-cmd^2/0.25),
            # 0.70 of the tracking reward at cmd 0.3 m/s, and pose + yaw paid up to 1.0/s more.
            # Forward error is now relative to the command: v = 0 earns exp(-4) = 0.02, and
            # pose / yaw are paid in proportion to forward progress.
            sigma_x = max((V2_SIGMA_FRAC * self.cmd) ** 2, 1e-4)
            track = np.exp(-(v[0] - self.cmd) ** 2 / sigma_x - v[1] ** 2 / TRACKING_SIGMA)
            progress = float(np.clip(v[0] / self.cmd, 0.0, 1.0))
            pose, yaw = pose * progress, yaw * progress
            # Fix 2: Playground's torque cost is ||tau||_2 + ||tau||_1, not sum(tau^2)
            # (sum(tau^2) was ~8x larger under exploration noise).
            torques = float(np.sqrt(np.sum(tau ** 2)) + np.sum(np.abs(tau)))
        return dict(
            tracking_lin_vel=float(track), tracking_ang_vel=float(yaw),
            lin_vel_z=v[2] ** 2, ang_vel_xy=w[0] ** 2 + w[1] ** 2,
            orientation=g[0] ** 2 + g[1] ** 2,
            dof_pos_limits=float(np.sum(np.clip(self.soft_lo - q, 0, None)
                                        + np.clip(q - self.soft_hi, 0, None))),
            pose=float(pose), termination=float(fell), torques=torques,
            action_rate=float(np.sum((a - self.last_action) ** 2)),
            energy=float(np.sum(np.abs(qd * tau))),
            feet_slip=slip, feet_air_time=air_rew)

    def _total(self, terms, fell):
        raw = self.dt * sum(WEIGHTS[k] * terms[k] for k in WEIGHTS)
        if self.reward_version == "playground":
            # Playground (and legged_gym "only positive rewards"): clip at zero
            return float(np.clip(raw, 0.0, 1e4))
        # Fix 3: a hard zero floor hid every penalty in mud (60% of noisy steps scored
        # exactly 0). Under PPO's initial exploration noise the summed terms have a median
        # of -0.44/s on the rigid road and -1.3/s in peak-monsoon mud, so a floor at -1/s
        # keeps most slip / tilt / sinking signal visible. The fall penalty sits outside
        # the clip and must beat the most a policy could save by falling to escape the
        # floor forever: 1.0/s * dt / (1 - gamma) = 0.02 / 0.01 = 2.0, so 3.0.
        r = float(np.clip(raw, V2_CLIP_LO * self.dt, 1e4))
        return r - V2_FALL_PENALTY * float(fell)

    # ------------------------------------------------------------------ helpers
    def _foot_contacts(self):
        d = self.data
        on = np.zeros(4, dtype=bool)
        for i in range(d.ncon):
            c = d.contact[i]
            for k, f in enumerate(self.feet):
                if c.geom1 == f or c.geom2 == f:
                    on[k] = True
        return on

    def _trunk_hit(self):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self.trunk_geoms and c.geom2 == self.terrain) or \
               (c.geom2 in self.trunk_geoms and c.geom1 == self.terrain):
                return True
        return False

    def _scan(self):
        """Heights below the trunk at 9 x 5 points ahead of it, as a scanner would see
        them: water hides pothole depth (Water.perceived_height)."""
        d = self.data
        x, y, z = d.qpos[0:3]
        yaw = np.arctan2(d.xmat[self.trunk][3], d.xmat[self.trunk][0])
        c, s = np.cos(yaw), np.sin(yaw)
        out = []
        for px in np.linspace(-0.2, 0.6, 9):
            for py in np.linspace(-0.2, 0.2, 5):
                wx, wy = x + c * px - s * py, y + s * px + c * py
                ground = self.surface(wx, wy)
                out.append(z - self.water.perceived_height(wx, wy, ground))
        return np.array(out)

    def _obs(self):
        d = self.data
        R = d.xmat[self.trunk].reshape(3, 3)
        u = (lambda n, s: self.rng.uniform(-s, s, n)) if self.obs_noise else (lambda n, s: 0.0)
        parts = [R.T @ d.qvel[0:3] + u(3, NOISE["linvel"]), d.qvel[3:6] + u(3, NOISE["gyro"]),
                 R.T @ np.array([0.0, 0.0, -1.0]) + u(3, NOISE["gravity"]), [self.cmd],
                 d.qpos[7:19] - self.home[7:] + u(12, NOISE["joint_pos"]),
                 0.05 * (d.qvel[6:18] + u(12, NOISE["joint_vel"])), self.last_action]
        if self.height_scan:
            parts.append(self._scan())
        return np.concatenate(parts).astype(np.float32)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._cam = mujoco.MjvCamera()
            self._cam.distance, self._cam.elevation, self._cam.azimuth = 2.0, -18, 120
        self._cam.lookat[:] = self.data.qpos[:3]
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
