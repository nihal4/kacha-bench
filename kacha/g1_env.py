"""Stage 1 humanoid task: the Unitree G1 crosses a muddy, rutted village road.

By default the policy does not drive the joints from scratch. Each control step it adds
a small correction to the scripted mud-walking controller (kacha/gait.py), which already
has the gait style and a basic balance reflex. This keeps early training from falling
at every step and keeps exploration noise around a gait that roughly works.

Baseline variants (docs/humanoid.md, "Baselines"):
- reference=False: no scripted controller. Leg targets are Unitree's default standing
  angles plus 0.25 rad x action, the convention of unitree_rl_gym, so Unitree's
  pretrained G1 policy runs in this mode too (scripts/g1_evaluate.py).
- height_scan=True: 9 x 5 heights ahead of the robot (water hides pothole depth).
- history=k: the last k proprioceptive observations, as a simple memory.

Reward (task-specific; docs/humanoid.md "Stage 1 reward"): pay for progress toward the
end of the road, a large one-off penalty for falling or leaving the road, a bonus for
arriving, and small terms for staying upright, following the reference style, smooth
actions and not wasting energy. Standing still earns only the small alive term.
"""

import gymnasium as gym
import mujoco
import numpy as np

from .g1_scene import build_g1_scene, foot_geoms
from .gait import GaitParams, ScriptedWalker
from .mud import SEASONS, FlatFootMud, MudParams, breakout_ratio, shear_strength
from .robot_scene import terrain_height
from .terrain import generate_village_road, muddy_rut_road
from .water import SEASON_FILL, Water, WaterForces

W = dict(progress=1.0, alive=0.2, upright=0.3, imitation=0.3, lateral=-0.5,
         action_rate=-0.02, energy=-0.0005, foot_slip=-0.1)
FALL_PENALTY = 20.0       # one-off, on falling or leaving the road
GOAL_BONUS = 20.0         # one-off, on reaching the end of the road
ACTION_SCALE = np.array([0.25, 0.15, 0.10, 0.25, 0.20, 0.10] * 2)   # rad, per leg joint
# reference=False: Unitree's G1 RL convention (unitree_rl_gym, g1 config)
UNITREE_DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
UNITREE_SCALE = 0.25
N_PROPRIO = 50
# Failure-cause thresholds (heuristics; docs/study.md, "Failure causes"), judged over the
# last 0.5 s before a fall
CAUSE_WINDOW = 25            # control steps
CAUSE_SKIP = 5               # ignore the last 0.1 s: the collapse itself slides and sinks feet
DEBOUNCE = 3                 # control steps a foot's contact must hold to count as a change
SANK_M = 0.08                # a loaded foot this deep below the mud surface
STUCK_N = 60.0               # suction pulling a lifting foot down
SLIP_M = 0.04                # a loaded foot sliding this far within the window (brief
                             # touchdown slides under impact are normal and much shorter)
LOADED = 0.15                # share of body weight that makes a foot "loaded"
SCAN_X, SCAN_Y = np.linspace(-0.2, 0.8, 9), np.linspace(-0.3, 0.3, 5)
NOISE = dict(gyro=0.1, gravity=0.03, joint_pos=0.01, joint_vel=0.5, linvel=0.05)
# Balance assist (curriculum; Yu, Turk & Liu 2018): a virtual torque on the pelvis that
# resists tipping, scaled by `assist` from 1 (start of training) to 0. It only resists
# tilt; it does not push the robot forward or hold it up. Evaluation uses assist = 0.
ASSIST_K, ASSIST_D = 400.0, 40.0      # N m per rad of tilt, N m per rad/s


class G1MudRoadEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, season="peak_monsoon", seed=None, speed_range=(0.25, 0.4),
                 max_seconds=40.0, regen_every=20, timestep=0.005, control_dt=0.02,
                 obs_noise=True, look=False, spawn_x=-4.0, goal_x=4.3, road=None,
                 assist=0.0, reference=True, height_scan=False, history=1,
                 mud=None, render_mode=None):
        # mud: overrides for robustness sweeps, applied on top of the season:
        #   LI (sets strength and suction together), su_kpa, depth (m), beta (suction
        #   ratio), mu (surface friction), fill (share of each hole filled with water)
        self.mud_override = dict(mud or {})
        self.assist = float(assist)
        self.reference, self.height_scan = bool(reference), bool(height_scan)
        self.history = max(int(history), 1)
        self.season, self.speed_range = season, speed_range
        self.rng = np.random.default_rng(seed)
        self.max_steps = int(max_seconds / control_dt)
        self.regen_every, self.timestep, self.control_dt = regen_every, timestep, control_dt
        self.obs_noise, self.look = obs_noise, look
        self.spawn_x, self.goal_x = spawn_x, goal_x
        self.road_kw = road or {}
        self.render_mode = render_mode
        self.walker = ScriptedWalker(GaitParams())
        self._resets, self._renderer, self.model = 0, None, None
        self._build()
        n_obs = N_PROPRIO * self.history + (len(SCAN_X) * len(SCAN_Y) if self.height_scan else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

    # ------------------------------------------------------------------ world
    def _pick_season(self):
        if self.season == "mix":
            return str(self.rng.choice(["dry", "early_monsoon", "peak_monsoon"]))
        return self.season

    def _build(self):
        self.cfg = muddy_rut_road(**self.road_kw)
        for _ in range(20):       # keep the first metre free of potholes and puddles
            heights, _, _, info = generate_village_road(self.cfg, seed=int(self.rng.integers(1 << 30)))
            if all(abs(p["center"][0] - self.spawn_x) > p["radius"] + 0.6 or abs(p["center"][1]) > 0.6
                   for p in info["potholes"]):
                break
        self.heights, self.potholes = heights, info["potholes"]
        self.cur_season = self._pick_season()
        self.mud_p, fill = self._mud_params()
        depth = self.mud_p.depth if self.mud_p else 0.0
        self.water = Water(self.potholes, fill=fill)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model, self.home = build_g1_scene(heights, self.cfg.length, self.cfg.width,
                                               self.cfg.road_width, mud_depth=depth,
                                               water=self.water, base_xy=(self.spawn_x, 0.0),
                                               timestep=self.timestep, look=self.look)
        m = self.model
        self.data = mujoco.MjData(m)
        self.surface = lambda x, y: terrain_height(self.heights, self.cfg.length, self.cfg.width, x, y)
        self.mud = FlatFootMud(m, self.mud_p, self.surface, foot_geoms()) if depth > 0 else None
        self.wforces = WaterForces(m, self.water) if self.water.pools else None
        self.sub = int(round(self.control_dt / m.opt.timestep))
        self.pelvis, self.torso = m.body("pelvis").id, m.body("torso_link").id
        self.soles = [[m.geom(g).id for g in gs] for gs in foot_geoms().values()]
        self.sole_set = [set(s) for s in self.soles]
        self.upper = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (self.pelvis, self.torso)
                      and m.geom_contype[g]}
        self.terrain = m.geom("terrain").id
        self.foot_bodies = [int(m.geom_bodyid[s[0]]) for s in self.soles]
        self._v6 = np.zeros(6)

    def _mud_params(self):
        """(MudParams or None, water fill) for the current season and any overrides."""
        o = self.mud_override
        base = SEASONS.get(self.cur_season, {"LI": -0.2, "depth": 0.0})
        LI = o.get("LI", base["LI"])
        depth = o.get("depth", base["depth"])
        if depth <= 0:
            return None, 0.0
        su = o["su_kpa"] * 1e3 if "su_kpa" in o else shear_strength(LI)
        p = MudParams(su=su, depth=depth, beta=o.get("beta", breakout_ratio(LI)))
        if "mu" in o:
            p.mu_surface = o["mu"]
        fill = o.get("fill", SEASON_FILL.get(self.cur_season, 1.0 if LI >= 1.0 else 0.6))
        return p, fill

    def mud_summary(self):
        """The physical conditions of the current road, for logging."""
        p = self.mud_p
        return dict(season=self.cur_season, su_kpa=p.su / 1e3 if p else None,
                    depth=p.depth if p else 0.0, beta=p.beta if p else 0.0,
                    mu=p.mu_surface if p else None, fill=self.water_fill_share())

    def water_fill_share(self):
        holes = [h for h in self.potholes if h["depth"] > 0]
        return len(self.water.pools) / len(holes) if holes else 0.0

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
        u0, _ = self.walker(0.0, 0.0, np.zeros(3), np.zeros(3), np.eye(3), np.zeros((2, 3)),
                            np.array([True, True]))
        if not self.reference:
            u0 = u0.copy()
            u0[:12] = UNITREE_DEFAULT
        d.qpos[7:] = u0 + np.r_[self.rng.uniform(-0.02, 0.02, 12), np.zeros(17)]
        mujoco.mj_forward(m, d)
        low = min(d.geom_xpos[g][2] - 0.005 - self.surface(*d.geom_xpos[g][:2])
                  for s in self.soles for g in s)
        d.qpos[2] -= low
        mujoco.mj_forward(m, d)
        if self.mud:
            self.mud.zp[:] = 0.0
        self.cmd = float(self.rng.uniform(*self.speed_range))
        self.t, self.steps = 0.0, 0
        self.last_action = np.zeros(12)
        self.x0 = float(d.qpos[0])
        self._ref()
        self._hist = None
        self._start_metrics()
        return self._obs(), {}

    def _phase_speed(self):
        """The clock starts after 0.5 s of standing; speed ramps up over 1 s."""
        T = self.walker.gait.g.period
        t = self.t - 0.5
        if t <= 0:
            return 0.0, 0.0
        return (t / T) % 1.0, self.cmd * min(t / 1.0, 1.0)

    def _ref(self):
        d = self.data
        self.phase, self.speed = self._phase_speed()
        R = d.xmat[self.pelvis].reshape(3, 3)
        sc = np.array([d.geom_xpos[s].mean(axis=0) for s in self.soles])
        stance = self.walker.gait.targets(self.phase, self.speed)[3]
        self.u_ref, self.stance_ref = self.walker(self.phase, self.speed, d.subtree_com[1],
                                                  d.qvel[0:3], R, sc, stance)
        if not self.reference:              # legs: no scripted controller
            self.u_ref = self.u_ref.copy()
            self.u_ref[:12] = UNITREE_DEFAULT
        return self.u_ref

    def step(self, action, ctrl_fn=None):
        """ctrl_fn: optional controller called before every physics step as
        ctrl_fn(data, dt) -> actuator targets; it replaces the action (model-based
        baselines such as kacha.dcm run at the physics rate)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        m, d = self.model, self.data
        u = self.u_ref.copy()
        u[:12] += (ACTION_SCALE if self.reference else UNITREE_SCALE) * a
        d.ctrl[:] = np.clip(u, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        suction = np.zeros(2)
        for _ in range(self.sub):
            if self.mud:
                self.mud.apply(m, d)
                suction = np.maximum(suction, self.mud.suction)
            else:
                d.qfrc_applied[:] = 0.0
            if self.wforces:
                self.wforces.apply(m, d)
            if self.assist > 0.0:
                self._apply_assist()
            if ctrl_fn is not None:
                d.ctrl[:] = ctrl_fn(d, m.opt.timestep)
            mujoco.mj_step(m, d)
        self.t += self.control_dt
        self.steps += 1

        R = d.xmat[self.pelvis].reshape(3, 3)
        v = R.T @ d.qvel[0:3]
        g = R.T @ np.array([0.0, 0.0, -1.0])
        x, y = float(d.qpos[0]), float(d.qpos[1])
        ground = self.surface(x, y)
        height = float(d.qpos[2] - ground)
        q = d.qpos[7:19]
        contact = self._sole_contacts()
        slip, slip_speed = 0.0, np.zeros(2)
        for k in np.flatnonzero(contact):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.foot_bodies[k], self._v6, 0)
            slip += float(self._v6[3] ** 2 + self._v6[4] ** 2)
            slip_speed[k] = float(np.hypot(self._v6[3], self._v6[4]))
        fell = bool(g[2] > -0.7 or height < 0.45 or self._upper_hit())     # tilt > ~45 deg
        off_road = abs(y) > self.cfg.road_width / 2 - 0.2
        reached = x >= self.goal_x
        vx_world = float(d.qvel[0])
        moving = self.speed > 0.0
        terms = dict(
            progress=float(np.clip(vx_world, -0.3, self.cmd) / self.cmd) if moving else 0.0,
            alive=1.0,
            upright=float(-g[2]),
            imitation=float(np.exp(-np.sum((q - self.u_ref[:12]) ** 2) / 0.25))
            if self.reference else 0.0,
            lateral=float(y ** 2),
            action_rate=float(np.sum((a - self.last_action) ** 2)),
            energy=float(np.sum(np.abs(d.qvel[6:18] * d.actuator_force[:12]))),
            foot_slip=slip)
        reward = self.control_dt * sum(W[k] * terms[k] for k in W)
        failed = fell or off_road
        if failed:
            reward -= FALL_PENALTY
        if reached and not failed:
            reward += GOAL_BONUS
        self.last_action = a
        self.terms = terms
        terminated = bool(failed or reached)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        info = {"distance": x - self.x0, "season": self.cur_season, "speed_cmd": self.cmd,
                "speed": vx_world, "fell": fell, "off_road": off_road, "assist": self.assist,
                "sinkage": self.mud.sinkage.copy() if self.mud else np.zeros(2)}
        self._track(contact, slip_speed, suction)
        if terminated or truncated:
            info["is_success"] = bool(reached and not failed)
            info["outcome"] = self._outcome(reached, fell, off_road)
            info["fall_mode"] = self._fall_mode(g, height) if fell else None
            info["metrics"] = self.episode_metrics(info["outcome"])
            info["metrics"]["fall_mode"] = info["fall_mode"]
        self._ref()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ study metrics
    def _start_metrics(self):
        self._win = []                        # (max loaded sinkage, max suction, max loaded slip, in water)
        self._weight = float(np.sum(self.model.body_mass)) * 9.81
        self._f6 = np.zeros(6)
        self._last_contact = self._sole_contacts()
        self._held = self._last_contact.copy()   # debounced contact
        self._pending = np.zeros(2, dtype=int)
        self._touch_x = []                    # (foot, x) at each touchdown while walking
        self._swing_peak = np.zeros(2)
        self._lifts = []
        self._n_walk = self._n_double = 0
        self._sink_sum = self._sink_n = 0.0
        self._slip_m = self._energy = 0.0

    def _foot_state(self):
        d = self.data
        sc = np.array([d.geom_xpos[s].mean(axis=0) for s in self.soles])
        bottom = np.array([d.geom_xpos[s][:, 2].min() - 0.005 for s in self.soles])
        surf = np.array([self.surface(*sc[k][:2]) for k in range(2)])
        return sc, bottom - surf

    def _corner_measures(self, contact):
        """Per loaded foot: sinkage, and the speed of the slowest sole corner (a rolling
        foot pivots on a still corner; only a foot whose every corner moves is sliding).
        Sinkage is the mud model's plastic yield under the foot (the floor of the hole it
        has pushed), not the geometric depth of a corner: the support plate is flat, so
        on a rut slope a corner can sit below the sloped surface without any yielding."""
        d, m = self.data, self.model
        sink, slide = np.zeros(2), np.zeros(2)
        for k in range(2):
            if not contact[k]:
                continue
            speeds = []
            for g in self.soles[k]:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, self._v6, 0)
                speeds.append(float(np.hypot(self._v6[3], self._v6[4])))
            sink[k] = float(self.mud.zp[k]) if self.mud else 0.0
            slide[k] = min(speeds)
        return sink, slide

    def _sole_forces(self):
        """Normal force (N) on each foot from the ground, the mud plates and the terrain."""
        d, f = self.data, np.zeros(2)
        for i in range(d.ncon):
            c = d.contact[i]
            for k in range(2):
                if c.geom1 in self.sole_set[k] or c.geom2 in self.sole_set[k]:
                    mujoco.mj_contactForce(self.model, d, i, self._f6)
                    f[k] += self._f6[0]
        return f

    def _track(self, contact, slip_speed, suction):
        d = self.data
        sc, clear = self._foot_state()
        loaded = self._sole_forces() > LOADED * self._weight
        sink, slip_speed = self._corner_measures(loaded)
        in_water = any(self.water.level_at(*sc[k][:2]) is not None for k in range(2))
        self._win.append((float(np.max(sink)), float(np.max(suction)),
                          float(np.max(slip_speed)), in_water))
        self._win = self._win[-CAUSE_WINDOW:]
        self._energy += self.control_dt * float(np.sum(np.abs(d.qvel[6:18] * d.actuator_force[:12])))
        walking = self.t > 1.5                # after the stand and the speed ramp
        if walking:
            self._n_walk += 1
            self._n_double += int(self._held.all())
            self._slip_m += self.control_dt * float(np.sum(slip_speed))
            if loaded.any():
                self._sink_sum += float(np.sum(sink)) / loaded.sum()
                self._sink_n += 1
        changed = np.zeros(2, dtype=bool)
        for k in range(2):                            # debounce: a change must hold 3 steps
            self._pending[k] = self._pending[k] + 1 if contact[k] != self._held[k] else 0
            if self._pending[k] >= DEBOUNCE:
                self._held[k], self._pending[k], changed[k] = contact[k], 0, True
        for k in range(2):
            if not self._held[k]:
                self._swing_peak[k] = max(self._swing_peak[k], clear[k])
            elif changed[k]:                          # touchdown
                if walking:
                    self._touch_x.append((k, float(sc[k][0])))
                    self._lifts.append(self._swing_peak[k])
                self._swing_peak[k] = 0.0
        self._last_contact = contact.copy()

    def _outcome(self, reached, fell, off_road):
        """success, off_road, too_slow, or the most likely cause of a fall. Over the 0.5 s
        before the collapse (the last 0.1 s excluded), the cause is whichever problem
        appeared first: hidden_water (a foot sinking where water hid the hole), stuck
        (suction holding a lifting foot), sank, slipped; no_trigger if none (a balance
        failure). How it fell is recorded separately (fall_mode). Heuristic thresholds
        (docs/study.md)."""
        if reached and not (fell or off_road):
            return "success"
        if off_road and not fell:
            return "off_road"
        if not fell:
            return "too_slow"
        w = np.array(self._win[:-CAUSE_SKIP]) if len(self._win) > CAUSE_SKIP else np.zeros((1, 4))
        flags = {"hidden_water": (w[:, 3] > 0) & (w[:, 0] > 0.5 * SANK_M),
                 "stuck": w[:, 1] > STUCK_N, "sank": w[:, 0] > SANK_M,
                 "slipped": np.cumsum(w[:, 2]) * self.control_dt > SLIP_M}
        first = {k: int(np.argmax(v)) for k, v in flags.items() if v.any()}
        if not first:
            return "no_trigger"               # fell without sinking, suction or sliding
        order = list(flags)                   # ties: the order above
        return min(first, key=lambda k: (first[k], order.index(k)))

    @staticmethod
    def _fall_mode(g, height):
        """tipped: the trunk went over (tilt past ~45 deg, or the trunk hit the ground);
        collapsed: the pelvis dropped below 0.45 m with the trunk still fairly upright
        (legs gave way, splits, or sank)."""
        tilt = np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0)))
        return "collapsed" if height < 0.45 and tilt < 30.0 else "tipped"

    def episode_metrics(self, outcome=None):
        """Gait and effort measures for the episode so far (walking part only)."""
        d = self.data
        dist = float(d.qpos[0] - self.x0)
        steps = [abs(b[1] - a[1]) for a, b in zip(self._touch_x, self._touch_x[1:]) if a[0] != b[0]]
        walk_s = self._n_walk * self.control_dt
        mass = float(np.sum(self.model.body_mass))
        return dict(
            outcome=outcome, distance=dist, seconds=self.t, speed_cmd=self.cmd,
            mean_speed=dist / max(self.t, 1e-6),
            step_length=float(np.mean(steps)) if steps else None,
            cadence_per_min=60.0 * len(self._touch_x) / walk_s if walk_s > 0 else None,
            double_support=self._n_double / self._n_walk if self._n_walk else None,
            # peak height of the sole above the mud surface in swing; 0 = never left the mud
            foot_lift=float(np.mean(self._lifts)) if self._lifts else None,
            stance_sinkage=self._sink_sum / self._sink_n if self._sink_n else 0.0,
            slip_m=self._slip_m,
            cost_of_transport=self._energy / (mass * 9.81 * dist) if dist > 0.2 else None,
            **{f"mud_{k}": v for k, v in self.mud_summary().items()})

    def set_assist(self, value):
        self.assist = float(np.clip(value, 0.0, 1.0))

    def _apply_assist(self):
        """Torque on the pelvis toward upright: k * (z_body x z_world) - d * (w_x, w_y)."""
        d = self.data
        z = d.xmat[self.pelvis].reshape(3, 3)[:, 2]
        tilt = np.cross(z, [0.0, 0.0, 1.0])            # axis and sine of the tilt
        w = d.qvel[3:6]
        R = d.xmat[self.pelvis].reshape(3, 3)
        w_world = R @ w
        tau = self.assist * (ASSIST_K * tilt - ASSIST_D * np.array([w_world[0], w_world[1], 0.0]))
        mujoco.mj_applyFT(self.model, d, np.zeros(3), tau, d.xpos[self.pelvis], self.pelvis,
                          d.qfrc_applied)

    # ------------------------------------------------------------------ helpers
    def _sole_contacts(self):
        d = self.data
        on = np.zeros(2, dtype=bool)
        for i in range(d.ncon):
            c = d.contact[i]
            for k in range(2):
                if c.geom1 in self.sole_set[k] or c.geom2 in self.sole_set[k]:
                    on[k] = True
        return on

    def _upper_hit(self):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self.upper and c.geom2 == self.terrain) or \
               (c.geom2 in self.upper and c.geom1 == self.terrain):
                return True
        return False

    def _scan(self):
        """Heights below the pelvis at 9 x 5 points around and ahead of it, as a scanner
        would see them: the mud surface, and the water surface over flooded holes."""
        d = self.data
        x, y, z = d.qpos[0:3]
        R = d.xmat[self.pelvis].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        out = []
        for px in SCAN_X:
            for py in SCAN_Y:
                wx, wy = x + c * px - s * py, y + s * px + c * py
                out.append(z - self.water.perceived_height(wx, wy, self.surface(wx, wy)))
        return np.array(out)

    def _obs(self):
        p = self._proprio()
        if self._hist is None:
            self._hist = [p] * self.history
        else:
            self._hist = self._hist[1:] + [p]
        parts = self._hist[:]
        if self.height_scan:
            parts.append(self._scan().astype(np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _proprio(self):
        d = self.data
        R = d.xmat[self.pelvis].reshape(3, 3)
        u = (lambda n, s: self.rng.uniform(-s, s, n)) if self.obs_noise else (lambda n, s: 0.0)
        ph = 2 * np.pi * self.phase
        parts = [d.qvel[3:6] + u(3, NOISE["gyro"]),
                 R.T @ np.array([0.0, 0.0, -1.0]) + u(3, NOISE["gravity"]),
                 R.T @ d.qvel[0:3] + u(3, NOISE["linvel"]),
                 [self.speed, np.sin(ph), np.cos(ph)],
                 d.qpos[7:19] - self.u_ref[:12] + u(12, NOISE["joint_pos"]),
                 0.05 * (d.qvel[6:18] + u(12, NOISE["joint_vel"])),
                 self.last_action, self.stance_ref.astype(float)]
        return np.concatenate(parts).astype(np.float32)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 720, 1280)
            self._cam = mujoco.MjvCamera()
            self._cam.distance, self._cam.elevation, self._cam.azimuth = 3.2, -14, 125
        self._cam.lookat[:] = self.data.qpos[:3] + np.array([0, 0, -0.15])
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def env_kwargs_from_config(cfg):
    """G1MudRoadEnv settings stored in a run's config.json (for evaluation and videos)."""
    season = cfg.get("season", "peak_monsoon")
    return dict(season="dry" if season == "none" else season,
                reference=cfg.get("g1_reference", True),
                height_scan=cfg.get("scan", False), history=cfg.get("g1_history", 1))
