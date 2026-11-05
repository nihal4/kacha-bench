"""Model-based walking controller for the Unitree G1: capture-point (DCM) walking.

The classical alternative to learned walking, used as a baseline in the study.

- Footsteps: a fixed plan at the commanded speed, re-planned every control step by
  capture-point step adjustment (Khadiv et al. 2016): the swinging foot is sent to where
  the divergent component of motion (DCM, Englsberger et al. 2015) will be at
  touchdown, minus the nominal offset, so a push or a slip is caught by stepping.
- Balance within a step: DCM tracking with the zero-moment point (ZMP, Kajita et al.
  2003) commanded inside the support foot, r = p + K (xi - xi_ref), clipped to the sole.
- Forces: the ground force that realises that ZMP (m (g + c_ddot), acting at r) is split
  between the feet in contact, and turned into joint torques with the full rigid-body
  model, tau = bias(q, q_dot) - J_c^T lambda (gravity, Coriolis and contact forces).
- Kinematics: numerical inverse kinematics puts the pelvis at the walking height and
  upright, the stance foot where it is, and the swing foot on its trajectory.
- Output: the G1's joints are position servos, so the torque is sent as a target
  offset, u = q_des + tau / kp. The servo position limits are lifted for this so that
  a target can carry torque; the motor torque limits (forcerange) still apply.

Everything uses the true simulator state (positions, velocities, contacts): it is an
idealised, fully informed controller, which suits a baseline.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

SIDES = ("left", "right")
SOLE_LOCAL = np.array([0.035, 0.0, -0.03])    # sole centre in the ankle-roll frame (m)
SOLE_HALF = np.array([0.09, 0.033])           # half length, half width of the sole (m)
# Footsteps and the ZMP reference are planned at the point under the ankle, where the
# G1's weight naturally sits; the support area is the whole sole, 3.5 cm further forward.
ANCHOR_FWD = -0.035                           # anchor relative to the sole centre, along the foot (m)
SOLE_BOTTOM = 0.005                           # sole centre to the bottom of the sole (m)
G = 9.81


@dataclass
class DCMParams:
    t_init: float = 0.8           # s of double support before the first step
    width: float = 0.27           # m between the feet
    pelvis_height: float = 0.72   # m above the stance sole
    swing_height: float = 0.10    # m peak foot lift
    t_ss: float = 0.50            # s of single support per step
    t_ds: float = 0.20            # s of double support per step
    k_dcm: float = 5.0            # DCM feedback: r = r_ref + (1 + k/omega) (xi - xi_ref)
    k_rot: float = 60.0           # N m/rad, heading correction (yaw moment)
    d_rot: float = 6.0            # N m s/rad
    kz: float = 60.0              # 1/s^2, pelvis-height force feedback
    dz: float = 12.0              # 1/s
    max_step_x: float = 0.40      # m, step adjustment limits relative to the stance foot
    min_step_x: float = -0.20
    min_width: float = 0.16
    max_width: float = 0.42
    ik_iters: int = 3
    # joint servo gains used by this controller (kp N m/rad, kd N m s/rad). Classical
    # position-controlled walkers track their joint plan stiffly; the motor torque
    # limits are unchanged. None keeps the scene's gains (Unitree's RL gains).
    kp_leg: float = 250.0
    kd_leg: float = 6.0
    kp_ankle: float = 60.0
    kd_ankle: float = 2.0
    k_swing: float = 0.6          # swing-foot position feedback (share of the error added)
    k_trunk: float = 300.0        # N m/rad, trunk pitch and roll through the stance hips
    d_trunk: float = 20.0         # N m s/rad
    k_yaw: float = -100.0          # N m/rad, heading through the stance hip yaw joints (the
    d_yaw: float = -7.0           # sign was found in simulation: the positive sign spun the robot)
    k_speed: float = 0.0          # (m/s) of planned speed per m of DCM lag at touchdown (off:
    relax_speed: float = 0.0      # changing the step length mid-walk made the plan jump)
    stance_rot: str = "ground"    # planted foot's target tilt: "ground" slope, "flat", "actual"
    torsion: float = 0.0          # m, twisting moment asked of a foot per N of load (0: asking for
                                  # any made the stance foot spin loose; heading is held at the hips)
    foothold_radius: float = 0.10 # m, terrain-aware footstep search around the target (0 = off)
    arm_swing: float = 0.0        # rad of shoulder pitch per m of foot separation (counter-swing;
                                  # off: it did not help in the parameter search)


def _quat_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rot_err(R_cur, R_des):
    """Rotation vector taking R_cur to R_des (world frame)."""
    R = R_des @ R_cur.T
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return axis * ang


def _convex_hull(P):
    """Counter-clockwise convex hull of 2-D points (monotone chain)."""
    P = sorted(map(tuple, P))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def _inside(r, hull):
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if (b[0] - a[0]) * (r[1] - a[1]) - (b[1] - a[1]) * (r[0] - a[0]) < 0:
            return False
    return True


def _smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class DCMWalker:
    def __init__(self, model, params=None, surface=None, surface_vec=None):
        self.m, self.p = model, params or DCMParams()
        self.surface = surface or (lambda x, y: 0.0)
        self.surface_vec = surface_vec or (lambda x, y: np.vectorize(self.surface)(x, y))
        self._calls, self._rot_cache = 0, {}
        self.scratch = mujoco.MjData(model)
        self.mass = float(np.sum(model.body_mass))
        self.feet = [model.body(f"{s}_ankle_roll_link").id for s in SIDES]
        self.sole_geoms = [{g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE}
                           for b in self.feet]
        self.pelvis = model.body("pelvis").id
        self.leg_dofs = [np.arange(6, 12), np.arange(12, 18)]
        if self.p.kp_leg is not None:
            for i in range(12):
                ankle = i % 6 >= 4
                kp = self.p.kp_ankle if ankle else self.p.kp_leg
                kd = self.p.kd_ankle if ankle else self.p.kd_leg
                model.actuator_gainprm[i, 0] = kp
                model.actuator_biasprm[i, 1] = -kp
                model.actuator_biasprm[i, 2] = -kd
        self.kp = model.actuator_gainprm[:, 0].copy()
        self.kd = -model.actuator_biasprm[:, 2].copy()
        self.tau_max = model.actuator_forcerange[:, 1].copy()
        model.actuator_ctrllimited[:] = 0            # targets may carry torque (see module doc)
        self._jp, self._jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self._jcom = np.zeros((3, model.nv))
        self.upper_pose = None

    # ------------------------------------------------------------------ frames
    def sole(self, d, k):
        R = d.xmat[self.feet[k]].reshape(3, 3)
        return d.xpos[self.feet[k]] + R @ SOLE_LOCAL, R

    def anchor(self, d, k):
        """Planning point of foot k: under the ankle, on the sole."""
        pos, R = self.sole(d, k)
        return pos[:2] + R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9) * ANCHOR_FWD

    def _fwd(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def com(self, d):
        mujoco.mj_jacSubtreeCom(self.m, d, self._jcom, self.pelvis)
        return d.subtree_com[self.pelvis].copy(), self._jcom @ d.qvel

    # ------------------------------------------------------------------ start pose
    def settle(self, d, upper_pose, ground_z=None):
        """Put the robot in the walking crouch before control starts: pelvis at the
        walking height above its feet, legs solved by IK, soles flat on the ground,
        waist and arms at upper_pose, everything at rest."""
        m = self.m
        mujoco.mj_forward(m, d)
        R = d.xmat[self.pelvis].reshape(3, 3)
        self.yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        soles = [self.sole(d, k)[0] for k in range(2)]
        # each sole on the ground under it (uneven road), pelvis above the higher one
        gz = [self.surface(soles[k][0], soles[k][1]) if ground_z is None else ground_z
              for k in range(2)]
        targets = {k: np.array([soles[k][0], soles[k][1], gz[k] + SOLE_BOTTOM]) for k in range(2)}
        pel = d.xpos[self.pelvis].copy()
        pel[2] = max(gz) + self.p.pelvis_height
        self.q_des = d.qpos[7:].copy()
        self.q_des[:12] = np.array([-0.2, 0.0, 0.0, 0.42, -0.22, 0.0] * 2)
        rots = {k: self.ground_rot(soles[k][0], soles[k][1]) for k in range(2)}
        for _ in range(30):
            q = self._ik(d, pel, targets, rots=rots)
        d.qpos[0:3] = pel
        d.qpos[3:7] = _quat_yaw(self.yaw)
        d.qpos[7:19] = q
        d.qpos[19:] = upper_pose
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ plan
    def reset(self, d, speed, upper_pose):
        """Start a walk from double support. upper_pose: waist and arm targets (17)."""
        self.speed, self.upper_pose = speed, np.asarray(upper_pose, dtype=float)
        self.v_plan = speed           # planned speed, adapted when the robot falls behind
        mujoco.mj_forward(self.m, d)
        soles = [self.sole(d, k)[0] for k in range(2)]
        R = d.xmat[self.pelvis].reshape(3, 3)
        self.yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self.phase, self.t, self.n_steps = "init", 0.0, 0
        self.stance = 0                        # left foot stands first; right steps first
        self.A = 0.5 * (self.anchor(d, 0) + self.anchor(d, 1))   # ZMP segment start
        self.B = self.anchor(d, 0)                                # ZMP segment end
        self.land = soles[1].copy()            # sole-centre target of the swing foot
        self.lift = soles[1].copy()
        self.q_des = d.qpos[7:].copy()
        # seed the IK away from the straight-knee singularity
        self.q_des[:12] = np.array([-0.2, 0.0, 0.0, 0.42, -0.22, 0.0] * 2)
        self.omega = np.sqrt(G / max(d.subtree_com[self.pelvis][2] - soles[0][2], 0.4))
        self.step_i = -1

    def _step_length(self, k):
        """Length of the k-th step from now: the speed ramps up over the first 3 steps."""
        n = self.n_steps + k
        return self.v_plan * (self.p.t_ss + self.p.t_ds) * min(n / 3.0, 1.0)

    def _footsteps(self, base, base_side, n=3):
        """Nominal anchors of the next n footsteps after the foot at `base` (side
        base_side: 0 left, 1 right), alternating sides, in the walking heading."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        fwd, left = np.array([c, s]), np.array([-s, c])
        out, pos, side = [], base.copy(), base_side
        for k in range(1, n + 1):
            nxt = 1 - side
            lat = self.p.width * (1.0 if nxt == 0 else -1.0)      # move across by the width
            pos = pos + fwd * self._step_length(k) + left * lat
            out.append(pos.copy())
            side = nxt
        return out

    def _segments(self):
        """ZMP reference from the current phase onward, as (r0, r1, duration) pieces:
        constant under the stance foot in single support, a straight move to the next
        foot in double support (Englsberger et al. 2015)."""
        p = self.p
        if self.phase == "init":
            segs = [(self.A, self.B, p.t_init)]
            base, side = self.B, self.stance
            first = [(self.B, self.B, p.t_ss)]
        elif self.phase == "ss":
            segs = [(self.B, self.B, p.t_ss)]
            base, side = self.B, self.stance
            first = []
        else:   # ds from A to B, B = the foot that just landed
            segs = [(self.A, self.B, p.t_ds)]
            base, side = self.B, self.stance
            first = [(self.B, self.B, p.t_ss)]
        segs += first
        prev = base
        for f in self._footsteps(base, side):
            segs += [(prev, f, p.t_ds), (f, f, p.t_ss)]
            prev = f
        return segs

    def _xi_ref(self):
        """DCM reference now and at the end of the current single support, from the ZMP
        pieces solved backward (the walk is assumed to come to rest on the last step).
        For a piece with r(t) = r0 + (r1 - r0) t / T:
        xi(t) = r(t) + D + e^{w (t - T)} (xi_T - r1 - D),  D = (r1 - r0) / (w T)."""
        w = self.omega
        segs = self._segments()
        xi_T = [None] * len(segs)
        xi = segs[-1][1]
        for i in range(len(segs) - 1, -1, -1):
            r0, r1, T = segs[i]
            xi_T[i] = xi
            D = (r1 - r0) / (w * T)
            xi = r0 + D + np.exp(-w * T) * (xi - r1 - D)          # value at the piece start
        r0, r1, T = segs[0]
        t = min(self.t, T)
        D = (r1 - r0) / (w * T)
        r_ref = r0 + (r1 - r0) * t / T
        ref = r_ref + D + np.exp(w * (t - T)) * (xi_T[0] - r1 - D)
        # end of the single support that is under way or next
        ss_i = 0 if self.phase == "ss" else 1
        return ref, r_ref, xi_T[ss_i], segs[ss_i + 2][1] if self.phase == "ss" else None

    # ------------------------------------------------------------------ control
    def __call__(self, d, dt):
        """Joint targets (29, actuator order) for the current state; dt since last call."""
        p, m = self.p, self.m
        mujoco.mj_forward(m, d)
        self.t += dt
        soles = [self.sole(d, k) for k in range(2)]
        c, cv = self.com(d)
        w = self.omega
        xi = c[:2] + cv[:2] / w
        ref, r_ref, xi_ss_end, next_nom = self._xi_ref()
        r = r_ref + (1 + p.k_dcm / w) * (xi - ref)
        swing_target = None
        if self.phase == "ss":
            support = [self.stance]
            r = self._clip_to_sole(r, soles[self.stance])
            swing = 1 - self.stance
            t_rem = max(p.t_ss - self.t, 0.0)
            # capture-point step adjustment: land where the DCM will be at touchdown,
            # minus the planned offset between that DCM and the next footstep
            if t_rem > 0.06:
                xi_T = r + np.exp(w * t_rem) * (xi - r)
                land = self._limit_step(xi_T - (xi_ss_end - next_nom))
                # terrain-aware foothold: chosen while the step is young, then kept as
                # an offset from the (still live) capture-point target
                self._calls += 1
                if self.t < 0.5 * p.t_ss and self._calls % 4 == 1:
                    sole_xy = land - self._fwd() * ANCHOR_FWD
                    self._fh = self.pick_foothold(sole_xy) - sole_xy
                land = land + getattr(self, "_fh", np.zeros(2))
                self.land[:2] = land - self._fwd() * ANCHOR_FWD
                self.land[2] = self.surface(*self.land[:2]) + SOLE_BOTTOM
            s = self.t / p.t_ss
            xy = self.lift[:2] + _smooth((s - 0.1) / 0.75) * (self.land[:2] - self.lift[:2])
            z = self.lift[2] + (self.land[2] - self.lift[2]) * _smooth(s) \
                + p.swing_height * np.sin(np.pi * np.clip(s, 0, 1)) - 0.015 * _smooth((s - 0.85) / 0.3)
            swing_target = np.array([xy[0], xy[1], z])
            touched = self._in_contact(d, swing)
            if (s > 0.7 and touched) or s >= 1.25:
                # speed adaptation: if the DCM lags its plan along the walking direction,
                # plan the next steps slower, then relax back toward the command
                lag = float(np.dot(xi - ref, self._fwd()))
                self.v_plan = float(np.clip(self.v_plan + p.k_speed * lag
                                            + p.relax_speed * (self.speed - self.v_plan),
                                            0.3 * self.speed, self.speed))
                self.A, self.B = self.B, self.anchor(d, swing)
                self.stance, self.phase, self.t = swing, "ds", 0.0
                self.n_steps += 1
                self.step_i = self.n_steps
        else:
            support = [0, 1]
            r = self._clip_to_hull(r, soles)
            dur = p.t_init if self.phase == "init" else p.t_ds
            if self.t >= dur:
                self.phase, self.t = "ss", 0.0
                self.A = self.B
                self._fh = np.zeros(2)
                sw = 1 - self.stance
                self.lift = soles[sw][0].copy()
                self.land = soles[sw][0].copy()
                if self.step_i < 0:
                    self.step_i = 0
        self.stance_xy = self.B
        self.debug = dict(xi=xi.copy(), r=r.copy(), xi_ref=ref.copy(),
                          support=list(support))
        # ---- desired ground force and its split between the feet in contact
        ground = np.mean([soles[k][0][2] for k in support]) - SOLE_BOTTOM
        z_ref = ground + p.pelvis_height - (d.xpos[self.pelvis][2] - c[2])
        acc = np.r_[w * w * (c[:2] - r), p.kz * (z_ref - c[2]) - p.dz * cv[2]]
        F = self.mass * (acc + np.array([0.0, 0.0, G]))
        F[2] = max(F[2], 0.0)
        lam = self._distribute(d, c, F, r, support, soles)
        # ---- inverse kinematics: pelvis at height and upright, feet on target
        pel_des = np.array([d.xpos[self.pelvis][0], d.xpos[self.pelvis][1],
                            ground + p.pelvis_height])
        targets = {}
        for k in (0, 1):
            pos, _ = soles[k]
            if k in support or swing_target is None:
                targets[k] = pos.copy()
            else:
                targets[k] = swing_target
        swing_k = None if swing_target is None else 1 - support[0]
        if swing_k is not None:
            # feedback on the swing foot: aim past the target by part of the tracking
            # error, so low-gain servos and body tilt do not leave the foot behind or ahead
            err = targets[swing_k] - soles[swing_k][0]
            targets[swing_k] = targets[swing_k] + p.k_swing * err
        # sole orientations: a foot on the ground lies on the ground's slope under it;
        # the swing foot turns from its lift-off slope to the slope where it will land
        rots = {}
        for k in (0, 1):
            if k == swing_k:
                s_sw = np.clip(self.t / p.t_ss, 0.0, 1.0)
                R_land = self.ground_rot(*self.land[:2])
                rots[k] = R_land if s_sw > 0.5 else self.ground_rot(*soles[k][0][:2])
            elif p.stance_rot == "actual":
                rots[k] = soles[k][1]
            elif p.stance_rot == "flat":
                rots[k] = self.ground_rot(*soles[k][0][:2]) if False else np.array(
                    [[np.cos(self.yaw), -np.sin(self.yaw), 0], [np.sin(self.yaw), np.cos(self.yaw), 0], [0, 0, 1]])
            else:
                rots[k] = self.ground_rot(*soles[k][0][:2])
        q = self._ik(d, pel_des, targets, actual_base_for=swing_k, rots=rots)
        # ---- joint torques: tau = bias - J_c^T lambda (legs), bias for waist and arms
        tau = d.qfrc_bias[6:].copy()
        for k, (Fk, Mk) in lam.items():
            pk = soles[k][0]
            mujoco.mj_jac(m, d, self._jp, self._jr, pk, self.feet[k])
            tau -= (self._jp[:, 6:].T @ Fk + self._jr[:, 6:].T @ Mk)
        # trunk attitude (hip strategy): torque at the stance hips that tilts the pelvis
        # back to upright. A hip torque tau acts on the pelvis as -tau, so
        # tau = k * angle + d * rate rotates the pelvis against its pitch and roll.
        Rp = d.xmat[self.pelvis].reshape(3, 3)
        pitch = np.arcsin(np.clip(Rp[0, 2], -1, 1))
        roll = np.arctan2(-Rp[1, 2], Rp[2, 2])
        wb = d.qvel[3:6]                                  # pelvis angular velocity, body frame
        fz = {k: max(lam[k][0][2], 0.0) for k in lam}
        tot = sum(fz.values()) or 1.0
        for k in lam:
            share = fz[k] / tot
            o = 6 * k
            tau[o + 0] += share * (p.k_trunk * pitch + p.d_trunk * wb[1])
            tau[o + 1] += share * (p.k_trunk * roll + p.d_trunk * wb[0])
        # heading: the swinging leg twists the body about the vertical. Hold the
        # heading through the stance hips' yaw joints (a hip torque acts on the pelvis
        # as -tau, so tau = -k * error + d * yaw rate turns it back).
        yaw_now = np.arctan2(Rp[1, 0], Rp[0, 0])
        yaw_err = (self.yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi
        wz = (Rp @ wb)[2]
        for k in lam:
            share = fz[k] / tot
            tau[6 * k + 2] += share * (-p.k_yaw * yaw_err + p.d_yaw * wz)
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        upper = self.upper_pose.copy()
        if p.arm_swing:
            # each arm swings opposite to the leg on its own side (as in human walking),
            # which cancels much of the twist the legs put into the body
            c_, s_ = np.cos(self.yaw), np.sin(self.yaw)
            fx = [c_ * soles[k][0][0] + s_ * soles[k][0][1] for k in (0, 1)]
            sep = fx[0] - fx[1]                        # left foot ahead of the right: > 0
            # waist (3) then left arm (7), right arm (7); shoulder pitch: + = arm back
            upper[3] += p.arm_swing * sep
            upper[10] -= p.arm_swing * sep
        q_des = np.r_[q, upper]
        return q_des + tau / self.kp

    def _distribute(self, d, c, F, r, support, soles):
        """Contact wrenches (force, moment about the sole centre) for the feet in support.
        F is split by where the ZMP r lies between the feet, each share acting at its own
        centre of pressure inside its sole. The net yaw moment about the CoM that this
        leaves would slowly twist the robot on its feet, so it is cancelled (plus a small
        heading correction) through the feet's twisting moments."""
        p = self.p
        feet = list(support)
        if len(feet) == 2:
            pl, pr = self.anchor(d, 0), self.anchor(d, 1)
            seg = pl - pr
            a = float(np.clip(np.dot(r - pr, seg) / max(np.dot(seg, seg), 1e-6), 0.0, 1.0))
            share = {0: a, 1: 1.0 - a}
            on_line = a * pl + (1 - a) * pr
            cop = {k: self._clip_to_sole(self.anchor(d, k) + (r - on_line), soles[k]) for k in feet}
        else:
            share, cop = {feet[0]: 1.0}, {feet[0]: r}
        out, yaw_tot = {}, 0.0
        for k in feet:
            pk = soles[k][0]
            rk3 = np.array([cop[k][0], cop[k][1], pk[2] - SOLE_BOTTOM])
            Fk = share[k] * F
            Mk = np.cross(rk3 - pk, Fk)
            out[k] = [Fk, Mk]
            yaw_tot += (np.cross(pk - c, Fk) + Mk)[2]
        R = d.xmat[self.pelvis].reshape(3, 3)
        yaw_err = (self.yaw - np.arctan2(R[1, 0], R[0, 0]) + np.pi) % (2 * np.pi) - np.pi
        wz = (R @ d.qvel[3:6])[2]
        yaw_des = p.k_rot * yaw_err - p.d_rot * wz
        fz = sum(out[k][0][2] for k in feet)
        for k in feet:
            frac = out[k][0][2] / fz if fz > 1e-6 else 1.0 / len(feet)
            out[k][1][2] += frac * (yaw_des - yaw_tot)
            lim = p.torsion * max(out[k][0][2], 0.0)     # torsional friction limit
            out[k][1][2] = float(np.clip(out[k][1][2], -lim, lim))
        return {k: (v[0], v[1]) for k, v in out.items()}

    # ------------------------------------------------------------------ helpers
    def slope(self, x, y):
        """Steepest ground slope (rise per metre) across a sole-sized patch at (x, y);
        x and y may be arrays."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        h = self.surface_vec
        gx = (h(x + 0.08, y) - h(x - 0.08, y)) / 0.16
        gy = (h(x, y + 0.03) - h(x, y - 0.03)) / 0.06
        return np.hypot(gx, gy)

    def pick_foothold(self, xy, n=7, w_dist=1.5):
        """Terrain-aware footstep: the point within foothold_radius of the capture-point
        target that minimises slope + w_dist x distance (avoids landing on rut walls)."""
        radius = self.p.foothold_radius
        if radius <= 0:
            return xy
        g = np.linspace(-radius, radius, n)
        dx, dy = np.meshgrid(g, g, indexing="ij")
        dx, dy = dx.ravel(), dy.ravel()
        cost = self.slope(xy[0] + dx, xy[1] + dy) + w_dist * np.hypot(dx, dy)
        cost0 = float(self.slope(xy[0], xy[1]))
        i = int(np.argmin(cost))
        if cost[i] < cost0:
            return xy + np.array([dx[i], dy[i]])
        return xy

    def ground_rot(self, x, y):
        """Sole orientation lying on the ground at (x, y), in the walking heading: pitch
        and roll from the terrain slope over the length and width of the sole (cached to
        the millimetre: planted feet ask for the same point every step)."""
        key = (round(float(x), 3), round(float(y), 3), round(float(self.yaw), 3))
        hit = self._rot_cache.get(key)
        if hit is not None:
            return hit
        if len(self._rot_cache) > 4096:
            self._rot_cache.clear()
        R = self._ground_rot(x, y)
        self._rot_cache[key] = R
        return R

    def _ground_rot(self, x, y):
        c, s_ = np.cos(self.yaw), np.sin(self.yaw)
        f, l = np.array([c, s_]), np.array([-s_, c])
        h = lambda q: self.surface(q[0], q[1])
        q0 = np.array([x, y])
        gx = (h(q0 + 0.08 * f) - h(q0 - 0.08 * f)) / 0.16
        gy = (h(q0 + 0.03 * l) - h(q0 - 0.03 * l)) / 0.06
        # capped at what the G1's ankle can reach (roll +-0.26 rad, pitch -0.87..0.52)
        pitch = float(np.clip(-np.arctan(gx), -0.45, 0.45))
        roll = float(np.clip(np.arctan(gy), -0.22, 0.22))
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    def _clip_to_hull(self, r, soles):
        """Nearest point to r inside the convex hull of both soles (shrunk by 1 cm)."""
        pts = []
        for pos, R in soles:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            c, s_ = np.cos(yaw), np.sin(yaw)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    lx, ly = sx * (SOLE_HALF[0] - 0.01), sy * (SOLE_HALF[1] - 0.01)
                    pts.append(pos[:2] + np.array([c * lx - s_ * ly, s_ * lx + c * ly]))
        hull = _convex_hull(np.array(pts))
        if _inside(r, hull):
            return r
        best, bd = r, np.inf
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            ab = b - a
            t = np.clip(np.dot(r - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
            q = a + t * ab
            dd = np.sum((r - q) ** 2)
            if dd < bd:
                best, bd = q, dd
        return best

    def _in_contact(self, d, k):
        body = self.feet[k]
        for i in range(d.ncon):
            c = d.contact[i]
            if self.m.geom_bodyid[c.geom1] == body or self.m.geom_bodyid[c.geom2] == body:
                return True
        return False

    def _contact_points(self, d, k):
        """World positions of foot k's sole corners that touch the ground now; all four
        corners if none does (a foot about to land)."""
        geoms = self.sole_geoms[k]
        on = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in geoms:
                on.add(c.geom1)
            elif c.geom2 in geoms:
                on.add(c.geom2)
        use = [g for g in geoms if g in on] or list(geoms)
        return np.array([d.geom_xpos[g][:2] for g in use])

    def _clip_to_support(self, d, r, feet, soles):
        """Nearest point to r inside the support area actually available: the convex hull
        of the sole corners of `feet` that are in contact (on rough ground often only two
        or three of a sole's four corners touch), shrunk 5 mm toward its centre."""
        pts = np.vstack([self._contact_points(d, k) for k in feet])
        ctr = pts.mean(axis=0)
        pts = ctr + (pts - ctr) * np.clip(1 - 0.005 / np.maximum(np.linalg.norm(pts - ctr, axis=1), 1e-6), 0, 1)[:, None]
        if len(pts) == 1:
            return pts[0]
        hull = _convex_hull(pts) if len(pts) >= 3 else pts
        if len(hull) >= 3 and _inside(r, hull):
            return r
        best, bd = hull[0], np.inf
        n = len(hull)
        for i in range(n if n >= 3 else 1):
            a, b = hull[i], hull[(i + 1) % n]
            ab = b - a
            t = np.clip(np.dot(r - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
            q = a + t * ab
            dd = np.sum((r - q) ** 2)
            if dd < bd:
                best, bd = q, dd
        return best

    def _clip_to_sole(self, r, sole):
        pos, R = sole
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        rel = r - pos[:2]
        loc = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
        loc = np.clip(loc, -(SOLE_HALF - 0.01), SOLE_HALF - 0.01)
        return pos[:2] + np.array([c * loc[0] - s * loc[1], s * loc[0] + c * loc[1]])

    def _limit_step(self, land):
        p = self.p
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rel = land - self.stance_xy
        x = c * rel[0] + s * rel[1]
        y = -s * rel[0] + c * rel[1]
        side = 1.0 if self.stance == 1 else -1.0     # swing foot is left (+y) of a right stance
        x = np.clip(x, p.min_step_x, p.max_step_x)
        y = side * np.clip(side * y, p.min_width, p.max_width)
        return self.stance_xy + np.array([c * x - s * y, s * x + c * y])

    def _ik(self, d, pel_pos, targets, actual_base_for=None, rots=None):
        """Leg joint angles (12) that put the sole centres at the targets (flat, in the
        walking heading). Stance legs are solved with the pelvis at pel_pos, upright, in
        the walking heading (so the servos hold the trunk up); the swing leg
        (actual_base_for) with the pelvis where it really is, so trunk tilt does not
        move the swing foot."""
        q = self._ik_base(d, pel_pos, _quat_yaw(self.yaw), targets, legs=[k for k in (0, 1)
                          if k != actual_base_for], rots=rots)
        if actual_base_for is not None:
            q2 = self._ik_base(d, d.qpos[0:3].copy(), d.qpos[3:7].copy(), targets,
                               legs=[actual_base_for], rots=rots)
            idx = self.leg_dofs[actual_base_for] - 6
            q[idx] = q2[idx]
        self.q_des[:12] = q
        return q

    def _ik_base(self, d, base_pos, base_quat, targets, legs, rots=None):
        m, sd = self.m, self.scratch
        sd.qpos[:] = d.qpos
        sd.qpos[0:3] = base_pos
        sd.qpos[3:7] = base_quat
        sd.qpos[7:19] = self.q_des[:12]
        R_des = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                          [np.sin(self.yaw), np.cos(self.yaw), 0], [0, 0, 1]])
        lo, hi = m.jnt_range[1:13, 0], m.jnt_range[1:13, 1]
        for _ in range(self.p.ik_iters):
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            for k in legs:
                pos, R = self.sole(sd, k)
                Rk = rots[k] if rots is not None else R_des
                err = np.r_[targets[k] - pos, _rot_err(R, Rk)]
                mujoco.mj_jac(m, sd, self._jp, self._jr, pos, self.feet[k])
                J = np.vstack([self._jp, self._jr])[:, self.leg_dofs[k]]
                dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ err)
                idx = self.leg_dofs[k] + 1
                sd.qpos[idx] = np.clip(sd.qpos[idx] + np.clip(dq, -0.3, 0.3),
                                       lo[idx - 7] + 0.05, hi[idx - 7] - 0.05)
        return sd.qpos[7:19].copy()
