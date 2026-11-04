"""Scripted mud-walking reference gait for the Unitree G1 (no motion capture).

The gait is generated from foot trajectories and solved to joint angles with analytic
leg inverse kinematics. Its style follows measured human adaptations to slippery and
unstable ground (docs/humanoid.md, "Reference gait"):

- shorter strides and a flat-footed landing          (Cham & Redfern 2002)
- higher foot clearance (more knee and hip flexion
  in swing) and a flat foot at contact                (Gates et al. 2012)
- longer double support (slower loading of the foot)  (Cham & Redfern 2002)
- a vertical lift before the forward swing, to break suction in mud (our assumption)
- arms held slightly out from the body for balance  (our assumption)

The policy does not copy this gait exactly. It adds a learned correction on top of it
(kacha/g1_env.py), so the reference only has to be roughly right.
"""

from dataclasses import dataclass

import numpy as np

# G1 leg geometry (MuJoCo Menagerie unitree_g1, standing keyframe)
THIGH = 0.3366        # hip-pitch axis to knee axis (m)
SHANK = 0.3000        # knee axis to ankle-pitch axis (m)
HIP_Y = 0.1185        # lateral offset of the ankle under the hip in the zero pose (m)

LEG_JOINTS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]


@dataclass
class GaitParams:
    period: float = 1.1          # s per full cycle (two steps)
    duty: float = 0.65           # share of the cycle each foot is on the ground
    hip_height: float = 0.63     # hip-pitch axis to ankle axis in stance (m); straight leg
                                 # 0.637. Gives knee 0.29 rad, Unitree's G1 default: deeper
                                 # crouches sag under the real knee gain in single support
    clearance: float = 0.10      # peak foot lift in swing (m)
    lift_first: float = 0.2      # share of the swing spent lifting before moving forward
    width: float = 0.0           # extra stance width per foot (m); literature: no change
    sway: float = 0.025          # lateral pelvis shift toward the stance foot (m)
    arm_out: float = 0.30        # shoulder roll away from the body (rad)
    arm_swing: float = 0.15      # shoulder pitch counter-swing amplitude (rad)
    elbow: float = 0.90          # elbow bend (rad)

    def stride(self, speed):
        """Distance a foot moves (relative to the hip) while it is on the ground: the
        body moves at `speed` for the stance time duty * period."""
        return speed * self.period * self.duty


def leg_ik(x, z):
    """Hip pitch, knee and ankle pitch (rad) that put the ankle at (x forward, z up)
    relative to the hip-pitch axis, with the sole level. G1 signs: hip pitch and knee
    positive move the foot backward; knee positive is flexion."""
    D = np.clip(np.hypot(x, z), 0.05, THIGH + SHANK - 1e-4)
    c = (THIGH ** 2 + SHANK ** 2 - D ** 2) / (2 * THIGH * SHANK)
    knee = np.pi - np.arccos(np.clip(c, -1.0, 1.0))
    gamma = np.arctan2(-x, -z)                       # direction hip -> ankle, from straight down
    hip = gamma - np.arctan2(SHANK * np.sin(knee), THIGH + SHANK * np.cos(knee))
    return hip, knee, -(hip + knee)


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def foot_path(phase, stride, g):
    """Ankle position (x, z) of one foot relative to the hip, for phase in [0, 1):
    stance first (foot moves back under the body), then swing."""
    if phase < g.duty:
        s = phase / g.duty
        return stride * (0.5 - s), 0.0, True
    s = (phase - g.duty) / (1.0 - g.duty)
    forward = _smooth((s - g.lift_first) / (1.0 - g.lift_first))   # lift, then swing forward
    x = stride * (-0.5 + forward)
    z = g.clearance * np.sin(np.pi * s) ** 1.5
    return x, z, False


class MudGait:
    """Reference joint targets for the G1 walking at a commanded forward speed."""

    def __init__(self, params=None):
        self.g = params or GaitParams()

    def targets(self, phase, speed, place=(0.0, 0.0)):
        """Return (leg_targets[12], arm_targets[14], waist[3], stance[2]) for the
        phase of the left foot (right foot is half a cycle later). Order matches the
        Menagerie G1 actuators: left leg, right leg, then waist, then left/right arm.
        place: (dx, dy) shift of where a swinging foot lands, from the balance
        controller (zero for the plain reference)."""
        g = self.g
        stride = g.stride(speed)
        legs, stance = [], []
        # pelvis sways over the foot that is on the ground: the right foot swings while
        # the left phase is in [duty - 0.5, 0.5), centred on duty / 2, so the sway
        # toward the left (+y) peaks there (and toward the right half a cycle later)
        sway = g.sway * np.cos(2 * np.pi * (phase - g.duty / 2))
        for side, (ph, sgn) in enumerate(((phase, 1.0), ((phase + 0.5) % 1.0, -1.0))):
            x, z, on = foot_path(ph, stride, g)
            dy = sgn * g.width - sway                       # foot relative to its hip, in y
            if not on:                                      # steer the landing point
                s_sw = (ph - g.duty) / (1.0 - g.duty)
                x += place[0] * _smooth(s_sw)
                dy += place[1] * _smooth(s_sw)
            hip, knee, ankle = leg_ik(x, -g.hip_height + z)
            roll = np.arctan2(dy, g.hip_height - z)
            legs += [hip, roll, 0.0, knee, ankle, -roll]
            stance.append(on)
        # arms: slightly out, elbows bent, small swing opposite the legs
        swing = g.arm_swing * np.sin(2 * np.pi * phase)
        arms = []
        for sgn, s in ((1.0, swing), (-1.0, -swing)):
            arms += [0.2 + s, sgn * g.arm_out, 0.0, g.elbow, 0.0, 0.0, 0.0]
        return np.array(legs), np.array(arms), np.zeros(3), np.array(stance)

    def standing(self):
        """Joint targets for standing still in the gait's crouch."""
        legs, arms, waist, _ = self.targets(0.0, 0.0)
        return legs, arms, waist


@dataclass
class BalanceGains:
    """Scripted balance (tuned in simulation on rigid ground, docs/humanoid.md).
    Ankle torque from the centre-of-mass error over the support feet (ankle strategy),
    hip feedback on trunk tilt (hip strategy), and step placement from the velocity
    error (Raibert 1986)."""
    com: float = 20.0        # rad of ankle per m of CoM error
    com_d: float = 2.0       # rad of ankle per m/s of CoM velocity error
    hip: float = 1.0         # rad of hip per rad of trunk tilt
    place_x: float = 0.25    # m of step shift per m/s of forward-speed error
    place_y: float = 0.0     # m of step shift per m/s of sideways velocity (off: the
                             # planned sway itself moves the body sideways)
    max_place: float = 0.12  # m


class ScriptedWalker:
    """Reference gait plus balance feedback: a complete scripted controller. It is the
    baseline in the benchmark and the base the learned policy corrects."""

    def __init__(self, params=None, gains=None):
        self.gait = MudGait(params)
        self.k = gains or BalanceGains()

    def __call__(self, phase, speed, com, com_vel, pelvis_R, sole_centres, stance_ref):
        """Joint targets (29, actuator order). sole_centres: (2, 3) world positions of
        the left and right soles; stance_ref: which feet the reference has down."""
        k = self.k
        # heading frame of the pelvis
        yaw = np.arctan2(pelvis_R[1, 0], pelvis_R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        to_local = lambda v: np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        v = to_local(com_vel)
        dx = np.clip(k.place_x * (v[0] - speed), -k.max_place, k.max_place)
        dy = np.clip(k.place_y * v[1], -k.max_place, k.max_place)
        legs, arms, waist, stance = self.gait.targets(phase, speed, (dx, dy))
        # centre of mass relative to the feet that are down (both if none or both)
        down = stance_ref if stance_ref.any() else np.array([True, True])
        sup = sole_centres[down].mean(axis=0)
        e = to_local(com[:2] - sup[:2])
        # forward: hold the CoM over the support while walking at the command
        ax = k.com * e[0] + k.com_d * (v[0] - speed)
        ay = k.com * e[1] + k.com_d * v[1]
        pitch = np.arcsin(np.clip(pelvis_R[0, 2], -1, 1))
        roll = np.arctan2(-pelvis_R[1, 2], pelvis_R[2, 2])
        for o, on in ((0, stance_ref[0]), (6, stance_ref[1])):
            w = 1.0 if on else 0.3
            legs[o + 4] += w * ax
            legs[o + 5] -= w * ay
            legs[o + 0] += k.hip * pitch
            legs[o + 1] -= k.hip * roll
        return np.concatenate([legs, waist, arms]), stance
