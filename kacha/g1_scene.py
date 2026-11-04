"""Put the Unitree G1 humanoid on a generated village road, with mud, water and the
scene look from kacha.look."""

import mujoco
import mujoco_menagerie as mm
import numpy as np

from .look import add_environment, add_materials
from .mud import add_flat_mud_plates
from .robot_scene import _add_hfield, terrain_height

SIDES = ("left", "right")
# Joint PD gains, from Unitree's G1 RL setup (unitree_rl_gym, g1 config): legs
# kp hip 100, knee 150, ankle 40; kd 2, 4, 2. Waist and arms are only held in pose in
# stage 1, with moderate gains.
KP = {"hip_pitch": 100, "hip_roll": 100, "hip_yaw": 100, "knee": 150,
      "ankle_pitch": 40, "ankle_roll": 40, "waist": 150, "shoulder": 40, "elbow": 40,
      "wrist": 20}
KD = {"hip_pitch": 2, "hip_roll": 2, "hip_yaw": 2, "knee": 4, "ankle_pitch": 2,
      "ankle_roll": 2, "waist": 3, "shoulder": 1, "elbow": 1, "wrist": 0.5}


def _gain_key(joint):
    for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
              "waist", "shoulder", "elbow", "wrist"):
        if k in joint:
            return k
    raise KeyError(joint)


def foot_geoms():
    return {s: [f"{s}_sole_{i}" for i in range(4)] for s in SIDES}


def build_g1_scene(heights, length, width, road_width=3.0, mud_depth=0.0, water=None,
                   base_xy=(0.0, 0.0), timestep=0.005, seed=0, look=True):
    """Return (model, home_qpos). Only the soles, pelvis and torso touch the world (as
    in legged-robot RL setups); the robot does not collide with itself."""
    entry = mm.get("unitree_g1")
    spec = entry.spec(entry.default_model)
    spec.option.timestep = timestep
    counts = {s: 0 for s in SIDES}
    for g in spec.geoms:
        body = g.parent.name if g.parent else ""
        if body.endswith("ankle_roll_link") and g.type == mujoco.mjtGeom.mjGEOM_SPHERE:
            side = body.split("_")[0]
            g.name = f"{side}_sole_{counts[side]}"
            counts[side] += 1
            g.contype, g.conaffinity = 1, 1
        elif body in ("pelvis", "torso_link") and (g.contype or g.conaffinity):
            g.contype, g.conaffinity = 1, 0          # touches the ground (falls), not itself
        else:
            g.contype, g.conaffinity = 0, 0
    if look:
        add_materials(spec, length, width)
        add_environment(spec, length, width, road_width, seed=seed)
    else:
        spec.add_material(name="earth", rgba=[0.55, 0.42, 0.28, 1])
        spec.add_material(name="mud", rgba=[0.33, 0.24, 0.15, 0.6])
    _add_hfield(spec, "terrain", heights - mud_depth, length, width, material="earth")
    if mud_depth > 0:
        _add_hfield(spec, "mud", heights, length, width, material="mud",
                    contype=0, conaffinity=0, group=1)
        add_flat_mud_plates(spec, foot_geoms())
    if water is not None:
        water.add_visuals(spec, material="water" if look else None)
    model = spec.compile()
    for i in range(model.nu):
        name = model.actuator(i).name
        kp, kd = KP[_gain_key(name)], KD[_gain_key(name)]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
    home = model.key("stand").qpos.copy()
    home[0], home[1] = base_xy
    home[2] += terrain_height(heights, length, width, home[0], home[1])
    return model, home
