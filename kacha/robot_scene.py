"""Place a MuJoCo Menagerie robot on a generated village-road terrain."""

import mujoco
import mujoco_menagerie as mm
import numpy as np


def terrain_height(heights, length, width, x, y):
    """Height of the terrain (metres) at world point (x, y), nearest grid cell."""
    nx, ny = heights.shape
    i = int(round((x + length / 2) / length * (nx - 1)))
    j = int(round((y + width / 2) / width * (ny - 1)))
    return float(heights[np.clip(i, 0, nx - 1), np.clip(j, 0, ny - 1)])


def build_robot_scene(heights, length, width, robot="unitree_go1"):
    """Return (model, home_qpos) with the robot standing above the road centre."""
    entry = mm.get(robot)
    spec = entry.spec(entry.default_model)

    zmin, zmax = float(heights.min()), float(heights.max())
    elev = max(zmax - zmin, 1e-3)
    data = ((heights - zmin) / elev).T          # MuJoCo hfield rows run along y
    nrow, ncol = data.shape

    spec.add_hfield(name="road", nrow=nrow, ncol=ncol,
                    size=[length / 2, width / 2, elev, 0.1],
                    userdata=data.ravel().astype(np.float32).tolist())
    spec.add_material(name="earth", rgba=[0.55, 0.42, 0.28, 1])
    spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname="road", pos=[0, 0, zmin], material="earth")
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    model = spec.compile()

    home = model.key("home").qpos.copy()
    home[2] += terrain_height(heights, length, width, home[0], home[1])
    return model, home
