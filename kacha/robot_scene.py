"""Place a MuJoCo Menagerie robot on a generated village-road terrain."""

import mujoco
import mujoco_menagerie as mm
import numpy as np

from .mud import add_mud_plates

GO1_FEET = ["FR", "FL", "RR", "RL"]


def terrain_height(heights, length, width, x, y):
    """Bilinear height of the terrain (metres) at world point (x, y)."""
    nx, ny = heights.shape
    fx = np.clip((x + length / 2) / length * (nx - 1), 0, nx - 1)
    fy = np.clip((y + width / 2) / width * (ny - 1), 0, ny - 1)
    i0, j0 = int(min(fx, nx - 2)), int(min(fy, ny - 2))
    tx, ty = fx - i0, fy - j0
    h = heights
    return float((1 - tx) * (1 - ty) * h[i0, j0] + tx * (1 - ty) * h[i0 + 1, j0]
                 + (1 - tx) * ty * h[i0, j0 + 1] + tx * ty * h[i0 + 1, j0 + 1])


def _add_hfield(spec, name, heights, length, width, **geom_kw):
    zmin, zmax = float(heights.min()), float(heights.max())
    elev = max(zmax - zmin, 1e-3)
    data = ((heights - zmin) / elev).T          # MuJoCo hfield rows run along y
    nrow, ncol = data.shape
    spec.add_hfield(name=name, nrow=nrow, ncol=ncol,
                    size=[length / 2, width / 2, elev, 0.1],
                    userdata=data.ravel().astype(np.float32).tolist())
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname=name, pos=[0, 0, zmin], **geom_kw)


def build_robot_scene(heights, length, width, robot="unitree_go1", mud_depth=0.0,
                      feet=GO1_FEET):
    """Return (model, home_qpos) with the robot standing on the road centre.

    heights is the ground surface. With mud_depth > 0 the rigid geometry ("terrain")
    is the firm base mud_depth below the surface, and a see-through, non-colliding
    "mud" surface is drawn at the original heights; mud forces come from kacha.mud.
    """
    entry = mm.get(robot)
    spec = entry.spec(entry.default_model)

    spec.add_material(name="earth", rgba=[0.55, 0.42, 0.28, 1])
    spec.add_material(name="mud", rgba=[0.33, 0.24, 0.15, 0.55])
    _add_hfield(spec, "terrain", heights - mud_depth, length, width, material="earth")
    if mud_depth > 0:
        _add_hfield(spec, "mud", heights, length, width, material="mud",
                    contype=0, conaffinity=0, group=1)
        add_mud_plates(spec, feet)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    model = spec.compile()
    home = model.key("home").qpos.copy()
    home[2] += terrain_height(heights, length, width, home[0], home[1])
    return model, home
