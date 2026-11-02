"""Build a MuJoCo model containing a generated terrain height field."""

import mujoco
import numpy as np

SCENE_XML = """
<mujoco model="kacha_road">
  <option timestep="0.002"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <asset>
    <hfield name="road" nrow="{nrow}" ncol="{ncol}" size="{rx} {ry} {elev} 0.1"/>
    <material name="earth" rgba="0.55 0.42 0.28 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="terrain" type="hfield" hfield="road" pos="0 0 {zmin}" material="earth"/>
    <body name="probe" pos="0 0 1.0">
      <freejoint/>
      <geom type="box" size="0.15 0.1 0.05" mass="5" rgba="0.2 0.4 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_model(heights, length, width):
    """Return a compiled MjModel with the given heights (metres).

    MuJoCo stores height fields normalised to [0, 1] and scales them by the
    elevation size, so the heights are normalised here and the terrain geom is
    shifted down by the minimum height.
    """
    zmin, zmax = float(heights.min()), float(heights.max())
    elev = max(zmax - zmin, 1e-3)
    # MuJoCo hfield rows run along y, columns along x.
    data = ((heights - zmin) / elev).T
    nrow, ncol = data.shape
    xml = SCENE_XML.format(nrow=nrow, ncol=ncol, rx=length / 2, ry=width / 2,
                           elev=elev, zmin=zmin)
    model = mujoco.MjModel.from_xml_string(xml)
    model.hfield_data[:] = data.ravel().astype(np.float32)
    return model
