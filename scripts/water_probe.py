"""Probe test for the water model: computed forces vs hand-calculated values.

Uses Go1-sized parts: a lower-leg capsule (radius 1 cm, 13.2 cm long) and a foot
sphere (radius 2.3 cm). Gravity is off so only the water force is measured.

Usage: python scripts/water_probe.py
"""

import mujoco
import numpy as np

from kacha.water import CD_CYL, G, RHO, Water, WaterForces

A_LEG, L_LEG = 0.010, 0.066        # capsule radius and half-length (Go1 calf)
A_FOOT = 0.023                     # Go1 foot sphere radius

XML = """
<mujoco><option gravity="0 0 0"/>
  <worldbody>
    <body name="part" pos="0 0 0" {euler}>
      <freejoint/>
      <geom name="g" type="{type}" size="{size}" mass="0.2"/>
    </body>
  </worldbody></mujoco>"""

POOL = [{"center": (0.0, 0.0), "radius": 1.0, "depth": 0.5, "rim": 0.5}]  # level 0.49


def measure(geom_type, size, euler="", z=0.0, vel=None):
    m = mujoco.MjModel.from_xml_string(XML.format(type=geom_type, size=size, euler=euler))
    d = mujoco.MjData(m)
    d.qpos[2] = z
    if vel is not None:
        d.qvel[:3] = vel
    mujoco.mj_forward(m, d)
    water = Water(POOL, fill=1.0)
    wf = WaterForces(m, water)
    d.qfrc_applied[:] = 0
    wf.apply(m, d)
    return d.qfrc_applied[:3].copy(), water.pools[0]["level"]


def row(name, got, expected):
    err = abs(got - expected) / expected * 100
    print(f"{name:52s} {got:8.4f} N   expected {expected:8.4f} N   ({err:4.1f}% off)")


def main():
    lvl = Water(POOL).pools[0]["level"]
    print("Water forces on Go1-sized parts (fresh water, still)\n")

    F, _ = measure("capsule", f"{A_LEG} {L_LEG}", z=lvl)
    row("Buoyancy, vertical leg, half submerged", F[2], RHO * G * np.pi * A_LEG**2 * L_LEG)

    F, _ = measure("capsule", f"{A_LEG} {L_LEG}", z=lvl - 0.5)
    row("Buoyancy, vertical leg, fully submerged", F[2], RHO * G * np.pi * A_LEG**2 * 2 * L_LEG)

    F, _ = measure("sphere", f"{A_FOOT}", z=lvl)
    row("Buoyancy, foot sphere, half submerged", F[2], RHO * G * 2 / 3 * np.pi * A_FOOT**3)

    v = 0.5
    F, _ = measure("capsule", f"{A_LEG} {L_LEG}", euler='euler="90 0 0"', z=lvl - 0.2,
                   vel=[v, 0, 0])
    row("Drag, leg moving sideways at 0.5 m/s, submerged", -F[0],
        0.5 * RHO * CD_CYL * (2 * A_LEG) * (2 * L_LEG) * v * v)

    F, _ = measure("capsule", f"{A_LEG} {L_LEG}", z=lvl - 0.5, vel=[0, 0, v])
    print(f"{'Drag, leg moving along its own axis (should be ~0)':52s} {-F[2] + RHO*G*np.pi*A_LEG**2*2*L_LEG:8.4f} N")

    re = RHO * v * 2 * A_LEG / 1.0e-3
    print(f"\nReynolds number of a 2 cm leg at 0.5 m/s: {re:,.0f} (subcritical, Cd ~ 1)")
    print("For scale: Go1 weight is 125 N; a quarter is 31 N.")


if __name__ == "__main__":
    main()
