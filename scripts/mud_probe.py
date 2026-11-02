"""Foot-drop probe test for the mud model (no robot, no learning).

A. Force-depth curves from the equations: push a Go1-sized foot into the mud, then pull
   it out, for each season. Saves docs/figures/mud_force_depth.png.
B. Sinkage in simulation: a foot carrying a quarter of the Go1's weight is loaded onto
   the mud in MuJoCo; its settled sinkage is compared with the equation prediction.

Usage: python scripts/mud_probe.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from kacha.mud import (PADDY_CLAY, SEASONS, MudContact, MudParams, add_mud_plates,
                       bearing_capacity, quasi_static_curve, water_content)

R_FOOT = 0.023                    # m, Go1 foot sphere radius
LEG_D = 0.02                      # m, approximate lower-leg diameter
LOAD = 12.74 * 9.81 / 4           # N, quarter of the Go1 weight
WET = [s for s in SEASONS if SEASONS[s]["depth"] > 0]


def curves():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    print("A. Push in to 10 cm (or mud depth), then pull out (equations)")
    print(f"{'season':15s} {'LI':>5s} {'w %':>5s} {'su kPa':>7s} {'F at 2.3 cm':>12s} {'F at 8 cm':>10s} {'peak suction':>13s}")
    for season in WET:
        p = MudParams.from_season(season)
        depths = np.linspace(0, min(0.10, p.depth), 200)
        F_in, F_out = quasi_static_curve(depths, p, R_FOOT, LEG_D)
        LI = SEASONS[season]["LI"]
        line, = ax.plot(depths * 100, F_in, label=f"{season} (LI {LI:.1f}, su {p.su/1e3:.1f} kPa)")
        ax.plot(depths[::-1] * 100, F_out, "--", color=line.get_color())
        print(f"{season:15s} {LI:5.2f} {water_content(PADDY_CLAY, LI):5.1f} {p.su/1e3:7.2f} "
              f"{np.interp(0.023, depths, F_in):10.1f} N {np.interp(0.08, depths, F_in):8.1f} N "
              f"{-F_out.min():11.1f} N")
    ax.axhline(LOAD, color="grey", lw=0.8)
    ax.text(0.2, LOAD * 1.08, "quarter of Go1 weight", color="grey", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("foot sinkage (cm)")
    ax.set_ylabel("vertical mud force on foot (N)\nsolid: pushing in, dashed: pulling out")
    ax.set_title("Mud model: Go1-sized foot in Bangladeshi paddy clay")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig("docs/figures/mud_force_depth.png", dpi=150)
    print("Saved docs/figures/mud_force_depth.png")


def predicted_sinkage(p):
    """Depth where bearing + leg adhesion first equal the load (equations only)."""
    for d in np.linspace(1e-5, p.depth, 20000):
        if sum(bearing_capacity(d, p, R_FOOT, LEG_D)) >= LOAD:
            return d
    return p.depth   # reaches the hardpan


PROBE_XML = """
<mujoco><option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="hardpan" type="plane" size="1 1 0.1" pos="0 0 {hard}"/>
    <body name="foot" pos="0 0 {r}">
      <joint type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size="{r}" mass="{m}"/>
    </body>
  </worldbody></mujoco>"""


def drop_test():
    print("\nB. Sinkage under a quarter of the Go1 weight (31 N), simulated")
    print("   slow = load ramped over 30 s; fast = ramped over 1 s")
    print(f"{'season':15s} {'predicted':>10s} {'sim slow':>9s} {'sim fast':>9s}  note")
    for season in WET:
        p = MudParams.from_season(season)
        res = []
        for ramp in (30.0, 1.0):
            spec = mujoco.MjSpec.from_string(
                PROBE_XML.format(hard=-p.depth, r=R_FOOT, m=LOAD / 9.81))
            add_mud_plates(spec, ["ball"])
            m = spec.compile()
            d = mujoco.MjData(m)
            mud = MudContact(m, p, lambda x, y: 0.0, ["ball"], leg_d=LEG_D)
            for i in range(int((ramp + 2.0) / m.opt.timestep)):
                mud.apply(m, d)
                d.qfrc_applied[0] -= LOAD * min(i * m.opt.timestep / ramp, 1.0)
                mujoco.mj_step(m, d)
            res.append(mud.sinkage[0])
        pred = predicted_sinkage(p)
        note = "reaches hardpan" if pred >= p.depth - 1e-4 else ""
        print(f"{season:15s} {pred*100:8.2f} cm {res[0]*100:6.2f} cm {res[1]*100:6.2f} cm  {note}")


if __name__ == "__main__":
    print(f"Soil: {PADDY_CLAY.name}; LL {PADDY_CLAY.LL}%, PL {PADDY_CLAY.PL}%")
    curves()
    drop_test()
