"""Water in potholes: buoyancy and drag on submerged robot parts, hidden pothole depth.

See docs/water_physics.md for sources and assumptions. Units: SI.

Water fills each pothole to just below its rim. The water is muddy, so its surface is
all a robot can "see": perceived_height() returns the water surface, not the floor.
Forces on every submerged collision shape of the robot:
  buoyancy  F = rho g V_submerged                       (Archimedes)
  drag      F = -1/2 rho Cd A |v_n| v_n                 (Morison drag term, still water)
"""

import mujoco
import numpy as np

RHO = 1000.0          # kg/m^3, fresh water (muddy water is slightly denser)
G = 9.81
CD_CYL = 1.0          # cylinder cross-flow drag, subcritical Re (GenTe uses 0.82-1.0)
CD_SPH = 0.47         # sphere drag, subcritical Re (standard textbook value)
N_SEG = 6             # segments per capsule/cylinder

SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)

# Fraction of each pothole filled with water, by season (ASSUMPTION, see docs).
SEASON_FILL = {"dry": 0.0, "early_monsoon": 0.6, "peak_monsoon": 1.0, "extreme": 1.0}
FREEBOARD = 0.01      # m, water surface below the rim when full


class Water:
    def __init__(self, potholes, fill=1.0, freeboard=FREEBOARD):
        self.pools = []
        for p in potholes:
            floor = p["rim"] - p["depth"]
            level = floor + fill * max(p["depth"] - freeboard, 0.0)
            if fill > 0 and level > floor + 1e-3:
                self.pools.append({"cx": p["center"][0], "cy": p["center"][1],
                                   "r": p["radius"], "level": level, "floor": floor})

    def level_at(self, x, y):
        """Water surface height at (x, y), or None if dry there."""
        for w in self.pools:
            if (x - w["cx"]) ** 2 + (y - w["cy"]) ** 2 < w["r"] ** 2:
                return w["level"]
        return None

    def perceived_height(self, x, y, ground):
        """What a height scanner would report: the water surface hides the floor."""
        lvl = self.level_at(x, y)
        return ground if lvl is None else max(ground, lvl)

    def add_visuals(self, spec):
        """Opaque muddy-water discs (visual only, no collision)."""
        for i, w in enumerate(self.pools):
            spec.worldbody.add_geom(
                name=f"water_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[w["r"], 0.002, 0], pos=[w["cx"], w["cy"], w["level"] - 0.002],
                rgba=[0.45, 0.36, 0.22, 0.95], contype=0, conaffinity=0, group=1)


def _submerged_cap(a, h):
    """Volume of a sphere of radius a submerged to depth h below its bottom."""
    h = np.clip(h, 0.0, 2 * a)
    return np.pi * h * h * (3 * a - h) / 3.0


class WaterForces:
    """Applies buoyancy and drag to a robot's collision geoms. Call after mud.apply()
    (which clears qfrc_applied) and before mj_step."""

    def __init__(self, model, water, geoms=None):
        self.water = water
        if geoms is None:
            geoms = [g for g in range(model.ngeom)
                     if model.geom_contype[g] and model.body_mocapid[model.geom_bodyid[g]] < 0
                     and model.geom_bodyid[g] != 0]
        self.geoms = geoms
        self._vel = np.zeros(6)
        self.last_buoyancy = 0.0
        self.last_drag = 0.0

    def _velocity(self, model, data, g, point):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, g, self._vel, 0)
        w, v = self._vel[:3], self._vel[3:]
        return v + np.cross(w, point - data.geom_xpos[g])

    def apply(self, model, data):
        self.last_buoyancy = self.last_drag = 0.0
        if not self.water.pools:
            return
        for g in self.geoms:
            c = data.geom_xpos[g]
            rb = model.geom_rbound[g]
            lvl = self.water.level_at(c[0], c[1])
            if lvl is None or c[2] - rb >= lvl:
                continue
            b = model.geom_bodyid[g]
            typ = int(model.geom_type[g])
            size = model.geom_size[g]
            if typ == SPHERE:
                a = size[0]
                V = _submerged_cap(a, lvl - (c[2] - a))
                frac = V / (4 / 3 * np.pi * a ** 3)
                v = self._velocity(model, data, g, c)
                F = np.array([0, 0, RHO * G * V]) \
                    - 0.5 * RHO * CD_SPH * np.pi * a * a * frac * np.linalg.norm(v) * v
                self._push(model, data, F, c, b)
            elif typ in (CAPSULE, CYLINDER):
                a, L = size[0], size[1]
                e = data.geom_xmat[g].reshape(3, 3)[:, 2]
                ds = 2 * L / N_SEG
                for s in np.linspace(-L + ds / 2, L - ds / 2, N_SEG):
                    p = c + e * s
                    if p[2] >= lvl:
                        continue
                    # fraction of this segment's cross-section under water
                    frac = np.clip((lvl - (p[2] - a)) / (2 * a), 0.0, 1.0)
                    v = self._velocity(model, data, g, p)
                    vn = v - np.dot(v, e) * e                      # cross-flow component
                    F = np.array([0, 0, RHO * G * np.pi * a * a * ds * frac]) \
                        - 0.5 * RHO * CD_CYL * (2 * a * ds * frac) * np.linalg.norm(vn) * vn
                    self._push(model, data, F, p, b)
            else:   # boxes (exact volume) and other shapes (bounding sphere); the
                    # submerged share is taken from the vertical extent (approximation)
                if typ == BOX:
                    V_full = 8.0 * size[0] * size[1] * size[2]
                else:
                    V_full = 4 / 3 * np.pi * rb ** 3
                frac = np.clip((lvl - (c[2] - rb)) / (2 * rb), 0.0, 1.0)
                v = self._velocity(model, data, g, c)
                A = V_full ** (2.0 / 3.0)
                F = np.array([0, 0, RHO * G * V_full * frac]) \
                    - 0.5 * RHO * CD_SPH * A * frac * np.linalg.norm(v) * v
                self._push(model, data, F, c, b)

    def _push(self, model, data, F, point, body):
        self.last_buoyancy += F[2] if F[2] > 0 else 0.0
        self.last_drag += np.linalg.norm(F[:2])
        mujoco.mj_applyFT(model, data, F, np.zeros(3), point, body, data.qfrc_applied)
