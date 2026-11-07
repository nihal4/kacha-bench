"""Reduced-order wet-clay mud model for legged feet.

Every relation comes from published soil mechanics; see docs/mud_physics.md for the
sources, the assumptions and the limitations. Units: SI (m, N, Pa, m/s).

Simulation design: the rigid height field is the firm base (hardpan) and a mud layer
of thickness H sits on top of it. Under each foot an invisible "support plate" marks
the floor of the hole that foot has pushed. MuJoCo's contact solver holds the foot on
the plate (stable at any stiffness); when the contact force exceeds the mud's bearing
capacity, the plate is moved down. That is plastic sinkage, as in terramechanics
simulators. Sideways resistance is set through the plate friction, and suction and
drag during swing are explicit forces with a damping gain kept below the stability
limit of the integrator.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

SU_LL = 1.7e3      # Pa, undrained shear strength at the liquid limit (Sharma & Bora 2003)
SU_RATIO = 100.0   # strength at PL / strength at LL (Sharma & Bora 2003; Sharma & Sridharan 2018)
MU_MAX = float(__import__("os").environ.get("KACHA_MU_MAX", 1.0))   # cap on the sole-ground friction coefficient (G1 flat foot); was 2.0 until 2026-11-07 (docs/mud_physics.md, section 9)
MU_WET_CLAY = 0.3  # surface friction of very wet clay (Birch et al. 2016: tan 16-28 deg)


@dataclass
class Soil:
    name: str
    LL: float   # liquid limit, % water
    PL: float   # plastic limit, % water


# Measured Bangladeshi soils
PADDY_CLAY = Soil("Bangladeshi paddy-field sandy fat clay (Hasnat et al. 2020)", 54.48, 29.68)
DHAKA_CLAY = Soil("Dhaka soft organic clay, mid-range (Islam et al. 2004)", 83.0, 38.0)

# Seasons are defined by liquidity index LI = (w - PL)/(LL - PL) and mud-layer depth.
# These values are ASSUMPTIONS until field moisture is measured (docs/mud_physics.md).
SEASONS = {
    "dry":           {"LI": -0.2, "depth": 0.00},
    "early_monsoon": {"LI": 0.70, "depth": 0.08},
    "peak_monsoon":  {"LI": 1.00, "depth": 0.15},
    "extreme":       {"LI": 1.20, "depth": 0.20},
}


def liquidity_index(soil, w):
    return (w - soil.PL) / (soil.LL - soil.PL)


def water_content(soil, LI):
    return soil.PL + LI * (soil.LL - soil.PL)


def shear_strength(LI):
    """Remoulded undrained shear strength (Pa); log-linear in water content between
    1.7 kPa at the liquid limit (LI = 1) and 170 kPa at the plastic limit (LI = 0)."""
    return SU_LL * SU_RATIO ** (1.0 - LI)


def breakout_ratio(LI, beta_max=0.6, lo=0.1, hi=1.4):
    """Fraction of bearing capacity resisting lift-off (suction). Zero when the soil is
    firm (LI <= lo) or nearly liquid (LI >= hi), largest in between, following the
    trend measured by Godon et al. (2025). beta_max < 1 because centrifuge tests rarely
    reach tension = compression (ASCE 1990). Shape and peak are assumptions."""
    x = np.clip((LI - lo) / (hi - lo), 0.0, 1.0)
    return beta_max * np.sin(np.pi * x)


@dataclass
class MudParams:
    su: float              # Pa
    depth: float           # m, mud-layer thickness above the hardpan
    beta: float            # suction (breakout) ratio
    alpha: float = 1.0     # leg-soil adhesion factor, ~1 for soft clay
    E_ratio: float = 300.0 # undrained Young's modulus / su (soft clays ~100-1500)
    J: float = 0.5         # Matlock lateral-resistance depth coefficient
    mu_surface: float = MU_WET_CLAY

    @classmethod
    def from_season(cls, season, soil=PADDY_CLAY):
        s = SEASONS[season]
        return cls(su=shear_strength(s["LI"]), depth=s["depth"], beta=breakout_ratio(s["LI"]))


def contact_area(d, r):
    """Horizontal area of a sphere of radius r pushed d into the ground."""
    d = np.clip(d, 0.0, r)
    return np.pi * (2.0 * r * d - d * d)


def bearing_capacity(d, p, r, leg_d):
    """(base, shaft) forces (N) the mud supports with the foot bottom d below the surface."""
    D = 2.0 * r
    Nc = np.minimum(6.0 * (1.0 + 0.2 * d / D), 9.0)          # Skempton (1951), circular
    F_base = Nc * p.su * contact_area(d, r)
    shaft = np.maximum(d - D, 0.0)                            # leg length buried above foot
    F_shaft = p.alpha * p.su * np.pi * leg_d * shaft          # adhesion along the leg
    return F_base, F_shaft


def lateral_capacity(d, p, r):
    """Sideways resistance (N) of a foot buried d deep (Matlock 1970, at mid-depth)."""
    D = 2.0 * r
    Np = np.minimum(3.0 + p.J * (d / 2.0) / D, 9.0)
    return Np * p.su * D * d


def punch_stiffness(p, r):
    """Elastic stiffness (N/m) of a rigid circular punch on clay: k = 2 E r / (1 - nu^2),
    undrained nu = 0.5, E = E_ratio * su."""
    return 2.0 * p.E_ratio * p.su * r / 0.75


def quasi_static_curve(depths, p, r, leg_d):
    """Analytic push-in then pull-out force curve for docs/figures (no simulation)."""
    F_in = [sum(bearing_capacity(d, p, r, leg_d)) for d in depths]
    zp = depths[-1]
    F_b, _ = bearing_capacity(zp, p, r, leg_d)
    F_out = [-(p.beta * F_b * d / max(zp, 1e-9)
               + p.alpha * p.su * np.pi * leg_d * max(d - 2 * r, 0.0)) for d in depths[::-1]]
    return np.array(F_in), np.array(F_out)


# --------------------------------------------------------------------- MuJoCo coupling

def add_mud_plates(spec, foot_geoms):
    """Add one invisible mocap support plate per foot to an MjSpec, each colliding only
    with its own foot. Call before spec.compile()."""
    for k, name in enumerate(foot_geoms):
        bit = 1 << (k + 1)
        foot = spec.geom(name)
        foot.contype = int(foot.contype) | bit
        body = spec.worldbody.add_body(name=f"plate_{name}", mocap=True, pos=[0, 0, -10])
        body.add_geom(name=f"plate_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.05, 0.05, 0.01], contype=0, conaffinity=bit,
                      priority=5, friction=[MU_WET_CLAY, 0.02, 0.01],
                      rgba=[0, 0, 0, 0], group=3)


class MudContact:
    """Mud for a set of feet. Call apply(model, data) before every mj_step."""

    HALF_T = 0.01   # plate half-thickness (m)

    def __init__(self, model, params, surface_fn, foot_geoms, leg_d=0.02):
        self.p = params
        self.surface_fn = surface_fn
        self.feet = [model.geom(n).id for n in foot_geoms]
        self.plates = [model.geom(f"plate_{n}").id for n in foot_geoms]
        self.mocap = [model.body_mocapid[model.body(f"plate_{n}").id] for n in foot_geoms]
        self.bodies = [model.geom_bodyid[g] for g in self.feet]
        self.radius = [float(model.geom_size[g][0]) for g in self.feet]
        self.leg_d = leg_d
        # stable explicit damping gain: c * dt / m_body <= 0.5
        self.c_max = [0.5 * model.body_subtreemass[b] / model.opt.timestep for b in self.bodies]
        n = len(self.feet)
        dt = model.opt.timestep
        self.filt = dt / (0.02 + dt)       # 20 ms low-pass on the contact force
        self.dz_max = 0.1 * dt             # yield at most 0.1 m/s (keeps foot on its plate)
        self.N_avg = np.zeros(n)
        self.zp = np.zeros(n)              # plastic sinkage (hole-floor depth) per foot
        self.hole_xy = np.zeros((n, 2))
        self.sinkage = np.zeros(n)
        self.normal = np.zeros(n)
        self._vel = np.zeros(6)
        self._f = np.zeros(6)

    def _plate_force(self, model, data, k):
        total = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if self.plates[k] in (c.geom1, c.geom2):
                mujoco.mj_contactForce(model, data, i, self._f)
                total += self._f[0]
        return total

    def apply(self, model, data):
        data.qfrc_applied[:] = 0.0
        p = self.p
        for k, (g, b, r) in enumerate(zip(self.feet, self.bodies, self.radius)):
            pos = data.geom_xpos[g]
            surface = self.surface_fn(pos[0], pos[1])
            d = surface - (pos[2] - r)          # depth of the foot bottom below the surface
            self.sinkage[k] = max(d, 0.0)
            N = self._plate_force(model, data, k)
            self.N_avg[k] += self.filt * (N - self.N_avg[k])
            self.normal[k] = N

            if p.depth <= 0.0:
                data.mocap_pos[self.mocap[k]] = [pos[0], pos[1], -10.0]
                continue
            if d <= 0.0:                        # out of the mud: the hole closes
                self.zp[k] = 0.0
                self.hole_xy[k] = pos[:2]
            elif N <= 0.0 and np.linalg.norm(pos[:2] - self.hole_xy[k]) > r:
                self.zp[k] = min(d, p.depth)    # moved sideways into unyielded mud
                self.hole_xy[k] = pos[:2]

            if N > 0.0:                         # yield if the (smoothed) load exceeds capacity
                cap = sum(bearing_capacity(self.zp[k], p, r, self.leg_d))
                if self.N_avg[k] > cap:
                    dz = (self.N_avg[k] - cap) / punch_stiffness(p, r)
                    self.zp[k] = min(self.zp[k] + min(dz, self.dz_max), p.depth)
                self.hole_xy[k] = pos[:2]

            data.mocap_pos[self.mocap[k]] = [pos[0], pos[1], surface - self.zp[k] - self.HALF_T]

            depth_in = min(max(d, 0.0), p.depth)
            Fh_cap = lateral_capacity(depth_in, p, r)
            model.geom_friction[self.plates[k], 0] = float(
                np.clip(max(p.mu_surface, Fh_cap / max(N, 1.0)), p.mu_surface, 2.0))

            if d > 0.0 and N <= 0.0:            # foot inside the mud but off its plate
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, g, self._vel, 0)
                v = self._vel[3:]
                F = np.zeros(3)
                if v[2] > 0.0 and np.linalg.norm(pos[:2] - self.hole_xy[k]) <= r:
                    F_b, _ = bearing_capacity(self.zp[k], p, r, self.leg_d)
                    cap = p.beta * F_b * depth_in / max(self.zp[k], 1e-6) \
                        + p.alpha * p.su * np.pi * self.leg_d * max(depth_in - 2 * r, 0.0)
                    F[2] = -min(self.c_max[k] * v[2], cap)          # suction
                vh = v[:2]
                sp = np.linalg.norm(vh)
                if sp > 1e-9:
                    F[:2] = -min(self.c_max[k] * sp, Fh_cap) * vh / sp   # drag in swing
                mujoco.mj_applyFT(model, data, F, np.zeros(3), pos, b, data.qfrc_applied)


# ----------------------------------------------------------- flat feet (humanoids)
#
# A humanoid sole is a flat rectangle, not a sphere, so its full area bears from the
# first millimetre of sinkage. Same soil strength su as above; only the footing
# geometry changes (docs/mud_physics.md, section "Flat feet").

def rect_bearing_capacity(d, p, B, L, frac=1.0, leg_d=0.07, foot_h=0.04):
    """(base, sides) forces (N) the mud supports under a flat B x L sole whose bottom is
    d below the surface. Skempton (1951) for a rectangular footing on clay:
    Nc = 5 (1 + 0.2 D/B)(1 + 0.2 B/L), with D/B capped at 2.5. frac is the share of
    the sole touching (heel or toe only during roll-off)."""
    Nc = 5.0 * (1.0 + 0.2 * min(d / B, 2.5)) * (1.0 + 0.2 * B / L)
    F_base = Nc * p.su * B * L * frac
    side = p.alpha * p.su * 2.0 * (B + L) * min(max(d, 0.0), foot_h)     # sole edges
    side += p.alpha * p.su * np.pi * leg_d * max(d - foot_h, 0.0)          # ankle / shin
    return F_base, side


def rect_lateral_capacity(d, p, B, L):
    """Sideways resistance (N) of a sole buried d deep: adhesion under the sole plus
    passive resistance on its front face (Matlock 1970 factor, face width B, the
    narrow side, which is the one pushed when a walking foot slips forward or back)."""
    Np = min(3.0 + p.J * (d / 2.0) / B, 9.0)
    return p.alpha * p.su * B * L + Np * p.su * B * d


def rect_punch_stiffness(p, B, L):
    """Elastic stiffness of a rigid rectangular punch, using the equal-area circle."""
    return punch_stiffness(p, np.sqrt(B * L / np.pi))


def add_flat_mud_plates(spec, feet):
    """One invisible mocap plate per foot, sized to the sole, colliding only with that
    foot's contact geoms. feet: {foot_name: [geom names]}. Call before compile."""
    for k, (foot, geoms) in enumerate(feet.items()):
        bit = 1 << (k + 1)
        for gname in geoms:
            g = spec.geom(gname)
            g.contype = int(g.contype) | bit
        body = spec.worldbody.add_body(name=f"plate_{foot}", mocap=True, pos=[0, 0, -10])
        body.add_geom(name=f"plate_{foot}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.15, 0.08, 0.01], contype=0, conaffinity=bit, priority=5,
                      friction=[MU_WET_CLAY, 0.02, 0.01], rgba=[0, 0, 0, 0], group=3)


class FlatFootMud:
    """Mud under flat feet. Same scheme as MudContact (a yielding support plate under
    each foot, plate friction for sideways resistance, explicit suction and swing drag),
    with the sole geometry of a humanoid. Call apply(model, data) before every mj_step.

    feet: {foot_name: [contact geom names]}, all geoms of one foot on one body."""

    HALF_T = 0.01

    def __init__(self, model, params, surface_fn, feet, leg_d=0.07, foot_h=0.04):
        self.p, self.surface_fn, self.leg_d, self.foot_h = params, surface_fn, leg_d, foot_h
        self.names = list(feet)
        self.geoms = [[model.geom(g).id for g in gs] for gs in feet.values()]
        self.bodies = [int(model.geom_bodyid[gs[0]]) for gs in self.geoms]
        self.plates = [model.geom(f"plate_{f}").id for f in self.names]
        self.mocap = [model.body_mocapid[model.body(f"plate_{f}").id] for f in self.names]
        # sole size from the contact points (sphere centres plus their radius)
        self.B, self.L, self.r_pt = [], [], []
        for gs in self.geoms:
            pos, r = model.geom_pos[gs], float(model.geom_size[gs[0]][0])
            self.L.append(float(np.ptp(pos[:, 0]) + 2 * r))
            self.B.append(float(np.ptp(pos[:, 1]) + 2 * r))
            self.r_pt.append(r)
        dt = model.opt.timestep
        self.c_max = [0.5 * model.body_subtreemass[b] / dt for b in self.bodies]
        self.filt = dt / (0.02 + dt)
        self.dz_max = 0.1 * dt
        n = len(self.names)
        self.N_avg, self.zp = np.zeros(n), np.zeros(n)
        self.hole_xy = np.zeros((n, 2))
        self.sinkage, self.normal, self.touching = np.zeros(n), np.zeros(n), np.zeros(n)
        self.suction = np.zeros(n)          # downward pull on a lifting foot (N), last step
        self._f, self._vel = np.zeros(6), np.zeros(6)

    def _plate_contacts(self, model, data, k):
        N, pts = 0.0, set()
        for i in range(data.ncon):
            c = data.contact[i]
            if self.plates[k] in (c.geom1, c.geom2):
                mujoco.mj_contactForce(model, data, i, self._f)
                N += self._f[0]
                pts.add(c.geom1 if c.geom2 == self.plates[k] else c.geom2)
        return N, len(pts)

    def apply(self, model, data):
        data.qfrc_applied[:] = 0.0
        p = self.p
        self.suction[:] = 0.0
        for k in range(len(self.names)):
            gs, b, B, L = self.geoms[k], self.bodies[k], self.B[k], self.L[k]
            pts = data.geom_xpos[gs]
            centre = pts.mean(axis=0)
            bottom = float(pts[:, 2].min()) - self.r_pt[k]
            surface = self.surface_fn(centre[0], centre[1])
            d = surface - bottom
            self.sinkage[k] = max(d, 0.0)
            N, n_pts = self._plate_contacts(model, data, k)
            frac = max(n_pts / len(gs), 0.25)
            self.N_avg[k] += self.filt * (N - self.N_avg[k])
            self.normal[k], self.touching[k] = N, n_pts
            if p.depth <= 0.0:
                data.mocap_pos[self.mocap[k]] = [centre[0], centre[1], -10.0]
                continue
            far = np.linalg.norm(centre[:2] - self.hole_xy[k]) > 0.5 * B
            if d <= 0.0:
                self.zp[k], self.hole_xy[k] = 0.0, centre[:2]
            elif N <= 0.0 and far:
                self.zp[k], self.hole_xy[k] = min(d, p.depth), centre[:2]
            if N > 0.0:
                cap = sum(rect_bearing_capacity(self.zp[k], p, B, L, frac, self.leg_d, self.foot_h))
                if self.N_avg[k] > cap:
                    dz = (self.N_avg[k] - cap) / rect_punch_stiffness(p, B, L)
                    self.zp[k] = min(self.zp[k] + min(dz, self.dz_max), p.depth)
                self.hole_xy[k] = centre[:2]
            # plate: flat, under the sole, turned with the foot's heading
            R = data.xmat[b].reshape(3, 3)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            data.mocap_pos[self.mocap[k]] = [centre[0], centre[1], surface - self.zp[k] - self.HALF_T]
            data.mocap_quat[self.mocap[k]] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            depth_in = min(max(d, 0.0), p.depth)
            Fh = rect_lateral_capacity(depth_in, p, B, L)
            model.geom_friction[self.plates[k], 0] = float(
                np.clip(max(p.mu_surface, Fh / max(N, 1.0)), p.mu_surface, MU_MAX))
            if d > 0.0 and N <= 0.0:           # in the mud, lifting or swinging
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, self._vel, 0)
                v, F = self._vel[3:], np.zeros(3)
                if v[2] > 0.0 and not far:
                    F_b, side = rect_bearing_capacity(self.zp[k], p, B, L, 1.0, self.leg_d, self.foot_h)
                    cap = p.beta * F_b * depth_in / max(self.zp[k], 1e-6) + side
                    F[2] = -min(self.c_max[k] * v[2], cap)                    # suction
                    self.suction[k] = -F[2]
                sp = np.linalg.norm(v[:2])
                if sp > 1e-9:
                    F[:2] = -min(self.c_max[k] * sp, Fh) * v[:2] / sp          # drag
                mujoco.mj_applyFT(model, data, F, np.zeros(3), centre, b, data.qfrc_applied)
