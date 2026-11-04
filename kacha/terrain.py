"""Height-field generators for rural village-road terrain.

All functions work on a 2D array of heights in metres. The x axis runs along
the road and the y axis across it.
"""

from dataclasses import dataclass

import numpy as np


# Pothole severity classes for rural roads, LGED (2025) guideline: depth and diameter (m).
# LGED gives only lower bounds for "severe"; the upper bounds here are our choice.
POTHOLE_CLASSES = {
    "minor":    {"depth": (0.02, 0.05), "diameter": (0.15, 0.30)},
    "moderate": {"depth": (0.05, 0.10), "diameter": (0.30, 0.50)},
    "severe":   {"depth": (0.10, 0.20), "diameter": (0.50, 0.80)},
}


@dataclass
class RoadConfig:
    length: float = 10.0          # metres along the road (x)
    width: float = 6.0            # metres across the whole patch (y)
    resolution: float = 0.04      # metres per cell (smallest LGED pothole, 15 cm, spans ~4 cells)
    road_width: float = 3.0       # LGED smallest rural-road carriageway (design type 8)
    embankment_height: float = 0.3  # road surface above the surrounding ground
    ditch_depth: float = 0.2      # extra depth of the ditches beside the road
    rut_depth: float = 0.05       # earthen-road wheel ruts (LGED "severe" on paved is > 20 mm)
    rut_spacing: float = 1.0      # distance between the two wheel ruts
    n_potholes: int = 4
    pothole_mix: tuple = (0.4, 0.4, 0.2)   # share of minor, moderate, severe
    roughness: float = 0.01       # std of small surface bumps (m)
    roughness_scale: float = 0.10 # bump size (m): Gaussian correlation length
    rut_sigma: float = 0.08       # rut half-width (m, Gaussian std)
    rut_wobble: float = 0.0       # sideways meander of the ruts (m)
    extra_ruts: int = 0           # more, shallower tracks at random places across the road
    rut_puddles: int = 0          # small water-filled dips along the ruts


# Road surfaces for the study: "rutted" (default, the monsoon road) or "smooth" (the same
# embankment road with no ruts, dips, potholes or surface roughness), to separate the
# effect of the mud from the effect of the road surface.
SMOOTH_ROAD = dict(rut_depth=0.0, extra_ruts=0, rut_puddles=0, n_potholes=0, roughness=0.0)
ROADS = {"rutted": {}, "smooth": SMOOTH_ROAD}


def muddy_rut_road(**kw):
    """Preset for a monsoon earthen road torn up by wheels (deep, meandering ruts with
    water standing in them, as in field photos of Bangladeshi village roads). The rut
    depth, meander and puddle counts are visual estimates, not measurements."""
    base = dict(rut_depth=0.08, rut_sigma=0.11, rut_wobble=0.12, extra_ruts=3,
                rut_puddles=8, n_potholes=2, roughness=0.015)
    base.update(kw)
    return RoadConfig(**base)


def _grid(cfg):
    nx = int(round(cfg.length / cfg.resolution)) + 1
    ny = int(round(cfg.width / cfg.resolution)) + 1
    x = np.linspace(-cfg.length / 2, cfg.length / 2, nx)
    y = np.linspace(-cfg.width / 2, cfg.width / 2, ny)
    return np.meshgrid(x, y, indexing="ij")


def embankment(X, Y, cfg):
    """Raised road with sloped sides dropping into ditches."""
    half = cfg.road_width / 2
    side = np.abs(Y) - half
    slope_w = 0.6
    h = np.full_like(X, cfg.embankment_height)
    sloped = (side > 0) & (side < slope_w)
    h[sloped] = cfg.embankment_height * (1 - side[sloped] / slope_w)
    outside = side >= slope_w
    h[outside] = 0.0
    ditch = outside & (side < slope_w + 0.8)
    h[ditch] -= cfg.ditch_depth
    return h


def rut_lines(cfg, rng):
    """Rut centre lines as (offset, depth, sigma, wobble phase, wobble wavelength)."""
    ruts = [(c, cfg.rut_depth, cfg.rut_sigma, rng.uniform(0, 2 * np.pi), rng.uniform(4, 8))
            for c in (-cfg.rut_spacing / 2, cfg.rut_spacing / 2)]
    half = cfg.road_width / 2 - 0.3
    for _ in range(cfg.extra_ruts):
        ruts.append((rng.uniform(-half, half), cfg.rut_depth * rng.uniform(0.3, 0.7),
                     cfg.rut_sigma * rng.uniform(0.6, 1.0), rng.uniform(0, 2 * np.pi),
                     rng.uniform(3, 7)))
    return ruts


def rut_centre(r, x, cfg):
    c, _, _, ph, lam = r
    return c + cfg.rut_wobble * np.sin(2 * np.pi * x / lam + ph)


def add_ruts(h, X, Y, cfg, rng=None):
    """Smooth grooves where cart and van wheels run; optionally meandering, plus
    shallower extra tracks. Returns (h, rut lines)."""
    rng = rng or np.random.default_rng(0)
    ruts = rut_lines(cfg, rng)
    for r in ruts:
        h -= r[1] * np.exp(-((Y - rut_centre(r, X, cfg)) ** 2) / (2 * r[2] ** 2))
    return h, ruts


def add_rut_puddles(h, X, Y, cfg, ruts, rng):
    """Shallow dips along the ruts where rain water stands. Returned in the pothole
    format, so kacha.water fills them like potholes."""
    out = []
    for _ in range(cfg.rut_puddles):
        r = ruts[rng.integers(len(ruts))]
        cx = rng.uniform(-cfg.length / 2 + 1.0, cfg.length / 2 - 0.5)
        cy = rut_centre(r, cx, cfg)
        rad, dep = rng.uniform(0.15, 0.30), rng.uniform(0.02, 0.05)
        dist = np.sqrt(((X - cx) / 1.6) ** 2 + (Y - cy) ** 2)      # stretched along the road
        inside = dist < rad
        ring = (dist >= rad) & (dist < rad + 2 * cfg.resolution)
        if not ring.any():
            continue
        rim = float(np.min(h[ring]))
        t = np.clip((rad - dist) / (0.5 * rad), 0.0, 1.0)
        h -= dep * (t * t * (3 - 2 * t)) * inside
        out.append({"center": (cx, cy), "radius": rad, "depth": dep, "severity": "rut_puddle",
                    "rim": rim})
    return h, out


def add_potholes(h, X, Y, cfg, rng):
    """Steep-walled potholes (flat floor, walls over the outer 30% of the radius),
    sized by the LGED severity classes. Returns the pothole list; each records its
    rim height, which is where rain water fills it to."""
    holes = []
    half = cfg.road_width / 2
    names = list(POTHOLE_CLASSES)
    for _ in range(cfg.n_potholes):
        sev = names[rng.choice(3, p=np.array(cfg.pothole_mix) / sum(cfg.pothole_mix))]
        c = POTHOLE_CLASSES[sev]
        r = rng.uniform(*c["diameter"]) / 2
        d = rng.uniform(*c["depth"])
        cx = rng.uniform(-cfg.length / 2 + 1.0, cfg.length / 2 - 1.0)
        cy = rng.uniform(-half + r, half - r)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        inside = dist < r
        rim = float(np.min(h[(dist >= r) & (dist < r + 2 * cfg.resolution)]))
        t = np.clip((r - dist) / (0.3 * r), 0.0, 1.0)            # 0 at rim, 1 on floor
        h -= d * (t * t * (3 - 2 * t)) * inside                    # smooth steep wall
        holes.append({"center": (cx, cy), "radius": r, "depth": d,
                      "severity": sev, "rim": rim})
    return h, holes


def smooth_noise(shape, sigma_cells, rng):
    """Spatially correlated noise (unit std): white noise blurred by a Gaussian."""
    n = rng.normal(0.0, 1.0, shape)
    r = max(int(3 * sigma_cells), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sigma_cells, 1e-6)) ** 2)
    k /= k.sum()
    n = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, n)
    n = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, n)
    return n / (n.std() + 1e-12)


def generate_village_road(cfg=None, seed=0):
    """Generate a village-road height field.

    Returns (heights, X, Y, info) where heights is in metres and info lists the
    potholes, which later become hidden-depth water regions.
    """
    cfg = cfg or RoadConfig()
    rng = np.random.default_rng(seed)
    X, Y = _grid(cfg)
    h = embankment(X, Y, cfg)
    on_road = np.abs(Y) <= cfg.road_width / 2
    # ruts draw from their own generator, so roads from earlier versions (same seed,
    # default config) keep their pothole positions
    rutted, ruts = add_ruts(h.copy(), X, Y, cfg, np.random.default_rng([seed, 1]))
    h = np.where(on_road, rutted, h)
    h, holes = add_potholes(h, X, Y, cfg, rng)
    if cfg.rut_puddles:
        h, puddles = add_rut_puddles(h, X, Y, cfg, ruts, rng)
        holes += puddles
    # surface roughness correlated over ~10 cm (cell-by-cell noise makes tiny spikes,
    # which look wrong and create many contacts per foot)
    h += cfg.roughness * smooth_noise(h.shape, cfg.roughness_scale / cfg.resolution, rng)
    return h, X, Y, {"potholes": holes, "config": cfg}
