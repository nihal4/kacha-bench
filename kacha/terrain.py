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
    resolution: float = 0.025     # metres per cell
    road_width: float = 3.0       # LGED smallest rural-road carriageway (design type 8)
    embankment_height: float = 0.3  # road surface above the surrounding ground
    ditch_depth: float = 0.2      # extra depth of the ditches beside the road
    rut_depth: float = 0.05       # earthen-road wheel ruts (LGED "severe" on paved is > 20 mm)
    rut_spacing: float = 1.0      # distance between the two wheel ruts
    n_potholes: int = 4
    pothole_mix: tuple = (0.4, 0.4, 0.2)   # share of minor, moderate, severe
    roughness: float = 0.01       # small random bumps on the surface


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


def add_ruts(h, Y, cfg):
    """Two smooth grooves where cart and van wheels run."""
    for c in (-cfg.rut_spacing / 2, cfg.rut_spacing / 2):
        h -= cfg.rut_depth * np.exp(-((Y - c) ** 2) / (2 * 0.08 ** 2))
    return h


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
    h = np.where(on_road, add_ruts(h.copy(), Y, cfg), h)
    h, holes = add_potholes(h, X, Y, cfg, rng)
    h += rng.normal(0, cfg.roughness, h.shape)
    return h, X, Y, {"potholes": holes, "config": cfg}
