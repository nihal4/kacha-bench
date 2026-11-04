"""Scene appearance: sky, sun and shadows, wet earth and mud, water, roadside plants.

Visual only. Nothing here collides or changes the physics, so videos look like a
monsoon village road without affecting training.
"""

import mujoco
import numpy as np

GREENS = [(0.16, 0.36, 0.12), (0.22, 0.45, 0.15), (0.12, 0.30, 0.10), (0.30, 0.50, 0.18)]


def _tex(spec, name, rgb1, rgb2, random=0.35, size=512, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT):
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=builtin,
                     mark=mujoco.mjtMark.mjMARK_RANDOM, random=random,
                     rgb1=list(rgb1), rgb2=list(rgb2), width=size, height=size)


def _mat(spec, name, tex=None, rgba=(1, 1, 1, 1), repeat=(1, 1), specular=0.1,
         shininess=0.1, reflectance=0.0):
    m = spec.add_material(name=name, rgba=list(rgba), texrepeat=list(repeat),
                          specular=specular, shininess=shininess, reflectance=reflectance)
    if tex:
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
    return m


def add_materials(spec, length, width):
    """Materials used by the terrain ("earth", "mud") and water ("water")."""
    # colours sampled by eye from field photos: dry earth is pale beige, wet clay a
    # grey-brown with a sheen, standing water grey and reflective
    _tex(spec, "earth_tex", (0.66, 0.57, 0.44), (0.58, 0.49, 0.37), random=0.08)
    _tex(spec, "mud_tex", (0.53, 0.47, 0.39), (0.46, 0.40, 0.32), random=0.05)
    rep = (length / 2.5, width / 2.5)
    _mat(spec, "earth", "earth_tex", repeat=rep, specular=0.05, shininess=0.05)
    # opaque, so feet that sink disappear into the mud as they do in reality
    _mat(spec, "mud", "mud_tex", repeat=rep, specular=0.6, shininess=0.75, reflectance=0.06)
    _mat(spec, "water", rgba=(0.40, 0.38, 0.33, 0.95), specular=1.0, shininess=1.0,
         reflectance=0.45)
    _mat(spec, "grass", rgba=(0.30, 0.42, 0.20, 1), specular=0.05, shininess=0.05)


def add_environment(spec, length, width, road_width, seed=0):
    """Sky, light, haze and roadside vegetation."""
    vis = spec.visual
    vis.global_.offwidth, vis.global_.offheight = 1280, 720
    vis.quality.shadowsize = 4096
    vis.quality.offsamples = 8
    vis.headlight.ambient = [0.25, 0.25, 0.25]
    vis.headlight.diffuse = [0.25, 0.25, 0.25]
    vis.headlight.specular = [0.0, 0.0, 0.0]
    vis.map.haze = 0.4
    vis.rgba.haze = [0.78, 0.82, 0.85, 1.0]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.70, 0.78], rgb2=[0.92, 0.93, 0.92], width=512, height=3072)
    wb = spec.worldbody
    # overcast monsoon sun: one shadow-casting key light plus a soft fill from the sky
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 pos=[-4, -3, 8], dir=[0.45, 0.35, -0.82], castshadow=True,
                 diffuse=[0.75, 0.72, 0.65], specular=[0.3, 0.3, 0.3], ambient=[0.15, 0.15, 0.15])
    wb.add_light(name="sky_fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 pos=[3, 4, 6], dir=[-0.3, -0.5, -0.8], castshadow=False,
                 diffuse=[0.25, 0.28, 0.32], specular=[0, 0, 0])
    # grass ground around and beyond the road patch, so the road does not end in a void
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], pos=[0, 0, -0.22],
                material="grass", contype=0, conaffinity=0, group=1)
    # a continuation of the road beyond the simulated patch (visual only)
    for sx in (-1, 1):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[6, road_width / 2, 0.16],
                    pos=[sx * (length / 2 + 6), 0, 0.14], material="mud", contype=0,
                    conaffinity=0, group=1)
    # dense bushes on both banks, beyond the road edges (visual only)
    rng = np.random.default_rng(seed + 7)
    for i in range(240):
        side = 1 if i % 2 else -1
        x = rng.uniform(-length / 2 - 4.0, length / 2 + 4.0)
        y = side * rng.uniform(road_width / 2 + 0.7, width / 2 + 1.5)
        r = rng.uniform(0.12, 0.45)
        c = GREENS[rng.integers(len(GREENS))]
        shade = rng.uniform(0.85, 1.15)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[r, r * rng.uniform(0.7, 1.2), r * rng.uniform(0.6, 1.4)],
                    pos=[x, y, r * 0.5], rgba=[min(c[0] * shade, 1), min(c[1] * shade, 1), min(c[2] * shade, 1), 1],
                    contype=0, conaffinity=0, group=1)
    # a few taller trees further back
    for i in range(14):
        side = 1 if i % 2 else -1
        x = rng.uniform(-length / 2, length / 2)
        y = side * rng.uniform(width / 2 + 0.8, width / 2 + 2.5)
        h = rng.uniform(2.5, 4.5)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.08, h / 2, 0], pos=[x, y, h / 2],
                    rgba=[0.33, 0.25, 0.17, 1], contype=0, conaffinity=0, group=1)
        c = GREENS[rng.integers(len(GREENS))]
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[1.0, 1.0, 0.8], pos=[x, y, h + 0.4],
                    rgba=[c[0], c[1], c[2], 1], contype=0, conaffinity=0, group=1)
