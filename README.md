# KachaBench

**A procedurally generated rural monsoon-road terrain benchmark for legged-robot locomotion with reinforcement learning, built on MuJoCo and MJX.**

> *Kacha rasta* (কাঁচা রাস্তা) is Bangla for an unpaved earthen road. These are the roads that connect most villages in Bangladesh, and for several months each year the monsoon turns them into mud, ruts and water-filled potholes.

**Status:** early development. The terrain generator is a prototype; no results yet.

---

## Motivation

Legged-robot locomotion is usually trained and tested on rigid terrain: flat ground, stairs, slopes, and random height maps. Recent work has started to model deformable and wet ground (sand, snow, wading water), but the test scenes are generic and nobody has modelled the terrain of rural South Asian roads in the monsoon season.

These roads combine several hazards at once:

- **Wet clay mud:** the foot sinks, slips, and is held by suction when it lifts.
- **Waterlogged potholes:** the water surface hides how deep the hole is, so a robot cannot see the hazard and must infer it by touch.
- **Raised embankment roads:** narrow earthen roads with ditches or ponds on either side.
- **Wheel ruts:** grooves left by rickshaws and vans, which trap feet and tilt the body.
- **Herringbone brick roads:** paved with bricks, often broken or missing.
- **Bamboo bridges (*sãko*):** narrow, uneven and slippery.
- **Seasonal change:** the same road behaves differently in the dry season, early monsoon and peak monsoon.

## Research questions

1. How well do standard locomotion controllers (MLP, recurrent, height-map-based) handle hidden-depth and sticky-mud hazards that existing benchmarks do not model?
2. Do policies trained on dry-season and early-monsoon conditions generalise to peak-monsoon conditions they have not seen?
3. How do quadrupeds and humanoids differ on the same terrain?

## How this differs from existing work

| | GenTe (2025) | MILD (2026) | **KachaBench** |
|---|---|---|---|
| Focus | Generic terrain generation from text/images | Accurate deformable contact model | Region-grounded benchmark and evaluation protocol |
| Water | Wading: drag, buoyancy, flow | — | Water-filled potholes with **hidden depth** |
| Soft ground | Sand-style (friction + bulldozing) | Discrete-element yielding terrain | **Wet clay** with sinkage, slip and **suction** |
| Scenes | 100 VLM-generated generic terrains | — | Procedural village-road layouts |
| Generalisation test | Unseen generated terrains | — | **Seasonal held-out splits** |
| Robots | Biped | Biped | Quadruped + humanoids |
| Simulator | Isaac Gym | Custom solver | MuJoCo / **MJX** (free GPU) |

See [`docs/related_work.md`](docs/related_work.md) for notes on the literature.

## Physics approach

Exact mud and water physics (MPM/DEM particle simulation) is far too slow for RL training. KachaBench uses **physics-grounded reduced-order models**, as GenTe, KAIST and others do:

- **Geometry:** height fields for roads, ruts, potholes, embankments and ditches.
- **Mud:** soft contact settings plus extra forces for sinkage, bulldozing resistance and lift-off suction.
- **Water:** potholes sized by the LGED (2025) rural-road guideline fill with opaque water that hides their depth; submerged robot parts feel buoyancy and Morison drag. See [`docs/water_physics.md`](docs/water_physics.md).

The mud model is built from classical soil mechanics: mud strength comes from soil moisture (liquidity index), and strength sets the bearing, sideways and suction forces on each foot, using measured properties of Bangladeshi clays. Full derivation, sources, verification and limitations: [`docs/mud_physics.md`](docs/mud_physics.md). These are approximations, and the paper will say so.

## Planned benchmark design

- **Robots:** Unitree Go1 (quadruped, used first to validate the environment), Berkeley Humanoid, Unitree G1.
- **Difficulty:** each terrain element has parameters (mud depth, puddle density, pothole size, road width, brick damage) grouped into difficulty levels for curriculum training.
- **Seasons:** dry, early monsoon, peak monsoon.
- **Splits:** train on dry + early monsoon, test on peak monsoon and on held-out terrain combinations.
- **Metrics:** success rate, survival time, velocity-tracking error, energy per metre, and fall rate.
- **Baselines:** PPO with MLP, PPO with GRU (memory), PPO with height-map input, and domain randomisation, each over multiple seeds.

## Compute plan

The same terrain code runs in standard MuJoCo (CPU) and MJX (GPU).

| Task | Where |
|---|---|
| Environment building, physics tuning, visual debugging | Local CPU (MuJoCo) |
| Short sanity training runs | Local CPU |
| Main training runs (all seeds and robots) | Kaggle GPU (MJX) |
| Evaluation, videos, analysis | Local CPU |

## Repository layout

```
kacha-bench/
├── kacha/                 # Python package
│   ├── terrain.py         # Height-field generators (road, ruts, potholes, embankment)
│   ├── scene.py           # Builds a MuJoCo model from a generated terrain
│   ├── robot_scene.py     # Puts a Menagerie robot (Go1) on the terrain, with optional mud layer
│   ├── mud.py             # Wet-clay mud model (soil mechanics) and its MuJoCo coupling
│   └── water.py           # Water-filled potholes: hidden depth, buoyancy and drag
├── scripts/
│   ├── smoke_test.py      # Builds a terrain, drops a test body, checks the simulation runs
│   ├── go1_on_road.py     # Go1 standing on the road or in mud for a season, renders a video
│   ├── mud_probe.py       # Foot-drop probe test: mud forces vs the equations, plus figure
│   ├── water_probe.py     # Water forces on Go1-sized parts vs hand calculations
│   └── go1_pothole.py     # Go1 with one foot over a water-filled pothole, per season
├── docs/
│   ├── related_work.md    # Literature notes and positioning
│   ├── mud_physics.md     # Mud model: equations, sources, verification, limitations
│   ├── water_physics.md   # Potholes and water: sources, verification, limitations
│   └── figures/           # Generated figures
├── requirements.txt
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/nihal4/kacha-bench.git
cd kacha-bench
pip install -e .
python scripts/smoke_test.py            # headless check
python scripts/smoke_test.py --view     # opens the MuJoCo viewer
```

## Roadmap

- [x] Repository and project description
- [x] Prototype height-field generator (road, ruts, potholes, embankment)
- [x] Go1 standing on the village road (`scripts/go1_on_road.py`, renders a video headless with `MUJOCO_GL=egl`)
- [x] Mud contact model (sinkage, sideways resistance, suction) from soil mechanics, verified against its equations (`scripts/mud_probe.py`)
- [x] Go1 standing in mud for each season (`scripts/go1_on_road.py --season peak_monsoon`)
- [ ] Validate the mud model against measured Go1 foot-force data (Godon et al. 2025)
- [x] Potholes sized by the LGED rural-road guideline; water-filled potholes with hidden depth, buoyancy and drag (`docs/water_physics.md`, `scripts/water_probe.py`, `scripts/go1_pothole.py`)
- [ ] Brick road and bamboo bridge elements
- [ ] Seasonal parameter sets and difficulty levels
- [ ] Go1 locomotion environment and first PPO baseline
- [ ] Physics parameter grounding and validation check
- [ ] MJX port and Kaggle training notebook
- [ ] Humanoid environments (Berkeley Humanoid, G1)
- [ ] Full baseline sweep and evaluation protocol

## Author

S. M. Nihal Ahmed, Department of Software Engineering, Daffodil International University, Dhaka, Bangladesh.

## Licence

MIT. See [`LICENSE`](LICENSE). Robot models from MuJoCo Menagerie keep their own licences.
