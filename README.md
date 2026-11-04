# KachaBench

**A robustness study of humanoid locomotion on rural monsoon roads: when, how and why a walking humanoid fails on wet clay, ruts and water-hidden potholes. Built on MuJoCo, with mud physics from soil mechanics and Bangladeshi clay data.**

> *Kacha rasta* (কাঁচা রাস্তা) is Bangla for an unpaved earthen road. These are the roads that connect most villages in Bangladesh, and for several months each year the monsoon turns them into mud, ruts and water-filled potholes.

**Status:** early development. Humanoid stage 1 (Unitree G1 crossing a muddy rutted road) is built, with failure-cause recording and robustness sweeps; no results yet. Study plan: [`docs/study.md`](docs/study.md).

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

The project is a robustness study rather than a leaderboard: a few controllers, each differing in one way, tested in depth across controlled mud conditions ([`docs/study.md`](docs/study.md)).

1. **Breakdown:** at what soil strength, mud depth, friction, suction and water level does humanoid walking break down, and how sharply?
2. **Mechanism:** why does it fail: slips, sinking, suction trapping a foot, hidden-depth water, or balance loss, and how does the mix change with the mud?
3. **Training conditions:** does a policy trained on rigid ground (Unitree's) or on one season hold up in another?
4. **Load:** does carrying a box in the hands or a load on the head bring the breakdown earlier?
5. **Gait adaptation:** do trained policies adapt their gait to mud the way people do on slippery ground?

## How this differs from existing work

| | GenTe (2025) | MILD (2026) | **KachaBench** |
|---|---|---|---|
| Focus | Generic terrain generation from text/images | Accurate deformable contact model | Region-grounded robustness study: when and why humanoids fail on mud |
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

## Study design

Full plan: [`docs/study.md`](docs/study.md).

- **Robot:** Unitree G1 humanoid ([`docs/humanoid.md`](docs/humanoid.md)). The Unitree Go1 quadruped was used first to validate the mud and water models.
- **Conditions:** the muddy rutted road, with one mud property varied at a time (soil wetness, mud depth, friction, suction, water in holes) across its plausible range, so results do not rest on one assumed value per season.
- **Load (later):** none, a box in both hands, and a load on the head steadied with one hand plus a box in the other, as station porters carry.
- **Controllers:** Unitree's pretrained rigid-ground policy, the scripted mud gait alone, and PPO policies trained on the dry road, early-monsoon mud, mixed seasons, and with memory, 3 seeds each.
- **Measurements:** success with confidence intervals; the trigger of every fall (slip, sinking, suction trap, hidden water, balance loss) and how it fell; gait (step length, cadence, double support, foot lift, sinkage, sliding, cost of transport).

## Compute plan

Training currently runs on an always-on 4-core Linux PC (CPU MuJoCo + Stable-Baselines3 PPO), controlled with the `kacha` command and watched on a web dashboard. See [`docs/training.md`](docs/training.md). An MJX (GPU) port for Kaggle is planned for larger sweeps.

| Task | Where |
|---|---|
| Environment building, physics tuning, tests | Local CPU (MuJoCo) |
| Training runs | Always-on PC (`kacha start`, dashboard on port 8081) |
| Large multi-seed sweeps (later) | Kaggle GPU (MJX port) |

## Repository layout

```
kacha-bench/
├── kacha/                 # Python package
│   ├── terrain.py         # Height-field generators (road, ruts, potholes, embankment)
│   ├── scene.py           # Builds a MuJoCo model from a generated terrain
│   ├── robot_scene.py     # Puts a Menagerie robot (Go1) on the terrain, with optional mud layer
│   ├── mud.py             # Wet-clay mud model (soil mechanics) and its MuJoCo coupling
│   ├── water.py           # Water-filled potholes: hidden depth, buoyancy and drag
│   ├── walk_env.py        # Go1 walking environment (Gymnasium), seasons, height scan
│   ├── g1_scene.py        # Unitree G1 humanoid on the road (contacts, Unitree PD gains)
│   ├── gait.py            # Scripted mud-walking gait (leg IK) and balance controller
│   ├── g1_env.py          # Humanoid stage 1: cross the muddy rutted road; failure causes, gait metrics
│   ├── g1_policies.py     # Controllers for evaluation: scripted, trained, Unitree pretrained
│   ├── look.py            # Sky, sun and shadows, wet clay, water, roadside plants
│   └── safety.py          # RAM / CPU-temperature guards for long training runs
├── scripts/
│   ├── smoke_test.py      # Builds a terrain, drops a test body, checks the simulation runs
│   ├── go1_on_road.py     # Go1 standing on the road or in mud for a season, renders a video
│   ├── mud_probe.py       # Foot-drop probe test: mud forces vs the equations, plus figure
│   ├── water_probe.py     # Water forces on Go1-sized parts vs hand calculations
│   ├── go1_pothole.py     # Go1 with one foot over a water-filled pothole, per season
│   ├── g1_preview.py      # G1 on the muddy road with the scripted controller, video
│   ├── g1_evaluate.py     # G1 per-season table: scripted, trained runs, Unitree's policy
│   ├── g1_sweep.py        # Robustness curves: one mud property at a time, with fall causes
│   ├── train.py           # PPO training (profiles, safe checkpoints, --resume)
│   ├── record.py          # Film a trained policy on three fixed test roads
│   └── desktop/           # Always-on PC: setup.sh, kacha command, services, dashboard
├── docs/
│   ├── related_work.md    # Literature notes and positioning
│   ├── mud_physics.md     # Mud model: equations, sources, verification, limitations
│   ├── water_physics.md   # Potholes and water: sources, verification, limitations
│   ├── training.md        # Environment, reward, PPO settings, PC workflow
│   ├── study.md           # Robustness study: questions, factors, controllers, measurements
│   ├── humanoid.md        # G1 tasks, flat-foot mud, scripted gait, stage 1 reward
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
- [x] Go1 walking environment (Playground Go1 joystick reward) and PPO training pipeline with dashboard (`docs/training.md`)
- [ ] First Go1 walking policy on the rigid road
- [ ] Physics parameter grounding and validation check
- [ ] MJX port and Kaggle training notebook
- [x] G1 humanoid stage 1 (muddy rutted road): flat-foot mud model, scripted mud-walking gait, residual PPO with balance-assist curriculum (`docs/humanoid.md`)
- [x] Failure-cause recording, gait metrics and robustness sweeps (`docs/study.md`)
- [ ] G1 stage 1 policies trained (dry, early monsoon, mix; 3 seeds each)
- [ ] Robustness curves and failure mix for stage 1
- [ ] G1 stage 2 (box in both hands) and stage 3 (head load + hand box)
- [ ] Uphill earthen road and muddy steps
- [ ] Full baseline sweep and evaluation protocol

## Author

S. M. Nihal Ahmed, Department of Software Engineering, Daffodil International University, Dhaka, Bangladesh.

## Licence

MIT. See [`LICENSE`](LICENSE). Robot models from MuJoCo Menagerie keep their own licences.
