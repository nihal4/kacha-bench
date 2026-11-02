# Related work notes

Literature check done on 2 November 2026. Not exhaustive; a full Google Scholar search is still needed before writing the paper.

## Closest work

### GenTe: Generative Real-world Terrains (2025), arXiv:2504.09997
- Generates simulation terrains from text or images using vision-language models; 100-terrain benchmark (50 from text, 50 from images).
- Physical terrain: wading water (drag, added mass, buoyancy, flow force) and deformable sand (friction + bulldozing resistance, Vanderkop et al. model).
- Isaac Gym, bipedal robot, PPO with curriculum.
- Stated future work: real-world deployment and more realistic physical terrain models.
- **Gaps relevant to us:** water has no hidden depth or potholes; soft ground is sand-like, with no clay cohesion or suction; scenes are generic, not region-grounded; no seasonal or structured generalisation splits; no comparison of controller types; biped only.

### MILD: Tractable Terrain Modeling for Bipedal Locomotion on Deformable Surfaces (RA-L 2026)
- Discrete-element contact solver for spatially varying yielding terrain; hardware validation.
- Focus is physics accuracy, not benchmark or scene design.

## Deformable and granular terrain
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*, Science Robotics 2020: robust to mud, snow and water without modelling them in training.
- Choi et al., *Learning Quadrupedal Locomotion on Deformable Terrain*, Science Robotics 2023 (KAIST): granular media model for RL, beach sand to asphalt.
- GM-Loco, *Terrain-Adaptive Humanoid Locomotion on Granular Media* (2026), arXiv:2609.10286: 3D resistive force theory model, validated against MPM; humanoid.
- Godon et al., *Walking in Mud*, IEEE/ASME TMECH 2025: Go1 in real 12 cm deep mud.
- CMU Robomechanics Lab: ongoing project on learned simulation for quadrupeds in mud.

## Benchmarks and evaluation
- Zhang et al., *Generating a Terrain-Robustness Benchmark for Legged Locomotion*, ICRA 2023: rigid height-map terrains.
- Review (Univ. of Brescia): very few papers rigorously test legged locomotion on irregular terrain; no agreed protocols or metrics.

## Simulation platform
- MuJoCo Playground (Zakka et al., 2025), arXiv:2502.08844: MJX locomotion environments for Go1, Spot, Barkour, Berkeley Humanoid, H1, G1, T1, OP3.
- MuJoCo Menagerie: robot models.

## Not found
- No work modelling South Asian rural roads, waterlogged potholes with hidden depth, or monsoon-season variation for legged locomotion.
