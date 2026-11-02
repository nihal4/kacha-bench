# Water-filled potholes in KachaBench

How potholes and the water in them are modelled, where the numbers come from, how the
implementation was checked, and what is still an assumption. Literature review done on
2 November 2026. Code: `kacha/terrain.py` (potholes), `kacha/water.py` (water).
Tests: `scripts/water_probe.py`, `scripts/go1_pothole.py`.

## 1. Potholes: sized by the LGED rural-road guideline

The Local Government Engineering Department's guideline for evaluating rural paved roads
(LGED 2025) classifies potholes by depth and diameter:

| Severity | Depth | Diameter | Used in KachaBench |
|---|---|---|---|
| Minor | < 50 mm | < 300 mm | 20–50 mm, 150–300 mm |
| Moderate | 50–100 mm | 300–500 mm | 50–100 mm, 300–500 mm |
| Severe | > 100 mm | > 500 mm | 100–200 mm, 500–800 mm |

LGED gives only lower bounds for "severe"; the upper bounds (200 mm deep, 800 mm across)
are our choice. The default mix is 40% minor, 40% moderate, 20% severe (assumption).
Potholes have a flat floor and steep walls (outer 30% of the radius), so stepping in is
a sudden drop, not a gentle bowl. The terrain grid is 4 cm, so the smallest pothole
(15 cm) spans about four cells.

The same guideline gives rut severity for paved roads (minor < 10 mm, moderate 10–20 mm,
severe > 20 mm) and carriageway widths from 3.0 m (design type 8) to 7.3 m. KachaBench
uses a 3.0 m road and 50 mm wheel ruts, deeper than the paved-road "severe" class,
because earthen roads rut more; LGED gives no rut classes for earthen roads.

## 2. Hidden depth

Rain fills each pothole to just below its rim (1 cm freeboard). Village-road water is
muddy and opaque, so a height scanner or camera sees only the water surface.
`Water.perceived_height()` returns the water surface over a pothole, not its floor. The
robot can learn the true depth only by stepping in. Below the water, the pothole floor
is covered by the season's mud layer (`docs/mud_physics.md`).

Fill by season (**assumption**): dry 0%, early monsoon 60%, peak monsoon 100%,
extreme 100%.

## 3. Forces from the water

Water is treated as still, fresh water (density 1000 kg/m³; muddy water is slightly
denser). On every submerged collision shape of the robot:

| Force | Equation | Source |
|---|---|---|
| Buoyancy | `F = rho g V_submerged` (exact spherical-cap volume for feet; segment-wise for legs) | Archimedes |
| Drag on legs | `F = -1/2 rho Cd D ds |v_n| v_n` per segment, `v_n` = velocity across the leg axis, `Cd = 1.0` | Morison drag term; GenTe uses Cd 0.82–1.0 for the same regime |
| Drag on feet | `F = -1/2 rho Cd A |v| v`, `Cd = 0.47` | standard sphere drag at subcritical Reynolds number |

Reynolds number of a 2 cm Go1 leg moving at 0.5 m/s is about 10,000, in the subcritical
range where cylinder drag is roughly constant, so a constant `Cd` is justified.

**Added mass is left out.** For a circular cylinder the added-mass coefficient is 1 in
potential flow (lower in real oscillating flow; SNU lecture notes after Sarpkaya &
Isaacson 1981). For a Go1 calf it would add about 0.04 kg to a 0.2 kg leg; it needs
the leg's acceleration and was omitted to keep the force explicit and stable.

## 4. Verification

`scripts/water_probe.py` measures the computed force on Go1-sized parts with gravity off
and compares it with a hand calculation:

| Case | Computed | Hand calculation |
|---|---|---|
| Buoyancy, leg half submerged | 0.2034 N | 0.2034 N |
| Buoyancy, leg fully submerged | 0.4068 N | 0.4068 N |
| Buoyancy, foot half submerged | 0.2500 N | 0.2500 N |
| Drag, leg moving sideways at 0.5 m/s | 0.3300 N | 0.3300 N |
| Drag, leg moving along its axis | 0.0000 N | 0 |

A first version had a type-comparison bug that treated legs as large bounding spheres
(forces 44 times too large); this test caught it.

## 5. What the scenario shows (`scripts/go1_pothole.py`, seed 0)

The Go1 holds its standing pose with its front-right foot over the deepest pothole
(severe: 18.6 cm deep, 50 cm across):

| Season | Scanner sees vs real floor | FR foot drop | Result |
|---|---|---|---|
| Dry | no water, floor visible | 1.7 cm (foot hangs over the hole) | stays upright on three legs |
| Early monsoon | 9.6 cm of water hides the floor | 12.9 cm | tips over |
| Peak monsoon | 16.6 cm of water hides the floor | 16.0 cm | tips over |

**Main finding:** the water's own forces on a walking-size robot are small (leg
buoyancy and drag below 1 N each, against 31 N of weight per leg). The hazard is the
**hidden depth** combined with the soft mud: the foot drops where the robot expects
ground. A controller that only holds a pose cannot recover; a learned policy will have
to.

## 6. Limitations

1. Pothole size limits for "severe" and the severity mix are our choices.
2. Seasonal fill levels are assumptions; no field data found.
3. Water is still: no flow, waves or splash; the water surface does not deform.
4. Added mass is omitted (Section 3).
5. Body shapes other than spheres, capsules and cylinders (the Go1 trunk box) use a
   simplified submerged fraction from their vertical extent.
6. Water and mud are separate layers; real pothole water carries suspended sediment,
   which raises its density a little and makes the mud below it softer.

## Sources

- LGED (2025). *Guideline on Effectiveness and Performance Evaluation of Rural Paved Roads of LGED*, with Islamic University of Technology. https://roadsforwater.org/wp-content/uploads/2025/08/Guideline-on-Effectiveness-and-Performance-Evaluation-of-Rural-Paved-Roads-of-LGED-V2-28-April-2025-1.pdf
- Wan, H. et al. (2025). GenTe: Generative real-world terrains for general legged robot locomotion control. arXiv:2504.09997. https://arxiv.org/pdf/2504.09997
- Seoul National University OCW, Chapter 8: Wave forces (Morison equation, added mass), citing Sarpkaya & Isaacson (1981), *Mechanics of Wave Forces on Offshore Structures*. https://ocw.snu.ac.kr/sites/default/files/NOTE/8946.pdf
- Morison equation overview. https://en.wikipedia.org/wiki/Morison_equation
