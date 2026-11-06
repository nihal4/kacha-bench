# Mud physics in KachaBench

This document explains how wet-clay mud is modelled, where every equation and number
comes from, how the implementation was checked, and what is still an assumption.
Literature review done on 2 November 2026. Code: `kacha/mud.py`. Test: `scripts/mud_probe.py`.

## 1. Approach

Exact simulation of mud (particle methods such as MPM or DEM) is far too slow for
reinforcement learning. KachaBench uses a **reduced-order model** built from classical
soil mechanics for saturated clay under undrained (fast) loading. A walking robot loads
the ground in a fraction of a second, so the water in clay cannot drain and the
undrained shear strength `su` controls the response.

The model has one central idea: **the moisture of the soil sets its strength, and the
strength sets every force on the foot.** The season changes the moisture.

## 2. Strength from moisture

Liquidity index: `LI = (w - PL) / (LL - PL)`, where `w` is water content and `LL`, `PL`
are the liquid and plastic limits.

Remoulded undrained shear strength (Sharma & Bora 2003; Sharma & Sridharan 2018):

```
su = 1.7 kPa x 100^(1 - LI)
```

This gives 1.7 kPa at the liquid limit (LI = 1) and 170 kPa at the plastic limit
(LI = 0), log-linear in between. Both anchor values and the log-linear form are reported
from fall-cone data on many soils (Sharma & Bora 2003); Sharma & Sridharan (2018) cite a
range of 0.7 to 3 kPa at the liquid limit across studies.

## 3. Forces on a foot

The foot is a sphere of radius `r` (Go1: 0.023 m), diameter `D = 2r`, pushed a depth `d`
below the mud surface.

| Force | Equation | Source |
|---|---|---|
| Bearing (vertical) | `F = Nc su A(d)`, `Nc = 6(1 + 0.2 d/D) <= 9`, `A(d) = pi (2 r d - d^2)` for `d < r` | Skempton (1951), circular footing on clay |
| Leg adhesion | `F = alpha su pi D_leg (d - D)`, `alpha = 1` | adhesion factor ~1 for soft clay |
| Sideways resistance | `F = Np su D d`, `Np = 3 + 0.5 z/D <= 9` (z = d/2) | Matlock (1970), lateral load on piles in soft clay |
| Suction on lift-off | `F = beta Nc su A(zp) (d/zp)` + leg adhesion | trend from Godon et al. (2025); `beta < 1` from ASCE (1990) |
| Surface friction | `mu >= 0.3` on very wet clay | Birch et al. (2016) |
| Elastic stiffness | `k = 2 E r / (1 - nu^2)`, `nu = 0.5`, `E = 300 su` | rigid circular punch; E/su ratio typical of soft clay |

Suction ratio `beta` is zero when the soil is firm (LI <= 0.1) or nearly liquid
(LI >= 1.4) and largest in between (peak 0.6). This follows the measured trend that
suction on a Go1 foot peaks at intermediate water content and vanishes when mud is very
dry or very wet (Godon et al. 2025), and the centrifuge result that breakout force
approaches but rarely reaches compressive capacity (ASCE 1990). **The exact shape and
peak value are assumptions.**

## 4. How it is simulated

The rigid height field is the firm base (hardpan). A mud layer of thickness `H` sits on
top of it, drawn as a see-through surface.

- **Vertical support:** under each foot an invisible plate marks the floor of the hole
  the foot has pushed (plastic sinkage `zp`). MuJoCo's contact solver holds the foot on
  the plate, which is numerically stable at any stiffness. Each step the contact force
  is low-pass filtered (20 ms); if it exceeds the bearing capacity at `zp`, the plate
  moves down by `(N - capacity)/k`, at most 0.1 m/s.
- **Sideways:** the plate's friction is set each step so that the friction limit equals
  the lateral capacity of the buried foot, but never below 0.3.
- **Suction and drag during swing:** explicit forces, capped at the capacities above,
  with a damping gain kept under the integrator's stability limit (`c dt / m <= 0.5`).
- **Holes:** a foot that leaves the mud resets its hole (the mud flows back); a foot
  that moves sideways into undisturbed mud starts a new hole at its current depth.

A first version applied all mud forces explicitly as a spring. It was unstable: the Go1
lower leg weighs about 0.2 kg, so `omega dt` was 1.6 and the foot chattered and
ratcheted down to the hardpan. The plate design replaced it.

## 5. Bangladeshi soil data used

| Soil | LL (%) | PL (%) | Natural w (%) | su | Source |
|---|---|---|---|---|---|
| Paddy-field sandy fat clay (default) | 54.5 | 29.7 | 43.1 | UCS 264 kPa (compacted) | Hasnat et al. (2020) |
| Dhaka soft organic clay | 72–94 | 30–46 | 46–83 | 3–19 kPa | Islam et al. (2004) |
| Coastal rural-road shoulders | lean clay | | | | Alam et al. (2020) |

Consistency check: Dhaka clay with LL 83, PL 38 and w 65 gives LI 0.6 and, from
Section 2, su = 11 kPa, inside the measured 3–19 kPa range.

## 6. Seasons (assumption)

| Season | LI | su (kPa) | Mud depth (m) |
|---|---|---|---|
| Dry | -0.2 | (rigid, no mud layer) | 0 |
| Early monsoon | 0.7 | 6.8 | 0.08 |
| Peak monsoon | 1.0 | 1.7 | 0.15 |
| Extreme | 1.2 | 0.7 | 0.20 |

**I did not find published field measurements of surface moisture on Bangladeshi
village roads by season.** The LI values and mud depths are reasoned choices, not
measurements. Mud depth 0.12 m is the "deep mud" setting in the Go1 experiments of
Godon et al. (2025). Measuring surface water content (and Atterberg limits) of real
village roads in each season would turn this into a strong claim; a fall-cone test can
measure su directly.

## 7. Verification (`scripts/mud_probe.py`)

A Go1-sized foot (r = 0.023 m) carrying a quarter of the Go1's weight (31 N), paddy clay:

| Season | Predicted sinkage (equations) | Simulated, slow load | Simulated, 1 s load |
|---|---|---|---|
| Early monsoon | 0.60 cm | 0.60 cm | 0.60 cm |
| Peak monsoon | 10.64 cm | 10.67 cm | 10.68 cm |
| Extreme | 20.0 cm (hardpan) | 20.0 cm | 19.0 cm (still sinking at 0.1 m/s cap) |

Go1 standing (motors holding the home pose, 4 s, `scripts/go1_on_road.py --season ...`,
seed 0): early monsoon 0.7–1.0 cm sinkage; peak monsoon 8–15 cm; extreme ~20 cm (on the
hardpan). Inside the training environment (4 ms physics step, Kp 35) the results are
similar: early ~1 cm, peak 7–15 cm, extreme 20 cm.

This verifies that **the simulation does what the equations say**. It does not yet
validate the equations against measured mud; see Section 8.

## 8. Limitations and open questions

1. **No direct validation against measured foot forces yet.** The Godon et al. (2025)
   paper has measured Go1 force–sinkage curves; reproducing their soil and comparing
   is the obvious next check. Their full paper could not be retrieved here.
2. **Seasonal moisture and mud depth are assumptions** (Section 6).
3. **Rate effects are left out.** Clay strength usually rises with loading speed, but
   Godon et al. observed the opposite trend in mud (slower intrusion gave more force).
   With conflicting evidence, the model is rate-independent.
4. **Viscosity is left out.** Locat & Demers (1988) report plastic viscosity of only
   0.2 to 0.007 Pa·s for liquidity index 2 to 5; extrapolating to LI 1 gives about
   2.5 Pa·s, which would resist a Go1 foot by well under 1 N.
5. **Suction shape (`beta`) is an assumption** fitted only to a qualitative trend.
6. **Numerical choices:** the 0.1 m/s maximum sinking speed, the 20 ms force filter
   and the damping cap are for stability, not physics. Their effect should be reported
   in a sensitivity study.
7. **Holes do not persist** after a foot leaves the mud, and the mud surface does not
   deform visually.
8. **MJX port:** the plate friction is changed per step through the model, which MJX
   batches cannot do per environment; this part will need a different implementation
   on GPU.

## Sources

- Sharma, B. & Bora, P. K. (2003). Plastic limit, liquid limit and undrained shear strength of soil: reappraisal. *J. Geotech. Geoenviron. Eng.* 129(8). https://ascelibrary.com/doi/10.1061/(ASCE)1090-0241(2003)129:8(774)
- Sharma, B. & Sridharan, A. (2018). Liquid and plastic limits of clays by cone method. *Int. J. Geo-Engineering* 9:22. https://link.springer.com/content/pdf/10.1186/s40703-018-0092-0.pdf
- Skempton, A. W. (1951). The bearing capacity of clays. *Building Research Congress*, London. (Values as commonly tabulated; original not retrieved.)
- Matlock, H. (1970). Correlations for design of laterally loaded piles in soft clay. *Offshore Technology Conference*, OTC 1204. (Original not retrieved.)
- Extraction forces for offshore foundations under undrained loading (1990). *J. Geotech. Eng.* 116(5). https://ascelibrary.com/doi/10.1061/%28ASCE%290733-9410%281990%29116%3A5%28868%29
- Godon, S., Prados, C., Ristolainen, A. & Kruusmaa, M. (2025). Walking in mud: modeling, control, and experiments of quadruped locomotion. *IEEE/ASME Trans. Mechatronics*. https://hal-lirmm.ccsd.cnrs.fr/lirmm-05103366v1 ; thesis summary: https://digikogu.taltech.ee/et/Download/be217d0d-89b7-45c8-8a4c-1a27f953812f
- Birch, R. A., Ekwue, E. I. & Phillip, C. J. (2016). Soil-metal sliding resistance forces of some Trinidadian soils at high water contents. *West Indian J. Eng.* 38(2). https://sta.uwi.edu/eng/wije/vol3802_jan2016/documents/Vol38No2ManRABirchJan2016.pdf
- Locat, J. & Demers, D. (1988). Viscosity, yield stress, remolded strength, and liquidity index relationships for sensitive clays. *Can. Geotech. J.* 25. https://trid.trb.org/View/290691
- Hasnat, A. et al. (2020). Improvement of bearing capacity of clay soil using fly ash. *AIUB J. Sci. Eng.* 19(2). https://ajse.aiub.edu/index.php/ajse/article/download/85/68
- Islam, M. S., Siddique, A. & Muqtadir, A. (2004). Mechanical properties of soft organic Dhaka clay. *J. Civil Eng. (IEB)* 32(2). https://mail.jce-ieb.org/doc_file/ce320205.pdf
- Alam, M. J., Karim, M. E. & Hoque, M. S. (2020). Causes of damage of rural road in coastal areas of Bangladesh. *Geotech. Eng. J. SEAGS & AGSSEA* 51(4). https://ph01.tci-thaijo.org/index.php/SEAGS_AGSSEA_Journal/article/view/258681

## 8. What the model does to a G1 foot as the soil gets wetter (design review, 2026-11-06)

Raised by the author: in heavy monsoon the mud is sticky and the leg sinks and gets stuck (water-filled holes too), it does
not simply get slippery like a wet tile floor. Computed from the model's own equations (assumed sole 9 x 22 cm, 343 N on one
foot, 15 cm mud layer; `rect_bearing_capacity`, `rect_lateral_capacity`, `breakout_ratio`). Foot size and the equations are
assumptions; the figures show what the model says, not what a real road does.

| LI | su (kPa) | Sinkage to carry the weight | Sideways grip (N) and as friction coefficient | Suction capacity (N, share of weight) |
|---|---|---|---|---|
| 0.5 | 17.0 | 0 | 337 N, 1.0 | 899 N, 2.6 |
| 0.7 | 6.8 | 0 | 134 N, 0.39 | 432 N, 1.3 |
| 0.8 | 4.3 | 0 | 85 N, 0.25 (floor 0.3) | 272 N, 0.8 |
| 0.85 | 3.4 | 0 | 67 N, 0.2 (floor 0.3) | 212 N, 0.6 |
| 0.9 | 2.7 | 2.4 cm | 71 N | 211 N, 0.6 |
| 1.0 | 1.7 | to the hardpan (15 cm) | 112 N | 203 N, 0.6 |

Reading (model behaviour):

- From LI 0.5 to 0.85 the mud acts as a firm but increasingly slippery surface: no sinkage, grip falling from about 1.0 to
  the 0.3 floor. This is the "more monsoon, more slippery" behaviour, and it is a consequence of su falling about 5 times
  while the foot still sits on top. It is soil failing in shear under a sliding foot, not a low-friction film.
- The foot sinks only from LI 0.88. Suction is largest in the firm band (relative to weight) and falls with su, so heavy
  mud has less suction capacity in absolute terms, and "stuck" is rare in the sweeps (0 to 5% of falls).
- Whether real heavy mud sticks more, and from which wetness the foot sinks, is not validated. Likely weak points: suction
  proportional to su with no time or rate dependence, a single peak value (beta_max 0.6, assumed), no slurry-over-firm-base layering.
- Water-filled holes with hidden depth exist in the model (`water.py`) but the smooth road used for the baseline has none,
  and the rutted road fails at the start for geometry reasons, so holes have not been tested yet.
