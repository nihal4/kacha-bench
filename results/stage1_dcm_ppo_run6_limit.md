# Run 6: where the working range ends (LI 0.82 to 0.9), 2026-11-07

PC, smooth road, base peak_monsoon (15 cm), 20 roads per LI (seeds 70000+k), time limit 300 s, no noise. Run after the contact fix
(dry on plates, friction cap 1.0). Run 6 = `g1-road-peak_monsoon-dcm-smooth-wet0.4-dr-max0.85-spd-t300-h5-s0`.

| LI | su (kPa) | DCM | Run 6 | Run 6 distance (m) | Run 6 cost of transport | Falls |
|---|---|---|---|---|---|---|
| 0.80 (earlier, cap 2.0) | 4.3 | 0% | 100% | 8.30 | 1.58 | - |
| 0.82 | 3.9 | 0% (0-16) | 0% (0-16) | 1.63 | 9.09 | slipped 100% |
| 0.85 | 3.4 | 0% | 0% | -0.03 | 17.76 | slipped 100% |
| 0.87 | 3.0 | 0% | 0% | -0.16 | - | slipped 95%, stuck 5% |
| 0.90 | 2.7 | 0% | 0% | -0.16 | - | sank 85%, slipped 15% |

Reading:

- The edge is sharp: 100% at LI 0.8, 0% at 0.82 (one step of 0.02). The cost of transport jumps from 1.6 to 9.1 and the walk covers
  1.6 m before the foot slides out. The falls up to 0.87 are slips (the sideways grip of the soil falls below what a step needs; the
  model's grip is about 0.25 at LI 0.8 and 0.2 at 0.85, with a floor of 0.3), sinking starts at 0.9.
- The earlier LI 0.8 result (100%) was measured with friction cap 2.0; this table used cap 1.0. Whether the edge moved with the cap
  is not known. To check: LI 0.75, 0.78, 0.8, 0.82 for both caps (`KACHA_MU_MAX=2.0` and `1.0`).
- The slow plain DCM (0.1 m/s, 6 roads) also crossed LI 0.8 and fell at 0.85 (`dcm_slow_walking_check.md`), so for this
  controller family the wall sits between LI 0.8 and 0.85, close to the bearing-capacity estimate (LI about 0.86 for 343 N on
  a 9 x 22 cm sole; assumed foot and equations).
- Target check: the aim was at least 90% from LI 0.5 to 0.85. Met to LI 0.8; not met at 0.82 and 0.85.
