# Run 6 on the corrected dry ground (2026-11-07)

After the contact fix of `docs/mud_physics.md` section 9 (dry ground on a plate like mud roads; plate friction cap 1.0). PC, run 6
(`g1-road-peak_monsoon-dcm-smooth-wet0.4-dr-max0.85-spd-t300-h5-s0`), final `model.zip`, smooth road, 20 roads per season (road
seeds 50000+k), time limit 300 s for both methods, no assist, no observation noise.

| Season | DCM | Run 6 | Run 6 before the fix (bare floor) |
|---|---|---|---|
| dry | 100%, 43.6 s | 95% (19/20), 49.5 s, 1 fall (no_trigger) | 0% |
| early monsoon | 100%, 45.0 s | 95% (19/20), 49.5 s, 1 fall (slipped) | 100% (51.7 s) |

Dry split (`g1_trace.py --ground dry`, 10 roads, 300 s):

| Mode | Success | Ended at (s) |
|---|---|---|
| full policy | 100% | 52.0 |
| speed choice only (leg corrections zeroed) | 100% | 53.5 |
| leg corrections only, speed 0.15 / 0.20 | 100% / 50% | 49.4 / 23.5 |
| plain DCM 0.15 / 0.20 | 100% / 100% | 54.6 / 43.7 |

Reading:

- The dry failures were the bare-floor contact artifact: the same policy goes from 0% to 95% to 100% on dry ground without retraining.
- Early monsoon dropped from 100% to 95% (one slip in 20, 95% interval 76 to 99) and the DCM's time changed from 45.4 to 45.0 s in that
  season. These roads (LI 0.70, 8 cm) should not be touched by a friction cap of 1.0 if grip stayed below 1, so the transient
  friction demand (touchdown impacts) apparently exceeds 1.0 at times. One slip is not significant, but the cap effect on early
  monsoon is to be checked: the same evaluation with `KACHA_MU_MAX=2.0` should give back 100% and 51.7 s if the cap is the cause.
