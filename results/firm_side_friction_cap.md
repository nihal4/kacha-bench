# Firm-mud artifact: plate friction cap 2.0 makes very firm mud unwalkable for DCM (2026-11-07)

Found while looking at why run 6 fails on dry ground. Plain DCM, smooth road, 15 cm mud layer, 10 roads per LI, 300 s limit,
no noise (scratch environment). The G1 flat-foot mud model sets the plate friction to `max(0.3, lateral capacity / load)`
capped at `MU_MAX` (`kacha/mud.py`; was a fixed 2.0, now `KACHA_MU_MAX`, default still 2.0).

| LI | su (kPa) | DCM, MU_MAX 2.0 | DCM, MU_MAX 1.0 |
|---|---|---|---|
| -0.2 | 170 | 0% (falls at once, -0.20 m) | 100% |
| 0.0 | 170 | 0% | - |
| 0.3 | 27 | 0% (-0.09 m) | 100% |
| 0.4 | 27 | 100% | 100% |
| 0.5 | 17 | 100% | 100% |

Bare dry ground (rigid floor, MuJoCo default friction 1.0): plain DCM 90% on the first 10 roads here (65% on 20 roads earlier,
40 to 60% on other 10-road sets in the PC diagnostic), so dry results vary a lot between road sets; n = 100 is needed.

Reading: firmer ground should never be harder, so the plain DCM failing on LI <= 0.3 mud is a model artifact: grip above about 1
makes the foot catch at touchdown. Foot-ground friction of rubber on firm soil is usually below 1 (assumption, to be checked
against literature). Affects only firm mud: the lateral grip is about 1.0 at LI 0.5 and lower for wetter mud, so results at
LI >= 0.5 do not change. Run 6 trained on LI 0.4 and up, so its firm training roads had grip between 1 and 2 while dry ground has 1.0;
a possible reason the policy's leg corrections fail on dry ground (hypothesis, test below).
