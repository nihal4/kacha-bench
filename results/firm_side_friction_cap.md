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

## Run 6 with the cap at 2.0 and at 1.0 (PC, 20 roads per LI, smooth road, 300 s), 2026-11-07

| LI | MU_MAX 2.0 | MU_MAX 1.0 |
|---|---|---|
| 0.4 | 100%, cost of transport 0.85 | 100%, 0.88 |
| 0.5 | 100%, 0.84 | 100%, 0.87 |
| 0.6 | 100%, 0.86 | 100%, 0.89 |

Result: run 6 does not depend on the extra grip (success and gait are the same, cost of transport 3 to 4% higher). The hypothesis
that its dry failures come from having trained on grip between 1 and 2 is not supported. The cap itself remains a model artifact
for plain DCM at LI <= 0.3 (table above) and should be lowered when the model is next revised; it does not affect results at
LI >= 0.4 for this policy.

Open for dry ground: the model's "dry" path has no mud plates at all (rigid floor, default contact), while every mud road has
plates (soft, damped contact). Run 6 crosses plate-covered ground from LI 0.4 up but fails on the bare floor. Next test: run 6 on very firm
mud with plates (LI -0.2 to 0.3, cap 1.0), which has grip like dry ground but the plate contact.

## Dry ground: bare floor against firm plates (2026-11-07)

Run 6 on very firm mud with plates (PC, `KACHA_MU_MAX=1.0`, 10 roads per LI, smooth road, 300 s): **100% at LI -0.2, 0.0, 0.2
and 0.3** (8.30 m, cost of transport 0.88, no falls). On the bare dry floor (no plates) the same policy gets 0% (20 roads) and falls
in the first seconds (g1_trace: 0% on 10 roads). Grip is about 1.0 in both.

Plain DCM, same 20 roads (seeds 70000+k, smooth road, 300 s, scratch environment):

| Ground | DCM success (95% CI) | Distance (m) |
|---|---|---|
| bare floor, no plates (`--axis depth --values 0.0`) | 85% (64-95) | 7.55 |
| firm plates, LI -0.2, cap 1.0 | 100% (84-100) | 8.30 |

Reading: the only difference between the two grounds is the contact the foot meets (the road's height-field mesh against a flat
sole-sized plate); the policy is extremely sensitive to it (0% against 100%), plain DCM only mildly (85% against 100%, intervals
overlap). The road is perfectly flat (roughness 0), so bumps are not the cause. Not known: which of the two contacts is closer to a
real dry compacted road; both are simulator representations. The policy trained on 25% bare-floor roads and still fails on them,
which is not explained (per-condition training success was not logged).
