Sweep of `LI` (base season peak_monsoon, smooth road), 10 episodes per value, road seeds 70000+k, no assist, no observation noise. Success with 95% Wilson intervals; fall causes as shares of falls.

| Method | LI | Success (95% CI) | Fell | Left road | Distance (m) | hidden_water | stuck | sank | slipped | no_trigger | Step (m) | Foot lift (m) | Stance sinkage (m) | Cost of transport |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DCM capture-point controller | 0.7 | 100% (72-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0% | 0% | 0.13 | 0.097 | 0.002 | 1.12 |
| DCM capture-point controller | 0.75 | 0% (0-28) | 30% | 0% | 6.65 | 0% | 0% | 0% | 100% | 0% | 0.09 | 0.091 | 0.009 | 2.43 |
| DCM capture-point controller | 0.8 | 0% (0-28) | 100% | 0% | -0.09 | 0% | 10% | 0% | 90% | 0% | 0.11 | 0.086 | 0.018 | - |
| DCM capture-point controller | 0.85 | 0% (0-28) | 100% | 0% | -0.25 | 0% | 0% | 0% | 100% | 0% | 0.16 | 0.094 | 0.024 | - |
| g1-dcm-wet-check | 0.7 | 20% (6-51) | 40% | 40% | 6.00 | 0% | 0% | 25% | 75% | 0% | 0.11 | 0.098 | 0.002 | 1.87 |
| g1-dcm-wet-check | 0.75 | 90% (60-98) | 10% | 0% | 7.73 | 0% | 0% | 0% | 100% | 0% | 0.11 | 0.096 | 0.004 | 1.61 |
| g1-dcm-wet-check | 0.8 | 100% (72-100) | 0% | 0% | 8.30 | 0% | 0% | 0% | 0% | 0% | 0.10 | 0.090 | 0.010 | 1.54 |
| g1-dcm-wet-check | 0.85 | 0% (0-28) | 0% | 0% | 7.46 | 0% | 0% | 0% | 0% | 0% | 0.09 | 0.080 | 0.021 | 2.02 |
