# DCM walking speed and time limit at wet LI (smooth road), 2026-11-06

Question: is the failure of DCM and the policies at LI 0.75 to 0.85 a balance wall, or a speed/time effect?
Method: `g1_sweep.py --axis LI --methods dcm:SPEED --base peak_monsoon --road smooth`, 6 roads per cell (road seeds
70000+k, no observation noise), run in the scratch environment (not the training PC). Small samples: 95% CI for 6/6 is 61-100%.
`KACHA_MAX_SECONDS` sets the time limit (default 60 s, the limit used in all other results).

| DCM speed | Limit | LI 0.75 | LI 0.8 | LI 0.85 | LI 0.9 |
|---|---|---|---|---|---|
| 0.20 m/s (normal) | 60 s | 0% (falls) | 0% (falls) | 0% | 0% |
| 0.25 m/s | 60 s | 0% (falls) | 0% (falls) | - | - |
| 0.15 m/s | 60 s | no falls, 6.9 m (timeout) | falls (0.37 m) | - | - |
| 0.15 m/s | 300 s | - | 0% (falls) | 0% (falls) | 0% (falls) |
| 0.10 m/s | 60 s | no falls, 5.6 m (timeout) | no falls, 2.5 m (timeout) | - | - |
| 0.10 m/s | 300 s | - | **100% (8.30 m), 6/6** | 0% (falls) | 0% (falls) |

Reading:

- At 0.1 m/s with 5 minutes the DCM crosses LI 0.8 on all 6 roads; at the normal 0.2 m/s it falls (slipping). So LI 0.8 is
  not a balance wall: it is a walking-speed (slip) problem plus the 60 s limit.
- LI 0.85 still falls at 0.1 m/s (immediately), so a physical limit exists somewhere near 0.82 to 0.85 for this controller.
- The 60 s limit and a fixed 0.2 m/s base speed are what keep policies from crossing LI 0.8 to 0.85; the policy cannot change
  the speed (DCM speed is a constant, the policy only adds leg-joint corrections, arms are held fixed).
