"""Robustness sweep: run G1 controllers over a range of one mud property at a time.

Each value of the swept property is tested on the same road seeds (SWEEP_SEED + k), so
curves for different controllers and values are directly comparable. Everything else
stays at the base season. No balance assist, no observation noise.

Axes (any one per run; docs/study.md, "Factors"):
    LI      liquidity index: sets soil strength and suction together (-0.2 .. 1.3)
    su_kpa  soil strength alone (kPa)
    depth   mud depth above the firm base (m)
    mu      surface friction of the wet clay
    beta    suction ratio alone (0 .. 0.6)
    fill    share of each pothole and rut dip filled with (opaque) water (0 .. 1)

Usage:
    python scripts/g1_sweep.py --axis LI --values -0.2 0.2 0.5 0.7 0.9 1.0 1.1 1.2 \\
        --methods scripted run:~/kacha/runs/g1-road-early_monsoon-s0 \\
                  unitree:~/unitree_rl_gym/deploy/pre_train/g1/motion.pt \\
        --base peak_monsoon --episodes 30 --workers 4 --out results/sweep_LI
Writes OUT.csv (one row per episode), OUT_summary.md (success with 95% Wilson
intervals, fall causes, gait measures) and OUT.png (success against the swept value).
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from kacha.terrain import ROADS  # noqa: E402

SWEEP_SEED = 70_000
MAX_SECONDS = 60.0      # time to cross the road, the same for every method
AXES = ("LI", "su_kpa", "depth", "mu", "beta", "fill")
CAUSES = ("hidden_water", "stuck", "sank", "slipped", "no_trigger")
GAIT = ("step_length", "cadence_per_min", "double_support", "foot_lift", "stance_sinkage",
        "slip_m", "cost_of_transport", "mean_speed")


def _job(args):
    """One worker task: a method, one swept value, a block of episodes."""
    spec, axis, value, base, seeds, road, fix = args
    from kacha.g1_env import G1MudRoadEnv
    from kacha.g1_policies import make_method, run_episode
    from kacha.terrain import ROADS
    method = make_method(os.path.expanduser(spec) if ":" not in spec else
                         spec.split(":", 1)[0] + ":" + os.path.expanduser(spec.split(":", 1)[1]))
    env = G1MudRoadEnv(season=base, seed=seeds[0], obs_noise=False, assist=0.0, max_seconds=MAX_SECONDS,
                       mud={**fix, axis: value}, road=ROADS[road], **method.env_kw)
    rows = []
    for s in seeds:
        m = run_episode(method, env, s)
        m.update(method=method.name, axis=axis, value=value)
        rows.append(m)
    env.close()
    return rows


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["value"])].append(r)
    out = []
    for (meth, val), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n, k = len(rs), sum(r["success"] for r in rs)
        lo, hi = wilson(k, n)
        falls = [r for r in rs if r["fell"]]
        causes = Counter(r["outcome"] for r in falls)
        mean = lambda key: (float(np.mean([r[key] for r in rs if r.get(key) is not None]))
                            if any(r.get(key) is not None for r in rs) else float("nan"))
        out.append(dict(method=meth, value=val, n=n, success=k / n, lo=lo, hi=hi,
                        fell=len(falls) / n, off_road=sum(r["off_road"] for r in rs) / n,
                        distance=mean("distance"),
                        causes={c: causes.get(c, 0) / max(len(falls), 1) for c in CAUSES},
                        **{g: mean(g) for g in GAIT}))
    return out


def write_summary(summary, axis, base, episodes, path):
    lines = [f"Sweep of `{axis}` (base season {base}), {episodes} episodes per value, road seeds "
             f"{SWEEP_SEED}+k, no assist, no observation noise. Success with 95% Wilson "
             f"intervals; fall causes as shares of falls.", "",
             f"| Method | {axis} | Success (95% CI) | Fell | Left road | Distance (m) | "
             + " | ".join(CAUSES) + " | Step (m) | Foot lift (m) | Stance sinkage (m) | Cost of transport |",
             "|---|---|---|---|---|---|" + "---|" * len(CAUSES) + "---|---|---|---|"]
    f = lambda v, d=2: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    for s in summary:
        lines.append(
            f"| {s['method']} | {s['value']:g} | {100 * s['success']:.0f}% "
            f"({100 * s['lo']:.0f}-{100 * s['hi']:.0f}) | {100 * s['fell']:.0f}% | "
            f"{100 * s['off_road']:.0f}% | {f(s['distance'])} | "
            + " | ".join(f"{100 * s['causes'][c]:.0f}%" for c in CAUSES)
            + f" | {f(s['step_length'])} | {f(s['foot_lift'], 3)} | {f(s['stance_sinkage'], 3)} | "
            f"{f(s['cost_of_transport'])} |")
    open(path, "w").write("\n".join(lines) + "\n")


def plot(summary, axis, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for meth in sorted({s["method"] for s in summary}):
        ss = sorted((s for s in summary if s["method"] == meth), key=lambda s: s["value"])
        x = [s["value"] for s in ss]
        y = [100 * s["success"] for s in ss]
        ax.plot(x, y, marker="o", label=meth)
        ax.fill_between(x, [100 * s["lo"] for s in ss], [100 * s["hi"] for s in ss], alpha=0.15)
    ax.set_xlabel(axis)
    ax.set_ylabel("success (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=AXES)
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["scripted"],
                    help="scripted, dcm, run:PATH, run-best:PATH, unitree:PATH")
    ap.add_argument("--base", default="peak_monsoon",
                    help="season for every property not swept (dry has no mud: give depth)")
    ap.add_argument("--road", default="rutted", choices=list(ROADS),
                    help="rutted (the monsoon road) or smooth (no ruts, dips or roughness)")
    ap.add_argument("--fix", nargs="*", default=[], metavar="AXIS=VALUE",
                    help="hold other mud properties at a value, e.g. --fix LI=0.7")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    fix = {}
    for kv in a.fix:
        k, _, v = kv.partition("=")
        if k not in AXES or not v:
            ap.error(f"--fix expects AXIS=VALUE with AXIS in {AXES}, got {kv!r}")
        fix[k] = float(v)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    block = max(1, a.episodes // max(a.workers, 1))
    jobs = []
    for spec in a.methods:
        for v in a.values:
            seeds = [SWEEP_SEED + k for k in range(a.episodes)]
            for i in range(0, len(seeds), block):
                jobs.append((spec, a.axis, v, a.base, seeds[i:i + block], a.road, fix))
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(_job, jobs):
                rows += r
                print(f"[sweep] {len(rows)} / {len(a.methods) * len(a.values) * a.episodes} episodes",
                      flush=True)
    else:
        for j in jobs:
            rows += _job(j)
            print(f"[sweep] {len(rows)} / {len(a.methods) * len(a.values) * a.episodes} episodes",
                  flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    held = "".join(f", {k} = {v:g}" for k, v in fix.items())
    write_summary(summary, a.axis, f"{a.base}{held}, {a.road} road", a.episodes, a.out + "_summary.md")
    plot(summary, a.axis, a.out + ".png")
    print(open(a.out + "_summary.md").read())
    print(f"wrote {a.out}.csv, {a.out}_summary.md, {a.out}.png")


if __name__ == "__main__":
    main()
