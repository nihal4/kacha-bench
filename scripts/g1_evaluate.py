"""Evaluate G1 stage-1 controllers on the same fixed test roads (the baseline table).

Every method sees identical roads, speed commands and seasons: episode k uses road seed
TEST_SEED + k, so results are directly comparable. No balance assist, no observation
noise, deterministic policies.

Methods:
    --scripted               the scripted mud-walking controller alone (zero corrections)
    --dcm                    the model-based capture-point controller (kacha/dcm.py)
    --run RUN_DIR [...]      trained runs (final model; --best for the best checkpoint)
    --unitree MOTION_PT      Unitree's pretrained G1 walking policy from unitree_rl_gym
                             (deploy/pre_train/g1/motion.pt), trained on rigid ground.
                             A TorchScript file can run code when loaded: only load
                             files from a source you trust.

Usage:
    python scripts/g1_evaluate.py --scripted --run runs/g1-road-early_monsoon-s0 \\
        --unitree ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt \\
        --seasons dry early_monsoon peak_monsoon --episodes 10 --out results/stage1.md
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from kacha.g1_env import G1MudRoadEnv  # noqa: E402
from kacha.terrain import ROADS  # noqa: E402
from kacha.g1_policies import run_episode  # noqa: E402

TEST_SEED = 50_000
# time to cross the 8.3 m road; the same for every method (the DCM controller walks at
# 0.2 m/s and needs about 45 s)
MAX_SECONDS = 60.0


# ----------------------------------------------------------------------------- evaluation
def _job(args):
    """One worker task: a method (built in the worker), a season, a block of road seeds."""
    spec, season, seeds, road = args
    from kacha.g1_policies import make_method
    method = make_method(spec)
    env = G1MudRoadEnv(season=season, seed=TEST_SEED, obs_noise=False, assist=0.0,
                       max_seconds=MAX_SECONDS, road=ROADS[road], **method.env_kw)
    rows = [run_episode(method, env, s) for s in seeds]
    env.close()
    return spec, method.name, season, rows


def summarise(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    causes = Counter(r["outcome"] for r in rows if r["fell"])
    return dict(success=m("success"), fell=m("fell"), off_road=m("off_road"),
                distance=m("distance"), seconds=m("seconds"), n=len(rows),
                causes=", ".join(f"{k} {v}" for k, v in causes.most_common()) or "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--dcm", action="store_true")
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--unitree")
    ap.add_argument("--seasons", nargs="+", default=["dry", "early_monsoon", "peak_monsoon"])
    ap.add_argument("--road", default="rutted", choices=list(ROADS),
                    help="rutted (the monsoon road) or smooth (no ruts, dips or roughness)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (use the core count)")
    ap.add_argument("--out", help="markdown table; a .json with every episode is written next to it")
    a = ap.parse_args()
    specs = (["scripted"] if a.scripted else []) + (["dcm"] if a.dcm else []) \
        + [("run-best:" if a.best else "run:") + os.path.expanduser(r) for r in a.run] \
        + ([f"unitree:{os.path.expanduser(a.unitree)}"] if a.unitree else [])
    if not specs:
        ap.error("give at least one of --scripted, --dcm, --run, --unitree")
    seeds = [TEST_SEED + k for k in range(a.episodes)]
    block = max(1, -(-a.episodes // max(a.workers, 1)))
    jobs = [(sp, season, seeds[i:i + block], a.road) for sp in specs for season in a.seasons
            for i in range(0, len(seeds), block)]
    results, names = {}, {}
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            out = list(ex.map(_job, jobs))
    else:
        out = [_job(j) for j in jobs]
    for spec, name, season, rows in out:
        names[spec] = name
        results.setdefault((spec, season), []).extend(rows)
    road = 4.3 - (-4.0)
    lines = [f"Stage 1 ({a.road} road, {road:.1f} m), {a.episodes} episodes per season, "
             f"road seeds {TEST_SEED}+k, no assist, no observation noise.", "",
             "| Method | Season | Success | Fell | Left road | Distance (m) | Time (s) | Fall causes |",
             "|---|---|---|---|---|---|---|---|"]
    detail = {}
    for sp in specs:
        for season in a.seasons:
            rows = results[(sp, season)]
            s = summarise(rows)
            detail[f"{names[sp]} | {season}"] = rows
            line = (f"| {names[sp]} | {season} | {100 * s['success']:.0f}% | {100 * s['fell']:.0f}% | "
                    f"{100 * s['off_road']:.0f}% | {s['distance']:.2f} | {s['seconds']:.1f} | {s['causes']} |")
            lines.append(line)
            print(line, flush=True)
    text = "\n".join(lines) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(text)
        json.dump(detail, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
