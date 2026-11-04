"""Evaluate G1 stage-1 controllers on the same fixed test roads (the baseline table).

Every method sees identical roads, speed commands and seasons: episode k uses road seed
TEST_SEED + k, so results are directly comparable. No balance assist, no observation
noise, deterministic policies.

Methods:
    --scripted               the scripted mud-walking controller alone (zero corrections)
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
from kacha.g1_policies import Scripted, Trained, Unitree, run_episode  # noqa: E402

TEST_SEED = 50_000


# ----------------------------------------------------------------------------- evaluation
def evaluate(method, season, episodes):
    rows = []
    env = G1MudRoadEnv(season=season, seed=TEST_SEED, obs_noise=False, assist=0.0, **method.env_kw)
    for k in range(episodes):
        rows.append(run_episode(method, env, TEST_SEED + k))
    env.close()
    return rows


def summarise(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    causes = Counter(r["outcome"] for r in rows if r["fell"])
    return dict(success=m("success"), fell=m("fell"), off_road=m("off_road"),
                distance=m("distance"), seconds=m("seconds"), n=len(rows),
                causes=", ".join(f"{k} {v}" for k, v in causes.most_common()) or "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--unitree")
    ap.add_argument("--seasons", nargs="+", default=["dry", "early_monsoon", "peak_monsoon"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", help="markdown table; a .json with every episode is written next to it")
    a = ap.parse_args()
    methods = ([Scripted()] if a.scripted else []) + [Trained(r, a.best) for r in a.run] \
        + ([Unitree(a.unitree)] if a.unitree else [])
    if not methods:
        ap.error("give at least one of --scripted, --run, --unitree")
    road = 4.3 - (-4.0)
    lines = [f"Stage 1 (muddy rutted road, {road:.1f} m), {a.episodes} episodes per season, "
             f"road seeds {TEST_SEED}+k, no assist, no observation noise.", "",
             "| Method | Season | Success | Fell | Left road | Distance (m) | Time (s) | Fall causes |",
             "|---|---|---|---|---|---|---|---|"]
    detail = {}
    for meth in methods:
        for season in a.seasons:
            rows = evaluate(meth, season, a.episodes)
            s = summarise(rows)
            detail[f"{meth.name} | {season}"] = rows
            line = (f"| {meth.name} | {season} | {100 * s['success']:.0f}% | {100 * s['fell']:.0f}% | "
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
