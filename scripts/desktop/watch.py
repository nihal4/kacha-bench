"""Live dashboard + progress videos for the training PC (runs as the kacha-watch service).
Adapted from connectome-tray-balance/scripts/desktop/watch.py.

Every 10 s: read only the NEW lines of the training log and write ~/kacha/status.json
(live numbers, learning curves, video list, CPU temperature, RAM). The page
~/kacha/index.html (copied from dashboard.html) fetches it and redraws in the browser.
The kacha-web service serves ~/kacha on port 8081.

Every 60 s: when a run has passed another --every steps, film its newest checkpoint on
the three test roads and save runs/<run>/progress/step_<N>k.mp4.

Runs at low priority with 1 thread, so it barely slows training.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from kacha.safety import cpu_temp_c  # noqa: E402

SERVICE = "kacha-train"


def last_eval(run):
    f = os.path.join(run, "evaluations.npz")
    try:
        d = np.load(f)
        return int(d["timesteps"][-1]) if len(d["timesteps"]) else None
    except Exception:
        return None


def start_video(run, step_k, py):
    os.makedirs(os.path.join(run, "progress"), exist_ok=True)
    out = os.path.join(run, "progress", f"step_{step_k:05d}k.mp4")
    proc = subprocess.Popen([py, os.path.join(REPO, "scripts", "record.py"), "--run", run, "--out", out],
                            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl")),
                            preexec_fn=lambda: os.nice(10))
    return proc, out, time.time()


def read_job(root):
    job = {}
    try:
        for line in open(os.path.join(root, "job.env")):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                job[k] = v.strip('"')
    except OSError:
        pass
    return job


def service_state():
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", SERVICE], capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def ram_free_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return None


def _num(x):
    try:
        v = float(str(x).replace(",", "").replace("%", "").replace("m", "").strip())
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _clean(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class LogTail:
    """Reads a training log incrementally and keeps the latest numbers plus the
    training-return / walked / distance history from the [progress] lines."""

    KEYS = {"fps", "std", "approx_kl", "ep_rew_mean", "total_timesteps", "learning_rate",
            "clip_fraction", "explained_variance"}

    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, ""
        self.live, self.train = {}, {"t": [], "r": [], "s": [], "d": []}
        self.events = []

    def update(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:
            self.__init__(self.path)
        if size == self.pos:
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read(size - self.pos).decode("utf-8", "replace")
        self.pos = size
        lines = (self.buf + chunk).split("\n")
        self.buf = lines.pop()
        for ln in lines:
            self._line(ln.strip())
        self.live["log_mtime"] = os.path.getmtime(self.path)

    def _line(self, ln):
        if ln.startswith("[progress]"):
            # [progress] 1,150,012 steps | training episodes: return 3.16 | solved 20% | distance 1.20 m
            parts = ln.split("|")
            steps = _num(parts[0].split()[1])
            ret = sol = dist = None
            for p in parts[1:]:
                if "return" in p:
                    ret = _num(p.split("return")[1])
                elif "solved" in p:
                    sol = _num(p.split("solved")[1])
                elif "distance" in p:
                    dist = _num(p.split("distance")[1])
            if steps is not None and ret is not None:
                if self.train["t"] and steps <= self.train["t"][-1]:      # resumed: drop overlap
                    while self.train["t"] and self.train["t"][-1] >= steps:
                        for k in self.train:
                            self.train[k].pop()
                for k, v in zip("trsd", (steps, ret, sol, dist)):
                    self.train[k].append(v)
                self.live.update(train_steps=steps, train_return=ret, train_solved=sol,
                                 train_distance=dist)
        elif ln.startswith("|") and ln.count("|") >= 3:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) == 2 and cells[0] in self.KEYS:
                v = _num(cells[1])
                if v is not None:
                    self.live[cells[0]] = v
        elif ln.startswith("Eval num_timesteps="):
            try:
                self.live["eval_steps"] = int(ln.split("=")[1].split(",")[0])
                self.live["eval_return"] = float(ln.split("episode_reward=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif ln.startswith(("[train]", "[safety]")):
            self.events = (self.events + [ln[:200]])[-8:]


def eval_history(run):
    try:
        d = np.load(os.path.join(run, "evaluations.npz"))
        t = d["timesteps"].tolist()
        r = np.nan_to_num(d["results"].mean(1)).round(2).tolist() if len(t) else []
        s = (d["successes"].mean(1) * 100).round(1).tolist() \
            if "successes" in d.files and d["successes"].size else []
        return {"t": t, "r": r, "s": s, "n_episodes": int(d["results"].shape[1]) if len(t) else 0}
    except Exception:
        return {"t": [], "r": [], "s": [], "n_episodes": 0}


class Dashboard:
    def __init__(self, root):
        self.root, self.tails, self.cache = root, {}, {}
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
        dst = os.path.join(root, "index.html")
        if os.path.exists(page):
            with open(page) as fi, open(dst + ".tmp", "w") as fo:
                fo.write(fi.read())
            os.replace(dst + ".tmp", dst)

    def run_info(self, run, active):
        name = os.path.basename(run)
        f = os.path.join(run, "evaluations.npz")
        mt = os.path.getmtime(f) if os.path.exists(f) else 0
        if self.cache.get(run, (None,))[0] != mt:
            self.cache[run] = (mt, eval_history(run))
        ev = self.cache[run][1]
        vids = sorted((v for v in glob.glob(os.path.join(run, "progress", "*.mp4"))
                       if ".part." not in v and os.path.getsize(v) > 10_000),
                      key=os.path.getmtime, reverse=True)
        log = os.path.join(self.root, "logs", name + ".log")
        train = {"t": [], "r": [], "s": [], "d": []}
        if os.path.exists(log):
            self.tails.setdefault(log, LogTail(log)).update()
            train = self.tails[log].train
        try:
            cfg = json.load(open(os.path.join(run, "config.json")))
        except Exception:
            cfg = {}
        stamps = [mt] + [os.path.getmtime(v) for v in vids[:1]]
        if os.path.exists(log):
            stamps.append(os.path.getmtime(log))
        return {"name": name, "season": cfg.get("season", ""), "scan": cfg.get("scan", False),
                "active": name == active, "eval": ev, "train": train,
                "best_return": max(ev["r"]) if ev["r"] else None,
                "best_solved": max(ev["s"]) if ev["s"] else None,
                "videos": [os.path.relpath(v, self.root) for v in vids], "last_update": max(stamps)}

    def write(self, runs):
        job = read_job(self.root)
        active = job.get("NAME")
        infos = sorted((self.run_info(r, active) for r in runs),
                       key=lambda i: (not i["active"], -i["last_update"]))
        live, events = {}, []
        tail = self.tails.get(os.path.join(self.root, "logs", f"{active}.log"))
        if tail:
            live, events = dict(tail.live), tail.events
        t = cpu_temp_c()
        status = {"updated": time.time(), "cpu_temp": round(t, 1) if t else None,
                  "ram_free_gb": ram_free_gb(), "service": service_state(),
                  "job": {"name": active, "season": job.get("SEASON"),
                          "steps_total": _num(job.get("STEPS", "0") or "0"), "extra": job.get("EXTRA", "")},
                  "live": live, "events": events, "runs": infos,
                  "results": results_info(self.root)}
        tmp = os.path.join(self.root, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(_clean(status), f, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, os.path.join(self.root, "status.json"))


def results_info(root):
    """Result tables and plots for the dashboard's Results section: every run's
    results/ folder (written by --then-evaluate) and the repository's results/ folder
    (sweeps and tables run by hand). Image paths are relative to the served root."""
    groups = []
    dirs = [(os.path.basename(os.path.dirname(d)), d)
            for d in glob.glob(os.path.join(root, "runs", "*", "results"))]
    dirs.append(("hand-run results (repo/results)", os.path.join(root, "repo", "results")))
    for title, d in dirs:
        if not os.path.isdir(d):
            continue
        items = []
        for f in sorted(glob.glob(os.path.join(d, "*.md")) + glob.glob(os.path.join(d, "*.png")),
                        key=os.path.getmtime, reverse=True):
            item = {"name": os.path.basename(f), "updated": os.path.getmtime(f)}
            if f.endswith(".md"):
                try:
                    item["text"] = open(f).read()[:20000]
                except OSError:
                    continue
            else:
                item["img"] = os.path.relpath(f, root)
            items.append(item)
        if items:
            groups.append({"title": title, "done": os.path.exists(os.path.join(d, "DONE")),
                           "items": items, "updated": max(i["updated"] for i in items)})
    return sorted(groups, key=lambda g: -g["updated"])


def find_runs(root):
    return sorted(r for r in glob.glob(os.path.join(root, "runs", "*"))
                  if os.path.exists(os.path.join(r, "config.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/kacha"))
    ap.add_argument("--every", type=int, default=500_000, help="steps between progress videos")
    ap.add_argument("--live", type=int, default=10, help="seconds between dashboard updates")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    state_f = os.path.join(args.root, "watch_state.json")
    dash = Dashboard(args.root)
    last_check, pending = 0.0, None
    while True:
        runs = find_runs(args.root)
        if pending is not None:
            proc, out, t0, run, mark = pending
            if proc.poll() is None and time.time() - t0 > 3600:
                proc.kill()
            if proc.poll() is not None:
                state = json.load(open(state_f)) if os.path.exists(state_f) else {}
                state[run + "|tried"] = mark
                if os.path.exists(out):
                    state[run] = mark
                json.dump(state, open(state_f, "w"))
                pending = None
        if pending is None and time.time() - last_check >= 60:
            last_check = time.time()
            state = json.load(open(state_f)) if os.path.exists(state_f) else {}
            for run in runs:
                steps = last_eval(run)
                if not steps or not os.path.exists(os.path.join(run, "model.zip")):
                    continue
                mark = steps // args.every * args.every
                if mark >= args.every and mark > max(state.get(run, 0), state.get(run + "|tried", 0)):
                    pending = (*start_video(run, mark // 1000, sys.executable), run, mark)
                    break
        dash.write(runs)
        if args.once:
            break
        time.sleep(args.live)


if __name__ == "__main__":
    main()
