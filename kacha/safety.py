"""Keep a training PC healthy (adapted from connectome-tray-balance/ctb/safety.py).

limit_threads()  caps PyTorch / BLAS threads
SafetyCallback   stops (after saving) when RAM runs low, rests when the CPU is very hot,
                 and stops after an optional wall-clock limit.
"""
import os
import time

import psutil
from stable_baselines3.common.callbacks import BaseCallback


def limit_threads(n):
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(n)
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def cpu_temp_c():
    """Hottest CPU sensor in deg C on Linux (None if not available)."""
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    vals = [t.current for name in ("coretemp", "k10temp", "cpu_thermal", "acpitz")
            for t in temps.get(name, []) if t.current]
    return max(vals) if vals else None


class SafetyCallback(BaseCallback):
    def __init__(self, save_fn, max_minutes=0, min_free_gb=1.0, max_temp_c=90, check_every_s=10):
        super().__init__(0)
        self.save_fn = save_fn
        self.max_s = max_minutes * 60 if max_minutes else None
        self.min_free_gb, self.max_temp_c = min_free_gb, max_temp_c
        self.check_every_s = check_every_s
        self.stop_reason = None
        self._t0 = self._t_check = None
        self._rest = 0.0

    def _on_training_start(self):
        self._t0 = self._t_check = time.time()

    def _on_step(self):
        now = time.time()
        if now - self._t_check < self.check_every_s:
            return True
        self._t_check = now
        if self.max_s and now - self._t0 - self._rest > self.max_s:
            return self._stop("time limit reached")
        free = psutil.virtual_memory().available / 1e9
        if free < self.min_free_gb:
            return self._stop(f"low memory ({free:.1f} GB free)")
        t = cpu_temp_c()
        if t is not None and t >= self.max_temp_c:
            print(f"[safety] CPU at {t:.0f} C: resting 60 s", flush=True)
            time.sleep(60)
            self._rest += 60
        return True

    def _stop(self, reason):
        self.stop_reason = reason
        print(f"\n[safety] stopping: {reason}. Saving checkpoint...", flush=True)
        self.save_fn()
        return False
