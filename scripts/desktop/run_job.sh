#!/usr/bin/env bash
# Runs the job in ~/kacha/job.env. Always --resume: after a reboot or power cut it
# continues from the last checkpoint (or starts fresh if there is none).
set -euo pipefail
ROOT="$HOME/kacha"
source "$ROOT/job.env"
cd "$ROOT/repo"
mkdir -p "$ROOT/logs"
exec "$ROOT/.venv/bin/python" -u scripts/train.py --season "$SEASON" --profile desktop \
    --total_steps "$STEPS" --runs_dir "$ROOT/runs" --name "$NAME" --resume $EXTRA \
    >> "$ROOT/logs/$NAME.log" 2>&1
