#!/usr/bin/env bash
# One-time setup of the always-on Ubuntu training PC for KachaBench (tested target:
# Ubuntu 24.04, i3-8100, 12 GB). Safe to run again: finished steps are skipped and the
# services are refreshed. Adapted from connectome-tray-balance/scripts/desktop/setup.sh.
#
#   bash ~/kacha/repo/scripts/desktop/setup.sh
#
# Creates ~/kacha/{repo,.venv,runs,logs}, installs CPU PyTorch + training packages,
# runs a quick check, and installs:
#   kacha-train   training service (started only by 'kacha start'; never at boot)
#   kacha-watch   progress videos every 500k steps + dashboard data (starts at boot)
#   kacha-web     dashboard at http://<this-pc>:8081 (starts at boot)
#   kacha         command to control it all (kacha help)
set -euo pipefail
ROOT="$HOME/kacha"
REPO="$ROOT/repo"
PY="$ROOT/.venv/bin/python"

echo "== system packages (asks for your password once)"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-dev libegl1 libgl1 libosmesa6 ffmpeg tmux lm-sensors zip >/dev/null

echo "== code"
mkdir -p "$ROOT/runs" "$ROOT/logs"
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull -q; else git clone -q https://github.com/nihal4/kacha-bench "$REPO"; fi

echo "== python environment (CPU PyTorch: smaller, no GPU needed)"
[ -x "$PY" ] || python3 -m venv "$ROOT/.venv"
"$PY" -m pip install -q -U pip
"$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install -q -e "$REPO[train]"

echo "== quick check (about 1 minute)"
cd "$REPO"
"$PY" scripts/smoke_test.py | tail -1
"$PY" -W ignore scripts/train.py --profile smoke --runs_dir "$ROOT/check" --name setup-check 2>&1 | grep -E "^\[train\] finished|Error" || true
rm -rf "$ROOT/check"

echo "== services"
TOTAL_MB=$(free -m | awk '/Mem/{print $2}')
MEM_MAX="$(( TOTAL_MB - 3000 ))M"; MEM_HIGH="$(( TOTAL_MB - 3800 ))M"
mkdir -p "$HOME/.config/systemd/user" "$HOME/.local/bin"
ln -sf "$REPO/scripts/desktop/kacha" "$HOME/.local/bin/kacha"
chmod +x "$REPO/scripts/desktop/kacha" "$REPO/scripts/desktop/run_job.sh"
cat > "$HOME/.config/systemd/user/kacha-train.service" <<UNIT
[Unit]
Description=KachaBench training (resumes from the last checkpoint)
After=network-online.target
[Service]
ExecStart=$REPO/scripts/desktop/run_job.sh
Restart=on-failure
RestartSec=60
Nice=10
MemoryHigh=$MEM_HIGH
MemoryMax=$MEM_MAX
MemorySwapMax=0
CPUWeight=50
[Install]
WantedBy=default.target
UNIT
cat > "$HOME/.config/systemd/user/kacha-watch.service" <<UNIT
[Unit]
Description=KachaBench progress videos and dashboard
[Service]
ExecStart=$PY $REPO/scripts/desktop/watch.py --root $ROOT
Restart=always
RestartSec=60
Nice=19
MemoryMax=1500M
MemorySwapMax=0
CPUWeight=10
Environment=MUJOCO_GL=egl
[Install]
WantedBy=default.target
UNIT
cat > "$HOME/.config/systemd/user/kacha-web.service" <<UNIT
[Unit]
Description=KachaBench dashboard on port 8081
[Service]
ExecStart=$PY -m http.server 8081 --directory $ROOT
Restart=always
[Install]
WantedBy=default.target
UNIT
[ -f "$ROOT/job.env" ] || "$REPO/scripts/desktop/kacha" job none >/dev/null
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now kacha-watch kacha-web
systemctl --user disable kacha-train 2>/dev/null || true
if sudo ufw status 2>/dev/null | grep -q "Status: active"; then sudo ufw allow 8081/tcp >/dev/null; fi

echo
echo "Done. Open a new terminal (or run: source ~/.profile), then:"
echo "   kacha job       # show what will be trained"
echo "   kacha start     # start training"
echo "   kacha status    # progress"
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8081"
