#!/usr/bin/env bash
# Training deadline. Start it FIRST, in its own tmux window:
#     bash scripts/watchdog.sh 300      # minutes until training is killed
# At the deadline it kills the sweep driver and every train.py, and nothing else.
# It never powers off or destroys the box.
#
# DigitalOcean: the spending cap is separate. This Droplet has a systemd timer
# (selfdestruct.timer -> /root/selfdestruct.sh) that deletes it at a fixed time; see
# CLAUDE.md. Never shut the box down: a powered-off Droplet is still billed and the
# timer cannot fire.
set -u
MIN=${1:?usage: watchdog.sh MINUTES}
END=$(( $(date +%s) + MIN * 60 ))
echo "watchdog: kills training at $(date -d @$END)"
while [ "$(date +%s)" -lt "$END" ]; do
  left=$(( (END - $(date +%s)) / 60 ))
  [ $((left % 30)) -eq 0 ] && echo "watchdog: ${left} min left"
  sleep 60
done
echo "watchdog: time is up, stopping training at $(date)"
pkill -f scripts/sweep.py || true   # first, so it cannot launch the next run
pkill -f train.py || true
sleep 20
pkill -9 -f train.py || true
echo "watchdog: done (box left running; selfdestruct.timer is the spending cap)"
