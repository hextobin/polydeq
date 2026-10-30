#!/usr/bin/env bash
# Automatic finish: once the final runs are done, summarise, commit and push to GitHub,
# and only if the push verifiably landed, delete this Droplet via /root/selfdestruct.sh.
# Run in its own tmux session:  tmux new -s finish -d "bash scripts/finish.sh"
#
# Waits for all of:
#   FINAL_RUNS_LAUNCHED     touched by scripts/rerun_implicit.sh after the last runs start
#   DEPLOY_KEY_CONFIRMED    touched once the user confirms the deploy key is on the repo
#   no train.py / sweep.py process for IDLE_MIN straight minutes
# If the push fails, it logs the error and leaves the box up (selfdestruct.timer remains
# the backstop at its fixed time).
set -u
cd /root/polydeq
IDLE_MIN=${IDLE_MIN:-10}
LOG=logs/finish.log
mkdir -p logs
log() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
busy() { pgrep -f "python[^ ]* (scripts/sweep|train)\.py" >/dev/null; }

log "finish: waiting (FINAL_RUNS_LAUNCHED, DEPLOY_KEY_CONFIRMED, ${IDLE_MIN} min idle)"
idle=0
n=0
while true; do
  if [ -f FINAL_RUNS_LAUNCHED ] && [ -f DEPLOY_KEY_CONFIRMED ] && ! busy; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  [ "$idle" -ge "$IDLE_MIN" ] && break
  n=$((n + 1))
  [ $((n % 60)) -eq 0 ] && log "finish: still waiting (flags: $(ls FINAL_RUNS_LAUNCHED DEPLOY_KEY_CONFIRMED 2>/dev/null | tr '\n' ' ')idle ${idle} min)"
  sleep 60
done

log "finish: idle for ${IDLE_MIN} min, summarising"
.venv/bin/python scripts/summarize.py runs/main --md runs/main/summary.md >>"$LOG" 2>&1 \
  || log "finish: summarize.py failed (continuing to push the raw runs)"

git add -A >>"$LOG" 2>&1
git commit -q -m "polydeq: code and runs from the DigitalOcean GPU box ($(date -u '+%F %H:%M') UTC)" >>"$LOG" 2>&1 \
  || log "finish: nothing new to commit"
if ! git push origin HEAD:main >>"$LOG" 2>&1; then
  log "finish: PUSH FAILED; box left up (selfdestruct.timer is the backstop)"
  exit 1
fi
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote origin refs/heads/main 2>>"$LOG" | cut -f1)
if [ "$remote_sha" != "$local_sha" ]; then
  log "finish: remote main ($remote_sha) != local HEAD ($local_sha); box left up"
  exit 1
fi
log "finish: pushed $local_sha to origin/main; running /root/selfdestruct.sh"
/root/selfdestruct.sh >>"$LOG" 2>&1
log "finish: selfdestruct.sh exit=$?"
