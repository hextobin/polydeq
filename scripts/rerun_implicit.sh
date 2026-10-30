#!/usr/bin/env bash
# Implicit-arm reruns with the solver settings from scripts/probe_solver.py, gated on
# seed 0: seeds 1 and 2 start only if the monotone arms stay converged late in training.
# FINAL_RUNS_LAUNCHED is touched only after seeds 1 and 2 are launched (scripts/finish.sh
# waits for it). If the gate fails, nothing more runs and the flag is NOT touched.
set -u
cd /root/polydeq
COMMON="--data data/enwik8 --steps 5000 --batch_size 64 --dtype bf16 --compile --eval_every 500 --eval_batches 20"
ARMS="elem_implicit group_implicit group_free"
args_for() {
  case $1 in
    elem_implicit)  echo "--newton_init cardano --fixed_iters 1" ;;            # Cardano is exact for k=1
    group_implicit) echo "--newton_init cardano --fixed_iters 4 --line_search 2" ;;
    group_free)     echo "--fixed_iters 6 --line_search 4" ;;                  # no Cardano for the free form
  esac
}
run() {
  local arm=$1 seed=$2 out=runs/main/$1_s$2
  mkdir -p "$out"
  .venv/bin/python train.py $COMMON --arm "$arm" --seed "$seed" --out "$out" $(args_for "$arm") \
    > "$out/stdout.txt" 2>&1
  echo "$(date -u +%H:%M) $arm s$seed exit=$?"
}

echo "$(date -u +%H:%M) phase 1: seed 0"
for a in $ARMS; do run "$a" 0 & done
wait
if .venv/bin/python scripts/check_converged.py runs/main/elem_implicit_s0 runs/main/group_implicit_s0; then
  echo "$(date -u +%H:%M) gate passed; phase 2: seeds 1, 2"
  for a in $ARMS; do (run "$a" 1; run "$a" 2) & done
  sleep 30
  touch FINAL_RUNS_LAUNCHED
  echo "$(date -u +%H:%M) FINAL_RUNS_LAUNCHED touched"
  wait
  echo "$(date -u +%H:%M) all reruns done"
else
  echo "$(date -u +%H:%M) GATE FAILED: seeds 1, 2 not launched; FINAL_RUNS_LAUNCHED not touched" \
    | tee runs/main/RERUN_GATE_FAILED.txt
fi
