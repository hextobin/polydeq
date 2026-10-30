#!/usr/bin/env bash
# CPU smoke test: unit tests, NumPy math checks, then every arm for a few steps.
# Run from the repo root: bash scripts/smoke.sh
set -euo pipefail
PY=${PYTHON:-python3}
$PY -m pytest -q tests
(cd ref && $PY check_ref.py)
$PY -c "from polydeq.data import synth_arith; synth_arith('data/arith', n_lines=50000)"
for arm in gelu none elem_explicit elem_implicit group_explicit group_implicit group_free \
           lowrank_implicit gelu_matched slot_linear slot_implicit slot_group slot_explicit slot_mlp_matched; do
  $PY train.py --data data/arith --arm "$arm" --out "runs/smoke/$arm" --device cpu --dtype fp32 \
    --d_model 64 --n_layer 2 --n_head 2 --block_size 64 --batch_size 16 \
    --group_size 16 --rank 8 --lowrank_rank 16 --steps 60 --eval_every 30 --eval_batches 4 \
    | tail -n 2
done
$PY scripts/summarize.py runs/smoke
