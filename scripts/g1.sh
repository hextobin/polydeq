#!/usr/bin/env bash
# G1 throughput check: 200 steps each of gelu, group_implicit, group_explicit, one at a time.
#     bash scripts/g1.sh runs/g1 [extra train.py args...]
set -u
ROOT=${1:-runs/g1}; shift || true
mkdir -p "$ROOT"
for arm in gelu group_implicit group_explicit; do
  .venv/bin/python train.py --data data/enwik8 --arm "$arm" --out "$ROOT/$arm" --seed 0 \
    --steps 200 --batch_size 64 --dtype bf16 --eval_every 100 --eval_batches 20 "$@" \
    > "$ROOT/$arm.log" 2>&1
  echo "$arm exit=$?"
done
echo G1 DONE
