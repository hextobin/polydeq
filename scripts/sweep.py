"""Run arms x seeds on one GPU, --jobs at a time, skipping runs that already have final.json.

    python scripts/sweep.py --data data/tinystories --root runs/main \
        --arms gelu,none,elem_implicit,group_explicit,group_implicit,gelu_matched \
        --seeds 0,1,2 --budget_minutes 300 -- --steps 5000 --d_model 384 --n_layer 6

Everything after "--" is passed to train.py. The sweep stops launching new runs once
--budget_minutes of wall time is used (the pod watchdog is the hard limit; this is the
soft one). Arms are interleaved by seed (all arms seed 0, then seed 1, ...) so a
partial sweep is still a balanced comparison.
"""
import argparse
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--budget_minutes", type=float, default=0)
ap.add_argument("--jobs", type=int, default=1, help="runs in parallel on the GPU")
args, rest = ap.parse_known_args()
rest = [r for r in rest if r != "--"]

t0 = time.time()
running = {}  # out -> Popen


def reap(block):
    while running:
        for out, proc in list(running.items()):
            if proc.poll() is not None:
                print(f"  {out} -> exit {proc.returncode} ({(time.time() - t0) / 60:.1f} min)",
                      flush=True)
                del running[out]
        if not block or len(running) < args.jobs:
            return
        time.sleep(5)


for seed in args.seeds.split(","):
    for arm in args.arms.split(","):
        out = os.path.join(args.root, f"{arm}_s{seed}")
        if os.path.exists(os.path.join(out, "final.json")):
            print(f"skip {out} (done)")
            continue
        reap(block=True)
        if args.budget_minutes and time.time() - t0 > 60 * args.budget_minutes:
            print("sweep budget used up; stopping before", out)
            break
        cmd = [sys.executable, "train.py", "--data", args.data, "--arm", arm, "--seed", seed,
               "--out", out] + rest
        print(" ".join(cmd), flush=True)
        os.makedirs(out, exist_ok=True)
        running[out] = subprocess.Popen(cmd, stdout=open(os.path.join(out, "stdout.txt"), "w"),
                                        stderr=subprocess.STDOUT)
    else:
        continue
    break
while running:
    reap(block=False)
    time.sleep(5)
print("sweep done", flush=True)
