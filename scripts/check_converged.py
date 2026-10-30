"""Exit 0 if every run dir finished and its solver stayed converged late in training.

    python scripts/check_converged.py RUN_DIR [RUN_DIR ...] [--from_step 2500] [--max_frac 0.01]
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--from_step", type=int, default=2500)
ap.add_argument("--max_frac", type=float, default=0.01)
a = ap.parse_args()
ok = True
for d in a.runs:
    if not os.path.exists(os.path.join(d, "final.json")):
        print(f"{d}: no final.json (crashed or still running)"); ok = False; continue
    evals = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if '"val_loss"' in l]
    late = [r for r in evals if r["step"] >= a.from_step and r.get("solver")]
    worst = max((s.get("frac_unconverged", 0.0) for r in late for s in r["solver"].values()), default=None)
    if worst is None:
        print(f"{d}: no solver stats after step {a.from_step}"); ok = False; continue
    res = max(s.get("res_max", 0.0) for r in late for s in r["solver"].values())
    good = worst < a.max_frac
    ok &= good
    print(f"{d}: worst frac_unconverged {worst:.4f}, res_max {res:.1e} after step {a.from_step} -> {'ok' if good else 'FAIL'}")
sys.exit(0 if ok else 1)
