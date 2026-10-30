"""Summarize runs: python scripts/summarize.py runs/<sweep>  [--md out.md]

Reads every <run>/final.json and <run>/log.jsonl below the folder. Prints, per arm:
seeds, params, final val loss (mean and std over seeds), gap to gelu, throughput, and
solver health (worst fraction of unconverged tokens, max Newton iterations, last q/h).
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict


def load(root):
    rows = []
    for f in glob.glob(os.path.join(root, "**", "final.json"), recursive=True):
        d = json.load(open(f))
        logp = os.path.join(os.path.dirname(f), "log.jsonl")
        unconv, iters, qh = 0.0, 0, None
        if os.path.exists(logp):
            for line in open(logp):
                r = json.loads(line)
                for st in r.get("solver", {}).values():
                    unconv = max(unconv, st.get("frac_unconverged", 0.0))
                    iters = max(iters, st.get("newton_iters", 0))
                    qh = st.get("q_ratio", qh)
        d.update(unconv=unconv, iters=iters, qh=qh)
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md")
    a = ap.parse_args()
    by = defaultdict(list)
    for r in load(a.root):
        by[r["arm"]].append(r)
    if not by:
        print("no runs found"); return

    def ms(v):
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
        return m, s

    base = ms([r["final_val_loss"] for r in by["gelu"]])[0] if "gelu" in by else None
    lines = ["| arm | seeds | params | val loss (mean ± sd) | Δ vs gelu | tok/s | max unconverged | max Newton iters | q/h |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(by, key=lambda k: ms([r["final_val_loss"] for r in by[k]])[0]):
        rs = by[arm]
        m, s = ms([r["final_val_loss"] for r in rs])
        tps = sum(r["tokens_per_s"] for r in rs) / len(rs)
        d = f"{m - base:+.4f}" if base is not None else "-"
        qh = rs[0]["qh"]
        lines.append(f"| {arm} | {len(rs)} | {rs[0]['n_params']:,} | {m:.4f} ± {s:.4f} | {d} | "
                     f"{tps:,.0f} | {max(r['unconv'] for r in rs):.3f} | {max(r['iters'] for r in rs)} | "
                     f"{'-' if qh is None else f'{qh:.3f}'} |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()
