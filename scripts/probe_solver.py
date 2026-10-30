"""Solver probe: from a checkpoint, capture each PolyLayer's input on a val batch and
report, per solver variant, the worst-layer fraction of tokens with residual > tol.

    python scripts/probe_solver.py runs/probe/group_implicit/ckpt.pt data/enwik8
"""
import itertools
import sys

import torch

sys.path.insert(0, ".")
from polydeq.data import ByteData  # noqa: E402
from polydeq.model import GPT, GPTConfig  # noqa: E402

ckpt = torch.load(sys.argv[1], map_location="cuda")
cfg = GPTConfig(**{k: v for k, v in ckpt["cfg"].items()})
model = GPT(cfg).cuda().eval()
model.load_state_dict(ckpt["model"])
data = ByteData(sys.argv[2], cfg.block_size, "cuda")
x, y = data.batch("val", 32, torch.Generator().manual_seed(1))

inputs = {}
hooks = [m.register_forward_pre_hook(lambda mod, inp, i=i: inputs.__setitem__(i, inp[0].detach()))
         for i, m in model.poly_layers()]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model(x, y)
for h in hooks:
    h.remove()
layers = dict(model.poly_layers())
k = next(iter(layers.values())).k


def worst(fixed_iters, line_search, init):
    fr, rm = 0.0, 0.0
    for i, lay in layers.items():
        lay.fixed_iters, lay.line_search, lay.newton_init = fixed_iters, line_search, init
        lay.track_stats = True
        with torch.no_grad():
            lay(inputs[i])
        fr, rm = max(fr, lay.stats["frac_unconverged"]), max(rm, lay.stats["res_max"])
    return fr, rm


# reference: adaptive damped Newton, many iterations
for lay in layers.values():
    lay.max_iters, lay.tol = 100, 1e-4
fr, rm = worst(0, 2, "a")
print(f"k={k}  adaptive damped (max 100 iters): unconverged {fr:.4f}  res_max {rm:.1e}")
for lay in layers.values():
    lay.max_iters, lay.tol = cfg.max_iters, cfg.tol
fr, rm = worst(0, 2, "a")
print(f"k={k}  adaptive damped (max {cfg.max_iters} iters, old default): unconverged {fr:.4f}  res_max {rm:.1e}")

iters = [0, 1, 2] if k == 1 else [2, 3, 4, 6, 8]
for init, ls, n in itertools.product(["a", "cardano"], [1, 2, 4], iters):
    if init == "a" and n == 0:
        continue
    fr, rm = worst(n, ls, init)
    print(f"init={init:8s} line_search={ls}  iters={n}:  unconverged {fr:.4f}  res_max {rm:.1e}")
