# polydeq

**Question:** if you replace the GELU in a transformer's MLP with an implicit
layer defined by a polynomial equation (solved by Newton's method, trained
through the implicit function theorem), does the algebraic structure buy
anything?

**Answer (at this scale):** no. Every polynomial variant did worse than GELU.
The implicit (solved) versions were barely better than having no activation
at all. The unconstrained version diverged.

## The layer

The layer's output y solves

    y = W x + U p(Uᵀ y)

where p is a cubic applied per channel. The polynomial term is low-rank and
grouped: the hidden width is split into groups of 16 units with coupling rank 8,
so Newton only solves an 8×8 system per token per group.

In the default **monotone** form, p = −(α t + β t³) with α, β ≥ 0. The equation
is then the gradient of a strongly convex function, so it has exactly one
solution for every input. Without the polynomial term, the layer is an ordinary
linear layer.

## Setup

- **Model:** a byte-level GPT (6 layers, d=384, about 11M parameters) on enwik8,
  for 5000 steps at batch 64 × 256 bytes. Metric: validation bits per byte.
- **Arms**, all within about 1% of GELU's parameter count:
  - `gelu`: standard MLP
  - `gelu_matched`: GELU with parameters matched to `group_implicit`
  - `none`: no activation (the MLP is linear)
  - `elem_explicit`, `group_explicit`: the same polynomial applied directly, no
    solve (per unit, or grouped)
  - `elem_implicit`, `group_implicit`: the polynomial equation solved by Newton
    (per unit, or grouped). `group_implicit` is the main arm.
  - `group_free`: grouped and implicit, with no constraints on U or p
- 3 seeds per arm, on one RTX 6000 Ada. A difference counts only if it exceeds
  twice the seed standard deviation.

## Results (mean ± sd over 3 seeds)

| arm | bits per byte | vs gelu | throughput (tok/s) |
|---|---|---|---|
| gelu_matched | **1.533 ± 0.003** | −0.002 | 252k |
| gelu | 1.535 ± 0.002 | – | 320k |
| elem_explicit | 1.617 ± 0.006 | +0.082 | 219k |
| group_explicit | 1.639 ± 0.008 | +0.104 | 164k |
| group_implicit | 1.808 ± 0.003 | +0.273 | 61k |
| elem_implicit | 1.809 ± 0.006 | +0.274 | 179k |
| none (linear MLP) | 1.822 ± 0.004 | +0.287 | 355k |
| group_free | diverged by step 151 in all seeds | | |

Every Newton solve in the monotone arms converged (0% of tokens unconverged).
That needed a per-channel closed-form (Cardano) starting point; from the naive
start y = Wx, three Newton steps stopped converging partway through training.

## Findings

1. **GELU wins clearly.** All polynomial arms are worse by far more than seed noise.
2. **Solving the equation made things worse, not better.** The implicit arms
   land almost on the no-activation baseline. Their polynomial term stays small
   relative to the linear part (‖q‖/‖h‖ ≈ 0.3–0.4, vs ≈ 1.4–1.5 for the
   explicit arms). The likely reason: along the coupled directions, the monotone
   implicit layer grows like a cube root, so it is close to linear at the
   scales the network uses. Training didn't escape that.
3. **The explicit polynomial is a real but weak activation:** better than linear,
   well short of GELU.
4. **The unconstrained form is unusable.** A float64 reference check already
   showed Newton losing the root for 10–60% of tokens. In training, the loss
   went non-finite within 151 steps in every seed.
5. **It is also slow.** `group_implicit` runs at about 1/5 of GELU's throughput,
   even with torch.compile and a fused small-matrix solver.

## Predictions vs outcome

Written down before any run:

| outcome | predicted odds | happened? |
|---|---|---|
| group_implicit beats gelu at equal steps | ~15% | no |
| group_implicit beats group_explicit (the solve helps) | ~30% | no |
| elem_implicit within noise of gelu | ~40% | no |
| free form trains without diverging | ~30% | no |

## Next, if revisited

- Start the implicit layer in a strongly nonlinear regime (larger α, β), so it
  can't settle near linear, and check whether it stays there.
- Compare at equal wall-clock as well as equal steps, since the implicit layer
  costs 5× per step.
- Try the polynomial only in some layers, keeping GELU elsewhere.

## Prior art

Implicit (deep-equilibrium) layers in language models are well studied: Bai,
Kolter & Koltun, *Deep Equilibrium Models* (NeurIPS 2019); implicit language
models (arXiv 2502.07827); and Equilibrium Language Models (ICLR 2026). The
specific twist here, a polynomial equation solved by Newton that reduces to a
linear layer when the polynomial is zero, did not turn up in a short search.

## Reproduce

```bash
pip install torch numpy pytest
python -m pytest -q tests                 # layer vs NumPy reference, gradcheck, IFT
(cd ref && python check_ref.py)            # float64 math checks, no torch needed
curl -LO http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip
python -c "from polydeq.data import prepare_text; prepare_text('enwik8', 'data/enwik8')"
python scripts/sweep.py --data data/enwik8 --root runs/main \
  --arms gelu,group_implicit --seeds 0,1,2 -- --steps 5000
python scripts/summarize.py runs/main
```

`scripts/rerun_implicit.sh` has the exact solver settings for the implicit arms.
Per-run logs are in `runs/main/`. `runs/main_it3/` holds the earlier implicit
runs with the naive start, which are excluded because their solves stopped
converging.
