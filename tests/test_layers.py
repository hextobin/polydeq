"""Correctness tests for the PyTorch layer.  Run from the repo root:  python -m pytest -q

The NumPy reference (ref/npref.py) is checked against finite differences by
ref/check_ref.py; here the PyTorch layer is checked against that reference and
against torch.autograd.gradcheck.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import npref  # noqa: E402
from polydeq.layers import PolyLayer  # noqa: E402

torch.set_default_dtype(torch.float64)
CONFIGS = [(1, 8, 3), (4, 2, 2), (8, 1, 1)]   # (G, g, k): global, grouped, elementwise


def make_layer(G, g, k, form, din=6, scale=1.0, seed=0, **kw):
    torch.manual_seed(seed)
    lay = PolyLayer(G * g, g, k, in_dim=din, form=form, max_iters=60, tol=1e-13,
                    solve_dtype=torch.float64, **kw)
    with torch.no_grad():
        if form == "free":
            lay.U.normal_(0, scale / g**0.5)
            lay.c.copy_(torch.tensor([0.3, 0.2, 0.5])[:, None, None].expand(3, G, k))
        else:
            lay.alpha.fill_(0.2)
            lay.beta.fill_(0.5)
        lay.s.normal_(0, 0.1)
    return lay


def to_ref(lay: PolyLayer):
    P = dict(W=lay.proj.weight.detach().numpy().copy(), b=lay.proj.bias.detach().numpy().copy(),
             U=lay.U.detach().numpy().copy(), s=lay.s.detach().numpy().copy())
    c1, c2, c3 = (ci.detach().numpy() for ci in lay.coeffs())
    P["c"] = np.stack([c1, c2, c3])
    P["V"] = (lay.V.detach().numpy().copy() if lay.form == "free"
              else np.transpose(P["U"], (0, 2, 1)).copy())
    return P


@pytest.mark.parametrize("G,g,k", CONFIGS)
@pytest.mark.parametrize("form", ["free", "monotone"])
def test_matches_numpy_reference(G, g, k, form):
    lay = make_layer(G, g, k, form, scale=0.3)
    x = torch.randn(10, 6) * (0.2 if form == "free" else 1.0)
    x.requires_grad_(True)
    y = lay(x)
    R = torch.randn_like(y)
    loss = (R * y).sum() + 0.1 * (y ** 2).sum()
    loss.backward()

    P = to_ref(lay)
    tied = form == "monotone"
    y_ref, cache = npref.forward(P, x.detach().numpy(), tied=tied)
    # free form: a test regime where the root branch from Wx exists (see check_ref.py [5])
    assert cache["hist"][-1] < 1e-10, "reference Newton did not converge"
    assert np.allclose(y.detach().numpy(), y_ref, atol=1e-9), "forward mismatch"
    g_ref = npref.backward(P, x.detach().numpy(), R.numpy() + 0.2 * y_ref, cache, tied=tied)
    assert np.allclose(lay.proj.weight.grad.numpy(), g_ref["W"], atol=1e-8)
    assert np.allclose(lay.proj.bias.grad.numpy(), g_ref["b"], atol=1e-8)
    assert np.allclose(lay.U.grad.numpy(), g_ref["U"], atol=1e-8)
    assert np.allclose(lay.s.grad.numpy(), g_ref["s"], atol=1e-8)
    if form == "free":
        assert np.allclose(lay.V.grad.numpy(), g_ref["V"], atol=1e-8)
        assert np.allclose(lay.c.grad.numpy(), g_ref["c"], atol=1e-8)


@pytest.mark.parametrize("G,g,k", CONFIGS)
@pytest.mark.parametrize("form", ["free", "monotone"])
def test_gradcheck(G, g, k, form):
    lay = make_layer(G, g, k, form, scale=0.3)
    x = (torch.randn(4, 6) * (0.2 if form == "free" else 1.0)).requires_grad_(True)
    names = [n for n, _ in lay.named_parameters()]
    assert torch.autograd.gradcheck(lambda x_: lay(x_), (x,), eps=1e-6, atol=1e-6)
    # gradients with respect to every parameter, through a functional wrapper
    assert torch.autograd.gradcheck(
        lambda *ps: torch.func.functional_call(lay, dict(zip(names, ps)), (x.detach(),)),
        tuple(p.detach().clone().requires_grad_(True) for p in lay.parameters()),
        eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("G,g,k", CONFIGS)
def test_zero_Q_is_linear(G, g, k):
    for form in ("free", "monotone"):
        lay = make_layer(G, g, k, form)
        with torch.no_grad():
            lay.U.zero_()
        x = torch.randn(5, 6)
        assert torch.equal(lay(x), lay.proj(x))
    # free form is exactly linear at initialisation (U = 0)
    lay = PolyLayer(G * g, g, k, in_dim=6, form="free")
    x = torch.randn(5, 6, dtype=torch.float32)
    lay = lay.float()
    assert torch.equal(lay(x), lay.proj(x))


def test_ift_equals_long_unroll():
    lay = make_layer(4, 2, 2, "monotone")
    x = torch.randn(7, 6)
    y1 = lay(x); y1.square().sum().backward()
    g1 = [p.grad.clone() for p in lay.parameters()]
    lay.zero_grad()
    lay.backward_mode, lay.unroll_steps = "unroll", 40
    y2 = lay(x); y2.square().sum().backward()
    g2 = [p.grad.clone() for p in lay.parameters()]
    assert torch.allclose(y1, y2, atol=1e-9)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-7)


def test_explicit_mode():
    lay = make_layer(4, 2, 2, "monotone", mode="explicit")
    x = torch.randn(3, 6)
    h = lay.proj(x).reshape(-1, 4, 2)
    U = lay.U
    a = torch.einsum("gik,ngi->ngk", U, h)
    expect = h + torch.einsum("gik,ngk->ngi", U, lay._p(a, lay.coeffs(), lay.s))
    assert torch.allclose(lay(x), expect.reshape(3, 8))


def test_monotone_always_converges():
    torch.manual_seed(1)
    lay = PolyLayer(64, 16, 8, in_dim=32, form="monotone", max_iters=20, tol=1e-6,
                    solve_dtype=torch.float64, init_u=2.0, init_c1=1.0, init_c3=3.0)
    lay.track_stats = True
    x = torch.randn(512, 32) * 5
    lay(x)
    assert lay.stats["frac_unconverged"] == 0.0, lay.stats
    # J = I + (U^T U) D with D >= 0 is similar to an SPD matrix: eigenvalues are real, >= 1
    with torch.no_grad():
        h = lay.proj(x).reshape(-1, lay.G, lay.g)
        U, V = lay._UV()
        a = torch.einsum("gki,ngi->ngk", V, h)
        M = torch.einsum("gki,gil->gkl", V, U)
        c = lay.coeffs()
        z, _, _ = lay._newton(a, M, c, lay.s)
        ev = torch.linalg.eigvals(lay._jac(M, lay._dp(z, c, lay.s)))
    assert ev.real.min() >= 1 - 1e-9 and ev.imag.abs().max() < 1e-6


def test_float32_and_bf16_autocast():
    lay = PolyLayer(64, 16, 8, in_dim=32).float()
    x = torch.randn(2, 5, 32, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = lay(x)
    assert y.shape == (2, 5, 64) and torch.isfinite(y).all()
    y.float().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in lay.parameters() if p.grad is not None)


@pytest.mark.parametrize("G,g,k", CONFIGS)
@pytest.mark.parametrize("form", ["free", "monotone"])
def test_fixed_iters_matches_adaptive(G, g, k, form):
    """Sync-free fixed-iteration Newton reaches the same root, and its IFT gradient is exact."""
    lay = make_layer(G, g, k, form, scale=0.3)
    x = (torch.randn(10, 6) * (0.2 if form == "free" else 1.0)).requires_grad_(True)
    y_ad = lay(x)
    lay.fixed_iters = 30
    y_fx = lay(x)
    assert torch.allclose(y_fx, y_ad, atol=1e-9)
    assert torch.autograd.gradcheck(lambda x_: lay(x_), (x[:4].detach().requires_grad_(True),),
                                    eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("G,g,k", CONFIGS)
def test_cardano_start(G, g, k):
    """Closed-form start: exact for k = 1 with zero Newton steps; a valid start otherwise."""
    lay = make_layer(G, g, k, "monotone")
    with torch.no_grad():
        lay.beta.fill_(2.0)                     # strong cubic, where z0 = Wx is far off
    x = (torch.randn(10, 6) * 3).requires_grad_(True)
    y_ad = lay(x)
    lay.newton_init = "cardano"
    lay.fixed_iters = 0 if k == 1 else 30
    lay.line_search = 4
    y_c = lay(x)
    assert torch.allclose(y_c, y_ad, atol=1e-8)
    assert torch.autograd.gradcheck(lambda x_: lay(x_), (x[:4].detach().requires_grad_(True),),
                                    eps=1e-6, atol=1e-6)
