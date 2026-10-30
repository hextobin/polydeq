"""Polynomial implicit (deep-equilibrium) layers. PyTorch.

Transliteration of ref/npref.py (whose math is checked against finite differences in
float64). Read that file's docstring for the full definition. In short:

    hidden width D = G*g, split into G groups of size g, coupling rank k per group
    h = W x + b                       (or h = x when there is no input projection)
    z solves  z = V h + M p(z),  M = V U          (per group, z has k entries)
    y = h + U p(z)                                ->  y = h + Q(y),  Q(y) = U p(V y)
    p(t) = c1 (t-s) + c2 (t-s)^2 + c3 (t-s)^3     (per coupling channel)

form="monotone": V = U^T, c1 = -softplus(alpha), c2 = 0, c3 = -softplus(beta).
    y + U g(U^T y) = h with g increasing: unique root for every input (recommended).
form="free": U, V, c unconstrained. Contains everything, but the root that starts at
    Wx can vanish (fold) or Newton can land on another branch; see ref/check_ref.py [5].

mode="implicit": solve for z with damped Newton (no autograd), then backprop with the
    implicit function theorem: one k x k transpose solve per token and group.
mode="explicit": no solve: z = V h, so y = h + U p(V h). Same parameters and
    polynomial features, zero Newton steps. The control arm that separates
    "polynomial features" from "solving an equation".
backward="unroll": differentiate through a fixed number of undamped Newton steps
    instead of IFT (for comparing the two gradient estimators).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


def _apply_M(M, v):  # (G,k,k) x (N,G,k)
    return torch.einsum("gkl,ngl->ngk", M, v)


def _chol_solve_unrolled(A, b):
    """Solve A x = b for SPD A (..., k, k), b (..., k), with Cholesky unrolled over k.

    Every operation is elementwise over the batch, so torch.compile fuses the whole
    solve into a few kernels: ~3x faster than batched LU and ~7x faster than
    torch.linalg.cholesky for k = 8. Slow in eager mode; meant for small k.
    """
    k = A.shape[-1]
    L = [[None] * k for _ in range(k)]
    for j in range(k):
        d = A[..., j, j] - sum(L[j][m] * L[j][m] for m in range(j)) if j else A[..., j, j]
        L[j][j] = d.clamp_min(1e-12).sqrt()
        for i in range(j + 1, k):
            v = A[..., i, j] - sum(L[i][m] * L[j][m] for m in range(j)) if j else A[..., i, j]
            L[i][j] = v / L[j][j]
    y = [None] * k
    for i in range(k):
        y[i] = (b[..., i] - sum(L[i][m] * y[m] for m in range(i)) if i else b[..., i]) / L[i][i]
    x = [None] * k
    for i in reversed(range(k)):
        x[i] = (y[i] - sum(L[m][i] * x[m] for m in range(i + 1, k)) if i < k - 1 else y[i]) / L[i][i]
    return torch.stack(x, -1)


def _spd_solve(A, b):
    if A.shape[-1] <= 16:
        return _chol_solve_unrolled(A, b)
    L = torch.linalg.cholesky_ex(A)[0]
    return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)


def _lin_solve(M, dp, r, spd: bool, transpose: bool = False):
    """Solve J x = r (or J^T x = r) with J = I - M diag(dp), batched over (N, G).

    spd (monotone form): M = S = U^T U is PSD and D = -dp >= 0, so J = I + S D. With
    A = I + D^1/2 S D^1/2 (SPD, Cholesky):
        J x = r:    A w = D^1/2 r,      x = r - S D^1/2 w
        J^T x = r:  A v = D^1/2 S r,    x = r - D^1/2 v
    """
    k = M.shape[-1]
    if k == 1:
        return r / (1 - M[:, 0, 0] * dp[..., 0]).unsqueeze(-1)
    if spd:
        sq = (-dp).clamp_min(0).sqrt()
        A = torch.eye(k, device=M.device, dtype=M.dtype) + sq.unsqueeze(-1) * M * sq.unsqueeze(-2)
        if transpose:
            return r - sq * _spd_solve(A, sq * _apply_M(M, r))
        return r - _apply_M(M, sq * _spd_solve(A, sq * r))
    J = torch.eye(k, device=M.device, dtype=M.dtype) - M.unsqueeze(0) * dp.unsqueeze(-2)
    if transpose:
        J = J.transpose(-1, -2)
    # solve_ex: no per-call singularity check (avoids a GPU sync)
    return torch.linalg.solve_ex(J, r.unsqueeze(-1))[0].squeeze(-1)


class _IFTGrad(torch.autograd.Function):
    """Identity in the forward pass; in the backward pass maps g -> J^{-T} g.

    Applied to f = a + M p(z*) computed with z* detached, this gives the exact
    implicit-function-theorem gradient of z* with respect to everything inside f.
    J = I - M diag(dp) is passed as (M, dp), so only (N, G, k) is stored per layer.
    """

    @staticmethod
    def forward(ctx, f, M, dp, spd):
        ctx.save_for_backward(M, dp)
        ctx.spd = spd
        return f.clone()

    @staticmethod
    def backward(ctx, grad):
        M, dp = ctx.saved_tensors
        return _lin_solve(M, dp, grad, ctx.spd, transpose=True), None, None, None


class PolyLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        group_size: int,
        rank: int,
        in_dim: int | None = None,
        form: str = "monotone",
        mode: str = "implicit",
        backward: str = "ift",
        max_iters: int = 8,
        tol: float = 1e-4,
        fixed_iters: int = 0,
        line_search: int = 2,
        newton_init: str = "a",
        unroll_steps: int = 3,
        init_u: float = 1.0,
        init_c1: float = 0.05,
        init_c3: float = 0.05,
        bias: bool = True,
        solve_dtype: torch.dtype = torch.float32,
    ):
        """
        dim:        output width D (= G * group_size)
        group_size: g. Use g = dim for one global coupling, g = 1 for elementwise.
        rank:       k, coupling rank per group (k <= g)
        fixed_iters: if > 0, run exactly this many Newton steps with no GPU sync (no
                    convergence check, branch-free half-step safeguard); else damped
                    Newton up to max_iters with an early exit at tol.
        line_search: fixed path only: step lengths tried per Newton step (1, 1/2, ...,
                    2^-(n-1)); each token keeps the one with the smallest residual.
        newton_init: "a" starts Newton at z0 = V h (y0 = Wx). "cardano" (monotone only)
                    starts at the closed-form root of each channel's scalar cubic with
                    the off-diagonal coupling dropped; for k = 1 that is the exact root.
        in_dim:     if given, an input projection W: in_dim -> dim is included (the
                    literal 'F = y - Wx - Q(y)' layer). If None, h = x (the layer acts
                    as a nonlinearity inside an MLP that has its own projections).
        """
        super().__init__()
        assert dim % group_size == 0, "dim must be divisible by group_size"
        assert 1 <= rank <= group_size
        assert form in ("monotone", "free") and mode in ("implicit", "explicit")
        assert backward in ("ift", "unroll")
        self.dim, self.g, self.k, self.G = dim, group_size, rank, dim // group_size
        self.form, self.mode, self.backward_mode = form, mode, backward
        self.max_iters, self.tol, self.unroll_steps = max_iters, tol, unroll_steps
        self.fixed_iters = fixed_iters
        assert line_search >= 1 and newton_init in ("a", "cardano")
        assert newton_init == "a" or form == "monotone", "cardano start needs the monotone form"
        self.line_search, self.newton_init = line_search, newton_init
        self.solve_dtype = solve_dtype
        G, g, k = self.G, self.g, self.k

        self.proj = nn.Linear(in_dim, dim, bias=bias) if in_dim is not None else None

        if form == "monotone":
            # V = U^T. Orthonormal-ish columns scaled by init_u, so M = U^T U ~ init_u^2 I.
            U = torch.randn(G, g, k)
            U = torch.linalg.qr(U).Q[..., :k] * init_u if g > 1 else torch.full((G, 1, 1), init_u)
            self.U = nn.Parameter(U.contiguous())
            self.register_parameter("V", None)
            self.alpha = nn.Parameter(torch.full((G, k), _inv_softplus(init_c1)))
            self.beta = nn.Parameter(torch.full((G, k), _inv_softplus(init_c3)))
            self.register_parameter("c", None)
        else:
            # U = 0 at init: the layer is exactly linear at step 0 and U gets gradient.
            self.U = nn.Parameter(torch.zeros(G, g, k))
            self.V = nn.Parameter(torch.randn(G, k, g) / math.sqrt(g))
            c = torch.zeros(3, G, k)
            c[2] = 1.0
            self.c = nn.Parameter(c)
            self.register_parameter("alpha", None)
            self.register_parameter("beta", None)
        self.s = nn.Parameter(torch.zeros(G, k))

        self.track_stats = False
        self.stats: dict[str, float] = {}

    # --- polynomial -------------------------------------------------------------
    def coeffs(self):
        if self.form == "monotone":
            c1 = -F.softplus(self.alpha)
            c3 = -F.softplus(self.beta)
            return c1, torch.zeros_like(c1), c3
        return self.c[0], self.c[1], self.c[2]

    @staticmethod
    def _p(z, c, s):
        t = z - s
        return c[0] * t + c[1] * t * t + c[2] * t * t * t

    @staticmethod
    def _dp(z, c, s):
        t = z - s
        return c[0] + 2 * c[1] * t + 3 * c[2] * t * t

    def _UV(self):
        if self.form == "monotone":
            return self.U, self.U.transpose(-1, -2)
        return self.U, self.V

    _apply_M = staticmethod(_apply_M)

    @staticmethod
    def _jac(M, dp):  # I - M diag(p'(z)) : (N,G,k,k)
        k = M.shape[-1]
        eye = torch.eye(k, device=M.device, dtype=M.dtype)
        return eye - M.unsqueeze(0) * dp.unsqueeze(-2)

    def _solve_step(self, M, dp, G_):
        return _lin_solve(M, dp, G_, spd=self.form == "monotone")

    @torch.no_grad()
    def _newton(self, a, M, c, s):
        """Damped Newton on z = a + M p(z), starting at z0 = a (i.e. y0 = Wx)."""
        z = a.clone()
        scale = 1.0 + a.abs().amax(dim=(1, 2))
        iters = 0
        res = None
        for it in range(self.max_iters):
            Gz = z - a - self._apply_M(M, self._p(z, c, s))
            res = Gz.abs().amax(dim=(1, 2)) / scale
            if bool((res < self.tol).all()):
                break
            iters = it + 1
            step = self._solve_step(M, self._dp(z, c, s), Gz)
            # Backtracking per token: halve the step until the residual decreases.
            t = torch.ones_like(res)
            for _ in range(6):
                zn = z - t[:, None, None] * step
                rn = (zn - a - self._apply_M(M, self._p(zn, c, s))).abs().amax(dim=(1, 2)) / scale
                bad = rn > (1 - 1e-4 * t) * res
                if not bool(bad.any()):
                    break
                t = torch.where(bad, t * 0.5, t)
            z = z - t[:, None, None] * step
        Gz = z - a - self._apply_M(M, self._p(z, c, s))
        res = Gz.abs().amax(dim=(1, 2)) / scale
        return z, res, iters

    @staticmethod
    def _cardano_diag(a, M, c, s):
        """Per channel, the one real root of z - m p(z) = a with m = M_ii >= 0 and the
        monotone p(t) = c1 t + c3 t^3 (t = z - s, c1, c3 <= 0). In t: A t^3 + B t = R with
        A = -m c3 >= 0, B = 1 - m c1 >= 1, R = a - s. Hyperbolic form of Cardano, stable
        as A -> 0 (where t -> R / B)."""
        m = torch.diagonal(M, dim1=-2, dim2=-1)                      # (G, k)
        A = (-m * c[2]).clamp_min(1e-12)
        B = 1 - m * c[0]
        R = a - s
        t = 2 * torch.sqrt(B / (3 * A)) * torch.sinh(
            torch.asinh(1.5 * R / B * torch.sqrt(3 * A / B)) / 3)
        return t + s

    @torch.no_grad()
    def _newton_fixed(self, a, M, c, s):
        """fixed_iters Newton steps on z = a + M p(z), without a GPU sync.

        Damping per token, branch-free: try step lengths 1, 1/2, ..., 2^-(line_search-1)
        and keep the one with the smallest max-norm residual (never a non-finite one; if
        none decreases the residual, the shortest step is kept).
        """
        z = self._cardano_diag(a, M, c, s) if self.newton_init == "cardano" else a.clone()
        for _ in range(self.fixed_iters):
            Gz = z - a - self._apply_M(M, self._p(z, c, s))
            step = self._solve_step(M, self._dp(z, c, s), Gz)
            best_z, best_r = None, None
            for i in range(self.line_search):
                zn = z - (0.5 ** i) * step
                rn = (zn - a - self._apply_M(M, self._p(zn, c, s))).abs().amax(dim=(1, 2))
                rn = torch.nan_to_num(rn, nan=float("inf"))
                if best_z is None:
                    best_z, best_r = zn, rn
                else:
                    better = rn < best_r
                    best_z = torch.where(better[:, None, None], zn, best_z)
                    best_r = torch.minimum(rn, best_r)
            z = best_z
        res = None
        if self.track_stats:  # final residual is a diagnostic only
            res = (z - a - self._apply_M(M, self._p(z, c, s))).abs().amax(dim=(1, 2))
            res = res / (1.0 + a.abs().amax(dim=(1, 2)))
        return z, res, self.fixed_iters

    # --- forward ----------------------------------------------------------------
    def forward(self, x):
        h = self.proj(x) if self.proj is not None else x
        # The solve runs in solve_dtype (float32) even under bf16/fp16 autocast.
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self._forward_core(h)

    def _forward_core(self, h):
        shape = h.shape
        out_dtype = h.dtype
        h = h.reshape(-1, self.G, self.g).to(self.solve_dtype)
        U, V = self._UV()
        U, V = U.to(self.solve_dtype), V.to(self.solve_dtype)
        c = tuple(ci.to(self.solve_dtype) for ci in self.coeffs())
        s = self.s.to(self.solve_dtype)

        a = torch.einsum("gki,ngi->ngk", V, h)
        M = torch.einsum("gki,gil->gkl", V, U)

        if self.mode == "explicit":
            z = a
            stats = None
        elif self.backward_mode == "unroll":
            z = a
            for _ in range(self.unroll_steps):
                Gz = z - a - self._apply_M(M, self._p(z, c, s))
                z = z - self._solve_step(M, self._dp(z, c, s), Gz)
            stats = None
        else:
            newton = (self._newton_fixed if self.fixed_iters > 0 or self.newton_init == "cardano"
                  else self._newton)
            z_star, res, iters = newton(a.detach(), M.detach(),
                                        tuple(ci.detach() for ci in c), s.detach())
            f = a + self._apply_M(M, self._p(z_star, c, s))     # = z* at convergence
            if torch.is_grad_enabled() and f.requires_grad:
                dp = self._dp(z_star, tuple(ci.detach() for ci in c), s.detach())
                z = _IFTGrad.apply(f, M.detach(), dp, self.form == "monotone")
            else:
                z = f
            stats = (res, iters, z_star)

        q = torch.einsum("gik,ngk->ngi", U, self._p(z, c, s))
        y = h + q

        if self.track_stats:
            self._record(h, q, M, c, s, stats)
        return y.to(out_dtype).reshape(shape)

    @torch.no_grad()
    def _record(self, h, q, M, c, s, stats):
        st = {"q_ratio": float(q.norm() / (h.norm() + 1e-12))}
        c1, c2, c3 = c
        st["c1_mean"], st["c3_mean"] = float(c1.mean()), float(c3.mean())
        if stats is not None:
            res, iters, z_star = stats
            st["newton_iters"] = iters
            st["res_max"] = float(res.max())
            st["res_mean"] = float(res.mean())
            st["frac_unconverged"] = float((res > self.tol).float().mean())
            if self.k > 1:
                idx = torch.randperm(z_star.shape[0], device=z_star.device)[:256]
                J = self._jac(M, self._dp(z_star[idx], c, s))
                sv = torch.linalg.svdvals(J)
                cond = sv[..., 0] / sv[..., -1].clamp_min(1e-12)
                st["jac_cond_max"] = float(cond.max())
                st["jac_smin_min"] = float(sv[..., -1].min())
            else:
                J = 1 - (M[..., 0, 0] * self._dp(z_star, c, s)[..., 0])
                st["jac_smin_min"] = float(J.abs().min())
        self.stats = st

    def extra_repr(self):
        return (f"dim={self.dim}, groups={self.G}, group_size={self.g}, rank={self.k}, "
                f"form={self.form}, mode={self.mode}, backward={self.backward_mode}, "
                f"proj={'yes' if self.proj is not None else 'no'}")
