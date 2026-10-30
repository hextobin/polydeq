"""NumPy float64 reference for the polynomial implicit layer (grouped form).

Checks the math (Newton solve, IFT backward) independently of PyTorch. The PyTorch
layer in polydeq/layers.py is a line-for-line transliteration of this file.

Shapes (row form): x (N, d_in); hidden width D = G * g, split into G groups of size g.
Each group has its own coupling of rank k:

    h        = x W^T + b                                   (N, D)  -> (N, G, g)
    a        = V h            per group, V: (G, k, g)      (N, G, k)
    M        = V U            per group, U: (G, g, k)      (G, k, k)
    z  solves  z = a + M p(z)                              (N, G, k)
    y        = h + U p(z)                                  (N, G, g)  -> (N, D)

    p(t) = c1 (t-s) + c2 (t-s)^2 + c3 (t-s)^3, per channel: c (3, G, k), s (G, k)

So Q(y) = U p(V y) is a degree-3 polynomial in y, block-diagonal over groups.
  G=1, g=D, k=r : one global low-rank coupling (the original proposal)
  g=k=1         : elementwise, i.e. a learned activation y = (t - p(t))^-1 (h)
  1 < k <= g    : grouped coupling (polynomial systems of size k per group)

"free"     : U, V independent, any c.
"monotone" : V = U^T, c1 <= 0, c2 = 0, c3 <= 0. Then y + U g(U^T y) = h with g
             monotone, which is the gradient of a strongly convex function: unique
             root for every input, Jacobian similar to an SPD matrix with eigenvalues >= 1.
"""
import numpy as np


def poly(z, c, s):
    t = z - s
    return c[0] * t + c[1] * t**2 + c[2] * t**3


def dpoly(z, c, s):
    t = z - s
    return c[0] + 2 * c[1] * t + 3 * c[2] * t**2


def apply_M(M, v):            # (G,k,k) x (N,G,k) -> (N,G,k)
    return np.einsum("gkl,ngl->ngk", M, v)


def jac(M, dp):               # J = I - M diag(p'(z)) : (N,G,k,k)
    k = M.shape[-1]
    return np.eye(k) - M[None] * dp[:, :, None, :]


def newton(a, M, c, s, iters=50, tol=1e-12, damp=True):
    z = a.copy()
    hist = []
    for it in range(iters):
        G = z - a - apply_M(M, poly(z, c, s))
        res = np.abs(G).max(axis=(1, 2))
        hist.append(res.max())
        if res.max() < tol:
            break
        step = np.linalg.solve(jac(M, dpoly(z, c, s)), G[..., None])[..., 0]
        t = np.ones(len(z))
        if damp:
            for _ in range(30):
                zn = z - t[:, None, None] * step
                rn = np.abs(zn - a - apply_M(M, poly(zn, c, s))).max(axis=(1, 2))
                bad = rn > (1 - 1e-4 * t) * res
                if not bad.any():
                    break
                t = np.where(bad, t * 0.5, t)
        z = z - t[:, None, None] * step
    return z, np.array(hist), it


def unpack(P, tied):
    U = P["U"]
    V = np.transpose(U, (0, 2, 1)) if tied else P["V"]
    return U, V


def forward(P, x, tied=False, iters=50, G_=None):
    W, b, c, s = P["W"], P["b"], P["c"], P["s"]
    U, V = unpack(P, tied)
    G, g, k = U.shape
    h = (x @ W.T + b).reshape(len(x), G, g)
    a = np.einsum("gki,ngi->ngk", V, h)
    M = np.einsum("gki,gil->gkl", V, U)
    z, hist, it = newton(a, M, c, s, iters=iters)
    y = h + np.einsum("gik,ngk->ngi", U, poly(z, c, s))
    return y.reshape(len(x), G * g), dict(h=h, M=M, z=z, U=U, V=V, hist=hist, k=it)


def backward(P, x, gy, cache, tied=False):
    c, s = P["c"], P["s"]
    h, M, z, U, V = cache["h"], cache["M"], cache["z"], cache["U"], cache["V"]
    N, G, g = h.shape
    gy = gy.reshape(N, G, g)
    pz, dpz = poly(z, c, s), dpoly(z, c, s)
    gp_direct = np.einsum("gik,ngi->ngk", U, gy)              # U^T gy
    J = jac(M, dpz)
    lam = np.linalg.solve(np.swapaxes(J, -1, -2), (gp_direct * dpz)[..., None])[..., 0]
    gM = np.einsum("ngk,ngl->gkl", lam, pz)                   # sum_n lam p^T
    gU = np.einsum("ngi,ngk->gik", gy, pz) + np.einsum("gki,gkl->gil", V, gM)
    gV = np.einsum("gkl,gil->gki", gM, U) + np.einsum("ngk,ngi->gki", lam, h)
    gh = gy + np.einsum("gki,ngk->ngi", V, lam)
    gp = gp_direct + np.einsum("gkl,ngk->ngl", M, lam)         # incl. z-equation path
    t = z - s
    gc = np.stack([(gp * t**e).sum(0) for e in (1, 2, 3)])
    gs = -(gp * dpz).sum(0)
    gh = gh.reshape(N, G * g)
    out = dict(W=gh.T @ x, b=gh.sum(0), U=gU, V=gV, c=gc, s=gs)
    if tied:
        out["U"] = gU + np.transpose(gV, (0, 2, 1))
        out["V"] = np.zeros_like(P["V"])
    return out


def unrolled_forward(P, x, K, tied=False):
    """K plain Newton steps from z0 = a (no damping)."""
    W, b, c, s = P["W"], P["b"], P["c"], P["s"]
    U, V = unpack(P, tied)
    G, g, k = U.shape
    h = (x @ W.T + b).reshape(len(x), G, g)
    a = np.einsum("gki,ngi->ngk", V, h)
    M = np.einsum("gki,gil->gkl", V, U)
    z = a.copy()
    for _ in range(K):
        Gz = z - a - apply_M(M, poly(z, c, s))
        z = z - np.linalg.solve(jac(M, dpoly(z, c, s)), Gz[..., None])[..., 0]
    return (h + np.einsum("gik,ngk->ngi", U, poly(z, c, s))).reshape(len(x), G * g)
