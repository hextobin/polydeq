"""Math checks on the NumPy reference. Run from ref/:  python3 check_ref.py"""
import numpy as np
from npref import forward, backward, unrolled_forward, poly

rng = np.random.default_rng(0)
N, din = 12, 10
ok = True


def make(G, g, k, scale_U=0.3, tied=False, c=(0.0, 0.0, 1.0), shift=0.0):
    D = G * g
    return dict(
        W=rng.normal(size=(D, din)) / np.sqrt(din),
        b=rng.normal(size=D) * 0.1,
        U=rng.normal(size=(G, g, k)) * scale_U / np.sqrt(g),
        V=rng.normal(size=(G, k, g)) / np.sqrt(g),
        c=np.tile(np.array(c, float)[:, None, None], (1, G, k)),
        s=rng.normal(size=(G, k)) * shift,
    )


x = rng.normal(size=(N, din))


def lossfn(y, R):
    return (R * y).sum() + 0.1 * (y**2).sum(), R + 0.2 * y


def fd_check(P, tied, label, x=x):
    global ok
    y, cache = forward(P, x, tied=tied)
    R = np.random.default_rng(1).normal(size=y.shape)
    _, gy = lossfn(y, R)
    g = backward(P, x, gy, cache, tied=tied)
    worst = 0.0
    for name in ["W", "b", "U", "V", "c", "s"]:
        if tied and name == "V":
            continue
        A = P[name]
        num = np.zeros_like(A)
        for i in np.ndindex(A.shape):
            if tied and name == "c" and i[0] == 1:
                num[i] = g["c"][i]
                continue
            old = A[i]
            A[i] = old + 1e-6; lp = lossfn(forward(P, x, tied=tied)[0], R)[0]
            A[i] = old - 1e-6; lm = lossfn(forward(P, x, tied=tied)[0], R)[0]
            A[i] = old
            num[i] = (lp - lm) / 2e-6
        worst = max(worst, np.abs(num - g[name]).max() / (np.abs(num).max() + 1e-12))
    print(f"{label}: Newton iters={cache['k']}, residual={cache['hist'][-1]:.1e}, "
          f"IFT vs finite-diff worst rel err={worst:.1e}")
    ok &= worst < 1e-6


# 1. Q = 0 reduces to the linear layer exactly.
for G, g, k in [(1, 8, 3), (4, 2, 2)]:
    P = make(G, g, k); P["U"][:] = 0
    e1 = np.abs(forward(P, x)[0] - (x @ P["W"].T + P["b"])).max()
    P = make(G, g, k); P["c"][:] = 0
    e2 = np.abs(forward(P, x)[0] - (x @ P["W"].T + P["b"])).max()
    print(f"[1] G={G},g={g},k={k}: U=0 -> linear err {e1:.0e};  c=0 -> linear err {e2:.0e}")
    ok &= e1 == 0 and e2 == 0

# 2. IFT gradients vs finite differences. Free variant in a mild regime so every token
#    stays on the branch connected to Wx (check [5] shows what happens otherwise).
fd_check(make(1, 8, 3, 0.3, shift=0.1), False, "[2] free,     global low-rank  G=1 g=8 k=3", x=0.2 * x)
fd_check(make(4, 2, 2, 0.3, shift=0.1), False, "[2] free,     grouped          G=4 g=2 k=2", x=0.2 * x)
fd_check(make(8, 1, 1, 0.3, shift=0.1), False, "[2] free,     elementwise      G=8 g=1 k=1", x=0.2 * x)
mono = (-0.5, 0.0, -1.0)
fd_check(make(1, 8, 3, 3.0, True, mono, 0.3), True, "[3] monotone, global low-rank  G=1 g=8 k=3")
fd_check(make(4, 2, 2, 3.0, True, mono, 0.3), True, "[3] monotone, grouped          G=4 g=2 k=2")
fd_check(make(8, 1, 1, 3.0, True, mono, 0.3), True, "[3] monotone, elementwise      G=8 g=1 k=1")

# 4. Few-step accuracy: max|y_K - y*| after K undamped Newton steps from z0 = Wx.
print("[4] max|y_K - y*| after K Newton steps (K=0..5), unit-scale inputs")
for label, P, tied in [
    ("free  G=1 k=3 sU=0.5", make(1, 8, 3, 0.5), False),
    ("mono  G=1 k=3 sU=1  ", make(1, 8, 3, 1.0, True, mono), True),
    ("mono  G=4 k=2 sU=1  ", make(4, 2, 2, 1.0, True, mono), True),
]:
    y_star, cache = forward(P, x, tied=tied)
    conv = cache["hist"][-1] < 1e-10
    errs = [np.abs(unrolled_forward(P, x, K, tied) - y_star).max() for K in range(6)]
    print(f"    {label}: " + " ".join(f"{e:.0e}" for e in errs) + ("" if conv else "  (y* not converged)"))

# 5. Robustness: fraction of rows where damped Newton fails to find the root that is
#    continuously connected to Wx (|G| > 1e-8 after 50 steps).
print("[5] fraction of tokens NOT converged after 50 damped Newton steps (G=1, g=8, k=3)")
base = make(1, 8, 3, 1.0)
for xs_ in (0.5, 1.0, 2.0):
    xs = rng.normal(size=(2000, din)) * xs_
    row = []
    for sU in (0.25, 1, 4):
        for tied, c in [(False, (0, 0, 1.0)), (False, (0, 1.0, 0)), (True, mono)]:
            P = {kk: v.copy() for kk, v in base.items()}
            P["U"] *= sU
            P["c"] = np.tile(np.array(c, float)[:, None, None], (1, 1, 3))
            _, cache = forward(P, xs, tied=tied)
            z, h, M, U, V = cache["z"], cache["h"], cache["M"], cache["U"], cache["V"]
            a = np.einsum("gki,ngi->ngk", V, h)
            res = np.abs(z - a - np.einsum("gkl,ngl->ngk", M, poly(z, P["c"], P["s"]))).max(axis=(1, 2))
            row.append((res > 1e-8).mean())
    print(f"    |x|~{xs_}:  " + "   ".join(
        f"sU={sU}: cubic {row[3*i]:.2f} quad {row[3*i+1]:.2f} mono {row[3*i+2]:.2f}"
        for i, sU in enumerate((0.25, 1, 4))))

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
