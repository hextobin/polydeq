# polydeq

Polynomial deep-equilibrium layers as a drop-in for a transformer's MLP nonlinearity.

    y solves  y = W x + U p(V y)        (p cubic per channel; Newton forward, IFT backward)

See `CLAUDE.md` for the design, current status and run plan.

Quick start (needs PyTorch):

    pip install torch numpy pytest
    python -m pytest -q tests
    bash scripts/smoke.sh

NumPy-only math checks (no torch): `cd ref && python check_ref.py`
