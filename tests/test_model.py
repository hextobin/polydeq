"""Every arm builds, runs forward/backward, and matched arms really are matched."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from polydeq.model import ARMS, GPT, GPTConfig, build_slot, slot_params  # noqa: E402


def tiny(arm, **kw):
    return GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, d_model=32,
                     arm=arm, group_size=8, rank=4, lowrank_rank=8, **kw)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_forward_backward(arm):
    torch.manual_seed(0)
    cfg = tiny(arm)
    m = GPT(cfg).float()
    x = torch.randint(0, 256, (3, 16))
    _, loss = m(x, x)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_matched_params():
    cfg = tiny("gelu_matched", match_to="group_implicit")
    target = slot_params(cfg, "group_implicit")
    got = sum(p.numel() for p in build_slot(cfg, "gelu_matched").parameters())
    assert abs(got - target) / target < 0.02
    cfg = tiny("slot_mlp_matched", match_to="slot_implicit")
    target = slot_params(cfg, "slot_implicit")
    got = sum(p.numel() for p in build_slot(cfg, "slot_mlp_matched").parameters())
    assert abs(got - target) / target < 0.05


def test_poly_layers_subset():
    m = GPT(tiny("group_implicit", poly_layers="1"))
    assert [i for i, _ in m.poly_layers()] == [1]
