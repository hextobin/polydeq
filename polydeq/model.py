"""Small GPT with a swappable MLP slot (nanoGPT-style, byte-level by default).

Arms (what goes in the MLP slot of the chosen layers; other layers keep GELU):

Family "act" -- keep the MLP's projections d -> m -> d, replace the activation:
    gelu              Linear(d,m) . GELU . Linear(m,d)                 (standard)
    gelu_matched      same, hidden width chosen to match --match_to's parameter count
    none              Linear(d,m) . Linear(m,d)                        (linear floor)
    elem_explicit     ... . y = h + p(h) elementwise (monotone form) . ...
    elem_implicit     ... . y solves y = h + p(y) elementwise   . ...   (learned activation)
    group_explicit    ... . y = h + U p(U^T h) per group of g units . ...
    group_implicit    ... . y solves y = h + U p(U^T y) per group     (coupled; main arm)
    group_free        group_implicit with the free form (U, V, c unconstrained)
    lowrank_implicit  one global coupling of rank r over all m units (G=1)

Family "slot" -- replace the whole MLP with one d -> d layer (the literal proposal
F(x,y) = y - Wx - Q(y)):
    slot_linear       y = W x
    slot_implicit     y solves y = W x + U p(U^T y), one global coupling of rank r
    slot_group        same, grouped coupling
    slot_explicit     y = W x + U p(U^T W x)
    slot_mlp_matched  GELU MLP with hidden width matched to --match_to if it is a slot
                      arm, else to slot_implicit

Implicit arms default to the monotone form (unique root); see layers.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PolyLayer

ACT_ARMS = ["gelu", "gelu_matched", "none", "elem_explicit", "elem_implicit",
            "group_explicit", "group_implicit", "group_free", "lowrank_implicit"]
SLOT_ARMS = ["slot_linear", "slot_implicit", "slot_group", "slot_explicit", "slot_mlp_matched"]
ARMS = ACT_ARMS + SLOT_ARMS


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.0
    arm: str = "gelu"
    poly_layers: str = "all"          # "all" or comma-separated layer indices
    group_size: int = 16              # g for group arms
    rank: int = 8                     # k for group arms
    lowrank_rank: int = 32            # r for lowrank_implicit / slot_implicit
    max_iters: int = 8
    fixed_iters: int = 3              # > 0: sync-free fixed Newton steps; 0: adaptive
    line_search: int = 2              # step lengths tried per fixed Newton step
    newton_init: str = "a"            # "a" (y0 = Wx) or "cardano" (monotone arms only)
    tol: float = 1e-4
    backward: str = "ift"             # "ift" or "unroll"
    unroll_steps: int = 3
    init_u: float = 1.0
    init_c1: float = 0.05
    init_c3: float = 0.05
    match_to: str = "group_implicit"  # reference arm for *_matched arms
    extra: dict = field(default_factory=dict)


class GeluMLP(nn.Module):
    def __init__(self, d, m, act=True, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d, m)
        self.proj = nn.Linear(m, d)
        self.act = act
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc(x)
        if self.act:
            h = F.gelu(h)
        return self.drop(self.proj(h))


class ActMLP(nn.Module):
    """Linear(d,m) . PolyLayer(m) . Linear(m,d)"""

    def __init__(self, d, m, poly: PolyLayer, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d, m)
        self.poly = poly
        self.proj = nn.Linear(m, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(self.poly(self.fc(x))))


def _poly(cfg: GPTConfig, dim, g, k, form="monotone", mode="implicit", in_dim=None):
    return PolyLayer(dim, g, k, in_dim=in_dim, form=form, mode=mode, backward=cfg.backward,
                     max_iters=cfg.max_iters, tol=cfg.tol, fixed_iters=cfg.fixed_iters,
                     line_search=cfg.line_search,
                     newton_init=cfg.newton_init if form == "monotone" else "a",
                     unroll_steps=cfg.unroll_steps,
                     init_u=cfg.init_u, init_c1=cfg.init_c1, init_c3=cfg.init_c3)


def slot_params(cfg: GPTConfig, arm: str) -> int:
    return sum(p.numel() for p in build_slot(cfg, arm, match=False).parameters())


def _matched_hidden(cfg: GPTConfig, arm: str) -> int:
    d = cfg.d_model
    ref = cfg.match_to
    if arm == "slot_mlp_matched" and ref not in SLOT_ARMS:
        ref = "slot_implicit"  # the slot control matches a slot arm, not the act family
    target = slot_params(cfg, ref)
    return max(1, round((target - d) / (2 * d + 1)))  # GELU MLP params = 2dm + m + d


def build_slot(cfg: GPTConfig, arm: str, match=True) -> nn.Module:
    d, m = cfg.d_model, cfg.mlp_ratio * cfg.d_model
    g, k = cfg.group_size, cfg.rank
    if arm == "gelu":
        return GeluMLP(d, m, True, cfg.dropout)
    if arm == "none":
        return GeluMLP(d, m, False, cfg.dropout)
    if arm in ("gelu_matched", "slot_mlp_matched"):
        if not match:
            raise ValueError("matched arms cannot be a match_to reference")
        return GeluMLP(d, _matched_hidden(cfg, arm), True, cfg.dropout)
    if arm == "elem_explicit":
        return ActMLP(d, m, _poly(cfg, m, 1, 1, mode="explicit"), cfg.dropout)
    if arm == "elem_implicit":
        return ActMLP(d, m, _poly(cfg, m, 1, 1), cfg.dropout)
    if arm == "group_explicit":
        return ActMLP(d, m, _poly(cfg, m, g, k, mode="explicit"), cfg.dropout)
    if arm == "group_implicit":
        return ActMLP(d, m, _poly(cfg, m, g, k), cfg.dropout)
    if arm == "group_free":
        return ActMLP(d, m, _poly(cfg, m, g, k, form="free"), cfg.dropout)
    if arm == "lowrank_implicit":
        return ActMLP(d, m, _poly(cfg, m, m, cfg.lowrank_rank), cfg.dropout)
    if arm == "slot_linear":
        return nn.Linear(d, d)
    if arm == "slot_implicit":
        return _poly(cfg, d, d, cfg.lowrank_rank, in_dim=d)
    if arm == "slot_group":
        return _poly(cfg, d, g, k, in_dim=d)
    if arm == "slot_explicit":
        return _poly(cfg, d, d, cfg.lowrank_rank, mode="explicit", in_dim=d)
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, arm: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = build_slot(cfg, arm)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.poly_layers == "all":
            poly_idx = set(range(cfg.n_layer))
        else:
            poly_idx = {int(i) for i in cfg.poly_layers.split(",") if i.strip()}
        self.poly_idx = sorted(poly_idx)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg, cfg.arm if i in poly_idx else "gelu") for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for n, p in self.named_parameters():   # GPT-2 scaled init on residual projections
            if n.endswith("proj.weight") and p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(mod):
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0.0, 0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, 0.0, 0.02)

    def poly_layers(self):
        return [(i, m) for i, b in enumerate(self.blocks) for m in b.modules()
                if isinstance(m, PolyLayer)]

    def set_tracking(self, on: bool):
        for _, m in self.poly_layers():
            m.track_stats = on

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
