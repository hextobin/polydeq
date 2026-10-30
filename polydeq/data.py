"""Byte-level data (vocab 256, no tokenizer dependency).

A dataset is a directory with train.bin and val.bin (raw uint8 bytes).
  prepare_text(src_txt, out_dir)  -> splits a text file 90/10
  synth_arith(out_dir)            -> small synthetic corpus for CPU smoke tests
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def prepare_text(src: str, out_dir: str, val_frac: float = 0.1) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = np.fromfile(src, dtype=np.uint8)
    n = int(len(data) * (1 - val_frac))
    data[:n].tofile(os.path.join(out_dir, "train.bin"))
    data[n:].tofile(os.path.join(out_dir, "val.bin"))


def synth_arith(out_dir: str, n_lines: int = 200_000, seed: int = 0) -> None:
    """Lines like 'a*b+c=r;' and 'x^2=n, x=k;'. Has algebraic structure, easy to generate."""
    rng = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        t = rng.random()
        if t < 0.4:
            a, b, c = rng.randint(0, 99), rng.randint(0, 99), rng.randint(0, 999)
            lines.append(f"{a}*{b}+{c}={a*b+c};")
        elif t < 0.7:
            x = rng.randint(0, 999)
            lines.append(f"sqrt {x*x}={x};")
        else:
            a, x = rng.randint(1, 30), rng.randint(0, 99)
            b = rng.randint(0, 500)
            lines.append(f"{a}x+{b}={a*x+b}, x={x};")
    text = "\n".join(lines).encode()
    os.makedirs(out_dir, exist_ok=True)
    n = int(len(text) * 0.9)
    np.frombuffer(text[:n], dtype=np.uint8).tofile(os.path.join(out_dir, "train.bin"))
    np.frombuffer(text[n:], dtype=np.uint8).tofile(os.path.join(out_dir, "val.bin"))


class ByteData:
    def __init__(self, data_dir: str, block_size: int, device: str):
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint8, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint8, mode="r")
        self.block_size, self.device = block_size, device

    def batch(self, split: str, batch_size: int, gen: torch.Generator):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
