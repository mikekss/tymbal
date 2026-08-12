#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streaming_tcn.py — StreamingTCN (torch): the base for training (T1) and for the
regular ONNX export. Reference — guide §3.1. Machine: GTX 1060 (Pascal) → torch
2.1–2.4 + cu118, fp32 (guide §1.2); the selftest here also works on CPU.

Both wirings of D-5:
  (a) batch = voices×bands: c_in=9 (band skeleton + amp/tA/tB + band-id one-hot(4)
      + drive), c_out=1; batch B = V×4.
  (b) bands-as-channels: c_in=8 (4 skeleton bands + amp/tA/tB + drive),
      C=192, c_out=4; batch B = V.

Difference from the code in guide §3.1: head/tail have no bias — that is how the
budget calculator counts (streaming_tcn_check.budget) and how the hand-built T0
graphs are made (export_t0.py); the net does not care (the head bias is absorbed
by the layer biases).

Selftest: chunked==full for both wirings + torch.onnx.export with the same I/O
names as in export_t0.py. IMPORTANT (see docs/t0_instructions.md): before the
final training, run stedgeai analyze on the torch-exported graph too — the
structure must match the hand-built one (a green T0 is valid for both sources of
the graph).

Run: python3 streaming_tcn.py
"""
import os

import torch
import torch.nn as nn


class StreamConv(nn.Module):
    def __init__(self, c, k=3, d=1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, dilation=d)      # valid, no padding
        self.pad = (k - 1) * d

    def forward(self, x, state):                        # x:[B,C,T] state:[B,C,pad]
        xin = torch.cat([state, x], dim=2)
        return self.conv(xin), xin[:, :, -self.pad:]


class StreamingTCN(nn.Module):
    def __init__(self, c_in, c=128, c_out=1, layers=24,
                 cycle=(1, 2, 4, 8, 16, 32), k=3, with_tail=True):
        super().__init__()
        assert layers % len(cycle) == 0
        self.head = nn.Conv1d(c_in, c, 1, bias=False)
        self.blocks = nn.ModuleList(
            [StreamConv(c, k, d) for d in (list(cycle) * (layers // len(cycle)))])
        self.tail = nn.Conv1d(c, c_out, 1, bias=False) if with_tail else None
        self.c = c

    def forward(self, x, *states):
        h = self.head(x)
        outs = []
        for blk, s in zip(self.blocks, states):
            z, s2 = blk(h, s)
            h = h + torch.relu(z)
            outs.append(s2)
        y = self.tail(h) if self.tail is not None else h
        return (y, *outs)

    def zero_states(self, b):
        return [torch.zeros(b, self.c, blk.pad) for blk in self.blocks]


def wiring_batch(c=128, with_tail=True):
    """Wiring (a): batch = voices×bands."""
    return StreamingTCN(c_in=9, c=c, c_out=1, with_tail=with_tail)


def wiring_channels(with_tail=True):
    """Wiring (b): bands-as-channels, the fallback option of D-5."""
    return StreamingTCN(c_in=8, c=192, c_out=4, with_tail=with_tail)


@torch.no_grad()
def check_stream_eq(net, c_in, b=2, t_total=480, hop=48):
    """chunked+state == full (the full sequence = one big chunk with zeroed
    states — it is the same computation)."""
    x = torch.randn(b, c_in, t_total)
    y_full = net(x, *net.zero_states(b))[0]
    states = net.zero_states(b)
    ys = []
    for t0 in range(0, t_total, hop):
        out = net(x[:, :, t0:t0 + hop], *states)
        ys.append(out[0])
        states = list(out[1:])
    y_stream = torch.cat(ys, dim=2)
    return (y_full - y_stream).abs().max().item()


def export_onnx(net, path, b, t, c_in):
    """§3.2: opset 17, static shapes, names as in export_t0.py."""
    net.eval()
    x = torch.zeros(b, c_in, t)
    states = net.zero_states(b)
    torch.onnx.export(
        net, (x, *states), path, opset_version=17,
        input_names=["x"] + [f"state_in_{i}" for i in range(len(states))],
        output_names=["y"] + [f"state_out_{i}" for i in range(len(states))])


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "models", "t0")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    for tag, net, c_in in [("(a) batch C=128", wiring_batch(128), 9),
                           ("(b) channels C=192", wiring_channels(), 8)]:
        e = check_stream_eq(net, c_in)
        print(f"[{tag}] chunked==full: {e:.3e} "
              f"({'OK' if e < 1e-5 else 'FAIL'})")
        assert e < 1e-5

    export_onnx(wiring_batch(128), os.path.join(out_dir, "torch_a_c128_b12_t48.onnx"),
                b=12, t=48, c_in=9)
    export_onnx(wiring_channels(), os.path.join(out_dir, "torch_b_c192_b3_t48.onnx"),
                b=3, t=48, c_in=8)
    print("OK: both wirings stream; torch-ONNX is in models/t0/ — run it through"
          " stedgeai with the same bat file for a cross-check")
