#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_bench.py — discriminating benchmarks after the first DevCloud numbers
(d16: 69.75 ms, d30: 35.89 ms against a slot budget of 3.2 ms; CSV: total_flash=0,
total_ram = weights+activations => EVERYTHING is in external RAM; duration ∝ size).

Two competing hypotheses about where the time went:
  H-A "dispatch": ~330 µs of fixed cost for each of the 209 epochs;
  H-B "weight traffic": the weights sit in hyperRAM, every epoch pulls its own
      weights over a narrow external bus => time ∝ weight bytes (the observed
      1.94x at 2x weights is consistent with both H-A and H-B — a discriminator
      is needed).

Models for this round:
  e1_ctrl_c32_l12_qdq   — a clone of d30 with C=32: the same structure/epoch
      count (~105), weights ×36 smaller (~35 kB). H-A predicts the same ~35 ms,
      H-B predicts a collapse by a large factor. THE MAIN DISCRIMINATOR.
  e5_mono_c128_l12_qdq  — a "survivable" recompute monolith (Gemini's plan, but
      with honest arithmetic): NO states, causal paddings, window W=300,
      output 48; C=128, L=12, N=3: 0.53 GMAC/hop => ~1.8 ms ideal.
      We measure the monolith's real epochs/GOPS.
  e3_mono_c192_l24_qdq  — a monolith on the TARGET shape (C=192, L=24, W=552):
      4.4 GMAC/hop => ~15 ms ideal. Expected to be OVER budget — a data point
      that closes the "full recompute" question with a number.

Plus at zero cost (no models): in DevCloud re-run d30 with external memory
turned off (Change parameters -> use external RAM/Flash off), if the UI allows
it — a direct test of H-B.

The decision that follows (the ladder in §3.5): on-chip placement + EC (H0) /
a block hybrid (states once every 6 layers) / a C=128 monolith / revising slot S.
Run: python3 export_t0_bench.py
"""
import os
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP, numpy_helper as NH
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import streaming_tcn_check as ref
from export_t0_diag2 import build_bh, verify_bh
from export_t0_diag3 import BHReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")
K = 3
CYCLE = (1, 2, 4, 8, 16, 32)


def build_mono(net, c_in, c_out, N, W, T_out, name):
    """A monolith without states: causal paddings (the tensor length stays W),
    one Slice at the end (the last T_out columns). Validity: outputs at positions
    >= RF do not see the padding zeros (RF = (k-1)*Σd <= W - T_out)."""
    C = net.c_hidden
    inits, nodes = [], []

    def vi(n, shape):
        return H.make_tensor_value_info(n, TP.FLOAT, shape)

    def w4(w):
        return w[:, :, None, :].astype(np.float32)

    inputs = [vi("x", [1, c_in, N, W])]
    inits.append(NH.from_array(w4(net.head[:, :, None]), "W_head"))
    nodes.append(H.make_node("Conv", ["x", "W_head"], ["h0"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    h = "h0"
    for i, cv in enumerate(net.convs):
        pad = (cv.k - 1) * cv.d
        inits.extend([NH.from_array(w4(cv.w), f"W_{i}"),
                      NH.from_array(cv.b.astype(np.float32), f"B_{i}")])
        nodes.append(H.make_node("Conv", [h, f"W_{i}", f"B_{i}"], [f"z_{i}"],
                                 kernel_shape=[1, cv.k], dilations=[1, cv.d],
                                 pads=[0, pad, 0, 0], strides=[1, 1]))
        nodes.append(H.make_node("Relu", [f"z_{i}"], [f"r_{i}"]))
        nodes.append(H.make_node("Add", [h, f"r_{i}"], [f"h_{i+1}"]))
        h = f"h_{i+1}"
    inits.append(NH.from_array(w4(net.tail[:, :, None]), "W_tail"))
    nodes.append(H.make_node("Conv", [h, "W_tail"], ["y_full"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    inits.extend([NH.from_array(np.array([W - T_out], np.int64), "ss"),
                  NH.from_array(np.array([W], np.int64), "se"),
                  NH.from_array(np.array([3], np.int64), "sa")])
    nodes.append(H.make_node("Slice", ["y_full", "ss", "se", "sa"], ["y"]))
    outputs = [vi("y", [1, c_out, N, T_out])]

    g = H.make_graph(nodes, name, inputs, outputs, inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)], producer_name="n6-bench")
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(OUT, name + ".onnx"))


def verify_mono(name, net, c_in, N, W, T_out):
    """The monolith's last T_out == the tail of numpy full (same left zero-pad)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(OUT, name + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(11)
    x = rng.standard_normal((N, c_in, W)).astype(np.float32)
    y = sess.run(None, {"x": x.transpose(1, 0, 2)[None]})[0][0].transpose(1, 0, 2)
    y_ref = np.stack([net.full(x[n].astype(np.float64))[:, -T_out:] for n in range(N)])
    e = float(np.max(np.abs(y[:, :, :] [..., :] - y_ref)) /
              (np.max(np.abs(y_ref)) + 1e-12))
    # y is already [N?]: y shape (c_out, N, T)? transposed above into (N, c_out, T)
    print(f"[{name}] monolith vs numpy tail: {e:.2e} rel. "
          f"({'OK' if e < 1e-4 else 'FAIL'})")
    assert e < 1e-4


class MonoReader(BHReader.__bases__[0]):
    def __init__(self, c_in, N, W, n_calib=8):
        rng = np.random.default_rng(7)
        self.data = [{"x": rng.standard_normal((1, c_in, N, W)).astype(np.float32)}
                     for _ in range(n_calib)]
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


def qdq(src, reader):
    quantize_static(os.path.join(OUT, src + ".onnx"),
                    os.path.join(OUT, src + "_qdq.onnx"), reader,
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print(f"{src}_qdq written")


if __name__ == "__main__":
    # e1: the hypothesis discriminator — a clone of d30 with tiny weights
    rng = np.random.default_rng(2030)
    net_e1 = ref.TCN(rng, 8, 32, 4, layers=12, cycle=CYCLE, k=K)
    build_bh(net_e1, 8, 4, 3, 48, "e1_ctrl_c32_l12")
    verify_bh("e1_ctrl_c32_l12", net_e1, 8, 3, 48)
    qdq("e1_ctrl_c32_l12", BHReader(net_e1, 8, 3, 48))

    # e5: the survivable monolith (C=128, L=12, W = RF(252)+48 = 300)
    net_e5 = ref.TCN(rng, 8, 128, 4, layers=12, cycle=CYCLE, k=K)
    b = ref.budget(128, 12, 3, k=K, c_in=8, c_out=4, bands=1)
    mac = 12 * 128 * 128 * K * 300 * 3 / 1e9
    print(f"e5: monolith {mac:.2f} GMAC/hop (streaming would be {b['mac_step']*48*3/1e9:.2f})")
    build_mono(net_e5, 8, 4, 3, 300, 48, "e5_mono_c128_l12")
    verify_mono("e5_mono_c128_l12", net_e5, 8, 3, 300, 48)
    qdq("e5_mono_c128_l12", MonoReader(8, 3, 300))

    # e3: a monolith on the target shape — the honest price of a full recompute
    net_e3 = ref.TCN(rng, 8, 192, 4, layers=24, cycle=CYCLE, k=K)
    mac = 24 * 192 * 192 * K * 552 * 3 / 1e9
    print(f"e3: monolith {mac:.2f} GMAC/hop — expected to be over budget, a data point")
    build_mono(net_e3, 8, 4, 3, 552, 48, "e3_mono_c192_l24")
    verify_mono("e3_mono_c192_l24", net_e3, 8, 3, 552, 48)
    qdq("e3_mono_c192_l24", MonoReader(8, 3, 552))
    print("done: e1/e5/e3 in models/t0/diag — run them on DevCloud + d30 without external memory")
