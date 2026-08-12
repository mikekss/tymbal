#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag2.py — diagnostics round 2: working around stedgeai's "batch = 1" limit.

Diagnosis from round 1 (diag_4810): B>1 breaks shape inference in ANY
representation (d05, d07); at B=1 the whole graph is green in both (d08, d09).
The workaround is "batch-as-height": N independent sequences are laid out along
the H axis of the conv2d representation, batch = 1:

    x        [1, c_in, N, T]      (N = voices×bands or voices)
    weights  [Cout, Cin, 1, k], dilations [1, d], pads 0
    states   [1, C, N, pad], Concat/Slice along the W axis (axis=3)

A kernel of height 1 => the H rows do not interact: mathematically IDENTICAL to
the batch wiring (verified right here against the numpy reference, row by row).
NB: "bands-as-channels" (wiring (b)) does not change in substance because of
this — only the packing of voices changes.

Graphs:
  d10_bh_c128_v12  — wiring (a) batch->height: C=128, N=12 (3 voices × 4 bands), T=48
  d11_bh_c192_v3   — wiring (b) channels, voices->height: C=192, N=3, T=48

Run: python3 export_t0_diag2.py  ->  models/t0/diag/*.onnx
"""
import os
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP, numpy_helper as NH

import streaming_tcn_check as ref

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")
K = 3
CYCLE = (1, 2, 4, 8, 16, 32)
LAYERS = 24


def build_bh(net, c_in, c_out, N, T, name):
    """StreamingTCN in the batch-as-height layout: [1, C, N, T]."""
    C = net.c_hidden
    inits, nodes, inputs, outputs = [], [], [], []

    def vi(n, shape):
        return H.make_tensor_value_info(n, TP.FLOAT, shape)

    def w4(w):            # [Cout,Cin,k] -> [Cout,Cin,1,k]
        return w[:, :, None, :].astype(np.float32)

    inputs.append(vi("x", [1, c_in, N, T]))
    inits.append(NH.from_array(w4(net.head[:, :, None]), "W_head"))
    nodes.append(H.make_node("Conv", ["x", "W_head"], ["h0"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    h = "h0"
    for i, cv in enumerate(net.convs):
        pad = cv.pad
        inputs.append(vi(f"state_in_{i}", [1, C, N, pad]))
        nodes.append(H.make_node("Concat", [f"state_in_{i}", h], [f"cat_{i}"], axis=3))
        inits += [NH.from_array(w4(cv.w), f"W_{i}"),
                  NH.from_array(cv.b.astype(np.float32), f"B_{i}")]
        nodes.append(H.make_node("Conv", [f"cat_{i}", f"W_{i}", f"B_{i}"], [f"z_{i}"],
                                 kernel_shape=[1, cv.k], dilations=[1, cv.d],
                                 pads=[0, 0, 0, 0], strides=[1, 1]))
        nodes.append(H.make_node("Relu", [f"z_{i}"], [f"r_{i}"]))
        nodes.append(H.make_node("Add", [h, f"r_{i}"], [f"h_{i+1}"]))
        inits += [NH.from_array(np.array([T], np.int64), f"ss_{i}"),
                  NH.from_array(np.array([T + pad], np.int64), f"se_{i}"),
                  NH.from_array(np.array([3], np.int64), f"sa_{i}")]
        nodes.append(H.make_node("Slice", [f"cat_{i}", f"ss_{i}", f"se_{i}", f"sa_{i}"],
                                 [f"state_out_{i}"]))
        outputs.append(vi(f"state_out_{i}", [1, C, N, pad]))
        h = f"h_{i+1}"
    inits.append(NH.from_array(w4(net.tail[:, :, None]), "W_tail"))
    nodes.append(H.make_node("Conv", [h, "W_tail"], ["y"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    outputs.insert(0, vi("y", [1, c_out, N, T]))

    g = H.make_graph(nodes, name, inputs, outputs, inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)], producer_name="n6-diag2")
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(OUT, name + ".onnx"))
    return m


def verify_bh(name, net, c_in, N, T, t_total=240):
    """Chunks with states through ORT == the numpy reference, H rows = independent
    sequences (we check 2 rows)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(OUT, name + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(6)
    x = rng.standard_normal((N, c_in, t_total)).astype(np.float32)   # N sequences
    states = {f"state_in_{i}": np.zeros((1, net.c_hidden, N, cv.pad), np.float32)
              for i, cv in enumerate(net.convs)}
    ys = []
    for t0 in range(0, t_total, T):
        # [N,c_in,T] -> [1,c_in,N,T]
        feed = {"x": x[:, :, t0:t0 + T].transpose(1, 0, 2)[None]} | states
        res = sess.run(None, feed)
        ys.append(res[0][0].transpose(1, 0, 2))                      # -> [N,c_out,T]
        states = {f"state_in_{i}": res[1 + i] for i in range(len(net.convs))}
    y = np.concatenate(ys, axis=2)
    nb = 2
    y_ref = np.stack([net.full(x[n].astype(np.float64)) for n in range(nb)])
    e = float(np.max(np.abs(y[:nb] - y_ref)) / (np.max(np.abs(y_ref)) + 1e-12))
    print(f"[{name}] height rows vs numpy reference: {e:.2e} rel. "
          f"({'OK' if e < 1e-4 else 'FAIL'})")
    assert e < 1e-4


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    net_a = ref.TCN(rng, 9, 128, 1, LAYERS, CYCLE, k=K)
    build_bh(net_a, 9, 1, 12, 48, "d10_bh_c128_v12")
    verify_bh("d10_bh_c128_v12", net_a, 9, 12, 48)

    net_b = ref.TCN(rng, 8, 192, 4, LAYERS, CYCLE, k=K)
    build_bh(net_b, 8, 4, 3, 48, "d11_bh_c192_v3")
    verify_bh("d11_bh_c192_v3", net_b, 8, 3, 48)
    print("done: d10/d11 in", OUT)
