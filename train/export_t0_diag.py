#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag.py — diagnostic matrix for the "TOOL ERROR: Error in
computation of shapes" from stedgeai 4.0 on T0 graphs (ladder §3.5, diagnosis).

Each micro-graph isolates ONE hypothesis; d07/d08 are full fallback graphs.
Run: tools/run_t0_diag.bat -> the logs show which feature breaks the shapes.

  d01_conv1d_plain      Conv1d k=3, causal pads=[2,0], B=1         — basic 1D support
  d02_conv1d_dil        + dilation 32                              — dilation in 1D
  d03_concat_state      state input + Concat(axis=2) + valid Conv  — Concat
  d04_slice_state       d03 + Slice of the tail into state_out     — Slice (our scheme)
  d05_batch12           d04 with B=12                              — batch > 1
  d06_opset13           d04 in opset 13 / ir 8                     — opset version
  d07_conv2d_c128_b12   FULL a_c128_b12_t48 as Conv2d [B,C,1,T],
                        kernel [1,3], dilations [1,d], axis 3       — guide §3.2 fallback
  d08_conv2d_c128_b1    d07 with B=1                               — fallback + no batch
  d09_conv1d_c128_b1    full 1D graph with B=1                     — batch hypothesis only

d07 is numerically cross-checked against the numpy reference (it is the same
computational graph).
Run: python3 export_t0_diag.py
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


def save(model, name, opset=17, ir=9):
    model.ir_version = ir
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(OUT, name + ".onnx"))
    print("written", name)


def micro(name, B=1, T=48, C=8, dil=1, concat=False, slice_out=False, opset=17, ir=9):
    """Micro-graph: [head 1x1] -> (concat state?) -> conv k=3 (valid with concat,
    otherwise causal pads) -> relu -> add -> [tail 1x1] (+ slice state_out?)."""
    rng = np.random.default_rng(1)
    pad = (K - 1) * dil
    inits, nodes, inputs, outputs = [], [], [], []
    inputs.append(H.make_tensor_value_info("x", TP.FLOAT, [B, C, T]))
    w1 = rng.standard_normal((C, C, K)).astype(np.float32) / 5
    b1 = rng.standard_normal(C).astype(np.float32) / 10
    inits += [NH.from_array(w1, "W1"), NH.from_array(b1, "B1")]
    src = "x"
    if concat:
        inputs.append(H.make_tensor_value_info("state_in_0", TP.FLOAT, [B, C, pad]))
        nodes.append(H.make_node("Concat", ["state_in_0", "x"], ["cat0"], axis=2))
        src = "cat0"
        conv_pads = [0, 0]
    else:
        conv_pads = [pad, 0]
    nodes.append(H.make_node("Conv", [src, "W1", "B1"], ["z0"], kernel_shape=[K],
                             dilations=[dil], pads=conv_pads, strides=[1]))
    nodes.append(H.make_node("Relu", ["z0"], ["r0"]))
    nodes.append(H.make_node("Add", ["x", "r0"], ["y"]))
    outputs.append(H.make_tensor_value_info("y", TP.FLOAT, [B, C, T]))
    if slice_out:
        inits += [NH.from_array(np.array([T], np.int64), "ss"),
                  NH.from_array(np.array([T + pad], np.int64), "se"),
                  NH.from_array(np.array([2], np.int64), "sa")]
        nodes.append(H.make_node("Slice", ["cat0", "ss", "se", "sa"], ["state_out_0"]))
        outputs.append(H.make_tensor_value_info("state_out_0", TP.FLOAT, [B, C, pad]))
    g = H.make_graph(nodes, name, inputs, outputs, inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid("", opset)], producer_name="n6-diag")
    save(m, name, opset, ir)


def full_graph(net, c_in, c_out, B, T, name, conv2d=False, opset=17, ir=9):
    """Full StreamingTCN; conv2d=True -> [B,C,1,T] representation (§3.2 fallback)."""
    C = net.c_hidden
    ax = 3 if conv2d else 2
    inits, nodes, inputs, outputs = [], [], [], []

    def vi(n, shape):
        return H.make_tensor_value_info(n, TP.FLOAT, shape)

    def wshape(w):        # [Cout,Cin,k] -> [Cout,Cin,1,k] for conv2d
        return w[:, :, None, :] if conv2d else w

    def kattr(k, d):
        return (dict(kernel_shape=[1, k], dilations=[1, d], pads=[0, 0, 0, 0],
                     strides=[1, 1]) if conv2d else
                dict(kernel_shape=[k], dilations=[d], pads=[0, 0], strides=[1]))

    xsh = [B, c_in, 1, T] if conv2d else [B, c_in, T]
    inputs.append(vi("x", xsh))
    inits.append(NH.from_array(wshape(net.head.astype(np.float32)[:, :, None]), "W_head"))
    nodes.append(H.make_node("Conv", ["x", "W_head"], ["h0"], **kattr(1, 1)))
    h = "h0"
    for i, cv in enumerate(net.convs):
        pad = cv.pad
        ssh = [B, C, 1, pad] if conv2d else [B, C, pad]
        inputs.append(vi(f"state_in_{i}", ssh))
        nodes.append(H.make_node("Concat", [f"state_in_{i}", h], [f"cat_{i}"], axis=ax))
        inits.append(NH.from_array(wshape(cv.w.astype(np.float32)), f"W_{i}"))
        inits.append(NH.from_array(cv.b.astype(np.float32), f"B_{i}"))
        nodes.append(H.make_node("Conv", [f"cat_{i}", f"W_{i}", f"B_{i}"], [f"z_{i}"],
                                 **kattr(cv.k, cv.d)))
        nodes.append(H.make_node("Relu", [f"z_{i}"], [f"r_{i}"]))
        nodes.append(H.make_node("Add", [h, f"r_{i}"], [f"h_{i+1}"]))
        inits += [NH.from_array(np.array([T], np.int64), f"ss_{i}"),
                  NH.from_array(np.array([T + pad], np.int64), f"se_{i}"),
                  NH.from_array(np.array([ax], np.int64), f"sa_{i}")]
        nodes.append(H.make_node("Slice", [f"cat_{i}", f"ss_{i}", f"se_{i}", f"sa_{i}"],
                                 [f"state_out_{i}"]))
        outputs.append(vi(f"state_out_{i}", ssh))
        h = f"h_{i+1}"
    inits.append(NH.from_array(wshape(net.tail.astype(np.float32)[:, :, None]), "W_tail"))
    nodes.append(H.make_node("Conv", [h, "W_tail"], ["y"], **kattr(1, 1)))
    ysh = [B, c_out, 1, T] if conv2d else [B, c_out, T]
    outputs.insert(0, vi("y", ysh))
    g = H.make_graph(nodes, name, inputs, outputs, inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid("", opset)], producer_name="n6-diag")
    save(m, name, opset, ir)
    return m


def verify_conv2d(model_name, net, c_in, B, T, t_total=240):
    """conv2d graph == numpy reference (chunks with states, 2 batch elements)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(OUT, model_name + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(5)
    x = rng.standard_normal((B, c_in, t_total)).astype(np.float32)
    states = {f"state_in_{i}": np.zeros((B, net.c_hidden, 1, cv.pad), np.float32)
              for i, cv in enumerate(net.convs)}
    ys = []
    for t0 in range(0, t_total, T):
        feed = {"x": x[:, :, None, t0:t0 + T]} | states
        res = sess.run(None, feed)
        ys.append(res[0][:, :, 0, :])
        states = {f"state_in_{i}": res[1 + i] for i in range(len(net.convs))}
    y = np.concatenate(ys, axis=2)
    nb = min(B, 2)
    y_ref = np.stack([net.full(x[b].astype(np.float64)) for b in range(nb)])
    e = float(np.max(np.abs(y[:nb] - y_ref)) / (np.max(np.abs(y_ref)) + 1e-12))
    print(f"[{model_name}] conv2d chunks vs numpy reference: {e:.2e} rel. "
          f"({'OK' if e < 1e-4 else 'FAIL'})")
    assert e < 1e-4


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    micro("d01_conv1d_plain")
    micro("d02_conv1d_dil", dil=32)
    micro("d03_concat_state", dil=32, concat=True)
    micro("d04_slice_state", dil=32, concat=True, slice_out=True)
    micro("d05_batch12", B=12, dil=32, concat=True, slice_out=True)
    micro("d06_opset13", dil=32, concat=True, slice_out=True, opset=13, ir=8)

    rng = np.random.default_rng(2026)
    net = ref.TCN(rng, 9, 128, 1, LAYERS, CYCLE, k=K)
    full_graph(net, 9, 1, 12, 48, "d07_conv2d_c128_b12", conv2d=True)
    verify_conv2d("d07_conv2d_c128_b12", net, 9, 12, 48)
    full_graph(net, 9, 1, 1, 48, "d08_conv2d_c128_b1", conv2d=True)
    full_graph(net, 9, 1, 1, 48, "d09_conv1d_c128_b1", conv2d=False)
    print("done:", OUT)
