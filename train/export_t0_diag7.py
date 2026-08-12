#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag7.py — round 7: int8-NATIVE graphs assembled by hand.

Diagnosis of round 6: the scales have nothing to do with it — atonn does not
fuse fp32 Concat/Slice between DQ and Q (24+24 hybrids), and on top of that it
inserts S2D/D2S itself on dilated convolutions (20+20). The shape of the cure:
data between layers lives in int8; Concat/Slice operate on int8 tensors directly
(that is how their own TFLite importer builds them); fp32 exists only inside the
DQ->Conv(->Relu)->Q pattern and around Add.

Layer scheme (all tensors int8 unless stated otherwise):
  cat_i  = Concat(state_in_i, h_i)                    # int8, W axis
  conv:    DQ(cat_i) -> Conv(DQ(W_i), b_f32) -> Relu -> Q  => r_i
  resid:   DQ(h_i), DQ(r_i) -> Add -> Q               => h_{i+1}
  state_out_i = Slice(cat_i, last pad columns)         # int8, NO Q/DQ
The states are int8 graph I/O: half the bytes and traffic, one scale for the
whole path (S=0.05) — the in/out state shares a scale by construction (what
QAT/brevitas will do for the real net, §3.3).

Output: d19_bh_c128_v12_i8.onnx (a), d20_bh_c192_v3_i8.onnx (b).
Run: tools/run_t0_diag_r7.bat. Smoke test: ORT, int8 states circulate, no NaNs;
the FIFO shift mechanics in int8 are exact (a byte move).
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
S = np.float32(0.05)          # single scale for activations/states


def qparams(prefix, inits):
    inits.append(NH.from_array(np.array(S, np.float32), f"{prefix}_s"))
    inits.append(NH.from_array(np.array(0, np.int8), f"{prefix}_z"))
    return f"{prefix}_s", f"{prefix}_z"


def wq(w):
    """int8 per-channel (axis 0) quantization of weights [Cout,Cin,1,k]."""
    scale = np.abs(w).max(axis=(1, 2, 3)) / 127.0 + 1e-12
    q = np.clip(np.round(w / scale[:, None, None, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def build_i8(net, c_in, c_out, N, T, name):
    C = net.c_hidden
    inits, nodes, inputs, outputs = [], [], [], []

    def vi(n, shape, tp=TP.INT8):
        return H.make_tensor_value_info(n, tp, shape)

    def conv_qdq(idx, src_q, w3, b, kk, dd, out_name, relu):
        """DQ(src)->Conv->(Relu)->Q => out_name (int8). w3: [Cout,Cin,k]."""
        w = w3[:, :, None, :].astype(np.float32)
        q, ws = wq(w)
        inits.append(NH.from_array(q, f"Wq_{idx}"))
        inits.append(NH.from_array(ws, f"Ws_{idx}"))
        inits.append(NH.from_array(np.zeros(len(ws), np.int8), f"Wz_{idx}"))
        s, z = qparams(f"aq_{idx}", inits)
        nodes.append(H.make_node("DequantizeLinear", [src_q, s, z], [f"xf_{idx}"], axis=3))
        nodes.append(H.make_node("DequantizeLinear",
                                 [f"Wq_{idx}", f"Ws_{idx}", f"Wz_{idx}"],
                                 [f"wf_{idx}"], axis=0))
        cin = [f"xf_{idx}", f"wf_{idx}"]
        if b is not None:
            inits.append(NH.from_array(b.astype(np.float32), f"B_{idx}"))
            cin.append(f"B_{idx}")
        nodes.append(H.make_node("Conv", cin, [f"cf_{idx}"], kernel_shape=[1, kk],
                                 dilations=[1, dd], pads=[0, 0, 0, 0], strides=[1, 1]))
        pre = f"cf_{idx}"
        if relu:
            nodes.append(H.make_node("Relu", [pre], [f"rf_{idx}"]))
            pre = f"rf_{idx}"
        so, zo = qparams(f"oq_{idx}", inits)
        nodes.append(H.make_node("QuantizeLinear", [pre, so, zo], [out_name]))

    # fp32 input -> int8
    inputs.append(vi("x", [1, c_in, N, T], TP.FLOAT))
    s0, z0 = qparams("inq", inits)
    nodes.append(H.make_node("QuantizeLinear", ["x", s0, z0], ["x_q"]))
    conv_qdq("head", "x_q", net.head[:, :, None], None, 1, 1, "h_q_0", relu=False)

    h = "h_q_0"
    for i, cv in enumerate(net.convs):
        pad = cv.pad
        inputs.append(vi(f"state_in_{i}", [1, C, N, pad]))
        nodes.append(H.make_node("Concat", [f"state_in_{i}", h], [f"cat_{i}"], axis=3))
        conv_qdq(f"l{i}", f"cat_{i}", cv.w, cv.b, cv.k, cv.d, f"r_q_{i}", relu=True)
        # residual: DQ+DQ -> Add -> Q
        sa, za = qparams(f"ad_{i}", inits)
        nodes.append(H.make_node("DequantizeLinear", [h, sa, za], [f"hf_{i}"]))
        nodes.append(H.make_node("DequantizeLinear", [f"r_q_{i}", sa, za], [f"rf2_{i}"]))
        nodes.append(H.make_node("Add", [f"hf_{i}", f"rf2_{i}"], [f"sum_{i}"]))
        sq, zq = qparams(f"hq_{i}", inits)
        nodes.append(H.make_node("QuantizeLinear", [f"sum_{i}", sq, zq], [f"h_q_{i+1}"]))
        # state_out: a pure int8 slice
        inits += [NH.from_array(np.array([T], np.int64), f"ss_{i}"),
                  NH.from_array(np.array([T + pad], np.int64), f"se_{i}"),
                  NH.from_array(np.array([3], np.int64), f"sa_{i}")]
        nodes.append(H.make_node("Slice", [f"cat_{i}", f"ss_{i}", f"se_{i}", f"sa_{i}"],
                                 [f"state_out_{i}"]))
        outputs.append(vi(f"state_out_{i}", [1, C, N, pad]))
        h = f"h_q_{i+1}"

    # tail: DQ -> Conv -> fp32 out (the M55 does the mix)
    st, zt = qparams("tl", inits)
    nodes.append(H.make_node("DequantizeLinear", [h, st, zt], ["hf_tail"]))
    q, ws = wq(net.tail[:, :, None, None].astype(np.float32))
    inits += [NH.from_array(q, "Wq_tail"), NH.from_array(ws, "Ws_tail"),
              NH.from_array(np.zeros(len(ws), np.int8), "Wz_tail")]
    nodes.append(H.make_node("DequantizeLinear", ["Wq_tail", "Ws_tail", "Wz_tail"],
                             ["wf_tail"], axis=0))
    nodes.append(H.make_node("Conv", ["hf_tail", "wf_tail"], ["y"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    outputs.insert(0, vi("y", [1, c_out, N, T], TP.FLOAT))

    g = H.make_graph(nodes, name, inputs, outputs, inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)], producer_name="n6-i8")
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(OUT, name + ".onnx"))
    print("written", name)
    return m


def smoke(name, net, c_in, N, T, hops=4):
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(OUT, name + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(4)
    states = {f"state_in_{i}": np.zeros((1, net.c_hidden, N, cv.pad), np.int8)
              for i, cv in enumerate(net.convs)}
    for _ in range(hops):
        feed = {"x": rng.standard_normal((1, c_in, N, T)).astype(np.float32)} | states
        res = sess.run(None, feed)
        states = {f"state_in_{i}": res[1 + i] for i in range(len(net.convs))}
    y = res[0]
    live = sum(int(np.any(v)) for v in states.values())
    assert not np.isnan(y).any()
    print(f"[{name}] smoke OK: y range [{y.min():.2f},{y.max():.2f}], "
          f"live int8 states {live}/{len(states)}")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    net_a = ref.TCN(rng, 9, 128, 1, LAYERS, CYCLE, k=K)
    net_b = ref.TCN(rng, 8, 192, 4, LAYERS, CYCLE, k=K)
    build_i8(net_a, 9, 1, 12, 48, "d19_bh_c128_v12_i8")
    smoke("d19_bh_c128_v12_i8", net_a, 9, 12, 48)
    build_i8(net_b, 8, 4, 3, 48, "d20_bh_c192_v3_i8")
    smoke("d20_bh_c192_v3_i8", net_b, 8, 3, 48)
    print("done")
