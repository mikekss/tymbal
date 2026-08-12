#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag8.py — round 8: int8-native graphs with QDQ "passports" at the edges.

Diagnosis from r.7: our local ONNX is valid (strict shape inference passes), but
the stedgeai frontend re-exports the graph and trips over the "bare" int8 outputs
(WARNING state_out is not quantized): in ONNX a tensor's quantization parameters
live only in Q/DQ nodes, and our int8 state outputs had none.

The r.8 shape (the only difference from r.7):
  state_in_i  (fp32 input)  -> QuantizeLinear(S) -> int8 -> Concat(...)
  state_out_i = Slice(int8) -> DequantizeLinear(S) -> fp32 output
Inside, everything is as in r.7: Concat/Slice on int8, fp32 only in
DQ->Conv->Relu->Q and around the Add. The scale S is single for the whole chain
=> the edge Q/DQ are identity transforms, and the standard CLI flag
`--input-data-type int8 --output-data-type int8` strips them, opening an int8
interface (the ST route; checked in the r8 bat both ways).

Output: d21_bh_c128_v12_i8e.onnx (a), d22_bh_c192_v3_i8e.onnx (b).
Run: tools/run_t0_diag_r8.bat
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
S = np.float32(0.05)


def qparams(prefix, inits):
    inits.append(NH.from_array(np.array(S, np.float32), f"{prefix}_s"))
    inits.append(NH.from_array(np.array(0, np.int8), f"{prefix}_z"))
    return f"{prefix}_s", f"{prefix}_z"


def wq(w):
    scale = np.abs(w).max(axis=(1, 2, 3)) / 127.0 + 1e-12
    q = np.clip(np.round(w / scale[:, None, None, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def build_i8_edged(net, c_in, c_out, N, T, name):
    C = net.c_hidden
    inits, nodes, inputs, outputs = [], [], [], []

    def vi(n, shape, tp=TP.FLOAT):
        return H.make_tensor_value_info(n, tp, shape)

    def conv_qdq(idx, src_q, w3, b, kk, dd, out_name, relu):
        w = w3[:, :, None, :].astype(np.float32)
        q, ws = wq(w)
        inits.extend([NH.from_array(q, f"Wq_{idx}"), NH.from_array(ws, f"Ws_{idx}"),
                      NH.from_array(np.zeros(len(ws), np.int8), f"Wz_{idx}")])
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

    inputs.append(vi("x", [1, c_in, N, T]))
    s0, z0 = qparams("inq", inits)
    nodes.append(H.make_node("QuantizeLinear", ["x", s0, z0], ["x_q"]))
    conv_qdq("head", "x_q", net.head[:, :, None], None, 1, 1, "h_q_0", relu=False)

    h = "h_q_0"
    for i, cv in enumerate(net.convs):
        pad = cv.pad
        # edge: state fp32 -> Q -> int8
        inputs.append(vi(f"state_in_{i}", [1, C, N, pad]))
        si, zi = qparams(f"sq_{i}", inits)
        nodes.append(H.make_node("QuantizeLinear", [f"state_in_{i}", si, zi],
                                 [f"state_q_{i}"]))
        nodes.append(H.make_node("Concat", [f"state_q_{i}", h], [f"cat_{i}"], axis=3))
        conv_qdq(f"l{i}", f"cat_{i}", cv.w, cv.b, cv.k, cv.d, f"r_q_{i}", relu=True)
        sa, za = qparams(f"ad_{i}", inits)
        nodes.append(H.make_node("DequantizeLinear", [h, sa, za], [f"hf_{i}"]))
        nodes.append(H.make_node("DequantizeLinear", [f"r_q_{i}", sa, za], [f"rf2_{i}"]))
        nodes.append(H.make_node("Add", [f"hf_{i}", f"rf2_{i}"], [f"sum_{i}"]))
        sq, zq = qparams(f"hq_{i}", inits)
        nodes.append(H.make_node("QuantizeLinear", [f"sum_{i}", sq, zq], [f"h_q_{i+1}"]))
        # edge: int8 slice -> DQ -> fp32 output
        inits += [NH.from_array(np.array([T], np.int64), f"ss_{i}"),
                  NH.from_array(np.array([T + pad], np.int64), f"se_{i}"),
                  NH.from_array(np.array([3], np.int64), f"sa_{i}")]
        nodes.append(H.make_node("Slice", [f"cat_{i}", f"ss_{i}", f"se_{i}", f"sa_{i}"],
                                 [f"state_s_{i}"]))
        sd, zd = qparams(f"so_{i}", inits)
        nodes.append(H.make_node("DequantizeLinear", [f"state_s_{i}", sd, zd],
                                 [f"state_out_{i}"]))
        outputs.append(vi(f"state_out_{i}", [1, C, N, pad]))
        h = f"h_q_{i+1}"

    st, zt = qparams("tl", inits)
    nodes.append(H.make_node("DequantizeLinear", [h, st, zt], ["hf_tail"]))
    q, ws = wq(net.tail[:, :, None, None].astype(np.float32))
    inits += [NH.from_array(q, "Wq_tail"), NH.from_array(ws, "Ws_tail"),
              NH.from_array(np.zeros(len(ws), np.int8), "Wz_tail")]
    nodes.append(H.make_node("DequantizeLinear", ["Wq_tail", "Ws_tail", "Wz_tail"],
                             ["wf_tail"], axis=0))
    nodes.append(H.make_node("Conv", ["hf_tail", "wf_tail"], ["y"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    outputs.insert(0, vi("y", [1, c_out, N, T]))

    g = H.make_graph(nodes, name, inputs, outputs, inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)], producer_name="n6-i8e")
    m.ir_version = 9
    onnx.checker.check_model(m)
    from onnx import shape_inference
    shape_inference.infer_shapes(m, strict_mode=True, check_type=True)
    onnx.save(m, os.path.join(OUT, name + ".onnx"))
    print("written", name, "(strict shape inference OK)")


def smoke(name, net, c_in, N, T, hops=4):
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(OUT, name + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(4)
    states = {f"state_in_{i}": np.zeros((1, net.c_hidden, N, cv.pad), np.float32)
              for i, cv in enumerate(net.convs)}
    for _ in range(hops):
        feed = {"x": rng.standard_normal((1, c_in, N, T)).astype(np.float32)} | states
        res = sess.run(None, feed)
        states = {f"state_in_{i}": res[1 + i] for i in range(len(net.convs))}
    assert not np.isnan(res[0]).any()
    live = sum(int(np.any(v)) for v in states.values())
    print(f"[{name}] smoke OK, live states {live}/{len(states)}")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    net_a = ref.TCN(rng, 9, 128, 1, LAYERS, CYCLE, k=K)
    net_b = ref.TCN(rng, 8, 192, 4, LAYERS, CYCLE, k=K)
    build_i8_edged(net_a, 9, 1, 12, 48, "d21_bh_c128_v12_i8e")
    smoke("d21_bh_c128_v12_i8e", net_a, 9, 12, 48)
    build_i8_edged(net_b, 8, 4, 3, 48, "d22_bh_c192_v3_i8e")
    smoke("d22_bh_c192_v3_i8e", net_b, 8, 3, 48)
    print("done")
