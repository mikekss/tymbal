#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_m0_d31.py — M0 experiment (b): d31 = d30 WITHOUT the dilation attribute.

M0 diagnosis (the M0 report, 1 Aug): 91% of d30's t_call is eaten by the
S2D/D2S wrapping that atonn 4.0 lowers dilations>1 into; the cost is ~quadratic
in C (empirically C=32 vs C=192). Workaround: a dilated k=3 convolution = three
taps x[t], x[t-d], x[t-2d] — we cut them out with explicit narrow Slices along W
from cat(state, h), glue them along CHANNELS (axis=1 -> [1, 3C, N, T]) and press
a Conv 1x1 with repacked weights:
    W1x1[:, k*C:(k+1)*C] = W[:, :, k]   (order = the tap order in Concat)
Mathematically IDENTICAL (checked by verify_bh against a numpy reference with
real dilations). In the graph dilations = [1,1] everywhere.

The net and the seed are the same as d30's (diag12, seed 2029) — the weights
match bit for bit.
Output: models/t0/diag/d31_bh_c192_v3_l12{,_qdq}.onnx
"""
import os
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP, numpy_helper as NH
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import streaming_tcn_check as ref
from export_t0_diag2 import verify_bh, OUT
from export_t0_diag3 import BHReader


def build_bh_gather(net, c_in, c_out, N, T, name):
    """batch-as-height, dilation via gather taps + Conv 1x1."""
    C = net.c_hidden
    inits, nodes, inputs, outputs = [], [], [], []
    vi = lambda n, s: H.make_tensor_value_info(n, TP.FLOAT, s)
    w4 = lambda w: w[:, :, None, :].astype(np.float32)

    inputs.append(vi("x", [1, c_in, N, T]))
    inits.append(NH.from_array(w4(net.head[:, :, None]), "W_head"))
    nodes.append(H.make_node("Conv", ["x", "W_head"], ["h0"], kernel_shape=[1, 1],
                             dilations=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]))
    h = "h0"
    for i, cv in enumerate(net.convs):
        pad, d, k = cv.pad, cv.d, cv.k
        assert pad == (k - 1) * d
        inputs.append(vi(f"state_in_{i}", [1, C, N, pad]))
        nodes.append(H.make_node("Concat", [f"state_in_{i}", h], [f"cat_{i}"], axis=3))
        # taps: cat[..., j*d : j*d+T], j=0..k-1
        taps = []
        for j in range(k):
            inits += [NH.from_array(np.array([j * d], np.int64), f"ts_{i}_{j}"),
                      NH.from_array(np.array([j * d + T], np.int64), f"te_{i}_{j}"),
                      NH.from_array(np.array([3], np.int64), f"ta_{i}_{j}")]
            nodes.append(H.make_node("Slice",
                                     [f"cat_{i}", f"ts_{i}_{j}", f"te_{i}_{j}", f"ta_{i}_{j}"],
                                     [f"tap_{i}_{j}"]))
            taps.append(f"tap_{i}_{j}")
        nodes.append(H.make_node("Concat", taps, [f"g_{i}"], axis=1))  # [1,kC,N,T]
        w1 = np.concatenate([cv.w[:, :, j] for j in range(k)], axis=1)  # [C, kC]
        inits += [NH.from_array(w1[:, :, None, None].astype(np.float32), f"W_{i}"),
                  NH.from_array(cv.b.astype(np.float32), f"B_{i}")]
        nodes.append(H.make_node("Conv", [f"g_{i}", f"W_{i}", f"B_{i}"], [f"z_{i}"],
                                 kernel_shape=[1, 1], dilations=[1, 1],
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
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)], producer_name="n6-m0-d31")
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(OUT, name + ".onnx"))
    return m


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    name = "d31_bh_c192_v3_l12"
    rng = np.random.default_rng(2029)          # same seed as d30 (diag12)
    net = ref.TCN(rng, 8, 192, 4, layers=12, cycle=(1, 2, 4, 8, 16, 32), k=3)
    build_bh_gather(net, 8, 4, 3, 48, name)
    # sanity: the serialized graph has no dilations != 1
    m = onnx.load(os.path.join(OUT, name + ".onnx"))
    for n in m.graph.node:
        for a in n.attribute:
            if a.name == "dilations":
                assert list(a.ints) == [1, 1], (n.name, list(a.ints))
    print(f"[{name}] dilations==1 everywhere: OK")
    verify_bh(name, net, 8, 3, 48)             # the reference COMPUTES with dilations
    quantize_static(os.path.join(OUT, name + ".onnx"),
                    os.path.join(OUT, name + "_qdq.onnx"),
                    BHReader(net, 8, 3, 48),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print(f"{name}_qdq written ->", OUT)
