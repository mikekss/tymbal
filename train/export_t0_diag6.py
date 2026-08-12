#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag6.py — round 6: unifying the activation scales in the QDQ graphs
(emulating the QAT discipline with shared scales, §3.3).

Diagnosis from r.5: the ORT quantizer gives every tensor its own scale → every
Q/DQ junction is a real requantization (fp32 on the M55): 88 hybrid epochs, fp32
buffers, no allocation on the on-chip map (21233664 B for d12 and d15 alike — the
graph is the same).

The cure here: ALL activation Q/DQ get one scale (0.05, zp=0); the weights are
left alone (per-channel). Q→DQ with equal scales is an identity, and the
Concat/Slice/Add junctions become int8 moves. This is exactly what the graph
will look like after honest QAT (brevitas, a shared quantizer for
state_in/state_out). Accuracy does not matter here (T0, random weights) — what
matters is the shape of the graph for atonn.

Output: d17_bh_c128_v12_qdqu.onnx (a), d18_bh_c192_v3_qdqu.onnx (b)
Run: tools/run_t0_diag_r6.bat
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper as NH

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")
SCALE = np.float32(0.05)


def unify(src, dst):
    m = onnx.load(os.path.join(OUT, src + ".onnx"))
    inits = {i.name: i for i in m.graph.initializer}
    n_act, n_w = 0, 0
    for node in m.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        # an activation Q/DQ: the data (input 0) is NOT an initializer
        if node.input[0] in inits:
            n_w += 1
            continue
        s = inits[node.input[1]]
        arr = NH.to_array(s)
        NH.from_array(np.full_like(arr, SCALE), s.name)
        s.CopyFrom(NH.from_array(np.full_like(arr, SCALE), s.name))
        if len(node.input) > 2 and node.input[2] in inits:
            z = inits[node.input[2]]
            z.CopyFrom(NH.from_array(np.zeros_like(NH.to_array(z)), z.name))
        n_act += 1
    onnx.save(m, os.path.join(OUT, dst + ".onnx"))
    print(f"{dst}: unified {n_act} activation Q/DQ "
          f"(weight ones untouched: {n_w})")


def smoke(dst, c_in, C, N, T, layers=24, cycle=(1, 2, 4, 8, 16, 32), k=3):
    """ORT run: several chunks with state carry-over, NaN check."""
    import onnxruntime as ort
    dil = (list(cycle) * (layers // len(cycle)))[:layers]
    sess = ort.InferenceSession(os.path.join(OUT, dst + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(3)
    states = {f"state_in_{i}": np.zeros((1, C, N, (k - 1) * d), np.float32)
              for i, d in enumerate(dil)}
    for _ in range(4):
        feed = {"x": rng.standard_normal((1, c_in, N, T)).astype(np.float32)} | states
        res = sess.run(None, feed)
        states = {f"state_in_{i}": res[1 + i] for i in range(layers)}
    assert not any(np.isnan(r).any() for r in res), "NaN in the output"
    print(f"{dst}: ORT smoke OK (4 chunks, the states circulate, no NaN)")


if __name__ == "__main__":
    unify("d15_bh_c128_v12_qdq2", "d17_bh_c128_v12_qdqu")
    smoke("d17_bh_c128_v12_qdqu", 9, 128, 12, 48)
    unify("d16_bh_c192_v3_qdq2", "d18_bh_c192_v3_qdqu")
    smoke("d18_bh_c192_v3_qdqu", 8, 192, 3, 48)
    print("done")
