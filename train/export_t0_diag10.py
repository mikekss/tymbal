#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag10.py — round 10: surgery on the ORT graph instead of building from scratch.

Lessons from r.8–9: graphs assembled by hand are stubbornly executed by atonn in
float (the matcher's conventions are undocumented and cannot be reproduced
blind), while the output of the ORT quantizer it does accept (d12/d15/d17: int8
weights, 0 SW epochs). So we take the MATCHER-ACCEPTED d17/d18 (ORT-QDQ + the
scale unification of r.6) and do a minimal rewiring only around Concat/Slice:

  before: DQ(state_q) --\
          DQ(h_q)     ---> Concat(fp32) -> Q -> DQ -> Conv...  (+ Slice from DQ)
  after:  state_q --\
          h_q      ---> Concat(int8) -> DQ -> Conv...          (+ Slice from int8)

The Conv/Add layers and their Q/DQ islands are not touched by a single byte — the
matcher has nothing to complain about; the fp32 buffers around concat/slice (the
main memory eater, [1,128,12,112] fp32 × 24) become int8. The scales are already
unified (r.6), so the rewiring is mathematically an identity.

Output: d25_bh_c128_v12_i8s.onnx (a), d26_bh_c192_v3_i8s.onnx (b).
Run: tools/run_t0_diag_r10.bat
"""
import os
import numpy as np
import onnx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")


def surgical(src, dst):
    m = onnx.load(os.path.join(OUT, src + ".onnx"))
    g = m.graph
    del g.value_info[:]          # drop the now-wrong ORT types: inference will recompute them
    prod = {o: n for n in g.node for o in n.output}
    gouts = {o.name for o in g.output}

    def consumers(t):
        return [n for n in g.node if t in n.input]

    def bypass_q_after(t):
        """If tensor t has a QuantizeLinear consumer — bypass it."""
        for q in [n for n in consumers(t) if n.op_type == "QuantizeLinear"]:
            qo = q.output[0]
            assert qo not in gouts, "a Q on a graph output cannot be bypassed"
            for c in consumers(qo):
                c.input[:] = [t if x == qo else x for x in c.input]
            g.node.remove(q)

    n_cat = n_sl = 0
    for node in list(g.node):
        if node.op_type == "Concat":
            for j, x in enumerate(list(node.input)):
                p = prod.get(x)
                if p is not None and p.op_type == "DequantizeLinear":
                    node.input[j] = p.input[0]        # the int8 side
            bypass_q_after(node.output[0])
            n_cat += 1
        elif node.op_type == "Slice":
            p = prod.get(node.input[0])
            if p is not None and p.op_type == "DequantizeLinear":
                node.input[0] = p.input[0]
            bypass_q_after(node.output[0])
            n_sl += 1

    # clean up orphans (DQ whose outputs nobody needs), iteratively
    removed = 1
    while removed:
        removed = 0
        used = {x for n in g.node for x in n.input} | gouts
        for n in list(g.node):
            if all(o not in used for o in n.output):
                g.node.remove(n)
                removed += 1

    onnx.checker.check_model(m)
    from onnx import shape_inference
    shape_inference.infer_shapes(m, strict_mode=True, check_type=True)
    onnx.save(m, os.path.join(OUT, dst + ".onnx"))
    print(f"{dst}: rewired Concat={n_cat}, Slice={n_sl}; the graph is valid")


def smoke(name, c_in, C, N, T, layers=24, cycle=(1, 2, 4, 8, 16, 32), k=3):
    import onnxruntime as ort
    dil = (list(cycle) * (layers // len(cycle)))[:layers]
    sess = ort.InferenceSession(os.path.join(OUT, name + ".onnx"),
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(4)
    states = {f"state_in_{i}": np.zeros((1, C, N, (k - 1) * d), np.float32)
              for i, d in enumerate(dil)}
    for _ in range(4):
        feed = {"x": rng.standard_normal((1, c_in, N, T)).astype(np.float32)} | states
        res = sess.run(None, feed)
        states = {f"state_in_{i}": res[1 + i] for i in range(layers)}
    assert not np.isnan(res[0]).any()
    live = sum(int(np.any(v)) for v in states.values())
    print(f"[{name}] smoke OK, live states {live}/{layers}")


if __name__ == "__main__":
    surgical("d17_bh_c128_v12_qdqu", "d25_bh_c128_v12_i8s")
    smoke("d25_bh_c128_v12_i8s", 9, 128, 12, 48)
    surgical("d18_bh_c192_v3_qdqu", "d26_bh_c192_v3_i8s")
    smoke("d26_bh_c192_v3_i8s", 8, 192, 3, 48)
    print("done")
