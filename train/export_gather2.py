#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_gather2.py — CANONICAL EXPORT of a trained net into a gather2 graph (D-10).

Where it came from: the function `build_bh_gather2` is called canonical in D-10,
m0_report, the v1.2 guide and the project contracts file, but it is not in
`train/export_m0_d31.py` — there is only `build_bh_gather` there (the d31 form:
C=192, N=3, taps on ALL layers). The generator of the M0 matrix (d34..d48) was
never committed to the repository, and neither was the weights_*.c generator
(m0_report says so honestly: "the generator itself was never committed, only
its output"). The artifacts survived:
`models/t0/diag/d44_bh_c88_v2_l12_g2_qdq.onnx` — exactly the point that gave
2866 us/hop. The structure here is RECONSTRUCTED FROM IT and cross-checked
against it (`--compare`), not written from memory.

STRUCTURE (verified against d44):
  input  x[1, c_in, V, T],  state_in_i[1, C, V, 2*d_i]
  output y[1, c_out, V, T], state_out_i[1, C, V, 2*d_i]
  h0 = Conv1x1(x, W_head)                       # no bias
  for layer i with dilation d:
    cat_i       = Concat(state_in_i, h_i, axis=3)          # width 2d + T
    d == 1:  r_i = Conv(cat_i, W_i[C,C,1,3], B_i)          # native, k=1x3
    d  > 1:  tap_j = Slice(cat_i, start=j*d, end=j*d+T, axis=3), j=0,1,2
             g_i   = Concat(tap_0, tap_1, tap_2, axis=1)   # 3C channels
             r_i   = Conv(g_i, W_i[C,3C,1,1], B_i)         # k=1x1
    state_out_i = Slice(cat_i, start=T, end=T+2d, axis=3)
    h_{i+1}     = Add(h_i, Relu(r_i))
  y = Conv1x1(h_L, W_tail)                       # no bias

WEIGHT REPACKING for d>1: tap j corresponds to W[:, :, j] of the native
convolution. Tap 0 is the OLDEST sample (cat = [state, h], slice from zero), tap
2 is the current one. A mistake here is not caught by eye: the net will simply
compute something else. That is why the selftest cross-checks the graph against
the numpy reference StreamingTCN AND checks streaming.

ABOUT RELU: in d44 (QDQ) there are no Relu nodes — ORT folds Relu into the next
QuantizeLinear. Here the graph is fp32, and Relu MUST be present in it,
otherwise it is not equal to the trained net. After quantize_static, cross-check
the node types against d44 (--compare does this automatically if you give it the
QDQ reference).

RUN
  python export_gather2.py --selftest                  # no torch: numpy+ORT
  python export_gather2.py --compare ../models/t0/diag/d44_bh_c88_v2_l12_g2_qdq.onnx
  python export_gather2.py --ckpt ../teacher_search/A2_ottpress/ckpt_delta_p360.pt \\
                           --out ../models/t1/n6_gather2.onnx
"""
import argparse
import collections
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CYCLE = (1, 2, 4, 8, 16, 32)
K = 3


# ------------------------------------------------------------------ building
def _init(name, arr):
    return numpy_helper.from_array(np.ascontiguousarray(arr, np.float32), name)


def _slice(nodes, inits, out, src, start, end, axis=3, tag=""):
    s, e, a = f"{out}_s{tag}", f"{out}_e{tag}", f"{out}_a{tag}"
    inits += [numpy_helper.from_array(np.array([start], np.int64), s),
              numpy_helper.from_array(np.array([end], np.int64), e),
              numpy_helper.from_array(np.array([axis], np.int64), a)]
    nodes.append(helper.make_node("Slice", [src, s, e, a], [out]))


def build_gather2(w, c_in, C, c_out, V, T, layers=12, cycle=CYCLE, name="n6_gather2"):
    """w: dict with head[C,c_in], tail[c_out,C], conv{i}_w[C,C,3], conv{i}_b[C]."""
    assert layers % len(cycle) == 0
    dil = list(cycle) * (layers // len(cycle))
    nodes, inits = [], []
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                            [1, c_in, V, T])]
    outputs = []

    inits.append(_init("W_head", w["head"].reshape(C, c_in, 1, 1)))
    nodes.append(helper.make_node("Conv", ["x", "W_head"], ["h_0"],
                                  kernel_shape=[1, 1], dilations=[1, 1],
                                  pads=[0, 0, 0, 0], group=1))

    for i, d in enumerate(dil):
        pad = (K - 1) * d
        inputs.append(helper.make_tensor_value_info(
            f"state_in_{i}", TensorProto.FLOAT, [1, C, V, pad]))
        outputs.append(helper.make_tensor_value_info(
            f"state_out_{i}", TensorProto.FLOAT, [1, C, V, pad]))

        cat = f"cat_{i}"
        nodes.append(helper.make_node("Concat", [f"state_in_{i}", f"h_{i}"],
                                      [cat], axis=3))
        wi, bi = w[f"conv{i}_w"], w[f"conv{i}_b"]
        if d == 1:                                   # native convolution k=1x3
            inits += [_init(f"W_{i}", wi.reshape(C, C, 1, K)), _init(f"B_{i}", bi)]
            nodes.append(helper.make_node("Conv", [cat, f"W_{i}", f"B_{i}"],
                                          [f"r_{i}"], kernel_shape=[1, K],
                                          dilations=[1, 1], pads=[0, 0, 0, 0],
                                          group=1))
        else:                                        # taps + Concat + 1x1
            taps = []
            for j in range(K):
                t = f"tap_{i}_{j}"
                _slice(nodes, inits, t, cat, j * d, j * d + T, tag=f"_{j}")
                taps.append(t)
            nodes.append(helper.make_node("Concat", taps, [f"g_{i}"], axis=1))
            # tap j <-> W[:, :, j]: tap 0 is the OLDEST (cat = [state, h])
            wg = np.concatenate([wi[:, :, j] for j in range(K)], axis=1)
            inits += [_init(f"W_{i}", wg.reshape(C, K * C, 1, 1)),
                      _init(f"B_{i}", bi)]
            nodes.append(helper.make_node("Conv", [f"g_{i}", f"W_{i}", f"B_{i}"],
                                          [f"r_{i}"], kernel_shape=[1, 1],
                                          dilations=[1, 1], pads=[0, 0, 0, 0],
                                          group=1))
        _slice(nodes, inits, f"state_out_{i}", cat, T, T + pad, tag="_st")
        nodes.append(helper.make_node("Relu", [f"r_{i}"], [f"a_{i}"]))
        nodes.append(helper.make_node("Add", [f"h_{i}", f"a_{i}"], [f"h_{i+1}"]))

    inits.append(_init("W_tail", w["tail"].reshape(c_out, C, 1, 1)))
    nodes.append(helper.make_node("Conv", [f"h_{layers}", "W_tail"], ["y"],
                                  kernel_shape=[1, 1], dilations=[1, 1],
                                  pads=[0, 0, 0, 0], group=1))
    outputs.insert(0, helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                                    [1, c_out, V, T]))
    g = helper.make_graph(nodes, name, inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


# -------------------------------------------------------- numpy reference
def ref_forward(w, x, states, layers=12, cycle=CYCLE):
    """StreamingTCN transcribed to numpy: h = h + relu(conv(cat(state,h)))."""
    dil = list(cycle) * (layers // len(cycle))

    def conv(inp, W, b=None, d=1):
        # inp[C,V,W], W[Co,Ci,k] -> out[Co,V,W-(k-1)d]
        Ci, V, Wd = inp.shape
        Co, _, k = W.shape
        ow = Wd - (k - 1) * d
        out = np.zeros((Co, V, ow), np.float64)
        for j in range(k):
            seg = inp[:, :, j * d: j * d + ow]                  # [Ci,V,ow]
            out += np.einsum('oi,ivw->ovw', W[:, :, j].astype(np.float64),
                             seg.astype(np.float64))
        if b is not None:
            out += b.astype(np.float64)[:, None, None]
        return out

    h = conv(x, w["head"][:, :, None])
    outs = []
    for i, d in enumerate(dil):
        cat = np.concatenate([states[i].astype(np.float64), h], axis=2)
        r = conv(cat, w[f"conv{i}_w"], w[f"conv{i}_b"], d=d)
        outs.append(cat[:, :, x.shape[2]:])
        h = h + np.maximum(r, 0.0)
    y = conv(h, w["tail"][:, :, None])
    return y, outs


def rand_weights(rng, c_in, C, c_out, layers):
    w = {"head": rng.standard_normal((C, c_in)) * 0.1,
         "tail": rng.standard_normal((c_out, C)) * 0.1}
    for i in range(layers):
        w[f"conv{i}_w"] = rng.standard_normal((C, C, K)) * (0.5 / np.sqrt(C))
        w[f"conv{i}_b"] = rng.standard_normal(C) * 0.01
    return {k: v.astype(np.float32) for k, v in w.items()}


# ------------------------------------------------------------------- checks
def selftest():
    import onnxruntime as ort
    rng = np.random.default_rng(20260802)
    c_in, C, c_out, V, T, L = 8, 88, 4, 2, 48, 12
    dil = list(CYCLE) * (L // len(CYCLE))
    w = rand_weights(rng, c_in, C, c_out, L)
    m = build_gather2(w, c_in, C, c_out, V, T, L)

    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    x = (rng.standard_normal((1, c_in, V, T)) * 0.3).astype(np.float32)
    st = [(rng.standard_normal((1, C, V, 2 * d)) * 0.2).astype(np.float32)
          for d in dil]
    feed = {"x": x}
    feed.update({f"state_in_{i}": s for i, s in enumerate(st)})
    got = sess.run(None, feed)

    y_ref, so_ref = ref_forward(w, x[0], [s[0] for s in st], L)
    e = np.abs(got[0][0] - y_ref).max() / (np.abs(y_ref).max() + 1e-30)
    print("1) graph vs numpy reference StreamingTCN: max rel. |diff| = %.3e" % e)
    assert e < 1e-5, e
    es = max(np.abs(got[1 + i][0] - so_ref[i]).max() for i in range(L))
    print("   state slices: max |diff| = %.3e" % es)
    assert es < 1e-5, es

    # 2) streaming: two consecutive chunks of T == one run over 2T
    x2 = (rng.standard_normal((1, c_in, V, 2 * T)) * 0.3).astype(np.float32)
    m2 = build_gather2(w, c_in, C, c_out, V, 2 * T, L)
    s2 = ort.InferenceSession(m2.SerializeToString(),
                              providers=["CPUExecutionProvider"])
    z = {f"state_in_{i}": np.zeros((1, C, V, 2 * d), np.float32)
         for i, d in enumerate(dil)}
    full = s2.run(None, dict(z, x=x2))[0]
    cur = {k: v.copy() for k, v in z.items()}
    chunks = []
    for c in range(2):
        r = sess.run(None, dict(cur, x=x2[:, :, :, c * T:(c + 1) * T]))
        chunks.append(r[0])
        cur = {f"state_in_{i}": r[1 + i] for i in range(L)}
    ch = np.concatenate(chunks, axis=3)
    e2 = np.abs(ch - full).max() / (np.abs(full).max() + 1e-30)
    print("2) chunks of 48 == one run of 96: max rel. |diff| = %.3e" % e2)
    assert e2 < 1e-5, e2

    # 3) tap repacking: shifting a tap breaks the answer (a "not mixed up" test)
    wbad = dict(w)
    wbad["conv1_w"] = w["conv1_w"][:, :, ::-1].copy()
    mb = build_gather2(wbad, c_in, C, c_out, V, T, L)
    sb = ort.InferenceSession(mb.SerializeToString(),
                              providers=["CPUExecutionProvider"])
    yb = sb.run(None, feed)[0]
    d3 = np.abs(yb - got[0]).max() / (np.abs(got[0]).max() + 1e-30)
    print("3) reversed taps in one layer change the output by %.1f%% — "
          "tap order is checked, not assumed" % (100 * d3))
    assert d3 > 1e-3, d3
    print("\nSELFTEST OK")
    return m


def compare(m, ref_path):
    ref = onnx.load(ref_path)
    def prof(g):
        c = collections.Counter(n.op_type for n in g.node)
        io = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(g.input) + list(g.output)}
        return c, io
    cm, iom = prof(m.graph)
    cr, ior = prof(ref.graph)
    qdq = {"QuantizeLinear", "DequantizeLinear"}
    print("nodes (the reference's Q/DQ are ignored — our graph is fp32):")
    keys = sorted(set(cm) | set(cr))
    ok = True
    for k in keys:
        a, b = cm.get(k, 0), cr.get(k, 0)
        mark = ""
        if k in qdq:
            mark = "(only in the QDQ reference)"
        elif k == "Relu":
            mark = "(folded into Quantize in QDQ — expected)" if b == 0 else ""
        elif a != b:
            mark = "<<< MISMATCH"; ok = False
        print("  %-18s ours %3d | reference %3d  %s" % (k, a, b, mark))
    # Slice — we cross-check not only the count but the parameters one by one:
    # this is exactly where "looks right but is not" slips through unnoticed
    def sl(g):
        ini = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        c = collections.Counter()
        for n in g.node:
            if n.op_type != "Slice":
                continue
            v = [ini[i].tolist() if i in ini else None for i in n.input[1:4]]
            c[tuple(tuple(z) if z else () for z in v)] += 1
        return c
    sm, sr = sl(m.graph), sl(ref.graph)
    dif = (sm - sr) + (sr - sm)
    print("\nSlice (starts/ends/axes) one by one: matched %d of %d%s" %
          (sum((sm & sr).values()), sum(sm.values()),
           "" if not dif else "  <<< MISMATCHES: %s" % dict(dif)))
    if dif:
        ok = False

    print("\ninputs/outputs:")
    for n in sorted(set(iom) | set(ior)):
        a, b = iom.get(n), ior.get(n)
        if a != b:
            print("  %-14s ours %s | reference %s  <<< MISMATCH" % (n, a, b)); ok = False
    print("  matched: %d names/shapes" % sum(1 for n in iom if iom.get(n) == ior.get(n)))
    print("\nSTRUCTURE %s" % ("MATCHES d44" if ok else "DIVERGES"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default="n6_gather2.onnx")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare")
    ap.add_argument("--C", type=int, default=88)
    ap.add_argument("--V", type=int, default=2)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--layers", type=int, default=12)
    a = ap.parse_args()

    if a.selftest:
        m = selftest()
        if a.compare:
            print(); compare(m, a.compare)
        return
    if a.compare and not a.ckpt:
        rng = np.random.default_rng(1)
        w = rand_weights(rng, 8, a.C, 4, a.layers)
        m = build_gather2(w, 8, a.C, 4, a.V, a.T, a.layers)
        return compare(m, a.compare)

    import torch
    # weights_only=False deliberately: OUR checkpoint holds not only tensors,
    # but also fir_W (numpy) and form (dict). The file is ours, not external.
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    C = ck.get("form", {}).get("c", a.C)
    L = ck.get("form", {}).get("layers", a.layers)
    w = {"head": sd["head.weight"].numpy()[:, :, 0],
         "tail": sd["tail.weight"].numpy()[:, :, 0]}
    for i in range(L):
        w[f"conv{i}_w"] = sd[f"blocks.{i}.conv.weight"].numpy()
        w[f"conv{i}_b"] = sd[f"blocks.{i}.conv.bias"].numpy()
    m = build_gather2(w, w["head"].shape[1], C, w["tail"].shape[0],
                      a.V, a.T, L)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(m, a.out)
    print("%s: C=%d L=%d V=%d T=%d, nodes %d" %
          (a.out, C, L, a.V, a.T, len(m.graph.node)))
    if a.compare:
        compare(m, a.compare)


if __name__ == "__main__":
    main()
