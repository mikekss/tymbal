#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0.py — the T0 matrix of StreamingTCN ONNX graphs for stedgeai (guide §3.1–3.2).

The graph is built BY HAND through onnx.helper (not torch.onnx.export): the
protobuf is clean, without exporter quirks — minimum surprises for the atonn
compiler. The numerical reference is the numpy TCN from streaming_tcn_check.py
(with chunked==full proven). Before the FINAL training run stedgeai once more on
the graph from torch.onnx.export (train/streaming_tcn.py) — the structures must match.

The matrix (all shapes static, the batch is fixed for good — §3.2):
  wiring (a) batch=voices×bands:  c_in=9, c_out=1, C∈{96,128}, B=12, T=48
      + C=128 T=96 (time batch 2×hop)  + C=128 notail (tail in f32 on the M55, §3.3)
  wiring (b) bands-as-channels:   c_in=8, c_out=4, C=192,  B=3,  T=48 (D-5)
  + QDQ-int8 variants of the two main ones (through onnxruntime.quantization; the
    state scales are NOT tied in/out here — for T0 compilability that does not
    matter, in the final version QAT/brevitas holds the scales, §3.3).

For each graph: an equivalence test through onnxruntime —
  (1) THE MAIN ONE: T chunks with state carry-over == the float64 numpy reference
      (exactly the proven scheme from streaming_tcn_check), tolerance 1e-4 rel.;
  (2) for reference: ORT chunks vs the full ORT graph in one call. It diverges by
      ~1e-5 rel. — reassociation of fp32 sums in the ORT conv kernels on a long T,
      it has nothing to do with the state scheme (the numpy cross-check proves it).

Output: models/t0/*.onnx + models/t0/t0_manifest.md (shapes, parameters, MAC,
cross-check with the budget calculator). Run: python3 export_t0.py
"""
import os
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP, numpy_helper as NH

import streaming_tcn_check as ref

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0")
K = 3
CYCLE = (1, 2, 4, 8, 16, 32)
LAYERS = 24


# ----------------------------------------------------------- ONNX construction
def build_graph(net, c_in, c_out, B, T, with_tail=True):
    """net — ref.TCN (float64 weights -> cast to float32). Returns a ModelProto.
    Input x[B,c_in,T], states state_in_i[B,C,pad_i]; outputs y + state_out_i.
    state_out_i = the last pad_i columns of Concat(state_in_i, h) — it works
    even when T < pad_i (the tail grabs part of the old state)."""
    C = net.c_hidden
    inits, nodes, inputs, outputs = [], [], [], []

    inputs.append(H.make_tensor_value_info("x", TP.FLOAT, [B, c_in, T]))
    inits.append(NH.from_array(net.head.astype(np.float32)[:, :, None], "W_head"))
    nodes.append(H.make_node("Conv", ["x", "W_head"], ["h0"],
                             kernel_shape=[1], dilations=[1], pads=[0, 0], strides=[1]))
    h = "h0"
    for i, cv in enumerate(net.convs):
        pad = cv.pad
        s_in, s_out, cat = f"state_in_{i}", f"state_out_{i}", f"cat_{i}"
        inputs.append(H.make_tensor_value_info(s_in, TP.FLOAT, [B, C, pad]))
        nodes.append(H.make_node("Concat", [s_in, h], [cat], axis=2))
        inits.append(NH.from_array(cv.w.astype(np.float32), f"W_{i}"))
        inits.append(NH.from_array(cv.b.astype(np.float32), f"B_{i}"))
        nodes.append(H.make_node("Conv", [cat, f"W_{i}", f"B_{i}"], [f"z_{i}"],
                                 kernel_shape=[cv.k], dilations=[cv.d],
                                 pads=[0, 0], strides=[1]))
        nodes.append(H.make_node("Relu", [f"z_{i}"], [f"r_{i}"]))
        nodes.append(H.make_node("Add", [h, f"r_{i}"], [f"h_{i + 1}"]))
        # state slice: [:, :, T : T+pad] of cat (length pad+T)
        inits.append(NH.from_array(np.array([T], np.int64), f"ss_{i}"))
        inits.append(NH.from_array(np.array([T + pad], np.int64), f"se_{i}"))
        inits.append(NH.from_array(np.array([2], np.int64), f"sa_{i}"))
        nodes.append(H.make_node("Slice", [cat, f"ss_{i}", f"se_{i}", f"sa_{i}"], [s_out]))
        outputs.append(H.make_tensor_value_info(s_out, TP.FLOAT, [B, C, pad]))
        h = f"h_{i + 1}"

    if with_tail:
        inits.append(NH.from_array(net.tail.astype(np.float32)[:, :, None], "W_tail"))
        nodes.append(H.make_node("Conv", [h, "W_tail"], ["y"],
                                 kernel_shape=[1], dilations=[1], pads=[0, 0], strides=[1]))
        y_info = H.make_tensor_value_info("y", TP.FLOAT, [B, c_out, T])
    else:  # the M55 runs the tail in f32 — expose the last block's activations (§3.3)
        nodes.append(H.make_node("Identity", [h], ["y"]))
        y_info = H.make_tensor_value_info("y", TP.FLOAT, [B, C, T])
    outputs.insert(0, y_info)

    graph = H.make_graph(nodes, "n6_streaming_tcn", inputs, outputs, inits)
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)],
                         producer_name="n6-export_t0")
    model.ir_version = 9          # conservative, for older parsers
    onnx.checker.check_model(model)
    return model


# ----------------------------------------------------------- equivalence (ORT)
def ort_run_chunked(sess, x_full, net, B, T):
    """The full sequence in T chunks with state carry-over through the I/O."""
    states = {f"state_in_{i}": np.zeros((B, net.c_hidden, cv.pad), np.float32)
              for i, cv in enumerate(net.convs)}
    ys = []
    for t0 in range(0, x_full.shape[2], T):
        feed = {"x": x_full[:, :, t0:t0 + T]} | states
        res = sess.run(None, feed)
        ys.append(res[0])
        states = {f"state_in_{i}": res[1 + i] for i in range(len(net.convs))}
    return np.concatenate(ys, axis=2)


def equivalence(model_path, net, c_in, c_out, B, T, t_total=None, with_tail=True):
    import onnxruntime as ort
    t_total = t_total or 5 * T   # a multiple of T (chunks with no remainder)
    rng = np.random.default_rng(0xA11CE)
    x = rng.standard_normal((B, c_in, t_total)).astype(np.float32)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    y_chunk = ort_run_chunked(sess, x, net, B, T)

    # the full graph in one call (T = t_total), the same weights — a temporary file
    full = build_graph(net, c_in, c_out, B, t_total, with_tail)
    fp = model_path + ".full.tmp"
    onnx.save(full, fp)
    sessf = ort.InferenceSession(fp, providers=["CPUExecutionProvider"])
    feed = {"x": x} | {f"state_in_{i}": np.zeros((B, net.c_hidden, cv.pad), np.float32)
                       for i, cv in enumerate(net.convs)}
    y_full = sessf.run(None, feed)[0]
    os.remove(fp)

    # numpy reference (float64, per batch element); for notail — hidden activations
    def ref_one(xb):
        if with_tail:
            return net.full(xb)
        h = net.head @ xb
        for cv in net.convs:
            h = h + np.maximum(cv.full(h), 0.0)
        return h
    nb = min(B, 2)   # the batch is trivially parallel — 2 elements are enough here
    y_ref = np.stack([ref_one(x[b].astype(np.float64)) for b in range(nb)])
    denom = float(np.max(np.abs(y_ref))) + 1e-12
    e_ref = float(np.max(np.abs(y_chunk[:nb] - y_ref)) / denom)  # THE MAIN criterion
    e_stream = float(np.max(np.abs(y_chunk - y_full)) / denom)   # for reference
    return e_stream, e_ref


# --------------------------------------------------------------------- QDQ int8
def make_qdq(src_path, dst_path, net, c_in, B, T, n_calib=8):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            rng = np.random.default_rng(7)
            self.data = []
            for _ in range(n_calib):
                d = {"x": rng.standard_normal((B, c_in, T)).astype(np.float32)}
                for i, cv in enumerate(net.convs):
                    d[f"state_in_{i}"] = (0.3 * rng.standard_normal(
                        (B, net.c_hidden, cv.pad))).astype(np.float32)
                self.data.append(d)
            self.it = iter(self.data)

        def get_next(self):
            return next(self.it, None)

    quantize_static(src_path, dst_path, Reader(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)


# ----------------------------------------------------------------------- matrix
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    manifest = ["# T0 manifest of the ONNX graphs (autogenerated by export_t0.py)\n",
                "| file | wiring | C | B | T | c_in→c_out | parameters | MMAC/call | ORT-vs-ORT | chunks vs reference |",
                "|---|---|---|---|---|---|---|---|---|---|"]

    cases = [
        # name, wiring, C, B, T, c_in, c_out, tail
        ("a_c96_b12_t48",  "batch",    96, 12, 48, 9, 1, True),
        ("a_c128_b12_t48", "batch",   128, 12, 48, 9, 1, True),
        ("a_c128_b12_t96", "batch",   128, 12, 96, 9, 1, True),
        ("a_c128_b12_t48_notail", "batch", 128, 12, 48, 9, 1, False),
        ("b_c192_b3_t48",  "channels", 192, 3, 48, 8, 4, True),
    ]
    nets = {}
    for name, wiring, C, B, T, c_in, c_out, tail in cases:
        key = (C, c_in, c_out)
        if key not in nets:   # one net per shape — notail shares weights with the tail version
            nets[key] = ref.TCN(rng, c_in, C, c_out, LAYERS, CYCLE, k=K)
        net = nets[key]
        model = build_graph(net, c_in, c_out, B, T, tail)
        path = os.path.join(OUT, name + ".onnx")
        onnx.save(model, path)
        e_s, e_r = equivalence(path, net, c_in, c_out, B, T, with_tail=tail)
        n_par = (net.head.size + net.tail.size * tail
                 + sum(cv.w.size + cv.b.size for cv in net.convs))
        mmac = (LAYERS * C * C * K + c_in * C + (C * c_out if tail else 0)) * T * B / 1e6
        ok = "OK" if e_r <= 1e-4 else "FAIL"
        manifest.append(f"| {name}.onnx | {wiring} | {C} | {B} | {T} | {c_in}→"
                        f"{c_out if tail else C} | {n_par/1e6:.2f}M | {mmac:.0f} | "
                        f"{e_s:.1e} | {e_r:.1e} {ok} |")
        print(f"[{name}] chunks vs numpy reference: {e_r:.2e} rel. ({ok}); "
              f"ORT-vs-ORT for reference: {e_s:.2e}; {n_par/1e6:.2f}M params.")
        assert e_r <= 1e-4, name

    # cross-check with the budget calculator (§12 of the guide)
    b128 = ref.budget(128, LAYERS, 3)
    b192 = ref.budget(192, LAYERS, 3, bands=1, c_in=8, c_out=4)
    manifest.append(f"\nCross-check: batch C=128 {b128['gops_voice']:.0f} GOPS/voice (spec ~113), "
                    f"weights {b128['weights_mb']:.2f} MB; channels C=192 "
                    f"{b192['gops_voice']:.0f} GOPS/voice (§3.1 ~64), weights {b192['weights_mb']:.2f} MB.")

    # QDQ-int8 of the two main ones
    for base, wiring, C, B, T, c_in, c_out in [
            ("a_c128_b12_t48", "batch", 128, 12, 48, 9, 1),
            ("b_c192_b3_t48", "channels", 192, 3, 48, 8, 4)]:
        src = os.path.join(OUT, base + ".onnx")
        dst = os.path.join(OUT, base + "_qdq.onnx")
        try:
            make_qdq(src, dst, nets[(C, c_in, c_out)], c_in, B, T)
            print(f"[{base}_qdq] QDQ-int8 written")
            manifest.append(f"| {base}_qdq.onnx | {wiring}, int8 QDQ | {C} | {B} | {T} "
                            f"| {c_in}→{c_out} | — | — | — | — |")
        except Exception as e:
            print(f"[{base}_qdq] QDQ skipped: {e}")
            manifest.append(f"QDQ {base}: not created ({e}) — do it through brevitas on the training machine.")

    with open(os.path.join(OUT, "t0_manifest.md"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print("done:", OUT)


if __name__ == "__main__":
    main()
