#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_gather2.py — int8 quantization of the canonical graph (step (c) of the
agenda).

THREE THINGS DONE HERE DIFFERENTLY FROM "JUST quantize_static":

1. CALIBRATION WITH THE CORPUS AND IN STREAMING. `export_t0.py:152` fed the
   calibration with Gaussian noise (x ~ N(0,1), state ~ 0.3·N(0,1)). Real
   subbands are sparse and spread 30-40 dB apart in level: a per-tensor scale
   taken from a Gaussian puts the residual of the upper bands inside half a
   quantization step, and on the board this looks like "the net does nothing".
   Here the calibration set is real corpus frames, and the states are NOT zero
   and NOT random: the graph is run by an fp32 session hop by hop, and the
   state_in values it generated itself go into the calibration. Otherwise the
   state range is understated by a factor of two or three.

2. TYING THE STATE SCALES. ORT quantizes state_in_i and state_out_i as
   different tensors and assigns them different scale/zero_point. But the
   output of hop n is the input of hop n+1: with different steps a
   requantization happens every 4 ms, and the rounding error recirculates
   around the state ring. Here the pairs are forced to a common (wider) scale
   and checked by an assert. m0_report closed the question of SPEED from tying
   (d36, zero), but the question of CORRECTNESS stayed open — it is here.

3. DEAD-BAND REPORT. It computes the fraction of output samples that landed
   exactly on zero, FOR EACH BAND separately, and the int8-vs-fp32 SNR per
   band. The upper bands are tens of dB quieter than the lower ones — if their
   dead-band is close to 100%, the per-tensor scale is not enough and we have
   to go per-channel or to int16.

RUN
  python quantize_gather2.py --selftest            # synthetic, without torch
  python quantize_gather2.py --onnx ../models/t1/n6_gather2.onnx \\
      --out ../models/t1/n6_gather2_qdq.onnx --hops 400 \\
      --compare ../models/t0/diag/d44_bh_c88_v2_l12_g2_qdq.onnx
"""
import argparse
import collections
import os
import sys

import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

HERE = os.path.dirname(os.path.abspath(__file__))
CYCLE = (1, 2, 4, 8, 16, 32)


def graph_dims(model):
    """(c_in, C, c_out, V, T, layers) from the declared graph shapes."""
    d = {v.name: [q.dim_value for q in v.type.tensor_type.shape.dim]
         for v in list(model.graph.input) + list(model.graph.output)}
    _, c_in, V, T = d["x"]
    _, C, _, _ = d["state_in_0"]
    c_out = d["y"][1]
    L = sum(1 for n in d if n.startswith("state_in_"))
    return c_in, C, c_out, V, T, L


# ---------------------------------------------------------- calibration reader
class StreamingReader(CalibrationDataReader):
    """Runs the fp32 graph hop by hop and yields REAL (x, state_in) sets.

    xs: a list of arrays [c_in, n12] — one per voice track. Voices are taken
    in pairs from DIFFERENT phrases (in the runtime they are independent)."""

    def __init__(self, model_path, xs, V, T, L, dil, max_hops):
        self.sess = ort.InferenceSession(model_path,
                                         providers=["CPUExecutionProvider"])
        self.C = graph_dims(onnx.load(model_path))[1]
        self.items = []
        st = {f"state_in_{i}": np.zeros((1, self.C, V, 2 * d), np.float32)
              for i, d in enumerate(dil)}
        tracks = [xs[i % len(xs)] for i in range(V)]
        nmin = min(t.shape[1] for t in tracks)
        n_hops = min(max_hops, nmin // T)
        for h in range(n_hops):
            x = np.stack([t[:, h * T:(h + 1) * T] for t in tracks], axis=1)
            x = x[None].astype(np.float32)                   # [1,c_in,V,T]
            feed = dict(st, x=x)
            self.items.append({k: v.copy() for k, v in feed.items()})
            out = self.sess.run(None, feed)
            st = {f"state_in_{i}": out[1 + i] for i in range(L)}
        self.it = iter(self.items)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.items)


# ------------------------------------------------ tying the scales of pairs
def tie_state_scales(path_in, path_out):
    """state_in_i and state_out_i -> a common scale/zp (take the wider step)."""
    m = onnx.load(path_in)
    init = {i.name: i for i in m.graph.initializer}

    def find(prefix, idx):
        """names of the (scale, zp) initializers belonging to the tensor."""
        s = z = None
        for n in m.graph.node:
            if n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
                continue
            names = list(n.input) + list(n.output)
            if any(x.startswith(f"{prefix}_{idx}") and
                   (x == f"{prefix}_{idx}" or x.startswith(f"{prefix}_{idx}_"))
                   for x in names):
                if len(n.input) >= 3 and n.input[1] in init and n.input[2] in init:
                    s, z = n.input[1], n.input[2]
        return s, z

    L = sum(1 for v in m.graph.input if v.name.startswith("state_in_"))
    tied, skipped, spread = 0, [], []
    for i in range(L):
        si, zi = find("state_in", i)
        so, zo = find("state_out", i)
        if not (si and so):
            skipped.append(i); continue
        a = float(numpy_helper.to_array(init[si]))
        b = float(numpy_helper.to_array(init[so]))
        wide = max(a, b)
        spread.append(abs(a - b) / max(min(a, b), 1e-30))
        for nm in (si, so):
            arr = numpy_helper.to_array(init[nm]).copy()
            init[nm].CopyFrom(numpy_helper.from_array(
                np.array(wide, arr.dtype), nm))
        if zi and zo:
            za = numpy_helper.to_array(init[zi])
            for nm in (zi, zo):
                init[nm].CopyFrom(numpy_helper.from_array(
                    np.array(za, numpy_helper.to_array(init[nm]).dtype), nm))
        tied += 1
    onnx.save(m, path_out)
    return tied, skipped, spread


def check_state_scales(path):
    """Assert: every state_in_i / state_out_i pair has the same step."""
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    got = collections.defaultdict(dict)
    for n in m.graph.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for x in list(n.input) + list(n.output):
            for kind in ("state_in", "state_out"):
                if x.startswith(kind + "_") and len(n.input) >= 2:
                    idx = x[len(kind) + 1:].split("_")[0]
                    if idx.isdigit() and n.input[1] in init:
                        got[int(idx)][kind] = float(init[n.input[1]])
    bad = [i for i, d in got.items()
           if "state_in" in d and "state_out" in d
           and not np.isclose(d["state_in"], d["state_out"], rtol=1e-6)]
    return got, bad





# --------------------------------- channel equalization (cross-layer equalization)
def channel_rms(fp32_path, reader, L, dil, C, V, nhops=60):
    """Mean RMS of each C-space channel over all h_i / a_i."""
    g = onnx.load(fp32_path)
    keep = [f"h_{i}" for i in range(L + 1)] + [f"a_{i}" for i in range(L)]
    have = {v.name for v in g.graph.output}
    for t in keep:
        if t not in have:
            g.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    sess = ort.InferenceSession(g.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    acc = np.zeros(C); n = 0
    st = {f"state_in_{i}": np.zeros((1, C, V, 2 * d), np.float32)
          for i, d in enumerate(dil)}
    for it in reader.items[:nhops]:
        d = dict(zip(outs, sess.run(None, dict(st, x=it["x"]))))
        for t in keep:
            a = d[t][0].astype(np.float64)
            acc += (a ** 2).mean(axis=(1, 2)); n += 1
        st = {f"state_in_{i}": d[f"state_out_{i}"] for i in range(L)}
    return np.sqrt(acc / max(n, 1))


def equalize_channels(fp32_path, out_path, rms, L, C):
    """A diagonal D in C-space, exact by construction.

    h'_i = D h_i for ALL i at once:
      W_head' = D W_head          W_i' = D W_i D^-1 (per tap block)
      b_i'    = D b_i             W_tail' = W_tail D^-1
    ReLU is positively homogeneous (D>0), Add is linear => the graph function
    does NOT change, and neither do the input x and the output y. Only the
    internal states change — and they are opaque ping-pong buffers, nobody looks
    at them from outside. That is why THE RUNTIME NEEDS NO FIX, unlike with band
    balancing.

    The point: a per-tensor scale serves the whole tensor, while our channels are
    spread 60-116 dB apart against an int8 budget of ~48 dB. We equalize them
    with a single diagonal."""
    m = onnx.load(fp32_path)
    init = {i.name: i for i in m.graph.initializer}
    r = np.asarray(rms, np.float64).copy()
    live = r > 0
    d = np.ones(C)
    d[live] = 1.0 / r[live]
    d[live] /= np.exp(np.log(d[live]).mean())          # geometric mean = 1
    dinv = 1.0 / d

    def put(name, arr):
        init[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    wh = numpy_helper.to_array(init["W_head"]).astype(np.float64)
    put("W_head", wh * d[:, None, None, None])
    wt = numpy_helper.to_array(init["W_tail"]).astype(np.float64)
    put("W_tail", wt * dinv[None, :, None, None])
    for i in range(L):
        w = numpy_helper.to_array(init[f"W_{i}"]).astype(np.float64)
        cin = w.shape[1]
        reps = cin // C                                  # 1 (d=1) or 3 (taps)
        put(f"W_{i}", w * d[:, None, None, None] *
            np.tile(dinv, reps)[None, :, None, None])
        b = numpy_helper.to_array(init[f"B_{i}"]).astype(np.float64)
        put(f"B_{i}", b * d)
    mp = m.metadata_props.add(); mp.key = "n6_equalized"; mp.value = "1"
    onnx.save(m, out_path)
    return d


# ------------------------------------------------- per-tensor diagnostics
def tensor_stats(fp32_path, reader, L, dil, C, V, T, topn=14):
    """Exactly where per-tensor int8 is bound to break.

    For every intermediate tensor we compute the SPREAD ACROSS CHANNELS
    (max channel RMS / min channel RMS, in dB) and the ratio |mean|/stddev.
    One scale for a whole tensor physically cannot serve channels spread
    40+ dB apart: the quiet ones fall into half a step. This is the test for
    "have we hit the per-tensor limit or not yet"."""
    m = onnx.load(fp32_path)
    keep = [n.output[0] for n in m.graph.node
            if n.op_type in ("Conv", "Add", "Concat", "Relu")]
    g = onnx.load(fp32_path)
    names = {v.name for v in g.graph.output}
    for t in keep:
        if t not in names:
            g.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    sess = ort.InferenceSession(g.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    acc = {t: [] for t in keep}
    st = {f"state_in_{i}": np.zeros((1, C, V, 2 * d), np.float32)
          for i, d in enumerate(dil)}
    for it in reader.items[:60]:
        r = sess.run(None, dict(st, x=it["x"]))
        d = dict(zip(outs, r))
        for t in keep:
            a = d[t]
            if a.ndim == 4:
                acc[t].append(np.sqrt((a[0].astype(np.float64) ** 2)
                                      .mean(axis=(1, 2))))
        st = {f"state_in_{i}": d[f"state_out_{i}"] for i in range(L)}
    rows = []
    for t, v in acc.items():
        if not v:
            continue
        r = np.mean(np.stack(v), axis=0)
        r = r[r > 0]
        if r.size < 2:
            continue
        rows.append((t, 20 * np.log10(r.max() / max(r.min(), 1e-30)), r.size))
    rows.sort(key=lambda z: -z[1])
    print("\nRMS spread ACROSS CHANNELS inside a tensor (how much ONE "
          "per-tensor scale has to cover):")
    for t, db, n in rows[:topn]:
        flag = "  <<< per-tensor does not work here" if db > 40 else ""
        print("  %-14s %6.1f dB  (%d channels)%s" % (t, db, n, flag))
    worst = rows[0][1] if rows else 0.0
    print("  worst tensor: %.1f dB. int8 gives ~48 dB of full range, "
          "so %.0f dB is left per channel." % (worst, max(0, 48 - worst)))
    return rows


# ------------------------------------------------- band balancing
def band_stats(fp32_path, reader, c_in, c_out, V, T, L, dil, C):
    """RMS of EACH band at the input (channels 0..3) and at the output, fp32."""
    s = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    st = {f"state_in_{i}": np.zeros((1, C, V, 2 * d), np.float32)
          for i, d in enumerate(dil)}
    ex = np.zeros(4); ey = np.zeros(c_out); n = 0
    for it in reader.items:
        o = s.run(None, dict(st, x=it["x"]))
        xb, y = it["x"][0][:4], o[0][0]
        for b in range(4):
            ex[b] += float((xb[b] ** 2).sum())
        for b in range(c_out):
            ey[b] += float((y[b] ** 2).sum())
        n += y[0].size
        st = {f"state_in_{i}": o[1 + i] for i in range(L)}
    return np.sqrt(ex / max(n, 1)), np.sqrt(ey / max(n, 1))


def balance_graph(fp32_path, out_path, rms_in, rms_out):
    """Diagonal band normalization folded into W_head and W_tail.

    Per-tensor quantization gives ONE step for the whole tensor, while the bands
    are spread 30-40 dB apart in level => the quiet bands fall inside half a
    step and are rounded to zero. The cure is to move the diagonal gain across
    the quantization boundary: the graph sees equalized bands, and the M55
    restores the original balance with eight multiplications per sample.

      input:  the graph gets x'[b] = x[b] / s_in[b]  =>  W_head'[:,b] = W_head[:,b]*s_in[b]
      output: the graph emits y'[b] = y[b] / s_out[b]  =>  W_tail'[b,:] = W_tail[b,:]/s_out[b]

    The fp32 function does NOT change (checked numerically in the selftest)."""
    m = onnx.load(fp32_path)
    init = {i.name: i for i in m.graph.initializer}
    s_in = (rms_in / max(rms_in.max(), 1e-30)).astype(np.float64)
    s_out = (rms_out / max(rms_out.max(), 1e-30)).astype(np.float64)
    s_in = np.maximum(s_in, 1e-4); s_out = np.maximum(s_out, 1e-4)

    wh = numpy_helper.to_array(init["W_head"]).astype(np.float64).copy()
    for b in range(min(4, wh.shape[1])):
        wh[:, b] *= s_in[b]
    init["W_head"].CopyFrom(numpy_helper.from_array(
        wh.astype(np.float32), "W_head"))

    wt = numpy_helper.to_array(init["W_tail"]).astype(np.float64).copy()
    for b in range(wt.shape[0]):
        wt[b] /= s_out[b]
    init["W_tail"].CopyFrom(numpy_helper.from_array(
        wt.astype(np.float32), "W_tail"))
    # marker: balancing is NOT idempotent. On a second pass s_out comes out ~1
    # (the output is already equalized), but s_in is computed from the RAW input,
    # which balancing does not change, and would be folded into W_head a second
    # time. The identity check does not see this: it compares the new graph with
    # the old one, not with the original.
    mp = m.metadata_props.add(); mp.key = "n6_balanced"; mp.value = "1"
    mp = m.metadata_props.add()
    mp.key = "n6_s_in"; mp.value = ",".join("%.9g" % v for v in s_in)
    mp = m.metadata_props.add()
    mp.key = "n6_s_out"; mp.value = ",".join("%.9g" % v for v in s_out)
    onnx.save(m, out_path)
    return s_in, s_out


def already_balanced(model):
    d = {p.key: p.value for p in model.metadata_props}
    if d.get("n6_balanced") != "1":
        return None
    return (np.array([float(v) for v in d["n6_s_in"].split(",")]),
            np.array([float(v) for v in d["n6_s_out"].split(",")]))


def _cfloat(v):
    """A valid C float literal. "%.9g" on a round number gives "1", and gluing
    "f" onto it produces "1f" — which is NOT a floating-point constant: the C
    standard requires either a dot or an exponent. The build failed on the very
    first include. Found on 3 Aug while preparing M2, before the first firmware
    build."""
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("non-numeric coefficient for the header: %r" % v)
    s = "%.9g" % v
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def emit_scales_header(path, s_in, s_out, meta):
    body = ["/* n6_npu_scales.h — GENERATED by train/quantize_gather2.py.",
            " * Band balancing around the int8 boundary (see balance_graph).",
            " *   before submit: x_cond[b][v][t] /= n6_npu_s_in[b]",
            " *   after DONE:    residual[b][v][t] *= n6_npu_s_out[b]",
            " * Source: %s" % meta, " */",
            "#ifndef N6_NPU_SCALES_H", "#define N6_NPU_SCALES_H", "",
            "static const float n6_npu_s_in[4]  = { %s };"
            % ", ".join(_cfloat(v) for v in s_in),
            "static const float n6_npu_s_out[4] = { %s };"
            % ", ".join(_cfloat(v) for v in s_out), "", "#endif", ""]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(body))


# ---------------------------------------------------------- int8 report
def deadband_report(fp32_path, int8_path, reader, c_out, V, T, L, dil, C):
    s32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    st32 = {f"state_in_{i}": np.zeros((1, C, V, 2 * d), np.float32)
            for i, d in enumerate(dil)}
    st8 = {k: v.copy() for k, v in st32.items()}
    num = np.zeros(c_out); den = np.zeros(c_out); zero = np.zeros(c_out); tot = 0
    for it in reader.items:
        x = it["x"]
        o32 = s32.run(None, dict(st32, x=x)); o8 = s8.run(None, dict(st8, x=x))
        y32, y8 = o32[0][0], o8[0][0]                      # [c_out,V,T]
        for b in range(c_out):
            den[b] += float((y32[b] ** 2).sum())
            num[b] += float(((y8[b] - y32[b]) ** 2).sum())
            zero[b] += float((np.abs(y8[b]) < 1e-12).sum())
        tot += y32[0].size
        st32 = {f"state_in_{i}": o32[1 + i] for i in range(L)}
        st8 = {f"state_in_{i}": o8[1 + i] for i in range(L)}
    snr = 10 * np.log10(np.maximum(den, 1e-30) / np.maximum(num, 1e-30))
    return snr, zero / max(tot, 1)


def node_profile(path):
    return collections.Counter(n.op_type for n in onnx.load(path).graph.node)


# --------------------------------------------------------------- run
def run(fp32_path, out_path, xs, hops, method, compare_ref=None,
        balance=True, hdr_path=None, stats=False, act16=False,
        equalize=True):
    m = onnx.load(fp32_path)
    c_in, C, c_out, V, T, L = graph_dims(m)
    dil = list(CYCLE) * (L // len(CYCLE))
    print("graph: c_in=%d C=%d c_out=%d V=%d T=%d L=%d" % (c_in, C, c_out, V, T, L))

    reader = StreamingReader(fp32_path, xs, V, T, L, dil, hops)
    print("calibration: %d hops, states are ITS OWN (not zeros, not Gaussian)"
          % len(reader.items))
    if stats:
        tensor_stats(fp32_path, reader, L, dil, C, V, T)
        return None, None, []

    src = fp32_path
    if equalize and not any(p.key == "n6_equalized" for p in m.metadata_props):
        rms = channel_rms(fp32_path, reader, L, dil, C, V)
        live = int((rms > 0).sum())
        rl = rms[rms > 0]
        print("C-space channels: %d live of %d, spread %.1f dB"
              % (live, C, 20 * np.log10(rl.max() / max(rl.min(), 1e-30))))
        src = out_path.replace(".onnx", "_eq.onnx")
        dv = equalize_channels(fp32_path, src, rms, L, C)
        sa = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        sb = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
        it0 = reader.items[len(reader.items) // 2]
        ya = sa.run(None, it0)[0]
        z = {k: (v if k == "x" else v * dv.astype(np.float32)[None, :, None, None])
             for k, v in it0.items()}
        yb = sb.run(None, z)[0]
        e = np.abs(ya - yb).max() / (np.abs(ya).max() + 1e-30)
        print("  equalization identity check: %.2e %s"
              % (e, "OK" if e < 1e-4 else "<<< VIOLATED"))
        reader = StreamingReader(src, xs, V, T, L, dil, hops)
        m = onnx.load(src)
        fp32_path = src
    prev = already_balanced(m)
    if prev is not None and balance:
        s_in, s_out = prev
        print("the graph is ALREADY balanced (marker in metadata) — leaving it alone.")
        print("  scales from the graph: s_in %s | s_out %s"
              % (np.array2string(s_in, precision=4),
                 np.array2string(s_out, precision=4)))
        balance = False
    if balance:
        rin, rout = band_stats(fp32_path, reader, c_in, c_out, V, T, L, dil, C)
        print("RMS per band, fp32: input %s | output %s"
              % (np.array2string(rin, precision=4),
                 np.array2string(rout, precision=4)))
        print("  input spread %.1f dB, output spread %.1f dB — this is exactly "
              "what drowns the quiet bands in a per-tensor quant"
              % (20 * np.log10(rin.max() / max(rin.min(), 1e-30)),
                 20 * np.log10(rout.max() / max(rout.min(), 1e-30))))
        src = out_path.replace(".onnx", "_balanced.onnx")
        s_in, s_out = balance_graph(fp32_path, src, rin, rout)
        hdr = hdr_path or os.path.normpath(
            os.path.join(HERE, "..", "fw", "n6_npu_scales.h"))
        emit_scales_header(hdr, s_in, s_out, os.path.basename(fp32_path))
        # CHECK: balancing must be an identity in fp32
        sa = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        sb = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
        it0 = reader.items[len(reader.items) // 2]
        xa = dict(it0)
        xb = dict(it0); xb["x"] = xa["x"].copy()
        xb["x"][0, :4] /= s_in.astype(np.float32)[:, None, None]
        ya = sa.run(None, xa)[0][0]
        yb = sb.run(None, xb)[0][0] * s_out.astype(np.float32)[:, None, None]
        eb = np.abs(ya - yb).max() / (np.abs(ya).max() + 1e-30)
        print("  identity check in fp32: max rel. |diff| = %.2e %s"
              % (eb, "OK" if eb < 1e-4 else "<<< VIOLATED"))
        print("  scales: s_in %s | s_out %s -> %s"
              % (np.array2string(s_in, precision=4),
                 np.array2string(s_out, precision=4), hdr))
        # rebuild the reader: the ALREADY balanced graph is what must be calibrated
        reader = StreamingReader(src, xs, V, T, L, dil, hops)

    tmp = out_path + ".untied"
    quantize_static(src, tmp, reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=(QuantType.QInt16 if act16
                                     else QuantType.QInt8),
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    calibrate_method=method,
                    extra_options={"ActivationSymmetric": False})
    tied, skipped, spread = tie_state_scales(tmp, out_path)
    print("state scales tied: %d pairs%s" %
          (tied, "" if not skipped else ", NOT found: %s" % skipped))
    if spread:
        print("  before tying, the pair steps diverged: median %.1f%%, max %.1f%%"
              % (100 * float(np.median(spread)), 100 * max(spread)))
    got, bad = check_state_scales(out_path)
    print("assert scale(state_in_i)==scale(state_out_i): %s"
          % ("OK" if not bad else "VIOLATED for %s" % bad))

    reader.rewind()
    snr, dead = deadband_report(src, out_path, reader, c_out, V, T, L,
                                dil, C)
    print("\nint8 against fp32, PER BAND:")
    for b in range(c_out):
        flag = "  <<< DEAD BAND" if dead[b] > 0.5 else ""
        print("  band %d: SNR %+6.1f dB | exactly zero samples %5.1f%%%s"
              % (b, snr[b], 100 * dead[b], flag))
    if compare_ref:
        a, r = node_profile(out_path), node_profile(compare_ref)
        print("\nQDQ profile against d44:")
        for k in sorted(set(a) | set(r)):
            mark = "" if a.get(k, 0) == r.get(k, 0) else "  <<< mismatch"
            print("  %-18s ours %3d | d44 %3d%s" % (k, a.get(k, 0), r.get(k, 0), mark))
    if os.path.exists(tmp):
        os.remove(tmp)
    return snr, dead, bad


def selftest():
    """Synthetic: we check the MECHANICS (streaming calibration, tying, the
    report), not the ranges of the real corpus."""
    sys.path.insert(0, HERE)
    import export_gather2 as eg
    rng = np.random.default_rng(20260802)
    c_in, C, c_out, V, T, L = 8, 32, 4, 2, 48, 12       # smaller C — faster
    w = eg.rand_weights(rng, c_in, C, c_out, L)
    # spread the OUTPUT in level too (~35 dB) — otherwise the synthetic case
    # does not reproduce the production condition and the selftest proves nothing
    w["tail"] = (w["tail"] * np.array([1.0, 0.1, 0.03, 0.015])[:, None]
                 ).astype(np.float32)
    m = eg.build_gather2(w, c_in, C, c_out, V, T, L)
    td = tempfile.gettempdir()
    fp = os.path.join(td, "_st_fp32.onnx"); onnx.save(m, fp)
    # "subbands": the bands are spread 30 dB apart in level — as in real life
    lvl = np.array([1.0, 0.3, 0.05, 0.01])[:, None]
    xs = []
    for k in range(4):
        a = rng.standard_normal((c_in, 48 * 60)) * 0.1
        a[:4] *= lvl
        xs.append(a.astype(np.float32))
    print("\n--- WITHOUT balancing (control) ---")
    snr0, dead0, _ = run(fp, os.path.join(td, "_st_q0.onnx"), xs, 40,
                         CalibrationMethod.MinMax, balance=False)
    print("\n--- WITH balancing ---")
    snr, dead, bad = run(fp, os.path.join(td, "_st_q.onnx"), xs, 40,
                         CalibrationMethod.MinMax,
                         hdr_path=os.path.join(td, "_st_scales.h"))
    assert not bad, bad
    print("\n--- RESULT for the quiet bands (1..3) ---")
    for b in range(1, c_out):
        print("  band %d: SNR %+6.1f -> %+6.1f dB | zeros %5.1f%% -> %5.1f%%"
              % (b, snr0[b], snr[b], 100 * dead0[b], 100 * dead[b]))
    assert dead[1:].max() < max(0.2, dead0[1:].max()), (dead0, dead)
    assert snr[1:].min() > snr0[1:].min() + 3.0, (snr0, snr)
    print("\nSELFTEST OK — balancing pulls the quiet bands out of the dead band")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "..", "models", "t1",
                                                   "n6_gather2.onnx"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "models", "t1",
                                                  "n6_gather2_qdq.onnx"))
    ap.add_argument("--hops", type=int, default=400)
    ap.add_argument("--phrases", type=int, default=24)
    ap.add_argument("--method", default="minmax", choices=["minmax", "percentile"])
    ap.add_argument("--compare")
    ap.add_argument("--no-equalize", action="store_true",
                    help="no channel equalization — for a \"before/after\" comparison")
    ap.add_argument("--stats", action="store_true",
                    help="diagnostics only: channel spread inside the tensors")
    ap.add_argument("--act16", action="store_true",
                    help="int16 activations instead of int8 (check whether we hit per-tensor)")
    ap.add_argument("--no-balance", action="store_true",
                    help="no band balancing — for a \"before/after\" comparison")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    # the corpus is the same as teacher_search/exam_delta's (the same SEED)
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "dsp"))
    import teacher_search as ts
    ts.N_PHRASES = a.phrases
    print("[corpus] rendering %d dry phrases..." % a.phrases, flush=True)
    items = ts.dry_corpus()
    cand = dict((c[0], c) for c in ts.CANDS)["A2_ottpress"]
    xs = [ts.build_xy(it, np.asarray(cand[1](it["dry"])), cand[2])[0]
          for it in items]
    meth = (CalibrationMethod.MinMax if a.method == "minmax"
            else CalibrationMethod.Percentile)
    run(a.onnx, a.out, xs, a.hops, meth, a.compare,
        balance=not a.no_balance, stats=a.stats, act16=a.act16,
        equalize=not a.no_equalize)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
