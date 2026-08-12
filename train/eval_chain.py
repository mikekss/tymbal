#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_chain.py — the DECISIVE measurement of the refiner: three numbers on one scale.

The question that had been hanging open until now (I put it to myself on 2 Aug):
    "The FIR gives +8.77 dB, FIR+net in fp32 gives +17.95. If the int8 version
     gives +16, then everything is fine and my SNR is just scaring me for
     nothing. If it gives +7, the NPU is hurting."

Not one of the existing utilities answers that question:
  - exam_delta   measures FIR and FIR+net, but only in torch fp32;
  - quantize_gather2 measures the int8-against-fp32 SNR ON THE NETWORK
    RESIDUAL, that is, a relative quantity that says nothing about whether
    the network is useful at the output of the whole chain.
Here everything is reduced to ONE metric — residual suppression on the same
hold-out phrases, with the same probe_linear.sup_db formula. The numbers are
directly comparable with probe_linear.md, exam_delta*.md and report.md.

WHAT IS COMPUTED
  1. FIR alone                   — the baseline, a linear filter on the M55
  2. FIR + net, torch fp32       — the ceiling of the design
  3. FIR + net, ONNX fp32        — CONTROL: must match (2).
                                   A divergence = the artifacts are out of sync
                                   (checkpoint and graph from different runs),
                                   and then everything else is meaningless.
  4. FIR + net, ONNX int8 QDQ    — THE ANSWER. This is what actually plays on the board.
  5. FIR + net, ONNX int16       — if --int16 is passed

Plus a per-band breakdown: int8 dies precisely in bands 1-3, and the overall
number masks that.

IMPORTANT ABOUT BALANCING. quantize_gather2 quantizes NOT the original graph
but the balanced one (*_qdq_balanced.onnx): the input is divided by s_in, the
output is multiplied by s_out. So the qdq graph inherits the same convention,
and here the scales are read from the metadata of the balanced graph and applied
back. Without that the numbers will be garbage — plausible-looking garbage.

RUN
  cd C:\\ST\\Projects\\N6\\train
  python eval_chain.py --cand A2_ottpress \\
      --ckpt ../teacher_search/A2_ottpress/ckpt_delta_p360.pt \\
      --phrases 360 --val 3

  # quick run (the first 200 hops of each phrase) to check the mechanics
  python eval_chain.py --cand A2_ottpress --limit-hops 200
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))

CYCLE = (1, 2, 4, 8, 16, 32)


# --------------------------------------------------------------- metrics
def sup_db_bands(y, yhat, skip):
    """Residual suppression PER BAND, dB. The same formula as probe_linear,
    but without averaging over bands — the overall number masks the death of the top ones."""
    y = np.asarray(y, np.float64)[:, skip:]
    d = np.asarray(yhat, np.float64)[:, skip:] - y
    m0 = np.mean(y ** 2, axis=1)
    mp = np.mean(d ** 2, axis=1)
    return 10 * np.log10(np.maximum(m0, 1e-30) / np.maximum(mp, 1e-30))


# ------------------------------------------------------ ONNX run
def onnx_stream(sess, x, dims, s_in=None, s_out=None, limit=0):
    """x[c_in, n] -> y[c_out, nh*T], hop by hop, the states are carried over.

    The voice is written into all V slots, slot 0 is read: along the V axis the
    voices are independent (batch-as-height), so the contents of neighbouring
    slots do not affect slot 0 — but zeros could give degenerate states."""
    c_in, C, c_out, V, T, L = dims
    dil = list(CYCLE) * (L // len(CYCLE))
    n = x.shape[1]
    nh = n // T
    if limit:
        nh = min(nh, limit)
    st = {"state_in_%d" % i: np.zeros((1, C, V, 2 * d), np.float32)
          for i, d in enumerate(dil)}
    outs = []
    for h in range(nh):
        xc = np.asarray(x[:, h * T:(h + 1) * T], np.float32)
        if s_in is not None:
            xc = xc.copy()
            xc[:4] /= s_in.astype(np.float32)[:, None]
        xb = np.repeat(xc[None, :, None, :], V, axis=2)      # [1,c_in,V,T]
        out = sess.run(None, dict(st, x=xb))
        y = out[0][0, :, 0, :]                               # [c_out, T]
        if s_out is not None:
            y = y * s_out.astype(np.float32)[:, None]
        outs.append(y)
        st = {"state_in_%d" % i: out[1 + i] for i in range(L)}
    if not outs:
        raise SystemExit("the phrase is shorter than one hop — nothing to measure")
    return np.concatenate(outs, axis=1)


def load_scales(int8_path, explicit_balanced, quiet=False):
    """s_in/s_out from the balanced graph that the qdq was quantized from."""
    import onnx
    import quantize_gather2 as qg
    cand = explicit_balanced or int8_path.replace(".onnx", "_balanced.onnx")
    if not os.path.exists(cand):
        if not quiet:
            print("  [scales] %s not found — treating the graph as unbalanced"
                  % os.path.basename(cand))
        return None, None
    prev = qg.already_balanced(onnx.load(cand))
    if prev is None:
        if not quiet:
            print("  [scales] %s has no n6_balanced marker — no scales"
                  % os.path.basename(cand))
        return None, None
    s_in, s_out = prev
    if not quiet:
        print("  [scales] from %s: s_in %s | s_out %s"
              % (os.path.basename(cand),
                 np.array2string(s_in, precision=4),
                 np.array2string(s_out, precision=4)))
    return s_in, s_out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="A2_ottpress")
    ap.add_argument("--ckpt", default=None,
                    help="ckpt_delta*.pt (by default the candidate's ckpt_delta.pt)")
    ap.add_argument("--fp32", default=os.path.join(
        HERE, "..", "models", "t1", "n6_gather2.onnx"))
    ap.add_argument("--int8", default=os.path.join(
        HERE, "..", "models", "t1", "n6_gather2_qdq.onnx"))
    ap.add_argument("--int16", default=None,
                    help="e.g. ../models/t1/n6_gather2_q16.onnx")
    ap.add_argument("--balanced", default=None,
                    help="explicit path to *_balanced.onnx (usually derived)")
    ap.add_argument("--phrases", type=int, default=0,
                    help="corpus size (0 = as in teacher_search)")
    ap.add_argument("--val", type=int, default=0, help="phrases in hold-out (0 = 3)")
    ap.add_argument("--limit-hops", type=int, default=0,
                    help="truncate each phrase to N hops (quick run)")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="tolerance on the ONNX fp32 vs torch divergence")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "teacher_search"))
    args = ap.parse_args()

    import torch
    import onnx
    import onnxruntime as ort
    import teacher_search as ts
    import probe_linear as pl
    import quantize_gather2 as qg
    from train_b3 import RF
    from streaming_tcn import StreamingTCN

    if args.phrases: ts.N_PHRASES = args.phrases
    if args.val:     ts.N_VAL = args.val
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.ckpt or os.path.join(args.out, args.cand, "ckpt_delta.pt")
    if not os.path.exists(ckpt):
        sys.exit("no checkpoint %s" % ckpt)
    ck = torch.load(ckpt, map_location="cpu")
    if "fir_W" not in ck:
        sys.exit("the checkpoint has no fir_W — this is not a ckpt_delta*.pt")
    W, taps = np.asarray(ck["fir_W"]), int(ck.get("taps", 65))
    net = StreamingTCN(**ck["form"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    want = [c for c in ts.CANDS if c[0] == args.cand]
    if not want:
        sys.exit("no candidate %s; available: %s"
                 % (args.cand, ", ".join(c[0] for c in ts.CANDS)))
    name, fn, la = want[0]

    print("eval_chain: %s, checkpoint %s, FIR %d taps, corpus %d phrases "
          "(val %d), dev=%s" % (name, os.path.basename(ckpt), taps,
                                ts.N_PHRASES, ts.N_VAL, dev), flush=True)
    print("[corpus] rendering the dry phrases (the same SEED as teacher_search)...",
          flush=True)
    items = ts.dry_corpus()
    xys = [ts.build_xy(it, np.asarray(fn(it["dry"])), la)
           for it in items[:ts.N_VAL]]
    print("[corpus] hold-out: %d phrases, lengths %s subband samples"
          % (len(xys), [x.shape[1] for x, _ in xys]), flush=True)

    # ---- graphs
    sessions = []
    dims = None
    for tag, path in (("ONNX fp32", args.fp32), ("ONNX int8", args.int8),
                      ("ONNX int16", args.int16)):
        if not path:
            continue
        if not os.path.exists(path):
            print("  [%s] no %s — skipped" % (tag, path))
            continue
        m = onnx.load(path)
        d = qg.graph_dims(m)
        if dims is None:
            dims = d
            print("graph: c_in=%d C=%d c_out=%d V=%d T=%d L=%d" % d)
        elif d != dims:
            sys.exit("[%s] graph shapes do not match: %s against %s" % (tag, d, dims))
        if tag == "ONNX fp32":
            s_in = s_out = None      # the original graph is not balanced
        else:
            s_in, s_out = load_scales(path, args.balanced)
        sessions.append((tag, ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]), s_in, s_out))

    if dims is None:
        sys.exit("not a single ONNX graph was found — nothing to measure")
    c_out = dims[2]
    T = dims[4]

    # ---- compute
    rows = {}          # tag -> list of [c_out] arrays per phrase
    tot = {}           # tag -> list of overall numbers per phrase
    for pi, (x, y) in enumerate(xys):
        nh = x.shape[1] // T
        if args.limit_hops:
            nh = min(nh, args.limit_hops)
        n = nh * T
        xc = np.ascontiguousarray(x[:, :n])
        yc = np.asarray(y)[:, :n]
        f = pl.fir_predict(xc, W, taps).astype(np.float64)

        chain = {"FIR alone": f}
        with torch.no_grad():
            chain["FIR + net, torch fp32"] = f + pl.net_out(net, xc, dev)
        for tag, sess, s_in, s_out in sessions:
            chain["FIR + net, %s" % tag] = f + onnx_stream(
                sess, xc, dims, s_in, s_out, limit=nh)

        for k, p in chain.items():
            rows.setdefault(k, []).append(sup_db_bands(yc, p, RF))
            tot.setdefault(k, []).append(pl.sup_db(yc, p, RF))
        print("  phrase %d/%d: %d hops — done"
              % (pi + 1, len(xys), nh), flush=True)

    order = ["FIR alone", "FIR + net, torch fp32", "FIR + net, ONNX fp32",
             "FIR + net, ONNX int8", "FIR + net, ONNX int16"]
    order = [k for k in order if k in tot]
    agg = {k: float(np.mean(tot[k])) for k in order}
    bands = {k: np.mean(np.stack(rows[k]), axis=0) for k in order}

    # ---- CONTROL of artifact sync
    ok_sync = True
    if "FIR + net, ONNX fp32" in agg:
        d = abs(agg["FIR + net, ONNX fp32"] - agg["FIR + net, torch fp32"])
        ok_sync = d <= max(args.tol, 0.05)
        print("\nCONTROL: ONNX fp32 against torch fp32 differ by %.3f dB — %s"
              % (d, "OK" if ok_sync else "<<< ARTIFACTS ARE OUT OF SYNC"))
        if not ok_sync:
            print("  The graph was exported from a DIFFERENT checkpoint than --ckpt.")
            print("  Re-run: python export_gather2.py --ckpt %s"
                  % os.path.relpath(ckpt))
            print("  Until then the int8 numbers mean nothing.")

    # ---- report
    fir = agg["FIR alone"]
    lines = ["# eval_chain — the whole chain, one scale", "",
             "Candidate **%s** (la=%s), checkpoint `%s`, FIR %d taps × 2, "
             "hold-out %d phrases%s."
             % (name, la, os.path.basename(ckpt), taps, len(xys),
                ", truncated to %d hops" % args.limit_hops
                if args.limit_hops else ""),
             "",
             "The metric is residual suppression `10·log10(Σy²/Σ(y−ŷ)²)`, "
             "the same formula as in probe_linear.md and exam_delta*.md.", "",
             "| stage | total, dB | Δ to FIR | " +
             " | ".join("band %d" % b for b in range(c_out)) + " |",
             "|---" * (3 + c_out) + "|"]
    for k in order:
        b = " | ".join("%+.1f" % v for v in bands[k])
        dl = "—" if k == "FIR alone" else "**%+.2f**" % (agg[k] - fir)
        lines.append("| %s | %+.2f | %s | %s |" % (k, agg[k], dl, b))

    verdict = []
    if "FIR + net, ONNX int8" in agg:
        i8, f32 = agg["FIR + net, ONNX int8"], agg["FIR + net, torch fp32"]
        gain, loss = i8 - fir, f32 - i8
        verdict.append("")
        verdict.append("## Verdict")
        verdict.append("")
        verdict.append("- Ceiling of the design (fp32): **%+.2f dB** over the FIR."
                       % (f32 - fir))
        verdict.append("- Actually on the board (int8): **%+.2f dB** over the FIR."
                       % gain)
        verdict.append("- Cost of quantization: **%.2f dB**." % loss)
        verdict.append("")
        if gain <= 0:
            verdict.append("**THE NPU HURTS.** The int8 network works worse than "
                           "a single linear FIR on the M55. In this shape the NPU "
                           "has to be taken out of the chain, not optimized.")
        elif loss > 0.5 * (f32 - fir):
            verdict.append("**Quantization eats more than half of the gain.** "
                           "The design is alive, but the bottleneck is the quant, "
                           "not the architecture: look at per-channel activations, "
                           "int16 and the dead zone of the upper bands.")
        else:
            verdict.append("**The design works.** Quantization takes less than "
                           "half of the gain; the residual SNR was scary for "
                           "nothing — it is relative, and at the output of the "
                           "chain the useful signal dominates.")
        worst = int(np.argmin(bands["FIR + net, ONNX int8"]))
        verdict.append("")
        verdict.append("The worst band on int8 is **%d** (%+.1f dB against %+.1f "
                       "for fp32). The overall number masks it."
                       % (worst, bands["FIR + net, ONNX int8"][worst],
                          bands["FIR + net, torch fp32"][worst]))
    if not ok_sync:
        verdict += ["", "> **WARNING:** the artifact sync control did not "
                        "pass — the ONNX graph was not exported from this "
                        "checkpoint. All the ONNX figures above are invalid."]

    lines += verdict
    txt = "\n".join(lines) + "\n"
    sfx = "_p%d" % ts.N_PHRASES if args.phrases else ""
    p = os.path.join(args.out, "eval_chain_%s%s.md" % (name, sfx))
    open(p, "w", encoding="utf-8", newline="\n").write(txt)
    json.dump({"cand": name, "ckpt": os.path.basename(ckpt), "sync_ok": ok_sync,
               "total": {k: round(v, 3) for k, v in agg.items()},
               "bands": {k: [round(float(v), 2) for v in b]
                         for k, b in bands.items()}},
              open(os.path.join(args.out, name, "eval_chain%s.json" % sfx), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n" + txt)
    print("report: %s" % p, flush=True)


if __name__ == "__main__":
    main()
