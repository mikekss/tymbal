#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_linear.py — "did the net learn the effect or a static filter?"
(review 2 Aug)

teacher_search measures HOW MUCH the net suppresses the residual, but not BY
WHAT MEANS. An optimal linear filter also suppresses the residual — and if it
suppresses just as much, then the NPU is computing an equalizer and the whole
enterprise is pointless. Three probes on a candidate's finished checkpoint:

  1. HOMOGENEITY  f(a·x) vs a·f(x). For a linear system the deviation goes to
     −inf dB. Computed FOR BOTH THE NET AND THE TEACHER: the comparison must be
     against the teacher — if the teacher itself is nearly linear, there is
     nothing to ask of the net.
  2. LINEAR BASELINE ("exam 0"): a least-squares fit of a causal 4→4 FIR bank
     of K taps modulated by the t axis (W0 + t·W1) on the train phrases of the
     same corpus; residual suppression on the same val phrases, with the same
     formula as the net's res_db. FIR ≥ net ⇒ the net learned nothing beyond a
     frequency response.
  3. AXIS ZERO: at t=0 the target is exactly zero (wet = dry). Whatever the net
     puts out — if it is a lot, it learned a constant coloration, not a
     controllable effect.

The corpus, the split and the shift of the non-causal target are taken FROM
teacher_search — the figures are directly comparable with report.md.

Run:
  cd C:\\ST\\Projects\\N6\\train
  python probe_linear.py --cand A2_ottpress
  python probe_linear.py --cand all --taps 65
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))

import teacher_search as ts                                  # noqa: E402
from train_b3 import RES_SCALE, RF, FS, FR                   # noqa: E402
from streaming_tcn import StreamingTCN                       # noqa: E402

GAINS = (0.25, 0.5, 2.0)          # multipliers for the homogeneity probe


# ------------------------------------------------------------------ helpers
def sup_db(y, yhat, skip=RF):
    """Residual suppression, dB: 10·log10(E[y²] / E[(y−ŷ)²]). The same formula
    as teacher_search.res_db — the numbers are comparable with report.md."""
    d = (np.asarray(yhat, np.float64) - np.asarray(y, np.float64))[:, skip:]
    m0 = float(np.mean(np.asarray(y, np.float64)[:, skip:] ** 2))
    mp = float(np.mean(d ** 2))
    return 10 * np.log10(max(m0, 1e-30) / max(mp, 1e-30))


@torch.no_grad()
def net_out(net, x, dev):
    """x[8,n] -> residual·RES_SCALE [4,n] (in one piece, states zeroed)."""
    xt = torch.from_numpy(np.ascontiguousarray(x, np.float32)[None]).to(dev)
    y = net(xt, *[s.to(dev) for s in net.zero_states(1)])[0][0]
    return y.detach().cpu().numpy()


def load_net(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu")
    net = StreamingTCN(**ck["form"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, float(ck.get("res_scale", RES_SCALE))


# ---------------------------------------------------- probe 1: homogeneity
def homogeneity_net(net, xys_va, dev):
    """dB of net nonlinearity: 10·log10( E[(f(a·x) − a·f(x))²] / E[(a·f(x))²] ).
    ONLY the 4 band channels are scaled; the controls (amp/tA/tB/t) stay as
    they are: this is exactly "the same gesture, played softer/louder"."""
    out = {}
    for a in GAINS:
        num = den = 0.0
        for x, _ in xys_va:
            r0 = net_out(net, x, dev)
            xa = x.copy()
            xa[:4] *= a
            r1 = net_out(net, xa, dev)
            num += float(np.mean((r1 - a * r0)[:, RF:] ** 2))
            den += float(np.mean((a * r0)[:, RF:] ** 2))
        out[a] = 10 * np.log10(max(num, 1e-30) / max(den, 1e-30))
    return out


def homogeneity_teacher(fn, items_va):
    """The same for the teacher itself, on the 48k waveform
    (residual = wet_full − dry)."""
    out = {}
    for a in GAINS:
        num = den = 0.0
        for it in items_va:
            d = np.asarray(it["dry"], np.float64)
            r0 = np.asarray(fn(d), np.float64)[:len(d)] - d
            r1 = np.asarray(fn(a * d), np.float64)[:len(d)] - a * d
            num += float(np.mean((r1 - a * r0) ** 2))
            den += float(np.mean((a * r0) ** 2))
        out[a] = 10 * np.log10(max(num, 1e-30) / max(den, 1e-30))
    return out


# ------------------------------------------------ probe 2: linear baseline
def fir_features(x, taps):
    """x[8,n] -> Phi[n, 2·4·taps]: causal taps of the four bands and the same × t.
    Model: res[b,n] = Σ_{b',k} (W0 + t[n]·W1)[b,b',k] · x[b', n−k]."""
    n = x.shape[1]
    cols = np.empty((4 * taps, n), np.float32)
    for b in range(4):
        row = np.asarray(x[b], np.float32)
        for k in range(taps):
            c = cols[b * taps + k]
            if k:
                c[:k] = 0.0
                c[k:] = row[:n - k]
            else:
                c[:] = row
    P = cols.T                                   # [n, 4·taps]
    t = np.asarray(x[7], np.float32)[:, None]    # t axis channel (see build_xy)
    return np.concatenate([P, P * t], axis=1)    # [n, 8·taps]


def fit_fir(xys_tr, taps, ridge=1e-6):
    """Least squares via the normal equations (memory O(F²), not O(nF))."""
    F = 8 * taps
    G = np.zeros((F, F), np.float64)
    c = np.zeros((F, 4), np.float64)
    for x, y in xys_tr:
        Phi = fir_features(x, taps)[RF:].astype(np.float64)
        Y = np.asarray(y, np.float64).T[RF:]
        G += Phi.T @ Phi
        c += Phi.T @ Y
        del Phi
    G += ridge * (np.trace(G) / F + 1e-30) * np.eye(F)
    return np.linalg.solve(G, c)                 # [F, 4]


def fir_predict(x, W, taps):
    return (fir_features(x, taps) @ W).T          # [4, n]


# ---------------------------------------------------------------- report
def probe(name, fn, la, ckpt, xys_tr, xys_va, items_va, taps, dev):
    net, _ = load_net(ckpt, dev)

    # what teacher_search measures — reproduced so the numbers line up
    net_sup = float(np.mean([sup_db(y, net_out(net, x, dev))
                             for x, y in xys_va]))

    W = fit_fir(xys_tr, taps)
    fir_sup = float(np.mean([sup_db(y, fir_predict(x, W, taps))
                             for x, y in xys_va]))

    hn = homogeneity_net(net, xys_va, dev)
    ht = homogeneity_teacher(fn, items_va)

    # probe 3: axis to zero
    z_rel = []
    for x, y in xys_va:
        x0 = x.copy()
        x0[7] = 0.0
        r0 = net_out(net, x0, dev)[:, RF:]
        z_rel.append(np.sqrt(np.mean(r0 ** 2)) /
                     (np.sqrt(np.mean(np.asarray(y)[:, RF:] ** 2)) + 1e-30))
    z_rel = float(np.mean(z_rel))

    print(f"\n=== {name} (la={la}, FIR {taps} taps × 2) ===")
    print(f"  residual suppression: net {net_sup:+.2f} dB | "
          f"linear FIR {fir_sup:+.2f} dB | "
          f"NET−FIR {net_sup - fir_sup:+.2f} dB")
    print("  nonlinearity (higher = more nonlinear; linear → −inf):")
    for a in GAINS:
        print(f"    ×{a:<5}  net {hn[a]:+7.2f} dB   teacher {ht[a]:+7.2f} dB")
    print(f"  output at t=0 (target exactly 0): {20 * np.log10(max(z_rel, 1e-12)):+.1f} dB "
          f"of a typical residual")
    return dict(name=name, la=la, net_sup=net_sup, fir_sup=fir_sup,
                delta=net_sup - fir_sup,
                nl_net=hn[2.0], nl_teacher=ht[2.0], zero_rel=z_rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="A2_ottpress",
                    help="teacher_search candidate name or all")
    ap.add_argument("--taps", type=int, default=65,
                    help="FIR taps per band pair (65 ≈ 5.4 ms @12k)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "teacher_search"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    want = [c for c in ts.CANDS
            if args.cand == "all" or c[0] == args.cand]
    if not want:
        sys.exit(f"no candidate {args.cand}; available: "
                 f"{', '.join(c[0] for c in ts.CANDS)}")

    print(f"probe_linear: {len(want)} candidate(s), FIR {args.taps} taps, "
          f"dev={dev}", flush=True)
    print("[corpus] rendering dry phrases (the same SEED as teacher_search)...",
          flush=True)
    items = ts.dry_corpus()
    items_tr, items_va = items[ts.N_VAL:], items[:ts.N_VAL]

    rows = []
    for name, fn, la in want:
        ckpt = os.path.join(args.out, name, "ckpt.pt")
        if not os.path.exists(ckpt):
            print(f"[{name}] no {ckpt} — skipped", flush=True)
            continue
        print(f"[{name}] teacher on {len(items)} phrases...", flush=True)
        wets = [np.asarray(fn(it["dry"])) for it in items]
        xys = [ts.build_xy(it, w, la) for it, w in zip(items, wets)]
        rows.append(probe(name, fn, la, ckpt, xys[ts.N_VAL:], xys[:ts.N_VAL],
                          items_va, args.taps, dev))

    if rows:
        rows.sort(key=lambda r: -r["delta"])
        lines = ["# probe_linear — the net against a linear filter", "",
                 "| candidate | net, dB | FIR, dB | net−FIR | net nonlin. ×2 |"
                 " teacher nonlin. ×2 | output at t=0 |",
                 "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['name']} | {r['net_sup']:+.2f} | {r['fir_sup']:+.2f} | "
                f"**{r['delta']:+.2f}** | {r['nl_net']:+.1f} | "
                f"{r['nl_teacher']:+.1f} | "
                f"{20 * np.log10(max(r['zero_rel'], 1e-12)):+.1f} dB |")
        lines += ["",
                  "How to read: **net−FIR** is the only column the NPU is "
                  "needed for. ≤0 ⇒ what was learned is reproducible by a "
                  "static filter. Compare \"net nonlin.\" with \"teacher "
                  "nonlin.\": if the teacher is −30 dB, it is nearly linear."]
        p = os.path.join(args.out, "probe_linear.md")
        open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        print(f"\nreport: {p}", flush=True)


if __name__ == "__main__":
    main()
