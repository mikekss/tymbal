#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exam_delta.py — "FIR as a zeroth layer": train the net ONLY on the nonlinear
residue.

Context (probe_linear.md, 2 Aug): a linear FIR bank of 520 coefficients beats
the trained 281K-parameter network on ALL six candidates (net−FIR from −0.94 to
−3.73 dB), and the trained networks themselves measure as ~90% linear
(nonlinearity −8.5…−12.8 dB). Hypothesis: the linear solution is an attractor
of the optimization (the loss is 2/3 waveform L1 + zero-init tail => the first
gradient step points exactly toward least squares), and the net gets stuck
in it.

The check: make the FIR a FROZEN ZEROTH LAYER.
    pred_total = FIR(x) + net(x)
The linear part is now free, and the net can only score on what the FIR cannot
do. The loss is the same as in train_b3 (the spectral term is computed on
pred_total, so its meaning is preserved).

Three numbers per candidate:
  FIR         — residual suppression by the linear filter alone (baseline);
  FIR+net     — the same for the sum;
  DELTA       — the difference. That is everything the NPU is needed for.
plus exam A' — overfit of a single crop on the residue (y − FIR): the ceiling
of RESIDUE expressiveness. It separates two diagnoses:
  A' large, delta ≈ 0   -> the residue is expressible but does not generalize
                           (wrong domain);
  A' ≈ 0                -> the residue is not expressible at all (definitely
                           the wrong domain);
  delta > 0             -> we have a design: FIR on the M55 + net on the NPU.

The val crops are FIXED (deterministic offsets), the metric is sup_db on FULL
val phrases, as in probe_linear.md: the numbers are directly comparable.

Run:
  cd C:\\ST\\Projects\\N6\\train
  python exam_delta.py --cand all              # ~5-7 min per candidate
  python exam_delta.py --cand A2_ottpress --epochs 200
  python exam_delta.py --cand all --no-fir     # control: same training, no FIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))

import teacher_search as ts                                  # noqa: E402
import probe_linear as pl                                    # noqa: E402
from train_b3 import RES_SCALE, RF, stft_l1, C_FROZEN, LAYERS  # noqa: E402
from streaming_tcn import StreamingTCN                       # noqa: E402

CROP, BATCH, PER_ITEM = 1440, 16, 6


# ------------------------------------------------------------------ data
class Trip(torch.utils.data.Dataset):
    """Crops of triples (x, y, f). For train the offsets are random, for val
    DETERMINISTIC (review 2 Aug: a floating val gives up to +2 dB out of air)."""

    def __init__(self, items, seed, fixed):
        self.items = [it for it in items if it[0].shape[1] >= CROP]
        self.rng = np.random.default_rng(seed)
        self.fixed = fixed

    def __len__(self):
        return len(self.items) * PER_ITEM

    def __getitem__(self, k):
        x, y, f = self.items[k % len(self.items)]
        span = x.shape[1] - CROP
        if self.fixed:
            j = k // len(self.items)
            o = int(round(span * j / max(1, PER_ITEM - 1)))
        else:
            o = int(self.rng.integers(0, span + 1))
        s = slice(o, o + CROP)
        return (torch.from_numpy(x[:, s].copy()),
                torch.from_numpy(y[:, s].copy()),
                torch.from_numpy(f[:, s].copy()))


def fresh_net(dev):
    net = StreamingTCN(c_in=8, c=C_FROZEN, c_out=4, layers=LAYERS).to(dev)
    torch.nn.init.zeros_(net.tail.weight)
    return net


def loss_total(net, x, y, f, dev):
    """The same loss as train_b3.losses, but on pred_total = FIR + net."""
    x, y, f = x.to(dev), y.to(dev), f.to(dev)
    pred = net(x, *[s.to(dev) for s in net.zero_states(x.shape[0])])[0] + f
    m = torch.ones_like(y)
    m[:, :, :RF] = 0.0
    l1 = ((pred - y).abs() * m).sum() / m.sum()
    sp = stft_l1((x[:, :4] + pred / RES_SCALE)[:, :, RF:],
                 (x[:, :4] + y / RES_SCALE)[:, :, RF:])
    return 0.2 * l1 + sp


@torch.no_grad()
def sup_full(net, items, dev, with_fir=True):
    """Residual suppression on FULL phrases: FIR+net vs target."""
    out = []
    for x, y, f in items:
        p = pl.net_out(net, x, dev) + (f if with_fir else 0.0)
        out.append(pl.sup_db(y, p))
    return float(np.mean(out))


# ------------------------------------------------------------------ exams
def exam_A_delta(item, dev, iters, lr):
    """Overfit of a single one-second crop on the RESIDUE (y − FIR)."""
    x, y, f = item
    sl = slice(6000, 18000)
    xt = torch.from_numpy(x[None, :, sl]).to(dev)
    d = torch.from_numpy((y - f)[None, :, sl]).to(dev)
    net = fresh_net(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    best = float("inf")
    base = float(torch.mean(d[:, :, RF:] ** 2))
    for _ in range(iters):
        opt.zero_grad()
        pred = net(xt, *[s.to(dev) for s in net.zero_states(1)])[0]
        mse = torch.mean((pred - d)[:, :, RF:] ** 2)
        mse.backward(); opt.step()
        best = min(best, float(mse))
    return 10 * np.log10(max(base, 1e-30) / max(best, 1e-30))


def exam_B_delta(tr_items, va_items, dev, epochs, lr, with_fir):
    tr = torch.utils.data.DataLoader(Trip(tr_items, 1, False),
                                     batch_size=BATCH, shuffle=True)
    va = torch.utils.data.DataLoader(Trip(va_items, 2, True), batch_size=BATCH)
    net = fresh_net(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                     eta_min=lr * 0.05)
    best, best_sd = -1e9, None
    for ep in range(epochs):
        net.train()
        for x, y, f in tr:
            if not with_fir:
                f = torch.zeros_like(f)
            opt.zero_grad()
            loss_total(net, x, y, f, dev).backward()
            opt.step()
        sch.step()
        net.eval()
        s = sup_full(net, va_items, dev, with_fir)
        if s > best:
            best = s
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep+1}/{epochs}: suppression {s:+.2f} dB "
                  f"(best {best:+.2f})", flush=True)
    net.load_state_dict(best_sd)
    return best, net


# ------------------------------------------------------------------ run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="all")
    ap.add_argument("--taps", type=int, default=65)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--a-iters", type=int, default=1500)
    ap.add_argument("--a-lr", type=float, default=2e-3)
    ap.add_argument("--phrases", type=int, default=0,
                    help="corpus size in phrases (0 = as in teacher_search, 24). "
                         "The gap between A' and delta is about DATA: 21 training "
                         "phrases is ~100 s of audio for 281K parameters")
    ap.add_argument("--val", type=int, default=0, help="phrases in hold-out (0 = 3)")
    ap.add_argument("--no-fir", action="store_true",
                    help="control: the same training, but WITHOUT the zeroth layer")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "teacher_search"))
    args = ap.parse_args()
    with_fir = not args.no_fir

    if args.phrases: ts.N_PHRASES = args.phrases
    if args.val:     ts.N_VAL = args.val
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    want = [c for c in ts.CANDS if args.cand == "all" or c[0] == args.cand]
    if not want:
        sys.exit(f"no candidate {args.cand}; available: "
                 f"{', '.join(c[0] for c in ts.CANDS)}")

    tag = "" if with_fir else "_nofir"
    print(f"exam_delta{tag}: {len(want)} candidate(s), corpus {ts.N_PHRASES} phrases "
          f"(val {ts.N_VAL}), FIR {args.taps} taps, "
          f"B: {args.epochs} ep @ lr {args.lr}, A': {args.a_iters} it, dev={dev}",
          flush=True)
    print("[corpus] rendering dry phrases (the same SEED as teacher_search)...",
          flush=True)
    items = ts.dry_corpus()
    rows = []

    for name, fn, la in want:
        t0 = time.time()
        print(f"\n[{name}] teacher on {len(items)} phrases...", flush=True)
        # one pair at a time: a list of wets for 300+ phrases is hundreds of extra MB
        xys = [ts.build_xy(it, np.asarray(fn(it["dry"])), la) for it in items]

        print(f"[{name}] least-squares FIR fit...", flush=True)
        W = pl.fit_fir(xys[ts.N_VAL:], args.taps)
        trip = [(x, y, pl.fir_predict(x, W, args.taps).astype(np.float32))
                for x, y in xys]
        tr_items, va_items = trip[ts.N_VAL:], trip[:ts.N_VAL]
        fir_sup = float(np.mean([pl.sup_db(y, f) for _, y, f in va_items]))
        print(f"[{name}] FIR alone: {fir_sup:+.2f} dB", flush=True)

        gA = exam_A_delta(tr_items[0], dev, args.a_iters, args.a_lr)
        print(f"[{name}] exam A' (overfit of the RESIDUE): {gA:+.2f} dB", flush=True)

        tot, net = exam_B_delta(tr_items, va_items, dev, args.epochs, args.lr,
                                with_fir)
        delta = tot - (fir_sup if with_fir else 0.0)
        print(f"[{name}] TOTAL: FIR {fir_sup:+.2f} | FIR+net {tot:+.2f} | "
              f"DELTA {delta:+.2f} dB  [{(time.time()-t0)/60:.1f} min]",
              flush=True)

        cdir = os.path.join(args.out, name)
        os.makedirs(cdir, exist_ok=True)
        torch.save({"model": net.state_dict(), "res_scale": RES_SCALE,
                    "fir_W": W, "taps": args.taps, "with_fir": with_fir,
                    "form": dict(c_in=8, c=C_FROZEN, c_out=4, layers=LAYERS)},
                   os.path.join(cdir, "ckpt_delta%s%s.pt" % (
                       tag, ("_p%d" % ts.N_PHRASES) if args.phrases else "")))
        r = dict(name=name, la=la, fir=round(fir_sup, 2), total=round(tot, 2),
                 delta=round(delta, 2), aprime=round(gA, 2),
                 minutes=round((time.time() - t0) / 60, 1))
        json.dump(r, open(os.path.join(cdir, f"delta{tag}.json"), "w",
                          encoding="utf-8"), ensure_ascii=False)
        rows.append(r)

    rows.sort(key=lambda r: -r["delta"])
    head = ("# exam_delta — FIR as a zeroth layer" if with_fir
            else "# exam_delta (CONTROL, without FIR)")
    lines = [head, "",
             f"B: {args.epochs} ep @ lr {args.lr}; A': {args.a_iters} it @ "
             f"{args.a_lr}; FIR {args.taps} taps × 2 (modulated by the t axis).", "",
             "| candidate | FIR alone, dB | FIR+net, dB | DELTA | "
             "A' residue overfit | min |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['fir']:+.2f} | {r['total']:+.2f} | "
                     f"**{r['delta']:+.2f}** | {r['aprime']:+.2f} | "
                     f"{r['minutes']} |")
    lines += ["",
              "How to read: **DELTA** is everything the NPU is needed for. "
              "A' large with delta ≈ 0 => the residue is expressible but does not "
              "generalize (wrong domain). A' ≈ 0 => the residue is not expressible "
              "at all. Delta > 0 => the design: FIR (520 coeff.) on M55 + net on NPU."]
    sfx = f"_p{ts.N_PHRASES}" if args.phrases else ""
    p = os.path.join(args.out, f"exam_delta{tag}{sfx}.md")
    open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nreport: {p}", flush=True)


if __name__ == "__main__":
    main()
