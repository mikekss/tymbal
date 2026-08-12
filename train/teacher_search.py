#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teacher_search.py — a systematic search for a LEARNABLE teacher (T1-3, 2 Aug).

Context (CORRECTED on the evening of 2 Aug): the original "B3 is not learnable by
the frozen form, overfitting gives +0.6 dB — that is the limit of expressiveness"
is REFUTED. At A_ITERS=1500 without cosine-to-zero the same B3_512 overfits by
+15.8 dB; the +0.6 was an artifact of stopping before convergence. The real
finding is different: B3 IS MEMORIZED (A ~15 dB) but does NOT GENERALIZE
(B ~0 dB) — a discontinuous decision function with a threshold taken from the
frame peak cannot be inferred from a finite sample.
Here is the candidate matrix; each one gets two exams on the FROZEN form
(C=88/L=12, the T1-2 loss) plus listening tests:

  Exam A (expressiveness): overfit a one-second crop, 500 iterations.
      A failure here = the form CANNOT do it at all -> we do not torture it further.
  Exam B (learnability): training on a shared mini corpus (~2.5 min,
      24 phrases, hold-out 3 phrases) -> gain over the "residual=0" baseline on val.
  Listening tests: for 2 hold-out phrases — dry / teacher / PRED (what the
      network actually learned) — listen to pred vs teacher!

The t axis is uniform across ALL candidates for comparability: t = the depth of
the effect, wet(t) = dry + t·(teacher(dry) − dry); the t curves are as in the
corpus (a base per phrase + LFO). The network is required to use the t channel.

Candidates (teacher_candidates.py + soft derivatives):
  A1_foldsat   — per-band saturation+folder (expectation: it learns);
  A2_ottpress  — OTT dynamics;
  B2_crush     — SRR+mulaw (the network cannot know the hold jitter — partially);
  B3_512       — the canonical q9/g38 (control: known ~0.6 dB);
  B3_256       — the same, window 256/64 (more local);
  B3_soft      — a soft B3: a smooth "staircase" instead of round, a sigmoid gate
                 instead of a hard one (continuity is a chance for the network).

Robustness to "walked away for the whole day": on finishing, each candidate
writes <out>/<name>/done.json; a restart skips the finished ones. The result is
<out>/report.md (a ranking) + the printed table.

RUN (one command, then you can leave):
  cd C:\\ST\\Projects\\N6\\train; python teacher_search.py
Smoke:  N6_SMOKE=1 python3 teacher_search.py   (2 candidates, everything trimmed)
Time estimate on a 1060 Max-Q: ~30-60 min per candidate, the whole matrix is a working day.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))
import pqmf_design as pq                                    # noqa: E402
import skeleton_a as ska                                    # noqa: E402
import make_corpus as mc                                    # noqa: E402
import teacher_candidates as tc                             # noqa: E402
from streaming_tcn import StreamingTCN                      # noqa: E402
from train_b3 import (pqmf_analyze, losses, RES_SCALE,      # noqa: E402
                      C_FROZEN, LAYERS, RF, FS, FR)

SMOKE = os.environ.get("N6_SMOKE") == "1"
OUT = os.path.join(HERE, "..", "teacher_search")
SEED = 20260802

N_PHRASES = 6 if SMOKE else 24
N_VAL = 2 if SMOKE else 3
A_ITERS = 60 if SMOKE else 1500
B_EPOCHS = 2 if SMOKE else 30
CROP, BATCH, PER_ITEM = 1440, 16, 6


# ------------------------------------------------------------ candidates
def b3_soft(x, nfft=512, hop=128, qdb=9.0, gate_db=38.0, knee=3.0):
    """Soft B3: a smooth staircase (r - sin(2πr)/2π) + a sigmoid gate."""
    from scipy import signal
    f, t, Z = signal.stft(x, fs=FS, nperseg=nfft, noverlap=nfft - hop)
    mag, phz = np.abs(Z), np.angle(Z)
    mdb = 20 * np.log10(mag + 1e-9)
    r = mdb / qdb
    mdb_q = qdb * (r - np.sin(2 * np.pi * r) / (2 * np.pi))
    thr = mdb_q.max(axis=0, keepdims=True) - gate_db
    gain = 1.0 / (1.0 + np.exp(-(mdb_q - thr) / knee))
    mag_q = 10 ** (mdb_q / 20.0) * gain
    _, y = signal.istft(mag_q * np.exp(1j * phz), fs=FS, nperseg=nfft,
                        noverlap=nfft - hop)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


# (name, fn, lookahead_la): la = the teacher's non-causality in SUBBAND COLUMNS
# (review 2 Aug: (nfft-hop)/4 @48k -> /4; the target is shifted by la, otherwise
# a causal network fundamentally cannot do it — see t1_notes)
CANDS = [
    ("A1_foldsat", lambda d: tc.teacher_A1_foldsat(d), 0),
    ("A2_ottpress", lambda d: tc.teacher_A2_ottpress(d), 0),
    ("B2_crush", lambda d: tc.teacher_B2_crush(d), 0),
    ("B3_512", lambda d: tc.teacher_B3_residue(d, qdb=9.0, gate_db=38.0), 128),
    ("B3_256", lambda d: tc.teacher_B3_residue(d, nfft=256, hop=64,
                                               qdb=9.0, gate_db=38.0), 64),
    ("B3_soft", b3_soft, 128),
]
if SMOKE:
    CANDS = [CANDS[0], CANDS[3]]


# ------------------------------------------------------------ shared dry corpus
def dry_corpus():
    """A fixed set of phrases + t curves. Shared by all candidates."""
    rng = np.random.default_rng(SEED)
    reps = max(4, -(-N_PHRASES // len(mc.CATS)))   # fix of 2 Aug: *4 truncated it
    cats = (list(mc.CATS) * reps)[:N_PHRASES]     # corpus up to 24 for any --phrases
    items = []
    for cat in cats:
        dur = float(rng.uniform(4.0, 6.0))
        ph = mc.gen_phrase(cat, dur, rng)
        dry = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"],
                               ph["gate"], seed=int(rng.integers(1, 2 ** 31)))
        pk = np.max(np.abs(dry)) + 1e-9
        if pk > 0.98:
            dry = dry * (0.98 / pk)
        nfr = len(ph["f0"])
        t0 = rng.uniform(0.15, 1.0)
        tm = np.arange(nfr) / FR
        tcv = np.clip(t0 + 0.08 * np.sin(
            2 * np.pi * rng.uniform(0.05, 0.4) * tm + rng.uniform(0, 6.28)),
            0.0, 1.0)
        items.append(dict(cat=cat, ph=ph, dry=dry, t=tcv))
    return items


def build_xy(item, wet_full, la=0):
    """x[8,n12], y[4,n12] (residual*RES_SCALE) with wet(t)=dry+t·(wet_full−dry)."""
    dry, ph, tcv = item["dry"], item["ph"], item["t"]
    t48 = np.repeat(tcv, FS // FR)[:len(dry)]
    wet = dry + t48 * (np.asarray(wet_full)[:len(dry)] - dry)
    sd, sw = pqmf_analyze(dry), pqmf_analyze(wet)
    n12 = sd.shape[1]

    def up(v):
        u = np.repeat(np.asarray(v, np.float32), FS // 4 // FR)
        return (u[:n12] if len(u) >= n12
                else np.pad(u, (0, n12 - len(u)), mode="edge"))
    x = np.concatenate([sd, up(ph["amp"] * ph["gate"])[None],
                        up(ph["tA"])[None], up(ph["tB"])[None],
                        up(tcv)[None]], axis=0)
    y = (sw - sd) * RES_SCALE
    if la > 0:                                  # shift of the non-causal target
        y = np.pad(y, ((0, 0), (la, 0)))[:, :y.shape[1]]
    return x.astype(np.float32), y.astype(np.float32)


class Crops(torch.utils.data.Dataset):
    def __init__(self, xys, seed):
        self.xys = [xy for xy in xys if xy[0].shape[1] >= CROP]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.xys) * PER_ITEM

    def __getitem__(self, k):
        x, y = self.xys[k % len(self.xys)]
        o = int(self.rng.integers(0, x.shape[1] - CROP + 1))
        return (torch.from_numpy(x[:, o:o + CROP].copy()),
                torch.from_numpy(y[:, o:o + CROP].copy()))


def fresh_net(dev):
    net = StreamingTCN(c_in=8, c=C_FROZEN, c_out=4, layers=LAYERS).to(dev)
    torch.nn.init.zeros_(net.tail.weight)
    return net


def gain_db(base, val):
    return 20 * np.log10(max(base, 1e-9) / max(val, 1e-9))


def screen_A(xy, dev):
    """Overfit a one-second crop."""
    x = torch.from_numpy(xy[0][None, :, 6000:18000]).to(dev)
    y = torch.from_numpy(xy[1][None, :, 6000:18000]).to(dev)
    net = fresh_net(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)  # review: WITHOUT cosine-to-zero
    with torch.no_grad():
        _, _, _, base = losses(net, x, y, dev, with_base=True)
    best = float("inf")
    for _ in range(A_ITERS):
        opt.zero_grad()
        loss, *_ = losses(net, x, y, dev)
        loss.backward(); opt.step()
        best = min(best, loss.item())
    return gain_db(base, best)


def screen_B(xys_tr, xys_va, dev):
    """Training on the mini corpus, gain on hold-out; also returns the network."""
    tr = torch.utils.data.DataLoader(Crops(xys_tr, 1), batch_size=BATCH,
                                     shuffle=True)
    va = torch.utils.data.DataLoader(Crops(xys_va, 2), batch_size=BATCH)
    net = fresh_net(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=B_EPOCHS)
    best, best_sd = float("inf"), None
    vb_last = 1.0
    for ep in range(B_EPOCHS):
        net.train()
        for x, y in tr:
            opt.zero_grad()
            loss, *_ = losses(net, x, y, dev)
            loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            vl, vb = [], []
            for x, y in va:
                loss, _, _, base = losses(net, x, y, dev, with_base=True)
                vl.append(loss.item()); vb.append(base)
        vl, vb_last = float(np.mean(vl)), float(np.mean(vb))
        if vl < best:
            best = vl
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd)
    return gain_db(vb_last, best), net


@torch.no_grad()
def res_db(net, xys_va, dev):
    """Honest dB: 10*log10(MSE_residual(0)/MSE_residual(pred)) on val,
    the first RF columns excluded (review 2 Aug — instead of a ratio of losses)."""
    mp, m0 = 0.0, 0.0
    for x, y in xys_va:
        xt = torch.from_numpy(x[None]).to(dev)
        pred = net(xt, *[st.to(dev) for st in net.zero_states(1)])[0][0]
        d = (pred.cpu().numpy() - y)[:, RF:]
        mp += float(np.mean(d ** 2))
        m0 += float(np.mean(y[:, RF:] ** 2))
    return 10 * np.log10(max(m0, 1e-12) / max(mp, 1e-12))


_SYN = None
def synth48(sub):
    global _SYN
    if _SYN is None:
        _, _SYN = pq.filterbank(pq.prototype(0.066603, beta=8.80))
    return pq.synthesize(np.asarray(sub, np.float64), _SYN)


def auditions(cdir, items_va, wets_va, xys_va, net, dev):
    import soundfile as sf
    from scipy import signal as sg
    for item, wet_full, xy in zip(items_va, wets_va, xys_va):
        dry, tcv = item["dry"], item["t"]
        t48 = np.repeat(tcv, FS // FR)[:len(dry)]
        wet = dry + t48 * (np.asarray(wet_full)[:len(dry)] - dry)
        with torch.no_grad():
            res = net(torch.from_numpy(xy[0][None]).to(dev),
                      *[s.to(dev) for s in net.zero_states(1)])[0][0]
        res = res.cpu().numpy() / RES_SCALE
        sd = xy[0][:4]
        pred = synth48(sd + res)
        rt = synth48(sd)
        n = min(len(rt), len(dry), FS)
        lag = int(np.argmax(sg.correlate(rt[:n + 512], dry[:n], mode="valid")))
        pred = pred[lag:lag + len(dry)]
        if len(pred) < len(dry):
            pred = np.pad(pred, (0, len(dry) - len(pred)))
        for nm, sig in (("dry", dry), ("teacher", wet), ("pred", pred)):
            sf.write(os.path.join(cdir, f"{item['cat']}_{nm}.wav"),
                     np.asarray(sig, np.float32), FS, subtype="FLOAT")


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    print(f"teacher_search: {len(CANDS)} candidates, {N_PHRASES} phrases, "
          f"exams A={A_ITERS} iters / B={B_EPOCHS} epochs, dev={dev}", flush=True)

    print("[corpus] rendering the dry phrases...", flush=True)
    items = dry_corpus()
    items_tr, items_va = items[N_VAL:], items[:N_VAL]

    results = []
    for name, fn, la in CANDS:
        cdir = os.path.join(OUT, name)
        marker = os.path.join(cdir, "done.json")
        if os.path.exists(marker):
            results.append(json.load(open(marker, encoding="utf-8")))
            print(f"[{name}] already done — skipped", flush=True)
            continue
        os.makedirs(cdir, exist_ok=True)
        t0 = time.time()
        try:
            print(f"[{name}] teacher on {len(items)} phrases...", flush=True)
            wets = [np.asarray(fn(it["dry"])) for it in items]
            xys = [build_xy(it, w, la) for it, w in zip(items, wets)]
            xys_tr, xys_va = xys[N_VAL:], xys[:N_VAL]
            gA = screen_A(xys_tr[0], dev)
            print(f"[{name}] exam A (overfit): {gA:+.1f} dB", flush=True)
            gB, net = screen_B(xys_tr, xys_va, dev)
            rdb = res_db(net, xys_va, dev)
            print(f"[{name}] exam B: loss {gB:+.1f} dB, "
                  f"residual suppression {rdb:+.1f} dB", flush=True)
            auditions(cdir, items_va, wets[:N_VAL], xys_va, net, dev)
            torch.save({"model": net.state_dict(), "res_scale": RES_SCALE,
                        "form": dict(c_in=8, c=C_FROZEN, c_out=4,
                                     layers=LAYERS)},
                       os.path.join(cdir, "ckpt.pt"))
            r = dict(name=name, gainA_db=round(gA, 2), gainB_db=round(gB, 2),
                     resB_db=round(rdb, 2), la=la,
                     minutes=round((time.time() - t0) / 60, 1), error=None)
        except Exception as e:                              # do not bring the matrix down
            r = dict(name=name, gainA_db=None, gainB_db=None, resB_db=None,
                     la=la, minutes=round((time.time() - t0) / 60, 1),
                     error=str(e))
            print(f"[{name}] ERROR: {e}", flush=True)
        json.dump(r, open(marker, "w", encoding="utf-8"), ensure_ascii=False)
        results.append(r)

    results.sort(key=lambda r: -(r.get("resB_db") if r.get("resB_db")
                                 is not None else -99))
    lines = ["# teacher_search — report (T1-3, metrics from the 2 Aug review)", "",
             "| candidate | la | overfit A, dB | val loss, dB | val residual, dB | min |",
             "|---|---|---|---|---|---|"]
    for r in results:
        e = "" if not r["error"] else f" ERROR: {r['error'][:60]}"
        lines.append(f"| {r['name']} | {r.get('la')} | {r['gainA_db']} | "
                     f"{r['gainB_db']} | {r.get('resB_db')} | {r['minutes']}{e} |")
    lines += ["", "How to read it: A — can the form do it at all (its ceiling); "
              "B — does it learn with generalization. Listen to pred vs teacher for B leaders. "
              "The t axis is depth (wet/dry) for all, so candidates are comparable."]
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8",
         newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nreport: {os.path.join(OUT, 'report.md')}", flush=True)


if __name__ == "__main__":
    main()
