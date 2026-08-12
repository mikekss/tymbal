#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_b3.py — training the refiner on the B3 corpus (T1; the form is FROZEN as
of 2 Aug).

Form (decision_log, frozen 2 Aug): wiring (b) bands-as-channels,
**C=88, V=2, T=48, L=12** (d44 of the M0 matrix: 2866 us/hop, 10% headroom).
Net: streaming_tcn.StreamingTCN(c_in=8, c=88, c_out=4, layers=12),
head/tail without bias (T0 canon). Input (8 channels, 12 kHz grid):
  0..3  subbands of the DRY skeleton (PQMF analysis, bank fc=0.066603/β=8.80);
  4     ag = amp*gate (effective amplitude);
  5..6  tA, tB;
  7     t — the D-16 teacher macro axis (in the runtime = drive/antenna).
The 250 Hz controls are stretched by repetition ×48 (constant within a hop).
Target: residual = wet_sub − dry_sub (in the runtime the NPU adds it to the
skeleton, pipeline §8.1). T1-2 (2 Aug, after the first "close to zero"
training): the net predicts residual×RES_SCALE=32 (the target lands in the
working range of the gradients; the inverse scale is stored in the ckpt and
removed at inference/export — for int8 io this is also a better dynamic range).
Loss: 0.2·L1(scaled) with an RF=252 mask + log-STFT per band (256/64) on
(dry+res) vs wet — the spectrum dominates.

Training runs on a GTX 1060 (torch 2.1–2.4+cu118, fp32; API-compatible);
--smoke runs on CPU. Determinism: all seeds are fixed.

Run (production, on the training machine):
  python3 make_corpus_b3.py --minutes 30 --seed 1 --out ../corpus_b3
  python3 train_b3.py --corpus ../corpus_b3 --epochs 60
Smoke:  python3 train_b3.py --smoke   (builds a mini corpus itself, 3 epochs, CPU)
Output: models/b3/ckpt_best.pt (+ metrics), listening tests — audition_b3.py.
Convergence criterion: the val loss is noticeably below the "residual=0"
baseline (which is printed).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))
import pqmf_design as pq                                    # noqa: E402
from streaming_tcn import StreamingTCN                      # noqa: E402

FS, FR, K = 48000, 250, 4
C_FROZEN, LAYERS, RF = 88, 12, 252                          # frozen 2 Aug
RES_SCALE = 32.0                       # T1-2: the net learns residual*32
FC_OPT, BETA_OPT = 0.066603, 8.80                           # as in skeleton_a

_ANA = None
def pqmf_analyze(x):
    global _ANA
    if _ANA is None:
        ana, _ = pq.filterbank(pq.prototype(FC_OPT, beta=BETA_OPT))
        _ANA = ana
    return pq.analyze(np.asarray(x, np.float64), _ANA).astype(np.float32)


def load_pair(base):
    """base without extension -> (x[8,n12], y[4,n12]) float32."""
    from scipy.io import wavfile
    _, dry = wavfile.read(base + ".wav")
    _, wet = wavfile.read(base + "_wet.wav")
    z = np.load(base + ".npz", allow_pickle=True)
    sd = pqmf_analyze(dry)
    sw = pqmf_analyze(wet)
    n12 = sd.shape[1]

    def up(v):                                              # 250 Hz -> 12 kHz
        u = np.repeat(np.asarray(v, np.float32), FS // K // FR)
        return (u[:n12] if len(u) >= n12
                else np.pad(u, (0, n12 - len(u)), mode="edge"))
    ag = up(z["amp"] * z["gate"])
    x = np.concatenate([sd, ag[None], up(z["tA"])[None], up(z["tB"])[None],
                        up(z["t250"])[None]], axis=0)
    y = (sw - sd) * np.float32(RES_SCALE)
    # review 2 Aug: a non-causal teacher (lookahead of la subband columns) ->
    # the target is delayed by la; in the runtime the skeleton is delayed by
    # 4*la samples @48k before the sum with the residual (+la/12 ms of path
    # latency)
    la = int(z["teacher_la"]) if "teacher_la" in z.files else 0
    if la > 0:
        y = np.pad(y, ((0, 0), (la, 0)))[:, :y.shape[1]]
    return x, y


class CorpusB3(torch.utils.data.Dataset):
    """Fixed-length crops from a preloaded corpus."""
    def __init__(self, bases, crop, per_item, seed):
        self.pairs = [load_pair(b) for b in bases]
        self.crop, self.per_item = crop, per_item
        self.rng = np.random.default_rng(seed)
        self.index = []
        for i, (x, _) in enumerate(self.pairs):
            n = x.shape[1]
            if n >= crop:
                self.index += [i] * per_item

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        x, y = self.pairs[i]
        o = int(self.rng.integers(0, x.shape[1] - self.crop + 1))
        return (torch.from_numpy(x[:, o:o + self.crop].copy()),
                torch.from_numpy(y[:, o:o + self.crop].copy()))


def stft_l1(a, b, nfft=256, hop=64):
    """log-magnitude L1 per band: a,b [B,4,T]."""
    B, Kb, T = a.shape
    win = torch.hann_window(nfft, device=a.device)
    A = torch.stft(a.reshape(B * Kb, T), nfft, hop, window=win,
                   return_complex=True).abs()
    Bm = torch.stft(b.reshape(B * Kb, T), nfft, hop, window=win,
                    return_complex=True).abs()
    return (torch.log1p(A) - torch.log1p(Bm)).abs().mean()


def losses(net, x, y, dev, with_base=False):
    x, y = x.to(dev), y.to(dev)
    pred = net(x, *[s.to(dev) for s in net.zero_states(x.shape[0])])[0]
    m = torch.ones_like(y)
    m[:, :, :RF] = 0.0                                      # warm-up mask
    l1 = ((pred - y).abs() * m).sum() / m.sum()
    # review 2 Aug: spectral term also masks warmup (zero states)
    sp = stft_l1((x[:, :4] + pred / RES_SCALE)[:, :, RF:],
                 (x[:, :4] + y / RES_SCALE)[:, :, RF:])
    total = 0.2 * l1 + sp                                   # T1-2: spectrum dominates
    if not with_base:
        return total, l1.item(), sp.item(), None
    b1 = (y.abs() * m).sum() / m.sum()                      # residual=0 baseline
    b_sp = stft_l1(x[:, :4, RF:], (x[:, :4] + y / RES_SCALE)[:, :, RF:])
    return total, l1.item(), sp.item(), (0.2 * b1 + b_sp).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "..", "corpus_b3"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=1440)       # 120 ms (RF=252)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--per-item", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "models", "b3"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(20260802)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:                                     # builds a mini corpus itself
        args.corpus = "/tmp/corpus_b3_smoke"
        if not glob.glob(os.path.join(args.corpus, "*.npz")):
            import subprocess
            subprocess.run([sys.executable,
                            os.path.join(HERE, "make_corpus_b3.py"),
                            "--minutes", "1.0", "--seed", "5",
                            "--out", args.corpus], check=True)
        args.epochs, args.batch, args.per_item = 3, 8, 4

    bases = sorted(p[:-4] for p in glob.glob(os.path.join(args.corpus, "*.npz")))
    assert bases, f"empty corpus: {args.corpus}"
    n_val = max(1, len(bases) // 10)
    tr = CorpusB3(bases[n_val:], args.crop, args.per_item, seed=1)
    va = CorpusB3(bases[:n_val], args.crop, args.per_item, seed=2)
    print(f"corpus: {len(bases)} pairs (train {len(bases)-n_val}/val {n_val}), "
          f"device: {dev}")

    net = StreamingTCN(c_in=8, c=C_FROZEN, c_out=4, layers=LAYERS).to(dev)
    torch.nn.init.zeros_(net.tail.weight)   # start == residual 0 (the baseline);
    # does not change the topology — init only; the net can only improve the baseline
    npar = sum(p.numel() for p in net.parameters())
    print(f"net: C={C_FROZEN}, L={LAYERS}, parameters {npar/1e3:.0f}K")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True)
    dv = torch.utils.data.DataLoader(va, batch_size=args.batch)

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    for ep in range(args.epochs):
        net.train()
        t0, acc = time.time(), []
        for x, y in dl:
            opt.zero_grad()
            loss, *_ = losses(net, x, y, dev)
            loss.backward()
            opt.step()
            acc.append(loss.item())
        sch.step()
        net.eval()
        with torch.no_grad():
            vl, v1, vb = [], [], []
            for x, y in dv:
                loss, l1, sp, base = losses(net, x, y, dev, with_base=True)
                vl.append(loss.item()); v1.append(l1); vb.append(base)
        vl, v1, vb = float(np.mean(vl)), float(np.mean(v1)), float(np.mean(vb))
        tag = ""
        if vl < best:
            best = vl
            torch.save({"model": net.state_dict(),
                        "res_scale": RES_SCALE,
                        "form": dict(c_in=8, c=C_FROZEN, c_out=4,
                                     layers=LAYERS)},
                       os.path.join(args.out, "ckpt_best.pt"))
            tag = " <- best"
        gain_db = 20 * np.log10(max(vb, 1e-9) / max(vl, 1e-9))
        print(f"epoch {ep+1:3d}/{args.epochs}: train {np.mean(acc):.4f} "
              f"val {vl:.4f} vs baseline(res=0) {vb:.4f} "
              f"(gain {gain_db:+.1f} dB) [{time.time()-t0:.0f} s]{tag}")
    print(f"done: best val {best:.4f} (baseline {vb:.4f}) -> "
          f"{os.path.join(args.out, 'ckpt_best.pt')}")


if __name__ == "__main__":
    main()
