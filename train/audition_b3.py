#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audition_b3.py — listening test for the T1 checkpoint without hardware: host
inference of the refiner.

Renders a FRESH phrase (seed outside the corpus), runs it through the network
(ckpt_best.pt), synthesizes back to 48k and writes three wavs side by side: the
dry skeleton, the prediction (skeleton+residual) and the teacher (B3 reference
with the same axes). Listen: pred vs teacher — what the network learned;
pred vs dry — what it does at all.

Run:    python3 audition_b3.py [--ckpt ../models/b3/ckpt_best.pt]
        [--t 0.8] [--cat drone] [--seed 777] [--out ../dsp/audition]
Output: t1_dry.wav / t1_pred.wav / t1_teacher.wav (+ prints the rel RMS of
        pred-vs-teacher — a rough numeric convergence figure).
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))
import pqmf_design as pq                                    # noqa: E402
import skeleton_a as ska                                    # noqa: E402
import make_corpus as mc                                    # noqa: E402
from teacher_candidates import teacher_B3_residue           # noqa: E402
from make_corpus_b3 import macro_to_qg                      # noqa: E402
from streaming_tcn import StreamingTCN                      # noqa: E402
from train_b3 import pqmf_analyze, FS, FR                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "models", "b3",
                                                   "ckpt_best.pt"))
    ap.add_argument("--t", type=float, default=0.8, help="D-16 macro axis")
    ap.add_argument("--cat", default="drone", choices=list(mc.CATS))
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "dsp", "audition"))
    ap.add_argument("--teacher", default="B3",
                    help="B3 (canonical q/g) or a teacher_search candidate — MUST "
                         "match the teacher of the corpus the ckpt was trained on")
    ap.add_argument("--prefix", default="t1",
                    help="wav name prefix (do not overwrite other nets' tests)")
    args = ap.parse_args()

    import soundfile as sf
    ck = torch.load(args.ckpt, map_location="cpu")
    net = StreamingTCN(**ck["form"])
    net.load_state_dict(ck["model"])
    net.eval()

    rng = np.random.default_rng(args.seed)
    ph = mc.gen_phrase(args.cat, args.dur, rng)
    dry = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"], ph["gate"],
                           seed=int(rng.integers(1, 2 ** 31)))
    pk = np.max(np.abs(dry)) + 1e-9
    if pk > 0.98:
        dry = dry * (0.98 / pk)
    nfr = len(ph["f0"])
    from make_corpus_b3 import resolve_teacher
    tfn, la = resolve_teacher(args.teacher)
    if tfn is None:                                    # canonical B3: q/g axes
        q, g = macro_to_qg(np.full(nfr, args.t))
        wet = np.asarray(teacher_B3_residue(dry, qdb=q, gate_db=g))[:len(dry)]
    else:                                                   # T1-3: depth t
        wet = dry + args.t * (np.asarray(tfn(dry))[:len(dry)] - dry)

    sd = pqmf_analyze(dry)
    n12 = sd.shape[1]

    def up(v):
        u = np.repeat(np.asarray(v, np.float32), FS // 4 // FR)
        return (u[:n12] if len(u) >= n12
                else np.pad(u, (0, n12 - len(u)), mode="edge"))
    x = np.concatenate([sd, up(ph["amp"] * ph["gate"])[None],
                        up(ph["tA"])[None], up(ph["tB"])[None],
                        up(np.full(nfr, args.t))[None]], axis=0)

    with torch.no_grad():
        xt = torch.from_numpy(x[None])
        res = net(xt, *net.zero_states(1))[0][0].numpy()
    res = res / float(ck.get("res_scale", 1.0))
    rt_src = sd                  # measure circular delay on the UNSHIFTED skeleton
    if la > 0:                         # the net predicts residual with a lookahead
        sd = np.pad(sd, ((0, 0), (la, 0)))[:, :sd.shape[1]]  # skeleton catches up
        wet = np.pad(wet, (4 * la, 0))[:len(dry)]            # reference likewise

    _, syn = pq.filterbank(pq.prototype(0.066603, beta=8.80))
    pred48 = pq.synthesize((sd + res).astype(np.float64), syn)
    # circular delay of analysis+synthesis (~N-1): measure it on the dry signal
    # and remove it, else rel RMS compares shifted signals and is off by a lot
    rt = pq.synthesize(rt_src.astype(np.float64), syn)
    n = min(len(rt), len(dry), FS)
    from scipy import signal as sg
    lag = int(np.argmax(sg.correlate(rt[:n + 512], dry[:n], mode="valid")))
    pred48 = pred48[lag:lag + len(dry)]
    if len(pred48) < len(dry):
        pred48 = np.pad(pred48, (0, len(dry) - len(pred48)))

    os.makedirs(args.out, exist_ok=True)
    for name, sig in ((f"{args.prefix}_dry", dry),
                      (f"{args.prefix}_pred", pred48),
                      (f"{args.prefix}_teacher", wet)):
        sf.write(os.path.join(args.out, name + ".wav"),
                 np.asarray(sig, np.float32), FS, subtype="FLOAT")
    rel = (np.sqrt(np.mean((pred48 - wet[:len(pred48)]) ** 2)) /
           (np.sqrt(np.mean(wet ** 2)) + 1e-30))
    rel0 = (np.sqrt(np.mean((dry[:len(pred48)] - wet[:len(pred48)]) ** 2)) /
            (np.sqrt(np.mean(wet ** 2)) + 1e-30))
    print(f"[audition] {args.teacher} {args.cat} t={args.t}: pred-vs-teacher "
          f"rel RMS = {rel:.3f} (baseline dry-vs-teacher {rel0:.3f}) -> "
          f"{args.out}/{args.prefix}_*.wav")


if __name__ == "__main__":
    main()
