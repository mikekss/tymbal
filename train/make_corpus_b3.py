#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_corpus_b3.py — a corpus of "dry skeleton -> teacher B3" pairs (D-1a/D-16, §4.1).
It replaces the tape protocol of §4.2: the pairs are rendered OFFLINE,
deterministically, sample-aligned (no τ(t)/align_takes). The teacher is
dsp/teacher_candidates.py::teacher_B3_residue with the D-16 conditioning axes:
  q(t)  — the spectral quantization step, dB;
  g(t)  — the gate depth below the frame peak, dB.
Axes per phrase: a constant from a range + a slow LFO wander (<=0.4 Hz) — the
analogue of "takes at 2-3 drive levels" from the old §4.2(d). The curves are
stored in the NPZ as 250 Hz frames (like all controls) — in training they go in
as conditioning, and the D-8 antenna maps onto the same ones at runtime.

THE AXIS IS CANON (D-16 DECIDED 2 Aug, listening to steps on a static drone —
ALL of them are audible): a single MACRO axis t in [0,1], q = 2+11t (dB),
g = 60-32t (dB). The network is conditioned by one channel t (the D-8 antenna
maps onto it too); q/g in the NPZ are derived, for reproducibility.

Layout of phrase NNNN_cat: .wav (dry 48k f32), _wet.wav (teacher),
.npz (f0/amp/tA/tB/gate + q250/g250 + the render seed + cat). The PQMF split
into subbands is done on the training pipeline side (T1), not here.

Run:    python3 make_corpus_b3.py --minutes 5 --seed 1 --out ../corpus_b3
Smoke:  python3 make_corpus_b3.py --selftest
--teacher (T1-3): B3 (default, the canonical q/g axes) OR any teacher_search
candidate (A1_foldsat/A2_ottpress/B2_crush/B3_256/B3_soft) — then
wet(t) = dry + t·(teacher(dry) − dry) (depth, as in the search matrix);
in that case q250/g250 in the NPZ are derived from t only formally (for compatibility).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))
import skeleton_a as ska                                    # noqa: E402
import make_corpus as mc                                    # noqa: E402
from teacher_candidates import teacher_B3_residue, FS       # noqa: E402

FR = 250


def macro_to_qg(t):
    """Canon D-16: t in [0,1] -> (q, g) in dB."""
    t = np.clip(t, 0.0, 1.0)
    return 2.0 + 11.0 * t, 60.0 - 32.0 * t


def axis_curves(nfr, rng):
    """(t250, q250, g250): a base per phrase + a slow LFO (<=0.4 Hz),
    full coverage of the axis (the network must see the antenna's whole travel)."""
    tm = np.arange(nfr) / FR
    t0 = rng.uniform(0.03, 1.0)
    t = t0 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * tm
                           + rng.uniform(0, 6.28))
    t = np.clip(t, 0.0, 1.0)
    q, g = macro_to_qg(t)
    return t, q, g


def resolve_teacher(name):
    """-> (fn|None, lookahead_la). None = canonical B3 (q/g axes, la=128:
    STFT 512/128 — review 2 Aug). Otherwise a teacher_search candidate."""
    if name in (None, "B3"):
        return None, 128
    import teacher_search as ts
    d = {n: (f, la) for n, f, la in ts.CANDS}
    assert name in d, f"no candidate {name}; available: B3, {', '.join(d)}"
    return d[name]


def render_pair(cat, dur_s, rng, teacher_fn=None):
    """-> (dry, wet, ph, q250, g250, seed_r) — deterministic given rng."""
    ph = mc.gen_phrase(cat, dur_s, rng)
    seed_r = int(rng.integers(1, 2 ** 31))
    dry = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"], ph["gate"],
                           seed=seed_r)
    peak = np.max(np.abs(dry)) + 1e-9
    if peak > 0.98:
        dry = dry * (0.98 / peak)
    t250, q250, g250 = axis_curves(len(ph["f0"]), rng)
    if teacher_fn is None:                                  # canonical B3: q/g axes
        wet = np.asarray(teacher_B3_residue(dry, qdb=q250, gate_db=g250))
    else:                                                   # T1-3: depth t
        t48 = np.repeat(t250.astype(np.float64), FS // FR)[:len(dry)]
        wet = dry + t48 * (np.asarray(teacher_fn(dry))[:len(dry)] - dry)
    return dry, wet[:len(dry)], ph, t250, q250, g250, seed_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("..", "corpus_b3"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--teacher", default="B3",
                    help="B3 (canonical) or a teacher_search candidate")
    args = ap.parse_args()
    teacher_fn, teacher_la = resolve_teacher(args.teacher)

    if args.selftest:
        rng = np.random.default_rng(7)
        d1, w1, ph, t, q, g, s1 = render_pair("legato", 2.0, rng)
        rng = np.random.default_rng(7)
        d2, w2, _, _, _, _, _ = render_pair("legato", 2.0, rng)
        assert np.array_equal(d1, d2) and np.array_equal(w1, w2), "non-determinism"
        assert len(d1) == len(w1) == len(ph["f0"]) * (FS // FR), "pair lengths"
        assert np.all(np.isfinite(w1)) and np.max(np.abs(w1)) < 1.5
        assert t.shape == q.shape == g.shape == ph["f0"].shape, "axes != grid"
        qq, gg = macro_to_qg(t)
        assert np.allclose(qq, q) and np.allclose(gg, g), "q/g != macro(t)"
        d_db = 20 * np.log10(np.sqrt(np.mean((w1 - d1) ** 2)) /
                             (np.sqrt(np.mean(d1 ** 2)) + 1e-30))
        print(f"[selftest] pair OK: {len(d1)} samples/48k, diff {d_db:.1f} dB, "
              f"t[{t.min():.2f}..{t.max():.2f}] q[{q.min():.1f}..{q.max():.1f}] "
              f"g[{g.min():.1f}..{g.max():.1f}]")
        print("[selftest] ALL OK")
        return

    from scipy.io import wavfile
    rng = np.random.default_rng(args.seed)
    outdir = os.path.abspath(os.path.join(HERE, args.out))
    os.makedirs(outdir, exist_ok=True)
    total, idx, manifest = 0.0, 0, []
    while total < args.minutes * 60.0:
        cat = rng.choice(mc.CATS, p=mc.CAT_W)
        dur = float(rng.uniform(4.0, 10.0))
        dry, wet, ph, t250, q250, g250, seed_r = render_pair(cat, dur, rng,
                                                             teacher_fn)
        name = f"{idx:04d}_{cat}"
        wavfile.write(os.path.join(outdir, name + ".wav"), FS,
                      dry.astype(np.float32))
        wavfile.write(os.path.join(outdir, name + "_wet.wav"), FS,
                      wet.astype(np.float32))
        np.savez(os.path.join(outdir, name + ".npz"), **ph,
                 t250=t250.astype(np.float32),
                 q250=q250.astype(np.float32), g250=g250.astype(np.float32),
                 seed=np.uint32(seed_r), cat=cat, teacher=args.teacher,
                 teacher_la=np.int32(teacher_la))
        manifest.append(f"{name}\t{dur:.1f}s\tt~{np.mean(t250):.2f}")
        total += dur
        idx += 1
    with open(os.path.join(outdir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"corpus ({args.teacher}): {idx} pairs, {total/60:.1f} min -> {outdir}")


if __name__ == "__main__":
    main()
