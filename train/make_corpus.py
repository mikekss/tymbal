#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_corpus.py — training corpus generator (guide §4.1).
Procedural scores in the aesthetic of spec §1 -> 250 Hz control curves ->
skeleton render VARIANT A (48k; §4.1: A is mandatory for data preparation) ->
WAV float32 48k + NPZ of controls (the conditioning — must be saved).

Phrase categories (§4.1): sparse (sparse melodic material with rests), legato
(portamento/glissando — the showcase), jumps (fast HF<->LF leaps), staccato
(transients), drone (long notes, slow tA/tB — path memory),
tech (sweeps/single harmonics/noise bursts, 10-20%).

Run:     python3 make_corpus.py --minutes 5 --seed 1 --out ../corpus
Smoke:   python3 make_corpus.py --selftest
Tape dubs are recorded from these WAVs at 2-3 levels (§4.2(d)); the drive of
each phrase goes into the NPZ in advance (nominal 0.0 — the actual level at
recording time).
"""
import argparse
import os
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsp"))
import skeleton_a as ska                                   # noqa: E402

FS = 48000
FR = 250                       # frames/s
SCALE = np.array([0, 2, 3, 5, 7, 10])                      # minor pentatonic+
BASE = 110.0

CATS = ("sparse", "legato", "jumps", "staccato", "drone", "tech")
CAT_W = (0.24, 0.20, 0.16, 0.14, 0.12, 0.14)


def _note(rng, lo=-12, hi=36):
    deg = rng.integers(0, len(SCALE))
    octv = rng.integers(lo // 12, hi // 12 + 1)
    return BASE * 2.0 ** ((SCALE[deg] + 12 * octv) / 12.0)


def _adsr(F, rng, att=(0.005, 0.05), rel=(0.05, 0.4)):
    a = max(1, int(rng.uniform(*att) * FR))
    r = max(1, int(rng.uniform(*rel) * FR))
    g = np.ones(F)
    g[:a] = np.linspace(0, 1, a)
    if r < F:
        g[-r:] = np.linspace(1, 0, r)
    return g


def gen_phrase(cat, dur_s, rng):
    """-> dict(f0, amp, tA, tB, gate) 250 Hz frames."""
    F = int(dur_s * FR)
    f0 = np.full(F, 220.0)
    amp = np.zeros(F)
    tA = np.full(F, 0.5)
    tB = np.full(F, 0.15)
    gate = np.zeros(F)

    if cat == "sparse":
        pos = 0
        while pos < F - 25:
            ln = int(rng.uniform(0.25, 1.2) * FR)
            ln = min(ln, F - pos)
            f0[pos:pos + ln] = _note(rng)
            gate[pos:pos + ln] = _adsr(ln, rng)
            amp[pos:pos + ln] = rng.uniform(0.35, 0.6)
            pos += ln + int(rng.uniform(0.3, 1.5) * FR)      # air
        tA += 0.3 * np.sin(np.linspace(0, 2 * np.pi * rng.uniform(0.5, 2), F))
    elif cat == "legato":
        nn = rng.integers(3, 8)
        knots = np.sort(rng.choice(np.arange(1, F - 1), nn, replace=False))
        freqs = [_note(rng) for _ in range(nn + 1)]
        logf = np.zeros(F)
        prev = 0
        for i, kx in enumerate(list(knots) + [F]):
            logf[prev:kx] = np.log(freqs[i])
            prev = kx
        glide = max(1, int(rng.uniform(0.05, 0.3) * FR))
        ker = np.ones(glide) / glide
        logf = np.convolve(logf, ker, mode="same")           # expo portamento
        f0[:] = np.exp(logf)
        gate[:] = 1.0
        gate[:2] = 0
        gate[-3:] = 0
        amp[:] = rng.uniform(0.4, 0.6)
        tB[:] = rng.uniform(0.1, 0.35)
    elif cat == "jumps":
        pos = 0
        hi_lo = True
        while pos < F - 10:
            ln = int(rng.uniform(0.06, 0.15) * FR)
            ln = max(3, min(ln, F - pos))
            f0[pos:pos + ln] = _note(rng, 24, 48) if hi_lo else _note(rng, -12, 0)
            gate[pos:pos + ln] = _adsr(ln, rng, att=(0.004, 0.01), rel=(0.01, 0.05))
            amp[pos:pos + ln] = rng.uniform(0.4, 0.65)
            hi_lo = not hi_lo
            pos += ln
        tA[:] = rng.uniform(0.6, 0.9)
    elif cat == "staccato":
        pos = int(rng.uniform(0, 0.2) * FR)
        while pos < F - 8:
            ln = max(2, int(rng.uniform(0.02, 0.08) * FR))
            f0[pos:pos + ln] = _note(rng)
            gate[pos:pos + ln] = 1.0
            amp[pos:pos + ln] = rng.uniform(0.45, 0.7)
            pos += ln + int(rng.uniform(0.05, 0.35) * FR)
    elif cat == "drone":
        f0[:] = _note(rng, -12, 12)
        gate[:] = 1.0
        gate[:5] = np.linspace(0, 1, 5)
        gate[-13:] = np.linspace(1, 0, 13)
        amp[:] = 0.5
        tA[:] = 0.5 + 0.45 * np.sin(np.linspace(0, 2 * np.pi * rng.uniform(0.2, 0.8), F)
                                    + rng.uniform(0, 6.28))
        tB[:] = 0.2 + 0.15 * np.sin(np.linspace(0, 2 * np.pi * rng.uniform(0.1, 0.5), F))
    else:  # tech
        kind = rng.integers(0, 3)
        gate[:] = 1.0
        gate[:3] = 0
        gate[-3:] = 0
        if kind == 0:                                        # log sweep
            f0[:] = np.exp(np.linspace(np.log(50.0), np.log(rng.uniform(4000, 12000)), F))
            amp[:] = 0.4
            tA[:] = 0.1                                      # almost a sine
        elif kind == 1:                                      # single harmonic
            f0[:] = rng.uniform(100, 8000)
            amp[:] = 0.4
            tA[:] = 0.05
        else:                                                # noise bursts
            amp[:] = 0.0
            pos = 0
            while pos < F - 5:
                ln = max(2, int(rng.uniform(0.03, 0.2) * FR))
                amp[pos:pos + ln] = rng.uniform(0.3, 0.6)
                pos += ln + int(rng.uniform(0.1, 0.5) * FR)
            tB[:] = 0.9
            f0[:] = 220.0
    return dict(f0=f0, amp=amp, tA=tA, tB=tB, gate=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("..", "corpus"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        args.minutes, args.out = 0.2, os.path.join("..", "corpus_selftest")

    from scipy.io import wavfile
    rng = np.random.default_rng(args.seed)
    outdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out))
    os.makedirs(outdir, exist_ok=True)
    total = 0.0
    idx = 0
    manifest = []
    while total < args.minutes * 60.0:
        cat = rng.choice(CATS, p=CAT_W)
        dur = float(rng.uniform(4.0, 10.0))
        ph = gen_phrase(cat, dur, rng)
        seed_r = int(rng.integers(1, 2 ** 31))
        audio = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"],
                                 ph["gate"], seed=seed_r)
        peak = np.max(np.abs(audio)) + 1e-9
        if peak > 0.98:
            audio = audio * (0.98 / peak)
        name = f"{idx:04d}_{cat}"
        wavfile.write(os.path.join(outdir, name + ".wav"), FS,
                      audio.astype(np.float32))
        np.savez(os.path.join(outdir, name + ".npz"), **ph,
                 seed=np.uint32(seed_r), drive=np.float32(0.0), cat=cat)
        manifest.append(f"{name}\t{dur:.1f}s")
        total += dur
        idx += 1
    with open(os.path.join(outdir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"corpus: {idx} phrases, {total/60:.1f} min -> {outdir}")


if __name__ == "__main__":
    main()
