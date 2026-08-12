#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audition_delta.py — listening test for WHAT WILL ACTUALLY PLAY.

Why a separate script: `audition_b3.py` was written for the "dry + net" design
and knows nothing about the FIR. But the runtime since 2 Aug is the FIR bank on
the M55 PLUS the net on the NPU on top of it (D-17). So the comparison is not
pred against teacher, but four tracks:

  <pfx>_dry.wav      dry skeleton — how the instrument sounds with nothing
  <pfx>_teacher.wav  the target: what the teacher does (A2_ottpress) — ceiling
  <pfx>_fir.wav      linear FIR only (520 coefficients on the M55, NPU off)
  <pfx>_full.wav     FIR + net — what actually plays

Listen IN PAIRS:
  fir  vs full     — what the NPU buys. If the difference is inaudible, the
                     +9.18 dB measures something the ear misses, worth knowing.
  full vs teacher  — how close we got to the target.
  dry  vs teacher  — whether this teacher is needed at all. THE MOST
                     IMPORTANT ONE: A2_ottpress was picked by numbers, nobody
                     ever listened to it.

Prints the residual suppression on the same phrase for fir and for fir+net — so
that the number and the ear stand side by side.

RUN
  python audition_delta.py --ckpt ../teacher_search/A2_ottpress/ckpt_delta_p360.pt
  python audition_delta.py --cat drone --t 0.9 --seed 4242 --dur 10
  python audition_delta.py --no-net        # without torch: only dry/teacher/fir
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))

import pqmf_design as pq                                     # noqa: E402
import skeleton_a as ska                                     # noqa: E402
import make_corpus as mc                                     # noqa: E402
import teacher_candidates as tc                              # noqa: E402

FS, FR, K = 48000, 250, 4
FC_OPT, BETA_OPT = 0.066603, 8.80
RES_SCALE = 32.0
_ANA = _SYN = None


def analyze(x):
    global _ANA
    if _ANA is None:
        _ANA, _ = pq.filterbank(pq.prototype(FC_OPT, beta=BETA_OPT))
    return pq.analyze(np.asarray(x, np.float64), _ANA).astype(np.float32)


def synth(sub):
    global _SYN
    if _SYN is None:
        _, _SYN = pq.filterbank(pq.prototype(FC_OPT, beta=BETA_OPT))
    return pq.synthesize(np.asarray(sub, np.float64), _SYN)




# All candidates, including the REJECTED ones — so we can hear what was dropped.
# B3 is not reproducible by the net (it is memorised, but does not generalise),
# yet as an exact DSP block on the M55 it is cheap: an FFT with 2.5M cycles of
# headroom is pennies, a 256 window costs +5.3 ms of path latency. So "listening
# to B3" is not nostalgia, it is an assessment of whether it is worth that
# latency.
TEACHERS = {
    "A1_foldsat":  tc.teacher_A1_foldsat,      # folder+saturation = distortion
    "A2_ottpress": tc.teacher_A2_ottpress,     # OTT dynamics (current choice)
    "B2_crush":    tc.teacher_B2_crush,        # SRR + mu-law = bitcrush
    "B3_512":      lambda d: tc.teacher_B3_residue(d, qdb=9.0, gate_db=38.0),
    "B3_256":      lambda d: tc.teacher_B3_residue(d, nfft=256, hop=64,
                                                   qdb=9.0, gate_db=38.0),
    "B3_hard":     lambda d: tc.teacher_B3_residue(d, qdb=12.0, gate_db=30.0),
}


# -------------------------------------------------------------- musical phrases
# `make_corpus.gen_phrase` renders RANDOM material: random out-of-key notes,
# random durations, a random LFO on the timbre. That is a training corpus —
# coverage, not music. You cannot judge the sound by it.
# Here are a few hand-written scores. And separately: the project methodology
# (D-16) says outright — listen to the axes on STATIC material, sweeps on top of
# changing phrases get masked. Hence the first score is exactly that.
SEMI = 2.0 ** (1.0 / 12.0)


def _score_frames(notes, dur_s, tA_curve, tB=0.2, amp=0.5):
    """notes: [(midi, duration share)]; emits 250 Hz frames."""
    F = int(dur_s * FR)
    f0 = np.zeros(F); amp_a = np.zeros(F); gate = np.zeros(F)
    tot = sum(d for _, d in notes)
    pos = 0
    for i, (m, d) in enumerate(notes):
        ln = int(round(F * d / tot)) if i < len(notes) - 1 else F - pos
        ln = max(2, min(ln, F - pos))
        f0[pos:pos + ln] = 440.0 * SEMI ** (m - 69)
        amp_a[pos:pos + ln] = amp
        g = np.ones(ln)
        at = min(max(2, int(0.01 * FR)), ln // 3)
        rl = min(max(2, int(0.08 * FR)), ln // 3)
        g[:at] = np.linspace(0, 1, at)
        g[-rl:] = np.linspace(1, 0, rl) ** 2
        gate[pos:pos + ln] = g
        pos += ln
    tm = np.linspace(0, 1, F)
    tA = np.asarray(tA_curve(tm), np.float64)
    return dict(f0=f0, amp=amp_a, tA=np.clip(tA, 0, 1),
                tB=np.full(F, tB), gate=gate)


def score_phrase(kind, dur_s):
    if kind == "hold":          # one low note + slow timbre travel
        return _score_frames([(41, 1.0)], dur_s,
                             lambda t: 0.15 + 0.7 * t, tB=0.12)
    if kind == "line":          # calm in-key line, legato
        seq = [(57, 1), (60, 1), (62, 1), (64, 1.5), (62, 1), (60, 2)]
        return _score_frames(seq, dur_s, lambda t: 0.45 + 0.1 * np.sin(6 * t))
    if kind == "pulse":         # one note repeated — attack and tail audible
        return _score_frames([(48, 1)] * 8, dur_s, lambda t: 0.5 + 0.0 * t)
    raise SystemExit("no score %s" % kind)


def make_item(cat, dur, seed, t_val, score=None, noise_tilt_db=0.0):
    """Fresh phrase OUTSIDE the corpus + constant t axis (as in build_xy)."""
    rng = np.random.default_rng(seed)
    ph = score_phrase(score, dur) if score else mc.gen_phrase(cat, dur, rng)
    dry = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"], ph["gate"],
                           seed=int(rng.integers(1, 2 ** 31)),
                           noise_tilt_db=noise_tilt_db)
    pk = np.max(np.abs(dry)) + 1e-9
    if pk > 0.98:
        dry = dry * (0.98 / pk)
    return dict(cat=cat, ph=ph, dry=dry,
                t=np.full(len(ph["f0"]), float(t_val), np.float64))


def build_x(item, wet_full):
    """Net input [8, n12] and target [4, n12] — teacher_search.build_xy layout."""
    dry, ph, tcv = item["dry"], item["ph"], item["t"]
    t48 = np.repeat(tcv, FS // FR)[:len(dry)]
    wet = dry + t48 * (np.asarray(wet_full)[:len(dry)] - dry)
    sd, sw = analyze(dry), analyze(wet)
    n12 = sd.shape[1]

    def up(v):
        u = np.repeat(np.asarray(v, np.float32), FS // 4 // FR)
        return (u[:n12] if len(u) >= n12
                else np.pad(u, (0, n12 - len(u)), mode="edge"))
    x = np.concatenate([sd, up(ph["amp"] * ph["gate"])[None],
                        up(ph["tA"])[None], up(ph["tB"])[None],
                        up(tcv)[None]], axis=0)
    y = (sw - sd) * RES_SCALE
    return x.astype(np.float32), y.astype(np.float32), wet


def fir_features(x, taps):
    """Copy of probe_linear.fir_features (order: b major/outer, k minor)."""
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
    P = cols.T
    t = np.asarray(x[7], np.float32)[:, None]
    return np.concatenate([P, P * t], axis=1)


def sup_db(y, yhat, skip):
    d = (np.asarray(yhat, np.float64) - np.asarray(y, np.float64))[:, skip:]
    m0 = float(np.mean(np.asarray(y, np.float64)[:, skip:] ** 2))
    return 10 * np.log10(max(m0, 1e-30) / max(float(np.mean(d ** 2)), 1e-30))


def align_to(sig48, ref48):
    """Remove the circular analysis+synthesis delay (else we compare shifted)."""
    from scipy import signal as sg
    n = min(len(sig48), len(ref48), FS)
    lag = int(np.argmax(sg.correlate(sig48[:n + 512], ref48[:n], mode="valid")))
    out = sig48[lag:lag + len(ref48)]
    if len(out) < len(ref48):
        out = np.pad(out, (0, len(ref48) - len(out)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "..", "teacher_search", "A2_ottpress", "ckpt_delta_p360.pt"))
    ap.add_argument("--teacher", default="A2_ottpress")
    ap.add_argument("--all-teachers", action="store_true",
                    help="dry + ALL teachers on one phrase: pick the character by ear")
    ap.add_argument("--cat", default="legato", choices=list(mc.CATS))
    ap.add_argument("--score", default="hold",
                    choices=["hold", "line", "pulse", "corpus"],
                    help="hold/line/pulse — hand-written phrases; corpus — random training material (you cannot judge the sound by it)")
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--taps", type=int, default=65)
    ap.add_argument("--noise-tilt", type=float, default=0.0,
                    help="D-23: skeleton noise tilt, dB per subband (0 = as before)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "dsp", "audition"))
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--no-net", action="store_true",
                    help="without torch: only dry/teacher/fir")
    a = ap.parse_args()
    import soundfile as sf

    pfx = a.prefix or ("d_%s_%s_t%02d" % (
        a.teacher, a.cat if a.score == "corpus" else a.score,
        round(a.t * 10)))
    tfn = TEACHERS[a.teacher]

    item = make_item(a.cat, a.dur, a.seed, a.t,
                     None if a.score == "corpus" else a.score,
                     noise_tilt_db=a.noise_tilt)

    if a.all_teachers:
        # teacher character only: the net and the FIR play no part here
        os.makedirs(a.out, exist_ok=True)
        dry = item["dry"]
        ref = float(np.sqrt(np.mean(np.asarray(dry, np.float64) ** 2)))
        base = "cmp_%s_t%02d" % (a.score, round(a.t * 10))
        sf.write(os.path.join(a.out, base + "_00_dry.wav"),
                 np.asarray(dry, np.float32), FS, subtype="FLOAT")
        print("dry + %d teachers, score %s, t=%.2f:"
              % (len(TEACHERS), a.score, a.t))
        t48 = np.repeat(item["t"], FS // FR)[:len(dry)]
        for i, (nm, fn) in enumerate(sorted(TEACHERS.items()), 1):
            w = np.asarray(fn(dry))[:len(dry)]
            wet = dry + t48 * (w - dry)
            r = float(np.sqrt(np.mean(wet ** 2)))
            g = ref / r if r > 1e-12 else 1.0
            out = wet * g
            pk = float(np.max(np.abs(out))) + 1e-12
            if pk > 0.99:
                out = out * (0.99 / pk)
            sf.write(os.path.join(a.out, "%s_%02d_%s.wav" % (base, i, nm)),
                     np.asarray(out, np.float32), FS, subtype="FLOAT")
            diff = 20 * np.log10(np.sqrt(np.mean((out - dry) ** 2)) /
                                 (ref + 1e-30))
            print("  %-12s difference from dry %+6.1f dB, level trim %+.1f dB"
                  % (nm, diff, 20 * np.log10(g)))
        print("\n-> %s/%s_*.wav" % (a.out, base))
        print("files are numbered: 00 dry, then the teachers alphabetically")
        return
    x, y, wet = build_x(item, np.asarray(tfn(item["dry"])))
    sd = x[:4]

    W = taps = None
    net_res = np.zeros_like(y)
    if not a.no_net:
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        taps = int(ck.get("taps", a.taps))
        W = np.asarray(ck["fir_W"], np.float64)
        sys.path.insert(0, HERE)
        from streaming_tcn import StreamingTCN
        net = StreamingTCN(**ck["form"])
        net.load_state_dict(ck["model"])
        net.eval()
        with torch.no_grad():
            net_res = net(torch.from_numpy(x[None]),
                          *net.zero_states(1))[0][0].numpy()
    else:
        taps = a.taps
        import pickle
        W = np.zeros((8 * taps, 4))            # without a ckpt the FIR is empty

    fir_res = (fir_features(x, taps) @ W).T.astype(np.float32)
    RF = 252
    print("on this phrase (%s, t=%.2f, %.0f s):"
          % (a.cat if a.score == "corpus" else a.score, a.t, a.dur))
    print("  FIR alone     : %+6.2f dB" % sup_db(y, fir_res, RF))
    print("  FIR + net     : %+6.2f dB" % sup_db(y, fir_res + net_res, RF))

    os.makedirs(a.out, exist_ok=True)
    dry = item["dry"]
    tracks = {
        "dry": dry,
        "teacher": wet[:len(dry)],
        "fir": align_to(synth(sd + fir_res / RES_SCALE), dry),
        "full": align_to(synth(sd + (fir_res + net_res) / RES_SCALE), dry),
    }
    for nm, sig in tracks.items():
        p = os.path.join(a.out, "%s_%s.wav" % (pfx, nm))
        sf.write(p, np.asarray(sig, np.float32), FS, subtype="FLOAT")
    print("\n-> %s/%s_{dry,teacher,fir,full}.wav" % (a.out, pfx))
    print("listen in pairs: fir vs full (what the NPU gives), full vs teacher "
          "(did we get there), dry vs teacher (is such a teacher needed at all)")


if __name__ == "__main__":
    main()
