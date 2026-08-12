#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harm_extract.py — the GATE of a new direction (D-18), with no ML at all.

The question it answers: is the model "harmonics + noise on a 250 Hz control
grid", the one our skeleton lives in, sufficient at all to sound like a real
instrument? If not, training a decoder is pointless — the ceiling is below the goal.

Method: from a monophonic recording, frame by frame (250 Hz), we pull out f0,
loudness and the HARMONIC AMPLITUDE VECTOR A[h] — by measuring the spectrum
directly at h*f0. Then this is fed into the EXISTING `skeleton_a.render_voice` as
the decoder, and the result is written next to the original.

This is an upper bound: A[h] here is extracted exactly, not predicted. Whatever a
decoder on the NPU manages will be NO BETTER than this resynthesis. So this is
what has to be listened to.

What the model definitely loses (and it is honestly audible in the resynthesis):
  - the phase relations of the harmonics (the skeleton imposes its own master phase);
  - inharmonic partials and inexact harmonicity (string/reed);
  - noise beyond what the skeleton's xorshift generator gives;
  - everything faster than 4 ms — the attacks get smeared by the frame grid.

RUN
  python harm_extract.py --wav ../rec/flute.wav
  python harm_extract.py --wav ../rec/bass.wav --fmin 40 --fmax 400
  python harm_extract.py --selftest        # mechanics check on synthetic material
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))

import skeleton_a as ska                                     # noqa: E402

FS, FR = 48000, 250
HOP = FS // FR                      # 192
NH = ska.NH                         # 100
NFFT = 4096                         # analysis window (~85 ms) — frequency resolution


# ------------------------------------------------------------------ f0
def f0_frame(x, fs, fmin, fmax):
    """Normalized autocorrelation (YIN-lite) on a single window."""
    n = len(x)
    if n < 64:
        return 0.0, 0.0
    x = x - x.mean()
    e = float(np.dot(x, x))
    if e < 1e-12:
        return 0.0, 0.0
    lo, hi = max(2, int(fs / fmax)), min(n - 2, int(fs / fmin))
    if hi <= lo + 2:
        return 0.0, 0.0
    # the YIN difference function over the lag range
    d = np.empty(hi - lo + 1)
    for i, lag in enumerate(range(lo, hi + 1)):
        diff = x[:n - lag] - x[lag:]
        d[i] = float(np.dot(diff, diff))
    cum = np.cumsum(d) / (np.arange(1, d.size + 1))
    dn = d / np.maximum(cum, 1e-30)
    i = int(np.argmin(dn))
    # parabolic refinement
    if 0 < i < dn.size - 1:
        a, b, c = dn[i - 1], dn[i], dn[i + 1]
        denom = (a - 2 * b + c)
        i_ref = i + (0.5 * (a - c) / denom if abs(denom) > 1e-20 else 0.0)
    else:
        i_ref = i
    lag = lo + i_ref
    conf = float(1.0 - min(dn[i], 1.0))          # 1 = confidently tonal
    return (fs / lag if lag > 0 else 0.0), conf


def extract(x, fs=FS, fmin=50.0, fmax=1200.0, conf_thr=0.55):
    """-> f0[F], amp[F], gate[F], A[F, NH] (normalized to sum=1 per frame)."""
    F = len(x) // HOP
    f0 = np.zeros(F); amp = np.zeros(F); gate = np.zeros(F)
    A = np.zeros((F, NH))
    win = np.hanning(NFFT)
    half = NFFT // 2
    xp = np.pad(np.asarray(x, np.float64), (half, half))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)
    df = freqs[1]
    for k in range(F):
        c = k * HOP + half
        seg = xp[c - half:c + half] * win
        # f0 is taken from a shorter window — it has to keep up with the notes
        s2 = xp[c - 1024:c + 1024]
        f, conf = f0_frame(s2, fs, fmin, fmax)
        rms = float(np.sqrt(np.mean(seg ** 2))) * np.sqrt(2.0)
        amp[k] = rms
        if f <= 0 or conf < conf_thr or rms < 1e-5:
            gate[k] = 0.0
            f0[k] = f0[k - 1] if k else 220.0
            continue
        gate[k] = 1.0
        f0[k] = f
        S = np.abs(np.fft.rfft(seg))
        # the amplitude of harmonic h is the spectral peak near h*f0 (±half a bin)
        for h in range(1, NH + 1):
            fh = f * h
            if fh >= fs / 2 - 2 * df:
                break
            b = fh / df
            i0 = int(round(b))
            lo, hi = max(0, i0 - 1), min(len(S) - 1, i0 + 1)
            A[k, h - 1] = float(S[lo:hi + 1].max())
    s = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(s, 1e-12)
    return f0, amp, gate, A


# ------------------------------------------------------------------ resynthesis
def resynth(f0, amp, gate, A, tA=0.5, tB=0.15, seed=0xC0FFEE):
    """The same render_voice, but the decoder returns the EXTRACTED A[h]."""
    F = len(f0)
    Apad = np.vstack([A, A[-1:]])                # render_voice expects F+1 frames

    def dec(f0g, ampg, tAg, tBg):
        n = len(f0g)
        return Apad[:n] if n <= Apad.shape[0] else np.vstack(
            [Apad, np.repeat(Apad[-1:], n - Apad.shape[0], axis=0)])

    return ska.render_voice(f0, amp, np.full(F, tA), np.full(F, tB), gate,
                            seed=seed, decoder=dec)


def spec_dist(a, b, nfft=2048, hop=512, floor_db=60.0):
    """Log-magnitude distance, dB. The floor is measured from the FRAME PEAK:
    without it the bins between harmonics (where both values are ~0) give
    hundreds of dB of noise, and the metric becomes meaningless (checked: it was
    99 dB on a perfect resynthesis). The metric ignores phase — the skeleton
    imposes its own master phase."""
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)
    w = np.hanning(nfft)
    acc, cnt = 0.0, 0
    for i in range(0, n - nfft, hop):
        A = np.abs(np.fft.rfft(a[i:i + nfft] * w))
        B = np.abs(np.fft.rfft(b[i:i + nfft] * w))
        ref = max(A.max(), B.max())
        if ref < 1e-9:
            continue
        eps = ref * 10 ** (-floor_db / 20.0)
        m = (A > eps) | (B > eps)
        if not m.any():
            continue
        acc += float(np.mean(np.abs(20 * np.log10((A[m] + eps) / (B[m] + eps)))))
        cnt += 1
    return acc / max(cnt, 1)


def selftest():
    """Synthetic material with a KNOWN spectral trend: extraction must recover it."""
    fs, dur = FS, 2.0
    t = np.arange(int(fs * dur)) / fs
    f = 220.0
    x = np.zeros_like(t)
    true_A = []
    for k in range(int(dur * FR)):
        pass
    # harmonics 1..12, the tilt moves from steep to shallow
    ph = 2 * np.pi * f * t
    slope = np.interp(t, [0, dur], [3.0, 1.2])
    for h in range(1, 13):
        x += (h ** -slope) * np.sin(h * ph)
    x /= np.max(np.abs(x))
    f0, amp, gate, A = extract(x, fs, fmin=80, fmax=800)
    ok_f0 = np.median(f0[gate > 0])
    print("f0: median %.2f Hz (truth 220), tonal frames %d out of %d"
          % (ok_f0, int(gate.sum()), len(gate)))
    assert abs(ok_f0 - 220.0) < 1.0, ok_f0
    # check the tilt trend: the ratio A[2]/A[1] must grow
    r = A[:, 1] / np.maximum(A[:, 0], 1e-12)
    m = gate > 0
    r0, r1 = np.median(r[m][:20]), np.median(r[m][-20:])
    print("A2/A1: start %.3f -> end %.3f (expect growth: the tilt flattens out)"
          % (r0, r1))
    assert r1 > r0 * 1.5, (r0, r1)
    y = resynth(f0, amp, gate, A)
    d = spec_dist(x, y[:len(x)])
    print("spectral distance original/resynthesis: %.1f dB" % d)
    print("\nSELFTEST OK — the extraction catches both the pitch and the spectral trend")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "dsp", "audition"))
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=1200.0)
    ap.add_argument("--tA", type=float, default=0.5)
    ap.add_argument("--npz", default=None, help="where to put the frames for training")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.wav:
        sys.exit("need --wav (mono, one instrument, no reverb)")
    import soundfile as sf
    x, sr = sf.read(a.wav, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != FS:
        from scipy import signal as sg
        x = sg.resample_poly(x, FS, sr)
        print("resample %d -> %d Hz" % (sr, FS))
    x = np.asarray(x, np.float64)
    pk = np.max(np.abs(x)) + 1e-12
    if pk > 0.98:
        x = x * (0.98 / pk)

    f0, amp, gate, A = extract(x, FS, a.fmin, a.fmax)
    y = resynth(f0, amp, gate, A, tA=a.tA)
    n = min(len(x), len(y))
    d = spec_dist(x[:n], y[:n])
    tone = float(gate.mean())
    print("frames %d, tonal %.0f%%, f0 median %.1f Hz"
          % (len(f0), 100 * tone, float(np.median(f0[gate > 0])) if tone else 0))
    print("spectral distance original/resynthesis: %.1f dB" % d)

    os.makedirs(a.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.wav))[0]
    sf.write(os.path.join(a.out, "hx_%s_orig.wav" % base),
             np.asarray(x[:n], np.float32), FS, subtype="FLOAT")
    sf.write(os.path.join(a.out, "hx_%s_resynth.wav" % base),
             np.asarray(y[:n], np.float32), FS, subtype="FLOAT")
    if a.npz:
        np.savez_compressed(a.npz, f0=f0, amp=amp, gate=gate, A=A)
        print("frames -> %s" % a.npz)
    print("\n-> %s/hx_%s_{orig,resynth}.wav" % (a.out, base))
    print("LISTEN: this is the CEILING of the direction. A decoder on the NPU will not do better —")
    print("it will PREDICT what has been extracted exactly here.")


if __name__ == "__main__":
    main()
