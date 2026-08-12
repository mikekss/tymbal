#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pilot.py — analysis of the 3150 Hz pilot tone from the characterization
session §4.2(b): the delay trajectory τ(t), the wow and flutter spectrum (rotor
peaks + background) — parameters for the runtime wowflutter model (W-2).

Run:     python3 analyze_pilot.py pilot_recording.wav [--f0 3150] [--out x.npz]
Smoke:   python3 analyze_pilot.py --selftest
         (synthesizes a pilot through dsp/wowflutter with known peaks and
          checks that they are recovered)
"""
import argparse
import os
import sys
import numpy as np
import scipy.signal as sig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dsp"))


def analyze(x, fs, f0=3150.0):
    sos = sig.butter(4, [f0 * 0.9, f0 * 1.1], btype="band", fs=fs, output="sos")
    z = sig.hilbert(sig.sosfilt(sos, x))
    phi = np.unwrap(np.angle(z))
    n = np.arange(len(x))
    tau = (phi - 2 * np.pi * f0 * n / fs) / (2 * np.pi * f0)     # τ(t), s
    speed = np.gradient(phi) * fs / (2 * np.pi * f0)             # v(t) rel.
    dev = speed - 1.0
    cut = int(0.2 * fs)                                          # FFT/Hilbert edges
    dev_c = dev[cut:-cut]
    fw, P = sig.welch(dev_c, fs, nperseg=1 << 15)
    m = (fw > 0.5) & (fw < 100.0)
    pk_idx, props = sig.find_peaks(P[m], prominence=np.max(P[m]) * 0.05)
    order = np.argsort(props["prominences"])[::-1][:6]
    peaks = [(float(fw[m][pk_idx[i]]),
              float(np.sqrt(P[m][pk_idx[i]] * (fw[1] - fw[0])))) for i in order]
    return dict(tau=tau, dev_rms=float(np.std(dev_c)), welch_f=fw, welch_p=P,
                peaks=sorted(peaks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--f0", type=float, default=3150.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import wowflutter as wf
        fs = 48000
        F = 1500                                             # 6 s
        n = F * wf.HOP
        t = np.arange(n)
        pilot = 0.5 * np.sin(2 * np.pi * 3150.0 * t / fs)
        y, _ = wf.process(pilot, np.ones(F), np.full(F, 0.002), np.zeros(F))
        r = analyze(y[fs:], fs)
        got = [p[0] for p in r["peaks"][:6]]
        print(f"wow and flutter RMS: {r['dev_rms']*100:.3f}% (set ~0.2%·|m|)")
        print(f"peaks: {[round(f,1) for f in sorted(got)]} (set to 4.2/12.7/33.0)")
        hits = sum(any(abs(g - ref) < 0.7 for g in got) for ref in (4.2, 12.7, 33.0))
        assert hits >= 2, "the modulator peaks were not recovered"
        print(f"OK: {hits}/3 of the expected peaks found")
        return

    from scipy.io import wavfile
    fs, x = wavfile.read(args.wav)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    if x.ndim > 1:
        x = x[:, 0]
    r = analyze(x, fs, args.f0)
    print(f"RMS speed deviation: {r['dev_rms']*100:.3f}%")
    print("wow and flutter peaks (Hz, RMS contribution):")
    for f, a in r["peaks"]:
        print(f"  {f:6.2f}  {a:.5f}")
    out = args.out or os.path.splitext(args.wav)[0] + "_pilot.npz"
    np.savez(out, **{k: v for k, v in r.items() if k != "peaks"},
             peaks=np.array(r["peaks"]))
    print("->", out, "(peaks+background — into WOW_PEAKS/NOISE_MIX of wowflutter.py, W-2)")


if __name__ == "__main__":
    main()
