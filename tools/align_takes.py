#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_takes.py — alignment of a tape take against the skeleton (guide §4.3, canon
F-3): the τ(t) of a take is extracted FROM THE TAKE ITSELF; the recording is
resampled onto a uniform grid (removing wow and flutter) for the training pairs.

The method is TWO-STAGE (a single-stage waveform cross-correlation on
quasi-periodic material is ambiguous by a period — checked on a chirp, the
tracker wanders off by hundreds of samples):
  1) COARSE: correlation of the ENVELOPES (|x| -> LP 40 Hz, decimation x16): the
     envelope is aperiodic, there is no cycle ambiguity; 250 ms windows, 60 ms step,
     search ±SEARCH_MS -> a smooth τ_coarse(t).
  2) FINE: waveform correlation, 50 ms window, 10 ms step, search ±FINE_MS around
     τ_coarse (less than the half-period of the bass — the cycle was picked by the
     coarse stage), parabola over the peak -> sub-sample τ(t).
Inversion: t_rec = t + τ(t), resampling with a cubic spline.
Check: the residual |τ| over windows < 1 sample, otherwise the take is rejected (§4.3.3).

Run:     python3 align_takes.py skeleton.wav take.wav --out aligned.wav
Smoke:   python3 align_takes.py --selftest

STATUS (16 Jul, honestly): the frame (coarse-by-envelope -> fine-by-waveform + QC
with rejection) is correct by construction, but the synthetic smoke test on a
pathologically periodic chirp does NOT converge (<1 sample): the coarse stage is
off by more than the warp itself. Until there are real takes (T1) treat this tool
as a DRAFT.
Directions for whoever picks it up: (1) GCC-PHAT instead of plain correlation
(spectral whitening kills the periodic ambiguity); (2) a coarse stage on ONSETS
(the envelope derivative); (3) an iterative coarse->fine->coarse pass;
(4) tuning of windows/radii on real tape material.
Protection against garbage is already there: resid_max > 1 sample => the take is
rejected (§4.3.3).
"""
import argparse
import os
import numpy as np
import scipy.signal as sig
from scipy.interpolate import CubicSpline

SEARCH_MS = 15.0
FINE_MS = 1.25                        # < half-period of a 50 Hz bass (10 ms/2)
DEC = 16


def _envelope(x, fs):
    sos = sig.butter(4, 40.0, fs=fs, output="sos")
    e = sig.sosfilt(sos, np.abs(x))
    return e[::DEC]


def _corr_tau(a, b, rad):
    """b has length len(a)+2*rad; returns (τ, quality), parabola over the peak.
    The mean is subtracted: in envelopes the DC dominates and without this the
    correlation peak is flat (the coarse stage gave errors larger than the warp)."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    c = sig.correlate(b, a, mode="valid")
    i = int(np.argmax(c))
    d = 0.0
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
    q = c[i] / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-30)
    return (i + d) - rad, q


def tau_coarse(skel, take, fs):
    es, et = _envelope(skel, fs), _envelope(take, fs)
    fs_d = fs / DEC
    win = int(0.25 * fs_d)
    step = int(0.06 * fs_d)
    rad = max(2, int(SEARCH_MS * 1e-3 * fs_d))
    cs, ts, qs = [], [], []
    pos = rad
    n = min(len(es), len(et))
    while pos + win + rad < n:
        tau, q = _corr_tau(es[pos:pos + win], et[pos - rad:pos + win + rad], rad)
        cs.append((pos + win // 2) * DEC)
        ts.append(tau * DEC)
        qs.append(q)
        pos += step
    cs, ts, qs = map(np.array, (cs, ts, qs))
    ok = qs > 0.5 * np.median(qs[qs > 0])
    if ok.sum() < 4:
        raise RuntimeError("the envelopes do not correlate — the take is rejected")
    t_sm = sig.medfilt(ts[ok], 5)
    return cs[ok], t_sm


def tau_fine(skel, take, fs, c_coarse, t_coarse):
    win = int(0.05 * fs)
    step = int(0.01 * fs)
    rad = int(FINE_MS * 1e-3 * fs)
    n = min(len(skel), len(take))
    cs, ts, qs = [], [], []
    pos = win
    while pos + win < n:
        base = int(round(np.interp(pos, c_coarse, t_coarse)))
        lo, hi = pos + base - rad, pos + win + base + rad
        if lo >= 0 and hi <= n:
            tau, q = _corr_tau(skel[pos:pos + win], take[lo:hi], rad)
            cs.append(pos + win // 2)
            ts.append(base + tau)
            qs.append(q)
        pos += step
    return np.array(cs), np.array(ts), np.array(qs)


def align(skel, take, fs):
    cc, tc = tau_coarse(skel, take, fs)
    c, tau, q = tau_fine(skel, take, fs, cc, tc)
    ok = q > 0.2 * np.median(q[q > 0])
    if ok.sum() < 8:
        raise RuntimeError("too few correlating windows — the take is rejected")
    ts = sig.medfilt(tau[ok], 5)
    if len(ts) > 21:
        ts = sig.savgol_filter(ts, 21, 2)
    n = min(len(skel), len(take))
    t = np.arange(n, dtype=np.float64)
    tau_t = np.interp(t, c[ok], ts)                  # const. edges (no extrapolation)
    aligned = CubicSpline(t, take[:n])(np.clip(t + tau_t, 0, n - 1))
    # check: the fine stage on the aligned signal, base 0
    c2, tau2, q2 = tau_fine(skel, aligned, fs, np.array([0, n]), np.array([0, 0]))
    ok2 = q2 > 0.2 * np.median(q2[q2 > 0])
    resid = float(np.max(np.abs(tau2[ok2]))) if ok2.any() else np.inf
    return aligned, dict(tau_c=c, tau=tau, qual=q, resid_max=resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skeleton", nargs="?")
    ap.add_argument("take", nargs="?")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        fs = 48000
        rng = np.random.default_rng(3)
        n = 6 * fs
        t = np.arange(n)
        f = np.exp(np.linspace(np.log(150), np.log(3000), n))
        env = 1.0 + 0.5 * np.sin(2 * np.pi * 2.1 * t / fs) \
            * sig.square(2 * np.pi * 0.9 * t / fs + 1)       # phrase envelope
        x = 0.35 * np.sin(2 * np.pi * np.cumsum(f) / fs) * np.clip(env, 0, 2)
        tau_true = 25 * np.sin(2 * np.pi * 0.7 * t / fs) \
            + 12 * np.sin(2 * np.pi * 2.3 * t / fs + 1)
        take = CubicSpline(t, x)(np.clip(t + tau_true, 0, n - 1))
        take = np.tanh(1.5 * take) + 0.003 * rng.standard_normal(n)
        aligned, rep = align(x, take, fs)
        print(f"residual |τ| after alignment: {rep['resid_max']:.3f} samples "
              f"(criterion < 1; the warp was up to 37)")
        if rep["resid_max"] < 1.0:
            print("OK: the two-stage alignment removed the warp")
        else:
            print("WARNING: the smoke test does NOT converge — draft tool, see STATUS "
                  "in the header; tune it on real takes (the QC rejection protects us)")
        return

    from scipy.io import wavfile
    fs1, skel = wavfile.read(args.skeleton)
    fs2, take = wavfile.read(args.take)
    assert fs1 == fs2, "different fs"

    def norm(a):
        if a.dtype not in (np.float32, np.float64):
            a = a.astype(np.float64) / np.iinfo(a.dtype).max
        return a[:, 0] if a.ndim > 1 else a
    skel, take = norm(skel), norm(take)
    aligned, rep = align(skel, take, fs1)
    out = args.out or os.path.splitext(args.take)[0] + "_aligned.wav"
    wavfile.write(out, fs1, aligned.astype(np.float32))
    ok = rep["resid_max"] < 1.0
    print(f"-> {out}; residual |τ|max = {rep['resid_max']:.3f} samples — "
          f"{'OK' if ok else 'REJECT (>1 sample — throw it out, §4.3.3)'}")
    np.savez(os.path.splitext(out)[0] + "_tau.npz",
             tau_c=rep["tau_c"], tau=rep["tau"], qual=rep["qual"],
             resid_max=rep["resid_max"])


if __name__ == "__main__":
    main()
