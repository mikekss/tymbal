#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pqmf_design.py — design and verification of the PQMF bank for the N6 Neural Instrument.

K=4 subbands, N=128-tap prototype (=> 32-tap polyphase per branch, as in spec §5.1).
Cosine-modulated pseudo-QMF (near-perfect reconstruction):

    h_k[n] = 2*h[n]*cos( (2k+1) * pi/(2K) * (n - (N-1)/2) + (-1)^k * pi/4 )   (analysis)
    g_k[n] = 2*h[n]*cos( (2k+1) * pi/(2K) * (n - (N-1)/2) - (-1)^k * pi/4 )   (synthesis)

The prototype is a Kaiser-windowed sinc; the cutoff frequency fc is picked by a
1-D search over the empirical reconstruction error (this is more reliable than
analytic formulas and automatically accounts for the chosen sign conventions).

Output:
  - report: reconstruction SNR, alias rejection, delay;
  - pqmf_coeffs.h — synthesis/analysis coefficients (float32) for the firmware;
  - pqmf_golden.npz — golden vectors for the unit tests of the C implementation.

Run: python3 pqmf_design.py
"""

import numpy as np
from scipy.signal.windows import kaiser

FS = 48000
K = 4            # subbands
N = 128          # prototype length (=> synthesis delay (N-1)/2 ~ 1.32 ms @48k)
BETA = 10.06     # kaiser beta ~ 100 dB rejection


def prototype(fc: float, n_taps: int = N, beta: float = BETA) -> np.ndarray:
    """Kaiser-windowed lowpass prototype; fc is the normalized cutoff (cycles/sample)."""
    n = np.arange(n_taps)
    m = n - (n_taps - 1) / 2
    h = 2.0 * fc * np.sinc(2.0 * fc * m)
    h *= kaiser(n_taps, beta)
    return h / np.sum(h)          # DC gain normalization


def filterbank(h: np.ndarray, k_bands: int = K):
    """Returns (analysis[K,N], synthesis[K,N])."""
    n_taps = len(h)
    n = np.arange(n_taps)
    k = np.arange(k_bands)[:, None]
    arg = (2 * k + 1) * (np.pi / (2 * k_bands)) * (n - (n_taps - 1) / 2)
    phi = ((-1) ** k) * (np.pi / 4)
    ana = 2.0 * h * np.cos(arg + phi)
    syn = 2.0 * h * np.cos(arg - phi)
    return ana, syn


def analyze(x: np.ndarray, ana: np.ndarray) -> np.ndarray:
    """x[T] -> subbands sub[K, T//K] (convolution + decimation by K)."""
    t = len(x) // K * K
    x = x[:t]
    return np.stack([np.convolve(x, ana[k])[:t][::K] for k in range(K)])


def synthesize(sub: np.ndarray, syn: np.ndarray) -> np.ndarray:
    """sub[K, T/K] -> y[T] (interpolation by K + convolution + sum)."""
    t = sub.shape[1] * K
    y = np.zeros(t + N)
    for k in range(K):
        up = np.zeros(t)
        up[::K] = sub[k]
        y[: t + N - 1] += np.convolve(up, syn[k]) * K
    return y[:t]


def pr_snr(fc: float, beta: float = BETA, n_test: int = 1 << 15, seed: int = 0) -> float:
    """Empirical reconstruction SNR for (fc, beta) (white noise, edges discarded)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_test)
    ana, syn = filterbank(prototype(fc, N, beta))
    y = synthesize(analyze(x, ana), syn)
    d = N - 1                                 # full analysis+synthesis delay
    a, b = 4 * N, n_test - 4 * N
    err = y[a + d:b + d] - x[a:b]
    return 10 * np.log10(np.sum(x[a:b] ** 2) / np.sum(err ** 2))


def golden_search(lo: float, hi: float, beta: float, iters: int = 35):
    """Maximize SNR over fc by golden-section search at a fixed beta."""
    gr = (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc_, fd_ = pr_snr(c, beta), pr_snr(d, beta)
    for _ in range(iters):
        if fc_ > fd_:
            hi, d, fd_ = d, c, fc_
            c = hi - gr * (hi - lo)
            fc_ = pr_snr(c, beta)
        else:
            lo, c, fc_ = c, d, fd_
            d = lo + gr * (hi - lo)
            fd_ = pr_snr(d, beta)
    fc = (lo + hi) / 2
    return fc, pr_snr(fc, beta)


def search_2d():
    """Coarse grid over beta + golden section over fc. Peak ~ (beta 8.8, fc 0.0666)."""
    best = (BETA, 0.0625, -1.0)
    for beta in np.arange(8.0, 10.01, 0.2):
        fc, snr = golden_search(0.06, 0.075, beta)
        if snr > best[2]:
            best = (beta, fc, snr)
    return best


def alias_rejection(ana: np.ndarray) -> float:
    """Worst stopband rejection of the neighbouring band, dB (rough isolation estimate)."""
    worst = -np.inf
    f = np.fft.rfftfreq(1 << 16)
    for k in range(K):
        H = np.abs(np.fft.rfft(ana[k], 1 << 16))
        H /= H.max()
        # stopband: everything more than half a band away from its own band edges
        band_lo, band_hi = k / (2 * K), (k + 1) / (2 * K)
        guard = 1 / (2 * K) / 2
        stop = (f < band_lo - guard) | (f > band_hi + guard)
        if stop.any():
            worst = max(worst, 20 * np.log10(H[stop].max() + 1e-12))
    return worst


def export_header(ana: np.ndarray, syn: np.ndarray, fc: float, beta: float, path: str):
    def carr(name, m):
        rows = [", ".join(f"{v:+.9e}f" for v in row) for row in m]
        body = ",\n    ".join("{ " + r + " }" for r in rows)
        return f"static const float {name}[{m.shape[0]}][{m.shape[1]}] = {{\n    {body}\n}};\n"
    with open(path, "w") as f:
        f.write("/* Autogenerated by pqmf_design.py — do not edit by hand.\n")
        f.write(f" * K={K}, N={N}, fc={fc:.6f} (norm.), beta={beta:.2f}, fs={FS}.\n")
        f.write(" * Synthesis delay (N-1)/2 = %.2f ms; analysis+synthesis N-1 samples.\n */\n" % ((N - 1) / 2 / FS * 1e3))
        f.write(f"#define PQMF_BANDS {K}\n#define PQMF_TAPS {N}\n\n")
        f.write(carr("pqmf_analysis", ana))
        f.write("\n")
        f.write(carr("pqmf_synthesis", syn))


if __name__ == "__main__":
    beta, fc, snr = search_2d()
    ana, syn = filterbank(prototype(fc, N, beta))
    print(f"K={K}, N={N} (polyphase {N // K} tap/branch), beta={beta:.2f}")
    print(f"optimum fc = {fc:.6f} (norm., fs=1)  = {fc * FS:.1f} Hz @48k")
    print(f"reconstruction SNR (noise, analysis->synthesis): {snr:.1f} dB")
    print(f"worst out-of-band rejection: {alias_rejection(ana):.1f} dB")
    print(f"delay: analysis+synthesis {N - 1} samples = {(N - 1) / FS * 1e3:.2f} ms; "
          f"synthesis only ~{(N - 1) / 2 / FS * 1e3:.2f} ms (real N6 chain)")

    # golden vectors for the unit test of the C implementation
    rng = np.random.default_rng(42)
    x = rng.standard_normal(4096).astype(np.float64)
    sub = analyze(x, ana)
    y = synthesize(sub, syn)
    np.savez("pqmf_golden.npz", x=x, sub=sub, y=y, ana=ana, syn=syn, fc=fc, beta=beta)
    export_header(ana, syn, fc, beta, "pqmf_coeffs.h")
    print("written: pqmf_coeffs.h, pqmf_golden.npz")
