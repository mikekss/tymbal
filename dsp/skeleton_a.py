#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skeleton_a.py — the skeleton reference, variant A (render at 48 kHz -> PQMF analysis).
Phase P of the pipeline (§2.2 of the guide). This is NOT just a generator — it is
the CONTRACT with C: the `--selftest` block renders the same sound with a literal
scalar hop loop ("the C transcript") and cross-checks it against the vectorized
version down to ~1e-6.

Contract decisions (mirror them in C one for one):
  C-1. All control values — INCLUDING the harmonic amplitude vector A[h]
       and the Nyquist taper — live on the 250 Hz frame grid and are linearly
       interpolated up to samples: v(n) = v[k] + (v[k+1]-v[k]) * (n%hop)/hop,
       k = n//hop. n_frames+1 frames are needed (the last one is duplicated).
       In C this is (start, delta) per hop — exactly one scheme for everything.
  C-2. The timbre decoder is a function of frame controls -> A[h], called at 250 Hz.
       What is inside it (tilt, key spectra, whatever) is irrelevant to the
       contract; the decoder can be changed without touching the renderer.
  C-3. One master phase per voice: phi += 2*pi*f0/fs (mod 2*pi), harmonic h
       reads sin(h*phi). mod 2*pi is LEGAL for integer h (h*2*pi*k is a multiple
       of 2*pi) and keeps precision. In C: uint32 phi += (uint32)round(f0/fs * 2^32),
       harmonic phase = h*phi (the natural wrap == mod 2*pi), a 4096 table
       + linear interpolation.
  C-4. Noise is xorshift32, uniform [-1,1): u = (int32)state * 2^-31.
       The same sequence in Python and in C for the same seed.
       Gaussianity is not needed: spectrally white is white.
  C-5. gate is interpolated AS PART OF the product amp*gate (same as C-1) —
       attack/release get a free 4 ms declick, identical on both
       sides.

Nyquist: not a binary mask but a 22->24 kHz taper on the frame grid (C-1);
a hard mask produces an amplitude step when a glissando crosses the boundary.
Normalization: sum(A) == 1 before multiplying by amp — loudness does not depend
on tA or on the number of live harmonics (otherwise an upward gliss "thins out"
in level).
"""
import time
import numpy as np

import pqmf_design as pq

FS = 48000
HOP = 192                    # 4 ms @48k; 250 Hz frames
NH = 100                     # harmonics, maximum
FC_OPT, BETA_OPT = 0.066603, 8.80   # from the pqmf_design 2D search (SNR 67.5 dB)
TAPER_LO, TAPER_HI = 22000.0, 24000.0


# ---------------------------------------------------------------- xorshift32
def xorshift32_block(seed: int, n: int) -> np.ndarray:
    """Exactly the same sequence must come out in C:
       s^=s<<13; s^=s>>17; s^=s<<5;  output: (int32)s * 2**-31  -> [-1,1)."""
    s = np.uint32(seed if seed else 1)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s ^= np.uint32(s << np.uint32(13))
        s ^= np.uint32(s >> np.uint32(17))
        s ^= np.uint32(s << np.uint32(5))
        out[i] = np.float64(np.int32(s)) * 2.0 ** -31
    return out


# ------------------------------------------------------- timbre decoder (C-2)
def decoder_slope(f0, amp, tA, tB):
    """v0 decoder: spectral tilt 1/h^(3-2*tA) + Nyquist taper.
    Input: frame arrays (n_frames+1,). Output A: (n_frames+1, NH),
    normalized to sum(A)=1 per frame. The key-spectra decoder will later
    slot in here too — the signature will not change."""
    h = np.arange(1, NH + 1)[None, :]                    # (1, NH)
    p = (3.0 - 2.0 * tA)[:, None]                        # (F, 1)
    A = h ** (-p)
    fh = f0[:, None] * h                                 # harmonic frequency
    taper = np.clip((TAPER_HI - fh) / (TAPER_HI - TAPER_LO), 0.0, 1.0)
    A = A * taper
    norm = A.sum(axis=1, keepdims=True)
    return A / np.maximum(norm, 1e-12)


# --------------------------------------------------------------- render (vec)
def _frames_pad(v):
    """n_frames -> n_frames+1: the last frame is held (C-1)."""
    return np.concatenate([v, v[-1:]])


def _tilt_noise(noise, tilt_db):
    """D-23: noise tilt across the PQMF subbands, dB PER BAND.

    The K=4 bands cover 0-6, 6-12, 12-18, 18-24 kHz; the weight of band b =
    10^(-tilt*b/20). tilt=0 -> all weights are 1.0 and the function returns the
    input BYTE FOR BYTE: canon, golden and the corpora stay as they were.

    Why this change: right now the noise is white and the same in all bands,
    while the harmonics fall off with frequency, so up top the noise nearly
    equals the tone (measured 7 Aug: -2.4 dB above 4 kHz at timbreB=0.15).
    The tilt must remove the excess sand AND pull the spectral tilt towards
    the teacher's.
    """
    if tilt_db == 0.0:
        return noise
    n = len(noise)
    S = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    band = np.clip((f / (FS / 8.0)).astype(int), 0, 3)   # 4 bands of 6 kHz each
    S *= (10.0 ** (-tilt_db * band / 20.0))
    return np.fft.irfft(S, n)


def render_voice(f0_hz, amp, tA, tB, gate, seed=0xC0FFEE, decoder=decoder_slope,
                 noise_tilt_db=0.0):
    """Vectorized render of one voice, 48 kHz. All inputs are (n_frames,)
    at 250 Hz. Returns audio (n_frames*HOP,)."""
    F = len(f0_hz)
    n = F * HOP
    f0g, ampg, tAg, tBg, gg = map(_frames_pad, (f0_hz, amp, tA, tB, gate))

    k = np.arange(n) // HOP                              # frame index
    fr = (np.arange(n) % HOP) / HOP                      # fraction inside the hop

    def interp(v):                                       # C-1
        return v[k] + (v[k + 1] - v[k]) * fr

    f0_s = interp(f0g)
    ag_s = interp(ampg * gg)                             # C-5
    tB_s = interp(tBg)

    # master phase (C-3)
    phi = np.cumsum(2.0 * np.pi * f0_s / FS)
    phi = np.mod(phi, 2.0 * np.pi)

    # harmonic amplitudes on the frame grid -> interpolation (C-1/C-2)
    A = decoder(f0g, ampg, tAg, tBg)                     # (F+1, NH)
    A_s = A[k] + (A[k + 1] - A[k]) * fr[:, None]         # (n, NH)

    harm = np.sin(np.outer(phi, np.arange(1, NH + 1))) * A_s
    audio = harm.sum(axis=1) * ag_s

    noise = _tilt_noise(xorshift32_block(seed, n), noise_tilt_db)   # C-4 (+D-23)
    audio = audio + 0.3 * noise * tB_s * ag_s
    return audio


# ---------------------------------------------------- render ("the C transcript")
def render_voice_cstyle(f0_hz, amp, tA, tB, gate, seed=0xC0FFEE,
                        decoder=decoder_slope):
    """A literal scalar hop loop — this is how it will be written in C.
    Exists for the selftest: it proves the vectorized version IS the contract."""
    F = len(f0_hz)
    f0g, ampg, tAg, tBg, gg = map(_frames_pad, (f0_hz, amp, tA, tB, gate))
    A = decoder(f0g, ampg, tAg, tBg)
    out = np.zeros(F * HOP)
    phi = 0.0
    s = np.uint32(seed if seed else 1)
    for kk in range(F):                                  # over hops
        ag0 = ampg[kk] * gg[kk]
        ag1 = ampg[kk + 1] * gg[kk + 1]
        for i in range(HOP):                             # over samples
            fr = i / HOP
            f0 = f0g[kk] + (f0g[kk + 1] - f0g[kk]) * fr
            ag = ag0 + (ag1 - ag0) * fr
            tb = tBg[kk] + (tBg[kk + 1] - tBg[kk]) * fr
            phi = (phi + 2.0 * np.pi * f0 / FS) % (2.0 * np.pi)
            acc = 0.0
            for h in range(1, NH + 1):
                a = A[kk, h - 1] + (A[kk + 1, h - 1] - A[kk, h - 1]) * fr
                if a > 0.0:
                    acc += np.sin(h * phi) * a
            s ^= np.uint32(s << np.uint32(13))
            s ^= np.uint32(s >> np.uint32(17))
            s ^= np.uint32(s << np.uint32(5))
            u = np.float64(np.int32(s)) * 2.0 ** -31
            out[kk * HOP + i] = acc * ag + 0.3 * u * tb * ag
    return out


# --------------------------------------------------------------------- PQMF
_h = pq.prototype(FC_OPT, beta=BETA_OPT)
_ANA, _SYN = pq.filterbank(_h)


def render_and_analyze(f0_hz, amp, tA, tB, gate, seed=0xC0FFEE):
    audio = render_voice(f0_hz, amp, tA, tB, gate, seed)
    sub = pq.analyze(audio, _ANA)                        # (4, n/4) @12 kHz
    return audio, sub


# ------------------------------------------------------------------ selftest
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    F = 250 * 2                                          # 2 s

    # glissando 100->8000 Hz: crosses the 6 kHz boundaries and runs into the
    # taper of the top harmonics; tA sweeps, tB breathes, gate with attack/release
    f0 = np.exp(np.linspace(np.log(100.0), np.log(8000.0), F))
    amp = np.full(F, 0.5)
    tA = 0.5 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, F))
    tB = np.full(F, 0.2)
    gate = np.ones(F); gate[:2] = 0; gate[-2:] = 0

    t0 = time.time()
    a_vec = render_voice(f0, amp, tA, tB, gate)
    t_vec = time.time() - t0

    t0 = time.time()
    a_c = render_voice_cstyle(f0, amp, tA, tB, gate)
    t_c = time.time() - t0

    err = np.max(np.abs(a_vec - a_c))
    ref = np.max(np.abs(a_vec))
    print(f"[contract] max|vec - cstyle| = {err:.3e} (signal max {ref:.3f})")
    assert err < 1e-9 * max(ref, 1.0), "the vector and C versions diverged!"

    audio, sub = render_and_analyze(f0, amp, tA, tB, gate)
    y = pq.synthesize(sub, _SYN)
    d = pq.N - 1                                         # analysis+synthesis delay
    a, b = 4 * pq.N, len(audio) - 4 * pq.N
    snr = 10 * np.log10(np.sum(audio[a:b] ** 2)
                        / (np.sum((y[a + d:b + d] - audio[a:b]) ** 2) + 1e-30))
    print(f"[pqmf]     analysis->synthesis on the skeleton: SNR = {snr:.1f} dB, "
          f"bands {sub.shape}")

    print(f"[perf]     vec: {t_vec:.2f} s for 2 s of audio "
          f"(~{t_vec/2:.2f}x RT), cstyle: {t_c:.1f} s")
    print(f"[sanity]   peak {np.max(np.abs(audio)):.3f}, "
          f"NaN: {np.isnan(audio).any()}")
    print("OK: Python<->C contract confirmed on a glissando across band boundaries")
