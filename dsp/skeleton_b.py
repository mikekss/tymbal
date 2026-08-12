#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skeleton_b.py — the skeleton, variant B: DIRECT render into the PQMF subbands
(no analysis). Phase P §2.2, decides D-6. The real chain: sub_b ->
pqmf.synthesize -> sound (synthesis-only delay 1.32 ms — the latency budget of
spec §5.4 is met at hop 4).

Contract (in addition to C-1..C-5 from skeleton_a, mirror it in C):
  B-1. The master phase is IDENTICAL to skeleton_a: the accumulator runs ON
       THE 48k GRID (Φ48 += 2πf0/48000, add-then-use, as in C-3), the renderer
       uses every 4th sample: Φ12[n] = Φ48[4n]. The reason is train/runtime
       consistency: the network input in training = analyze(render A) with
       phases in the ska convention; any other accumulator gives an
       f-dependent shift (2πf/48000 per one sample of misalignment) + drift
       from the integration scheme on a glissando — measured as −3 dB of phase
       divergence in the subband domain. Output n of the 12k grid corresponds
       to 48k index 4n (the analysis decimation), i.e.
       Φ12[n] = Φ48[4n] — the slice [0::4]. In cstyle it is a literal inner
       loop of increments (m ≤ 4n); in C the optimization "a group of 4 =
       4·f(middle)/48000" is allowed, but the group m=4n−3..4n crosses a frame
       boundary at i=0 — a special case is needed. The g table is calibrated
       in THIS same convention (the probe renders a tone via cumsum@48k).
       KEY FACT: after ↓4 decimation the visible frequency of a tone is THE
       SAME in all bands and equals fold12(f) — no sign alternation or
       heterodyning is required; the frequency placement is done by the
       synthesis filters themselves.
  B-2. The contribution of a harmonic to band k is a COMPLEX coefficient g_k(f):
         sub_k[n] += A · ( re_k(f)·sin θ + im_k(f)·cos θ ) · alt_k[n]
       g_k(f) was taken EMPIRICALLY: a 25 Hz grid of tones through the actual
       pq.analyze + lock-in on the shared e^{jθ} (see _probe_band_response) —
       amplitude AND phase, including the decimation conventions:
         sub_k[n] += A · ( re_k(f)·sin θ + im_k(f)·cos θ )
       Both band components at the seam read ONE θ_h — there is no beating.
       The table is cached in dsp/golden/pqmf_bandresp.npz (commit it: C
       uses the same table).
  B-5. We neglect the bank's true alias residue (≤ −30 dB in the transition
       zones at the 6/12/18 kHz seams, outside them at the stopband level of
       −97 dB): the PQMF alias cancellation between bands is not reproduced
       in the B render (there is no second component). The acceptance
       threshold is listening test T1; if needed, a second lock-in at the
       mirrored frequency is added.
       Both band components read ONE θ_h — a crossfade with no beating.
       |g| < W_EPS — the band is not rendered. Linear interpolation of re/im
       over the grid (complex interpolation — safe against phase wraps).
  B-3. Noise: per band, xorshift32, seed_k = seed XOR BAND_SEED[k] (C-4);
       level 0.5·0.3·tB — energetically equal to the A path (analysis of white
       noise 0.3·u loses half the RMS per band). The A/B noise implementations
       differ — the contract is on spectrum and level, not on bits.
  B-4. The total delay of the B path EQUALS the A path (N−1): the phase of the
       g table already carries the group delay of the analysis filter (63.5),
       and synthesis adds its own 63.5.
       The B carriers ≡ analyze(A) down to the phases (checked by lock-in:
       Δψ=0.0000). THE EXCEPTION is the envelope: amp/gate are applied after
       the table and are NOT delayed => attacks in B are ~1.3 ms earlier than
       in analyze(A). For the instrument this is a plus (a harder attack); for
       train/runtime consistency it is a 1.3 ms shift of the conditioning at
       an RF of 21–42 ms, judged negligible; if wanted, the corpus generator
       can delay the control curves by 63 samples at 48k.

Selftest: (1) vec == cstyle; (2) D-6: A path vs B path, magnitude STFT;
(3) seam: gliss 5700->6300, envelope ripple; (4) perf.
Run: python3 skeleton_b.py
"""
import os
import time
import numpy as np

import pqmf_design as pq
import skeleton_a as ska

FS48 = 48000
FSB = 12000
HOPB = 48
BANDS = 4
NH = ska.NH
W_EPS = 1e-4
NOISE_B = 0.5 * 0.3
BAND_SEED = (0x00000000, 0x9E3779B9, 0x3C6EF372, 0xDAA66D2B)
GRID_HZ = 25.0
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden", "pqmf_bandresp.npz")

_h = pq.prototype(ska.FC_OPT, beta=ska.BETA_OPT)
_ANA, _SYN = pq.filterbank(_h)


# ------------------------------------------------ empirical complex response
def _probe_band_response(force=False):
    """g[b, f] = complex coefficient: analyze(sin θ)|band b ≡
    Im{g·e^{jθ12}}·alt_b. GRID_HZ grid, cached in golden/pqmf_bandresp.npz."""
    if (not force) and os.path.exists(_CACHE):
        z = np.load(_CACHE)
        if "note" in z and z["grid_hz"] == GRID_HZ and z["fc"] == ska.FC_OPT:
            return z["fgrid"], z["g"]
    n48 = 12000                                    # 0.25 s
    n12 = n48 // 4
    t = np.arange(n48)
    m0, m1 = 300, n12 - 300                        # steady-state window
    fgrid = np.arange(0.0, 24000.0 + GRID_HZ, GRID_HZ)
    g = np.zeros((BANDS, len(fgrid)), dtype=np.complex128)
    for i, f in enumerate(fgrid):
        th = np.cumsum(np.full(n48, 2.0 * np.pi * f / FS48))   # ska convention
        sub = pq.analyze(np.sin(th), _ANA)         # (4, n12)
        e = np.exp(-1j * th[::4][m0:m1])
        for b in range(BANDS):
            g[b, i] = 2j * np.mean(sub[b, m0:m1] * e)
    # degenerate nodes (f a multiple of 6000: Nyquist/DC of the 12k grid — the
    # lock-in collapses the quadrature and doubles |g|) — average of neighbours
    for i, f in enumerate(fgrid):
        if abs(f % 6000.0) < 1e-6 or abs(f % 6000.0 - 6000.0) < 1e-6:
            lo = max(i - 1, 0)
            hi = min(i + 1, len(fgrid) - 1)
            g[:, i] = 0.5 * (g[:, lo] + g[:, hi])
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    np.savez(_CACHE, fgrid=fgrid, g=g, grid_hz=GRID_HZ, fc=ska.FC_OPT,
             beta=ska.BETA_OPT, note="v5: ska phase convention (cumsum@48k); m*6000 nodes interpolated")
    return fgrid, g


_FGRID, _G = _probe_band_response()
_GRE = np.ascontiguousarray(_G.real)
_GIM = np.ascontiguousarray(_G.imag)
_NMAX = len(_FGRID) - 1


def band_coeff(f):
    """f [...] Hz -> (re, im) of shape (BANDS, ...) — linear interp. of the table."""
    idx = np.clip(f / GRID_HZ, 0.0, _NMAX - 1e-9)
    i0 = idx.astype(np.int64)
    fr = idx - i0
    re = _GRE[:, i0] * (1.0 - fr) + _GRE[:, i0 + 1] * fr
    im = _GIM[:, i0] * (1.0 - fr) + _GIM[:, i0 + 1] * fr
    return re, im


# ---------------------------------------------------------------- render (vec)
def render_voice_b(f0_hz, amp, tA, tB, gate, seed=0xC0FFEE,
                   decoder=ska.decoder_slope):
    """Vector render of a voice straight into subbands. Returns: sub (4, F*HOPB) @12k."""
    F = len(f0_hz)
    n = F * HOPB
    f0g, ampg, tAg, tBg, gg = map(ska._frames_pad, (f0_hz, amp, tA, tB, gate))

    k = np.arange(n) // HOPB
    fr = (np.arange(n) % HOPB) / HOPB

    def interp(v):                                   # C-1 on the 12k grid
        return v[k] + (v[k + 1] - v[k]) * fr

    f0_s = interp(f0g)
    ag_s = interp(ampg * gg)                         # C-5
    tB_s = interp(tBg)

    # B-1: master phase on the 48k grid (bit-exact with skeleton_a), take [::4]
    n48 = n * 4
    k48 = np.arange(n48) // (HOPB * 4)
    fr48 = (np.arange(n48) % (HOPB * 4)) / (HOPB * 4)
    f0_48 = f0g[k48] + (f0g[k48 + 1] - f0g[k48]) * fr48
    phi = np.mod(np.cumsum(2.0 * np.pi * f0_48 / FS48), 2.0 * np.pi)[0::4]

    A = decoder(f0g, ampg, tAg, tBg)                 # C-2
    A_s = A[k] + (A[k + 1] - A[k]) * fr[:, None]     # (n, NH)

    h_idx = np.arange(1, NH + 1)
    th = np.outer(phi, h_idx)                        # (n, NH)
    s_th = np.sin(th)
    c_th = np.cos(th)
    fh = np.outer(f0_s, h_idx)

    sub = np.zeros((BANDS, n))
    re, im = band_coeff(fh)                          # (BANDS, n, NH)
    for b in range(BANDS):
        mask = (re[b] ** 2 + im[b] ** 2) > W_EPS * W_EPS
        contrib = ((s_th * np.where(mask, re[b], 0.0)
                    + c_th * np.where(mask, im[b], 0.0)) * A_s).sum(axis=1)
        sub[b] = contrib * ag_s

    for b in range(BANDS):                           # B-3
        u = ska.xorshift32_block(seed ^ BAND_SEED[b], n)
        sub[b] += NOISE_B * u * tB_s * ag_s
    return sub


# ---------------------------------------------------- render ("the C transcript")
def render_voice_b_cstyle(f0_hz, amp, tA, tB, gate, seed=0xC0FFEE,
                          decoder=ska.decoder_slope):
    F = len(f0_hz)
    f0g, ampg, tAg, tBg, gg = map(ska._frames_pad, (f0_hz, amp, tA, tB, gate))
    A = decoder(f0g, ampg, tAg, tBg)
    out = np.zeros((BANDS, F * HOPB))
    phi = 0.0
    m_next = 0                     # next 48k index to be incremented (B-1)
    hop48 = HOPB * 4
    seeds = [np.uint32((seed ^ BAND_SEED[b]) or 1) for b in range(BANDS)]
    for kk in range(F):
        ag0 = ampg[kk] * gg[kk]
        ag1 = ampg[kk + 1] * gg[kk + 1]
        df = f0g[kk + 1] - f0g[kk]
        for i in range(HOPB):
            n = kk * HOPB + i
            fr = i / HOPB
            f0 = f0g[kk] + df * fr
            ag = ag0 + (ag1 - ag0) * fr
            tb = tBg[kk] + (tBg[kk + 1] - tBg[kk]) * fr
            # B-1: Φ12[n] = Φ48[4n] — catch up the 48k increments to m = 4n
            while m_next <= 4 * n:
                kk48 = m_next // hop48
                fr48 = (m_next % hop48) / hop48
                f48 = f0g[kk48] + (f0g[kk48 + 1] - f0g[kk48]) * fr48
                phi = (phi + 2.0 * np.pi * f48 / FS48) % (2.0 * np.pi)
                m_next += 1
            acc = [0.0] * BANDS
            for h in range(1, NH + 1):
                a = A[kk, h - 1] + (A[kk + 1, h - 1] - A[kk, h - 1]) * fr
                if a <= 0.0:
                    continue
                s = np.sin(h * phi)
                c = np.cos(h * phi)
                idx = h * f0 / GRID_HZ
                if idx > _NMAX - 1e-9:
                    idx = _NMAX - 1e-9
                i0 = int(idx)
                dfr = idx - i0
                for b in range(BANDS):
                    wre = _GRE[b, i0] + (_GRE[b, i0 + 1] - _GRE[b, i0]) * dfr
                    wim = _GIM[b, i0] + (_GIM[b, i0 + 1] - _GIM[b, i0]) * dfr
                    if wre * wre + wim * wim > W_EPS * W_EPS:
                        acc[b] += (wre * s + wim * c) * a
            for b in range(BANDS):
                s32 = seeds[b]
                s32 ^= np.uint32(s32 << np.uint32(13))
                s32 ^= np.uint32(s32 >> np.uint32(17))
                s32 ^= np.uint32(s32 << np.uint32(5))
                seeds[b] = s32
                u = np.float64(np.int32(s32)) * 2.0 ** -31
                out[b, n] = acc[b] * ag + NOISE_B * u * tb * ag
    return out


# ------------------------------------------------------------------ selftest
def _stft_mag(x, nfft=1024, hop=256):
    frames = 1 + (len(x) - nfft) // hop
    w = np.hanning(nfft)
    return np.abs(np.stack([np.fft.rfft(x[i * hop:i * hop + nfft] * w)
                            for i in range(frames)]))


if __name__ == "__main__":
    print(f"[table]    complex response of the bank: {_G.shape[1]} points x {BANDS} "
          f"bands, grid {GRID_HZ:.0f} Hz (cache golden/pqmf_bandresp.npz)")

    # --- 1) contract vec == cstyle (0.5 s, harmonics across 6 kHz) ---------
    F = 125
    f0 = np.exp(np.linspace(np.log(1000.0), np.log(2000.0), F))
    amp = np.full(F, 0.5)
    tA = 0.5 + 0.5 * np.sin(np.linspace(0, np.pi, F))
    tB = np.full(F, 0.2)
    gate = np.ones(F); gate[:2] = 0; gate[-2:] = 0

    t0 = time.time(); sb_vec = render_voice_b(f0, amp, tA, tB, gate)
    t_vec = time.time() - t0
    t0 = time.time(); sb_c = render_voice_b_cstyle(f0, amp, tA, tB, gate)
    t_c = time.time() - t0
    err = np.max(np.abs(sb_vec - sb_c)); ref = np.max(np.abs(sb_vec))
    print(f"[contract] max|vec - cstyle| = {err:.3e} (signal max {ref:.3f})")
    assert err < 1e-9 * max(ref, 1.0), "vec and cstyle diverged!"

    # --- 2) D-6: A path vs B path, magnitude STFT (2 s, 100->8000) ----------
    F2 = 500
    f0g = np.exp(np.linspace(np.log(100.0), np.log(8000.0), F2))
    ampg = np.full(F2, 0.5)
    tAg = 0.5 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, F2))
    tBg = np.zeros(F2)
    gg = np.ones(F2); gg[:2] = 0; gg[-2:] = 0

    audioA = ska.render_voice(f0g, ampg, tAg, tBg, gg)
    yA = pq.synthesize(pq.analyze(audioA, _ANA), _SYN)   # delay N-1
    subB = render_voice_b(f0g, ampg, tAg, tBg, gg)
    yB = pq.synthesize(subB, _SYN)                       # also N-1 (B-4)

    a, b = 4 * pq.N, min(len(yA), len(yB)) - 4 * pq.N
    MA = _stft_mag(yA[a:b]); MB = _stft_mag(yB[a:b])
    err_db = 20 * np.log10(np.linalg.norm(MA - MB) / np.linalg.norm(MA) + 1e-12)
    print(f"[D-6]      magnitude STFT error A vs B: {err_db:.1f} dB rel. "
          f"(RMS A {np.sqrt(np.mean(yA[a:b]**2)):.4f}, "
          f"B {np.sqrt(np.mean(yB[a:b]**2)):.4f})")
    # wav for the listening test (D-6 is decided by ear)
    from scipy.io import wavfile
    adir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audition")
    os.makedirs(adir, exist_ok=True)
    for nm, y in (("d6_A_analyze_path.wav", yA), ("d6_B_direct_path.wav", yB)):
        wavfile.write(os.path.join(adir, nm), FS48,
                      (np.clip(y, -1, 1) * 32767).astype(np.int16))
    print(f"[listening] dsp/audition/d6_A_analyze_path.wav, d6_B_direct_path.wav")

    # --- 3) seam: tone 5700->6300, envelope ripple -------------------------
    F3 = 500
    f0s = np.linspace(5700.0, 6300.0, F3)
    ones = np.ones(F3); z = np.zeros(F3)
    def dec_pure(f0g_, ampg_, tAg_, tBg_):
        A = np.zeros((len(f0g_), NH)); A[:, 0] = 1.0
        return A
    subS = render_voice_b(f0s, ones * 0.5, z, z, ones, decoder=dec_pure)
    yS = pq.synthesize(subS, _SYN)
    seg = yS[4 * pq.N:-4 * pq.N]
    env = np.sqrt(np.convolve(seg ** 2, np.ones(480) / 480, mode="valid"))
    env = env[480:-480]
    ripple_db = 20 * np.log10(env.max() / max(env.min(), 1e-12))
    print(f"[seam]     gliss 5700->6300: envelope ripple {ripple_db:.2f} dB")

    print(f"[perf]     vec: {t_vec:.2f} s for 0.5 s of audio (~{t_vec/0.5:.2f}x RT); "
          f"cstyle: {t_c:.1f} s")
    print(f"[sanity]   peak of bands {np.max(np.abs(sb_vec)):.3f}, "
          f"NaN: {np.isnan(sb_vec).any()}")
    print("OK: variant B is ready; the D-6 decision — from the numbers above + listening test")
