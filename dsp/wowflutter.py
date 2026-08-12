#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wowflutter.py — wow and flutter + hiss, explicit M55 models (guide §2.3).
Full band 48 kHz, AFTER PQMF synthesis (canon F-1). Python reference + C transcript.

Contracts (W-1..W-5, mirror them in C):
  W-1. Delay ring: a physical "read speed" model v(t): the lag
       L(t+1) = L(t) + (1 − v(t)) (in 48k samples), read position
       r = n − L0 − L(t), fractional read by LINEAR interpolation (v0;
       Lagrange-3/allpass is the upgrade if the HF modulation becomes
       audible). The ring is RING_S seconds (0.5–1, dictated by tape-stop,
       §2.3); the offline reference uses a straight buffer — in C a ring
       with a mask, the semantics are the same.
  W-2. Speed modulator: v(t) = v_macro(t)·(1 + depth(t)·m(t)),
       m(t) = Σ aᵢ·sin(2π fᵢ n/fs + φᵢ) + w_lp(t): peaks (rotors) + background.
       w_lp is xorshift32 noise through a one-pole (POLE), normalized to RMS=1.
       The default parameters are a STUB until the characterization session
       §4.2(b); the structure and the contract do not change when the
       parameters are replaced.
  W-3. Macros on top of the stochastics: v_macro(t) is a 250 Hz frame curve
       (C-1), varispeed = a ramp inside that curve. tape-stop policy (F-2):
       v_macro < V_EPS => fade to silence over FADE_MS and RESET the lag to
       nominal; restart (v_macro >= V_EPS) => fade-in. Otherwise the next
       gesture starts with an exhausted ring.
  W-4. Hiss: xorshift32 → a cascade of biquads (replacing the FIR, §2.3) →
       level hiss_lvl(t) (250 Hz frames). It is added AT THE INPUT of the
       delay line: tape noise wows and flutters along with the program
       (physics of the medium).
       The biquad coefficients are a stub for a "pinkish" slope until §4.2(a).
  W-5. All control curves are 250 Hz frames, linearly interpolated up to
       samples (C-1); the lag and the modulator phase run on the 48k grid.

Selftest: (1) vec == cstyle (wow+flutter+varispeed+tape-stop+hiss);
(2) wow and flutter spectrum of a 3150 Hz tone (Hilbert) — the peaks are in place;
(3) tape-stop: silence during the stop, the lag is reset; (4) hiss spectrum ~ the biquads.
Run: python3 wowflutter.py
"""
import numpy as np

import skeleton_a as ska

FS = 48000
HOP = 192                      # 250 Hz frames (C-1)
RING_S = 1.0
L0 = 24                        # nominal read lag, samples (> max depth)
V_EPS = 1e-3                   # tape-stop "stopped" threshold (W-3)
FADE_MS = 30.0
POLE = 0.995                   # one-pole of the wow/flutter background (W-2)

# stubs until characterization (§4.2) — the contract structure does not depend on them
WOW_PEAKS = ((4.2, 1.0, 0.3), (12.7, 0.45, 1.7), (33.0, 0.2, 4.1))  # (Hz, amp, phase)
NOISE_MIX = 0.35               # share of background in m(t)
HISS_BIQUADS = (               # (b0,b1,b2,a1,a2) — pinkish slope (stub)
    (0.05, 0.0, 0.0, -0.95, 0.0),   # leaky integrator: LF lift
    (0.30, 0.0, 0.0, -0.70, 0.0),   # gentle HF rolloff
)
# NB: the stub is two stable one-pole sections in biquad form; at
# characterization §4.2(a) replace it with a fit to the Welch PSD (4-6 biquads).


def _interp_frames(v, n):
    """250 Hz frames (F,) -> samples (n,) linearly (C-1); n = F*HOP."""
    vg = np.concatenate([v, v[-1:]])
    k = np.arange(n) // HOP
    fr = (np.arange(n) % HOP) / HOP
    return vg[k] + (vg[k + 1] - vg[k]) * fr


def modulator(n, seed=0xF1A77E12, peaks=WOW_PEAKS, noise_mix=NOISE_MIX):
    """m(t): peaks + one-pole background (background RMS = 1 before mixing). Vec. and determ."""
    t = np.arange(n)
    m = np.zeros(n)
    for f, a, ph in peaks:
        m += a * np.sin(2.0 * np.pi * f * t / FS + ph)
    u = ska.xorshift32_block(seed, n)
    w = np.empty(n)
    acc = 0.0
    for i in range(n):                                   # one-pole (transcript)
        acc = POLE * acc + (1.0 - POLE) * u[i]
        w[i] = acc
    w = w / (np.sqrt(np.mean(w * w)) + 1e-12)
    return m + noise_mix * w


def process(x, v_macro, depth, hiss_lvl, seed_noise=0xF1A77E12,
            seed_hiss=0x8155CAFE):
    """Vector reference. x (n,), frame curves (F,) = n/HOP.
    Returns: y (n,), lag (n,) — for golden/diagnostics."""
    n = len(x)
    F = n // HOP
    assert F * HOP == n
    v_m = _interp_frames(v_macro, n)
    dep = _interp_frames(depth, n)
    hl = _interp_frames(hiss_lvl, n)

    # W-4: hiss at the input of the line
    hs = ska.xorshift32_block(seed_hiss, n)
    for b0, b1, b2, a1, a2 in HISS_BIQUADS:
        y1 = y2 = x1 = x2 = 0.0
        out = np.empty(n)
        for i in range(n):                               # transcript biquad
            yi = b0 * hs[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, hs[i]
            y2, y1 = y1, yi
            out[i] = yi
        hs = out
    src = x + hl * hs

    # W-2/W-3: speed and lag
    m = modulator(n, seed_noise)
    v = v_m * (1.0 + dep * m)
    ring = int(RING_S * FS)
    fade_n = int(FADE_MS * 1e-3 * FS)

    y = np.empty(n)
    lag_dbg = np.empty(n)
    lag = 0.0
    gain = 1.0
    stopped = False
    for i in range(n):                                   # sequential part
        if v_m[i] < V_EPS:                               # tape-stop policy
            if not stopped:
                stopped = True
            gain = max(0.0, gain - 1.0 / fade_n)
            if gain == 0.0:
                lag = 0.0                                # reset to nominal (W-3)
        else:
            if stopped and gain == 0.0:
                lag = 0.0
                stopped = False
            gain = min(1.0, gain + 1.0 / fade_n)
            lag += 1.0 - v[i]
            if lag > ring - L0 - 4:
                lag = ring - L0 - 4                      # ring protection
            elif lag < -(L0 - 4):
                lag = -(L0 - 4)
        r = i - L0 - lag
        if r < 0.0:
            y[i] = 0.0
        else:
            i0 = int(r)
            frc = r - i0
            a = src[i0]
            b = src[i0 + 1] if i0 + 1 < n else src[i0]
            y[i] = (a + (b - a) * frc) * gain
        lag_dbg[i] = lag
    return y, lag_dbg


def process_cstyle(x, v_macro, depth, hiss_lvl, seed_noise=0xF1A77E12,
                   seed_hiss=0x8155CAFE):
    """Literal C loop: a ring with a mask, per-frame (start,delta) controls,
    incremental generators. Semantics == process()."""
    n = len(x)
    F = n // HOP
    v_g = np.concatenate([v_macro, v_macro[-1:]])
    d_g = np.concatenate([depth, depth[-1:]])
    h_g = np.concatenate([hiss_lvl, hiss_lvl[-1:]])

    ring = int(RING_S * FS)
    buf = np.zeros(ring)
    mask = None                                          # ring is not 2^k — modulo
    fade_n = int(FADE_MS * 1e-3 * FS)

    s_h = np.uint32(seed_hiss or 1)
    bq_state = [[0.0, 0.0, 0.0, 0.0] for _ in HISS_BIQUADS]
    s_n = np.uint32(seed_noise or 1)
    acc = 0.0
    # background normalization: the same as in the reference — from running the same noise
    w_all = np.empty(n)
    s_tmp = np.uint32(seed_noise or 1)
    a_tmp = 0.0
    for i in range(n):
        s_tmp ^= np.uint32(s_tmp << np.uint32(13))
        s_tmp ^= np.uint32(s_tmp >> np.uint32(17))
        s_tmp ^= np.uint32(s_tmp << np.uint32(5))
        u = np.float64(np.int32(s_tmp)) * 2.0 ** -31
        a_tmp = POLE * a_tmp + (1.0 - POLE) * u
        w_all[i] = a_tmp
    wnorm = 1.0 / (np.sqrt(np.mean(w_all * w_all)) + 1e-12)
    # (in C wnorm is a firmware constant, computed once offline)

    y = np.empty(n)
    lag = 0.0
    gain = 1.0
    stopped = False
    wpos = 0
    for kk in range(F):
        for i in range(HOP):
            nn = kk * HOP + i
            fr = i / HOP
            v_mac = v_g[kk] + (v_g[kk + 1] - v_g[kk]) * fr
            dep = d_g[kk] + (d_g[kk + 1] - d_g[kk]) * fr
            hlv = h_g[kk] + (h_g[kk + 1] - h_g[kk]) * fr
            # hiss
            s_h ^= np.uint32(s_h << np.uint32(13))
            s_h ^= np.uint32(s_h >> np.uint32(17))
            s_h ^= np.uint32(s_h << np.uint32(5))
            hz = np.float64(np.int32(s_h)) * 2.0 ** -31
            for bi, (b0, b1, b2, a1, a2) in enumerate(HISS_BIQUADS):
                x1, x2, y1, y2 = bq_state[bi]
                yo = b0 * hz + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                bq_state[bi] = [hz, x1, yo, y1]
                hz = yo
            s = x[nn] + hlv * hz
            buf[wpos % ring] = s
            # modulator
            s_n ^= np.uint32(s_n << np.uint32(13))
            s_n ^= np.uint32(s_n >> np.uint32(17))
            s_n ^= np.uint32(s_n << np.uint32(5))
            u = np.float64(np.int32(s_n)) * 2.0 ** -31
            acc = POLE * acc + (1.0 - POLE) * u
            m = NOISE_MIX * acc * wnorm
            for fp, ap, ph in WOW_PEAKS:
                m += ap * np.sin(2.0 * np.pi * fp * nn / FS + ph)
            v = v_mac * (1.0 + dep * m)
            if v_mac < V_EPS:
                if not stopped:
                    stopped = True
                gain = max(0.0, gain - 1.0 / fade_n)
                if gain == 0.0:
                    lag = 0.0
            else:
                if stopped and gain == 0.0:
                    lag = 0.0
                    stopped = False
                gain = min(1.0, gain + 1.0 / fade_n)
                lag += 1.0 - v
                hi = ring - L0 - 4
                if lag > hi:
                    lag = hi
                elif lag < -(L0 - 4):
                    lag = -(L0 - 4)
            r = nn - L0 - lag
            if r < 0.0:
                y[nn] = 0.0
            else:
                i0 = int(r)
                frc = r - i0
                a = buf[i0 % ring]
                b = buf[(i0 + 1) % ring] if i0 + 1 <= nn else a
                y[nn] = (a + (b - a) * frc) * gain
            wpos += 1
    return y


if __name__ == "__main__":
    import time
    F = 500                                              # 2 s
    n = F * HOP
    t = np.arange(n)
    x = 0.4 * np.sin(2 * np.pi * 3150.0 * t / FS)        # W&F pilot tone

    v_macro = np.ones(F)
    v_macro[300:350] = np.linspace(1.0, 0.0, 50)         # tape-stop at 1.2 s
    v_macro[350:420] = 0.0
    v_macro[420:450] = np.linspace(0.0, 1.0, 30)         # restart
    depth = np.full(F, 0.002)                            # 0.2% — cassette class
    hiss = np.full(F, 0.02)

    t0 = time.time()
    y_vec, lag = process(x, v_macro, depth, hiss)
    t_vec = time.time() - t0
    t0 = time.time()
    y_c = process_cstyle(x, v_macro, depth, hiss)
    t_c = time.time() - t0
    err = np.max(np.abs(y_vec - y_c))
    print(f"[contract] max|vec - cstyle| = {err:.3e} (peak {np.max(np.abs(y_vec)):.3f})")
    assert err < 1e-9, "vec and cstyle diverged!"

    # wow and flutter spectrum via Hilbert on the steady segment (before the stop)
    seg = y_vec[FS // 4:FS]
    z = np.fft.ifft(np.fft.fft(seg) * ((np.arange(len(seg)) < len(seg) // 2) * 2))
    phi = np.unwrap(np.angle(z))
    inst = np.diff(phi) * FS / (2 * np.pi)
    dev = inst / 3150.0 - 1.0
    Wd = np.abs(np.fft.rfft(dev * np.hanning(len(dev))))
    fx = np.fft.rfftfreq(len(dev), 1 / FS)
    pk = [fx[i] for i in np.argsort(Wd[(fx > 1) & (fx < 60)])[-3:]]
    print(f"[wow/flut] RMS speed deviation {np.std(dev)*100:.3f}% "
          f"(set ~0.2%); modulation peaks ~ {sorted(np.round(pk,1))} Hz "
          f"(set to 4.2/12.7/33)")

    # tape-stop: silence during the stop, the lag is reset to nominal
    stop_rms = np.sqrt(np.mean(y_vec[int(1.45 * FS):int(1.65 * FS)] ** 2))
    print(f"[tape-stop] RMS during the stop {stop_rms:.2e} (→0); "
          f"lag after restart {lag[-1]:+.2f} samples (started from 0)")
    assert stop_rms < 1e-3

    # hiss: spectral slope (sanity)
    xz = np.zeros(n)
    yh, _ = process(xz, np.ones(F), np.zeros(F), np.ones(F))
    Y = np.abs(np.fft.rfft(yh[FS//4:FS//2] * np.hanning(FS // 4)))
    fY = np.fft.rfftfreq(FS // 4, 1 / FS)
    lo = np.mean(Y[(fY > 100) & (fY < 400)])
    hi = np.mean(Y[(fY > 8000) & (fY < 16000)])
    print(f"[hiss]     slope LF(100-400)/HF(8-16k) = {20*np.log10(lo/hi):.1f} dB "
          f"(pinkish >0)")
    print(f"[perf]     vec {t_vec:.2f} s / cstyle {t_c:.1f} s for 2 s of audio")
    print("OK: wow/flutter+hiss ready (stub parameters until the §4.2 session)")
