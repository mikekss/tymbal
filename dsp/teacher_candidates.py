#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teacher_candidates.py — teacher candidates for the refiner INSTEAD of tape (D-1a,
the "not vintage" branch, session 1 Aug). Context: I dropped the vintage
aesthetic of the deck; the refiner does not care whose degradation it learns — what
is needed is an offline teacher with nonlinearity and memory that cannot run in
real time on the M55. There are two families here:

  A — "modern analog" (program-dependent saturation, folder, OTT dynamics):
      A1 foldsat  — 3 bands, tanh with drive from the envelope (memory ~15 ms),
                    sin folder in the mid band, oversampling ×8;
      A2 ottpress — 3 bands, downward+upward compression (OTT character, release ~25 ms),
                    light tanh, oversampling ×4.
  B — "digital glitch" (artifacts as aesthetics, Alva Noto):
      B1 codec   — mp3 32 kbit/s mono 48k through ffmpeg (pre-echo, whistling
                   highs; frame 1152/48000 = 24 ms), the output is aligned by
                   cross-correlation with the dry (codec delay compensated);
      B2 crush   — SRR with hold jitter (seed-deterministic) + mu-law
                   6 bit + LP 12 kHz;
      B3 residue — STFT 512/128: magnitude quantization in 6 dB steps + a gate −50 dB
                   below the frame peak → "birdies"/spectral garbage on the tails.

Constraint on the teacher's MEMORY (otherwise a TCN with RF 21–42 ms will not learn it):
all time constants/frames ≤ 30 ms: A1/A2 envelopes 15/25 ms, B1 frame 24 ms,
B2 hold <1 ms, B3 window 10.7 ms (+1 window of lookahead — non-causality within
the RF, acceptable for an offline teacher). All outputs are length-aligned with
the dry and brought to its RMS (a fair listening test), peak softly limited to 0.99.

The dry input is skeleton VARIANT A (as in make_corpus §4.1): a fixed
sequence of sparse/legato/staccato/jumps/drone/tech phrases with pauses,
the seed is fixed → bit-exact reproducible.

LISTENING TEST 1 Aug: the winner is B3_residue; A1_foldsat is good ONLY on
stationary material (drone/tech, >18 s), not on transients. Hence the
H family (hybrids): B3 is the core, always; A1 is blended in by the wet/dry
stationarity curve m(t) (250 Hz frames):
      H1 fold_residue — blend(x, A1(x), m) -> B3  (fold before the spectral stage);
      H2 residue_fold — B3 -> blend(., A1(.), m)  (we fold the birdies).
m(t) is the ratio of the slow/fast envelopes (40/4 ms): transient ~ clean,
sustain ~ full fold. In training m(t) is NOT recovered by the network from the sound —
it rides on a separate conditioning curve (the drive axis, §4.2(d); the D-8
antenna maps onto it too) => its time constants are NOT bounded by the network's RF.

Run:    python3 teacher_candidates.py            (wavs -> audition/)
Smoke:  python3 teacher_candidates.py --selftest
Output: audition/teacher_00_dry.wav + teacher_{A1,A2,B1,B2,B3,H1,H2}_*.wav
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import skeleton_a as ska                                    # noqa: E402
import make_corpus as mc                                    # noqa: E402

FS = 48000
SEED = 20260801


# ----------------------------------------------------------------- dry input
def render_dry(seconds_per_cat=None, seed=SEED):
    """Fixed sequence of phrases + 0.3 s pauses. -> float64 mono 48k."""
    seq = seconds_per_cat or [("sparse", 5.0), ("legato", 5.0), ("staccato", 4.0),
                              ("jumps", 3.0), ("drone", 6.0), ("tech", 3.0)]
    rng = np.random.default_rng(seed)
    gap = np.zeros(int(0.3 * FS))
    parts = []
    for cat, dur in seq:
        ph = mc.gen_phrase(cat, dur, rng)
        a = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"], ph["gate"],
                             seed=int(rng.integers(1, 2 ** 31)))
        pk = np.max(np.abs(a)) + 1e-12
        parts += [a * (0.6 / max(pk, 0.6)), gap]
    return np.concatenate(parts[:-1])


# -------------------------------------------------------------- utilities
def onepole(x, tau_s, fs=FS):
    """Causal one-pole (memory ~tau)."""
    a = np.exp(-1.0 / (tau_s * fs))
    return signal.lfilter([1 - a], [1, -a], x)


def split3(x, fs=FS, f1=250.0, f2=2500.0):
    """Causal 3-band split (butter-2, the sum is not perfectly flat — that's ok)."""
    sos_lo = signal.butter(2, f1, "low", fs=fs, output="sos")
    sos_hi = signal.butter(2, f2, "high", fs=fs, output="sos")
    sos_b1 = signal.butter(2, f1, "high", fs=fs, output="sos")
    sos_b2 = signal.butter(2, f2, "low", fs=fs, output="sos")
    lo = signal.sosfilt(sos_lo, x)
    mid = signal.sosfilt(sos_b2, signal.sosfilt(sos_b1, x))
    hi = signal.sosfilt(sos_hi, x)
    return lo, mid, hi


def match_rms(y, ref, peak=0.99):
    r = np.sqrt(np.mean(ref ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    y = y * r
    knee = 0.85 * peak                                      # safety net:
    a = np.abs(y)                                           # below the knee — identity,
    if np.max(a) > knee:                                    # above it — soft saturation
        y = np.where(a > knee, np.sign(y) *
                     (knee + (peak - knee) * np.tanh((a - knee) / (peak - knee))),
                     y)
    return y


# ------------------------------------------------------------------- family A
A1_ENV_REF = 0.25       # review 2 Aug: instead of np.max(env) over the phrase (a global
                        # = NON-causal statistic, unavailable to the network) — a constant
                        # reference; teacher causality is mandatory for learnability

def teacher_A1_foldsat(x):
    """Per-band program-dependent saturation + sin folder (OS ×8). CAUSAL."""
    lo, mid, hi = split3(x)
    out = []
    for band, base_drive, fold in ((lo, 2.2, 0.0), (mid, 3.0, 0.8), (hi, 1.6, 0.0)):
        env = onepole(np.abs(band), 0.015)                  # memory 15 ms
        up = signal.resample_poly(band, 8, 1)
        env_u = np.repeat(env, 8)[:len(up)]
        m = np.minimum(env_u * (1.0 / A1_ENV_REF), 1.5)     # causal modulation
        drive = base_drive * (1.0 + 1.5 * m)
        y = np.tanh(drive * up) / np.tanh(np.maximum(drive, 1e-3))
        if fold > 0.0:                                      # soft wavefolding
            y = y + fold * np.sin(2.5 * np.pi * y) * m
        out.append(signal.resample_poly(y, 1, 8)[:len(band)])
    y = 0.9 * out[0] + 1.0 * out[1] + 0.85 * out[2]
    return signal.sosfilt(signal.butter(1, 20, "high", fs=FS, output="sos"), y)


def _gain_updown(env_db, thr_dn=-18.0, ratio_dn=3.0, thr_up=-42.0, ratio_up=2.0,
                 max_up=12.0):
    g = np.zeros_like(env_db)
    over = env_db > thr_dn
    g[over] = -(env_db[over] - thr_dn) * (1.0 - 1.0 / ratio_dn)
    under = env_db < thr_up
    g[under] = np.minimum((thr_up - env_db[under]) * (1.0 - 1.0 / ratio_up), max_up)
    return g


def teacher_A2_ottpress(x):
    """3-band downward+upward compression (OTT character) + light tanh (OS ×4)."""
    bands = split3(x)
    out = []
    for band, w in zip(bands, (0.95, 1.0, 1.05)):
        env = onepole(np.abs(band), 0.003)                  # detector 3 ms
        env_db = 20 * np.log10(env + 1e-7)
        g_db = onepole(_gain_updown(env_db), 0.025)         # smoothing 25 ms
        y = band * 10 ** (g_db / 20.0)
        up = signal.resample_poly(y, 4, 1)
        up = np.tanh(1.4 * up) / np.tanh(1.4)
        out.append(w * signal.resample_poly(up, 1, 4)[:len(band)])
    return sum(out)


# ------------------------------------------------------------------- family B
def teacher_B1_codec(x, bitrate="32k", fs=FS):
    """mp3 low-bitrate through ffmpeg; the output is aligned by cross-correlation."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found — B1 skipped")
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        p_in, p_mp3, p_out = (os.path.join(td, n) for n in
                              ("in.wav", "mid.mp3", "out.wav"))
        sf.write(p_in, x.astype(np.float32), fs, subtype="FLOAT")
        for cmd in (["ffmpeg", "-y", "-v", "error", "-i", p_in,
                     "-ar", str(fs), "-ac", "1", "-b:a", bitrate, p_mp3],
                    ["ffmpeg", "-y", "-v", "error", "-i", p_mp3,
                     "-ar", str(fs), "-ac", "1", p_out]):
            subprocess.run(cmd, check=True)
        y, _ = sf.read(p_out, dtype="float64")
    n = min(len(x), 5 * fs)
    lag = np.argmax(signal.correlate(y[:n + 8192], x[:n], mode="valid"))
    y = y[lag:lag + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y, int(lag)


def teacher_B2_crush(x, hold_nom=6, jitter=2, bits=6, seed=SEED):
    """SRR with hold jitter + mu-law quantization + LP 12k."""
    rng = np.random.default_rng(seed)
    holds = hold_nom + rng.integers(-jitter, jitter + 1, size=len(x) // 2 + 16)
    holds = np.maximum(holds, 1)
    edges = np.cumsum(holds)
    edges = edges[edges < len(x)]
    idx = np.zeros(len(x), dtype=np.int64)
    idx[edges] = 1
    idx = np.take(np.concatenate(([0], edges)), np.cumsum(idx))
    y = x[idx]                                              # steps with jitter
    mu = 2 ** bits - 1
    s = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
    s = np.round(s * (mu / 2)) / (mu / 2)                   # quantization
    y = np.sign(s) * (np.expm1(np.abs(s) * np.log1p(mu))) / mu
    return signal.sosfilt(signal.butter(4, 12000, "low", fs=FS, output="sos"), y)


def teacher_B3_residue(x, nfft=512, hop=128, qdb=6.0, gate_db=50.0):
    """Spectral quantization + gate: birdies and garbage on the tails.
    D-16: qdb (quantization step, dB; ~0 => no quantization) and gate_db (gate
    depth below the frame peak, dB) — CONDITIONING AXES. A scalar or a curve
    (any length, resampled to the STFT frames linearly); in the corpus the axes
    are stored as 250 Hz frames (like f0/amp/..., §4.1). A scalar call is
    bit-exact equal to the earlier fixed version (the 1 Aug listening tests are
    reproducible)."""
    f, t, Z = signal.stft(x, fs=FS, nperseg=nfft, noverlap=nfft - hop)
    nfr = Z.shape[1]

    def to_frames(v):
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        if len(v) == 1:
            return np.full(nfr, v[0])
        return np.interp(np.linspace(0, 1, nfr), np.linspace(0, 1, len(v)), v)

    q = to_frames(qdb)[None, :]
    g = to_frames(gate_db)[None, :]
    mag = np.abs(Z)
    ph = np.angle(Z)
    mdb = 20 * np.log10(mag + 1e-9)
    mdb_q = np.where(q > 0.05, np.round(mdb / np.maximum(q, 0.05)) *
                     np.maximum(q, 0.05), mdb)              # q~0 => identity
    thr = mdb_q.max(axis=0, keepdims=True) - g              # gate below the frame peak
    mag_q = np.where(mdb_q >= thr, 10 ** (mdb_q / 20.0), 0.0)
    _, y = signal.istft(mag_q * np.exp(1j * ph), fs=FS, nperseg=nfft,
                        noverlap=nfft - hop)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def render_axis_sweeps(outdir, dry):
    """D-16: axis sweep listening tests — hear the whole axis in one pass.
    The curves are defined on the 250 Hz grid (the reference corpus format)."""
    import soundfile as sf
    nfr = int(len(dry) / FS * 250)
    sweeps = (
        ("axq", np.linspace(0.0, 12.0, nfr), 50.0,
         "q: 0->12 dB @ gate 50"),
        ("axg", 6.0, np.linspace(70.0, 30.0, nfr),
         "gate: 70->30 dB @ q 6"),
        ("corner_soft", 3.0, 62.0, "soft corner (q3/g62)"),
        ("corner_hard", 9.0, 38.0, "hard corner (q9/g38)"),
        # round 2 (listening test 2 Aug: hard beats everything, the axes are
        # indistinguishable one at a time):
        ("axmacro", np.linspace(2.0, 13.0, nfr), np.linspace(60.0, 28.0, nfr),
         "MACRO axis: diagonal (q2/g60)->(q13/g28) through hard"),
        ("corner_harder", 12.0, 32.0, "even harder (q12/g32) — vs hard A/B"),
    )
    for name, q, g, note in sweeps:
        y = teacher_B3_residue(dry, qdb=q, gate_db=g)
        y = match_rms(np.asarray(y)[:len(dry)], dry)
        assert np.all(np.isfinite(y))
        sf.write(os.path.join(outdir, f"teacher_B3_{name}.wav"),
                 y.astype(np.float32), FS, subtype="FLOAT")
        print(f"[sweep] B3_{name}: {note}")


# ------------------------------------------------------------------- family H
def stationarity_curve(x, fr=250, ratio=1.5, hold_fr=10, rise_s=0.35):
    """m(t) in [0,1]: an onset (fast > ratio*slow) drops m to 0 (+hold 40 ms),
    sustain grows it back over rise_s => the fold "blooms" INSIDE the note, the
    transients stay clean. 250 Hz frames (like the controls in §4.1) + upsampling by
    repetition. In training this is conditioning (the drive axis, §4.2(d)), NOT an
    input the network must infer from audio => the time constants are free of the
    RF. The parameters are listening-test knobs."""
    fast = onepole(np.abs(x), 0.003)
    slow = onepole(np.abs(x), 0.050)
    hop = FS // fr
    nfr = len(x) // hop
    f_fr = fast[:nfr * hop].reshape(nfr, hop).max(axis=1)
    s_fr = slow[:nfr * hop].reshape(nfr, hop).max(axis=1)
    onset = f_fr > ratio * s_fr + 1e-5
    mf = np.zeros(nfr)
    m, cool = 0.0, 0
    step = 1.0 / (rise_s * fr)
    for k in range(nfr):                                    # 250 Hz — cheap
        if onset[k]:
            m, cool = 0.0, hold_fr
        elif cool > 0:
            cool -= 1
        else:
            m = min(1.0, m + step)
        mf[k] = m
    ma = np.repeat(mf, hop)
    return np.pad(ma, (0, len(x) - len(ma)), mode="edge"), mf


def _blend(dry, wet_sig, m):
    return (1.0 - m) * dry + m * wet_sig


def teacher_H1_fold_residue(x):
    """blend(x, A1, m) -> B3: fold on the sustain, then the spectral residue."""
    m, _ = stationarity_curve(x)
    return teacher_B3_residue(_blend(x, teacher_A1_foldsat(x), m))


def teacher_H2_residue_fold(x):
    """B3 -> blend(., A1, m): we fold the already quantized spectrum (dirtier top)."""
    y = teacher_B3_residue(x)
    m, _ = stationarity_curve(x)                            # m from the DRY (control!)
    return _blend(y, teacher_A1_foldsat(y), m)


def render_axis_drone(outdir):
    """Round 3 (listening test 2 Aug: in axmacro the movement is inaudible — the
    material was changing on its own and masking the axis). A static A1 55 Hz
    drone (f0/amp/tA/tB are constants) — ONLY the axis moves: (a) a smooth sweep,
    (b) 5 steps of ~4 s each (the ear catches step boundaries better than drift).
    Macro axis t: q=2+11t, g=60-32t."""
    import soundfile as sf
    F = 20 * 250                                            # 20 s
    ph = dict(f0=np.full(F, 55.0), amp=np.full(F, 0.5),
              tA=np.full(F, 0.6), tB=np.full(F, 0.15), gate=np.ones(F))
    ph["gate"][:3] = 0; ph["gate"][-3:] = 0
    dry = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"],
                           ph["gate"], seed=42)
    t_sweep = np.linspace(0.0, 1.0, F)
    t_step = (np.minimum(np.arange(F) // (4 * 250), 4)) / 4.0
    for name, tt in (("dronesweep", t_sweep), ("dronestep", t_step)):
        y = teacher_B3_residue(dry, qdb=2.0 + 11.0 * tt,
                               gate_db=60.0 - 32.0 * tt)
        y = match_rms(np.asarray(y)[:len(dry)], dry)
        assert np.all(np.isfinite(y))
        sf.write(os.path.join(outdir, f"teacher_B3_{name}.wav"),
                 y.astype(np.float32), FS, subtype="FLOAT")
        print(f"[axdrone] B3_{name}: OK")
    sf.write(os.path.join(outdir, "teacher_B3_drone_dry.wav"),
             dry.astype(np.float32), FS, subtype="FLOAT")


TEACHERS = (("A1_foldsat", teacher_A1_foldsat),
            ("A2_ottpress", teacher_A2_ottpress),
            ("B1_codec", None),                             # special call (lag)
            ("B2_crush", teacher_B2_crush),
            ("B3_residue", teacher_B3_residue),
            ("H1_fold_residue", teacher_H1_fold_residue),
            ("H2_residue_fold", teacher_H2_residue_fold))


# ------------------------------------------------------------- run
def run(outdir, dry):
    import soundfile as sf
    os.makedirs(outdir, exist_ok=True)
    sf.write(os.path.join(outdir, "teacher_00_dry.wav"),
             dry.astype(np.float32), FS, subtype="FLOAT")
    rows = [("00_dry", np.sqrt(np.mean(dry ** 2)), np.max(np.abs(dry)), "-")]
    for name, fn in TEACHERS:
        try:
            if name == "B1_codec":
                y, lag = teacher_B1_codec(dry)
                note = f"lag={lag}"
            else:
                y, note = fn(dry), "-"
        except RuntimeError as e:
            print(f"[{name}] SKIPPED: {e}")
            continue
        y = match_rms(np.asarray(y, dtype=np.float64)[:len(dry)], dry)
        assert np.all(np.isfinite(y)) and len(y) == len(dry)
        sf.write(os.path.join(outdir, f"teacher_{name}.wav"),
                 y.astype(np.float32), FS, subtype="FLOAT")
        rows.append((name, np.sqrt(np.mean(y ** 2)), np.max(np.abs(y)), note))
    w = max(len(r[0]) for r in rows)
    print(f"{'variant':<{w}}  {'RMS':>7}  {'peak':>6}  note")
    for n, r, p, note in rows:
        print(f"{n:<{w}}  {r:7.4f}  {p:6.3f}  {note}")


def selftest():
    rng = np.random.default_rng(1)
    ph = mc.gen_phrase("legato", 2.0, rng)
    dry = ska.render_voice(ph["f0"], ph["amp"], ph["tA"], ph["tB"], ph["gate"],
                           seed=7) * 0.6
    for name, fn in TEACHERS:
        if name == "B1_codec":
            try:
                y, lag = teacher_B1_codec(dry)
                assert lag >= 0
            except RuntimeError:
                print("[selftest] B1 skipped (no ffmpeg)")
                continue
        else:
            y = fn(dry)
        y = match_rms(np.asarray(y, dtype=np.float64)[:len(dry)], dry)
        assert np.all(np.isfinite(y)) and len(y) == len(dry), name
        assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2)) /
                                 np.sqrt(np.mean(dry ** 2)) + 1e-12)) < 3.0, name
        print(f"[selftest] {name}: OK (len={len(y)}, "
              f"peak={np.max(np.abs(y)):.3f})")
    # B2 determinism
    a = teacher_B2_crush(dry)
    b = teacher_B2_crush(dry)
    assert np.array_equal(a, b), "B2 is not deterministic"
    print("[selftest] B2 determinism: OK")
    # stationarity curve
    ma, mf = stationarity_curve(dry)
    assert len(ma) == len(dry) and mf.min() >= 0.0 and mf.max() <= 1.0
    print(f"[selftest] m(t): OK (frames={len(mf)}, mean={mf.mean():.2f})")
    # D-16: scalar == constant curve (bit-exact), q~0 == identity quantization
    y_sc = teacher_B3_residue(dry, qdb=6.0, gate_db=50.0)
    y_cv = teacher_B3_residue(dry, qdb=np.full(500, 6.0),
                              gate_db=np.full(500, 50.0))
    assert np.array_equal(y_sc, y_cv), "B3: scalar != constant curve"
    y_q0 = teacher_B3_residue(dry, qdb=0.0, gate_db=200.0)
    assert np.sqrt(np.mean((y_q0 - dry) ** 2)) < 2e-3, "B3: q=0/g=200 is not ~identity"
    print("[selftest] D-16 B3 axes: OK (constant==scalar, q0/g200 ~ identity)")
    print("[selftest] ALL OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweeps", action="store_true",
                    help="D-16: axis sweep listening tests for B3 (q, gate, corners)")
    ap.add_argument("--axdrone", action="store_true",
                    help="D-16 r.3: the axis on a static drone (sweep+steps)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "audition"))
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.sweeps:
        os.makedirs(args.out, exist_ok=True)
        render_axis_sweeps(args.out, render_dry())
    elif args.axdrone:
        os.makedirs(args.out, exist_ok=True)
        render_axis_drone(args.out)
    else:
        run(args.out, render_dry())
