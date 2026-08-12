#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_golden.py — golden vectors §2.4: the Python↔C contract for the H1 unit
tests. Each block: exact input and output (float64 in the npz; C is cross-
checked in f32, tolerance ~1e-6 RMS). The fixed score covers: attack/release,
a glissando across the 6 kHz seam, tA/tB movement, tape-stop with restart, hiss.

Output (dsp/golden/):
  skeleton_b_io.npz  controls -> sub (4, n12)          [runtime canon, D-6]
  pqmf_synth_io.npz  sub -> y48                        [synthesis bank]
  wowflutter_io.npz  y48 + curves -> out               [wow and flutter+hiss]
  fullmix_io.npz     controls -> final (end-to-end chain)
  + audition/golden_mix.wav — to hear what this actually sounds like.
Run: python3 make_golden.py
"""
import os
import numpy as np
from scipy.io import wavfile

import pqmf_design as pq
import skeleton_a as ska
import skeleton_b as skb
import wowflutter as wf

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "golden")
FS = 48000

# ------------------------------------------------------ score (2.4 s, 250 Hz)
F = 600
f0 = np.full(F, 220.0)
f0[100:250] = np.exp(np.linspace(np.log(220.0), np.log(1568.0), 150))  # gliss
f0[250:] = 1568.0                                    # harmonic 4 at the 6272 Hz seam
amp = np.full(F, 0.5)
tA = 0.5 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, F))
tB = np.full(F, 0.25)
gate = np.ones(F)
gate[:3] = 0.0
gate[280:300] = 0.0                                  # re-trigger (declick C-5)
gate[-3:] = 0.0

v_macro = np.ones(F)
v_macro[430:470] = np.linspace(1.0, 0.0, 40)         # tape-stop
v_macro[470:520] = 0.0
v_macro[520:560] = np.linspace(0.0, 1.0, 40)         # restart
depth = np.full(F, 0.002)
hiss = np.full(F, 0.03)

if __name__ == "__main__":
    os.makedirs(GOLD, exist_ok=True)

    sub = skb.render_voice_b(f0, amp, tA, tB, gate)
    np.savez(os.path.join(GOLD, "skeleton_b_io.npz"),
             f0=f0, amp=amp, tA=tA, tB=tB, gate=gate,
             seed=np.uint32(0xC0FFEE), sub=sub)
    print(f"skeleton_b_io: sub {sub.shape}, RMS {np.sqrt(np.mean(sub**2)):.4f}")

    y48 = pq.synthesize(sub, skb._SYN)[:sub.shape[1] * 4]
    np.savez(os.path.join(GOLD, "pqmf_synth_io.npz"), sub=sub, y=y48)
    print(f"pqmf_synth_io: y {y48.shape}, RMS {np.sqrt(np.mean(y48**2)):.4f}")

    n = (len(y48) // wf.HOP) * wf.HOP
    Fw = n // wf.HOP
    out, lag = wf.process(y48[:n], v_macro[:Fw], depth[:Fw], hiss[:Fw])
    np.savez(os.path.join(GOLD, "wowflutter_io.npz"),
             x=y48[:n], v_macro=v_macro[:Fw], depth=depth[:Fw],
             hiss=hiss[:Fw], y=out, lag=lag,
             seeds=np.array([0xF1A77E12, 0x8155CAFE], np.uint32))
    print(f"wowflutter_io: y {out.shape}, RMS {np.sqrt(np.mean(out**2)):.4f}")

    np.savez(os.path.join(GOLD, "fullmix_io.npz"),
             f0=f0, amp=amp, tA=tA, tB=tB, gate=gate,
             v_macro=v_macro, depth=depth, hiss=hiss, final=out)
    adir = os.path.join(HERE, "audition")
    os.makedirs(adir, exist_ok=True)
    wavfile.write(os.path.join(adir, "golden_mix.wav"), FS,
                  (np.clip(out, -1, 1) * 32767).astype(np.int16))
    print("fullmix_io + audition/golden_mix.wav ready")
    assert not np.isnan(out).any()
    print("OK: golden vectors written (dsp/golden/*_io.npz)")
