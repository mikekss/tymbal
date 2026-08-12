#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag11.py — round 11: shrinking the architecture on the canonical
ORT path.

Verdicts of rounds 8-10 (dearly bought):
  - atonn 4.0 recognises int8 ONLY in graphs coming straight out of the ORT
    quantizer; any hand assembly or post-surgery => Conv(float) on the M55;
  - therefore the Q/DQ shape is untouchable, and the only way to fight for
    epochs/memory is the architecture (C, L, N, T) — luckily d14 showed: a
    graph half the size => 4x fewer epochs (53) and activations of 1.375 MB
    (which fits).
  - k=5 is forbidden by the hardware; k=3 with any of our dilations is fully
    in HW.

Candidates (both k=3, C=128, N=12, T=48, ORT-QDQ without post-processing):
  d27: L=12, cycle (1..32)x2, Σd=126 -> RF 21.1 ms, weights ~0.59 MB,
       int8 states 378 KB/voice (x2 ping-pong)
  d28: L=18, cycle (1..32)x3, Σd=189 -> RF 31.6 ms, weights ~0.89 MB
The price against L=24 is the receptive field 42 -> 21/31.6 ms: a question of
"path memory" quality, settled by a listening test in T1 (D-9 in decision_log).

Run: tools/run_t0_diag_r11.bat
"""
import os
import numpy as np
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import streaming_tcn_check as ref
from export_t0_diag2 import build_bh, verify_bh
from export_t0_diag3 import BHReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")

if __name__ == "__main__":
    for name, layers, reps in [("d27_bh_c128_v12_l12", 12, 2),
                               ("d28_bh_c128_v12_l18", 18, 3)]:
        rng = np.random.default_rng(2028)
        net = ref.TCN(rng, 9, 128, 1, layers=layers, cycle=(1, 2, 4, 8, 16, 32), k=3)
        b = ref.budget(128, layers, 3, k=3, c_in=9)
        print(f"{name}: {b['mac_step']/1e6:.2f} MMAC/step, {b['gops_voice']:.0f} GOPS/voice, "
              f"weights {b['weights_mb']:.2f} MB, state/voice {b['state_voice_kb']:.0f} KB, "
              f"RF {b['rf_ms']:.1f} ms")
        build_bh(net, 9, 1, 12, 48, name)
        verify_bh(name, net, 9, 12, 48)
        quantize_static(os.path.join(OUT, name + ".onnx"),
                        os.path.join(OUT, name + "_qdq.onnx"),
                        BHReader(net, 9, 12, 48),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        print(f"{name}_qdq written (canonical ORT path, no post-processing)")
    print("done")
