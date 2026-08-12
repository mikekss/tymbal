#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag12.py — round 12 (the final static one): candidates for closing T0.

The key arithmetic of r.11: "unallocated" = ~865 kB × L regardless of the map —
this is the fp32 staging of the S2D/D2S wrapping of the dilated HW convolutions,
∝ C·N·T. Wiring (a) C=128/N=12 does not fit into it; wiring (b) C=192/N=3
(staging ×4 smaller) DOES fit the nucleo map, already back in r5 (d16:
activations 1.9 MB, weights 2.56 MB int8 in NOR, SW=0). The compiler voted for
(b) in D-5.

Candidates:
  d29: (a) last chance — C=96, N=8 (2 voices), L=24: staging ×0.47 of
       C128/N12 => ~2.4 MB, right on the edge. If it squeezes through, (a) is
       alive for the D-5 quality test in T1 with reduced polyphony.
  d30: (b) C=192, N=3, L=12 (RF 21 ms): half as many epochs — headroom on t_call;
       the choice between RF 21/42 ms is made by listening in T1 (D-9).
The canonical (b) L=24 candidate already exists: d16_bh_c192_v3_qdq2.onnx (r5).

Run: tools/run_t0_diag_r12.bat (+ d16 with int8 io and the epoch controller).
"""
import os
import numpy as np
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import streaming_tcn_check as ref
from export_t0_diag2 import build_bh, verify_bh
from export_t0_diag3 import BHReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")

CASES = [
    # name, c_in, C, c_out, N, layers, comment
    ("d29_bh_c96_v8_l24", 9, 96, 1, 8, 24, "(a) C=96, 2 voices"),
    ("d30_bh_c192_v3_l12", 8, 192, 4, 3, 12, "(b) L=12, RF 21 ms"),
]

if __name__ == "__main__":
    for name, c_in, C, c_out, N, layers, tag in CASES:
        rng = np.random.default_rng(2029)
        net = ref.TCN(rng, c_in, C, c_out, layers=layers,
                      cycle=(1, 2, 4, 8, 16, 32), k=3)
        bands = 4 if c_out == 1 else 1
        b = ref.budget(C, layers, max(N // bands, 1) if bands == 4 else N,
                       k=3, c_in=c_in, c_out=c_out, bands=bands)
        print(f"{name} {tag}: {b['mac_step']/1e6:.2f} MMAC/step, "
              f"{b['gops_voice']:.0f} GOPS/voice, weights {b['weights_mb']:.2f} MB, RF {b['rf_ms']:.1f} ms")
        build_bh(net, c_in, c_out, N, 48, name)
        verify_bh(name, net, c_in, N, 48)
        quantize_static(os.path.join(OUT, name + ".onnx"),
                        os.path.join(OUT, name + "_qdq.onnx"),
                        BHReader(net, c_in, N, 48),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        print(f"{name}_qdq written")
    print("done")
