#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag4.py — round 4: the k=5/L=12 shape (ladder §3.5, the "fewer state
tensors" step) in the batch-as-height layout + QDQ.

Motive (from the d12 report, qdq_8249): int8 maps onto the NPU (0 SW epochs), but
there are 208 epochs, 88 of them hybrid — mostly the Concat/Slice of the state
scheme (a pair per layer). k=5, cycle (1..32)×2, L=12: Σd=126, (k−1)·Σd=504 — the
state bytes and the RF (42 ms) are THE SAME, but there are half as many state
tensors (12 instead of 24) => half as many Concat/Slice epochs; MAC per step
12·C²·5 against 24·C²·3 = −17%.

Output: models/t0/diag/d14_bh_c128_v12_k5l12_qdq.onnx (+ fp32 to cross-check shapes).
Equivalence: chunks+states == the numpy reference (verify_bh from diag2).
Run: tools/run_t0_diag_r4.bat (which also holds the EC flag and the nucleo-mpool experiments).
"""
import os
import numpy as np

import streaming_tcn_check as ref
from export_t0_diag2 import build_bh, verify_bh
from export_t0_diag3 import BHReader
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2027)
    # k=5, L=12, cycle ×2 repeats: Σd=126, a layer's state is (k-1)·d columns
    net = ref.TCN(rng, 9, 128, 1, layers=12, cycle=(1, 2, 4, 8, 16, 32), k=5)

    b = ref.budget(128, 12, 3, k=5, c_in=9)
    print(f"shape k=5/L=12: {b['mac_step']/1e6:.2f} MMAC/step (was 1.18), "
          f"GOPS/voice {b['gops_voice']:.0f}, states/voice {b['state_voice_kb']:.0f} kB "
          f"(must match k=3/L=24), RF {b['rf_ms']:.1f} ms")

    build_bh(net, 9, 1, 12, 48, "d14_bh_c128_v12_k5l12")
    verify_bh("d14_bh_c128_v12_k5l12", net, 9, 12, 48)

    quantize_static(os.path.join(OUT, "d14_bh_c128_v12_k5l12.onnx"),
                    os.path.join(OUT, "d14_bh_c128_v12_k5l12_qdq.onnx"),
                    BHReader(net, 9, 12, 48),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print("written d14_bh_c128_v12_k5l12_qdq")
