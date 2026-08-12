#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag5.py — round 5: k=3/L=24 with int8-domain quantization for the
data-movement ops (Concat/Slice) — an attack on fragmentation.

Diagnoses: d12 (k=3, naive QDQ) — all convolutions on HW, but 208 epochs and 5 MB
of activations (fp32 round-trips around every op; without hyperRAM it does not fit);
d14 (k=5) — rejected by the hardware (a dilated k=5 → SW Conv). The cure:
quantize Concat/Slice/Add/Relu AS WELL (extra_options ForceQuantizeNoInputCheck),
so that inter-layer data stays int8 and the Q/DQ pairs collapse.

Output:
  d15_bh_c128_v12_qdq2.onnx — wiring (a), weights of d10
  d16_bh_c192_v3_qdq2.onnx  — wiring (b), weights of d11
Run: tools/run_t0_diag_r5.bat (nucleo map; d15 is also run on the on-chip-only
map: the "weights in AXISRAM" test, spec §2).
"""
import os
import numpy as np
from onnxruntime.quantization import (QuantFormat, QuantType, quantize_static)

import streaming_tcn_check as ref
from export_t0_diag3 import BHReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")
K = 3
CYCLE = (1, 2, 4, 8, 16, 32)
LAYERS = 24

if __name__ == "__main__":
    rng = np.random.default_rng(2026)          # the same order as diag2/diag3
    net_a = ref.TCN(rng, 9, 128, 1, LAYERS, CYCLE, k=K)
    net_b = ref.TCN(rng, 8, 192, 4, LAYERS, CYCLE, k=K)

    for src, dst, net, c_in, N, T in [
            ("d10_bh_c128_v12", "d15_bh_c128_v12_qdq2", net_a, 9, 12, 48),
            ("d11_bh_c192_v3", "d16_bh_c192_v3_qdq2", net_b, 8, 3, 48)]:
        quantize_static(
            os.path.join(OUT, src + ".onnx"), os.path.join(OUT, dst + ".onnx"),
            BHReader(net, c_in, N, T),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "Relu", "Add", "Concat", "Slice"],
            extra_options={"ForceQuantizeNoInputCheck": True})
        print("written", dst)
    print("done")
