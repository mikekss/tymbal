#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_t0_diag3.py — round 3: QDQ int8 on top of the batch-as-height graphs
d10/d11.

Diagnosis of round 2: fp32 graphs compile, but the convolutions go into
Conv(float) SW epochs — Neural-ART computes in int8, float is emulated on the
M55. Expectation for QDQ: convolutions in HW epochs. The in/out state scales are
NOT tied here (ORT quantizer), which does not matter for checking the mapping —
QAT/brevitas will tie them in the final version (§3.3).

Output: models/t0/diag/d12_bh_c128_v12_qdq.onnx, d13_bh_c192_v3_qdq.onnx
Run: tools/run_t0_diag_qdq.bat
"""
import os
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

import streaming_tcn_check as ref

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "t0", "diag")
K = 3
CYCLE = (1, 2, 4, 8, 16, 32)
LAYERS = 24


class BHReader(CalibrationDataReader):
    """Calibration: random input and states in batch-as-height shapes."""

    def __init__(self, net, c_in, N, T, n_calib=8):
        rng = np.random.default_rng(7)
        self.data = []
        for _ in range(n_calib):
            d = {"x": rng.standard_normal((1, c_in, N, T)).astype(np.float32)}
            for i, cv in enumerate(net.convs):
                d[f"state_in_{i}"] = (0.3 * rng.standard_normal(
                    (1, net.c_hidden, N, cv.pad))).astype(np.float32)
            self.data.append(d)
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


if __name__ == "__main__":
    # nets with the same shapes as in diag2 (same seed/order — for state shapes)
    rng = np.random.default_rng(2026)
    net_a = ref.TCN(rng, 9, 128, 1, LAYERS, CYCLE, k=K)
    net_b = ref.TCN(rng, 8, 192, 4, LAYERS, CYCLE, k=K)

    for src, dst, net, c_in, N, T in [
            ("d10_bh_c128_v12", "d12_bh_c128_v12_qdq", net_a, 9, 12, 48),
            ("d11_bh_c192_v3", "d13_bh_c192_v3_qdq", net_b, 8, 3, 48)]:
        quantize_static(os.path.join(OUT, src + ".onnx"),
                        os.path.join(OUT, dst + ".onnx"),
                        BHReader(net, c_in, N, T),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        print("written", dst)
    print("done")
