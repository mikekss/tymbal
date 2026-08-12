#!/usr/bin/env python3
"""Minimal reproducer for the LL_ATON_LIB_Concat slow path (STM32N6 / Neural-ART).

Builds an ONNX graph containing ONLY the pattern that hits the generic branch
of LL_ATON_LIB_Concat: a concatenation along the innermost (width) axis of a
tensor whose left-hand dimensions are not all 1.

Shapes match our real streaming-TCN application:
    C = 88 channels, V = 2 (two independent voices carried as a spatial axis),
    T = 48 samples per inference, state width 2*d for dilation d.

The graph is arithmetic-free apart from the concatenations, so whatever the
compiler reports for it is the cost of the copy itself.

Usage:  python repro_concat.py [out.onnx]
Requires: onnx, numpy
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto

C, V, T = 88, 2, 48
DILATIONS = (1, 2, 4, 8, 16, 32)      # two stacks of six in the real graph


def build(path):
    inputs, outputs, nodes, inits = [], [], [], []

    # h: the running activation, common to every layer
    inputs.append(helper.make_tensor_value_info(
        "h", TensorProto.FLOAT, [1, C, V, T]))

    for i, d in enumerate(DILATIONS):
        w = 2 * d                       # state ring width for this dilation
        sname, cname = "state_%d" % i, "cat_%d" % i
        inputs.append(helper.make_tensor_value_info(
            sname, TensorProto.FLOAT, [1, C, V, w]))
        # THE PATTERN: concat along axis 3 (width). Dimensions to the left
        # are [1, 88, 2] — not all ones, so axis_is_leftmost is false.
        nodes.append(helper.make_node("Concat", [sname, "h"], [cname], axis=3))
        outputs.append(helper.make_tensor_value_info(
            cname, TensorProto.FLOAT, [1, C, V, w + T]))

    g = helper.make_graph(nodes, "concat_slowpath", inputs, outputs, inits)
    m = helper.make_model(g, producer_name="n6-repro",
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)

    total = sum(C * V * (2 * d) for d in DILATIONS) \
        + len(DILATIONS) * C * V * T
    print("wrote %s" % path)
    print("  %d Concat nodes, layout [1, %d, %d, W], axis=3"
          % (len(DILATIONS), C, V))
    print("  elements moved per inference: %d  (= %d bytes at int8)"
          % (total, total))
    print("  in our application the same 12 blocks (two stacks) moved")
    print("  145 728 bytes and cost 1 019 501 M55 cycles = 6.997 cycles/byte")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "repro_concat.onnx")
