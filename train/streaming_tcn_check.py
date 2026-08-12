#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streaming_tcn_check.py — two tools for phase T0 of the N6 project:

1) A NUMERICAL PROOF of the streaming scheme for a causal dilated TCN via
   explicit FIFO states (spec §5.2): frame-by-frame processing in chunks of
   hop=48 (4 ms @12 kHz) with state carry-over == processing the whole sequence
   at once. This is exactly the scheme the graph exports to ONNX: the states of
   each layer are ordinary input/output tensors.

   The causal dilated convolution (in correlation form):
       y[t] = sum_j W[:,:,j] @ x[:, t - (k-1-j)*d],  x[<0] = 0
   Streaming: S is the last P=(k-1)*d columns of the layer input's history;
       xin = [S | chunk];  y[t] = sum_j W[:,:,j] @ xin[:, t + j*d]
       S' = the last P columns of xin.
   Initializing S=0 is equivalent to left zero-padding of the full sequence,
   so the equivalence is exact (down to machine epsilon).

2) A BUDGET CALCULATOR for planning the M0 shmoo: MAC/cycle, GOPS/voice,
   weights, states (taking into account the batch over subbands and the
   ping-pong state buffers) for a C x L x V grid. Cross-checked against the
   reference figures of spec §5.2–5.3.

Run: python3 streaming_tcn_check.py
"""

import numpy as np

# ----------------------------------------------------------------------------
# Part 1. Proof of streaming equivalence
# ----------------------------------------------------------------------------

class DilConv:
    """Causal dilated conv1d C_in->C_out, kernel k, dilation d."""

    def __init__(self, rng, c_in, c_out, k, d):
        self.w = rng.standard_normal((c_out, c_in, k)) / np.sqrt(c_in * k)
        self.b = rng.standard_normal(c_out) * 0.1
        self.k, self.d = k, d
        self.pad = (k - 1) * d

    def full(self, x):
        """x[C_in, T] -> y[C_out, T], the full sequence (zero-pad on the left)."""
        c_in, t = x.shape
        xp = np.concatenate([np.zeros((c_in, self.pad)), x], axis=1)
        y = np.zeros((self.w.shape[0], t))
        for j in range(self.k):
            y += self.w[:, :, j] @ xp[:, j * self.d: j * self.d + t]
        return y + self.b[:, None]

    def stream(self, x_chunk, state):
        """One chunk. state[C_in, pad] -> (y[C_out, Lc], new_state)."""
        lc = x_chunk.shape[1]
        xin = np.concatenate([state, x_chunk], axis=1)
        y = np.zeros((self.w.shape[0], lc))
        for j in range(self.k):
            y += self.w[:, :, j] @ xin[:, j * self.d: j * self.d + lc]
        return y + self.b[:, None], xin[:, lc:]

    def zero_state(self, c_in):
        return np.zeros((c_in, self.pad))


class TCN:
    """head 1x1 -> [residual( ReLU(dilconv) )] x L -> tail 1x1.
    A block with no separate pointwise — as in the spec's budget (1.18 MMAC/sample at C=128, L=24).
    The variant with a pointwise inside the block: +C^2*L MAC (~+33%), the streaming equivalence is unchanged."""

    def __init__(self, rng, c_in, c_hidden, c_out, layers, cycle, k=3):
        self.head = rng.standard_normal((c_hidden, c_in)) / np.sqrt(c_in)
        self.tail = rng.standard_normal((c_out, c_hidden)) / np.sqrt(c_hidden)
        assert layers % len(cycle) == 0
        dil = (list(cycle) * (layers // len(cycle)))[:layers]
        self.convs = [DilConv(rng, c_hidden, c_hidden, k, d) for d in dil]
        self.c_hidden = c_hidden

    def full(self, x):
        h = self.head @ x
        for cv in self.convs:
            h = h + np.maximum(cv.full(h), 0.0)
        return self.tail @ h

    def init_states(self):
        return [cv.zero_state(self.c_hidden) for cv in self.convs]

    def stream(self, x_chunk, states):
        h = self.head @ x_chunk
        new_states = []
        for cv, s in zip(self.convs, states):
            z, s2 = cv.stream(h, s)
            h = h + np.maximum(z, 0.0)
            new_states.append(s2)
        return self.tail @ h, new_states

    def state_bytes(self, dtype_bytes=1):
        return sum(s.size for s in self.init_states()) * dtype_bytes


def prove_streaming(seed=1, c_in=4, c_hidden=16, layers=8,
                    cycle=(1, 2, 4, 8), t_total=3072, hop=48):
    rng = np.random.default_rng(seed)
    net = TCN(rng, c_in, c_hidden, c_out=1, layers=layers, cycle=cycle)
    x = rng.standard_normal((c_in, t_total))

    y_full = net.full(x)

    states = net.init_states()
    chunks = []
    for i in range(0, t_total, hop):
        y_c, states = net.stream(x[:, i:i + hop], states)
        chunks.append(y_c)
    y_stream = np.concatenate(chunks, axis=1)

    err = np.max(np.abs(y_full - y_stream))
    print("=== 1. Streaming equivalence (chunked + state == full) ===")
    print(f"proof config: C={c_hidden}, L={layers}, cycle={cycle}, "
          f"hop={hop} (4 ms @12 kHz), T={t_total}")
    print(f"max|y_full - y_stream| = {err:.3e}  "
          f"({'OK — exact equivalence' if err < 1e-9 else 'FAIL'})")
    print(f"total state of this mini network: {net.state_bytes(8)} bytes (float64)")
    assert err < 1e-9
    return err


# ----------------------------------------------------------------------------
# Part 2. Budget calculator (M0 planning)
# ----------------------------------------------------------------------------

def budget(c, layers, voices, k=3, cycle=(1, 2, 4, 8, 16, 32),
           bands=4, band_fs=12000, c_in=9, c_out=1, pingpong=True):
    # c_in=9 — conditioning as in guide §3.1: the band's skeleton (1) +
    # amp/tA/tB (3) + band-id one-hot (4) + drive (1). Numerically it barely
    # affects the budget (c_in*C << L*C^2*k), but the M0 tables must match
    # the production shape of the graph one for one.
    """The spec's wiring: batch = voices x subbands, each subband is its own
    12 kHz sequence, the weights are shared. Everything int8, states int8."""
    reps = layers // len(cycle)
    sum_d = sum(cycle) * reps
    pad_cols = (k - 1) * sum_d                      # total state columns per layer
    rf_ms = (pad_cols + 1) / band_fs * 1e3

    mac_step = layers * c * c * k + c_in * c + c * c_out   # per 1 subband sample
    gops_voice = mac_step * band_fs * bands * 2 / 1e9      # MAC=2 op

    weights = layers * (c * c * k + c) + c_in * c + c * c_out  # int8, bytes
    state_seq = c * pad_cols                                   # bytes per 1 sequence
    state_voice = state_seq * bands * (2 if pingpong else 1)   # ping-pong in/out

    return dict(mac_step=mac_step, gops_voice=gops_voice, gops_total=gops_voice * voices,
                weights_mb=weights / 2**20, state_voice_kb=state_voice / 2**10,
                state_total_mb=state_voice * voices / 2**20, rf_ms=rf_ms)


def print_budget_tables():
    print("\n=== 2. Budgets for the M0 shmoo (batch = voices x 4 subbands, int8) ===")
    print("State ping-pong x2 is included; if stedgeai confirms in-place I/O — divide by 2.")
    hdr = f"{'C':>4} {'L':>3} {'V':>2} | {'MMAC/step':>9} {'GOPS/vc':>9} {'GOPS ':>7} | " \
          f"{'wts,MB':>8} {'state/vc,kB':>11} {'state tot,MB':>13} | {'RF,ms':>6}"
    print(hdr)
    print("-" * len(hdr))
    for c in (96, 128):
        for v in (1, 2, 3, 4, 5):
            b = budget(c, 24, v)
            print(f"{c:>4} {24:>3} {v:>2} | {b['mac_step'] / 1e6:>9.2f} {b['gops_voice']:>9.1f} "
                  f"{b['gops_total']:>7.0f} | {b['weights_mb']:>8.2f} {b['state_voice_kb']:>11.0f} "
                  f"{b['state_total_mb']:>13.2f} | {b['rf_ms']:>6.1f}")
    b96, b128 = budget(96, 24, 1), budget(128, 24, 1)
    print(f"\ncross-check with spec §5.2: C=128 -> {b128['mac_step'] / 1e6:.2f} MMAC/sample "
          f"(spec: 1.18), {b128['gops_voice']:.0f} GOPS/voice (spec: ~113); "
          f"C=96 -> {b96['gops_voice']:.0f} GOPS/voice (spec: ~64)")
    print("WARNING: the states scale x4 by subbands (batch wiring) and x2 by ping-pong —")
    print("that is more than the '~0.5 MB' line in spec §5.3. Fix the final layout from M0.")


if __name__ == "__main__":
    prove_streaming()
    # The production dilation cycle: for the layers with d=16,32 the state
    # (k-1)*d = 32,64 columns is LONGER than the chunk hop=48 — the default
    # config (cycle 1..8, pad<=16) does not cover that regime. The rule
    # state' = the tail of concat(state, chunk) must work even when the tail
    # grabs part of the old state.
    prove_streaming(seed=3, c_in=9, c_hidden=16, layers=12,
                    cycle=(1, 2, 4, 8, 16, 32), t_total=3072, hop=48)
    print_budget_tables()
