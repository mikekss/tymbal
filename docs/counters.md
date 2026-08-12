# Glossary of counters and labels

Everything the heartbeat and the tests print, in one place. Until 7 Aug the
decoding existed only in the code (`fw/src/pipeline.c` PROF,
`fw/src/skeleton_b.c` SKB_P, `fw/test/qemu/qemu_prof.c`); the price was that
every new reader had to do archaeology on the logs.

## Heartbeat line (`[hb]`, main.c)

`hops` — hops since the previous print (printed once every 2 s ≈ 500).
`cyc(min/avg/max)` — cycles of the full `n6_pipe_hop` (MIDI drain and the
24-into-32 packing are OUTSIDE). Budget 3 200 000.
`underrun` — SAI did not get its buffer in time; zero is the only norm.
`midi` — bytes received; `err/drop` — UART errors / FIFO overflow.
`peak=N/1000` — output peak ×1000 (980 = 0.980); `clip` — limiter clips.

### `prof(skb/npu/pq/wf/fir)` — hop stages, `n6_prof[]`

| label | slot | what it measures |
|---|---|---|
| `skb` | 0 | skeleton render TOGETHER with NPU pumping (cooperative loop) |
| `npu` | 1 | finishing the inference after the render (waiting for DONE) |
| `pq`  | 2 | PQMF synthesis 4×12 → 48 kHz |
| `wf`  | 3 | wow/flutter + hiss |
| `fir` | 7 | FIR bank over the residual (inside the mix loop) |

The stages OVERLAP: pumping lives inside the render, and the waiting cycles
are booked where they physically pass. Hence the mirror image between silence
(`npu` large, `skb` small) and a chord (the other way round). The useful
formula is a maximum, not a sum: `hop ≈ inference wall + pq + wf + tail`
(breakdown in opt_npu.md).

### `skb(wt/pro/bod/dec/ph)` — inside the skeleton, `n6_skb_prof[]`

| label | slot | what it measures |
|---|---|---|
| `wt`  | 0 | reading band amplitudes from the `g_br8` table (30 kB, DTCM) |
| `pro` | 1 | prologue: seeding four sin/cos rotators over the segment |
| `bod` | 2 | harmonic body: two vfma accumulators across the bands |
| `dec` | 3 | per-frame envelope decoder (amplitudes, `skb_exp2`) |
| `ph`  | 4 | phase catch-up and preparing the top live harmonic |

### `tail(xc/notes/lim)` — `n6_prof[4..6]`

`xc` — building the NPU input (`build_xcond`); `notes` — clearing voice state
on NoteOn + ping/pong swap (with user-IO the swap is empty, the remaining
1–10 k shows up only on key presses); `lim` — the limiter. `userIO=1` — the
NPU buffers are ours and concatenated, there is no swap.

### `peaks(sk/rs/fir)` — "honest loudness" diagnostics

Peaks (×1000) of the skeleton, of the network residual and of the FIR output
over the window. `sk` and `rs` of the same order and both ≥ ~500 — the scales
are honest; `rs` an order of magnitude above `sk` — the residual is cranked
up and the verdict on the sound is void (see endgame).

### `npu: inference=… blocks=… m55(hyb/EC)=…`

`inference` — wall time of a full graph run (cycles); `blocks` — calls to
`LL_ATON_RT_RunEpochBlock`; `m55(hyb/EC)` — CPU cycles inside hybrid blocks /
EC blobs (do not confuse with wall: the difference is waiting for the NPU).
`prof_err` — contract checks at init: 0 is the norm; 8 — the memcpy
self-test; 71..77 — user-IO buffer binding (which one exactly is in
npu_neuralart.c). `npu_miss` — a hop where DONE did not arrive in time.

## Host test labels (`make test`, 11 of them)

`[pqmf_synth] [skeleton_b] [wowflutter]` — rel RMS against the Python golden;
`[midi]` — the parser; `[pipeline]` — 1 s smoke test (peak/nan/npu_miss);
`[npu layout]` — the `[CIN][V][T]` layout and the conditioning reach the
network; `[npu null]` — null teacher: muting the network zeroes the output;
`[sustain]` — CC64 (holds/releases/panic); `[fir]` — FIR against the
reference; `[fir chain]` — identity stub in the pipeline (independent of the
coefficients); `[unison]` — a pair 7 cents apart, voice steal with no glide
carry-over, drift.

## Graph shapes

`d30/d31/d44` — TCN generations: d30/d16 are historical (dilation attribute,
S2D/D2S lowering, T0); d31 is the first gather2; **d44 is the production
one** (C=88, V=2, T=48, L=12, gather2, int8 QDQ). CK4 canonical value = CRC
`86dbdfc5`.
