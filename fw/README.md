# fw/ — N6 firmware scaffold (host-verified)

`make test` (Linux/gcc) — builds the DSP building blocks and runs them against
the §2.4 golden vectors (`test/vectors/*.bin` from
`tools/export_fw_assets.py`).

**Status as of 7 Aug: eleven tests, all OK** — pqmf 4.319e-06, skeleton_b
1.512e-03, wowflutter 8.035e-06 (rel. RMS), midi, pipeline, npu layout,
npu null, sustain, fir 2.915e-07, fir chain (exact zero), unison.
Checked on 7 Aug by a clean unpack into an unfamiliar environment:
`rm -rf build &&
make test` → 11/11 without a single fix. The golden vectors never needed
regenerating — the thresholds are the same as before.

- `src/n6_config.h` — canonical constants; everything M0-dependent (V, hop)
  lives in `n6_params_t`.
- `src/pqmf_synth.c` — polyphase synthesis (Helium candidate: the inner loops).
- `src/skeleton_b.c` — rendering in subbands (B-1..B-5; the `n6_bandresp.h`
  table; uint32 phase + sincosf 1/sample + rotator over harmonics — target
  semantics).
- `src/wowflutter.c` — wow and flutter + hiss (W-1..W-5; f32 + peak rotators —
  target semantics).
- `src/midi.[ch]` — SPSC FIFO (C11 atomics) + FSM (running status, realtime).
- `src/voice.[ch]` — §4.2 allocation, glide in log-f0, retrig flags for §8.2.
- `src/pipeline.[ch]` — the §8.1 hop slot (wow AFTER synthesis, NoteOn slices
  after DONE).
- `src/npu_iface.h` + `npu_stub.c` — the NPU abstraction; the target
  implementation (`npu_neuralart.c`, LL_ATON) is written in H0 from the
  generate template.
- `src/main_target.c` — the firmware scaffold (-DN6_TARGET): ISR stubs,
  initialization order, cache/DMA pitfalls. Clock/MPU/linker — a PORT from the
  CubeN6 example.

Not to be done before the freeze: npu_neuralart.c with a concrete shape, the
final training. The H5 limiter is DONE (1 Aug): feed-forward peak limiter in
pipeline.c (threshold 0.98, instant attack, 80 ms release) — the smoke test
holds peak==0.980.

## Canonical mpool (item 19 of the review backlog — closed 7 Aug)

The production Neural-ART pool profile is **`stm32n6_nucleo_app_safe.mpool`**:
the current graph (`n6-app-safe`, see opt_npu.md) was built with it, and
`tools/atonn/run_variant.sh` takes the same one. The other five `*.mpool` are
T0/M0 history (`full`, `full_onchip`, `onchip`, `nucleo`, the base
`stm32n6`): do not delete them, reports refer to them, but do NOT take them
for new runs. Do not write comments inside an mpool — the parser is strict
JSON.
