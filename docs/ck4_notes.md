# CK4 — checkpoint 4 (H1): numeric validation of MVE on the board

Goal: cross-check the MVE branches of skeleton_b/pqmf (and the whole hop-slot
path with the limiter) numerically against the host reference. The host tests
run the scalar path — this closes the "golden on the board" gap (§2.4). The
scheme and the contracts are in `fw/src/ck4.h` (read that first), the score is
in `fw/src/ck4.c`.

## Host (already run 1 Aug, in the cloud)

```
cd fw && make ck4          # -> build/ck4_ref.bin (reference: 48000 f32, 1 s)
cd tools && python3 ck4_compare.py --selftest
```

Reference of 1 Aug: RMS=0.4377, peak=0.980 (the trio sits in the limiter),
CRC=9cb3b2cb. `make test` after adding ck4 — all previous tests OK, thresholds
untouched.

## Board (N6_m1, built by hand)

1. `cp fw/src/ck4.[ch] <Repo>/Examples/SAI/N6_m1/FSBL/DSP/src/` — the same
   manual sync as for the other DSP copies (do not forget it when editing!).
   In the IDE: Refresh (F5) + Clean after adding files (pitfall #5 in
   h1_notes).
2. Capture buffer — 192K in the `.ram2` section (the attribute is already in
   ck4.c under N6_TARGET). CHECK THE LINKER: g_pipe ~198K already lives in
   RAM2; together ~390K — the region @0x340A4000 has to fit them (the AXISRAM
   bank is 1M — OK, but cross-check against the .ld!).
3. Three insertions into main (all under `#ifdef N6_CK4`, the normal build
   does not change):

```c
/* render_into_half(), BEFORE the FIFO drain: */
#ifdef N6_CK4
    n6_ck4_pre_hop(&g_mf);
#endif
    /* ... drain, n6_pipe_hop(&g_pipe, out48); ... */
#ifdef N6_CK4
    n6_ck4_post_hop(out48, N6_HOP48);
#endif

/* super-loop (NOT the ISR), next to the heartbeat: */
#ifdef N6_CK4
    { static char _l[160]; int _n = n6_ck4_dump_line(_l, sizeof _l);
      if (_n > 0) { puts(_l); } }     /* VCP; one line per iteration */
#endif
```

4. Build with `-DN6_CK4`, turn on terminal logging (TeraTerm: File→Log;
   PuTTY: Session→Logging) TO A FILE, reset the board. Do NOT touch the
   KeyStep: real MIDI bytes in the capture window = non-determinism = a
   garbage cross-check (the score injects itself, the UART input can be
   disabled altogether).
   Capture: 0.1 s warmup + 1 s; the dump is ~440K of text @115200 ≈ 40 s
   (the audio may stutter meanwhile — that is fine, the capture is already in
   RAM).
   The heartbeat in the log does not get in the way (the parser skips it), but
   the CK4 lines must not be cut by the terminal (width ≥ 120).
5. Cross-check: `python3 tools/ck4_compare.py <log>.log`
   (it takes the reference from fw/build/ck4_ref.bin — rebuild with `make ck4`
   if the repository is newer than the cloud reference).

## Criterion and reading the result

- **PASS: rel RMS < 1e-4.** Expect 1e-6..1e-5 (vfma fusing + a ulp difference
  in sinf between host and target, amplified by the rotators across the
  harmonics).
- 1e-4..1e-3 — suspect libm sinf first (newlib vs glibc), look at the
  breakdown over 100 ms segments: libm smears evenly, an MVE bug is usually
  localised (worst-case chord, PB sawtooth, voice steal).
- Above 1e-3 or NaN — a bug in the MVE branch; turn MVE off one brick at a
  time (#undef __ARM_FEATURE_MVE by hand in skeleton_b / pqmf) and bisect.
- IMPORTANT: the score (ck4.c) is shared by the reference and the board. Any
  edit = rebuild BOTH sides (make ck4 + reflash), otherwise the cross-check
  lies.

## What the score covers

Worst-case trio of low keys (the 3.15M chord), PB sawtooth (phase catch-up of
B-1), CC1/channel pressure (timbres + parser need=1), NoteOff/NoteOn with a
retrig cut (§8.2), voice steal at full polyphony, running status on the final
Offs, release tails, the limiter at its working point (peak 0.980). NOT
covered: the real NPU (stub; will be closed in M2 by its own cross-check),
the §8.3 degradation (npu_miss=0).


## The reference has moved: a FIR stage in the path (4 Aug, late evening)

The production `n6_fir_coeffs.h` (from `ckpt_delta_p360.pt`) puts the D-17
linear layer into the pipeline, and the CK4 score now runs through it.
New canonical values: **CRC=0001842f, RMS=0.3886** (was bd9c8c83 / 0.4377).
`make qemu-ck4` with the new path: rel RMS 7.118e-06 (threshold 1e-4).
While at it, a dependency of the targets on `n6_fir_coeffs.h` was added to the
Makefile: without it a change of coefficients did NOT rebuild the reference,
and make ck4 handed back the old CRC.

Addition the same night: the scalar FIR cost 652 k cycles per hop on the board
(6.5 cycles/MAC — the backward step `s[-k]`); the weights are reversed at
recompute time, the dot product became forward×forward, and an MVE branch was
added (64 taps vectorised + 1 scalar). The summation order changed, so the
reference was rebuilt once more: **canonical CRC=86dbdfc5** (0001842f lived
one round and was never flashed anywhere), the RMS is the same 0.3886;
qemu-ck4 = 7.156e-06, [fir] golden 2.915e-07, [fir chain] exact zero.
