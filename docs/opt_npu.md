# The NPU stage: where 2.5 million cycles go

Journal, 4 Aug 2026. The subject is not our code: the refiner on Neural-ART,
graph `n6_gather2_qdq` of shape d44 (C=88, V=2, T=48, L=12), compiled by ST Edge
AI with the `n6-app-safe` profile. The hop budget is 3.2 Mcyc at 800 MHz.

## Starting point

After five rounds of skeleton optimization the picture was this (cycles per
hop):

| | silence | worst chord |
|---|---|---|
| hop | 3 111 545 | 3 669 167 |
| **`npu`** | **2 533 485** | **2 535 221** |
| `skb` | 53 149 | 605 149 |
| `wf` | 134 005 | 134 687 |
| `pqmf` | 73 944 | 74 411 |
| outside the counters | 316 962 | 319 699 |

The NPU stage costs 2.53 M **the same in silence and while playing** — 79% of
the budget goes before the first note sounds. In silence the hop is already
3.11 M with a completely silent skeleton. The conclusion that follows from
this, and that reordered every priority: **further skeleton optimization does
not close the budget.**

## The tool was inside the runtime

There was nothing to break the 2.53 M down across the 88 epochs, and the "1.5 M
in 54 Hybrid" estimate rested on indirect signs. It turned out ST ships what is
needed together with the compiler, just switched off by default:

- `LL_ATON_EB_DBG_INFO` in `ll_aton_config.h` (it sits there commented out)
  adds a number, a type and **the compiler's cost estimate** to every epoch
  block — `estimated_npu_cycles` (pure NPU) and `estimated_tot_cycles` (with
  the memory penalty);
- `LL_ATON_RT_SetEpochCallback` calls a callback at four points in a block:
  before and after `start_epoch_block`, before and after `end_epoch_block` —
  for EVERY block, including a hybrid one, where `start_epoch_block == NULL`.

That gives two quantities, and they must not be confused:

- **`m55`** = `(POST_START-PRE_START) + (POST_END-PRE_END)` — processor cycles
  INSIDE the block: programming the engines plus the software part of a hybrid.
  This is the price the stage charges the M55.
- **`wall`** = `POST_END-PRE_START` — the full block time, including waiting for
  the NPU. As a "cost" it is meaningless.

Even before the board, analysing `network.c` on the host gave a static picture:
the compiler promises **1 263 056** cycles for the whole inference — 703 376 in
15 EC blobs and 559 680 in 54 hybrids. The board showed 2.54 M. The profile had
to explain the 1.28 M difference, and it did.

## What the profile showed

`m55(hyb/EC) = 1 290 916 / 66 016`. Almost the whole million is in **twelve**
blocks out of 69:

| block | epoch | type | estimate | m55 | ratio |
|---|---|---|---|---|---|
| EP 01 | 4 | Concat | 8 800 | 62 253 | 7.1 |
| EP 05 | 8 | Concat | 9 152 | 63 956 | 7.0 |
| EP 11 | 16 | Concat | 9 856 | 68 709 | 7.0 |
| EP 17 | 24 | Concat | 11 264 | 78 427 | 7.0 |
| EP 23 | 32 | Concat | 14 080 | 97 697 | 6.9 |
| EP 29 | 39 | Concat | 19 712 | 135 466 | 6.9 |
| EP 35 | 46 | Concat | 8 800 | 62 086 | 7.1 |
| EP 39 | 50 | Concat | 9 152 | 64 509 | 7.0 |
| EP 45 | 58 | Concat | 9 856 | 69 037 | 7.0 |
| EP 51 | 66 | Concat | 11 264 | 78 588 | 7.0 |
| EP 57 | 74 | Concat | 14 080 | 103 526 | 7.4 |
| EP 63 | 81 | Concat | 19 712 | 135 730 | 6.9 |

The total is **1 019 501 cycles per hop, 32% of the budget.** The remaining 42
hybrids (`Slice`) cost 7 k each, 270 k together; the EC blobs take only 66 k
from the processor.

This is `cat_k` — concatenation of the state ring with the input before each of
the six dilated convolutions, twice (the stacks come in pairs). Widths
50/52/56/64/80/112 at 88 channels and V=2: `2*88*(48+2*d)` for dilations
1,2,4,8,16,32 — and these are **exactly** the `est` values. So the compiler
counts Concat as one element per cycle, and the M55 does it in seven. The input
(48 samples, 8448 bytes) is rewritten six times per stack.

**Conclusion: a third of the hop budget is software copying that the NPU can do
by itself.** The fix is regenerating the graph: either a shape without `Concat`
(a ring buffer the convolution reads from directly), or a placement in which
the compiler puts the concatenation on the streaming engines. It needs ST Edge
AI, that is, the Windows side.

## Second finding: pumping

The same profile showed something I could fix right away and with my own hands.

`LL_ATON_RT_RunEpochBlock` is **non-blocking**: it advances the state machine
and returns. The counter showed 13 209 calls for 69 blocks — that is about 190
idle polls per block, about 90 cycles each, **1.2 M cycles per hop with the
processor spinning for nothing**.

And `pipeline.c` rendered the WHOLE skeleton (0.6 M) before entering the
polling loop. Worse: before the first poll the NPU is not started at all — the
runtime is a state machine that the processor drives. A comment in the code
promised cooperative pumping; the code did not do it.

So 0.6 M of work was queued in front of 1.2 M of idling.

The fix: the render is cut into spans of `N6_SKB_SPAN` subband samples, with
pumping between spans. The arithmetic does not change by a single bit —
verified with CK4: the blob matched byte for byte, `CRC=bd9c8c83`.

The pumping step is **adaptive**. The first version used a fixed 8 polls per
span, and the render finished inside the first sixth of the stage — where there
is still little idle time, because in the early blocks the M55 is itself busy
with concatenations. Three quarters of the work got hidden, 115 k stuck out.
The right step is "the total poll count divided by the number of spans"; the
total is known from the previous hop and is stable (9.7–13 k). There can be no
fixed constant here: it depends on the chord.

## Along the way: DTCM

The cycles-per-instruction ratio (see `docs/opt_skeleton.md`) showed that
`skb_wtab` is memory-bound: 2.8–3.7 against 2.0–2.4 for the rest of the
skeleton. The Cortex-M55 has a DTCM — 128 kB right next to the core, zero wait
cycles, past the cache and past AXI, that is, with no contention against the
NPU. The project did not use it at all: the FSBL linker script knew only
AXISRAM.

Moved there: `g_br8` (30.0 kB), the sine table, `w_re`/`w_im`, `p_s`/`q_s`,
`dA` — 39 744 bytes. Result: **`wt` 228 366 → 107 367 cycles, −53%**, the whole
skeleton −26%.

Two things without which this would have been ghost debugging, and both found
by reading:

1. **The TCM has ECC.** After reset the contents are random and the ECC bits do
   not match them — the very first READ gives a precise hard fault (PECC, AFSR
   bit 17). The region must be written once, in words, before any access. Byte
   writes will not do: in ECC memory that is a read-modify-write.
2. **The FSBL lives in the secure world** (AXISRAM at 0x34…), so our DTCM is at
   `0x30000000`, not at the non-secure `0x20000000`.

## A side lesson about globals

Cutting the render into spans required moving `h_last`/`h_pad`/`seg_cur` from
locals to file statics — and that immediately cost **16% of the skeleton**: the
compiler has to re-read a global on every iteration, because it cannot prove
that `skb_wtab` does not touch it. A copy into a local variable put them back
into registers (`bod` 96 k → 136 k → 96 k).

Caught by the QEMU rig in a minute. On the board it would have been an extra
round trip and, very likely, the wrong conclusion that "interleaving is
expensive".

## The hop tail: 307 k in swap_states, and why it was not what I thought

Three tail counters (`build_xcond` / `swap_states` / the limiter) gave
4529 / 307460 / 3823 — the pipeline remainder is fully accounted for.
`swap_states` is 9.6% of the hop budget.

From here the order of steps matters. A DMA fix suggested itself, but first I
had to understand what exactly was expensive: the copying or the cache
maintenance (12 ranges, 43.7 kB; with a 32-byte line that is about 1400
invalidate operations plus as many cleans). Their fixes are different. We split
them: **270 362 against 34 575**. It is not the cache.

6.2 cycles per byte — and here it was easy to get it wrong a second time. The
obvious story was "newlib-nano is built for size, so memcpy is byte-by-byte".
**Not true:** disassembling `libc_nano.a` shows, for aligned pointers, an
unrolled `ldr`/`str` loop over 16 words, and the state buffers are aligned to 4
(the offsets are visible in `network_c_info.json`). The D-cache is on too —
checked in `main.c`.

So the limit is the latency of the processor → npuRAM path: 1366 cache lines,
and for each of them a fill on read, a fill on write (write-allocate) and a
later eviction. Three transactions per line, 4100 transactions in 270 k cycles
— about 66 cycles per transaction, and a plain `ldr`/`str` loop does not
overlap them: each one waits for the previous one.

The fix is **memory-level parallelism**: MVE loads 16 bytes per instruction,
and four loads in a row BEFORE the first store keep four misses in flight at
once instead of one. Measured: **copying 270 362 → 59 380 (−78%)**, the whole
`swap_states` 307 460 → 92 080. DMA was not needed.

On the side: the first version of this change did not build — the definition of
the copy was below the call. The cause is broader than a typo: the target-only
files (`npu_neuralart`, `npu_boot`, `n6_weights`) were built by NOBODY on the
development machine — the host does not take them, and neither does the QEMU
rig. So `make check-target` appeared, with CubeIDE's real flags (`-mcmse` is
critical: without it the secure aliases `RISAFx_S` are invisible).

## State swap: 307 460 → 237

The second largest item after `Concat` — about 317 k cycles that landed in no
counter and stood the same in silence and while playing. The suspect was named
at once (`n6_npu_swap_states`, the 43.7 kB state ring), but "the size looks
right" is a hypothesis, not a measurement, so three slots went into the hop
tail. The answer: `xc/swap/lim = 4539/307420/4163`. The hypothesis was right.

Then two steps, and the second cancels the first.

**Step 1 — a vector copy.** The breakdown inside the swap: 270 320 cycles on
the copying itself, 34 571 on cache maintenance. So the bottleneck is memory
bandwidth, not the cache. `memcpy` from newlib-nano for aligned pointers is not
a byte loop but a 16-word ldr/str (verified by disassembling `libc_nano.a`), and
on 43.7 kB that is exactly what shows. Replacing it with `n6_wide_copy` — four
16-byte MVE loads before the first store, so that the loads pipeline — gave
**270 320 → 59 380**, the whole swap 307 460 → 92 080. The copy is checked on
the board by a self-test against `memcpy` over all lengths 0..150 and four
alignments; a failure sets `prof_err = 8`.

**Step 2 — do not copy at all.** The swap exists only because the compiler
allocates the buffers: each of the six states has its own input and its own
output, and between hops the output has to be moved into the input. If the
buffers are OURS, the input and the output of one pair can be made one buffer —
and there is nothing to move.

stedgeai has the lever: `--no-inputs-allocation --no-outputs-allocation`
(frontend flags; `atonn` itself does not have them). The runtime then expects
`LL_ATON_Set_User_Input_Buffer` / `..._Output_Buffer` for every port; it checks
the 32-byte alignment and the minimum size itself. Regeneration did not touch
the weights — the blob has the same md5, and NOR did not have to be reflashed.

One pitfall cost a debug round trip. I gated the first version on the macro
`LL_ATON_NETWORK_USER_ALLOCATED_INPUTS` from `network.h` — and `npu_neuralart.c`
does not include that header. The branch silently compiled to nothing, the
buffers stayed unbound, and the old swap ran over garbage pointers: a hard fault
inside `n6_wide_copy`. The fix is not to repair the macro but to drop
conditional compilation entirely: the decision is made from the RUNTIME field
`n->ib[0]->is_user_allocated`. It is always there, it cannot lie, and it costs
one read at init.

`swap` = **237 cycles** — that is no longer copying, it is a call to a function
that returns immediately.

## AXI cache: −335 k, and why it is unnecessary here

With user IO the runtime calls NPU cache maintenance on our buffers, and ST's
`npu_cache_clean_invalidate_range` starts with
`assert(hcacheaxi_s.Instance == CACHEAXI)`; the handle is only set in
`npu_cache_enable`. The debug build stopped on that assert — so I turned the
cache on to get past it. The board answered with a number:

| | swap | npu | hop, silence |
|---|---|---|---|
| before user IO, cache off | 92 080 | 2 559 700 | 2 933 500 |
| user IO, cache on, buffers in `.ram2` | 213 | 2 902 000 | 3 181 000 |
| user IO, cache on, buffers in npuRAM6 | 211 | 2 894 500 | 3 173 700 |
| **user IO, cache off** | **237** | **2 550 700** | **2 826 500** |

The swap got 92 k cheaper, the NPU stage got 335 k more expensive — the result
is worse than the starting point. My first diagnosis was wrong: I decided it
was placement and moved 45.5 kB from `.ram2` into the unused npuRAM6 (448 kB at
0x34350000; the graph does not touch it). That returned 7 k out of 335 — so
placement has almost nothing to do with it. The mistake has a simple name: I
thought about who WRITES the buffers (the processor, once per hop) and did not
think about who READS them (the NPU, every epoch).

That left the cache. The assert turned out to be spurious: with the cache off
the function itself correctly does nothing — there is also an `if` inside on the
same condition. It is removed by a single `NDEBUG` in the project properties.
Our own checking does not rest on ST's asserts — we have `n6_npu_prof_err` and
codes 71..77. PROCESSOR cache maintenance (`LL_ATON_Cache_MCU_*` →
`mcu_cache.c`) is a separate pair of functions, always compiled; buffer
coherence between the M55 and the NPU is in place.

The result matches what ST writes in its own mpools: the internal npuRAM3..6
pools come WITHOUT `cacheable`, and `CACHEABLE_ON` is set only on the external
flash pool. On our profile the AXI cache is not an accelerator but an extra
level on the NPU's path to on-chip memory.

## What came out

Averages over 500 hops, cycles:

| | silence | chord (2 voices, peak 0.98) |
|---|---|---|
| **hop** | **2 826 500** | **2 972 700** (max 3 000 338) |
| `npu` — top-up at the end | 2 550 700 | 6 528 |
| `skb` — render together with pumping | 51 850 | 2 738 800 |
| `pqmf` | 80 250 | 80 180 |
| `wow/flutter` | 134 700 | 135 050 |
| tail (xc/notes/lim) | 4675/237/3955 | 4353/240/7421 |
| `underrun` | 0 | 0 |

Two things in this table are worth reading separately.

**The accounting is closed.** The slot total in silence is 2 826 367 against a
hop of 2 826 500. A difference of 130 cycles out of 2.8 million. The 317 k
outside the counters that this analysis started from are gone, and the
remainder of slot [5] while playing (1..10 k) is explained: it is not the swap
but `n6_npu_zero_voice` — clearing a voice's states on NoteOn, real work.

**The stages overlapped.** In silence `npu` = 2.55 M, `skb` = 52 k; on a chord
it is exactly the other way round — `npu` = 6.5 k, `skb` = 2.74 M. These are
not two modes but one loop: pumping lives INSIDE the render, and the waiting
cycles are charged to whichever counter's code they physically pass through.
The useful formula is now not a sum but a maximum: **hop ≈ inference wall +
PQMF + wow + tail.** The skeleton — 605 k on the worst chord — hides entirely
inside the wait for the NPU and costs the budget nothing. Check on a chord:
2 700 420 + 80 180 + 135 050 + 12 014 = 2 927 664 against a hop of 2 972 700,
that is, 45 k not overlapped.

A side effect: the inference wall grows from 2.557 M in silence to 2.70 M while
playing, even though the NPU's work is exactly the same (`m55(hyb/EC)` =
1 281 900 / 64 110 in both cases). What grows is not the work but the service
latency of the epoch state machine: the processor is busy rendering and
advances blocks less often. That is the price of overlap, and it is six times
smaller than what the overlap saves.

Residual risk: on a chord CHANGE the peak reaches 3.29 M — above the 3.2 M
budget, but `underrun` stays at zero and the double buffer absorbs it. Worth
digging into only if clicks appear.

## Concat: 32% of the budget turned out to be a library memcpy

The twelve `Concat` blocks cost 1 019 501 cycles per hop and did not move for
any compiler option. The hypotheses were all about the engines: put the
concatenation on the streaming DMA, change the channel layout, regenerate the
shape. All of them needed the Windows side and a NOR reflash. Not one of them
was needed.

What I should have read was not the options but the runtime source.
`LL_ATON_LIB_Concat` in `ll_aton_lib.c` picks the fast DMA path on the
condition `axis_is_leftmost` — every dimension to the LEFT of the
concatenation axis must equal one. Our shape is `[1, V=2, T, C=88]` and the
concatenation is along the width: there is a two on the left, the check fails,
and the work goes to the generic branch at the bottom of the function. And that
branch is four `memcpy` calls per block: two per input, one for each of the `V`
rows.

The arithmetic adds up and leaves no room for hypotheses. The twelve blocks
move 145 728 bytes per hop — exactly the compiler's total `est`, because it
counts Concat as one element per cycle. 1 019 501 / 145 728 = **6.997 cycles
per byte**, and that figure is the same for all six block sizes: 7.04, 7.01,
6.95, 6.96, 6.91, 6.87. A constant per-byte price with NO fixed overhead is the
signature of a copy loop, not of DMA setup: setup would add a fixed term, and
small blocks would cost relatively more.

So it is exactly the illness we already treated in the state swap: 6.19 cycles
per byte with plain `memcpy` and 1.36 with the wide MVE copy. The difference is
not in the instructions — `memcpy` from newlib-nano for aligned pointers is an
unrolled ldr/str over 16 words — but in the latency of the M55 → npuRAM path:
an ldr/str loop holds one miss at a time, the MVE copy holds four.

So the fix is not in the graph but in the link: **define `memcpy` ourselves**.
A strong symbol displaces newlib's, and the wide copy reaches all the code —
including ST's files inside the STEdgeAI installation, which must not be
edited: editing there would mean the build does not reproduce for anyone but
us. No graph regeneration, the weights do not change, no NOR reflash.

Two subtleties, both of which would have cost debugging. First: there must be
no byte loop inside `memcpy` — GCC can recognise it as an idiom and replace it
with a call to `memcpy`, that is, to the same function. The tail shorter than
16 bytes is covered by the `vctp8q` predicate, in one pass, so there is nothing
to recognise. Second: the self-test can no longer check against `memcpy` — it
would be checking against itself. The reference is our own and byte-by-byte,
with `volatile` on the pointers, for the same reason.

**Measurement (4 Aug, late evening): `prof_err=0`, the copy self-test passed on
the board.** The direction and the order of magnitude came true, the
coefficient did not:

| | prediction | measured |
|---|---|---|
| `Concat`, 12 blocks | ~200 k | **343 459** (2.36 cyc/B) |
| `m55(hyb)` | ~460 k | **578 554** |
| hop, silence | ~2.0 M | **2 104 000** (min 2 093 506) |
| hop, held chord | — | **2 242 000** (max 2 272 053) |

Cycles per byte by block: 2.21–2.23 for the large ones (d=16, 32) and 2.74–2.88
for the small ones (d=1) — a monotonic rise towards the small ones. That is the
signature of a fixed per-call overhead plus short rows: at d=1 a state row is
176 bytes and the load pipeline does not have time to spin up. The same code in
the swap, on solid 3.7 kB chunks, gave 1.36 cyc/B; where the cycles between 1.36
and 2.36 go was not investigated: with a hop of 2.10/2.24 M against a 3.2 M
budget the headroom is about 30%, and further Concat work (DMA, or a graph
shape without concatenations) does not justify its own cost.

The peak on a chord change is 2 570 565: for the first time ALL the traffic is
under budget (it was 3.29 M). A side gain that confirms the mechanism of the
fix: the strong symbol reached all the code — `pqmf` 80.25 k → 69.4 k (there is
a copy of about 11 k inside), `build_xcond` 4.7 k → 3.6 k, Slice hybrids 262 k
→ 235 k (−10%). Inference (wall) in silence 2 540 k → about 1 848 k cycles
(3176 → 2310 µs), while playing about 1 970 k — the price of overlap is
unchanged. The slot accounting adds up in both modes.

## What is left

1. **`Concat` is still 343 k** — beyond this it can only be fixed outside the
   link: DMA or a graph shape without concatenations (the Windows side, a pass
   through ST's compiler). With about 30% headroom it is not a priority; come
   back to it if the article needs a figure for "the NPU stage with no software
   copies".
2. **`peak=980/1000` — CLOSED by verdict (the night of 4→5 Aug, full path).**
   Peaks before the mix: while playing `sk` 0.5–0.8, `rs` 0.37–0.92, `fir`
   0.23–0.59 — three comparable terms, and their sum naturally lives in the
   limiter. In silence `rs` = 13–20 thousandths — pure int8 quantization noise
   (the graph scales are applied correctly); on note changes `rs` goes
   transiently up to 1.59 FS (state reset plus a fresh input). This is the
   honest loudness of the design: the teacher A2_ottpress is OTT compression
   that lifts quiet material, so by construction the residual is comparable to
   the skeleton. The answer is not to "fix the scale" but to normalize the
   output before the limiter; the coefficient is chosen by ear in listening
   tests with a DAC. (First heartbeat after startup: peak=596/rs=183 — a
   startup transient.)
3. **`pqmf` 73.9 → 80.2 k after DTCM** — the question lost its subject: with
   the strong memcpy the counter is 69.4 k, below the original. The rise itself
   was never explained, but the counter is known to be sensitive to copy speed
   (about 11 k of copying inside).
