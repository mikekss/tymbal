# Findings and pitfalls: STM32N657 + Neural-ART

A journal of facts paid for with debugging. One finding — one item, with how
it showed up and what confirms it. It is appended to ON THE SPOT, as soon as
the finding is made: what is not written down straight away is simply lost.
The order inside sections runs from hardware to tools; it is not
chronological.

## Memory and bringing the chip up

**The TCM has ECC, and the first READ after reset is a hard fault.** The
contents are random and the ECC bits do not match them. It is a precise fault;
the marker is PECC, bit 17 in AFSR. The area must be written once in WORDS
before any access; a byte-wise write will not do — in ECC memory that is a
read-modify-write, i.e. the same read again. For us this is `n6_dtcm_init()`
as the first statement of `main()`. (4 Aug, `docs/opt_npu.md`)

**The FSBL lives in the secure world**, so our DTCM sits at `0x30000000` and
not at the non-secure `0x20000000`. The same applies to AXISRAM — all
addresses are at `0x34…`.

**The AXISRAM3..6 banks are OFF after the bootloader.** The clock domain is
not started (`RCC->MEMENR`) and the banks sit in shutdown
(`RAMCFG_CR_SRAMSD`). The very first write there hangs the core — and that
write happens not in LL_ATON but in our own state `memset`. While the graph
shape contract did not add up, initialisation returned before that point and
the symptom stayed hidden. The symptom of a failure in general is not a
message but SILENCE: the bus fault turns into a hard fault before the terminal
comes up.

**The NPU as a bus master has to be started by hand through RIF** (secure +
privileged, CID 1), and RISAF has to be opened for the graph regions. The
`SystemIsolation_Config()` that CubeMX generates configures only GPDMA and
GPIO — there is nothing about the NPU in it.

**The graph does not touch npuRAM6 (`0x34350000`, 448 kB) at all** — free
memory next to the NPU, good for user buffers.

**A CPU copy into npuRAM costs 6–7 cycles per byte** with a plain ldr/str
loop, and this is NOT about instructions but about the latency of the M55 →
AXISRAM path: every next transaction waits for the previous one. An MVE copy
that keeps four loads in flight before the first store gives ~1.4 cycles per
byte — 4.5 times faster on the same data. Measured three times: the 43.7 kB
state swap — 6.19 → 1.36; the graph's `Concat` blocks — 6.997 → 2.36 (our own
`memcpy`, measured on the evening of 4 Aug). The price of a copy depends on
CONTEXT: for Concat, per block, it is 2.21–2.88 cycles/byte and rises towards
the small blocks — a fixed per-call overhead plus short rows (176 B at d=1)
that do not amortise the ramp-up of the load pipeline; for the swap the same
instructions on solid 3.7 kB chunks gave 1.36. Concat never got down to ~1.4;
with ~30% of the budget spare, it was not taken further.

**The AXI cache (CACHEAXI) on this profile is a penalty, not a speed-up:
+335 k cycles per hop.** Moving buffers between pools gave back only 7 k of
the 335, so it is not about placement. This agrees with ST's official mpools:
the internal npuRAM3..6 pools come WITHOUT `cacheable`, and `CACHEABLE_ON` is
set only on the external flash pool. Separately: the register
`RCC_MEMENR_NPUCACHERAMEN` does NOT EXIST AT ALL in `stm32n657xx.h`, there is
only `CACHEAXIRAMEN` — the "wrong power bit" hypothesis does not hold.

**The CPU cache and the NPU cache are different mechanisms.**
`LL_ATON_Cache_MCU_*` (→ `mcu_cache.c`) and `npu_cache_*` (→ CACHEAXI) are
always compiled in on the N6 and live independently. Turning the AXI cache off
does not break buffer coherency between the M55 and the NPU.

## The LL_ATON runtime

**`LL_ATON_RT_RunEpochBlock` is NOT blocking.** It advances the state machine
and returns. On our graph that is 13 209 calls over 69 blocks per hop, about
90 cycles each: if you just spin it in a loop, the CPU wastes 1.2 M cycles per
hop. The useful work has to be INSIDE that loop.

**The epoch profile is inside the runtime, just switched off.**
`LL_ATON_EB_DBG_INFO` in `ll_aton_config.h` (commented out there) adds a
number, a type and the compiler's estimate to every block —
`estimated_npu_cycles` and `estimated_tot_cycles`.

**`LL_ATON_RT_SetEpochCallback` is called at FOUR points in every block** —
before and after `start_epoch_block`, before and after `end_epoch_block` —
including for hybrid blocks, where `start_epoch_block == NULL` (the calls sit
outside the NULL check, verified against `ll_aton_runtime.c`). That gives two
quantities, and they must not be confused: **m55** =
`(POST_START−PRE_START) + (POST_END−PRE_END)` — CPU cycles inside the block,
and that is the price; **wall** = `POST_END−PRE_START` — the full time of the
block including the wait for the NPU, meaningless as a "cost".

**`LL_ATON_LIB_Concat` takes the slow path when the concatenation axis is not
the leftmost significant one.** The fast DMA branch is gated by
`axis_is_leftmost` — a check that ALL dimensions to the left of the axis equal
one. With the shape `[1, V=2, T, C=88]` and concatenation along the width,
there is a two to the left, the check fails, and the work goes to the generic
branch at the bottom of the function, which is just `memcpy` row by row. The
price on our graph: 1 019 501 cycles per hop, 32% of the budget. (4 Aug)
The fix is our own strong `memcpy` (MVE, four loads in flight) that displaces
the newlib one at link time: 1 019 501 → **343 459** per hop, `prof_err=0`
(measured on the evening of 4 Aug). As a side effect it reached all the code,
as intended: pqmf 80.2 k → 69.4 k, Slice hybrids 262 k → 235 k, build_xcond
4.7 k → 3.6 k.

**`npu_cache_clean_invalidate_range` starts with
`assert(hcacheaxi_s.Instance == CACHEAXI)`**, and the handle is set only in
`npu_cache_enable`. With the cache off, a Debug build stops on that assert,
even though the function itself correctly does nothing — there is an `if` on
the same condition inside as well. The assert is false; it goes away with
`-DNDEBUG` in the project properties.

**User-allocated IO: branch on the runtime field, not on the macro.**
`LL_ATON_NETWORK_USER_ALLOCATED_INPUTS` is declared in `network.h`, and if a
file does not include it, the branch silently compiles to nothing, the buffers
stay unbound, and the code runs off garbage pointers. The right marker is
`n->ib[0]->is_user_allocated`: that field is always there. The setters require
32-byte alignment and a size no smaller than declared; check the return value.

**The input and the output of a state pair can be merged into ONE buffer.**
The grounds are not general ones, they come from the generation report:
`state_out_k` is declared as `Slice(cat_k)`, that is a slice of the
CONCATENATION and not a slice of `state_in`. The graph reads `state_in` while
it builds `cat_k`, and only then writes `state_out` — the order is guaranteed
by the data dependency.

## The graph compiler

**`atonn` is a static x86-64 ELF and it sits INSIDE the Windows install**
of ST Edge AI (`<STEdgeAI>/Utilities/linux/atonn`), so it runs in a container.
Graph variants can be compiled with no Windows and no board.

**`atonn` is backend only.** 95 options, and none of them are frontend
switches: `--no-inputs-allocation`, `--no-outputs-allocation`,
`--inputs-ch-position chfirst|chlast`, `--outputs-ch-position`. Those belong
to `stedgeai generate` — the Python wrapper with ONNX preprocessing and
quantization. The working order: locally we throw out everything the epoch
table lets us throw out, and only what survives goes to `stedgeai`.

**The compiler's estimate for `Concat` is exactly the number of output
elements.** It counts a concatenation as one element per cycle. Useful: `est`
can be read as "how many bytes are moved", and the measured cycles can be
divided by it.

**`--Ox` is much worse for this shape:** 172 blocks and an estimate of
2 725 403 against 70 and 1 263 056 for the working profile.
`eliminate_concat_split`, `fuse_consecutive_concats_new`, `--ec-optimize` and
`-S` did not help either.

**`//` comments in the mpool are not parsed** — the file is read as strict
JSON.

**The QDQ form is untouchable:** `atonn` recognises int8 only in graphs that
come straight out of `ORT quantize_static`; the matcher rejects hand-made QDQ
and post-hoc surgery.

**The `dilations > 1` attribute is forbidden in ONNX:** `atonn` 4.0 emulates
it with an S2D/D2S pyramid whose cost is roughly quadratic in the number of
channels. The canonical form is gather2: explicit Slice taps + a channel
Concat + a 1×1 Conv.

## Tools and environment

**QEMU as a Cortex-M55 rig.** Machine `mps3-an547`, the same MVE decoder,
`-icount shift=0`. It gives INSTRUCTIONS and a numeric check of the vector
path with no board; it does NOT give absolute cycles (neither the cache nor
dual issue is modelled). The ratio "board cycles / QEMU instructions" is a
working diagnostic: around one means we are limited by the operation count,
much larger means memory.

**`objdump` 2.42 decodes MVE wrongly.** `ldc 15, crN, [rM], #16` is
`vldrw.u32`, `stc 15, …` is `vstrw`, `cdp 14, 3, …` is `vmul`/`vfma` with a
scalar operand. The immediate offset is shown multiplied by 4. Reliable signs
of vector code: ldc/stc with a stride, cdp nearby, `le lr` at the end of the
loop, and `__ARM_FEATURE_MVE == 3` at build time.

**Nano-printf SILENTLY skips `%f`.** Newlib-nano (`--specs=nano.specs`) does
not enable float formatting without the linker flag `-u _printf_float`:
printf with `%f`/`%.2f` prints NOTHING, with no compiler warning and no
garbage — the field simply disappears from the line. The project never ran
into it because all the telemetry is integer; the very first float banner
(D-19) came out empty. The fix in the spirit of the project is to print
integers (cents, hundredths) rather than drag float-printf into ROM (+~8 kB)
for a banner. (5 Aug)

**`-mcmse` is mandatory** in the syntax check of the target files, otherwise
`RISAF2_S` and its neighbours are undeclared.

**The newlib-nano `memcpy` is NOT a byte loop** for aligned pointers, it is an
unrolled ldr/str over 16 words (I looked at the `libc_nano.a` disassembly). It
is not slow because of the instructions — see the item about path latency into
npuRAM.

**A dot product with a backward step through memory costs 6.5 cycles per
MAC.** The 65-tap FIR dot product `acc += w[k]*s[-k]` cost 652 k cycles per
hop in scalar form (board, fir slot); GCC does not unroll that loop, and the
backward direction does not help the prefetcher. The fix is to reverse the
WEIGHTS once at recompute time (w_rev[j] = w[K-1-j]): the dot product becomes
forward×forward over two contiguous arrays, and MVE opens up (64 taps in
groups of 4 f32 + a scalar tail). The branches are cross-checked with
qemu-ck4. Measurement of the MVE branch on the board (the same night):
652 k → **245 k** per hop, 2.45 cycles/MAC over the 99 840 MACs of two voices.

**Globals in a hot loop cost real money.** Moving three variables from locals
to file-scope statics cost 16% of the skeleton: the compiler has to re-read a
global on every iteration, because it cannot prove that the function it calls
does not touch it. The fix is a copy into a local variable with a write-back
at the end.

**CubeIDE does not see files added to `.project` from outside.** The project
model lives in the IDE's memory, and at build time it regenerates
`Debug/**/subdir.mk` from that model. The only fix is a re-import: Delete
(without deleting from disk) → Import → Existing Projects. Editing EXISTING
files and adding `-D` to the project properties are safe — the file list does
not change.

## The TPA6120A2 amplifier runs on a split supply only (6 Aug, from board photos)

What was visible: the CIRMECH WM-004 amplifier board with an "AC GND AC"
terminal block — the temptation is to feed it ordinary 12 V DC from an
off-the-shelf adapter.

What it turned out to be: the board carries both a 78M12 AND a 79M12 — ±12 V
rails, so the secondary has to be centre-tapped (2×12 V AC). A single DC
supply gives only the positive rail: the amplifier either stays silent or
drives DC into the headphones. Check the regulator markings BEFORE applying
power — the TO-252 parts are labelled on the silkscreen (78M12/79M12), and
that is cheaper than guessing from the connector.

General rule: the power connector tells you the voltage but not the
polarities; read the rail topology from the regulators, not from the terminal
block.

## A block spec with no supply is not a spec (6 Aug, a lesson)

What was visible: the audio path was "written up" — the I2S pinout checked
against msp.c and UM3417, the DAC module chosen, the amplifier already in my
hands. A feeling of readiness.

What it turned out to be: the amplifier has a split supply, and there was no
source for it in the project, nor was it on the shopping list — the question
had been deferred "until we have a photo of the board". The surprise surfaced
at a stage where the hardware was already bought.

Rule: for any block in the path the spec consists of TWO paths — signal and
supply; if only the signal one is described, the block is not specified. The
symptom of the disease is the phrase "we will work the supply out later /
from the markings". For an MCU we would not accept that for a second (there
the supply is page one of the datasheet), but for external circuitry we did.

The fix in this project: docs/audio_bom.md — a full list FROM the board TO the
ears, with an explicit line for "what is missing".

## PCM5102A: tie SCK to ground, otherwise the PLL does not start (5 Aug)

What is visible: on the GY-PCM5102 module the SCK pin sticks out on the pin
header, and we have no MCLK — the temptation is to simply not connect it.

What it turned out to be: a floating SCK is NOT "there is no master clock, we
work without one". The chip brings its system clock up with an internal PLL
from BCK only when SCK is pulled to ground; a floating input leaves the PLL
unstarted — silence or noise, and it diagnoses horribly (the I2S is perfectly
valid all the while). On the ChipStudio "Audio Artwork PCM5102" board SCK is
not brought out — it is grounded on the board.

Consequence for the firmware: MCLK is not needed at all, the SAI puts out
three wires (BCK 3.072 MHz, LRCK 48 kHz, DIN) — standard I2S at 64·fs.

## The SSM3582 is not a DAC but a class-D amplifier with a digital input (5 Aug)

What is visible: in the shop it sits next to the DACs, "I2S — Audio", priced
like a DAC.

What it turned out to be: its input is I2S, but its output is bridged PWM,
2×31 W straight into a speaker; there is NO line output. Do not plug it into
the line input of somebody else's amplifier: what comes out is a square wave
of PVDD amplitude at hundreds of kHz.

From the ADI datasheet, what matters for us: there is a standalone mode with
NO I2C (slot, mono/stereo and fs are set by the ADDRx pins — no driver
needed); MCLK is not needed, BCLK is auto-detected from 2.048 MHz (our 3.072
passes); I2S 24 bit, 8–192 kHz — our frame with no changes; no output filter
is needed (filterless ΣΔ); PVDD 4.5–16 V; loads from 3 ohm stereo / 2 ohm
mono.

The niche: "the instrument is a box with its own speaker, there is no analogue
path". The danger: in standalone mode it has NO volume control — the only knob
in the whole path is our software multiplier, and the output goes up to 31 W.
The first power-up is at −40 dB and into a speaker you can afford to lose.

## Compute path levels from sensitivity, not "by ear" (5 Aug)

What is visible: the DAC gives 2.1 V RMS, the amplifier is a "headphone" one —
it looks compatible by definition.

What it turned out to be: the AKG K512 MkII is 32 ohm, 109 dB SPL/V. A
comfortable 85 dB is 0.063 V at the headphones; the DAC puts out 2.1 V and the
amplifier multiplies further (at the gain of 2 typical for the TPA6120A2, up
to 4.2 V = 121 dB SPL and 0.55 W into headphones with a limit of ~0.2 W).
There are ~30 dB too many in the path, and the whole working range of the knob
is squeezed into the first tenth of the travel of a linear B50K — where its
channel imbalance is worst. Three lines of arithmetic
(V = 10^((SPL−sens.)/20)) tell you this in advance, and also that the first
power-up happens without the headphones on your head.

## `peak=980` was not loudness, it was the limiter ceiling (7 Aug)

What was visible: in ALL the board logs across the whole project `peak` showed
980 — on a single note, on a chord, and at different key pressures. On 4 Aug
we read that as "honest loudness, we will defer normalisation until the DAC".

What it turned out to be: 0.98 is `LIM_THR`, the threshold of the output
limiter (`pipeline.c`). The indicator was not showing the signal level, it was
showing that the limiter was pinned. The raw path (measured with a host
render, limiter disabled) peaks at **1.77** on a chord at velocity 127 and
**1.40** on a single note — so the limiter, with its instant attack and 80 ms
release, engaged at vel≈60 on a chord and vel≈85 on a note, practically
always. By ear through the DAC this is exactly what I heard as clipping on
hard presses (7 Aug), and measurably it was a complete loss of velocity
dynamics above 90: the peak stood at 0.980 at any strike force.

Rule: **a saturating indicator must not be read as a measurement.** If a
quantity keeps hitting the same number under different input conditions, that
is not a level measurement, it is a limiter indicator. Checking it costs one
run with the limiter turned off.

The price of the mistake here: for three days we considered the level "honest"
and planned to pick it by ear, when it would have been enough to look once at
where the 980 comes from. The fix is D-22.

## My own rules: LF/CRLF (7 Aug, minor)

The first attempt to edit `n6_config.h`/`pipeline.c` in the container silently
did nothing: the file is CRLF, and I wrote the replacement pattern with `\n`.
`str.replace` does not fail when it finds nothing — it just returns the
original string, and the build passes "successfully" with the old code. This
is caught by comparing md5 BEFORE and AFTER, not by eye. The rule itself had
been written down long before, but I applied it to reading and not to the
search pattern.

## "Sand" on every voice: the skeleton noise has no spectral shape (7 Aug)

What was visible: after the first live playing — quite a lot of noise, sand
running alongside every sounding voice. The key word is "alongside": the
noise is tied to the voice and not to the background, so it is synthesised,
not picked up.

What it turned out to be (`skeleton_b.c`):

    sub[b][i] += acc[b] * ag + SKB_NOISE_B * u * tb * ag;

The noise component = 0.15 × timbreB × envelope, WHITE and with the SAME
coefficient in all four PQMF subbands. The harmonics meanwhile fall off with
frequency along their natural tilt — so in the top band the tone has almost
died out already, while the noise runs at full strength.

Measurement (note A3, vel 100, preset D-19, the noise isolated by subtracting
a render at timbreB = 0, the PRNG made deterministic):

| CC1 | timbreB | noise/signal | noise/signal ABOVE 4 kHz |
|---|---|---|---|
| 0  | 0.000 | none at all (exact zero) | — |
| 5  | 0.039 | −30.5 dB | −10.7 dB |
| 10 | 0.079 | −24.5 dB | −5.7 dB |
| **19** | **0.150 (default)** | **−19.0 dB** | **−2.4 dB** |
| 40 | 0.315 | −12.7 dB | −0.7 dB |

Broadband, −19 dB looks decent — and that is exactly why the problem was not
caught earlier. But the ear judges hiss by the top end, and there, at the
default setting, the noise is only 2.4 dB below the tone. Half the
high-frequency energy of every note is white noise.

Two separate facts worth keeping apart:
1. `timbreB` is initialised to **0.15** (`voice.c:10`) and edited over CC1.
   So the "sand knob" already exists — it is the modulation wheel, and at zero
   the noise disappears EXACTLY (not "almost").
2. The noise has no shape at all. In canonical DDSP the noise branch is
   filtered noise with a learned envelope across the bands; ours is one
   constant for all bands. This is not a setting, it is a missing stage.

Rule: **the broadband noise/signal ratio does not describe the audibility of
the noise.** It has to be measured in the band where the ear looks for it; for
us the difference between the two ways of counting is 16.6 dB.

## The network ADDS high-frequency noise, and the metric does not see it (7 Aug)

What was visible: a live A/B on button B1 (D-24). My verdict on the
network — more sand, but it is not critical, I trim it a bit with the
modulation strip.

Why this matters: we assumed that all the "sand" came from the skeleton (white
noise with no spectral shape, see the entry above). The A/B showed there are
TWO sources: the skeleton lays down the base, the network adds on top.

Likely mechanism (HYPOTHESIS, not measured): quantization of the residual to
int8. The quantization step produces a broadband sprinkle of roughly constant
amplitude; in the upper bands, where the wanted signal is small, that is what
is heard as sand.

The main thing here is not the fact itself but that **our metric did not see
it**. The price of quantization by eval_chain is −0.27 dB of spectral
distance, that is "almost free". A flat broadband sprinkle barely moves the
band-averaged distance, and the ear looks for it exactly where there is least
signal.

This is the SECOND case in one day where a broadband metric underestimated the
audibility of noise (the first was the skeleton noise component: −19 dB
broadband against −2.4 dB above 4 kHz). The two cases add up to a rule:
**compute the noise metric in the band where the ear looks for it, not over
the whole band at once.** For eval_chain that means adding a per-band
breakdown, not only the summary figure.

Testing the hypothesis (needs the Python side, `train/`): run eval_chain with
the residual in fp32 and in int8, and compare the HF part of the difference
against the teacher band by band. If the excess sand disappears in fp32, then
quantization is to blame, and from there it is a question of the quantization
scale rather than of the design.

## A remembered parent hash drops commits out of history (8 Aug)

What was visible: on re-reading an earlier list of defects in the repository,
half of them were already fixed. I explained that by the list having been drawn
up from an incomplete picture of the project, and wrote it up in the journal.

What it turned out to be: the disputed files had been worked on between the
list and the re-reading. It could be checked with one command — `git log` on
those files — which I did not run, because I was comparing "now" against "the
list" and never asked what happened BETWEEN them.

Worse: the same blindness cost history. When I resumed work, I built commits
off the remembered parent instead of `.git/refs/heads/<branch>`. Four commits
dropped out of the graph. The content survived by accident — it was sitting in
the index, and `write-tree` picked it up, which is why my commit "DAC pinout"
actually contains 2990 changed files and −940 MB. The history was restored by
a merge (`eff1bfd`), the tree did not change.

Two rules: **read the parent from the ref before every commit** and **explain
a divergence from an earlier report with history, not with a hypothesis**. A
plausible explanation accepted without checking is worth nothing: the check
here cost one command.

## PCM5102A filter delay: 21/fs, not "about 0.4 ms" (8 Aug)

The latency budget had "~0.4 ms (DAC)" in it from the very start of the
design — a plausible figure, but backed by nothing. Cross-checked against the
datasheet: the group delay of the x8 interpolation filter is given as 20/fs on
p. 4 and as 22/fs in table 4 (a contradiction inside the datasheet itself),
and TI's measurement on the EVM gives 21/fs. At 48 kHz that is **0.44 ms** —
so the figure we had put in "as typical" turned out to be right, but now it is
not an assumption.

A side note from the same source: the group delay can wander by up to 7.5 BCK
periods at power-up, because of the asynchronous crossing between the BCK and
SCK domains. Not significant for us (37 µs), but it explains why numbers from
different power-ups may fail to agree down to the microsecond.

The rule is the old one: a "typical value" is a placeholder, not a number. The
check cost one lookup in the datasheet.

## Running the project's python scripts: the system `py`, not the venv (8 Aug)

`C:\ST\venv-n6` has NO torch; the training and listening-test scripts run on
the system `py` 3.12.10 (torch 2.4.1+cu118, numpy 2.5, soundfile). The
directory you start from does not matter: the scripts resolve paths from their
own file.

A console pitfall: PowerShell does not always give stdout the same encoding —
a shell spawned non-interactively emits cp1252, and printing Cyrillic then
kills the script with `UnicodeEncodeError`. It does so AFTER the computation,
at the output stage, so it looks like "it crashed" even though the work is
done; what is lost is the printing and everything written after it (in
audition_delta, the wav files themselves). Fixed with
`$env:PYTHONIOENCODING="utf-8"`. In the interactive Russian console it does
not show up at all: this is an ENVIRONMENT pitfall, not a project one.

## A clamp does not catch NaN: with NaN every comparison is FALSE (8 Aug)

The output packing looked protected: the peak is computed, a clamp at
±0.999969 is in place, there is a clip counter. NaN went straight through,
because for NaN `x > 1.0f` and `x < -1.0f` are **both false** — the clamp does
not touch it, and `(int32_t)NaN` is already undefined behaviour, so an
arbitrary bit pattern goes off to the DAC. A range limiter is NOT protection
against NaN: it protects against large numbers, and NaN is not large, it is
incomparable.

The fix is to negate the range: `if (!(x >= -1e3f && x <= 1e3f)) x = 0.0f;` —
one condition catches NaN, both Infs and absurd values. With a counter, not
silently: the mere appearance of a NaN is a diagnosis of the network or of the
states, and I want to know that it happened.

Real UB was removed in the same place: `(int32_t)(x*8388607.f) << 8` — a shift
of a NEGATIVE signed number (C11 6.5.7p4). The compiler has so far done "what
was expected", but that is its goodwill, not a guarantee. We assemble the bit
pattern in a `uint32_t`, where the shift is defined.
