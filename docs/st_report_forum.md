# Report #1 — ST Community forum (LL_ATON runtime and the atonn compiler)

**Where:** community.st.com → the **Edge AI** section (if you can't find it —
"STM32 MCUs Embedded software"). *Ask a question* / *Start a topic* button.
**Tags (important, ST staff search by them):** `STM32N6`, `Neural-ART`,
`ST Edge AI Core`, `LL_ATON`, `Bug-Report`, `Performance`.
**Topic title:** copy it from "Subject" below.
**Language:** English — the forum is international, the engineers who answer
are in France and Italy.

Post it as one topic: the two findings share one graph and one build.
Below is the finished text, paste it as is.

---

**Subject:** `LL_ATON_LIB_Concat` falls back to 4× memcpy when the concat axis
is not leftmost — 32% of a 4 ms real-time budget on STM32N6

Hardware: NUCLEO-N657X0-Q (STM32N657X0H3Q), Cortex-M55 @ 800 MHz, Neural-ART.
Tools: ST Edge AI Core 4.0, STM32Cube_FW_N6 V1.4.0, STM32CubeIDE 2.1.1.
Application: real-time audio, hop = 192 samples @ 48 kHz = 4 ms = 3.2 M cycles,
hard deadline, `underrun` must stay 0.

## 1. Concat falls back to a byte loop, and it is one third of the budget

Our graph is a streaming TCN (int8, L=12, C=88, T=48) run per audio hop with
two voices, so the tensor layout has the voice axis leftmost: `[C][V][T]`.
The graph contains 12 `Concat` blocks (channel-wise taps).

Epoch profiling (`LL_ATON_EB_DBG_INFO`, DWT cycle counter on the M55) showed
those 12 blocks costing **1 019 501 cycles per inference — 32% of our entire
4 ms budget.** No compiler option changed the number.

The cause is in the runtime source: `LL_ATON_LIB_Concat` takes its fast DMA
path only when the concatenation axis is the leftmost *significant* axis. With
V=2 sitting to the left, that test fails and the block is executed by the
generic branch — four `memcpy` calls per block.

The arithmetic confirms it exactly. The 12 blocks move 145 728 bytes per hop
(this equals the compiler's own cost estimate, which counts Concat as one
element per cycle):

```
1 019 501 cycles / 145 728 bytes = 6.997 cycles per byte
```

and that figure is **identical across all six different block sizes** — the
signature of a copy loop, not of a DMA setup cost.

### Workaround, and why it is unsatisfying

We overrode `memcpy` with a strong symbol: an MVE implementation issuing four
16-byte loads before the first store, tail handled by predication. The strong
symbol displaces the newlib one and reaches all code including ST files we
must not modify. Result:

```
Concat: 1 019 501 -> 343 459 cycles   (6.997 -> 2.36 cycles/byte)
hop (silence): 2 826 500 -> 2 104 000 cycles
```

That is a 676 kcycle saving — 21% of the hop budget — recovered by replacing a
libc function under the vendor runtime. It works, but overriding `memcpy`
globally is a heavy hammer for an application to swing, and it silently
changes behaviour for every ST driver linked into the image.

### What we would ask for

1. Extend the fast path so it applies when the concat axis is contiguous in
   memory, regardless of whether it is leftmost.
2. Failing that, **document the gating condition** in the ST Edge AI docs, so
   that graph authors know the layout choice has a 30%-of-budget consequence.
3. Optionally: have the runtime use an internal optimized copy instead of libc
   `memcpy`, so applications do not have to override a standard function to
   get reasonable throughput.

Point 2 alone would have saved us several days: nothing in the documentation
suggests that the position of an axis changes Concat cost by 3×.

## 2. `dilations > 1` unsupported in ONNX import — pyramid emulation is ~9× over budget

Same setup. A causal TCN naturally uses dilated convolutions (1..32 across two
stacks). `atonn` 4.0 rejects `dilations > 1` on Conv and emulates dilation with
a pyramid of reshapes. On our shape the resulting graph executed in
**37.36 ms against a 4 ms hop — about nine times over budget.**

We worked around it by expressing dilation explicitly: `Slice` taps + channel
`Concat` + 1×1 `Conv` (we call this the "gather2" form). Mathematically the
same operations; roughly **10× faster** on this target.

This is presumably a known limitation, but the size of the gap seems worth
reporting: the naive export path is not slightly slower, it is unusable, and
the working form is not obvious. A note in the documentation — "express
dilation as explicit slices; the pyramid emulation is not intended for
real-time" — would be valuable.

## Reproduction

Both numbers come from on-target measurement with the DWT cycle counter and
per-epoch profiling, not from estimates. I can share the graph shape, the
epoch profile dump, and the memcpy implementation if that helps.

---

### What to expect

A forum reply does not always come quickly, but topics tagged `Bug-Report`
do get read. If it stays silent for a week, duplicate it as an **online
ticket** (you have an ST account): link to the forum topic from the ticket,
that way they get tied into one story. The Concat item is the only one worth
pushing hard: it is about the performance of a closed runtime, and nobody but
ST will fix it.
