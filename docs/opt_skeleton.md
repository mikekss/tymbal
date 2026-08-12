# The skeleton: five rounds of optimization and how they ended

Journal, 2–4 Aug 2026. The subject is `fw/src/skeleton_b.c`, rendering a voice
straight into the PQMF subbands. The hop budget is 3.2 Mcyc at 800 MHz; going
into this work the skeleton cost 1.11 Mcyc on a sounding triad, that is, a
third of the budget for a stage that ought to be cheap.

The value of this text is not the −45%, but **which lines of reasoning did not
work.**

## What the hot loop computes

For every subband sample (48 per hop) and every live harmonic (up to 100):

```
acc[b] += a_h * (w_re[h][b]*sin(h*phi) + w_im[h][b]*cos(h*phi)),  b = 0..3
```

`w` is the response of the four PQMF bands at frequency `h*f0`, interpolated
from the `n6_bandresp` table (961 nodes with a 25 Hz step). Four bands are
exactly one f32x4, so all the vectorization runs along `b`.

## Round 1 (2 Aug): interleaving the table

**The reasoning.** `re` and `im` sat in two arrays of 15.4 kB each, and each
harmonic meant four scattered accesses. Merge them into one array `g_br8`,
where node `i` stores `{re[0..3], im[0..3]}` back to back: a pair of nodes for
interpolation becomes 64 contiguous bytes.

**Prediction:** a gain of several times. **Measured:** −16%.

The change is correct (the numbers are bit-for-bit the same, locality really is
better), but the scale was predicted off the top of my head.

## Round 2 (3 Aug): swapping the loops

**The reasoning.** The table was combed through in full for every subband
sample — 96 passes per hop. Move the harmonics outward, in blocks.

**Measured:** `skb` 1.11 → 0.93 M. Bringing back the scalar prologue and the
branchless body after that gave **nothing** (0.93 → 0.92).

The conclusion drawn correctly at the time: since neither memory, nor
dependencies, nor branching gets in the way any more, the limit is the number
of operations. The conclusion drawn incorrectly at the time: that the number of
operations is the bottleneck everywhere.

## Round 3 (3 Aug): hoisting the response interpolation out of the sample loop

**The reasoning.** `w_re`/`w_im` depend on the sample only through `f0`, and
inside a hop that is 4 ms: with a 25 Hz grid step `f0` drifts by fractions of a
hertz. Compute the response once per hop.

**The first attempt failed numerically:** rel RMS 1.9e-02. The error in
reasoning is narrow and instructive — I was thinking about the drift of `f0`,
but what matters is the drift of `h*f0`: at the hundredth harmonic that is tens
of grid nodes. The fix is segments: the response is taken at the `f0` in the
middle of a segment of `SKB_WSEG` samples. The curve was measured on the host:

| SKB_WSEG | rel RMS | dB | share of work |
|---|---|---|---|
| 1 | 6.68e-06 | −103.5 | 100% |
| 2 | 6.82e-04 | −63.3 | 63% |
| **4** | **1.51e-03** | **−56.4** | **44%** |
| 8 | 3.02e-03 | −50.4 | 34% |
| 16 | 6.29e-03 | −44.0 | 27% |

4 was taken: 13 dB below the already accepted A-path equivalence contract
(−43.5 dB), while the work falls by more than half.

**The second attempt made things worse** (0.92 → 1.76 M): the precompute
function was copied from the scalar branch and reached into the
non-interleaved table — exactly what round 1 was moving away from. After the
fix: `skb` 0.759 M.

## Here the work stopped and the counters appeared

Three rounds in a row missed on scale, and in different directions. They have
one thing in common: the change was designed against a cost model in my head.
So the next step was three cycle counters inside `skb` (`n6_skb_prof`):
response precompute, prologue, vector body. The very first measurement:
227 k / 170 k / 186 k and **176 k of "remainder"**, that is, almost a quarter
of the skeleton lived outside all three counters.

## Round 4 (3 Aug): libm off the hot path

`sinf`/`cosf` are two newlib calls per sample, 96 times per hop, and they do
not inline. The phase is stored as a uint32 of a full turn anyway, so a table
is natural here: the top 10 bits are the index, the next 12 the fractional
part. The linear interpolation error over 1024 nodes is about 4.7e-06, around
−106 dB. `llrint` was replaced by adding 0.5 and a single VCVT.

**Prediction −76 k, measured −76 k.** The first prediction in two days that
came out right — precisely because it came from a counter and not from a model.

## Round 5 (4 Aug): `exp2f`, divisions and two more counters

`exp2f` in the decoder is 200 calls per hop. Replaced by the decomposition
`2^x = 2^n * 2^r`: `2^n` is assembled from the float exponent field, and `2^r`
on [−0.5, 0.5] is a fifth-degree polynomial (Chebyshev, coefficients rounded to
f32). Cross-check against libm over a whole CK4 run: **−139.4 dB**.

Along the way, in the phase catch-up `m/192` and `m%192` sat inside the loop
over `m` — four uint64 divisions by a constant for every subband sample. It
became one division per sample with an incremented remainder; **verified
bit-for-bit**, the CK4 blob matched byte for byte.

Counters `dec` and `ph` were added. And here is the most useful thing of the
day:

**`dec` = 13.8 k cycles, that is 1.8% of the skeleton.** The prediction was
"about 60 k". A miss by a factor of three, and again in the same direction: the
cost of a libm call was estimated by feel. The change stayed (it is free and
makes the file cleaner), but there was no money in it. What is useful here is
the counter, not the change.

## The rig: a Cortex-M55 model in QEMU

From here it was clear that guessing further was not an option, and that the
verification loop meant a firmware flash and somebody else's attention.
`qemu-system-arm` supports `mps3-an547`: the same Cortex-M55, the same MVE
decoder. A bare-metal rig was built (`fw/test/qemu/`): `make qemu` counts
instructions over the same sections, `make qemu-ck4` runs the CK4 score through
the **vector** branches and cross-checks it against the host's scalar
reference.

Two consequences right away:

**1. The vector path is checked numerically without a board for the first
time.** `make test` runs the scalar path — there is no MVE on x86. The
threshold is the same 1e-4, and it holds at 3.06e-06.

**2. Cycles per instruction.** Put the board measurement and the QEMU
measurement side by side over the same counters:

| section | instructions (QEMU) | cycles (board) | cycles/instruction |
|---|---|---|---|
| `wt` | 92 010 | 229 972 | **2.8–3.7** |
| `pro` | 68 282 | 111 270 | 2.4 |
| `bod` | 96 545 | 130 078 | 2.0 |
| `dec` | 15 671 | 13 784 | ~0.9 |
| `ph` | 29 690 | 30 965 | ~1.0 |

(QEMU was run on three voices, the board on two — the numbers have been brought
to a comparable basis and are good for the ratio, not for absolute values.)

Where the ratio is about one, the bottleneck is the number of operations. Where
it is much higher, it is memory. **`wt` is the only outlier, and that is proof
that three rounds were fixing its formulas while the table was what got in the
way.** The spread of `wt` across the chord register (110 k–230 k) confirms it:
the table step `step_ix = f0/25` grows with frequency, and the working set of a
pass falls out of the cache.

## Round 6 (4 Aug): a vector prologue

The prologue turned out to be **more expensive than the vector body it feeds**:
13 instructions per (harmonic, sample) pair against 9 for the body. It was
scalar for one reason — the `sin(h*phi)` rotator is recurrent in `h`, and I did
not want to drag that dependency chain into a vector.

The recurrence splits into four: start not from one angle but from four
(φ, 2φ, 3φ, 4φ) and rotate them together with a step of 4φ — that is exactly
one f32x4 and the same sequence of values. The seeds are taken from the table
**at the exact phase**: φ sits in a uint32 where 2^32 == 2π, so 2φ, 3φ, 4φ are
ordinary multiplications with modular overflow, with no loss. The chain became
four times shorter, so the accuracy is higher too.

The amplitude is folded in at the same time in the prologue: the body gets
ready-made `p_h = a_h sin(hφ)` and `q_h = a_h cos(hφ)` and accumulates with two
`vfma` instead of `vmul+vmul+vadd+vfma`, using two independent accumulators.

| | predicted (QEMU) | measured (board) |
|---|---|---|
| `pro` | −66% | **−38%** (179 880 → 111 270) |
| `bod` | −28% | **−26%** (176 110 → 130 078) |
| `skb` | −33% | **−15%** (709 621 → 605 149) |

The body matched exactly; the prologue is twice as bad as the prediction, and
the reason is clear: there are two thirds fewer instructions, but now they are
all vector, and a vector operation on the M55 takes two cycles against one for
a scalar one. **This is a limitation of the rig that has to be kept in mind: it
counts work, not time.** The right way to use it is to predict the sign and the
order of magnitude, and take the coefficient from the board.

## Result and what is left

`skb` on the worst chord: **1.11 M → 0.605 M cycles** (−45%). The largest
remaining piece is `wt` at 230 k, and it is a memory problem, not an arithmetic
one: the fix is the format of the `g_br8` table (30 kB), for example float16
with expansion on the fly. The accuracy of such a change can now be checked
locally, `make qemu-ck4`.

But the main conclusion of this round is outside the skeleton. The NPU stage
costs 2.53 M M55 cycles, the same in silence and while playing, that is, 79% of
the budget goes before the first note. Even a zero skeleton leaves the hop at
3.11 M against a budget of 3.2 M. **Further skeleton optimization does not
close the budget and should not be a priority** — the priorities are in
the handover notes, §7.
