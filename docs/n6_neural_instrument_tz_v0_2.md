# Requirements spec: N6 Neural Instrument — v0.2

**Date:** 2026-08-01 · **Status:** after the M0 measurements · **Platform:** NUCLEO-N657X0-Q (STM32N657X0)
**Performance controller:** Arturia KeyStep mk1 (MIDI DIN)

## Changes v0.1 → v0.2 (from the H0/M0 results, the M0 report)
- **Network shape**: the canonical export is **gather2** (D-10): dilation is NOT a Conv
  attribute but explicit Slice taps + a channel-wise Concat + a 1×1 Conv (atonn 4.0
  emulates the dilation attribute with an S2D/D2S pyramid, roughly quadratic in C —
  up to 91% of t_call). d30/d16 are the historical T0 baseline.
- **The ballparks in §5.2/5.3 are replaced by measurements**: a matrix of 14
  configurations, three finalists (§5.2).
- **NOR**: 64 MB MX25UM51245G (D-12), not 128 MB/MX66UW1G45G.
- **Latency §5.4**: two branches — strict (T=48, as in v0.1) and tape trio (T=96, +4 ms
  residual, split latency). The choice is D-11.
- **M0 in §8 — PASSED** (1 Aug); the state-I/O risk is closed; a shape-freeze risk is added.
- §12 points to the decision log as the single canonical record of decisions.

---

## 1. Purpose and scope

A polyphonic hybrid real-time synthesizer: a deterministic DDSP skeleton
(harmonic bank + filtered noise) on Cortex-M55/Helium, on top of which a
neural multiband refiner on the Neural-ART NPU works as a layer of learned
degradation and texture (a "neuro-tape": wow/flutter, saturation, dropout,
path memory). The aesthetic target is not acoustics but synthetics: sparse
melodic lines + precision glitch in the spirit of Alva Noto × Sakamoto; fast
pitch jumps HF↔LF are first-class gestures.
Control is a MIDI keyboard. Key requirement of the project: **the NPU is
loaded close to its limit** (target ≥30% of the measured peak in steady state;
the practical peak was fixed in M0 — the convolutional kernel loads the NPU
pipeline; the formal metric comes with the choice of finalist, D-2/D-11).

**Non-goals (on principle):** acoustic realism, imitation of existing instruments.
**Non-goals for v1:** audio-driven input, USB-host MIDI, MPE, effects processing on the NPU,
case/enclosure, standalone power.

---

## 2. System architecture

(unchanged from v0.1 — M55 skeleton ∥ NPU residual, PQMF 4×12 kHz, SAI→PCM5102A;
the diagram in v0.1 §2 still holds)

Principles: the network is a residual on top of the skeleton (graceful degradation); the weights
are resident in AXISRAM (confirmed by M0: every finalist fits entirely on-chip); polyphony = the
H axis of the graph (batch-as-height, T0).

---

## 3. Hardware — unchanged from v0.1
(MIDI optoisolator H11L1, PCM5102A without MCLK, power from STLK; see v0.1 §3)

## 4. MIDI and control — unchanged from v0.1 (§4)

---

## 5. Synthesis

### 5.1 DDSP skeleton (M55/Helium)
As in v0.1, with the D-6 refinement (closed by a listening test on 1 Aug): the runtime
render is **variant B** (straight into the subbands, dsp/skeleton_b), A is for
offline work and data preparation.

### 5.2 NPU refiner — PER THE M0 MEASUREMENTS
- A causal TCN k=3, dilation cycles (1,2,4,8,16,32)×2, L=12; **export is gather2**
  (D-10, train/export_m0_d31.py::build_bh_gather2): for d>1 — Slice taps x[t],x[t−d],x[t−2d]
  + a channel-wise Concat + a 1×1 Conv (weights repacked); for d=1 — a native convolution.
  The dilations>1 attribute in ONNX is BANNED for atonn 4.0.
- Streaming through explicit states (I/O tensors, AXISRAM) — confirmed on the die.
- int8 via QAT (PTQ is banned); QAT ties the state_in/out scales together.
- The tape model is split as in v0.1 (wow/flutter/hiss on the M55; the NPU
  takes saturation and path memory).
- **Shape finalists** (µs/hop on a live die, jitter <0.25%; the choice comes
  from the T1 listening tests, D-2/D-11):

| Candidate | C | V | T | t_call/hop | headroom to 3.2 ms | latency |
|---|---|---|---|---|---|---|
| Wide duet | 88 | 2 | 48 | 2866 | 10% | per §5.4-strict |
| Thrifty duet | 80 | 2 | 48 | 2635 | 18% | per §5.4-strict |
| Tape trio | 80 | 3 | 96 | 3077 | 4% | §5.4-trio (+4 ms residual) |

  Not viable (MEASURED): C≥96 at V=3 in strict; C=128 at V=2; L=24 in realtime;
  a strict trio at ≤C88 even with ideal state offloading (D-13 closed, see decision_log).
- **RF compromise**: all finalists are L=12, RF 21 ms (half the v0.1 ballpark).
  If the T1 listening test shows a lack of "path memory": (a) accept it; (b) offline rendering
  of long notes with L=24, outside v1; (c) the last resort of §3.5 of the guide — long memory
  as an f32 block on the M55. Do not bring L=24 back into realtime (5045 µs/hop measured).

### 5.3 Budgets — from the actual stedgeai/M0 reports
| Resource | Actual (finalists) |
|---|---|
| NPU | t_call/hop 2.6–3.1 ms out of a 4 ms slot; the whole network memory on-chip. **Peak (practical, M0)**: the convolutional path ≈113 GOPS (invariant in C — a property of the engines; the marketing peak is not reachable with this shape). Finalist duty cycle 66–77% of the slot; useful MAC utilization 12–17%. **The §1 target metric must be reworded at the freeze (D-14)**: the candidate is duty ≥60% + the largest feasible shape, not a % of marketing GOPS |
| AXISRAM | weights 0.23–0.28 MB int8 + activations/states ~0.3–0.5 MB — out of 4.2 MB; the free remainder goes to M55 audio |
| M55 | the harmonic bank + PQMF synthesis + wow/flutter (§5.1) — headroom confirmed by the host scaffold in fw/ |
| Flash **NOR 64 MB (MX25UM51245G, D-12)** | FSBL · app · model bank A/B — space to spare |

### 5.4 Latency (note → sound) — TWO BRANCHES (D-11)
**Strict (the duet finalists, T=48):** as in v0.1 — median ~8.7 / worst ~10.7 ms.
**Tape trio (T=96): split latency** — the skeleton answers on the strict table
(the attack is instant), the NPU residual joins +4 ms later (a two-hop batch).
Artistic hypothesis: "late tape" is acceptable, even desirable, for the §1 aesthetic —
tested by auditioning fast attacks and glitch articulation (T1). If not — the duet.
D-3 (hop 4→2 ms) is rejected (M0): the problem is the opposite one —
amortization, not subdivision.

---

## 6. Software structure — unchanged from v0.1
(bare metal, a pipeline 1 hop deep, watchdog, underrun telemetry;
boot: dev mode into RAM, release from a signed FSBL in NOR; A/B weight banks)
Addition: the working position of the BOOT jumpers and the debugging
pitfalls — the H0 notes.

## 7. Training — unchanged from v0.1 (§7)
Addition: **training starts ONLY after the finalist is frozen** (the C/V/T/L shape
+ the gather2 structure); the torch export must produce a structure identical to the hand-built
graph (an stedgeai analyze check before training — as was done for T0).

---

## 8. Milestones and acceptance criteria

| Milestone | Status |
|---|---|
| **M0 — shmoo** | **PASSED 1 Aug 2026**: 14 configurations measured, state I/O confirmed, the rig ≡ DevCloud (1.5%), the finalists chosen (§5.2). Remaining: fixing the shape by listening test (T1) |
| M1 — skeleton | the v0.1 criteria stand |
| M2 — offline refiner | the v0.1 criteria stand |
| M3 — realtime | latency per §5.4 of the chosen branch |
| M4 — saturation | NPU load ≥ the §1 target while meeting §5.4 |
| M5 — instrument | a 30-min session with no engineer |

### 8a. Listening-test protocol for choosing the finalist (D-2/D-11)
**Stage 1 — latency (no training, right after M1):** the trio delay is simulated
on the skeleton: the residual stub = golden degradation (the wow/hiss chain of
§2.4 of the guide), mixed in with a delay of 0 / +4 / +8 ms.
Material: (a) staccato at 120–180 BPM with velocity contrast;
(b) fast pitch jumps HF↔LF (the signature gesture of §1); (c) glissando across the 6 kHz seam;
(d) long notes + an active pitch-strip; (e) varispeed/tape-stop.
**Trio criterion**: on (a) and (b) the late residual does not read as a separate event
("an echo of the attack"); if it does — the trio is out, the decision = the
duet, with no stage 2 for V.
**Stage 2 — texture (a short QAT, hours on the GTX 1060):** rough training of both
duet shapes (C=80 vs C=88) on a small corpus; blind A/B on material (d)+(e).
The winner is not the "richer" one but the one "closer to the §1 aesthetic".
The outcome of both stages is a shape freeze in decision_log.

## 9. System acceptance criteria
Strict branch: median ≤9 ms, worst ≤12 ms. Trio branch: the same figures for the SKELETON,
residual +4 ms (total ≤13/16 ms for the full timbre) — accepted only if
the T1 listening test confirms artistic acceptability (otherwise the duet). Note jitter ≤1 hop.
NPU load — measured. No seam artefacts (long notes + pitch-strip). 24 h soak.

---

## 10. Risks (updated)
| Risk | Status/mitigation |
|---|---|
| R-1 state I/O | CLOSED by M0 (archive: t0_report, m0_report) |
| Lowering of dilations | SOLVED: gather2 (D-10), the dilation attribute banned (archive: m0_report) |
| **Training a shape that is not frozen** | The MAIN risk right now: training only after the finalist is chosen; a shape change only through decision_log + a repeat M0 measurement |
| Thin trio headroom (4%) | If the trio is chosen — a mandatory headroom test with full M55 traffic (M3); the fallback is the duet |
| QAT int8 degradation | as in v0.1 |
| GPU 6 GB | as in v0.1 |

## 11. BOM — as in v0.1 (+ FDC2214 per D-8, ordered 15 Jul)

## 12. Decisions
The canon is **the decision log** (the single source). Snapshot as of 1 Aug:
D-1/5/6/7/8/10/12 decided; D-13 closed (do not do it); D-3 rejected;
D-2/D-11 — the finalist chosen by the T1 listening tests; D-9 — L=24 is dead in realtime
(if "path memory" turns out to be critical — offline rendering only, outside v1);
D-14 (the NPU load metric for §1/M4) — to be formulated at the shape freeze; D-1a, D-4 are open.
