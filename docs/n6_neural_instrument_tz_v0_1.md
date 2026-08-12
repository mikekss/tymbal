# Requirements spec: N6 Neural Instrument — v0.1 (draft)

**Date:** 2026-07-15 · **Status:** draft to open the branch · **Platform:** NUCLEO-N657X0-Q (STM32N657X0)
**Performance controller:** Arturia KeyStep mk1 (MIDI DIN)

---

## 1. Purpose and scope

A polyphonic hybrid real-time synthesizer: a deterministic DDSP skeleton
(harmonic bank + filtered noise) on Cortex-M55/Helium, on top of which a
neural multiband refiner on the Neural-ART NPU works as a layer of learned
degradation and texture (a "neuro-tape": wow/flutter, saturation, dropout,
path memory). The aesthetic target is not acoustics but synthetics: sparse
melodic lines + precision glitch in the spirit of Alva Noto × Sakamoto; fast
pitch jumps HF↔LF are first-class gestures. Control is a MIDI keyboard. Key requirement of the project:
**the NPU is loaded close to its limit** (target ≥30% of measured peak
performance in steady state, the exact figure fixed by the M0 results).

**Non-goals (on principle):** acoustic realism, imitation of existing instruments.
**Non-goals for v1:** audio-driven input (microphone/guitar), USB-host MIDI, MPE,
effects processing on the NPU, case/enclosure, standalone power.

---

## 2. System architecture

```
KeyStep mk1 ──DIN MIDI──> opto H11L1 ──UART 31250──> MIDI parser (ISR)
                                                        │ SPSC FIFO
                                                        v
                                              Voice manager (250 Hz)
                                              control frames {f0, amp, tA, tB, gate} × N voices
                                                        │
                    ┌───────────────────────────────────┴──────────────────┐
                    v                                                      v
        M55: DDSP skeleton                                   NPU: multiband refiner
        harmonics + noise, 4 subbands                        dilated TCN, residual,
        (PQMF domain, 12 kHz × 4)                            batch = voices, state-passing
                    │                                                      │
                    └────────────> sum (skeleton + residual) <─────────────┘
                                            │
                              M55: PQMF synthesis (32-tap polyphase)
                                            │
                              SAI TX DMA ──> PCM5102A ──> line out
```

Principles: the network is a residual on top of the skeleton (graceful degradation: without
the NPU the instrument still sounds, just flatter); the weights are resident in AXISRAM;
polyphony = the batch dimension of the graph.

---

## 3. Hardware

### 3.1 Parts
| Block | Implementation |
|---|---|
| Compute | NUCLEO-N657X0-Q |
| MIDI input | DIN-5 → H11L1 (subst. 6N138) → USART RX @ 31250 8N1 |
| Audio output | PCM5102A (I2S, no MCLK — internal PLL), line out |
| Assembly | Nucleo-144 proto-shield / breadboard |
| Power | USB-C from STLINK; the KeyStep from its own USB supply, data over DIN only |

### 3.2 MIDI input (schematic)
```
DIN pin 4 ──[220R]──> H11L1 anode
DIN pin 5 ──────────> H11L1 cathode
1N4148 anti-parallel across the input LED
H11L1 output: 1–4.7k pull-up to 3V3 ──> UART_RX (3V3 domain)
```
Galvanic isolation is mandatory (ground loops between the keyboard's USB supply
and the debugger).

### 3.3 Audio output
SAI is master, TX, 48 kHz / 24-in-32, DMA double buffering on half/complete.
PCM5102A: SCK→GND (internal PLL mode), XSMT unmuted from a GPIO after the
clocks settle, the module's straps left at their defaults (I2S). Line-level
output; headphones go through an external amplifier (a BOM option).

---

## 4. MIDI and control

### 4.1 KeyStep mk1 map
| Source | Message | Assignment |
|---|---|---|
| Keys | NoteOn/Off, vel 1–127 | voice trigger; velocity → excitation energy + envelope peak |
| Aftertouch | Channel Pressure | timbre axis A (brightness / latent dim), smoothing ~20 ms |
| Pitch-strip | Pitch Bend | continuous f0 shift, ±2 semitones (config.) — glissando is where DDSP is strong |
| Mod-strip | CC1 | timbre axis B (noise share / second latent) |
| Pedal | CC64 | sustain (holds the gate) |
| Sequencer/arp | ordinary notes | a free modulation source, needs no separate handling |

Parser: a full state machine with running status and tolerance for Active Sensing;
the ISR puts bytes into a lock-free SPSC FIFO (reuse the ZeroCopySdrBuffer pattern),
drained in the control tick.

### 4.2 Modes
- **POLY**: N voices (N comes from M0), allocation: free → oldest in release → steal the oldest.
- **MONO/LEGATO**: last-note priority, glide (time from CC config). Portamento over a continuous f0 is mandatory to implement, it is the DDSP showcase.

### 4.3 Control frames
250 Hz, per voice: `{f0 (Hz, float), amp (0..1), timbreA, timbreB, gate}`.
Interpolation up to the graph's frame grid happens inside the pipeline (linear).

---

## 5. Synthesis

### 5.1 DDSP skeleton (M55/Helium)
A harmonic bank (frequencies are multiples of f0, amplitudes from the decoder/preset)
+ a noise path with a per-frame spectral envelope. Rendering goes straight into the
PQMF domain (4 subbands × 12 kHz) — this saves a band analysis ahead of the refiner.

### 5.2 NPU refiner
- A causal dilated TCN, k=3, dilation cycles; input is the subband skeleton + the control conditioning; output is the subband residual.
- **Streaming through explicit states**: the FIFO activations of every layer are input/output tensors of the graph, resident in AXISRAM; each call computes only the new columns. Recompute factor ≈ 1.0.
- **Batch = voices** (shared weights), plus a batch over time inside the hop to amortize the call overhead.
- Quantization: **int8 via QAT** (PTQ is banned as a method for the waveform domain). The output layer gets raised precision, or the final mixing is moved to fp32 on the M55.
- **Splitting the tape model**: wow/flutter is an explicit modulated fractional delay line on the M55 (a stochastic modulator driven by the measured wow and flutter spectrum; the varispeed/tape-stop macros hook in here too); the NPU learns only saturation/hysteresis/noise/path memory. The training pairs are aligned by removing the measured delay trajectory.
- The shape (C is channels, L is layers, V is voices) is fixed by M0. Ballpark: C=128, L=24 → 1.18 MMAC per subband sample ≈ 113 GOPS per voice, ~1.2 MB of weights; C=96 → 64 GOPS per voice.

### 5.3 Budgets (ballpark, before M0)
| Resource | Estimate |
|---|---|
| NPU | 2–3 voices @C=128 or 4–5 @C=96 → 180–360 GOPS sustained |
| AXISRAM | weights 1.2–2 MB + states ~0.5 MB + activations/audio <1 MB ≤ 4.2 MB |
| M55 | a bank of ≤100 harmonics × V + PQMF synthesis + reverb (partitioned FFT) — with headroom |
| Flash (NOR 128 MB) | FSBL · app · model bank A/B (safe weight updates) |

### 5.4 Latency (note → sound)
| Stage | ms |
|---|---|
| DIN MIDI (3 bytes @31250) | ~1.0 |
| Quantization to the hop (4 ms), average/worst | 2.0 / 4.0 |
| Pipeline slot (skeleton ∥ NPU) | 4.0 |
| PQMF group delay | ~1.3 |
| DAC | ~0.4 |
| **Total** | **~8.7 median / ~10.7 worst** |
Tuning option D-3: hop 4→2 ms, if M0 shows an acceptable call overhead.

---

## 6. Software structure

- Bare metal, no RTOS in v1: the audio ISR (SAI DMA half/complete) → a pipeline tick; the UART ISR → the FIFO; the super-loop does telemetry. Rationale: there is one hard period (the hop), and an RTOS adds nothing but jitter.
- A pipeline 1 hop deep: the M55 renders the skeleton of block i, the NPU processes block i−1 in parallel.
- Watchdog; underrun/overrun counters as first-class telemetry (UART/semihosting).
- Boot: development — dev mode through the debugger into RAM; release — FSBL from octal NOR + signed images (SigningTool as a post-build step), NOR programmed through the CubeProgrammer external loader.
- Model weights live in NOR at fixed offsets, with a header carrying version/CRC; A/B banks.

---

## 7. Training (the offline part)

- PyTorch; teacher (a full-size DDSP/vocoder) → student (shape from M0) by distillation.
- Losses: multi-resolution STFT (the main one), optionally an adversarial fine-tune at the end.
- QAT: Brevitas or equivalent → ONNX → `stedgeai` → Neural-ART.
- Teacher: a **degradation chain**. The priority is a real tape transport: self-recorded, perfectly aligned "skeleton → cassette/reel" pairs. Fallback is a plugin chain of tape emulation. The corpus of input material is generated by a script from our own skeleton — hours of data for free. Training is noticeably lighter than the original plan — within reach of a GTX 1060.
- Path characterization protocol: (a) silence → a noise/hiss model; (b) an impulse comb/chirps → the delay trajectory and the wow and flutter spectrum; (c) stepped sines across levels → saturation curves; (d) a corpus of the skeleton at 2–3 record levels → drive conditioning for the network (performable saturation depth).
- Hardware risk: a GTX 1060 6 GB can pull the student and STFT distillation with small batches, but slowly; the adversarial stage and a large teacher are candidates for rented GPU hours.

---

## 8. Milestones and acceptance criteria

| Milestone | Content | Pass criterion |
|---|---|---|
| **M0 — shmoo** | Random weights, a C×L×hop×V matrix on a live board through stedgeai; check state I/O and conv1d mapping, measure GOPS/ms/SRAM/call overhead | A table of measurements; the graph shape chosen; the state scheme confirmed. **Blocker for the whole project** |
| **M1 — skeleton** | The DDSP bank on the M55, the SAI path, the MIDI parser, POLY/MONO, no NPU | Playable with the KeyStep; latency ≤6 ms; 1 h without underrun |
| **M2 — offline refiner** | The trained student runs on the board non-real-time, A/B against the skeleton | Audible improvement (transients/texture) on the test phrases; int8 degradation vs fp32 ≤ the threshold on the STFT metric |
| **M3 — realtime** | Streaming with state passing, the pipeline, the full path | Latency per the table in §5.4; 24 h soak with no underruns and no SRAM leaks |
| **M4 — saturation** | A batch of voices up to the calculated V, load measurement | Steady-state NPU load ≥ the target from §1 while latency is met |
| **M5 — instrument** | Presets, glide/tuning config, model bank A/B, polish | A 30-minute live session with no engineer intervention |

---

## 9. System acceptance criteria

Latency: median ≤9 ms, worst ≤12 ms (note→sound). Note start jitter ≤1 hop.
NPU load per §1, measured, not calculated. No audible artefacts at block
seams (test: long notes + an active pitch-strip). 24 h of continuous operation.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| stedgeai will not support state tensors as I/O | Find out in M0 before training; fallback is a hybrid: short states, partial recompute with a reduced RF |
| QAT int8 sound degradation | fp16/fp32 output layer on the M55; QAT from the very start of training |
| NPU call overhead at a 4 ms hop | Batch over time; if that fails, an 8 ms hop compensated by skeleton lookahead |
| A 6 GB GPU cannot pull the teacher | Public teacher checkpoints; rent a GPU for the final run |
| Logistics (September) | Order the §11 BOM immediately; everything is cheap, only the lead time matters |

---

## 11. BOM — to order

| Item | Qty | Note |
|---|---|---|
| PCM5102A I2S DAC module | 2 | the second one is a spare |
| H11L1 (subst.: 6N138) | 5 | optoisolator for the MIDI input |
| DIN-5 180° socket | 3 | panel/board mount |
| MIDI cable DIN–DIN 1.5 m | 2 | |
| Nucleo-144 proto-shield or breadboard | 1–2 | mounting for the DIN + DAC |
| *(option)* headphone amplifier (module) | 1 | line→headphones |
| *(option)* PDM microphone MP34DT06 breakout | 2 | groundwork for the audio-driven branch in v2 |

Passives (220R, 470R, 1–4.7k, 1N4148) come from our own stock.

---

## 12. Open decisions

- **D-1 (decided 15 Jul)**: acoustics dropped; the target is Noto/Sakamoto synthetics, the teacher is a degradation chain (§7). What remains is **D-1a**: real tape vs a plugin teacher (depends on having a deck/reel-to-reel).
- **D-2**: priority — voice count vs channel width C (settled by the M0 data).
- **D-3**: hop 4 ms → 2 ms (settled by the call overhead from M0).
- **D-4**: the project name.
