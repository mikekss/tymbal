# Tymbal — a neural synthesizer on a single chip (working index N6)

> **Tymbal** is named after the mechanism. The character of this instrument is
> set by its teacher: change the teacher and the character changes completely
> while the construction stays put. Timbre here is a replaceable parameter, and
> the name points at the slot rather than at the sound. `N6` is the working
> index, taken from the chip; it survives in paths, in the branch name and in
> file names.
> A `tymbal` is the sound organ of a cicada: a ribbed membrane that clicks as it
> buckles. The word shares a root with "timbre", by way of `timbal` — a kettle
> drum — and the Greek `tympanon`, a drum.

A polyphonic hybrid real-time synthesizer on a NUCLEO-N657X0-Q (STM32N657):
a deterministic DDSP skeleton (harmonics plus noise, rendered in a 4×12 kHz
PQMF domain) on the Cortex-M55 with Helium, and a neural refiner (an FIR bank
plus a streaming int8 TCN) on the Neural-ART accelerator. MIDI over DIN, I2S
output. The hard real-time budget is 4 ms per hop — 3.2 M cycles — with
`underrun = 0`.

The full write-up is in `docs/article_draft.md`.

## Map of the repository

| path | what it is |
|---|---|
| `docs/` | requirements, design guide, chip and optimization journals, the write-up |
| `dsp/` | reference models in Python (PQMF, skeleton A/B, wow and flutter) with self-checks |
| `train/` | refiner training, ONNX export (gather2), quantization, evaluation |
| `fw/` | firmware: DSP primitives, the pipeline, the NPU driver (LL_ATON), tests |
| `tools/` | asset/weight/coefficient export, a local `atonn`, cross-checks |
| `models/` | graphs and generation artefacts (not tracked in git) |

## Reproducing without the board

```
cd fw
make test        # 11 host tests against the Python golden references
make ck4         # host reference for the CK4 score (worst-case DSP)
make qemu-ck4    # the same score through VECTOR branches on an M55 model (QEMU)
make qemu        # instructions per hop, broken down by hot-path section
```

You need `cc`, `arm-none-eabi-gcc` and `qemu-system-arm` (machine `mps3-an547`).
NPU graph variants are sifted locally with `tools/atonn/run_variant.sh`; the
`atonn` binary comes from an ST Edge AI installation. The frontend invocation
(`stedgeai generate`) and flashing the board are the Windows side.

## State (12 Aug 2026)

The instrument plays. A MIDI keyboard over DIN, sound through a PCM5102A, the
network running in every hop, 8.7 ms median latency end to end.

Milestones 0/P/T0/T1/M0/H0/H1/M2 are closed. The full chain — skeleton plus
MVE FIR bank plus int8 network — lives inside the budget: **2.11 M** cycles per
hop in silence, **2.47 M** on the worst chord, 2.81 M on chord-change peaks, out
of 3.2 M, with `underrun = 0`. Usefulness is measured by the canonical
eval_chain on hold-out: the FIR alone gives +11.25 dB of residual suppression,
the int8 chain +16.89, so the **network contributes +5.64 dB** over the linear
baseline. Canonical CK4: CRC 86dbdfc5.

Closed after first sound: a −22 dB master level ahead of the limiter (D-22);
a 6 dB per PQMF band tilt on the skeleton noise, which dropped high-frequency
noise by 7 dB and improved the metric rather than costing it (D-23); and a live
A/B between "network in the chain" and "skeleton plus FIR only" on a button on
the board (D-24). Two findings about the runtime and the toolchain have been
reported to ST.

## A note on language

This repository is translated. The work was done and journalled in Russian, and
the English here is a translation of that original — including the code
comments. Numbers, identifiers and file names are untouched. If a phrasing looks
odd, the measurement behind it is still the measurement.

## Licence

Source code, build scripts and tests are under Apache-2.0; the text is in
`LICENSE`. The documentation under `docs/` — the article, the journals and the
images — is under CC BY 4.0.

Two sets of files are covered by neither. The files in `fw/target/` come from
an STM32Cube example and stay under ST's SLA0048 (`fw/target/LICENSE.md`), and
`fw/src/mx25um51245g_conf.h` follows the macro names of ST's BSD-3-Clause
template. `NOTICE` has the details.
