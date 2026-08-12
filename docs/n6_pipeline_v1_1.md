# N6 Neural Instrument — project pipeline v1.1

**Applies to spec:** v0.1 (2026-07-15) · **Platform:** NUCLEO-N657X0-Q ·
**Purpose of this document:** a standalone step-by-step guide for the whole
project — from an empty repository to M5. It is meant to be worked from on its
own: every non-obvious decision, formula and pitfall is written down here.

**Included (already verified by running it):**
- `pqmf_design.py` — designs the PQMF bank; emits `pqmf_coeffs.h` and golden
  vectors. Result: reconstruction SNR **67.5 dB**, out-of-band suppression
  **−97 dB**, synthesis delay **1.32 ms** (agrees with §5.4 of the spec).
- `streaming_tcn_check.py` — a numerical **proof** of the TCN streaming scheme
  through explicit states (chunked-vs-full error = 0.0) plus a budget
  calculator, cross-checked against spec §5.2–5.3.
- `pqmf_coeffs.h` — ready coefficients for the firmware.

**Changes in v1.1** (from external review): F-1 — wow/flutter in the hop slot
strictly AFTER PQMF synthesis (§8.1); F-2 — wow and flutter ring 0.5–1 s plus
a tape-stop policy (§2.3, §11.1); F-3 — τ(t) of each take is taken from that
take by cross-correlation with the skeleton, stereo pilot as an option (§4.3);
F-4 — hiss as an explicit generator on the M55, the network does not learn
stochastic noise (§2.3, §4.4); F-5 — the D-5 band wiring is NOT fixed, the
test is moved into T1 (§3.1, §4.5, §7). New: §3.5 — the R-1 fallback ladder;
D-7 (reverb: dry in v1); seam phase coherence for variant B (§2.2); the
memcpy-fallback numbers are fixed (§8.2); die temperature in H4 (§9). Pencil
edits after acceptance: R-1 — the silence spectrum of §4.2(a) feeds the hiss
generator, not a "network check"; R-2 — a footnote at §3.1 for outcome (b);
R-3 — zeroing of states on NoteOn strictly after DONE (§8.2).

---

## 0. Strategy: what to do before the hardware arrives

The main idea of the whole plan: **the project blocker — M0 — is two thirds
solvable without a board.**

M0 consists of two questions:
1. *"Does the compiler swallow our graph with state tensors?"* — this is a
   pure host question: `stedgeai analyze/generate` runs on a PC and produces
   the epoch mapping, the memory layout and estimates. Risk R-1 from spec §10
   is closed (or surfaces) **today**, without hardware.
2. *"How many ms/GOPS do we really get?"* — a board is needed. Partly covered
   by the remote ST Edge AI Developer Cloud farm (see §3.4), fully — in H0.

Hence the order of phases:

| Phase | Hardware | Content | Spec reference |
|---|---|---|---|
| 0 | no | tools, repository, versions | — |
| P | no | Python prototype of the whole audio path (golden reference) | §5.1, D-6 |
| T0 | no | NPU stack validation: graph → ONNX → stedgeai; QAT skeleton | M0(a), R-1 |
| T1 | no | teacher: corpus, tape characterization, v0 training | §7 |
| H0 | **yes** | bring-up + shmoo on a live board | **M0(b)** |
| H1 | yes | skeleton in real time | M1 |
| H2 | yes | offline refiner on the board | M2 |
| H3 | yes | realtime streaming, pipeline, soak | M3 |
| H4 | yes | NPU saturation | M4 |
| H5 | yes | instrument | M5 |

**Iron rule of the pre-hardware period:** do not write a single line of
firmware that depends on the M0 unknowns (final graph shape C×L, hop, number
of voices). Everything that does not depend on them — PQMF, MIDI parser, FIFO,
voice manager, skeleton, wow/flutter — is written and tested on the host
against the golden vectors from phase P. The firmware code is then assembled
from bricks that are already verified.

And an organizational one: **order the BOM from spec §11 immediately**, before
anything else. Logistics is the only resource you cannot make up for.

---

## 1. Phase 0 — tools, repository, versions (1–2 days)

### 1.1 STM32 toolchain
- **STM32CubeIDE** of a recent version (with N6 support) — or the combination
  VSCode + CMake + `arm-none-eabi-gcc` (Arm GNU Toolchain), if that is more
  familiar. CubeMX is needed either way — the N6 clock tree cannot be built by
  hand.
- **STM32CubeProgrammer** — flashing, external loader for the NOR
  (MX66UW1G45G) on the Nucleo, ships with the **STM32 Signing Tool** (image
  signing for boot-from-flash; the N6 has no internal flash, a release image
  must be signed — dev mode through the debugger into RAM needs no signature).
- **The STM32CubeN6 package** (HAL/LL, examples) plus ST's AI examples for the
  N6 from the STMicroelectronics GitHub (packages such as
  *n6-ai-getting-started* / application templates). The examples are the
  source of truth for the clock tree, MPU regions and LL_ATON initialization:
  **do not invent your own initialization, port theirs**.
- **ST Edge AI Core** (CLI `stedgeai`) version 2.x with target `stm32n6`.
  Install check: `stedgeai --version`, `stedgeai generate --help` — make sure
  stm32n6 is in the target list and that the Neural-ART profile option (JSON
  with memory pools) is there.

### 1.2 Python environment (training + DSP prototypes)
```bash
python3 -m venv ~/venv-n6 && source ~/venv-n6/bin/activate
pip install numpy scipy soundfile matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu118   # see below
pip install brevitas onnx onnxruntime auraloss
```
**GTX 1060 pitfalls (Pascal, sm_61):** the newest PyTorch builds dropped
Pascal from their binaries. The zone that is known to work is **torch 2.1–2.4
+ cu118**. Right after installing:
`python -c "import torch; print(torch.cuda.get_device_name(0)); torch.zeros(1).cuda()"`.
If it complains about capability — roll back the torch version, not CUDA.
fp16/AMP makes no sense on Pascal (no tensor cores, half is slow) — we train
in fp32.

### 1.3 Repository
```
n6-instrument/            (branch: n6-instrument, as decided earlier)
├── docs/                 spec, this pipeline, journals and reports
├── dsp/                  python golden: pqmf_design.py, skeleton.py, wowflutter.py, golden/*.npz
├── train/                dataset generator, StreamingTCN, QAT, export
├── tools/                tape characterization: record_session.md, analyze_*.py
├── models/               *.onnx + stedgeai reports (commit them! these are experiment results)
├── fw/                   firmware (appears in H0–H1)
└── VERSIONS.md           exact versions of EVERYTHING (stedgeai, CubeN6, torch, brevitas, gcc)
```
`VERSIONS.md` is not bureaucracy: the work runs offline for weeks, and "it
built yesterday" has to be reproducible. the decision log is the log of
D-decisions (see §13), filled in as we go.

---

## 2. Phase P — Python audio prototype (golden reference), ~1–2 weeks

Goal: the whole audio path (except the neural network) runs in Python at
48 kHz, sounds, and produces golden vectors — the contract for the future C
code.

### 2.1 PQMF — done
Run `pqmf_design.py`, commit `pqmf_coeffs.h` and `pqmf_golden.npz`. What came
out and why it is enough:
- K=4, prototype N=128 (32-tap polyphase — as in spec §5.1), Kaiser β=8.8,
  fc≈0.0666 — found by a 2-D search over empirical SNR.
- **SNR of the full analysis→synthesis path: 67.5 dB.** That is plenty for
  this project: the modelled medium (the "neuro-tape") itself has an SNR of
  about 50–60 dB, and its noise is artistic material. The bank error sits
  ~10–17 dB below the tape noise.
- In the **real instrument path there is no analysis** — the skeleton and the
  residual are born in the subbands, only synthesis is applied: delay 1.32 ms
  (as in §5.4), and the only coloration is the ripple of the prototype's
  composite magnitude response (fractions of a dB).
- The full analysis+synthesis path is used offline: preparing training pairs
  (§4.4).
- Upgrade path, if >67 dB is ever wanted: N=192/256 (recompute the delay in
  §5.4!) or direct optimization of the prototype coefficients for flatness of
  the composite response (`scipy.optimize`, starting from the current
  solution).

### 2.2 Skeleton: where to render it — variant A vs B (decision D-6)
Spec §5.1 prescribes rendering directly in the subbands. Prototype **both**
variants and decide by metric — here is the fork, stated honestly:

**Variant A — render at 48 kHz → PQMF analysis.** Simple, no edge effects. The
cost of analysis on the M55 is pennies: polyphase analysis with N=128, K=4 is
32 MAC for each of the 4 subband samples per band input cycle, so ~128 MAC per
48 kHz input sample ≈ 6.1 MMAC/s for the whole thing — background noise on an
800 MHz M55. Downside: in the real path an analysis group delay appears
(~1.3 ms), and the median latency becomes ~10 ms — this **violates** the §9
criterion (median ≤9 ms) at hop 4 ms. So D-6 is coupled to D-3: variant A is
legal in real time only together with hop 2 ms (then the median is ~7–8 ms),
variant B — at any hop. For offline use (data preparation) A is mandatory
either way.

**Variant B — direct rendering in the subbands (as in the spec).** A harmonic
with absolute frequency f lands in band k=⌊f/6000⌋ and is rendered by an
oscillator at 12 kHz with frequency:
- even band k: f_sub = f − k·6000;
- odd band k: f_sub = (k+1)·6000 − f — **spectral inversion of the odd bands**
  of a cosine-modulated bank. Do not forget: the inversion flips the sign of
  the phase increment (a glissando up in an odd band is f_sub going down).

The delicate part of B: a harmonic closer than ~300–500 Hz to a band edge is
smeared by the prototype's transition band across **two** bands. Options:
(1) crossfade rendering into both bands with weights taken from the prototype
response at that frequency (a weight table indexed by f mod 6000), (2) accept
the artifact and listen. Important for (1): an amplitude crossfade is not
enough — both band components must inherit the phase of **one master
accumulator** (accounting for the odd-band inversion), otherwise two free
oscillators at slightly different effective frequencies will beat at the seam.
With active pitch bend across a band edge, the variant without a crossfade
gives an audible seam — the test is mandatory.

**Recommendation:** start with A (it is needed for preparing the teacher data
anyway), implement B in Python, compare by the error spectrum on glissandi
across band edges and by the latency budget. The decision goes into
the decision log as D-6. The NPU refiner does not care: it lives in the
subbands under either variant.

**The noise path** in the subband domain is trivial: per band, a white noise
generator × a per-frame envelope (frames = hop, linear interpolation of the
envelope between frames, otherwise zipper noise). Within-band shaping is
barely needed with 4 bands; if it is wanted — a short FIR of 4–8 taps per
band.

**Control:** the voice manager emits 250 Hz frames `{f0, amp, tA, tB, gate}`
(spec §4.3); in Python build exactly the same structure and linear
interpolation onto the hop grid — it will port to C one to one.

### 2.3 Wow/flutter and hiss — explicit models on the M55 (in Python first)
From spec §5.2: wow and flutter is NOT the network's job. The model:
- **A fractional delay line** on the full band, AFTER PQMF synthesis
  (physically honest: the tape modulates the common time axis). Ring buffer
  **0.5–1 s** (mono f32: 96–192 KB — enter it into the memory map §11.1): the
  size is dictated not by vibration but by tape-stop — the accumulated read
  lag = ∫(1−v)dt; a ramp to zero over ~0.4 s gives ~200 ms of lag, and a full
  stop grows it without bound. Interpolation: linear for v0; if the HF
  modulation is audible — 3rd-order Lagrange (4 samples) or a Thiran allpass.
- **A stochastic modulator** τ(t): a sum of 2–4 sinusoids at the peak
  frequencies of the measured wow and flutter spectrum (capstan/pinch roller
  rotation — they will come out of §4.2) + noise filtered to match the
  measured spectral floor (usually ~1/f²), + a slow drift. Depth comes from
  the measurements; the typical cassette scale is 0.1–0.3% of speed.
- The **varispeed / tape-stop macros** cut in here too: they are just a
  controlled speed trajectory on top of the stochastic one. They cost nothing
  to implement, and aesthetically (the fast pitch swings from spec §1) they
  are first-class gestures. Stop policy: on reaching v=0 — fade to silence
  over tens of ms and reset the read pointer to the nominal lag before
  restart (otherwise the next gesture starts with an exhausted ring). For
  varispeed without a stop the lag budget = depth × gesture duration — also
  cross-check it against the ring size.
- Consistency with training: in training, wow and flutter is *removed* from
  the recordings before they are fed to the network (§4.3); at inference it is
  *added* after the network. The symmetry is mandatory, otherwise the network
  learns the wow and flutter residue and it adds on top of the explicit model.

**Hiss is an explicit generator too, not the network (a refinement of the
split in spec §5.2).** A deterministic feedforward network under MR-STFT+L1
does not reproduce stochastic noise: L1 pulls the prediction toward the
conditional median — that is, toward the signal *without* noise — and a
spectral loss with no source of randomness produces a "buzzing" deterministic
texture (the classic failure of vocoder training). Therefore:
- a generator on the M55, fully symmetric with wow and flutter: white noise
  (xorshift) → a shaping filter matched to the measured spectrum of §4.2(a) —
  4–6 biquads fitted offline against the Welch PSD, cheaper than a long FIR;
  level from the silence measurement, dependence on drive to be checked
  against the three recording levels (d) — on most paths the hiss is nearly
  constant;
- the resulting split of the path: **explicit wow and flutter + explicit hiss
  (M55) + neural saturation/dynamics/memory (NPU)**;
- a noise-seed input to the network + an adversarial loss — an experiment for
  later, not the plan;
- the consequence for training — see §4.4.

### 2.4 Golden vectors — the Python↔C contract
For each block: `dsp/golden/<block>_io.npz` with the exact input and output
(float32): PQMF synthesis, one-voice skeleton (fixed control score),
wow/flutter (fixed τ trajectory), final mix. The future C unit tests in H1 run
these same vectors (the input is loaded by the debugger or over UART, the
output is dumped and compared on the PC; the tolerance for an f32
implementation is bit-close, ~1e-6 RMS). This is the only way to debug DSP on
the board without tears.

---
## 3. Phase T0 — NPU stack validation without a board (the most important pre-hardware phase)

Goal: by the time the hardware arrives, have a "template graph" — an
architecture that `stedgeai` is guaranteed to compile entirely onto the NPU,
with state tensors in SRAM and predictable memory. This removes R-1 (the main
risk in the spec) before training starts.

### 3.1 Streaming TCN with explicit states
The state scheme is mathematically proven (`streaming_tcn_check.py`, error
0.0): the state of a layer = the last (k−1)·d columns of the layer's
**input**; with zero initialization, streaming is identical to the full
sequence. Carry rule: `state' = tail of concat(state, chunk)`.

Reference implementation for training and export (**run the chunked==full test
yourself, following the numpy script, before any training**):

```python
import torch, torch.nn as nn

class StreamConv(nn.Module):
    def __init__(self, c, k=3, d=1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, dilation=d)      # valid, no padding
        self.pad = (k - 1) * d
    def forward(self, x, state):                        # x:[B,C,T] state:[B,C,pad]
        xin = torch.cat([state, x], dim=2)
        return self.conv(xin), xin[:, :, -self.pad:]

class StreamingTCN(nn.Module):
    def __init__(self, c_in, c=128, c_out=1, layers=24, cycle=(1,2,4,8,16,32), k=3):
        super().__init__()
        self.head = nn.Conv1d(c_in, c, 1)
        self.blocks = nn.ModuleList(
            [StreamConv(c, k, d) for d in (list(cycle) * (layers // len(cycle)))])
        self.tail = nn.Conv1d(c, c_out, 1)
    def forward(self, x, *states):
        h = self.head(x)
        outs = []
        for blk, s in zip(self.blocks, states):
            z, s2 = blk(h, s)
            h = h + torch.relu(z)
            outs.append(s2)
        return (self.tail(h), *outs)
```

**Subband wiring — decision D-5 (new, surfaced during the budget
cross-check).** The spec arithmetic (113 GOPS/voice at C=128) only adds up if
**batch = voices × subbands**: each band is an independent 12 kHz sequence, the
weights are shared, band-id goes into the conditioning. The weak point: a
band-local model physically cannot produce **cross-band** saturation products —
and those are not exotic: products of 2–6 kHz material fall into 6–12 kHz,
exactly the HF-glitch territory of this aesthetic. So we do NOT fix the
baseline: in T1 **both** wirings are trained at small size and discriminated by
a target signal (§4.5); the decision comes before it grows into the realtime
code, by the start of H3. Candidates: (a) batch wiring (as above);
(b) bands-as-channels at **C=192** — weights 2.65 MB, 64 GOPS/voice → 191 GOPS
for 3 voices, the saturation class is preserved, states shrink to ~0.6 MB with
ping-pong (AXISRAM fits, but only just — verify it with a T0 report the same
day); (c) a hybrid — a band-local stack with 1×1 cross-band mixing every ~6
layers. In T0 it is cheap to export **both** extreme forms and push them
through stedgeai. And remember: with batch wiring the **states are ×4 bands** —
the "~0.5 MB" line from spec §5.3 is really 1–2.5 MB (it fits in 4.2 MB, but
plan the memory layout from the calculator table, not from the spec).

*Footnote on outcome (b):* if bands-as-channels wins in §4.5, fix three places
written for batch wiring at the same time: the conditioning here (c_in = 8: the
four skeleton bands as channels + amp/tA/tB + drive; band-id disappears), the
export shapes in §3.2 (B = V, T the same 48), and the state slice on NoteOn in
§8.2 (the slice = one sequence per voice × L layers).

**Conditioning** (broadcast input channels, interpolation 250 Hz → 12 kHz):
band skeleton (1) + amp/timbreA/timbreB (3) + band-id one-hot (4) + drive (1)
→ **c_in = 9**. FiLM can come later; it is not needed for v1.

### 3.2 ONNX export and compilation
```python
net.eval()
B, T = 12, 48                      # batch is FIXED (voices×bands), hop 4 ms @12 kHz
x = torch.zeros(B, 9, T)
states = [torch.zeros(B, 128, blk.pad) for blk in net.blocks]
torch.onnx.export(net, (x, *states), "student_c128.onnx", opset_version=17,
    input_names=["x"] + [f"state_in_{i}"  for i in range(len(states))],
    output_names=["y"] + [f"state_out_{i}" for i in range(len(states))])
```
Rules: opset 17; **no** dynamic axes, no Loop/If, no GRU/LSTM ops; the batch is
fixed forever (changing the number of voices = a different graph — that is why
V is chosen in M0 and frozen). Right after export — an equivalence test
through `onnxruntime`: feed the full sequence in chunks with state carry,
compare against torch.

Compilation (check the exact syntax against `--help` of your version):
```bash
stedgeai analyze  --model student_c128.onnx --target stm32n6 \
                  --st-neural-art default@fw/neural_art.json
stedgeai generate --model student_c128.onnx --target stm32n6 \
                  --st-neural-art default@fw/neural_art.json --output models/out_c128/
```
`neural_art.json` is the profile with the memory pools; take the template from
ST's AI examples and write in your own pools (weights/activations/I-O in
AXISRAM). Look at four things in the reports:
1. **Epochs**: did the whole data path land on NPU epochs? An SW epoch
   (fallback to the M55) in the middle of the graph on every hop is a budget
   killer; it is tolerable only on the output layer.
2. **Where the I/O landed**: state tensors must live in SRAM pools, not in the
   "virtual" interface memory.
3. **Memory**: the report numbers vs the `streaming_tcn_check.py` table (a
   discrepancy >20% means investigate: padding/alignment/copies).
4. **Number of epochs**: every epoch is launch overhead; fewer = better for a
   4 ms hop.

The T0 iteration loop: simplify or change the graph until all four points are
green for C=96 and C=128. Typical substitutions when there are problems: move
the residual-add out of suspicious places, use ReLU instead of anything
exotic, make sure Conv1d does not unfold into something strange (sometimes it
helps to express Conv1d as a Conv2d of height 1 — if the compiler is
friendlier that way; it is quick to check).

**If the report is bad, do not rework the network blindly:** the R-1 fallback
ladder is in §3.5. Keep the spare graph branches ready, but do not spend time
on them until the main path is stuck.

### 3.3 QAT (int8) — from the very start, PTQ is forbidden (spec §5.2)
- Brevitas: `QuantConv1d` (int8 per-channel weights) + quantized activations; a
  short fp32 pretrain (tens of thousands of steps) → switch QAT on and train
  to the end.
- **States are quantized tensors too.** Critical: the quantization scale of
  `state_out_i` must be **equal** to the scale of `state_in_i` of the same
  layer — it is the same wire in time. In Brevitas — a shared quantizer at the
  layer input and at the state slice; if the scales drift apart, recurrent
  feeding accumulates drift and clicks appear at the hop joints.
- Output layer: leave it in fp — the "skeleton + residual" mix is done by the
  M55 anyway (the sum in the subbands before synthesis), so the tail conv can
  physically run on the M55 in f32, with the NPU graph emitting the
  activations of the second-to-last layer. Or an int8 output, if the M2 metric
  does not degrade. Build both variants into the export.
- Export the quantized graph as QDQ ONNX (`brevitas.export`; for the current
  function name see the docs of the installed version — their API is a moving
  target); check that `stedgeai` swallows QDQ and maps it into int8 epochs
  instead of emulating it in fp.

### 3.4 Early shmoo without a board — ST Edge AI Developer Cloud
ST has a cloud service with a farm of real boards: you upload ONNX and get
measured per-epoch times back. If the farm list has an N6 (usually as an
STM32N6570-DK — the same die), run the matrix {C=96,128} × {hop batch 1,2,4}
**before the hardware arrives** — that is almost all of M0(b). If there is no
farm or no N6 — no disaster, H0 is still there; but it is worth checking on
the very first day of T0.

### 3.5 R-1: the fallback ladder (if stedgeai breaks state streaming)
First a diagnosis from the T0 report — these are three different diseases:
1. **The graph built, few epochs, states in SRAM pools** → there is no
   problem, move on.
2. **The graph built but is fragmented**: SW epochs around concat, dozens of
   small epochs, hidden state copies. That is overhead, not a death sentence —
   first **measure** t_call (DevCloud/board), then treat it, by cost:
   - fewer state tensors at the same RF: fewer layers with a wider kernel —
     k=5, L=12, Σd=126 (cycle 1…32 ×2) gives the same RF≈42 ms and **the same
     state bytes** (C·(k−1)·Σd is invariant), but half as many I/O tensors and
     even −17% MAC (12·5 against 24·3 per C²); the fixed cost is paid per
     tensor — this step is free;
   - a bigger NPU slot S (time batching): the fixed call cost ∝ 1/S, but
     latency grows linearly with S — S=8 ms gives a median of ~14.7 ms, and
     the §9 criterion would have to be officially revised. Important: a small
     audio hop does NOT save this — NPU economics depend only on S, not on the
     hop;
   - an experimental "split latency" branch: the skeleton is mixed on time
     (hop 2–4 ms), the residual is blended in S later. This is not an echo
     (the residual is a different signal, not a copy), but the attack "blooms
     late" — judge it by listening test only.
3. **States-as-I/O export is structurally rejected** → the spec §10 fallback: a
   short RF (~8–12 ms, smaller Σd) without cross-call states in the deep
   layers — an overlapping window W, with the last hop taken at the output;
   compute factor W/hop = 2–3×, paired with shrinking C. What suffers is
   exactly the long memory of the path. The last step is a hybrid: move the
   long memory into a tiny recurrent f32 block on the M55 and leave the
   windowed convolutional part on the NPU.

For a live instrument, "lookahead in the DSP" compensates for nothing — the
performer's future is unknown; the only compensations for a large S are
revising the criterion or splitting the paths.

---

## 4. Phase T1 — teacher, data, training (in parallel with T0)

### 4.1 Corpus generator
The `train/make_corpus.py` script: procedural scores → 250 Hz control curves →
skeleton render (variant A from §2.2) → WAV 48k/24-bit + NPZ of controls (they
are also the conditioning, saving them is mandatory!). Score content matching
the aesthetic of spec §1:
- sparse melodic phrases with pauses (Noto/Sakamoto — air matters more than
  notes), wide range;
- legato/glissando lines (portamento is the showcase, there must be a lot of
  it in the data);
- fast HF↔LF swings (arpeggios across 3–4 octaves, leaps);
- staccato/impulsive notes (transients — the hardest thing for the refiner);
- long notes with slow timbreA/B envelopes (texture, memory of the path);
- 10–20% pure "technical" signals: sweeps, single harmonics, noise bursts —
  they stabilize training at the edges.
Hours of material are free; 2–4 h is enough for v0, aim for 8–12 h by the
final training run.

### 4.2 Tape transport characterization session (checklist)
The transport is still being looked for (D-1a); when it turns up — one careful
session following the protocol of spec §7, everything recorded from the
interface at 48k/24-bit, recording levels written into the log:
1. Preparation: warm up the mechanics for 10 min; set the nominal level and
   write it into the log (0 dB on the deck meter = X dBFS at the interface);
   do not touch azimuth or heads — the dirt of the path is the character.
2. **(a) Silence**: 5 min of pause with no signal + 5 min of recording a
   "zero" input → the noise/hiss spectrum — this is the source for the shaping
   filter of the hiss generator in §2.3 and the subtraction profile for §4.4,
   not a "network check": the network does not learn noise.
3. **(b) Wow and flutter**: a 3150 Hz pilot tone (the standard for wow &
   flutter measurement), 5 min, level −10 dB; plus log chirps 20 Hz–16 kHz ×3
   and a "comb" of short impulses once per second, 2 min (a backup way to
   track the delay).
4. **(c) Saturation**: stepped sines: frequencies on a log grid, ~20 points
   50 Hz–12 kHz × levels −30…+6 dB in 3 dB steps, 2 s of tone + 0.5 s of pause
   each (~20 min in total) → a family of amplitude curves per frequency. Use:
   a sanity check for the network + parameters for an emergency DSP saturator
   if the network does not fly.
5. **(d) Corpus**: the skeleton corpus from §4.1 at 2–3 recording levels (for
   example −10 / 0 / +4 dB relative to nominal) → performable saturation
   depth: the recording level of each phrase goes into the `drive`
   conditioning.

Pilot analysis (sketch of `tools/analyze_pilot.py`):
```python
import numpy as np, scipy.signal as sig
x, fs = ...                                   # pilot recording
z   = sig.hilbert(sig.sosfilt(bp_3150, x))    # narrow BP around 3150 Hz!
phi = np.unwrap(np.angle(z))
tau = (phi - 2*np.pi*3150*np.arange(len(x))/fs) / (2*np.pi*3150)   # τ(t), s
speed_dev = np.gradient(phi, 1/fs) / (2*np.pi*3150) - 1.0          # speed deviation
# wow/flutter spectrum: Welch of speed_dev -> peaks (rotors) + floor for the §2.3 model
```

### 4.3 Alignment of the training pairs
The recordings contain wow and flutter — it has to be taken out so that the
network learns only saturation/hysteresis/noise/memory (the split from spec
§5.2):
1. τ(t) **of each take comes from that take**: wow and flutter is stochastic
   from run to run; the pilot take gives the *spectrum* (for the runtime model
   of §2.3), but not the trajectory of some other recording. The main method
   is a per-frame cross-correlation of the recording with the exactly known
   skeleton input (window ~50 ms, step ~10 ms, parabolic peak interpolation →
   subsample accuracy, the trajectory is stitched together by smoothing). An
   option for a stereo deck: a 3150 Hz pilot on the second channel
   **simultaneously** with the program, at −20 dB — the interchannel crosstalk
   will sit around −50 dB and a narrowband Hilbert will pull it out; if the
   pilot is audible in the target channel — a narrow notch. A separate pilot
   take in the session is still needed: the wow and flutter spectrum and the
   calibration of the sign of τ.
2. Inversion: build the axis `t_rec = t + τ(t)` (check the sign by
   cross-correlation — it depends on the measurement convention) and resample
   the recording onto a uniform grid with cubic interpolation.
3. Quality control: residual drift across windows < 1 sample over the whole
   file; phrases where it did not converge get thrown away, not "nudged into
   place".
4. Pairs into the dataset: PQMF analysis of the skeleton → input; PQMF
   analysis of the aligned recording → target; + controls + drive.

### 4.4 Training on a GTX 1060 6 GB
- Segments of 8192–16384 subband samples (~0.7–1.4 s), batch 4–8 + grad
  accumulation up to an effective ~16; fp32; AdamW, lr 3e-4, cosine; gradient
  clip 1.0.
- Loss: multi-resolution STFT (windows 64…2048, auraloss) on the
  **reconstructed** signal — the PQMF synthesis matrix as a fixed convolution
  in torch, differentiable — plus L1 in the subbands with weight ~0.1.
  Adversarial only as a final stage, and more likely on a rented GPU (I
  confirm the risk from spec §10: 6 GB + Pascal for GAN tuning is pain).
- Hiss in the targets: the network will not learn it (§2.3) — the L1 optimum
  of a deterministic network is exactly the "signal without noise", so the
  student naturally converges to a noiseless output and the metric hits a
  floor at the level of the target's hiss. That is normal. If clean curves are
  wanted — a light spectral subtraction from the targets using the silence
  profile of §4.2(a). In the M2 A/B listening tests, add hiss with the
  explicit generator to **both** candidates, otherwise the comparison is
  dishonest.
- Monitor: the MR-STFT curve on held-out phrases; every N hours — export audio
  examples (skeleton / target / prediction) and **listen**, the metric is deaf
  to what matters most.
- QAT stage: from the fp32 checkpoint, the same data, usually it finishes
  training quickly; the M2 acceptance metric: MR-STFT(int8) − MR-STFT(fp32) ≤
  threshold (you set the threshold after the first listening test — typically
  "no worse than 5–10% relative").
- Realistic pace on a 1060: a v0 student is a day or two per run. Plan
  experiments in batches for overnight.

### 4.5 The D-5 test — band wiring (decided here, not in M2)
Train both wirings (batch and channels; small size C=48–64, short run, one
dataset). The discriminator is a target signal: steady partials at 2–5 kHz
with drive +4 dB; the metric is the energy and structure of the products in
band 1 (6–12 kHz) against the target, plus a listening test. If band-local
honestly loses on cross-band products — switch to (b)/(c) from §3.1 **now**:
by M2 the wiring will already have grown into the realtime code and the state
layout.

---

## 5. Phase H0 (=M0) — bring-up and shmoo (the first 2–4 days with hardware)

### 5.1 Bring-up, day one
1. Power/STLINK: the board is visible to CubeProgrammer. Boot switches into
   dev mode (execution from RAM through the debugger; see the board UM, boot
   section).
2. Build and flash **an example from the CubeN6 package / AI template** (not
   your own!): blink + UART printf at 115200. That pins down a working clock
   tree (M55 at nominal, NPU clock), MPU/cache settings and the linker script
   — you port them, you do not invent them.
3. `DWT->CYCCNT` for measurements:
```c
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
```
4. Flashing the NOR through the external loader (MX66UW1G45G) — check writing
   and reading a test block now, not in M5 when the weight banks are needed.

### 5.2 Shmoo firmware
One firmware = one graph variant (automate building the variants with a
script; results go into the the M0 report table by hand). The core of a
run is the LL_ATON loop from the `generate` template (check the names against
the generated code):
```c
LL_ATON_RT_RuntimeInit();
LL_ATON_RT_Init_Network(&NN_Instance_Default);
/* set the input/output and state buffer addresses in AXISRAM */
uint32_t t0 = DWT->CYCCNT;
LL_ATON_RT_RetValues_t s;
do { s = LL_ATON_RT_RunEpochBlock(&NN_Instance_Default);
     if (s == LL_ATON_RT_WFE) __WFE(); } while (s != LL_ATON_RT_DONE);
uint32_t cycles = DWT->CYCCNT - t0;
```
Measurement matrix (following the table from `streaming_tcn_check.py`, filled
in with reality):
- C ∈ {96, 128} × hop batch in time {1×48, 2×48, 4×48} × batch of voices×bands
  {4, 8, 12, 16, 20};
- for each point: t_call in ms; achieved GOPS = MAC_total·2/t; **call
  overhead** = t(2×hop) − 2·t(hop); actual SRAM (report + map file);
- a run of ≥1000 consecutive calls: mean/max (jitter!), stability of the state
  pointers.

### 5.3 Decisions from the M0 data (recorded in the decision log)
- **D-2**: V vs C — from the measured GOPS and the latency budget (pipeline
  slot = hop: t_call(of the whole batch) ≤ hop − a reserve of ~20%).
- **D-3**: hop 4→2 ms — only if the call overhead is ≤ ~15% of t_call at 2 ms.
- Freezing the graph shape and the batch → from this point the final student
  is trained (T1 finishes training on the exact shape).
- The spec §1 goal (≥30% of the measured peak) — the "peak" is fixed here too:
  the best GOPS achieved on a large batch is the practical ceiling.

**M0 pass criterion** (spec §8): the measurement table in the M0 report, the
shape chosen, the state scheme confirmed on a live die.

---
## 6. Phase H1 (=M1) — skeleton in real time (~1 week)

Strictly bottom-up order; each step is closed by its own test before moving to
the next.

### 6.1 SAI + DMA + PCM5102A
1. SAI: master, TX, 48 kHz, 32-bit slot / 24-bit data (MSB alignment as in
   I2S), **do not output MCLK** — the PCM5102A with SCK→GND synthesizes the
   system clock from BCK itself. DMA circular, two half-buffers of one hop
   each (192 samples × 2 channels), half/complete interrupts.
2. DAC start-up sequence: SAI clocks stable → pause ~10 ms → GPIO XSMT high
   (unmute). On shutdown — mute first, then the clocks. Otherwise you get
   clicks and occasional DAC "lock-ups".
3. Test 1: constant level/silence — use a scope to confirm that frames are
   going out (BCK/LRCK visible, LRCK exactly 48 kHz).
4. Test 2: a 440 Hz sine from a table — listen for cleanliness;
   "mirrors"/squeal = wrong slot format (24-in-32, alignment, channel order).
5. Test 3: underrun counters (in half/complete check the "previous half has
   been rendered" flag) — 1 hour of a clean sine, counter = 0.
6. **Cache and DMA — pitfall no. 1 on the M profile with D-Cache:** the DMA
   reads memory directly. Either put the audio buffers in a non-cacheable MPU
   region, or call `SCB_CleanDCache_by_Addr()` on the just-rendered half before
   the DMA gets to it. The symptom of missing this: "stuttering" with old
   data, dependent on buffer size.

### 6.2 MIDI input
1. Hardware: the H11L1 optoisolator output with a pull-up to 3V3 → scope:
   pressing a key on the KeyStep shows bursts at 31250 baud. If the line is
   "upside down" — DIN pins 4/5 are swapped or the LED polarity is wrong (the
   1N4148 saves it from burning out, but there will be no data).
2. USART RX 31250 8N1, per-byte interrupt → the byte goes into an SPSC FIFO.
   No parsing in the ISR, except (optionally) a filter for realtime bytes.
3. **Porting `ZeroCopySdrBuffer`:** the SPSC pattern carries over one to one;
   what changes is the memory-model justification — instead of RVWMO we have
   ARMv8-M: C11/C++ `atomic` with acquire/release maps onto LDA/STL natively.
   On a single core with an ISR producer, volatile plus compiler barriers
   would formally be enough, but we stay on atomics — the pattern is already
   verified and portable.
4. Parser (draining the FIFO in the control tick): a state machine with
   running status; realtime bytes (0xF8–0xFF, including Active Sensing 0xFE)
   can wedge **inside** a message — ignore them without resetting the state
   machine. Test: the KeyStep arpeggiator at maximum tempo plus working the
   mod strip at the same time, 10 min — not a single stuck note.
5. Measure note latency like this: a GPIO toggle at the moment NoteOn is
   recognized → scope GPIO vs line-out (start of the attack).

### 6.3 Voice manager and skeleton
1. The 250 Hz tick — count hops in the audio callback (each hop = 4 ms → every
   tick is a hop; at hop 2 ms — every second one). Voice allocation per spec
   §4.2: free → oldest in release → steal the oldest. Legato/mono: last-note
   priority, exponential glide in log-f0 (musically smoother), time from the
   CC config.
2. Harmonic bank on Helium: a uint32 phase accumulator (natural wrap), a 4096
   sine table + linear interpolation, MVE vectorization over 4 floats;
   harmonic amplitudes from the preset/decoder, harmonics above the band
   Nyquist hard-zeroed (otherwise aliasing on an upward glissando).
3. Budget, for peace of mind: hop 4 ms @ 800 MHz = **3.2 million cycles**. A
   bank of 100 harmonics × 3 voices × 192 samples ≈ 58 thousand
   sample-harmonics × ~2–4 cycles (Helium) ≈ 0.12–0.24 million cycles. Noise +
   envelopes + PQMF synthesis (32 MAC/sample ≈ 6 thousand MAC/hop) are
   background. The skeleton will take <15% of the core, everything else is
   headroom for wow/flutter, reverb and pumping the NPU epochs.
4. Order of sound assembly in this phase: skeleton (subbands) → PQMF synthesis
   → wow/flutter (the ready port from §2.3, cross-checked against golden) →
   DMA. There is no refiner yet — and this is **a playable instrument** (the
   graceful principle of spec §2 gets tested right here).

**M1 acceptance (spec §8):** playable with the KeyStep; latency ≤6 ms (without
the NPU slot it is shorter than the full figure); 1 hour with no underrun;
glissando on the pitch strip continuous and without steps.

---

## 7. Phase H2 (=M2) — offline refiner on the board

1. Run the trained int8 graph on the board in non-real time: the input (a
   golden phrase: skeleton subbands + controls) is loaded by the debugger into
   RAM or from the NOR; the output is dumped to RAM or over UART to the PC.
2. Comparison with the reference: torch fake-quant and NPU int8 are **not
   bit-exact** — compare output-to-output SNR (expect >35–40 dB) and the
   MR-STFT metric. The int8 vs fp32 drop ≤ the threshold (spec M2). Option:
   the standard `stedgeai validate` on the target — as a cross-check of the
   stack.
3. A/B listening test on the PC: skeleton / skeleton+residual (from the
   board) / target tape — on test phrases of every type from §4.1. Spec
   criterion: an audible improvement in transients/texture.
4. A check listening test of the wiring chosen in §4.5, on live hardware:
   decision D-5 has already been taken in T1 by this point — here it is only
   confirmation that on the board it sounds the way it did in simulation.

---

## 8. Phase H3 (=M3) — realtime streaming and the pipeline

### 8.1 Hop slot schedule (pipeline of depth 1)
```
IRQ half/complete (hop boundary, every 4 ms):
 1. hand the finished block i−2 to the DMA (already mixed in the previous slot)
 2. start the NPU on block i−1 (inputs: skeleton i−1 + controls; states: ping pointers)
 3. M55: drain the MIDI FIFO, voice tick, render the skeleton of block i (into a buffer)
 4. pump the NPU to DONE (cooperatively: RunEpochBlock; instead of __WFE —
    a quantum of the remaining M55 work; or an IRQ on completion, if LL_ATON gives one)
 5. mix block i−1 in the subbands: skeleton+residual → PQMF synthesis → wow/flutter (full band — only after synthesis, §2.3!) → into the free DMA half
 6. swap the ping/pong state pointers; timing telemetry (max per second)
```
Latency under this scheme is the spec §5.4 table (median ~8.7 ms at hop 4).

### 8.2 States in real time
- Ping-pong: two sets of state buffers, swapped every hop by setting the I/O
  addresses (if LL_ATON allows rebinding without reinitialization — verified
  in T0/H0). If not — memcpy out→in: the full state set at C=128 and 12
  sequences (3 voices × 4 bands) ≈ 756 KB, at 250 Hz that is ≈ 190 MB/s —
  AXISRAM will survive it, but the M55 cycles are too precious: pointers stay
  plan A.
- NoteOn on a voice = zero its state slice (otherwise the tail of another note
  "leaks" into the attack) — but **not in step 3**: at that moment the NPU may
  still be writing state_out (step 4), and the zeroing would be overwritten.
  Timing: **after DONE**, in the window of steps 5–6, zero the voice slice in
  the **output** set that was just written — after the swap that is exactly
  what becomes the input of the new note's first block. The slice by voice
  batch index is 4 bands × L layers (with outcome (b) from §4.5 — one sequence
  × L).

### 8.3 Degradation and telemetry
- The NPU did not finish by mix time → the residual of this hop = 0 (the
  instrument sounds as the skeleton), increment the `npu_miss` counter. No
  audio stops under any NPU error whatsoever.
- Counters: underrun/overrun, npu_miss, max t_call, max M55 occupancy per
  second, FIFO highwater. Printed once per second from the superloop.
  **Printing from an ISR is forbidden.**

### 8.4 Block joints (spec §9 test: long notes + active pitch strip)
Typical causes of clicks at hop boundaries, in order of likelihood:
1. a state buffer lost or swapped (ping-pong mixed up) — a click every hop;
2. quantization scale of state_in ≠ state_out (see §3.3) — slowly growing
   "sand";
3. the seam in control interpolation (250 Hz → hop) — zipper noise with an
   active pitch strip;
4. the DMA buffer cache (see 6.1).

**M3 acceptance (spec §8):** latency per §5.4; 24 h soak: underrun=0,
npu_miss≈0, no SRAM leaks (state pointers and highwater stable).

---

## 9. Phase H4 (=M4) — NPU saturation

1. Raise the batch to the calculated V (from M0), play dense sequencer
   material from the KeyStep (16th-note arpeggios at a high tempo — the "free"
   stress generator from spec §4.1).
2. Measure the **steady-state** utilization: active NPU time per second / 1 s;
   achieved GOPS = the MAC volume of the second / active time; compare against
   the peak recorded in M0.
3. Spec §1 criterion: ≥30% of the measured peak in steady state while meeting
   the §5.4 latency. If it falls short: raise C (revisit D-2), deepen the hop
   batch in time, raise V — in decreasing order of benefit to the sound.
4. Add the die temperature (internal sensor) to the soak telemetry: otherwise
   degradation from throttling over a long session cannot be told apart from a
   code regression.
5. Report `docs/m4_report.md`: numbers, configuration, temperature profile,
   conclusion.

---

## 10. Phase H5 (=M5) — the instrument

1. **Presets**: a structure in the NOR (harmonic amplitudes / skeleton decoder
   parameters, CC map, glide time, wow/flutter depth, model bank selection),
   header {magic, version, size, CRC32}. Switching by Program Change from the
   KeyStep.
2. **A/B weight banks** (spec §6): two slots at fixed NOR offsets, each with a
   header and a CRC; at startup — check the CRC of the active one, on failure
   — automatic rollback to the second. A bank update goes only into the
   inactive slot.
3. Tuning/glide/PB-range configuration over CC (document the map in
   `docs/midi_map.md`).
4. Polish: voice priorities on stealing (a click fade of ~2 ms on the stolen
   voice), an output limiter (soft, f32 on the M55) — insurance against int8
   outliers in the residual.
5. **M5 acceptance (spec §8): a 30-minute live session with no engineer
   intervention.** Record it — that is also the first demo material.

---

## 11. Cross-cutting matters

### 11.1 Memory map plan (refined by the stedgeai reports and the map file)
| Region | Contents | Estimate |
|---|---|---|
| AXISRAM (4.2 MB) | int8 weights | 0.65–1.15 MB (C=96/128) |
| | states ×4 bands ×V (×2 ping-pong) | 0.4–2.5 MB — from the calculator table! |
| | NPU activations (pool) | from the stedgeai report |
| | audio: DMA buffers (non-cacheable section), skeleton subbands, wow line 0.5–1 s (96–192 KB, §2.3) | ~0.3–0.4 MB |
| NOR 128 MB | FSBL · app · presets · model bank A · model bank B | fixed offsets, all with CRC |

### 11.2 Cache and DMA/NPU
- Audio DMA buffers: a non-cacheable MPU region **or**
  `SCB_CleanDCache_by_Addr` before handing over a half. Pick one and write a
  comment in the code saying why.
- NPU pools: the cacheability/alignment requirements come from ST's AI example
  (the MPU is already set up there) — port it, do not invent it.

### 11.3 Debug playbook (symptom → where to look)
| Symptom | Causes in decreasing order of likelihood |
|---|---|
| Complete silence | XSMT not raised; SAI clocks not running (scope the LRCK!); DMA not started; buffer of zeros |
| Squeal / "mirrored" sound | slot format: 24-in-32, MSB alignment, byte order |
| Periodic clicks | underrun (check the counter!); DMA buffer cache; halves swapped |
| Stuttering with old data | D-Cache not cleaned before the DMA |
| Skeleton sounds, residual is garbage | int8 state scales (§3.3); states not updated (pointers); graph input order |
| Crash when starting the network | pools/alignment from neural_art.json did not match the linker; LL_ATON reinitialization |
| MIDI is silent | optoisolator polarity (DIN pins 4/5); pull-up on the H11L1 output; baud 31250 |
| Notes "hang" | running status / realtime bytes break the parser state machine; NoteOff lost during voice stealing |
| Latency drifts | NPU t_call jitter (measure the max in H0); quantization to the hop — that is the expected ±hop/2 |

### 11.4 Hygiene
- Commit together with the code: the stedgeai reports, the m0/m4 reports,
  decision_log. Model checkpoints stay outside git (on disk); git holds the
  config plus a hash of the data.
- Every phase ends with a tag: `phase-T0-done`, `M0`, `M1`…

---

## 12. Formula reference (everything in one place)

- States, bytes: `C · (k−1) · Σd · bands · V · (2 if ping-pong)`; Σd = the sum
  of dilations over all layers (cycle × repeats). For k=3, cycle 1…32 ×4:
  Σd=252, (k−1)Σd=504.
- MAC per subband sample (per single sequence): `L·C²·k + c_in·C + C·c_out`.
- Total GOPS: `MAC/sample · 2 · 12000 · bands · V / 1e9`.
- Receptive field, ms: `((k−1)·Σd + 1) / 12` (at 12 kHz). For the baseline
  shape — 42 ms.
- Note→sound latency: `~1.0 (3 MIDI bytes @31250) + hop/2…hop (quantization) + hop (pipeline slot) + 1.32 (PQMF synthesis) + ~0.4 (DAC)`.
  hop=4 ms → 8.7 median / 10.7 worst (spec §5.4 ✓).
- hop 4 ms = 192 samples @48k = 48 samples per band @12k. M55 budget: hop_ms ×
  0.8 million cycles (3.2 million @4 ms).
- A MIDI byte = 10 bits / 31250 = 0.32 ms; NoteOn (status+2) ≈ 0.96 ms.
- PQMF: synthesis delay (N−1)/2 = 63.5 samples = 1.32 ms; the full
  analysis+synthesis path N−1 = 2.65 ms (offline only).

## 13. Acceptance and decision log

The M0–M5 criteria follow the spec §8 table, the system ones follow §9
(latency median ≤9 / worst ≤12 ms; jitter ≤1 hop; NPU utilization ≥ the §1
goal, **measured**; clean joints on the "long notes + pitch strip" test; 24 h
of continuous operation).

the decision log — starting state:
| ID | Question | Status |
|---|---|---|
| D-1a | real tape vs plugin teacher | looking for a transport; the §4.2 protocol is ready for a session |
| D-2 | voices vs width C | waiting for M0 data (§5.3) |
| D-3 | hop 4 → 2 ms | waiting for the call overhead from M0 |
| D-4 | project name | open |
| D-5 | band wiring: batch vs channels vs hybrid | **new**; NOT fixed; both wirings in T1, test §4.5 (partials 2–5 kHz, drive +4, products in band 1); fallback — channels @C=192 |
| D-6 | skeleton render: 48k+analysis vs directly in the subbands | **new**; A/B in phase P (§2.2), watch the latency budget |
| D-7 | reverb (allowed for in the M55 budget, spec §5.3) | **new**; in v1 — a dry path; for this aesthetic that is probably the answer; come back to it after M5 |

## 14. Calendar of the pre-hardware period (rough guide)

- **Today:** order the BOM (spec §11). Phase 0: environment, repository,
  VERSIONS.md. Run both attached scripts, commit the artifacts. Check whether
  the Developer Cloud farm has an N6.
- **Week 1:** phase P — skeleton prototype (A and B), wow/flutter, golden
  vectors. In parallel: the v0 corpus generator.
- **Weeks 2–3:** T0 — StreamingTCN, the equivalence test, export, stedgeai
  iterations up to a "green" graph at C=96/128; QAT skeleton. In parallel: the
  search for a tape transport.
- **Weeks 3–4:** T1 — the characterization session (as soon as there is a
  transport), alignment, training the v0 student, the first listening tests.
- **When the hardware arrives:** H0 (2–4 days) → final training on the frozen
  shape → H1 (a week) → H2 → H3 → H4 → H5.

A last remark on substance: the project is already unusually well specified —
the main danger now is not "I don't know how", but the temptation to start
training a large network **before** T0 confirms the graph is exportable and M0
freezes the shape. The order T0 → shape → final training is the only place
where a violation costs weeks. Everything else in this document survives a
rework cheaply.
