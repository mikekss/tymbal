# N6 against industry practice — an honest review (6 Aug)

The occasion: my own question — did I build a serious synthesizer by all the
rules, or just something that works? The review was done BY READING THE
CODE, not from memory: every line below has a file behind it. The goal is to
know our own holes before a reader of the article finds them.

## 1. What the industry treats as mandatory — and what we have

| Practice | Us | Where |
|---|---|---|
| Voice allocation with steal priorities | **yes**: free → unison slave → oldest in release → oldest | voice.c `alloc_voice` |
| Clickless stealing | **yes by construction**: the envelope is NOT reset on NoteOn, f0 glides from the current one (`if (v->env <= 0.001f)`) — there is no discontinuity | voice.c |
| CC64 sustain pedal | **yes**, with `held` and correct release when the pedal is lifted | voice.c |
| CC120/123 panic (All Sound/Notes Off) | **yes**, both send voices to release | voice.c |
| Pitch bend with a range | **yes**, ±2 semitones | voice.c, n6_config.h |
| Continuous timbre controllers | **yes**: aftertouch → tA, CC1 → tB | voice.c |
| Portamento/glide | **yes**, `glide_s`, exponential on the frame grid | voice.c `n6_vm_tick` |
| Control interpolation inside the block (anti-zipper) | **yes**: `render_span(f_k, f_k1, …)` — rendering between two frames, the DDSP canon | skeleton_b.c |
| A roll-off towards Nyquist instead of a hard cut of partials | **yes**: `SKB_TAPER_HI 24000` | skeleton_b.c |
| Output protection against overflow | **yes**, but a HARD clip at ±0.999969 with a `g_out_clip` counter | main.c |
| Micro-detune/breathing (what makes it sound "analogue") | **yes**: D-19 — unison, OU drift of f0, bloom | voice.c, skeleton_b.c |

Conclusion from this table: the playability layer was not thrown together
just to make it work. Clickless voice stealing is done more elegantly than in
the textbook (there they mute and restart — with us envelope continuity plus
glide gives the same thing for free).

## 2. What is missing — and what it costs

**(a) Events are quantized to the hop — 4 ms of jitter on every note.**
`while (n6_mf_pop(…)) n6_vm_event(…)` runs ONCE before `n6_pipe_hop`
(main.c). So a note-on that arrives mid-block will sound at the start of the
next one: from 0 to 4 ms of random delay. This is not "latency" (that is
heard as a delay and you get used to it) but JITTER — it is heard as
sloppiness in fast playing. The industry solves this by splitting the block
at the event (sample-accurate note timing) and has treated it as mandatory
since the 1990s.

**(b) Key velocity controls loudness only.** `vel_amp` goes into
`frames[i].amp`; brightness does not depend on it directly. On a live
instrument louder = brighter. The continuous controllers (tA/tB) partly cover
this, but by hand, not from the strike.

**(c) Mono.** `dst[2*i] = dst[2*i+1] = smp` (main.c). The DAC is stereo, the
D-19 unison pair is asking to be panned, and it is almost free.

**(d) Output: a hard clip, no DC blocking.** A hard clip is the
worst-sounding way to limit; soft saturation costs ten lines. There is no DC
blocker at all (grep comes back empty) — a DC component from the network or
the skeleton will go into the speaker as an offset.

**(e) No effects — and that gets in the way of LISTENING.** The decision is
right in substance (the sound must come from the refiner, not from a reverb),
but a dry synthesizer is perceived as "badly synthesized". Our fir↔full↔teacher
listening tests run absolutely dry — that is a systematic interference in the
assessment, and it must at least be named in the article.

**(f) No measurements IN THE AUDIO DOMAIN.** We have the spectral distance to
the teacher in dB (eval_chain: FIR +11.25, fp32 +17.16, int8 +16.89) — but
none of the classics: THD+N, noise floor, aliasing products on a sweep. For
the article that is a cheap and very convincing plot.

**(g) The denormal policy is not set explicitly.** FZ/FPSCR is not touched
anywhere. Our exact-zero-in-silence test removes the classic "the tail slides
into denormals and eats cycles" problem for the FIR history, but the PQMF and
one-pole states have not been checked. A cheap check: turn FZ on and measure
the hop during a decay.

**(h) No presets, no state saving.** Deliberately out of scope.

## 3. The structural conclusion (more important than the rest for the article)

Item (a) cannot be fixed all the way — and the reason is architectural, not
lazy. The skeleton ALREADY knows how to render parts of a block:
`render_span(…, i0, i1)`. So for the skeleton a sample-accurate note start is
cheap. But the neural refiner works on a whole hop: to refine with sample
accuracy the network has to be run more often, and the entire 3.2 M cycle
budget falls apart.

**A hybrid pays for its neural part with a block time grid.** That is the
general price of the approach, not a shortcoming of ours: any neural refiner
quantizes your attacks to its own hop. Hence an honest compromise worth both
making and describing: put the skeleton's attack exactly on the sample, leave
the refiner on the grid — it corrects the timbre anyway, not the moment of
the strike.

## 4. What to take, by "musical value / risk" ratio

1. **Stereo spread of the unison** — the effect is large, the risk zero: the
   panning is done where samples are written into the SAI, out48 and CK4 are
   not touched at all.
2. **Soft clip + a DC blocker on the output** — a dozen lines, audible
   immediately; it changes out48 → a zero default is mandatory, as in D-19.
3. **Key velocity → brightness** — mix vel into tA/tB or into bloom.
4. **Sample-accurate note start (skeleton)** — the most expensive and the
   most "professional" change; the span machinery already exists, but it
   touches pipe_hop.

All four follow the D-19 rule: zeros = the previous behaviour, golden/CK4 do
not move.
