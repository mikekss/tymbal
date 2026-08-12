# Listening tests: what to listen to and what follows from what (8 Aug)

The rule this file exists for: **a listening test only makes sense if
different answers lead to different actions.** Below, for every pair, what we
will do on each outcome is written down in advance. If the outcome changes
nothing — we do not audition that pair, we just play for our own pleasure.

## Material

Drones are useless — checked on 5 Aug ("twelve identical fragments, the whole
point is in the attacks and the decays"). Two scores work:

- `pulse` — repeating hits: shows the ATTACK and the start of the decay;
- `line` — a moving line: shows the DECAY, the transitions and tuning
  stability.

```
py train\audition_delta.py --score pulse --dur 8 --prefix pulse
py train\audition_delta.py --score line  --dur 8 --prefix line
```

The directory you launch from does not matter: the script derives paths from
its own file (`HERE`), the output always lands in `dsp/audition/`. The
interpreter is the **system `py`** (3.12.10, torch 2.4.1+cu118): torch is NOT
installed in `C:\ST\venv-n6`, there is no need to activate it. If the console
is in cp1252, printing Cyrillic kills the script with `UnicodeEncodeError`
AFTER the computation is done — cured by
`$env:PYTHONIOENCODING="utf-8"` (not needed in a normal Russian console).

**Both runs were already done on 8 Aug, the files are in `dsp/audition/`:**
`pulse_{dry,fir,full,teacher}.wav` and `line_{dry,fir,full,teacher}.wav`.

## Numbers from the 8 Aug run — and a missed prediction

Residual suppression relative to the teacher, fp32, on these very phrases:

| phrase | FIR alone | FIR + network | **network contribution** |
|---|---|---|---|
| `pulse` | +14.62 dB | +15.91 dB | **+1.29** |
| `line`  | +19.53 dB | +23.79 dB | **+4.26** |

The prediction before the run was exactly the opposite: "the network buys the
attack, on the steady state almost nothing". The network gives more than
three times as much on `line` as on `pulse`.

By the project's rule a missed prediction is a signal, not an annoyance, and
the next step here is not a new hypothesis but ears. Still, a sharpening of
the wording already suggests itself: the word "dynamics" in the author's
verdict ("the difference is exactly in the dynamics") we read straight away
as "attack". The numbers say it is more about how the sound DEVELOPS between
notes and inside a phrase, not about the hit as such: on `pulse` — repeating
identical hits — the linear part takes almost everything.

The listening test has to close this question: whether those 1.29 dB on
`pulse` are audible at all, and what the 4.26 dB on `line` sound like.

Each run puts four files into `dsp/audition/`: `dry` (the bare skeleton),
`fir` (skeleton + FIR), `full` (skeleton + FIR + network), `teacher` (the
target) — and prints the residual suppression in dB.

**Caveat:** on the host, `full` is the network in fp32. On the board it is
int8: minus 0.27 dB in spectral distance and noticeably more high-frequency
sand (D-23). That is, the host `full` sounds a little cleaner than the board.
For comparison against the teacher it is the right file; for "how it actually
is" — button B1 on the board.

## Pair 1. `fir` ↔ `full` — does the network earn its place

Live this is done with button B1, and it already gave a verdict on 7 Aug:
"the difference is exactly in the dynamics, the instrument becomes alive".
What is left is to find the BOUNDARY — where the network stops being audible.

| what is heard | conclusion | what we do |
|---|---|---|
| difference clear on `pulse`, weak on `line` | the network works on transients, almost nothing on the steady state | in the article we state the boundary honestly: the refiner buys the attack, not the timbre |
| clear on both | the contribution is wider than we thought | we strengthen the claim, add it to §4 |
| not audible anywhere | contradicts eval_chain (+5.64 dB) | we do NOT trust the ears and do not trust the metric — we go and check a third way (a per-band breakdown) |

## Pair 2. `full` ↔ `teacher` — how far from the target

| where the remaining difference sits | conclusion | what we do |
|---|---|---|
| in the midrange, on the steady-state sound | we got the dynamics, the remainder is timbral: the ceiling of the network's capacity or of the teacher's character | we write it into the article as an honest ceiling of the method; we do not start retraining |
| on the attacks | the network did NOT get the thing it was built for | serious: we revisit the shape and the training, D-2 is unfrozen |
| in the noise and the "sand" up top | that is our D-23, not the model | we fix the noise, not the network |

## Pair 3 (optional). `dry` ↔ `fir` — how much the linear part gives

Material for §4 of the article ("is a network even needed here?"). If FIR
alone already almost catches the teacher on `line`, that is a strong argument
that an honest linear baseline must come before any network — our main
transferable conclusion.

## Comparison hygiene (otherwise the result is about loudness, not sound)

1. **Level-match by ear** before comparing. Louder is always "better".
2. Chunks of 3–5 seconds, switch instantly. The ear adapts within seconds;
   after half a minute the difference "disappears" whether it is there or not.
3. **Write down the first impression before the second listen.** The second
   is already spoiled by the first.
4. Do not compare from memory a minute later — only back-to-back A/B.
5. No more than 15–20 minutes at a time.
6. Name NOT "better/worse" but a feature: attack density, length and
   character of the decay, breathing in the midrange, noise/sand, tuning
   stability.

## Results

_(in my own words; format — 3–5 phrases per pair, naming the score)_

### Pairs 1 and 3 — 8 Aug, what I wrote down at the time (verbatim)

> "They're all distinguishable. The network gives roundness and velvet, that
> fuzzy quality. Plus the attack came out with a hint of an organ or a
> shakuhachi, with that overblow you get on wind instruments. FIR is less
> fuzzy, but it's different from dry too."

Breakdown:

- **All three pairs are distinguishable by ear**, `pulse` included. So the
  ear DOES catch the network's +1.29 dB contribution on repeating hits — the
  boundary of the method lies lower than the number would suggest. A separate
  small result: here the measurable and the audible agreed instead of
  diverging (unlike the int8 quantization story, where −0.27 dB turned out to
  be clearly audible, and the skeleton noise story, where −19 dB turned out
  to be −2.4 dB where the ear listens).
- **"Roundness, velvet, that fuzzy quality"** — a description of TEXTURE, not
  of brightness and not of loudness. That is exactly the ear-signature of
  multiband compression: the teacher is called `A2_ottpress`, and OTT gives
  precisely that densification, envelope flattening and "fluffiness" in the
  midrange. The circle closed from an unexpected side: the teacher was chosen
  numerically, and the ear described its character independently.
- **The attack "with a hint of an organ or a shakuhachi, with the
  overblow"** — that is chiff: a burst of upper partials and noise at the
  moment of attack, as on a wind instrument being overblown. WHERE it comes
  from is an open question (see below), and it is not a trifle: it strongly
  affects how the instrument is perceived.
- **FIR alone is already different from `dry`, but "less fuzzy"** — the
  linear part takes the main share (the numbers: +14.6 and +19.5 dB out of
  ~+16 and ~+24), and the network adds exactly the texture. Material for §4
  of the article: an honest linear baseline must come before the network, but
  it does not replace it either.

### Pair 2 (`full` ↔ `teacher`) — CLOSED 8 Aug

> "The teacher has the overblow as well. The character is reproduced, quite
> well. The midrange is pretty similar. The teacher has the sand too. Our
> sound came out a bit brighter than the teacher."

Four answers, and each leads to a consequence written down in advance:

1. **The teacher has the overblow too** → the chiff is not an artefact but a
   reproduced feature of the target. The "keep it or fix it" question is
   settled: keep it, this is a success.
2. **The midrange is similar** → per the table above: we got the dynamics,
   the remainder is timbral, this is an honest ceiling of the method. **We do
   NOT start retraining**, D-2 stays frozen.
3. **The teacher has the sand too** → this changes D-23 at the root. Part of
   our noise is not a defect but fidelity to the target: `A2_ottpress` is
   multiband compression, and it lifts low-level content, the noise floor
   included. So flattening the noise "down to zero" would take us AWAY from
   the teacher.
4. **We are a bit brighter than the teacher** → the only remainder named,
   and it is spectral tilt, not structure.

### Synthesis: brightness and sand are probably the same thing

The skeleton noise is added with the SAME weight into all four subbands
(measurement of 7 Aug: above 4 kHz it is only 2.4 dB quieter than the tone).
Noise that is flat across the bands is exactly an excess of high-frequency
energy. So "sand" and "a bit brighter than the teacher" are with high
probability not two observations but one.

Hence a falsifiable prediction for D-23: **a noise tilt across the subbands
should simultaneously remove the excess sand AND pull our tilt towards the
teacher's — that is, IMPROVE the metric, not make it worse.** If the metric
gets worse, the noise was carrying a load and must not be touched.

Checkable without the board and without retraining: `eval_chain` with a
per-band breakdown, before and after the tilt. That is exactly the tool that
has to be fixed anyway after two misses with wideband metrics.



## Measurement following the ears (8 Aug): "brighter" turned out to be temporal, not spectral

Checking the ear's assessment "our sound is a bit brighter than the teacher"
with a number, on the same files (`dsp/audition/{line,pulse}_*.wav`, fp32,
equal RMS):

- **The spectral shape matches.** Above 220 Hz the full path lies within
  ±0.3 dB of the teacher across third-octave bands; the linear part alone
  deviates by 2…5 dB. Our share of energy above 4 kHz is even 0.1 dB LESS
  than the teacher's — so "brighter" is not about the amount of top end.
- **The typical dynamics match.** The median short-term crest factor
  (50 ms window) is 7.41 dB against the teacher's 7.42 on `line`, 5.66
  against 5.67 on `pulse`.
- **The tail diverges.** At the 99.9th percentile we are 1.8 dB higher, and
  by absolute peak 2.9 dB (`line`) and 3.8 dB (`pulse`). The teacher is
  multiband compression, it shaves the outliers off; the refiner reproduces
  them and in places amplifies them.
- The rest of the spectral divergence is below 200 Hz (we are hotter by
  3.5…7.5 dB), where these phrases have almost no energy.

Conclusion: **the ear reported a spectral difference, and the measurement
found a temporal one.** Rare short bursts on the attacks read as "brighter"
while the average spectrum stays put. The third case in the project of a
metric and audibility diverging — and the first where it was not the
instrument that was wrong but the interpretation.

Plots: `docs/img/spec_vs_teacher.png`, `docs/img/peaks_vs_teacher.png`.
