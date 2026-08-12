# N6 audio path BOM (the thing that was missing from the very start — 6 Aug)

Why this file only appears now: the project had no BOM at all. The signal
path (the I2S pinout) was written out honestly, because it touched the
firmware, while the amplifier supply was left as "we will work it out from the
markings once we have a photo". This is the same mistake we have already paid
for in this project: the interface is described, the power is forgotten. Here
it cost us a surprise at the purchasing stage.

## 1. From the board to the ears — the full list

| # | Item | Status | Note |
|---|---|---|---|
| 1 | GY-PCM5102 module (PCM5102A) | in transit | jumpers FLT/DEMP/FMT=L, XSMT=H; SCK→GND |
| 2 | Dupont female-to-female wires, 5 pcs | have | VIN, GND, BCK(D12), LRCK(D11), DIN(D10); < 15 cm, twisted in pairs with ground |
| 3 | 3.5 mm male-to-male cable | to buy | DAC → amplifier input (see §3) |
| 4 | CIRMECH WM-004 amplifier (TPA6120A2) | have | ±12 V rails, AC-GND-AC input |
| 5 | **±12 V amplifier supply** | **NO — this is the hole** | options in §2 |
| 6 | Headphones | have | preferably not the lowest-impedance ones (see §4) |
| 7 | For the first try: any powered speakers with 3.5 mm | have? | the amplifier is NOT needed to check "sound is coming out" |
| 8 | SSM3582 (class D, I2S input, 2×31 W) | ordered, ~7 Aug | NOT a DAC, a separate branch: drives a 3–8 ohm speaker directly, no analogue path; standalone mode without I2C; there is NO volume control — only our software multiplier, starting at −40 dB |

## 2. Amplifier supply — four routes, a transformer is not mandatory

The board needs two polarities. The AC-GND-AC terminal block feeds a
rectifier, and that feeds a 78M12 (+12) and a 79M12 (−12). So anything that
gives symmetry about a centre point will do.

**(a) Two identical ISOLATED 12–15 V DC adapters** — the cheapest route.
Connect the minus of the first to the plus of the second; that point = GND
(the middle contact of the terminal block). Plus of the first → one AC pin,
minus of the second → the other AC pin. The bridge diodes will simply take
0.7 V each. With 12 V units the regulators drop out and the output will be
≈ ±10 V — for the TPA6120A2 (datasheet ±5…±15 V) that is an operating point
with enormous headroom: even ±10 V give about a watt into 32 ohms, and
headphones need milliwatts. With 15 V units the regulators stay in
regulation — exactly ±12 V, as intended.
Condition: the adapters must be ISOLATED from each other (two-pin, no earth).
Cheap "chargers" almost always are.

**(b) An isolated pedalboard power supply** — if its outputs are isolated
sections (proper ones have a transformer per output; cheap daisy-chains share
a ground and will not do), two 9 V sections in series give ±9 V ahead of the
regulators and ≈ ±7 V after them. Also within the TPA6120A2 datasheet range,
and also louder than needed.
Isolation check: a meter between the "minus" of one section and the "minus"
of another — if it reads a short, the sections are common and this route is
no good.

**(c) A 2×12 V AC transformer, 15–30 VA** — as the designer intended, the
quietest one in terms of noise. 2×9 V is too little (the 78M12 wants ≳ 14.5 V
at its input), 2×15 V is the ceiling set by the 35 V capacitors and by TO-252
heating without a heatsink.

**(d) An isolated ±15 V DC-DC module** — compact, but the switching noise
will have to be suppressed; it only makes sense if everything is boxed up
together.

If after (a)/(d) you hear a whistle or a squeal in silence — that is the
adapter, not our synthesizer: check by unplugging the signal cable (the
whistle stays — it is the supply; it goes away — it is the path).

## 3. What plugs into what

Amplifier input: `AUDIO01#` (white JST, labelled R–GND–L) and `AUDIO02#`
(black 3.5 mm jack) — parallel inputs. The output is the green `EARPHONE`.
Before the first power-up, RING IT OUT: the `AUDIO02#` contacts should ring
through to the outer terminals of the volume pot (then it really is the
input). Simplest: jack-to-jack from the DAC into `AUDIO02#`, leave the JST
alone.

Grounds: the DAC is powered from the board's 5 V (USB), the amplifier from
its own source. Join the grounds ONLY through the signal cable, at a single
point. Otherwise you get a loop and 50 Hz hum.

Power-up order: volume to zero → amplifier supply → board → volume up.
Power-down is the reverse.

## 4. What to remember during a listening test

My headphones are AKG K512 MkII: 32 ohms, 109 dB SPL/V. Hence the level
arithmetic: 85 dB (comfortable) = 0.063 V at the headphones, 100 dB (loud) =
0.355 V, while the path delivers up to 2.1 V × the board's gain — at a
typical gain of 2 that is 121 dB SPL and 0.55 W into headphones rated at
about 0.2 W. **The path has about 30 dB too much**, and the whole useful
travel of the knob is the first tenth of a turn.

The B50K pot is LINEAR, and in that first tenth its channel mismatch is at
its worst. Fixes in increasing order: (1) a software attenuation of
−20…−24 dB at the 24-in-32 packing stage (3.3 bits out of 24, with the DAC's
112 dB dynamic range — free; the code does NOT exist yet, the candidate is
`g_out_gain` in main.c); (2) swap for an A50K (logarithmic, same footprint,
pennies).

The resistors were read off a photo of the board (5-band, 1%): the EARPHONE
output has 10.0 ohms ×2 (confirmed, not "typical"); the input has 47.0 ohms
×2 (exactly TI's recommendation); at the chip there is 1.00 kohm, almost
certainly the feedback resistor (the value is from the datasheet). Its
partner RG cannot be read → the gain is NOT confirmed: measure with a meter
from the inverting input to ground (power off), gain = 1+RF/RG; or feed a
known tone and read the output with a voltmeter in AC mode.

The 10 ohms in series at the amplifier output will noticeably colour the
bottom end with low-impedance earbuds. Do the fir↔full↔teacher comparisons on
the same headphones, preferably high-impedance ones, so that the amplifier
does not mix its own correction into a verdict about the synthesis. The first
power-up of any newly assembled path — headphones NOT on your head (the
available level is 121 dB: an error in the frame format means full-scale
noise).

## 5. Bring-up does not wait for the amplifier

The PCM5102A puts out line level (~2.1 V rms). The "sound comes out at all,
underrun=0, no clicks" check can and should be done into any powered speakers
with a 3.5 mm input, well before the amplifier supply question is settled.
The amplifier is only needed for listening on headphones — that is, for step
5, not for step 6.
