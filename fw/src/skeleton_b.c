/* skeleton_b.c — rendering a voice straight into the PQMF subbands (variant
 * B, D-6). A transcript of dsp/skeleton_b.py::render_voice_b_cstyle
 * (contracts B-1..B-5). The key point: the master phase is a 48k accumulator
 * (B-1); during catch-up the first up to three increments of the new hop
 * still belong to the PREVIOUS frame — the state keeps prev_f0. Table
 * g_k(f): fw/n6_bandresp.h (locked in, cumsum@48k convention).
 *
 * TARGET SEMANTICS: all the hot arithmetic is f32; the phase is uint32
 * (exact modular wrap instead of fmod); sin(h*phi) — one sine per sample plus
 * an incremental rotator over h; the spectral tilt h^(-p) — our own 2^x.
 * THERE ARE NO libm CALLS ON THE FRAME PATH (3 Aug): sinf/cosf replaced by a
 * table indexed by the uint32 phase, llrint by adding 0.5 and a single VCVT,
 * exp2f by a polynomial (skb_exp2). Verified with the disassembler: in the
 * whole file exactly two bl into libm remain — sin and log2f, both in the
 * one-time initialization of the tables. double is left only in the
 * conversion of f48 -> phase increment, where it is genuinely needed: in
 * float the coherent quantization error drags the phase off. */
#include <math.h>
#include <string.h>
#include "n6_dsp.h"
#include "xorshift.h"
#include "../n6_bandresp.h"

/* MVE/Helium: 4 bands == one f32x4 vector. At startup the response table is
 * repacked into the interleaved form [idx][band] (two contiguous vld1q per
 * harmonic instead of 16 scalar loads from 8 rows). The formulas and the
 * order of operations FOR EACH band are the same as in the scalar branch (the
 * reference of the host test); the only difference is the fused multiplies
 * (vfma) — within tolerance. */
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
#include <arm_mve.h>
#define SKB_HAVE_MVE 1
/* LOCALITY (2 Aug): re and im lay in TWO arrays of 15.4 kB each, and each
 * harmonic took 4 scattered accesses — two into re4, two into im4, with
 * 15 kB between them. With a grid step of step_ix = f0/25 (for f0=440 that is
 * 17.6 nodes = 281 bytes per harmonic) each of them is its own cache line,
 * and the working set per subband sample overflows the D-cache and is re-read
 * 48 times per hop for each voice. Here there is one interleaved array: node
 * i stores {re[0..3], im[0..3]} contiguously, and the pair of nodes (i, i+1)
 * for interpolation is 64 contiguous bytes. The values and the order of
 * operations are the same: the numbers are bit-for-bit as before
 * (cross-checked by CK4 on the board). */
static float g_br8[N6_BANDRESP_N * 8] N6_DTCM;      /* 30.0 kB — in DTCM */
static int   g_br_init;
static void skb_br_transpose(void)
{
    for (int i = 0; i < N6_BANDRESP_N; ++i)
        for (int b = 0; b < N6_BANDS; ++b) {
            g_br8[i * 8 + b]     = n6_bandresp_re[b][i];
            g_br8[i * 8 + 4 + b] = n6_bandresp_im[b][i];
        }
    g_br_init = 1;
}
#endif

/* Build guard: the scalar branch on the target is several times more
 * expensive than the vector one, and losing Helium happens silently — it is
 * enough for MCU Settings to carry -mfpu=fpv5-d16: on the Cortex-M55 this
 * flag overrides -mcpu and kills MVE. Cured by -mfpu=auto. (On 2 Aug the
 * build turned out to be correct, but diagnosing that had to be done in a
 * roundabout way — let it shout for itself from now on.)
 *
 * HOW TO CHECK MVE IN THE DISASSEMBLY (4 Aug, burnt twice). Counting
 * "vector" mnemonics is useless: vmul.f32/vfma.f32 exist on the scalar FPU
 * too, the only difference is the register (q versus s). Worse, objdump from
 * binutils 2.42 does not decode part of MVE and prints it as coprocessor
 * instructions: `ldc 15, crN, [rM], #16` — that is vldrw.u32 qN, [rM], #16,
 * `stc 15, ...` — vstrw, `cdp 14, 3, crD, crN, crM` — vmul/vfma with a
 * scalar operand. Reliable signs of the hot loop: a ldc/stc pair with a #16
 * step, a cdp next to it and `le lr, <start>` (low-overhead loop) at the end.
 * Plus the check `-E -dM | grep __ARM_FEATURE_MVE` — it must be 3
 * (MVE + FP). */
#ifdef SKB_HAVE_MVE
int n6_skb_have_mve(void) { return 1; }
#else
int n6_skb_have_mve(void) { return 0; }
#ifdef N6_TARGET
#warning "skeleton_b: MVE/Helium is OFF -> scalar branch. Check -mfpu: fpv5-d16 kills MVE, you need -mfpu=auto"
#endif
#endif

/* --- skb broken down by parts (3 Aug) ---------------------------------
 * prof[0] in pipeline.c gives the skeleton as a single number, and three
 * edits in a row missed by an order of magnitude: the body shrank from 15
 * operations to 4, and the gain turned out to be 11 cycles out of 96 per
 * harmonic-sample. So the cost is not where I modelled it. Here are three
 * separate counters — a measurement instead of yet another guess:
 *   [0] skb_wtab   — precomputing the response, 12 segments per hop
 *   [1] prologue   — amplitudes and rotator, a chain of dependent FMAs
 *   [2] body       — vector accumulation over harmonics
 *   [3] decoder    — skb_decoder, 1-2 calls per hop per voice       (4 Aug)
 *   [4] phase      — catch-up of 48k increments, 4 per sample       (4 Aug)
 * The first three closed 583 k out of 759 k, and the 176 k of "remainder"
 * stayed a guess for three attempts in a row; [3] and [4] close its two most
 * likely pieces. What is outside the counters now: the ag/tb interpolation,
 * the noise with the write into the subbands, and the loop overhead. On the
 * host there are no counters. */
uint32_t n6_skb_prof[5];
#ifdef N6_TARGET
extern uint32_t n6_cyc_now(void);
#define SKB_T0()      uint32_t _st = n6_cyc_now()
#define SKB_P(slot)   do { uint32_t _n = n6_cyc_now(); \
                           n6_skb_prof[slot] += _n - _st; _st = _n; } while (0)
#else
#define SKB_T0()      do {} while (0)
#define SKB_P(slot)   do {} while (0)
#endif

#define SKB_W_EPS    1e-4f
#define SKB_NOISE_B  N6_SKB_NOISE_B              /* B-3 */
#define SKB_TAPER_LO 22000.0f
#define SKB_TAPER_HI 24000.0f

/* --- sine from a table instead of sinf/cosf (3 Aug) --------------------
 * The measurement split skb into four parts, and the largest turned out to
 * be the one I had not considered for three attempts: 252 k cycles out of
 * 818 k — that is not the loop over harmonics but fixed work per sample. Two
 * newlib calls (sinf, cosf) plus llrint on double: they do not inline and
 * cost hundreds of cycles, and they are called 96 times per hop.
 *
 * The phase is stored as a uint32 of a full turn anyway, so a table is
 * natural here: the top 10 bits are the index, the next 12 the fractional
 * part for linear interpolation. The cosine is taken from the same table by
 * a shift of a quarter turn. The linear interpolation error over 1024 nodes
 * is of the order (2*pi/1024)^2/8 ~ 4.7e-06, that is about -106 dB: two
 * orders of magnitude quieter than the already accepted response
 * segmentation error (-56 dB), so it does not affect the overall metric. */
#define SKB_SIN_N    1024
static float g_sin[SKB_SIN_N + 1] N6_DTCM;          /* 4.1 kB */
static int   g_sin_init;
static void skb_sin_build(void)
{
    for (int i = 0; i <= SKB_SIN_N; ++i)
        g_sin[i] = (float)sin(6.283185307179586476925286766559
                              * (double)i / (double)SKB_SIN_N);
    g_sin_init = 1;
}
/* sin(2*pi*q/2^32) with linear interpolation */
static inline float skb_sin_q(uint32_t q)
{
    uint32_t i = q >> 22;                       /* top 10 bits */
    float    f = (float)((q >> 10) & 0xFFFu) * (1.0f / 4096.0f);
    float    a = g_sin[i], b = g_sin[i + 1];
    return a + (b - a) * f;
}
#define skb_cos_q(q)  skb_sin_q((uint32_t)((q) + 0x40000000u))

#define SKB_HZ_TO_Q  (4294967296.0 / (double)N6_FS48)   /* Hz -> quanta/step */
#define SKB_Q_TO_RAD (6.283185307179586476925286766559 / 4294967296.0)

static const uint32_t skb_band_seed[N6_BANDS] = {
    0x00000000u, 0x9E3779B9u, 0x3C6EF372u, 0xDAA66D2Bu };

void n6_skb_init(n6_skb_voice_t *v, uint32_t seed)
{
#ifdef SKB_HAVE_MVE
    if (!g_br_init) skb_br_transpose();
#endif
    if (!g_sin_init) skb_sin_build();
    memset(v, 0, sizeof *v);
    for (int b = 0; b < N6_BANDS; ++b) {
        v->nseed[b] = n6_xs32_seed(seed ^ skb_band_seed[b]);
        v->nw[b] = SKB_NOISE_B;                 /* D-23: 0 dB = as it was */
    }
    v->have_prev = 0;
}

/* --- 2^x without a libm call (4 Aug) ----------------------------------
 * The decoder computes the spectral tilt a_h = h^(-p) = 2^(-p*log2 h), a
 * hundred times per voice per hop — 200 exp2f calls per hop, and that was
 * all that was left of libm on the frame path. newlib's exp2f is 62
 * instructions plus the call prologue and epilogue; here it is the
 * decomposition 2^x = 2^n * 2^r, where n is the nearest integer,
 * r = x - n lies in [-0.5, 0.5], the factor 2^n is assembled directly from
 * the float exponent field, and 2^r is computed by a fifth-degree polynomial
 * (Chebyshev on [-0.5, 0.5]). The coefficients are rounded to f32; checked
 * numerically: the maximum relative error is 2.9e-07 on the polynomial and
 * 2.8e-07 on the finished a_h for p from 1 to 3 — that is the resolution
 * level of float itself, that is -131 dB against the -56 dB of response
 * segmentation accepted in the project.
 *
 * The argument here is always x = -p*log2(h) <= 0 and no more than ~20 in
 * magnitude, but n is clamped anyway: the network supplies tA, and guarding
 * against an outlier from it costs two comparisons per harmonic, whereas
 * debugging denormals costs more. */
static inline float skb_exp2(float x)
{
    int n = (int)(x < 0.0f ? x - 0.5f : x + 0.5f);   /* round to nearest */
    if (n < -126) return 0.0f;
    if (n >  127) n = 127;
    float r = x - (float)n;
    float p = 0.00133952798f;
    p = p * r + 0.00967076313f;
    p = p * r + 0.0555034068f;
    p = p * r + 0.240222117f;
    p = p * r + 0.6931472f;
    p = p * r + 1.0f;
    union { uint32_t u; float f; } e;
    e.u = (uint32_t)(n + 127) << 23;                 /* 2^n */
    return p * e.f;
}

/* v0 decoder C-2: tilt 1/h^(3-2tA) + Nyquist taper, sum(A)=1 (C-1/C-5).
 * D-19 (5 Aug): the tilt breathes with the envelope — p += bloom_k*(1-gate).
 * The attack opens up over the attack time of env (8 ms), the decay darkens
 * before the low end does: the behaviour of a filter with an envelope, the
 * main anti-"chiptune". bloom_k=0 is exactly the old decoder (golden/CK4 are
 * computed with zero). */
static void skb_decoder(const n6_skb_voice_t *v, const n6_frame_t *f,
                        float A[N6_NH])
{
    static float log2h[N6_NH];
    static int   log2h_init = 0;
    if (!log2h_init) {                  /* single thread: main loop/tick (§8.3) */
        for (int h = 1; h <= N6_NH; ++h) log2h[h - 1] = log2f((float)h);
        log2h_init = 1;
    }
    float p = 3.0f - 2.0f * f->tA, norm = 0.0f;
    if (v->bloom_k != 0.0f) {
        float g = f->gate;
        if (g < 0.0f) g = 0.0f; else if (g > 1.0f) g = 1.0f;
        p += v->bloom_k * (1.0f - g);
    }
    for (int h = 1; h <= N6_NH; ++h) {
        float a = skb_exp2(-p * log2h[h - 1]);
        float fh = f->f0 * (float)h;
        float tp = (SKB_TAPER_HI - fh) * (1.0f / (SKB_TAPER_HI - SKB_TAPER_LO));
        if (tp < 0.0f) tp = 0.0f; else if (tp > 1.0f) tp = 1.0f;
        a *= tp;
        A[h - 1] = a;
        norm += a;
    }
    if (norm < 1e-12f) norm = 1e-12f;
    float inv = 1.0f / norm;
    for (int h = 0; h < N6_NH; ++h) A[h] *= inv;
}

/* HOISTING THE RESPONSE INTERPOLATION OUT OF THE PER-SAMPLE LOOP (3 Aug,
 * my decision).
 *
 * The path up to this point. At first the g_br8 table (30.0 kB) was combed
 * through in full for every subband sample — 96 passes per hop, ~330 kB of
 * line traffic. Interchanging the loops (harmonics in blocks on the outside)
 * removed that: skb while playing 1.11 -> 0.93 Mcyc. Bringing back the scalar
 * prologue and the branchless body gave NOTHING (0.93 -> 0.92). So neither
 * memory, nor dependencies, nor branches get in the way any more: it is
 * limited by the number of operations. ~920 k cycles for 9600 (harmonic,
 * sample) pairs is ~90 cycles, while the body on the M55 costs about forty
 * cycles by construction.
 *
 * More than half of the body is the response interpolation: 4 loads and 4
 * arithmetic operations for EVERY pair. But wre/wim depend on the sample only
 * through f0, and inside a hop that is 4 ms: even on a fast glissando f0
 * moves by fractions of a hertz with a grid step of 25 Hz. We compute them
 * ONCE per hop from the average f0 and immediately with the mask applied
 * (below the threshold — exactly zeros, and then the body does not need a
 * predicate either).
 *
 * THE NUMBERS CHANGE, and that is a deliberate price: the band response is
 * taken from one f0 per hop instead of its own for every sample. The error is
 * small and smooth, but CK4 is NO longer bit-for-bit — the check is now by
 * metric (rel RMS against the Python reference) and by ear. The bit-exact
 * form is in the git history.
 *
 * The body shrank to four operations: two vmul_n, vadd, vfma. The working set
 * of the table became 3.2 kB instead of 30 and is read sequentially, so the
 * block interchange is no longer needed — the loop went back to the simple
 * form "samples outside, harmonics inside" with a scalar prologue. */

#ifndef SKB_WSEG
#define SKB_WSEG 4                      /* samples per one response recompute */
#endif

/* Band response for all live harmonics from f0 at moment fr inside the frame.
 * The threshold mask is applied here: below the threshold we store exactly
 * zeros, and the body of the hot loop becomes unconditional. */
static void skb_wtab(const n6_frame_t *f_k, const n6_frame_t *f_k1,
                     int h_last, float fr,
                     float w_re[][N6_BANDS], float w_im[][N6_BANDS])
{
    float f0m = f_k->f0 + (f_k1->f0 - f_k->f0) * fr;
    float step_ix = f0m * (1.0f / N6_BANDRESP_GRID_HZ);
    const float br_lim = (float)(N6_BANDRESP_N - 1) - 1e-3f;
    float idx = step_ix;
#ifdef SKB_HAVE_MVE
    /* READ ONLY THE INTERLEAVED g_br8. The first version of this function was
     * copied off the scalar branch and reached into n6_bandresp_re/im[b][i0]
     * — that is eight scattered loads per harmonic, exactly what g_br8 was
     * made for on 2 Aug to get away from. The precomputation came out more
     * expensive than what it saves: skb while playing grew 0.92 -> 1.76 Mcyc.
     * Here node i is 32 contiguous bytes {re[0..3], im[0..3]}, and the pair
     * of nodes for interpolation is 64 bytes in a row. */
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    for (int h = 1; h <= h_last; ++h) {
        float ix = idx;
        if (ix > br_lim) ix = br_lim;
        int i0 = (int)ix;
        float dfr = ix - (float)i0;
        const float *p8 = &g_br8[(size_t)i0 * 8];
        float32x4_t re0 = vld1q_f32(p8),     im0 = vld1q_f32(p8 + 4);
        float32x4_t re1 = vld1q_f32(p8 + 8), im1 = vld1q_f32(p8 + 12);
        float32x4_t wre = vfmaq_n_f32(re0, vsubq_f32(re1, re0), dfr);
        float32x4_t wim = vfmaq_n_f32(im0, vsubq_f32(im1, im0), dfr);
        float32x4_t mag = vfmaq_f32(vmulq_f32(wre, wre), wim, wim);
        mve_pred16_t pk = vcmpgtq_n_f32(mag, SKB_W_EPS * SKB_W_EPS);
        vst1q_f32(w_re[h - 1], vpselq_f32(wre, vzero, pk));
        vst1q_f32(w_im[h - 1], vpselq_f32(wim, vzero, pk));
        idx += step_ix;
    }
#else
    for (int h = 1; h <= h_last; ++h) {
        float ix = idx;
        if (ix > br_lim) ix = br_lim;
        int i0 = (int)ix;
        float dfr = ix - (float)i0;
        for (int b = 0; b < N6_BANDS; ++b) {
            float wre = n6_bandresp_re[b][i0]
                + (n6_bandresp_re[b][i0 + 1] - n6_bandresp_re[b][i0]) * dfr;
            float wim = n6_bandresp_im[b][i0]
                + (n6_bandresp_im[b][i0 + 1] - n6_bandresp_im[b][i0]) * dfr;
            int live = (wre * wre + wim * wim > SKB_W_EPS * SKB_W_EPS);
            w_re[h - 1][b] = live ? wre : 0.0f;
            w_im[h - 1][b] = live ? wim : 0.0f;
        }
        idx += step_ix;
    }
#endif
}

/* RENDERING IN CHUNKS (4 Aug). Previously the whole hop was computed here at
 * once, and pipeline.c called this BEFORE the NPU polling loop. The
 * measurement showed what that order costs: LL_ATON_RT_RunEpochBlock is not
 * blocking (13209 calls for 69 blocks, ~90 cycles each), that is, for 1.2 M
 * cycles per hop the processor spins in empty polling — while 0.6 M of
 * skeleton work has by then already been done and is queued BEFORE the idle
 * time. Worse, before the first poll the NPU is not started at all: the
 * runtime is a state machine that the processor drives.
 *
 * That is why the render is cut into spans of samples, and pipeline.c
 * alternates them with pumping. The arithmetic does not change by a single
 * bit (checked by CK4), only the order in time changes. Preparing the frame —
 * the decoder, h_last, dA — is expensive and is done once, on the span with
 * i0 == 0; finalizing the state is done on the last one. Hence the file-scope
 * statics instead of locals: pipeline.c must take a voice through to the end
 * before starting the next one (voices are not interleaved within a hop), and
 * that is the only requirement on the caller. */
static float A_next[N6_NH] N6_DTCM;   /* amplitudes of frame k+1 */
static float dA[N6_NH]     N6_DTCM;   /* A_next - A_prev, computed once per hop */
static float w_re[N6_NH][N6_BANDS] N6_DTCM;   /* 1.6 kB, band response */
static float w_im[N6_NH][N6_BANDS] N6_DTCM;   /* 1.6 kB */
static int   h_last, h_pad, seg_cur;

void n6_skb_render_span(n6_skb_voice_t *v, const n6_frame_t *f_k,
                        const n6_frame_t *f_k1,
                        float sub[N6_BANDS][N6_HOP12], int i0, int i1)
{
    if (i1 > N6_HOP12) i1 = N6_HOP12;
    if (i0 >= i1) return;

    if (i0 == 0) {
    {   SKB_T0();
        if (!v->have_prev) {                    /* the voice's first hop */
            skb_decoder(v, f_k, v->A_prev);
            v->have_prev = 1;
            v->phi_q = 0u;
            v->n48 = 0;
        }
        skb_decoder(v, f_k1, A_next);
        SKB_P(3); }

    /* the top live harmonic of both frames: we do not spin the rotator past it */
    h_last = N6_NH;
    while (h_last > 1 && v->A_prev[h_last - 1] <= 0.0f
                      && A_next[h_last - 1] <= 0.0f)
        --h_last;
    /* The vector prologue goes in fours, so we count up to a multiple of four.
     * The extra harmonics are harmless by the construction of h_last: there
     * A_prev and A_next are exactly zero (the decoder gives non-negative
     * amplitudes), so p_h and q_h are zero too, and the contribution to the
     * sum is zero whatever w_re/w_im contain. */
    h_pad = (h_last + 3) & ~3;
    for (int h = 0; h < N6_NH; ++h) dA[h] = A_next[h] - v->A_prev[h];
    seg_cur = -1;
    }
#ifndef SKB_HAVE_MVE
    (void)h_pad;                        /* the scalar branch goes up to h_last */
#endif

    /* LOCAL COPIES OF THE GLOBALS. h_last/h_pad/seg_cur became file-scope
     * statics for the sake of the span slicing, and that immediately cost 16%
     * of the skeleton: the compiler has to re-read the global on every
     * iteration, because it cannot prove that skb_wtab does not touch it. A
     * copy into a local variable puts them back into registers (measured in
     * QEMU: bod 136 k -> 96 k). */
    const int hl = h_last, hp = h_pad;
    int seg = seg_cur;
    (void)hl; (void)hp;

    float ag0 = f_k->amp  * f_k->gate;
    float ag1 = f_k1->amp * f_k1->gate;

    /* --- band response: recomputed every SKB_WSEG samples ------------------
     * Hoisting it out for the whole hop gave rel RMS 1.9e-2 against the
     * Python reference: f0 moves by fractions of a hertz inside a hop, but
     * the frequency of the h-th harmonic moves h times more, and on the top
     * harmonics that is dozens of nodes of the 25 Hz grid. So not "once per
     * hop" but in segments: the response is taken from f0 in the MIDDLE of
     * the segment, and the error falls proportionally to the length. The
     * curve The curve was taken on the host (rel RMS against the Python
     * reference, dsp/skeleton_b.py, a glissando test through all the seams):
     *
     *     SKB_WSEG   rel RMS     dB     share of work vs exact
     *         1      6.68e-06  -103.5      100%   (exact, as it was)
     *         2      6.82e-04   -63.3       63%
     *         4      1.51e-03   -56.4       44%   <- taken
     *         8      3.02e-03   -50.4       34%
     *        16      6.29e-03   -44.0       27%   (test threshold 5e-3 — misses)
     *        48      1.89e-02   -34.5       27%
     *
     * 4 was taken: -56.4 dB is 13 dB quieter than the A-path equivalence
     * contract already accepted in the project (-43.5 dB, dsp/skeleton_b.py),
     * and the work drops by more than half. Beyond that the return is small:
     * 8 saves another tenth but loses 6 dB. Setting 1 means going back to the
     * exact form without a single edit. */
    for (int i = i0; i < i1; ++i) {
        float fr = (float)i * (1.0f / (float)N6_HOP12);
        {   int s = i / SKB_WSEG;
            if (s != seg) { seg = s;
                SKB_T0();
                skb_wtab(f_k, f_k1, hl,
                         ((float)(s * SKB_WSEG) + 0.5f * (float)(SKB_WSEG - 1))
                             * (1.0f / (float)N6_HOP12), w_re, w_im);
                SKB_P(0); } }
        float ag = ag0 + (ag1 - ag0) * fr;
        float tb = f_k->tB + (f_k1->tB - f_k->tB) * fr;

        /* B-1: catch-up of the 48k increments up to m = 4n (n is the global
           12k index); the increments m belong to frame m/HOP48: "current-1"
           interpolates prev_f0..f_k, the current one f_k..f_k1.

           64-BIT DIVISION COSTS MONEY (4 Aug). m/HOP48 and m%HOP48 sat inside
           the loop over m — that is four uint64 divisions by a constant for
           every subband sample, and GCC expands each into a chain of umull/
           umlal. But m runs over no more than four consecutive values: one
           division per sample is enough, after that the remainder is
           incremented and carried at the frame boundary. frame_cur is
           likewise hoisted out of the loop — it does not depend on m. The
           arithmetic is the same, the result is bit-for-bit the same. */
        uint64_t n12 = v->n48;
        {   SKB_T0();
            uint64_t m_hi = 4 * n12;
            uint64_t m = (n12 == 0) ? 0 : (4 * (n12 - 1) + 1);
            uint64_t frame_of_m = m / N6_HOP48;
            uint32_t rem = (uint32_t)(m - frame_of_m * N6_HOP48);
            uint64_t frame_cur = n12 / N6_HOP12;
            for (; m <= m_hi; ++m, ++rem) {
                if (rem == (uint32_t)N6_HOP48) { rem = 0u; ++frame_of_m; }
                float fr48 = (float)rem * (1.0f / (float)N6_HOP48);
                float fA, fB;
                if (frame_of_m == frame_cur) { fA = f_k->f0; fB = f_k1->f0; }
                else { fA = (v->prev_f0 != 0.0f) ? v->prev_f0 : f_k->f0;
                       fB = f_k->f0; }
                float f48 = fA + (fB - fA) * fr48;
                /* It used to be llrint() — a libm call on every increment.
                 * The double precision is needed here (otherwise the coherent
                 * quantization error drags the phase off), but the rounding
                 * is done by adding 0.5 and a single VCVT instruction:
                 * f48 > 0 always, and the product is certainly less than
                 * 2^32. The call is gone. */
                v->phi_q += (uint32_t)((double)f48 * SKB_HZ_TO_Q + 0.5);
            }
            SKB_P(4); }

        float acc[N6_BANDS] = {0.0f, 0.0f, 0.0f, 0.0f};

        /* --- prologue: p_h = a_h*sin(h*phi), q_h = a_h*cos(h*phi) ----------
         * The prologue turned out to be MORE EXPENSIVE than the vector body
         * it feeds: 13 instructions per (harmonic, sample) pair against 9 for
         * the body (measured in QEMU on a Cortex-M55 model, fw/test/qemu).
         * The reason is that it was scalar, and scalar for a single reason:
         * the sin(h*phi) rotator is recurrent in h, and I did not want to
         * drag the dependency chain into the body.
         *
         * But the recurrence is easy to cut into four: if you start not from
         * one angle but from four (phi, 2phi, 3phi, 4phi) and spin them
         * TOGETHER with a step of 4phi, you get exactly one f32x4 vector and
         * the same sequence of values. The seeds are taken from the table by
         * the EXACT phase: phi lives in a uint32 where 2^32 == 2*pi, so
         * 2*phi, 3*phi, 4*phi are ordinary multiplication with modular
         * overflow, without any loss whatsoever. The chain became four times
         * shorter in the process (25 rotations per sample instead of 100),
         * that is, the accuracy is higher too.
         *
         * At the same time the amplitude is applied HERE and not in the body:
         * the body gets ready-made p_h and q_h and adds up with two vfma
         * instead of vmul+vmul+vadd+vfma. Mathematically
         * (w_re*sin + w_im*cos)*a is the same as w_re*(a*sin) + w_im*(a*cos);
         * only the rounding changes. The frame difference A_next-A_prev is
         * hoisted out of the per-sample loop into dA. */
        static float p_s[N6_NH] N6_DTCM;
        static float q_s[N6_NH] N6_DTCM;
        SKB_T0();
#ifdef SKB_HAVE_MVE
        {   uint32_t ph = v->phi_q;
            float sd[4] __attribute__((aligned(16))) = {
                skb_sin_q(ph), skb_sin_q(ph * 2u),
                skb_sin_q(ph * 3u), skb_sin_q(ph * 4u) };
            float cd[4] __attribute__((aligned(16))) = {
                skb_cos_q(ph), skb_cos_q(ph * 2u),
                skb_cos_q(ph * 3u), skb_cos_q(ph * 4u) };
            float32x4_t vs = vld1q_f32(sd), vc = vld1q_f32(cd);
            float s4 = sd[3], c4 = cd[3];       /* rotation step = 4*phi */
            for (int h = 0; h < hp; h += 4) {
                float32x4_t av = vfmaq_n_f32(vld1q_f32(&v->A_prev[h]),
                                             vld1q_f32(&dA[h]), fr);
                vst1q_f32(&p_s[h], vmulq_f32(av, vs));
                vst1q_f32(&q_s[h], vmulq_f32(av, vc));
                float32x4_t ns = vfmaq_n_f32(vmulq_n_f32(vs, c4), vc, s4);
                vc = vsubq_f32(vmulq_n_f32(vc, c4), vmulq_n_f32(vs, s4));
                vs = ns;
            }
        }
#else
        {   float sn1 = skb_sin_q(v->phi_q), cs1 = skb_cos_q(v->phi_q);
            float sn = sn1, cs = cs1;
            for (int h = 0; h < hl; ++h) {
                float a = v->A_prev[h] + dA[h] * fr;
                p_s[h] = a * sn;
                q_s[h] = a * cs;
                float sn_n = sn * cs1 + cs * sn1;   /* angle(sn,cs) += th1 */
                cs = cs * cs1 - sn * sn1;
                sn = sn_n;
            }
        }
#endif
        SKB_P(1);

#ifdef SKB_HAVE_MVE
        /* Two independent accumulators: a single vfma chain over accv would
         * be limited by its own latency, here re and im run in parallel. */
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        for (int h = 0; h < hp; ++h) {
            acc0 = vfmaq_n_f32(acc0, vld1q_f32(w_re[h]), p_s[h]);
            acc1 = vfmaq_n_f32(acc1, vld1q_f32(w_im[h]), q_s[h]);
        }
        vst1q_f32(acc, vaddq_f32(acc0, acc1));
        SKB_P(2);
#else
        for (int h = 0; h < hl; ++h)
            for (int b = 0; b < N6_BANDS; ++b)
                acc[b] += w_re[h][b] * p_s[h] + w_im[h][b] * q_s[h];
        SKB_P(2);
#endif
        for (int b = 0; b < N6_BANDS; ++b) {
            float u = n6_xs32_next_f32(&v->nseed[b]);
            sub[b][i] += acc[b] * ag + v->nw[b] * u * tb * ag;
        }
        v->n48 = n12 + 1;
    }

    seg_cur = seg;
    if (i1 >= N6_HOP12) {                       /* the last span of the hop */
        v->prev_f0 = f_k->f0;
        memcpy(v->A_prev, A_next, sizeof A_next);
    }
}

void n6_skb_render_hop(n6_skb_voice_t *v, const n6_frame_t *f_k,
                       const n6_frame_t *f_k1,
                       float sub[N6_BANDS][N6_HOP12])
{
    n6_skb_render_span(v, f_k, f_k1, sub, 0, N6_HOP12);
}
