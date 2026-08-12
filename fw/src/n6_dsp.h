/* n6_dsp.h — API of the M55 DSP building blocks. Every block is cross-checked
 * against the Python reference (the references are in the dsp directory) on
 * golden vectors (fw/test): contracts C-1..C-5, B-1..B-5,
 * W-1..W-5 are in the headers of the corresponding .py files.
 *
 * PRECISION (the plan from the v1 header is DONE, 1 Aug): hot paths are f32
 * (M55 FPU); the skeleton master phase is a uint32 accumulator (2^32 == 2π,
 * C-3); the sines of the harmonics and of the modulator peaks are incremental
 * rotators. double is kept only in spots where the accumulators have a large
 * dynamic range (rare operations, cheap even in soft-float): the Hz->phase
 * increment conversion, the wow and flutter lag, the one-off wnorm init (on
 * the target it is a firmware constant, §4.2). The cross-check against the
 * float64 golden is by relative RMS, thresholds are in host_test; the golden
 * did not have to be regenerated (measurements in fw/README). */
#ifndef N6_DSP_H
#define N6_DSP_H
#include "n6_config.h"

/* ---- PQMF synthesis: sub[4][hop12] -> y48[hop48], 32-tap polyphase (§2.1) */
typedef struct {
    /* ring with duplication: hist[b][i] == hist[b][i+32] — any window of
     * 32 samples is contiguous in memory (MVE path); the scalar path
     * reads only [0..31] as before */
    float hist[N6_BANDS][2 * N6_PQMF_PHASES];
    int   pos;                              /* ring index */
} n6_pqmf_synth_t;

void n6_pqmf_synth_init(n6_pqmf_synth_t *s);
void n6_pqmf_synth_hop(n6_pqmf_synth_t *s, const float sub[N6_BANDS][N6_HOP12],
                       float *y48 /*[hop48]*/, int hop12);

/* ---- Skeleton, variant B: render a voice into subbands (B-1..B-5) -------- */
typedef struct {
    uint32_t phi_q;                 /* master phase: 2^32 == 2π (B-1/C-3) */
    uint64_t n48;                   /* global 12k sample index */
    float    prev_f0;               /* f0 of the previous frame (catch-up B-1) */
    uint32_t nseed[N6_BANDS];       /* per-band noise (B-3) */
    float    nw[N6_BANDS];          /* noise weight per subband (D-23) */
    float    A_prev[N6_NH];         /* harmonic amplitudes of frame k (C-1/C-2) */
    int      have_prev;
    float    bloom_k;               /* D-19: 0 = v0 decoder, unchanged */
} n6_skb_voice_t;

void n6_skb_init(n6_skb_voice_t *v, uint32_t seed);
/* 1 = skeleton built with MVE/Helium, 0 = scalar branch (~14x slower). Print
   it in the firmware banner: silently losing MVE costs the whole hop budget. */
int  n6_skb_have_mve(void);
/* Render of one hop: frames f_k (current) and f_k1 (next), ADD into sub. */
void n6_skb_render_hop(n6_skb_voice_t *v, const n6_frame_t *f_k,
                       const n6_frame_t *f_k1,
                       float sub[N6_BANDS][N6_HOP12]);
/* The same render over a span of samples [i0, i1) — so that the caller can
 * interleave it with NPU pumping (see the header of skeleton_b.c and §8.1).
 * Frame preparation is done on the span with i0 == 0, state finalisation on
 * the last one.
 * REQUIREMENT ON THE CALLER: a voice is walked to the end before the next one
 * is started — frame preparation lives in shared statics, not in the voice
 * state (for the sake of DTCM and locality). */
void n6_skb_render_span(n6_skb_voice_t *v, const n6_frame_t *f_k,
                        const n6_frame_t *f_k1,
                        float sub[N6_BANDS][N6_HOP12], int i0, int i1);

/* ---- Wow/flutter + hiss (W-1..W-5), full band, after synthesis ----------- */
#define N6_WF_RING   48000          /* 1 s @48k (W-1, §2.3) */
#define N6_WF_NPEAKS 3
#define N6_WF_NBIQ   2

typedef struct {
    float  ring[N6_WF_RING];
    uint64_t n48;                   /* global write index */
    double lag;                     /* double: long delay accumulator */
    float  gain;
    int    stopped;
    uint32_t seed_noise, seed_hiss;
    float  onepole_acc;
    float  bq[N6_WF_NBIQ][4];       /* x1,x2,y1,y2 */
    float  wnorm;                   /* background normalisation (n6_wf_init) */
    /* modulator peak rotators (W-2): z_p = (c,s), step w_p — constants */
    float  rot_c[N6_WF_NPEAKS], rot_s[N6_WF_NPEAKS];
    float  stp_c[N6_WF_NPEAKS], stp_s[N6_WF_NPEAKS];
    float  amp_p[N6_WF_NPEAKS];
} n6_wf_t;

/* wnorm_len: the length over which the background is normalised (== the length
 * of the golden run for the host test; on the target it is a constant from the
 * characterisation, §4.2). */
void n6_wf_init(n6_wf_t *w, uint32_t seed_noise, uint32_t seed_hiss,
                uint32_t wnorm_len);
/* One hop: input x48[hop48] (after PQMF synthesis), frame controls (k, k+1):
 * v_macro, depth, hiss_lvl; output through x48 (in-place). */
void n6_wf_hop(n6_wf_t *w, float *x48, int hop48,
               float vmac0, float vmac1, float dep0, float dep1,
               float hl0, float hl1);

#endif
