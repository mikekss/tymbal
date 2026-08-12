/* pqmf_synth.c — polyphase PQMF synthesis 4x12k -> 48k.
 * Reference: dsp/pqmf_design.py::synthesize; y[4m+j] = 4 * sum_k sum_p
 * syn[k][4p+j] * sub[k][m-p]. Zero initial history == the left zero-pad of
 * numpy convolve => a match with golden FROM THE VERY FIRST sample.
 * Helium port: the 2 inner loops are a ready MVE candidate (4x f32). */
#include <string.h>
#include "n6_dsp.h"
#include "../pqmf_coeffs.h"

void n6_pqmf_synth_init(n6_pqmf_synth_t *s) { memset(s, 0, sizeof *s); }

/* MVE path: the history window is contiguous (a ring with duplication), at
 * startup the synthesis matrix is transposed into g_synT[j][k][p] with the
 * taps REVERSED — the window (old -> new by increasing address) is convolved
 * by a plain 32-float dot product. The sum over k/p is vector (the order of
 * the terms differs from the scalar reference — the reassociation is within
 * the test tolerance). */
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
#include <arm_mve.h>
#define PQMF_HAVE_MVE 1
static float g_synT[N6_BANDS][N6_BANDS][N6_PQMF_PHASES]
    __attribute__((aligned(16)));           /* [j][k][p from old to new] */
static int g_synT_init;
static void pqmf_synT_init(void)
{
    for (int j = 0; j < N6_BANDS; ++j)
        for (int k = 0; k < N6_BANDS; ++k)
            for (int p = 0; p < N6_PQMF_PHASES; ++p)
                g_synT[j][k][p] =
                    pqmf_synthesis[k][4 * (N6_PQMF_PHASES - 1 - p) + j];
    g_synT_init = 1;
}
#endif

void n6_pqmf_synth_hop(n6_pqmf_synth_t *s, const float sub[N6_BANDS][N6_HOP12],
                       float *y48, int hop12)
{
#ifdef PQMF_HAVE_MVE
    if (!g_synT_init) pqmf_synT_init();
#endif
    for (int m = 0; m < hop12; ++m) {
        /* push the new subband samples (pos points at the newest);
           writing into both copies of the ring — the window is always
           contiguous */
        s->pos = (s->pos + 1) & (N6_PQMF_PHASES - 1);
        for (int k = 0; k < N6_BANDS; ++k) {
            s->hist[k][s->pos] = sub[k][m];
            s->hist[k][s->pos + N6_PQMF_PHASES] = sub[k][m];
        }
#ifdef PQMF_HAVE_MVE
        int w0 = (s->pos + 1) & (N6_PQMF_PHASES - 1);  /* the oldest edge */
        for (int j = 0; j < N6_BANDS; ++j) {
            float32x4_t accv = vdupq_n_f32(0.0f);
            for (int k = 0; k < N6_BANDS; ++k) {
                const float *w = &s->hist[k][w0];      /* 32 in a row: old->new */
                const float *c = g_synT[j][k];
                for (int p = 0; p < N6_PQMF_PHASES; p += 4)
                    accv = vfmaq_f32(accv, vld1q_f32(w + p), vld1q_f32(c + p));
            }
            y48[4 * m + j] = 4.0f *
                (vgetq_lane_f32(accv, 0) + vgetq_lane_f32(accv, 1) +
                 vgetq_lane_f32(accv, 2) + vgetq_lane_f32(accv, 3));
        }
#else
        for (int j = 0; j < N6_BANDS; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N6_BANDS; ++k) {
                const float *hk = s->hist[k];
                int idx = s->pos;
                for (int p = 0; p < N6_PQMF_PHASES; ++p) {
                    acc += pqmf_synthesis[k][4 * p + j] * hk[idx];
                    idx = (idx - 1) & (N6_PQMF_PHASES - 1);
                }
            }
            y48[4 * m + j] = 4.0f * acc;
        }
#endif
    }
}
