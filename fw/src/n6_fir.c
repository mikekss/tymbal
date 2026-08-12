/* n6_fir.c — the linear zero layer of the refiner (see n6_fir.h).
 *
 * TWO BRANCHES, as in skeleton_b/pqmf: scalar is the reference (host, CK4
 * reference), MVE is the runtime; the branches are cross-checked by
 * make qemu-ck4 (threshold 1e-4) and the [fir] golden.
 *
 * COST, HISTORY (4 Aug, late evening). The first version kept the weights in
 * forward order and computed the dot product "backwards along the signal":
 * acc += w[k]*s[-k]. Measured on the board: 652 k cycles per hop with two
 * voices — 6.5 cycles per MAC, 20% of the budget (the "~3%" prediction was
 * off by a factor of three; the counter is the fir slot). A backwards step
 * through memory is not unrolled by the compiler and does not get along with
 * the prefetcher. The cure: the weights are REVERSED ONCE when w_eff is
 * recomputed (w_rev[j] = w[K-1-j]), and the dot product becomes
 * "forward×forward" over two contiguous arrays — this opens up MVE (64 taps
 * vectorized 4 f32 at a time, the 65th scalar) and makes the scalar version
 * unrollable. */
#include <string.h>
#include "n6_fir.h"

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
#include <arm_mve.h>
#define N6_FIR_HAVE_MVE 1
#endif

void n6_fir_init(n6_fir_t *f)
{
    memset(f, 0, sizeof *f);
    f->t_cur = -1.0f;                       /* deliberately != any valid t */
}

void n6_fir_reset(n6_fir_t *f)
{
    memset(f->hist, 0, sizeof f->hist);
}

/* Recomputing the effective coefficients: once per hop, if t has moved.
 * NOTE: w_eff is stored REVERSED in k (w_eff[..][j] = W[K-1-j]) — an
 * internal contract, it does not stick out. */
static void fir_update_w(n6_fir_t *f, const n6_fir_coeffs_t *c, float t)
{
    if (f->have_w && t == f->t_cur) return;
    for (int b = 0; b < N6_BANDS; ++b)
        for (int bp = 0; bp < N6_BANDS; ++bp) {
            const size_t off = ((size_t)b * N6_BANDS + bp) * N6_FIR_TAPS;
            const float *s0 = c->w0 + off, *s1 = c->w1 + off;
            float *d = f->w_eff[b][bp];
            for (int k = 0; k < N6_FIR_TAPS; ++k)
                d[N6_FIR_TAPS - 1 - k] = s0[k] + t * s1[k];
        }
    f->t_cur = t;
    f->have_w = 1;
}

void n6_fir_hop(n6_fir_t *f, const n6_fir_coeffs_t *c, float t,
                const float x[N6_BANDS][N6_HOP12],
                float y[N6_BANDS][N6_HOP12], int T)
{
    fir_update_w(f, c, t);

    /* Window [history | block]: a linear buffer instead of a ring — both dot
     * products run forward through contiguous memory. */
    float win[N6_BANDS][N6_FIR_HIST + N6_HOP12];
    for (int b = 0; b < N6_BANDS; ++b) {
        memcpy(win[b], f->hist[b], N6_FIR_HIST * sizeof(float));
        memcpy(win[b] + N6_FIR_HIST, x[b], (size_t)T * sizeof(float));
    }

    for (int n = 0; n < T; ++n) {
        /* output window n: s[0..K-1] = win[bp][n .. n+K-1], newest is s[K-1] */
        for (int b = 0; b < N6_BANDS; ++b) {
#ifdef N6_FIR_HAVE_MVE
            float32x4_t av = vdupq_n_f32(0.0f);
            float tail = 0.0f;
            for (int bp = 0; bp < N6_BANDS; ++bp) {
                const float *w = f->w_eff[b][bp];
                const float *s = &win[bp][n];
                for (int j = 0; j < N6_FIR_TAPS - 1; j += 4)
                    av = vfmaq_f32(av, vldrwq_f32(w + j), vldrwq_f32(s + j));
                tail += w[N6_FIR_TAPS - 1] * s[N6_FIR_TAPS - 1];
            }
            y[b][n] = tail + (((vgetq_lane_f32(av, 0) + vgetq_lane_f32(av, 1))
                             + (vgetq_lane_f32(av, 2) + vgetq_lane_f32(av, 3))));
#else
            float acc = 0.0f;
            for (int bp = 0; bp < N6_BANDS; ++bp) {
                const float *w = f->w_eff[b][bp];
                const float *s = &win[bp][n];
                for (int j = 0; j < N6_FIR_TAPS; ++j)
                    acc += w[j] * s[j];
            }
            y[b][n] = acc;
#endif
        }
    }

    /* new history — the window tail (y may alias x, so we take the window) */
    for (int b = 0; b < N6_BANDS; ++b)
        memcpy(f->hist[b], win[b] + T, N6_FIR_HIST * sizeof(float));
}
