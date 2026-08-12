/* n6_fir.h — the linear "zero layer" of the refiner: a causal 4->4 FIR bank
 * modulated by the macro axis t (D-16).
 *
 * WHY (measured 2 Aug, teacher_search/probe_linear.md and exam_delta*.md):
 * the trained network on its own converged to a linear solution — the optimal
 * FIR of 520 coefficients beat it for ALL six teacher candidates
 * (net−FIR from −0.94 to −3.73 dB). As soon as the linear part was given away
 * for free, by freezing a FIR as the zero layer (pred = FIR(x) + net(x)), the
 * network started scoring exactly on what the filter cannot do: the delta grew
 * +2.78 -> +7.09 -> **+9.18 dB** as the corpus grew (2 / 9 / 28 minutes).
 * Final breakdown: the FIR gives +8.77 dB, FIR+net +17.95.
 *
 * Hence the runtime design: the M55 computes the FIR, the NPU the remainder.
 *
 * CONTRACT
 *   y[b][n] = SUM_{b'} SUM_{k=0..K-1} (W0[b][b'][k] + t*W1[b][b'][k]) * x[b'][n-k]
 *   - x, y — the subbands of ONE voice, [N6_BANDS][T], 12 kHz;
 *   - t — the macro axis, piecewise constant over a hop (as in the corpus:
 *     np.repeat of the 250 Hz curve), so the effective coefficients are
 *     computed ONCE per hop;
 *   - the state is K-1 history samples per band, ITS OWN for each voice
 *     (clear on NoteOn in the same place where the NPU slices are cleared,
 *     §8.2/R-3);
 *   - the output is ADDED to the network residual before the voice mix.
 *
 * COST: 4*4*65 = 1040 MAC to recompute the coefficients + 4*4*65*48 = 49 920 MAC
 * per hop per voice; at V=2 that is ~100k MAC. The estimate "~3% of the budget
 * even in scalar" was NOT confirmed by measurement: the scalar version with the
 * backwards step cost 652 k cycles per hop (6.5 cycles/MAC, 20% of the budget)
 * — the analysis and the cure (weight reversal + MVE) are in the header of
 * n6_fir.c. w_eff is stored REVERSED in k — an implementation detail.
 */
#ifndef N6_FIR_H
#define N6_FIR_H
#include "n6_config.h"

#define N6_FIR_TAPS  65                    /* K; matches probe_linear --taps */
#define N6_FIR_HIST  (N6_FIR_TAPS - 1)

typedef struct {
    /* band history, CHRONOLOGICALLY: hist[b][HIST-1] is the newest */
    float hist[N6_BANDS][N6_FIR_HIST];
    float w_eff[N6_BANDS][N6_BANDS][N6_FIR_TAPS];   /* W0 + t*W1 for this hop */
    float t_cur;
    int   have_w;
} n6_fir_t;

/* The coefficients come from tools/export_fir.py (fw/n6_fir_coeffs.h).
 * Pointers, not copies: the tables are const and live in flash/ROM. */
typedef struct {
    const float *w0;                       /* [N6_BANDS][N6_BANDS][N6_FIR_TAPS] */
    const float *w1;
} n6_fir_coeffs_t;

void n6_fir_init(n6_fir_t *f);
/* Clear the history of one voice (NoteOn, §8.2). */
void n6_fir_reset(n6_fir_t *f);
/* One hop: x[N6_BANDS][T] -> y[N6_BANDS][T]; y may alias x. */
void n6_fir_hop(n6_fir_t *f, const n6_fir_coeffs_t *c, float t,
                const float x[N6_BANDS][N6_HOP12],
                float y[N6_BANDS][N6_HOP12], int T);
#endif
