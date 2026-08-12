#ifndef N6_PIPELINE_H
#define N6_PIPELINE_H
#include "n6_config.h"
#include "n6_dsp.h"
#include "voice.h"
#include "npu_iface.h"
#include "n6_fir.h"

/* Pipeline of depth 1 (§8.1): the NPU computes block i-1, the M55 renders i.
 *
 * Ping-pong layout (P0 fix, 2 Aug):
 *   sub[idx][v][b][t] — the block's skeleton PER VOICE (mixing only after
 *                       DONE: the net keeps its own state for each voice);
 *   xcond[idx][...]   — the assembled NPU input of the same block, flat
 *                       [CIN][V][T] with a RUNTIME stride V (npu_iface.h);
 *   idx = p->cur      — the block rendered last (at the start of a hop, i-1).
 * Two sets are needed so that the input handed to the NPU is not overwritten
 * by the render of the next block: on npu_miss the NPU still reads the old
 * buffer.
 */
#define N6_NPU_XCOND_MAX ((size_t)N6_NPU_CIN * N6_MAX_VOICES * N6_HOP12)

typedef struct {
    n6_params_t prm;
    n6_voicemgr_t vm;
    n6_skb_voice_t skv[N6_MAX_VOICES];
    n6_pqmf_synth_t pq;
    n6_wf_t wf;
    n6_npu_t *npu;
    n6_frame_t fr_cur[N6_MAX_VOICES], fr_next[N6_MAX_VOICES];
    /* frames the block in sub[idx] was RENDERED with — they condition the NPU */
    n6_frame_t fr_used[2][N6_MAX_VOICES];
    /* whether the voice was rendered in this block (fa OR fb non-zero is the
     * render criterion): it also decides whether to run the FIR; frame fa is
     * no good here, it is zero on the first block of the attack */
    uint8_t sub_live[2][N6_MAX_VOICES];
    float sub[2][N6_MAX_VOICES][N6_BANDS][N6_HOP12];
    float xcond[2][N6_NPU_XCOND_MAX];
    int   cur;                            /* index of the last one rendered */
    int   have_prev;
    /* D-16 conditioning macro axis (the D-8 antenna maps here) */
    float t_macro;
    /* wow/flutter macro curves of the frame (preset / D-8 antenna) */
    float vmac_cur, vmac_next, wdepth, hisslvl;
    /* output limiter (H5): peak envelope, instant attack /
     * exponential release (see pipeline.c) */
    float lim_env;
    /* FIR bank (D-17, the linear zeroth layer of the refiner): state per
     * voice. Coefficients are in fw/n6_fir_coeffs.h (production ones from
     * tools/export_fir.py; a zero stub until they are generated).
     * fir_c == NULL switches the stage off. */
    n6_fir_t fir[N6_MAX_VOICES];
    const n6_fir_coeffs_t *fir_c;
    uint8_t fir_flush[N6_MAX_VOICES];   /* blocks left to flush the tail */
    /* the macro axis t that xcond of block idx was assembled with: the FIR of
     * that block is modulated by the same one — the net and the filter must
     * see ONE t (D-16/D-17) */
    float t_used[2];
    /* telemetry §8.3 */
    uint32_t npu_miss, hops, npu_polls, npu_polls_max;
    /* D-24 (7 Aug): live A/B of the net. 1 = the net residual goes into the
     * mix (as before, byte for byte); 0 = the "skeleton + FIR" branch is
     * forced — exactly the one that already runs when the NPU did not make
     * it. The net KEEPS being computed: the state stays coherent, the cycles
     * are the same, switching back is instant and the comparison is honest. */
    uint8_t net_on;
} n6_pipe_t;

void n6_pipe_init(n6_pipe_t *p, const n6_params_t *prm);
/* One hop slot: MIDI events are already drained into vm; out48[hop48]. */
void n6_pipe_hop(n6_pipe_t *p, float *out48);
/* Swapping the FIR coefficients (tests, A/B comparisons): the history of all
 * voices is reset — history from foreign coefficients is meaningless. NULL
 * switches the stage off entirely (the chain as it was before D-17). */
/* Switch for the net residual (D-24). A click when switching on a sounding
 * note is expected: the residual disappears instantly, there is no smoothing.
 * Returns the new state. */
int  n6_pipe_set_net(n6_pipe_t *p, int on);
void n6_pipe_set_fir(n6_pipe_t *p, const n6_fir_coeffs_t *c);
#endif
