/* pipeline.c — the hop-slot schedule (§8.1), the order is canonical:
 * 1..6 see the guide; wow/flutter — only AFTER PQMF synthesis (F-1);
 * NoteOn zeroing of the NPU slices — after DONE (R-3). Host version: NPU stub.
 *
 * P0 FIX 2 Aug (three defects that the zero stub masked):
 *  1) voices are NO longer summed before the NPU — the network gets
 *     [CIN][V][T], the mix of skeleton with residual runs per voice AFTER
 *     DONE (npu_iface.h);
 *  2) n6_npu_poll is called IN A LOOP until DONE, not once: t_call 2866 us
 *     against ~350 us of render — a single poll gave npu_miss every hop;
 *  3) the NPU input lies in a ping-pong buffer and is not overwritten until
 *     DONE.
 */
#include <string.h>
#include <math.h>
#include "pipeline.h"
#include "../n6_fir_coeffs.h"   /* w0/w1 of FIR bank: production or zero stub */

/* Profiling of the hop stages (target only): DWT cycles through a hook in
 * main. Slots: 0 skeleton (together with the nested NPU pumping), 1 tail
 * polling of the NPU, 2 mix+PQMF, 3 wow/flutter.
 *
 * HOP TAIL (4 Aug, slots 4-6). The sum of the first four did not add up to
 * the full hop by ~317 k cycles, THE SAME in silence and while playing. It
 * matters that g_cyc in main.c measures exactly n6_pipe_hop: the MIDI drain
 * and the 24-in-32 conversion lie OUTSIDE the measurement. So those 317 k
 * were sitting here, between PROF(3) and the end of the function, and by size
 * there was exactly one candidate — n6_npu_swap_states, copying the 43.7 kB
 * state ring with cache maintenance for each of the 12 buffers. But "the size
 * looks about right" is a hypothesis, not a measurement, hence three separate
 * counters:
 *   [4] build_xcond — assembling the NPU input
 *   [5] zeroing states on NoteOn + ping/pong swap
 *   [6] limiter
 *
 * CLOSED. The hypothesis was confirmed: the swap gave 307 420 out of those
 * 317 k. It was then cured by the same route — a vector copy with 16-byte
 * loads brought it down to 92 080, and user-allocated IO (a shared buffer per
 * input/output pair, npu_neuralart.c) to zero: there is nothing left to copy.
 * What remains of slot [5] — 1..10 k and only when keys are pressed — is no
 * longer the swap but n6_npu_zero_voice: clearing the voice's state slice on
 * NoteOn, real and necessary work. In silence the tail as a whole is
 * 4676 + 236 + 3954, and the sum of all seven slots matches the hop to within
 * 200 cycles out of 2.83 M. The per-stage accounting is fully closed. */
/* [7] added 4 Aug (evening) together with the FIR stage: FIR + voice mix.
 * From this point slot [2] is PURE PQMF synthesis (the mix used to sit in it
 * too); the print in main.c is prof(...)+fir. The sum of the slots still
 * matches the hop. */
#ifdef N6_TARGET
extern uint32_t n6_cyc_now(void);
uint32_t n6_prof[8];
#define PROF_T0() uint32_t _pt = n6_cyc_now()
#define PROF(slot) do { uint32_t _n = n6_cyc_now(); \
                        n6_prof[slot] += _n - _pt; _pt = _n; } while (0)
#else
#define PROF_T0() (void)0
#define PROF(slot) (void)0
#endif

/* Diagnostics for peak=980 (agenda item (b)): the maxima of |skeleton|,
 * |network residual|, |FIR output| BEFORE the mix, in thousandths of full
 * scale, the maximum since the last reset (main clears it when printing the
 * heartbeat). It answers the question "who is pushing the output into the
 * limiter": the honest loudness of the skeleton or an inflated residual scale
 * (fw/n6_npu_scales.h). The cost is ~1.2 k f32 scans per hop, target only. */
#ifdef N6_TARGET
uint32_t n6_dbg_sk_pk, n6_dbg_rs_pk, n6_dbg_fir_pk;
static void dbg_pk(uint32_t *dst, const float *a, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; ++i) { float x = fabsf(a[i]); if (x > m) m = x; }
    uint32_t q = (uint32_t)(m * 1000.0f);
    if (q > *dst) *dst = q;
}
#define DBG_PK(d, a, n) dbg_pk((d), (a), (n))
#else
#define DBG_PK(d, a, n) (void)0
#endif

/* Output limiter (H5): feed-forward peak, instant attack,
 * release ~80 ms. 3 voices give a peak up to ~2.1 — hard clipping at the DAC
 * is unacceptable aesthetically (spec §1: silence/air, not punk distortion).
 * The 0.98 threshold leaves half a bit of headroom to 24-bit full scale. */
#define LIM_THR   0.98f
#define LIM_REL   0.99974f          /* exp(-1/(0.08*48000)) */
static void lim_hop(float *x, int n, float *env_st)
{
    float env = *env_st;
    /* protection against sticking: otherwise a single Inf/NaN in the chain
     * makes env permanent and mutes the output until reboot (a comparison
     * with NaN is always false) */
    if (!(env >= 0.0f && env < 1.0e6f)) env = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = (x[i] < 0.0f) ? -x[i] : x[i];
        env *= LIM_REL;
        if (ax > env) env = ax;             /* instant attack */
        if (env > LIM_THR) x[i] *= LIM_THR / env;
    }
    *env_st = env;
}

/* Assembling the NPU input for block idx: [CIN][V][T], stride T per voice,
 * V*T per channel. Conditioning uses the frame the block was rendered with
 * (piecewise constant over the hop: exactly the way the training corpus
 * repeats the 250 Hz curves). */
static void build_xcond(n6_pipe_t *p, int idx, int V, int T)
{
    float *x = p->xcond[idx];
    const size_t vt = (size_t)V * T;
    for (int v = 0; v < V; ++v) {
        const n6_frame_t *f = &p->fr_used[idx][v];
        for (int b = 0; b < N6_BANDS; ++b) {
            float *dst = x + (size_t)b * vt + (size_t)v * T;
            const float *src = p->sub[idx][v][b];
            for (int t = 0; t < T; ++t) dst[t] = src[t];
        }
        float *c4 = x + (size_t)4 * vt + (size_t)v * T;
        float *c5 = x + (size_t)5 * vt + (size_t)v * T;
        float *c6 = x + (size_t)6 * vt + (size_t)v * T;
        float *c7 = x + (size_t)7 * vt + (size_t)v * T;
        const float ag = f->amp * f->gate;
        for (int t = 0; t < T; ++t) {
            c4[t] = ag; c5[t] = f->tA; c6[t] = f->tB; c7[t] = p->t_macro;
        }
    }
}

void n6_pipe_init(n6_pipe_t *p, const n6_params_t *prm)
{
    memset(p, 0, sizeof *p);
    p->prm = *prm;
    if (p->prm.n_voices > N6_MAX_VOICES) p->prm.n_voices = N6_MAX_VOICES;
    if (p->prm.n_voices < 1)             p->prm.n_voices = 1;
    n6_vm_init(&p->vm, &p->prm);
    for (int i = 0; i < p->prm.n_voices; ++i) {
        n6_skb_init(&p->skv[i], 0xC0FFEEu + (uint32_t)i);
        /* D-23: noise tilt across the subbands, dB per band. 0 -> ones, that
         * is, the previous white noise at one level; the references do not
         * move. */
        /* the tilt is folded INTO THE NOISE COEFFICIENT ITSELF: the hot loop
         * keeps exactly as many multiplications as it had before D-23 —
         * the stage is free in cycles, it costs 16 bytes per voice. */
        for (int b = 0; b < N6_BANDS; ++b)
            p->skv[i].nw[b] = N6_SKB_NOISE_B
                * ((prm->noise_tilt_db == 0.0f) ? 1.0f
                   : powf(10.0f, -prm->noise_tilt_db * (float)b / 20.0f));
        p->skv[i].bloom_k = p->prm.bloom_k;        /* D-19; 0 = as before */
    }
    n6_pqmf_synth_init(&p->pq);
    n6_wf_init(&p->wf, 0xF1A77E12u, 0x8155CAFEu, 115200u /* host golden */);
    p->npu = n6_npu_create(&p->prm);
    /* FIR stage D-17. With a production n6_fir_coeffs.h it enables itself;
     * the zero stub TURNS IT OFF (fir_c = NULL): there is no point burning
     * ~0.2 M cycles multiplying zeros, and the chain with the stage off is
     * identical to the one with zeros (verified by [fir chain]). Flag for the
     * banner: p->fir_c != NULL. */
    for (int i = 0; i < N6_MAX_VOICES; ++i) n6_fir_init(&p->fir[i]);
    p->net_on = 1u;                      /* D-24: as it was */
    p->fir_c = &n6_fir_coeffs;
#ifdef N6_FIR_COEFFS_PLACEHOLDER
    p->fir_c = NULL;
#endif
    p->vmac_cur = p->vmac_next = 1.0f;
    p->wdepth = 0.002f; p->hisslvl = 0.02f;
    p->t_macro = 0.0f;                       /* D-16: axis at zero = dry */
    p->cur = 0;
}

void n6_pipe_hop(n6_pipe_t *p, float *out48)
{
    const int V = p->prm.n_voices;
    const int T = p->prm.hop48 / 4;
    const int prev = p->cur;                 /* block i-1 (rendered earlier) */
    const int now  = p->cur ^ 1;             /* we render block i here */

    PROF_T0();
    /* 2) start the NPU on block i-1 (the input was assembled during the last
     *    hop and has not been touched since — ping-pong; voices SEPARATELY,
     *    see npu_iface.h) */
    if (p->have_prev)
        n6_npu_submit(p->npu, p->xcond[prev]);

    /* 3) 250 Hz voice tick and render of the skeleton of block i — into the
     *    voice's OWN buffer */
    memcpy(p->fr_cur, p->fr_next, sizeof p->fr_cur);
    n6_vm_tick(&p->vm, p->fr_next);
    /* only the runtime V slices: with V=2 a full memset would clear 6 kB
       instead of 1.5 — on AXISRAM2 that is noticeable cycles in EVERY hop */
    memset(p->sub[now], 0, (size_t)V * sizeof p->sub[now][0]);
    /* The render goes in spans, with NPU pumping between them: see
       n6_config.h, N6_SKB_SPAN. The order of the arithmetic does not change
       by a single bit (cross-checked by CK4), only the distribution in time
       changes.

       HOW MANY POLLS PER SPAN IS ADAPTIVE (4 Aug, second attempt). The first
       version put a fixed 8 there, and the render managed to finish within
       the first sixth of the NPU stage: there is still little idle time there
       (in the early blocks the M55 is itself busy with concatenations), so
       three quarters of the work was hidden and 115 k stuck out. The goal is
       to smear the spans across the WHOLE stage, so the step equals "the
       total number of polls divided by the number of spans". The total is
       known from the previous hop and is stable (9.7-13 k); there can be no
       fixed constant here — it depends on the chord. */
    int done = !p->have_prev;
    uint32_t k = 0;
    int nv = 0;
    for (int v = 0; v < V; ++v)
        if (p->fr_cur[v].amp * p->fr_cur[v].gate != 0.0f
         || p->fr_next[v].amp * p->fr_next[v].gate != 0.0f) ++nv;
    uint32_t pump = N6_NPU_PUMP;
    if (nv > 0 && p->npu_polls > 0) {
        pump = p->npu_polls / (uint32_t)((N6_HOP12 / N6_SKB_SPAN) * nv);
        if (pump < 1u)    pump = 1u;
        if (pump > 4096u) pump = 4096u;
    }
    for (int v = 0; v < V; ++v) {
        const n6_frame_t *fa = &p->fr_cur[v], *fb = &p->fr_next[v];
        p->fr_used[now][v] = *fa;
        /* silent voice: both render terms are multiplied by ag==0 — the
           contribution is strictly zero, skipping it does not change the
           output (the phase of a silent voice consults no one) */
        if (fa->amp * fa->gate == 0.0f && fb->amp * fb->gate == 0.0f) {
            p->sub_live[now][v] = 0;
            continue;
        }
        p->sub_live[now][v] = 1;
        for (int i0 = 0; i0 < N6_HOP12; i0 += N6_SKB_SPAN) {
            for (uint32_t q = 0; q < pump && !done; ++q) {
                ++k; done = n6_npu_poll(p->npu);
            }
            n6_skb_render_span(&p->skv[v], fa, fb, p->sub[now][v],
                               i0, i0 + N6_SKB_SPAN);
        }
    }
    PROF(0);

    /* 4) finish pumping the NPU UNTIL DONE (§8.1 step 4). The stub answers
     *    instantly; on the target the bulk of the slot goes here; the §8.3
     *    degradation is residual=0 for this block, and the instrument sounds
     *    as a bare skeleton. */
    if (p->have_prev) {
        while (!done && k < N6_NPU_POLL_MAX) { ++k; done = n6_npu_poll(p->npu); }
        p->npu_polls = k;                    /* how many polls during the hop */
        if (k > p->npu_polls_max) p->npu_polls_max = k;
        if (!done) p->npu_miss++;
    }
    PROF(1);

    /* 5) mix of block i-1 PER VOICE: the voice's skeleton + FIR bank (D-17, a
     *    linear zero layer) + the network residual -> synthesis ->
     *    wow/flutter -> limiter.
     *
     *    The FIR is computed HERE and not during the render: block prev has
     *    not changed since the render (ping-pong), the t axis is the same one
     *    the NPU of this block was conditioned with (t_used[prev]), and the
     *    FIR state is per voice and advances exactly once per block, in
     *    chronological block order. The order of addition per sample:
     *    (skeleton + residual) + FIR — it must not be changed, the accuracy
     *    of the [fir chain] test relies on it.
     *
     *    A silent voice with zero FIR history is skipped EXACTLY, not
     *    approximately: after NoteOff the tail is flushed out by ceil(HIST/T)
     *    blocks of zero input, after which the history is strict zeros and
     *    the contribution is strictly zero. In silence the stage costs
     *    nothing. */
    if (p->have_prev) {
        const float *res = (done && p->net_on) ? n6_npu_residual(p->npu) : NULL;
        const size_t vt = (size_t)V * T;
        const int flushN = (N6_FIR_HIST + T - 1) / T;
        float mix[N6_BANDS][N6_HOP12];
        float fo[N6_BANDS][N6_HOP12];            /* FIR output of one voice */
        for (int b = 0; b < N6_BANDS; ++b)
            for (int t = 0; t < T; ++t) mix[b][t] = 0.0f;
        for (int v = 0; v < V; ++v) {
            int use_fir = 0;
            if (p->fir_c) {
                /* the flag comes from the render itself (sub_live), NOT from
                 * frame fa: on the first block of the attack fa is still
                 * zero, but the block already sounds */
                if (p->sub_live[prev][v]) {
                    p->fir_flush[v] = (uint8_t)flushN; use_fir = 1;
                } else if (p->fir_flush[v] > 0u) {
                    p->fir_flush[v]--;             use_fir = 1;
                }
            }
            if (use_fir) {
                n6_fir_hop(&p->fir[v], p->fir_c, p->t_used[prev],
                           (const float (*)[N6_HOP12])p->sub[prev][v], fo, T);
                for (int b = 0; b < N6_BANDS; ++b)
                    DBG_PK(&n6_dbg_fir_pk, fo[b], T);
            }
            for (int b = 0; b < N6_BANDS; ++b) {
                const float *sk = p->sub[prev][v][b];
                DBG_PK(&n6_dbg_sk_pk, sk, T);
                if (res && use_fir) {
                    const float *rs = res + (size_t)b * vt + (size_t)v * T;
                    for (int t = 0; t < T; ++t) mix[b][t] += sk[t] + rs[t] + fo[b][t];
                } else if (res) {
                    const float *rs = res + (size_t)b * vt + (size_t)v * T;
                    for (int t = 0; t < T; ++t) mix[b][t] += sk[t] + rs[t];
                } else if (use_fir) {
                    for (int t = 0; t < T; ++t) mix[b][t] += sk[t] + fo[b][t];
                } else {
                    for (int t = 0; t < T; ++t) mix[b][t] += sk[t];
                }
            }
        }
        if (res) DBG_PK(&n6_dbg_rs_pk, res, (int)((size_t)N6_NPU_COUT * vt));
        PROF(7);
        n6_pqmf_synth_hop(&p->pq, (const float (*)[N6_HOP12])mix, out48, T);
        PROF(2);
        n6_wf_hop(&p->wf, out48, p->prm.hop48,
                  p->vmac_cur, p->vmac_next, p->wdepth, p->wdepth,
                  p->hisslvl, p->hisslvl);
        PROF(3);
        /* Master level BEFORE the limiter (D-22): at 1.0 we do not touch a
         * single byte, so golden/CK4 do not move. The point is to give the
         * limiter back its role as a safety net: at unity it already fires at
         * vel≈85 on a single note and at vel≈60 on a chord, that is,
         * practically always. */
        if (p->prm.out_gain != 1.0f) {
            const float g = p->prm.out_gain;
            for (int i = 0; i < p->prm.hop48; ++i) out48[i] *= g;
        }
        lim_hop(out48, p->prm.hop48, &p->lim_env);
    } else {
        memset(out48, 0, sizeof(float) * (size_t)p->prm.hop48);
    }
    PROF(6);
    p->vmac_cur = p->vmac_next;

    /* 6) assemble the NPU input for block i (buffer now — we hand it over on
     *    the next hop and do not touch it until DONE) */
    build_xcond(p, now, V, T);
    p->t_used[now] = p->t_macro;          /* the same t modulates the block's FIR */
    PROF(4);

    /* 7) NoteOn slices (AFTER DONE, R-3) + ping/pong swap of the states.
     *    The voice's FIR history is erased right here: block prev of the old
     *    note has already been mixed in step 5, block now of the new note
     *    will be filtered on the next hop — from a clean state, in sync with
     *    the NPU slices. */
    for (int v = 0; v < V; ++v)
        if (p->vm.v[v].retrig) { n6_npu_zero_voice(p->npu, v);
                                 n6_fir_reset(&p->fir[v]);
                                 p->fir_flush[v] = 0;
                                 p->vm.v[v].retrig = 0; }
    n6_npu_swap_states(p->npu);
    PROF(5);

    /* block i becomes i-1 */
    p->cur = now;
    p->have_prev = 1;
    p->hops++;
}

int n6_pipe_set_net(n6_pipe_t *p, int on)
{
    p->net_on = on ? 1u : 0u;
    return (int)p->net_on;
}

void n6_pipe_set_fir(n6_pipe_t *p, const n6_fir_coeffs_t *c)
{
    p->fir_c = c;
    for (int i = 0; i < N6_MAX_VOICES; ++i) {
        n6_fir_init(&p->fir[i]);
        p->fir_flush[i] = 0;
    }
}
