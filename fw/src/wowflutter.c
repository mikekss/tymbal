/* wowflutter.c — wow and flutter + hiss (W-1..W-5). A transcript of
 * dsp/wowflutter.py::process_cstyle; hop-oriented API.
 * The parameters are stubs until the §4.2 characterization session — as in
 * Python.
 *
 * TARGET SEMANTICS (the n6_dsp.h plan, 1 Aug): the hot path is f32; the sines
 * of the modulator peaks are incremental rotators (the step constants are
 * computed in double at init, |z| is renormalized once per hop). double is
 * left for: lag (a long delay accumulator, f32 loses precision when lag >> 1)
 * and the wnorm init (one-off; on the target replace it with a firmware
 * constant, §4.2). */
#include <math.h>
#include <string.h>
#include "n6_dsp.h"
#include "xorshift.h"

#define WF_L0        24
#define WF_V_EPS     1e-3f
#define WF_FADE_N    ((int)(30.0e-3 * N6_FS48))
#define WF_POLE      0.995f
#define WF_NOISE_MIX 0.35f

static const double wf_peaks[N6_WF_NPEAKS][3] = {   /* Hz, amp, phase (§4.2b) */
    {4.2, 1.0, 0.3}, {12.7, 0.45, 1.7}, {33.0, 0.2, 4.1} };
static const float wf_biq[N6_WF_NBIQ][5] = {        /* b0 b1 b2 a1 a2 (§4.2a) */
    {0.05f, 0.0f, 0.0f, -0.95f, 0.0f}, {0.30f, 0.0f, 0.0f, -0.70f, 0.0f} };

void n6_wf_init(n6_wf_t *w, uint32_t seed_noise, uint32_t seed_hiss,
                uint32_t wnorm_len)
{
    memset(w, 0, sizeof *w);
    w->gain = 1.0f;
    w->seed_noise = n6_xs32_seed(seed_noise);
    w->seed_hiss  = n6_xs32_seed(seed_hiss);
    for (int p = 0; p < N6_WF_NPEAKS; ++p) {
        double wp = 2.0 * M_PI * wf_peaks[p][0] / (double)N6_FS48;
        w->stp_c[p] = (float)cos(wp);
        w->stp_s[p] = (float)sin(wp);
        w->rot_c[p] = (float)cos(wf_peaks[p][2]);   /* phase @ n=0 */
        w->rot_s[p] = (float)sin(wf_peaks[p][2]);
        w->amp_p[p] = (float)wf_peaks[p][1];
    }
    /* background normalization — as in the reference: RMS of the one-pole
     * noise over wnorm_len.
     * TARGET: replace with a firmware constant (an offline run of this same
     * code). */
    uint32_t s = w->seed_noise;
    double acc = 0.0, ss = 0.0;
    for (uint32_t i = 0; i < wnorm_len; ++i) {
        double u = n6_xs32_next(&s);
        acc = (double)WF_POLE * acc + (1.0 - (double)WF_POLE) * u;
        ss += acc * acc;
    }
    w->wnorm = (float)(1.0 / (sqrt(ss / (double)wnorm_len) + 1e-12));
}

void n6_wf_hop(n6_wf_t *w, float *x48, int hop48,
               float vmac0, float vmac1, float dep0, float dep1,
               float hl0, float hl1)
{
    /* rotator renormalization: |z| drifts ~1e-7/step — once per hop is enough */
    for (int p = 0; p < N6_WF_NPEAKS; ++p) {
        float g = 1.5f - 0.5f * (w->rot_c[p] * w->rot_c[p]
                                 + w->rot_s[p] * w->rot_s[p]);
        w->rot_c[p] *= g; w->rot_s[p] *= g;
    }
    float inv_hop  = 1.0f / (float)hop48;
    float inv_fade = 1.0f / (float)WF_FADE_N;
    for (int i = 0; i < hop48; ++i) {
        float fr    = (float)i * inv_hop;
        float v_mac = vmac0 + (vmac1 - vmac0) * fr;
        float dep   = dep0 + (dep1 - dep0) * fr;
        float hlv   = hl0 + (hl1 - hl0) * fr;
        uint64_t nn = w->n48;

        /* hiss: xorshift -> biquad cascade -> into the line input (W-4) */
        float hz = n6_xs32_next_f32(&w->seed_hiss);
        for (int b = 0; b < N6_WF_NBIQ; ++b) {
            float *st = w->bq[b];
            float yo = wf_biq[b][0]*hz + wf_biq[b][1]*st[0] + wf_biq[b][2]*st[1]
                       - wf_biq[b][3]*st[2] - wf_biq[b][4]*st[3];
            st[1] = st[0]; st[0] = hz; st[3] = st[2]; st[2] = yo;
            hz = yo;
        }
        float s = x48[i] + hlv * hz;
        w->ring[nn % N6_WF_RING] = s;

        /* modulator (W-2): peak rotators + one-pole background */
        float u = n6_xs32_next_f32(&w->seed_noise);
        w->onepole_acc = WF_POLE * w->onepole_acc + (1.0f - WF_POLE) * u;
        float m = WF_NOISE_MIX * w->onepole_acc * w->wnorm;
        for (int p = 0; p < N6_WF_NPEAKS; ++p) {
            m += w->amp_p[p] * w->rot_s[p];
            float sn_n = w->rot_s[p] * w->stp_c[p] + w->rot_c[p] * w->stp_s[p];
            w->rot_c[p] = w->rot_c[p] * w->stp_c[p] - w->rot_s[p] * w->stp_s[p];
            w->rot_s[p] = sn_n;
        }
        float v = v_mac * (1.0f + dep * m);

        /* tape-stop policy (W-3, F-2) */
        if (v_mac < WF_V_EPS) {
            w->stopped = 1;
            w->gain -= inv_fade; if (w->gain < 0.0f) w->gain = 0.0f;
            if (w->gain == 0.0f) w->lag = 0.0;
        } else {
            if (w->stopped && w->gain == 0.0f) { w->lag = 0.0; w->stopped = 0; }
            w->gain += inv_fade; if (w->gain > 1.0f) w->gain = 1.0f;
            w->lag += (double)(1.0f - v);
            double hi = (double)(N6_WF_RING - WF_L0 - 4);
            if (w->lag > hi) w->lag = hi;
            else if (w->lag < -(double)(WF_L0 - 4)) w->lag = -(double)(WF_L0 - 4);
        }
        double r = (double)nn - WF_L0 - w->lag;
        if (r < 0.0) {
            x48[i] = 0.0f;
        } else {
            uint64_t i0 = (uint64_t)r;
            float frc = (float)(r - (double)i0);
            /* i0+1 <= nn always, except at the edge; ring valid to depth RING */
            float a = w->ring[i0 % N6_WF_RING];
            float b = (i0 + 1 <= nn) ? w->ring[(i0 + 1) % N6_WF_RING] : a;
            x48[i] = (a + (b - a) * frc) * w->gain;
        }
        w->n48++;
    }
}
