/* npu_stub.c — the host-side NPU stub.
 *
 * Normal mode: residual = 0 (the graceful principle of spec §2 — without the
 * NPU the instrument sounds like the skeleton). The golden vectors and CK4 do
 * not change in this mode.
 *
 * The test modes (npu_test.h) exist so that LAYOUT and pipeline SCHEDULING
 * defects fail on `make test` instead of surfacing in M2 on the board: a null
 * stub masks both (residual=0 is correct under any indexing error, done=1
 * immediately — under any pumping error).
 */
#include <stdlib.h>
#include <string.h>
#include "npu_iface.h"
#include "npu_test.h"

struct n6_npu {
    int    V, T;
    float *res;                 /* [COUT][V][T] */
    float *xin;                 /* copy of the last input [CIN][V][T] */
    int    done;
    int    lat, lat_left;       /* test: how many polls until DONE */
    int    pattern;
};

n6_npu_t *n6_npu_create(const n6_params_t *prm)
{
    n6_npu_t *n = calloc(1, sizeof *n);
    if (!n) return NULL;
    n->V = prm->n_voices;
    n->T = prm->hop48 / 4;
    n->res = calloc((size_t)N6_NPU_COUT * n->V * n->T, sizeof(float));
    n->xin = calloc((size_t)N6_NPU_CIN  * n->V * n->T, sizeof(float));
    return n;
}

void n6_npu_submit(n6_npu_t *n, const float *x)
{
    const size_t nin  = (size_t)N6_NPU_CIN  * n->V * n->T;
    const size_t vt   = (size_t)n->V * n->T;

    /* THE NORMAL PATH (on the board too — the stub is still the production
     * one there): NOT A SINGLE cycle. res is already zero from calloc and is
     * not written in this mode, the input is not copied. Copying the input
     * and filling res happen only under the test hooks. */
    if (n->pattern == N6_NPUT_ZERO) {
        n->lat_left = n->lat;
        n->done = (n->lat_left == 0);
        return;
    }
    memcpy(n->xin, x, nin * sizeof(float));

    switch (n->pattern) {
    case N6_NPUT_NEGX:                       /* the "null teacher" */
        for (int b = 0; b < N6_NPU_COUT; ++b)
            for (size_t i = 0; i < vt; ++i)
                n->res[(size_t)b * vt + i] = -x[(size_t)b * vt + i];
        break;
    case N6_NPUT_VGAIN:                      /* dependence on the voice index */
        for (int b = 0; b < N6_NPU_COUT; ++b)
            for (int v = 0; v < n->V; ++v)
                for (int t = 0; t < n->T; ++t) {
                    size_t i = ((size_t)b * n->V + v) * n->T + t;
                    n->res[i] = 0.1f * (float)(v + 1) * x[i];
                }
        break;
    default:
        memset(n->res, 0, (size_t)N6_NPU_COUT * vt * sizeof(float));
        break;
    }
    n->lat_left = n->lat;
    n->done = (n->lat_left == 0);
}

int n6_npu_poll(n6_npu_t *n)
{
    if (!n->done && n->lat_left > 0 && --n->lat_left == 0)
        n->done = 1;
    return n->done;
}

const float *n6_npu_residual(const n6_npu_t *n) { return n->res; }

void n6_npu_zero_voice(n6_npu_t *n, int v)
{
    /* the stub has no states; the signature is kept for the §8.2 contract */
    (void)n; (void)v;
}

void n6_npu_swap_states(n6_npu_t *n) { (void)n; }

/* ------------------------------------------------------------ test hooks */
void n6_npu_test_pattern(n6_npu_t *n, int mode) { n->pattern = mode; }
void n6_npu_test_latency(n6_npu_t *n, int polls) { n->lat = polls; }
const float *n6_npu_test_last_input(const n6_npu_t *n) { return n->xin; }
