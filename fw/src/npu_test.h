/* npu_test.h — test hooks of the NULL NPU stub (host only, npu_stub.c).
 * On the target (npu_neuralart.c) they do not exist and are not needed: only
 * the tests in fw/test call them. The point is to make P0 pipeline defects
 * catchable on the host:
 *
 *  n6_npu_test_latency(k)  — poll must return 0 exactly k times before DONE.
 *      Catches "poll called once instead of in a loop" (done=0 forever then).
 *  n6_npu_test_pattern(m)  — what to fill the residual with:
 *      N6_NPUT_ZERO   the normal zero (golden/CK4 do not change);
 *      N6_NPUT_NEGX   residual[b][v][t] = -x_cond[b][v][t] — the "null
 *          teacher": with the CORRECT [COUT][V][T] layout the mix of the
 *          skeleton with the residual must give EXACTLY zero at the output;
 *          if the voices are summed before the NPU, or the residual is
 *          indexed without V — it will not;
 *      N6_NPUT_VGAIN  residual[b][v][t] = 0.1*(v+1) * x_cond[b][v][t] —
 *          dependence on the voice INDEX: catches swapped slices.
 *  n6_npu_test_last_input() — what the NPU actually got (to check the layout).
 */
#ifndef N6_NPU_TEST_H
#define N6_NPU_TEST_H
#include "npu_iface.h"

enum { N6_NPUT_ZERO = 0, N6_NPUT_NEGX = 1, N6_NPUT_VGAIN = 2 };

void n6_npu_test_pattern(n6_npu_t *n, int mode);
void n6_npu_test_latency(n6_npu_t *n, int polls_before_done);
const float *n6_npu_test_last_input(const n6_npu_t *n);   /* [CIN][V][T] */
#endif
