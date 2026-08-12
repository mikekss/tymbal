/* npu_iface.h — the NPU refiner abstraction. The graph shape was FROZEN on
 * 2 Aug (D-2/D-9/D-11): C=88, V=2, T=48, L=12, wiring (b), batch-as-height.
 *
 * BUFFER CONTRACT (the ONNX canon, export_m0_d31.py: [1, c, N, T], N == V):
 *   input  x_cond[N6_NPU_CIN][V][T]  — CHANNEL-major, the voice in "height";
 *          channels 0..3 — the skeleton subbands of a SINGLE voice,
 *          4 — amp*gate, 5 — tA, 6 — tB, 7 — the macro axis t (D-16, shared);
 *   output residual[N6_NPU_COUT][V][T] — the same layout;
 *   voice stride = T, channel stride = V*T (V is the RUNTIME prm.n_voices,
 *   not N6_MAX_VOICES: the graph is compiled for one specific V).
 *
 * IMPORTANT (the source of a P0 before 2 Aug): the voices MUST NOT be summed
 * before the NPU — the network keeps its own state per voice. The mix comes
 * only after DONE, from residual[b][v][t] and the skeleton of THE SAME voice.
 *
 * Implementations: npu_stub.c (host: zero or a test pattern, see npu_test.h)
 * and npu_neuralart.c (target: LL_ATON from the generate template, §5.2/§8.2
 * — ping-pong of the state pointers, RunEpochBlock pumping, EC).
 */
#ifndef N6_NPU_IFACE_H
#define N6_NPU_IFACE_H
#include "n6_config.h"

typedef struct n6_npu n6_npu_t;

n6_npu_t *n6_npu_create(const n6_params_t *prm);

/* Start inference on block i-1. The x_cond buffer belongs to the NPU until
 * DONE: the caller must not touch it (in the pipeline — ping-pong by parity). */
void n6_npu_submit(n6_npu_t *n, const float *x_cond /*[CIN][V][T]*/);

/* Cooperative pumping: 1 = DONE (the residual is ready), 0 = still computing.
 * CALL IN A LOOP UNTIL DONE (§8.1 step 4). A single call = a guaranteed
 * npu_miss every hop: t_call 2866 us against ~350 us of skeleton render. */
int  n6_npu_poll(n6_npu_t *n);

/* residual[COUT][V][T] of the last DONE hop. */
const float *n6_npu_residual(const n6_npu_t *n);

/* §8.2/R-3: clear the state slice of voice v — call ONLY after DONE, in the
 * OUTPUT set (before the ping-pong swap), otherwise it races with the
 * state_out write. */
void n6_npu_zero_voice(n6_npu_t *n, int v);
void n6_npu_swap_states(n6_npu_t *n);      /* ping <-> pong of pointers */
#endif
