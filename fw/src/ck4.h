/* ck4.h — checkpoint 4 (H1): numeric validation of the MVE branches ON THE BOARD.
 * Guide §2.4 "golden on the board": the host tests run the SCALAR reference,
 * the MVE branches (#ifdef __ARM_FEATURE_MVE in skeleton_b/pqmf) have never
 * been cross-checked numerically.
 *
 * Scheme: a fixed MIDI score (raw bytes, including running status and
 * channel pressure — the FSM gets exercised along the way) is injected into
 * THE SAME FIFO the UART ISR writes to; after NW_WARMUP hops, N6_CK4_HOPS
 * hops of the float output are captured (after the limiter, BEFORE the
 * 24-in-32 quantization); then a line-by-line hex dump with CRC32 to the VCP
 * — one line per super-loop iteration, so as not to block the render
 * (printing from an ISR is forbidden, §8.3).
 * The host computes the reference with THE SAME code (make ck4 ->
 * build/ck4_ref.bin); comparison: tools/ck4_compare.py, criterion rel RMS <
 * 1e-4 (headroom over vfma fusing and the ulp difference of libm sinf
 * host/target ~1e-6..1e-5; a failure at 1e-4..1e-3 — look per-band, suspect
 * libm, not MVE).
 *
 * Determinism is provided for: the pipe_init seeds are fixed, the NPU stub is
 * null, the events are aligned to hop boundaries, the uint32 phase is exact.
 * The only non-determinism in the chain — the real UART — is taken out of
 * play here.
 *
 * Integration into N6_m1 (three calls, all under #ifdef N6_CK4 — see docs/ck4_notes.md):
 *   render_into_half():  n6_ck4_pre_hop(&g_mf);   // BEFORE draining the FIFO
 *                        ... n6_pipe_hop(...);
 *                        n6_ck4_post_hop(out48, N6_HOP48); // AFTER pipe_hop
 *   super-loop:          if (n6_ck4_dump_line(line, sizeof line) > 0) puts(line);
 * Buffer 48000 f32 = 192K -> section .ram2 (check the region size: g_pipe
 * ~198K is already there; the RAM2 bank of AXISRAM must fit both — see
 * ck4_notes). */
#ifndef N6_CK4_H
#define N6_CK4_H
#include <stdint.h>
#include "midi.h"

#define N6_CK4_WARMUP 25            /* hops of silence before the score (100 ms) */
#define N6_CK4_HOPS   250           /* capture window: 1 s @ hop 4 ms */

/* Call once per hop BEFORE draining the MIDI FIFO: advances the hop counter
 * and injects the score bytes assigned to this hop. */
void n6_ck4_pre_hop(n6_midi_fifo_t *mf);

/* Call AFTER n6_pipe_hop: captures out48 into the window [0, N6_CK4_HOPS). */
void n6_ck4_post_hop(const float *out48, int hop48);

/* 1 == the capture is finished (the dump can begin). */
int n6_ck4_done(void);

/* Re-entrant dump: puts the next line into dst (no '\n', with '\0'),
 * returns: the line length; 0 — the capture is still running or the dump has
 * already finished.
 * Format: "CK4 BEGIN <n_words>" / 12 words %08lx separated by spaces /
 * "CK4 CRC <crc32 of the buffer's LE bytes>" / "CK4 END". cap >= 120. */
int n6_ck4_dump_line(char *dst, int cap);

/* For the host reference: direct access to the capture buffer. */
const float *n6_ck4_buf(void);

#endif
