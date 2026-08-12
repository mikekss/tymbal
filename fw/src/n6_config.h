/* n6_config.h — canonical constants of the N6 chain (NOT dependent on M0).
 * Everything M0 decides (V, C, L of the net, hop 4->2 ms) is in n6_params_t,
 * not in #define: hard rule of the guide §0. Canon of values: the project contracts file / docs. */
#ifndef N6_CONFIG_H
#define N6_CONFIG_H

#include <stdint.h>

#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

#define N6_FS48        48000
#define N6_FSB         12000
#define N6_BANDS       4
/* level of the skeleton noise component (B-3); the per-band tilt is D-23,
 * it is folded into this same coefficient when the voice is initialised */
#define N6_SKB_NOISE_B (0.5f * 0.3f)
#define N6_FRAME_HZ    250          /* control frames (C-1) */
#define N6_HOP48       192          /* 4 ms @48k (M0 may give 96) */
#define N6_HOP12       48
#define N6_NH          100          /* max harmonics (spec §5.3) */
#define N6_PQMF_TAPS   128          /* see fw/pqmf_coeffs.h */
#define N6_PQMF_PHASES 32           /* TAPS / BANDS */

/* NPU graph shape (FROZEN 2 Aug, D-2/D-9/D-11): buffer layout is in
 * npu_iface.h. c_in = 4 subbands + amp*gate + tA + tB + macro axis t. */
#define N6_NPU_CIN     8
#define N6_NPU_COUT    4
/* Ceiling on NPU pumping per hop (§8.3). On the target a DWT deadline is more
 * correct: 4 ms slot, t_call 2866 us — here it is only a guard against an
 * infinite loop. */
#define N6_NPU_POLL_MAX 1000000u

/* INTERLEAVING RENDER AND PUMPING (4 Aug). LL_ATON_RT_RunEpochBlock is not
 * blocking: it advances a state machine and returns, so the NPU runs exactly
 * as far as it is polled. Measurement: 13209 polls over 69 blocks, ~90 cycles
 * each — the processor spins idle for 1.2 M cycles per hop, and by that point
 * the skeleton has already been computed and was queued BEFORE the idle
 * stretch. Now the render is cut into spans of N6_SKB_SPAN samples, and
 * N6_NPU_PUMP polls are done between them.
 * Choice of the numbers: an epoch block lives ~17 k cycles on average, a span
 * of 4 samples costs ~25 k while playing, so 8 polls per span give one poll
 * about every 3 k cycles — the delay in detecting the end of a block is about
 * 1.5 k, over 69 blocks that is ~100 k, against 600 k of hidden work. SPAN is
 * a multiple of SKB_WSEG so that the span boundary coincides with the boundary
 * of a band-response segment. */
#define N6_SKB_SPAN   4
#define N6_NPU_PUMP   8

/* Parameters fixed by M0 / by the preset — runtime, canon by default */
typedef struct {
    int   n_voices;                 /* D-2 (frozen 2 Aug): 2 */
    int   hop48;                    /* D-3: 192 (4 ms) or 96 (2 ms) */
    float pb_range_semitones;       /* spec §4.1: 2.0 */
    float glide_s;                  /* MONO glide, CC-configured */
    /* Skeleton "analogueness" (5 Aug, D-19). ZEROS = behaviour before the
     * change: golden, CK4 and qemu-ck4 are computed with zeros and do not
     * move; the production preset is switched on in main.c (outside N6_CK4). */
    float uni_cents;    /* auto-unison on a free voice: detune, cents */
    float bloom_k;      /* A[h] tilt breathes with the envelope: p += k*(1-gate) */
    float drift_cents;  /* f0 micro-drift: std dev of the OU walk, cents */
    /* Master level BEFORE the limiter (D-22, 7 Aug). 1.0 = behaviour before
     * the change: the references are computed with one and do not move. The
     * raw chain peaks at 1.77 on a vel=127 chord — with one the limiter is
     * working almost all the time, and that is audible as clipping. */
    float out_gain;
    /* D-23 (8 Aug): tilt of the skeleton noise component across subbands,
     * dB PER BAND. 0 = as before — white noise at one level in all four
     * bands, because of which above 4 kHz it is almost equal to the tone.
     * Checked on the host: a 6 dB/band tilt removes ~7 dB of skeleton HF
     * noise and IMPROVES the metric (line +23.79 -> +24.36 dB), 12 dB
     * gives nothing more. The references are computed with zero and do not
     * move. */
    float noise_tilt_db;
} n6_params_t;

/* FROZEN CANON: wide duet C=88/V=2/T=48 (d44, 2866 us/hop).
 * CK4 deliberately runs a TRIO (host_ck4.c sets n_voices=3): its job is
 * worst-case DSP, not the runtime canon; the ck4_ref.bin reference does not
 * change when the canon changes. */
#define N6_PARAMS_DEFAULT { 2, N6_HOP48, 2.0f, 0.06f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f }

/* --- TCM (4 Aug) ---------------------------------------------------------
 * The Cortex-M55 has a DTCM — 128 kB glued to the core: zero wait states,
 * past the cache and past AXI, that is, no contention with the NPU. The
 * project did not use it at all: the FSBL linker script knew only AXISRAM.
 *
 * That is exactly what was hurting skb_wtab: 2.8-3.7 cycles per instruction
 * against 2.0-2.4 for the rest of the skeleton, and a spread of 110 k - 230 k
 * across the chord register — the step through the table grows with f0, and
 * the working set of the pass falls out of the cache. The n6_bandresp table
 * in interleaved form is 30.0 kB; it fits into DTCM whole, together with all
 * the hot skeleton arrays (~40 kB out of 128).
 *
 * TWO THINGS WITHOUT WHICH THERE WILL BE A HARD FAULT (established from ST's
 * write-up, not the hard way):
 *   1. TCM has ECC, and after reset its contents are random while the ECC bits
 *      do not match them. The VERY FIRST read gives PECC (AFSR bit 17). So the
 *      whole area must be written once, in words, before any access — that is
 *      what n6_dtcm_init() does at the start of main.
 *   2. The FSBL lives in the SECURE world (AXISRAM at 0x34...), so our DTCM is
 *      at 0x30000000 and not at 0x20000000 — the 0x2 alias is non-secure.
 *
 * On the host and on the QEMU rig the macro expands differently: they have
 * their own map. */
#if defined(N6_TARGET) && !defined(N6_NO_DTCM)
#define N6_DTCM  __attribute__((section(".dtcm"), aligned(32)))
#else
#define N6_DTCM  __attribute__((aligned(32)))
#endif

/* Control frame of one voice (spec §4.3) */
typedef struct {
    float f0, amp, tA, tB, gate;
} n6_frame_t;

#endif
