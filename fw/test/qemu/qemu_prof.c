/* qemu_main.c — a hot-path profiler on a real Cortex-M55 decoder.
 *
 * WHY. Three attempts in a row at editing the skeleton missed on scale, and
 * a hypothesis could only be checked by a rebuild in CubeIDE and a run on the
 * board: a loop several minutes long and costing someone else's attention.
 * QEMU 8.2 can do mps3-an547 — a board with a Cortex-M55, the same MVE
 * decoder. Absolute cycles must not be taken from there (QEMU models neither
 * the cache nor dual issue), but with -icount shift=0 a virtual cycle ==
 * an instruction, and SysTick turns into an exact INSTRUCTION counter. That
 * is enough to separate "the edit removed work" from "the edit removed
 * nothing" before going to the board.
 *
 * It is built separately from the firmware (fw/Makefile, target qemu) and is
 * not part of it: the NPU here is a stub, the sound goes nowhere. */
#include <stdint.h>
#include <string.h>
#include "../../src/n6_config.h"
#include "../../src/pipeline.h"
#include "../../src/midi.h"

#include "qemu_io.h"

extern uint32_t n6_prof[4];
extern uint32_t n6_skb_prof[5];
extern uint32_t n6_cyc_now(void);

static n6_pipe_t g_pipe;
static float     g_out[N6_HOP48];

#define HOPS 250                    /* 1 s of sound */

int main(void)
{
    n6_params_t prm = N6_PARAMS_DEFAULT;
    prm.n_voices = 3;               /* worst case, as in CK4 */
    n6_pipe_init(&g_pipe, &prm);

    sh_write0("--- QEMU mps3-an547 / cortex-m55 ---\n");
    say("MVE in the build: ", (uint32_t)n6_skb_have_mve());

    /* CALIBRATION. SysTick ticks on the board's MODEL clock (32 MHz on the
     * mps3-an547), while -icount sets the time PER INSTRUCTION, so one tick is
     * several instructions, and the coefficient must be measured, not guessed.
     * The reference is a loop of exactly 2 instructions per iteration. After
     * that everything printed is multiplied by this coefficient and becomes
     * instructions. */
    uint32_t c0 = n6_cyc_now();
    __asm__ volatile ("movw r3, #0xFFFF\n\t"
                      "1: subs r3, r3, #1\n\t"
                      "bne 1b\n\t" ::: "r3", "cc");
    uint32_t c1 = n6_cyc_now();
    uint32_t ipt_x100 = (uint32_t)((2u * 65535u * 100u) / (c1 - c0 ? c1 - c0 : 1u));
    say("instructions per tick x100: ", ipt_x100);

    /* a dense triad: three voices live for all 250 hops */
    static const uint8_t notes[3] = { 52, 59, 64 };
    for (int i = 0; i < 3; ++i) {
        n6_midi_ev_t ev = { N6_EV_NOTE_ON, 0, notes[i], 100 };
        n6_vm_event(&g_pipe.vm, &ev);
    }

    for (int h = 0; h < 8; ++h) n6_pipe_hop(&g_pipe, g_out);   /* warm-up */
    n6_prof[0] = n6_prof[1] = n6_prof[2] = n6_prof[3] = 0u;
    for (int i = 0; i < 5; ++i) n6_skb_prof[i] = 0u;

    uint32_t t0 = n6_cyc_now();
    for (int h = 0; h < HOPS; ++h) n6_pipe_hop(&g_pipe, g_out);
    uint32_t t1 = n6_cyc_now();

    sh_write0("--- INSTRUCTIONS per hop (3 voices, dense triad) ---\n");
#define SAY(lbl, ticks) say(lbl, (uint32_t)(((uint64_t)(ticks) * ipt_x100) / (100u * HOPS)))
    SAY("hop total:  ", (t1 - t0));
    SAY("  skb:      ", n6_prof[0]);
    SAY("    wt:     ", n6_skb_prof[0]);
    SAY("    pro:    ", n6_skb_prof[1]);
    SAY("    bod:    ", n6_skb_prof[2]);
    SAY("    dec:    ", n6_skb_prof[3]);
    SAY("    ph:     ", n6_skb_prof[4]);
    SAY("    rest:   ", (n6_prof[0] - n6_skb_prof[0] - n6_skb_prof[1]
                       - n6_skb_prof[2] - n6_skb_prof[3] - n6_skb_prof[4]));
    SAY("  npu(stub):", n6_prof[1]);
    SAY("  pqmf:     ", n6_prof[2]);
    SAY("  wf:       ", n6_prof[3]);
    sh_exit();
    return 0;
}
