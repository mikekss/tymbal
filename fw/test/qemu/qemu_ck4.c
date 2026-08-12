/* qemu_ck4.c — CK4 on the Cortex-M55 model instead of the board.
 *
 * WHY. `make test` runs the SCALAR branches: there is no MVE on the host, and
 * the vector path of skeleton_b/pqmf is not covered by it at all. Until now
 * the only check of it was CK4 on the board — that is, every edit to the hot
 * loop cost a flash and a capture over UART. QEMU gives the same MVE decoder,
 * so THE SAME ck4.c score is run here, and the capture buffer is written via
 * semihosting into a file next to the host reference. Comparison — tools/ck4
 * or numpy: the criterion is the same, rel RMS < 1e-4.
 *
 * This does not replace CK4 on the board (the cache, DMA and the real chain
 * are there as well), but it catches vectorization errors before they get
 * that far. */
#include <stdint.h>
#include <string.h>
#include "qemu_io.h"
#include "../../src/pipeline.h"
#include "../../src/ck4.h"

static n6_pipe_t        g_pipe;
static n6_midi_fifo_t   g_mf;
static n6_midi_parser_t g_mp;
static float            g_out[N6_HOP48];

int main(void)
{
    n6_params_t prm = N6_PARAMS_DEFAULT;
    prm.n_voices = 3;                   /* as in host_ck4.c */
    n6_mf_init(&g_mf);
    n6_midi_parser_init(&g_mp);
    n6_pipe_init(&g_pipe, &prm);

    sh_write0("--- CK4 in QEMU (cortex-m55, MVE branches) ---\n");
    say("MVE in the build: ", (uint32_t)n6_skb_have_mve());

    uint8_t b; n6_midi_ev_t ev;
    for (int hop = 0; hop < N6_CK4_WARMUP + N6_CK4_HOPS + 2; ++hop) {
        n6_ck4_pre_hop(&g_mf);
        while (n6_mf_pop(&g_mf, &b))
            if (n6_midi_parse_byte(&g_mp, b, &ev))
                n6_vm_event(&g_pipe.vm, &ev);
        n6_pipe_hop(&g_pipe, g_out);
        n6_ck4_post_hop(g_out, N6_HOP48);
    }
    if (!n6_ck4_done()) { sh_write0("capture not finished!\n"); sh_exit(); }

    const uint32_t nb = (uint32_t)N6_CK4_HOPS * N6_HOP48 * 4u;
    int fd = sh_open_wb("build/ck4_qemu.bin");
    if (fd < 0) { sh_write0("could not open build/ck4_qemu.bin\n"); sh_exit(); }
    int rest = sh_write(fd, n6_ck4_buf(), nb);
    sh_close(fd);
    say("bytes not written: ", (uint32_t)rest);
    sh_write0("-> build/ck4_qemu.bin\n");
    sh_exit();
    return 0;
}
