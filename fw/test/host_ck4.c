/* host_ck4.c — the host reference for checkpoint 4 (see src/ck4.h).
 * Runs THE SAME loop as render_into_half in N6_m1: pre_hop -> drain the
 * FIFO through the FSM -> vm_event -> pipe_hop -> post_hop. The scalar
 * branches (host without MVE) = the reference; the board with MVE is
 * cross-checked by ck4_compare.py.
 * Output: build/ck4_ref.bin (f32 LE) + CRC32 + a self-test of the dump
 * mechanism (reconstruction from the lines == the buffer bit for bit). The
 * ck4 state is global and single-use — one run per process (as on the
 * board). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/pipeline.h"
#include "../src/ck4.h"

static void run_once(float *dst)
{
    /* fresh state for every run */
    static n6_pipe_t pipe;                       /* ~198K — not on the stack */
    n6_midi_fifo_t mf;
    n6_midi_parser_t mp;
    n6_params_t prm = N6_PARAMS_DEFAULT;
    /* CK4 is worst-case DSP, not the runtime canon: the ck4.c score is built
     * for a TRIO of low keys (the 3.15M chord). We keep 3 explicitly, so that
     * freezing the V=2 canon (2 Aug) does not move the reference and does not
     * devalue the PASS on the board (rel RMS 4.085e-06, CRC 9cb3b2cb). */
    prm.n_voices = 3;
    n6_mf_init(&mf);
    n6_midi_parser_init(&mp);
    n6_pipe_init(&pipe, &prm);

    float out48[N6_HOP48];
    uint8_t b; n6_midi_ev_t ev;
    for (int hop = 0; hop < N6_CK4_WARMUP + N6_CK4_HOPS + 2; ++hop) {
        n6_ck4_pre_hop(&mf);
        while (n6_mf_pop(&mf, &b))
            if (n6_midi_parse_byte(&mp, b, &ev))
                n6_vm_event(&pipe.vm, &ev);
        n6_pipe_hop(&pipe, out48);
        n6_ck4_post_hop(out48, N6_HOP48);
    }
    if (!n6_ck4_done()) { fprintf(stderr, "capture not finished!\n"); exit(2); }
    memcpy(dst, n6_ck4_buf(), (size_t)N6_CK4_HOPS * N6_HOP48 * sizeof(float));
}

int main(void)
{
    enum { NW = N6_CK4_HOPS * N6_HOP48 };
    static float ref[NW];
    run_once(ref);

    /* dump through the same line-by-line mechanism that will go to the VCP on
     * the board — this doubles as ITS test: reconstruction from the lines ==
     * the buffer */
    static float recon[NW];
    char line[256];
    int len, wi = 0;
    unsigned long crc_dump = 0;
    while ((len = n6_ck4_dump_line(line, sizeof line)) > 0) {
        if (!strncmp(line, "CK4 CRC ", 8)) { crc_dump = strtoul(line + 8, 0, 16); continue; }
        if (!strncmp(line, "CK4", 3)) continue;  /* BEGIN/END */
        char *p = line;
        while (*p && wi < NW) {
            uint32_t w = (uint32_t)strtoul(p, &p, 16);
            memcpy(&recon[wi++], &w, 4);
            while (*p == ' ') ++p;
        }
    }
    if (wi != NW || memcmp(recon, ref, sizeof ref)) {
        fprintf(stderr, "[ck4] dump != buffer (wi=%d)\n", wi); return 2;
    }

    double rms = 0; float peak = 0;
    for (int i = 0; i < NW; ++i) {
        rms += (double)ref[i] * ref[i];
        float a = ref[i] < 0 ? -ref[i] : ref[i];
        if (a > peak) peak = a;
    }
    rms = __builtin_sqrt(rms / NW);

    FILE *f = fopen("build/ck4_ref.bin", "wb");
    if (!f || fwrite(ref, sizeof(float), NW, f) != NW) { perror("ck4_ref.bin"); return 2; }
    fclose(f);
    printf("[ck4_ref] %d samples (1 s), RMS=%.4f, peak=%.3f, CRC=%08lx -> build/ck4_ref.bin\n",
           (int)NW, rms, peak, crc_dump);
    printf("[ck4_ref] dump mechanism: bit-for-bit reconstruction OK\n");
    return 0;
}
