/* host_play.c — rendering the chain to WAV WITHOUT the board (endgame step 5,
 * D-19 A/B).
 *
 * Runs THE SAME C pipeline as the board: voices (unison/drift D-19), the
 * skeleton (bloom D-19), the FIR bank (the production n6_fir_coeffs.h),
 * wow/flutter, the limiter. The network is a null stub (residual 0): this is
 * the "skeleton+FIR" chain; the network's contribution is auditioned
 * separately via train/audition_delta.py.
 *
 * Score: three single notes (the unison/drift is audible), then a chord (the
 * unison gives up its voice), releases. ~9 s, 48 kHz, 16 bit, mono.
 *
 *   ./build/host_play out.wav [uni_cents bloom_k drift_cents]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/pipeline.h"

static void wav_write(const char *path, const float *x, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(2); }
    uint32_t sr = 48000, bps = sr * 2, dlen = (uint32_t)(n * 2);
    uint16_t blk = 2, bits = 16, fmt = 1, ch = 1;
    uint32_t riff = 36 + dlen, fmtlen = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmtlen, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr, 4, 1, f);
    fwrite(&bps, 4, 1, f); fwrite(&blk, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dlen, 4, 1, f);
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

typedef struct { int hop; int type; uint8_t d1, d2; } ev_t;

int main(int argc, char **argv)
{
    const char *out = argc > 1 ? argv[1] : "build/play.wav";
    n6_params_t prm = N6_PARAMS_DEFAULT;
    if (argc > 4) {
        prm.uni_cents   = (float)atof(argv[2]);
        prm.bloom_k     = (float)atof(argv[3]);
        prm.drift_cents = (float)atof(argv[4]);
    }
    n6_pipe_t *p = malloc(sizeof *p);
    n6_pipe_init(p, &prm);

    enum { NH = 2250 };                     /* 9 s */
    static const ev_t SCORE[] = {
        {  50, N6_EV_NOTE_ON,  45, 92 },    /* A2: unison beating in the low end */
        { 500, N6_EV_NOTE_OFF, 45, 0  },
        { 600, N6_EV_NOTE_ON,  52, 84 },    /* E3 */
        {1050, N6_EV_NOTE_OFF, 52, 0  },
        {1150, N6_EV_NOTE_ON,  60, 76 },    /* C4: bloom audible on the attack */
        {1500, N6_EV_NOTE_OFF, 60, 0  },
        {1650, N6_EV_NOTE_ON,  57, 88 },    /* chord: the unison gives way */
        {1652, N6_EV_NOTE_ON,  64, 88 },
        {2100, N6_EV_NOTE_OFF, 57, 0  },
        {2100, N6_EV_NOTE_OFF, 64, 0  },
    };
    const int NEV = (int)(sizeof SCORE / sizeof SCORE[0]);

    float *y = malloc((size_t)NH * N6_HOP48 * sizeof(float));
    float out48[N6_HOP48];
    int e = 0;
    for (int h = 0; h < NH; ++h) {
        while (e < NEV && SCORE[e].hop == h) {
            n6_midi_ev_t ev = { SCORE[e].type, 0, SCORE[e].d1, SCORE[e].d2 };
            n6_vm_event(&p->vm, &ev);
            ++e;
        }
        n6_pipe_hop(p, out48);
        memcpy(y + (size_t)h * N6_HOP48, out48, sizeof out48);
    }
    wav_write(out, y, (size_t)NH * N6_HOP48);
    printf("%s: %d hop, unison %.0f cents, bloom %.2f, drift %.0f cents\n",
           out, NH, (double)prm.uni_cents, (double)prm.bloom_k,
           (double)prm.drift_cents);
    free(y); free(p);
    return 0;
}
