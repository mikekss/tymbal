/* ck4.c — see ck4.h. The score and the capture/dump machinery.
 *
 * SCORE (hop 0 == the first hop after warmup; V=3, path coverage):
 *   5/10/15  NoteOn A0/C1/E1 (21/24/28) — WORST CASE: 3 low keys,
 *            ~100 harmonics x 3 voices (that very 3.15M chord from h1_notes);
 *   60..96   pitch-bend: ramp up to +8191 and back to center (step 3 hops) —
 *            B-1 phase catch-up with a live f0;
 *   110      CC1=96 (timbreB), 118 channel pressure 0xD0 (parser: need=1);
 *   130      NoteOff C1; 135 NoteOn C4 (60) — reallocation + retrig slice §8.2;
 *   150      NoteOn C3 (48) — STEALING the oldest voice (all 3 are busy);
 *   200      NoteOff on all WITHOUT repeating the status (running status
 *            0x80) — the parser and the release tails (~160 ms) to the end
 *            of the window.
 * The score may be changed, but the host reference must be rebuilt IN SYNC
 * (make ck4). */
#include <stdio.h>
#include <string.h>
#include "ck4.h"
#include "n6_config.h"

typedef struct { uint16_t hop; uint8_t len; uint8_t b[4]; } ck4_ev_t;

#define PB(h, v14) { (h), 3, { 0xE0, (uint8_t)((v14) & 0x7F), \
                               (uint8_t)(((v14) >> 7) & 0x7F), 0 } }
static const ck4_ev_t SCORE[] = {
    {   5, 3, { 0x90, 21, 100, 0 } },
    {  10, 3, { 0x90, 24, 110, 0 } },
    {  15, 3, { 0x90, 28, 120, 0 } },
    /* PB ramp: center 8192; up and back */
    PB( 60,  9500), PB( 63, 10800), PB( 66, 12100), PB( 69, 13400),
    PB( 72, 14700), PB( 75, 16000), PB( 78, 16383),
    PB( 81, 14000), PB( 84, 12000), PB( 87, 10500),
    PB( 90,  9300), PB( 93,  8600), PB( 96,  8192),
    { 110, 3, { 0xB0, 1, 96, 0 } },            /* CC1 -> timbreB */
    { 118, 2, { 0xD0, 90, 0, 0 } },            /* channel pressure -> timbreA */
    { 130, 3, { 0x80, 24, 64, 0 } },           /* NoteOff C1 */
    { 135, 3, { 0x90, 60, 80, 0 } },           /* NoteOn C4: retrig */
    { 150, 3, { 0x90, 48, 71, 0 } },           /* NoteOn C3: voice steal */
    { 200, 3, { 0x80, 21, 64, 0 } },           /* offs: running status below */
    { 200, 2, { 28, 64, 0, 0 } },
    { 200, 2, { 60, 64, 0, 0 } },
    { 200, 2, { 48, 64, 0, 0 } },
};
#define N_EV ((int)(sizeof SCORE / sizeof SCORE[0]))

#define CK4_NW (N6_CK4_HOPS * N6_HOP48)         /* 48000 f32 words */
#ifdef N6_TARGET
__attribute__((section(".ram2")))               /* NOLOAD region, like g_pipe */
#endif
static float g_ck4_buf[CK4_NW];

static int32_t g_hop = -(N6_CK4_WARMUP + 1);    /* pre_hop increments first */
static int g_next_ev = 0;
static int32_t g_cap = 0;                       /* samples captured */

void n6_ck4_pre_hop(n6_midi_fifo_t *mf)
{
    ++g_hop;
    while (g_next_ev < N_EV && g_hop >= 0 && SCORE[g_next_ev].hop == g_hop) {
        for (int i = 0; i < SCORE[g_next_ev].len; ++i)
            n6_mf_push(mf, SCORE[g_next_ev].b[i]);
        ++g_next_ev;
    }
}

void n6_ck4_post_hop(const float *out48, int hop48)
{
    if (g_hop < 0 || g_cap >= CK4_NW) return;
    int n = hop48;
    if (g_cap + n > CK4_NW) n = CK4_NW - g_cap;
    memcpy(g_ck4_buf + g_cap, out48, (size_t)n * sizeof(float));
    g_cap += n;
}

int n6_ck4_done(void) { return g_cap >= CK4_NW; }

const float *n6_ck4_buf(void) { return g_ck4_buf; }

/* CRC32 (IEEE, poly 0xEDB88320) over the LE bytes of the buffer, table-free. */
static uint32_t ck4_crc32(const void *p, uint32_t nbytes)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < nbytes; ++i) {
        c ^= b[i];
        for (int k = 0; k < 8; ++k)
            c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

#define CK4_WPL 12                              /* words per line */
int n6_ck4_dump_line(char *dst, int cap)
{
    static int32_t line = -1;                   /* -1 BEGIN, 0.. data */
    static const int32_t nlines = (CK4_NW + CK4_WPL - 1) / CK4_WPL;
    if (!n6_ck4_done() || line > nlines + 1 || cap < 120) return 0;
    int len;
    if (line < 0) {
        len = snprintf(dst, (size_t)cap, "CK4 BEGIN %d", (int)CK4_NW);
    } else if (line < nlines) {
        const uint32_t *w = (const uint32_t *)(const void *)g_ck4_buf;
        int i0 = line * CK4_WPL, pos = 0;
        for (int i = i0; i < i0 + CK4_WPL && i < CK4_NW; ++i)
            pos += snprintf(dst + pos, (size_t)(cap - pos),
                            i == i0 ? "%08lx" : " %08lx", (unsigned long)w[i]);
        len = pos;
    } else if (line == nlines) {
        len = snprintf(dst, (size_t)cap, "CK4 CRC %08lx",
                       (unsigned long)ck4_crc32(g_ck4_buf, CK4_NW * 4u));
    } else {
        len = snprintf(dst, (size_t)cap, "CK4 END");
    }
    ++line;
    return len;
}
