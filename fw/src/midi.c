#include <string.h>
#include "midi.h"

void n6_midi_parser_init(n6_midi_parser_t *p) { memset(p, 0, sizeof *p); }

static int need_of(uint8_t st) {
    switch (st & 0xF0) {
    case 0xC0: case 0xD0: return 1;      /* PC, Channel Pressure */
    default:              return 2;      /* NoteOn/Off, CC, PB, PolyAT */
    }
}

int n6_midi_parse_byte(n6_midi_parser_t *p, uint8_t b, n6_midi_ev_t *ev)
{
    if (b >= 0xF8) return 0;             /* realtime — passed through (§6.2.4) */
    if (b == 0xF0) { p->in_sysex = 1; p->status = 0; return 0; }
    if (b == 0xF7) { p->in_sysex = 0; return 0; }
    if (p->in_sysex) return 0;
    if (b >= 0xF0) { p->status = 0; return 0; }  /* other system common */

    if (b & 0x80) {                      /* status byte */
        p->status = b;
        p->need = need_of(b);
        p->have = 0;
        return 0;
    }
    if (!p->status) return 0;            /* data without status — garbage */
    p->data[p->have++] = b;
    if (p->have < p->need) return 0;
    p->have = 0;                         /* running status: status persists */

    uint8_t hi = p->status & 0xF0, ch = p->status & 0x0F;
    ev->ch = ch; ev->d1 = p->data[0]; ev->d2 = p->need > 1 ? p->data[1] : 0;
    switch (hi) {
    case 0x90: ev->type = ev->d2 ? N6_EV_NOTE_ON : N6_EV_NOTE_OFF; return 1;
    case 0x80: ev->type = N6_EV_NOTE_OFF; return 1;
    case 0xB0: ev->type = N6_EV_CC; return 1;
    case 0xE0: ev->type = N6_EV_PB; return 1;
    case 0xD0: ev->type = N6_EV_AT; return 1;
    case 0xC0: ev->type = N6_EV_PC; return 1;
    default: return 0;                   /* PolyAT 0xA0 — ignored in v1 */
    }
}
