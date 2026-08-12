/* midi.h — SPSC byte FIFO (the ISR is the producer) + the parser state machine
 * (§6.2: running status, realtime bytes 0xF8..0xFF do not break the machine,
 * NoteOn vel=0 == NoteOff, SysEx is skipped). C11 atomics — the
 * ZeroCopySdrBuffer pattern (LDA/STL are native on ARMv8-M). */
#ifndef N6_MIDI_H
#define N6_MIDI_H
#include <stdint.h>
#include <stdatomic.h>

#define N6_MIDI_FIFO 256                 /* power of two */

typedef struct {
    uint8_t buf[N6_MIDI_FIFO];
    _Atomic uint32_t wr, rd;
} n6_midi_fifo_t;

static inline void n6_mf_init(n6_midi_fifo_t *f) {
    atomic_store(&f->wr, 0); atomic_store(&f->rd, 0);
}
static inline int n6_mf_push(n6_midi_fifo_t *f, uint8_t b) {   /* from the ISR */
    uint32_t w = atomic_load_explicit(&f->wr, memory_order_relaxed);
    uint32_t r = atomic_load_explicit(&f->rd, memory_order_acquire);
    if (((w + 1) & (N6_MIDI_FIFO - 1)) == (r & (N6_MIDI_FIFO - 1))) return 0;
    f->buf[w & (N6_MIDI_FIFO - 1)] = b;
    atomic_store_explicit(&f->wr, w + 1, memory_order_release);
    return 1;
}
static inline int n6_mf_pop(n6_midi_fifo_t *f, uint8_t *b) {   /* from the tick */
    uint32_t r = atomic_load_explicit(&f->rd, memory_order_relaxed);
    uint32_t w = atomic_load_explicit(&f->wr, memory_order_acquire);
    if (r == w) return 0;
    *b = f->buf[r & (N6_MIDI_FIFO - 1)];
    atomic_store_explicit(&f->rd, r + 1, memory_order_release);
    return 1;
}

typedef enum {
    N6_EV_NOTE_ON, N6_EV_NOTE_OFF, N6_EV_CC, N6_EV_PB, N6_EV_AT, N6_EV_PC
} n6_midi_ev_type_t;

typedef struct {
    n6_midi_ev_type_t type;
    uint8_t ch, d1, d2;                  /* PB: d1=lsb d2=msb */
} n6_midi_ev_t;

typedef struct {
    uint8_t status;                      /* running status; 0 = none */
    uint8_t data[2];
    uint8_t need, have;
    int     in_sysex;
} n6_midi_parser_t;

void n6_midi_parser_init(n6_midi_parser_t *p);
/* Feed one byte; returns 1 if an event is ready (in *ev). */
int  n6_midi_parse_byte(n6_midi_parser_t *p, uint8_t byte, n6_midi_ev_t *ev);

#endif
