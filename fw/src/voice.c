#include <math.h>
#include <string.h>
#include "voice.h"
#include "xorshift.h"

void n6_vm_init(n6_voicemgr_t *m, const n6_params_t *prm)
{
    memset(m, 0, sizeof *m);
    m->prm = *prm;
    m->timbreA = 0.5f; m->timbreB = 0.15f;
    m->attack_fr  = 1.0f / 2.0f;         /* ~8 ms attack (2 frames) */
    m->release_fr = 1.0f / 40.0f;        /* ~160 ms release */
    m->drift_rng  = n6_xs32_seed(0xD21F7A11u);   /* D-19, deterministic */
}

static double note_logf0(uint8_t note) {
    return log(440.0) + ((int)note - 69) * (M_LN2 / 12.0);
}

static int alloc_voice(n6_voicemgr_t *m)
{
    int best = -1; uint32_t oldest = UINT32_MAX;
    for (int i = 0; i < m->prm.n_voices; ++i)      /* 1) a free one */
        if (m->v[i].st == V_FREE) return i;
    /* 1a) the unison slave (D-19): it is a duplicate, not a note — steal it
     * first, and a chord seamlessly displaces the unison of a single note */
    for (int i = 0; i < m->prm.n_voices; ++i)
        if (m->v[i].uni_slave) return i;
    for (int i = 0; i < m->prm.n_voices; ++i)      /* 2) oldest release */
        if (m->v[i].st == V_RELEASE && m->v[i].age < oldest)
            { oldest = m->v[i].age; best = i; }
    if (best >= 0) return best;
    oldest = UINT32_MAX;                           /* 3) steal the oldest */
    for (int i = 0; i < m->prm.n_voices; ++i)
        if (m->v[i].age < oldest) { oldest = m->v[i].age; best = i; }
    return best;
}

void n6_vm_event(n6_voicemgr_t *m, const n6_midi_ev_t *ev)
{
    switch (ev->type) {
    case N6_EV_NOTE_ON: {
        int i = alloc_voice(m);
        n6_voice_t *v = &m->v[i];
        double lf = note_logf0(ev->d1);
        v->st = V_ON; v->note = ev->d1; v->age = ++m->age_ctr;
        v->logf0_tgt = lf;
        if (v->env <= 0.001f) v->logf0_cur = lf;   /* new voice — no glide */
        v->vel_amp = 0.15f + 0.85f * (float)ev->d2 / 127.0f;
        v->retrig = 1;                             /* §8.2: NPU slice after DONE */
        v->held = 0;
        v->uni_slave = 0;
        /* Auto-unison (D-19): the note is the ONLY one and there is a free
         * voice — duplicate it with a detune. The slave is a full voice: its
         * own NPU/FIR slices via retrig, its own skeleton state; NoteOff
         * releases both by note match; a chord steals the slave first (see
         * alloc_voice). The budget is already paid: V=2 is computed by the
         * network every hop regardless of the notes. */
        if (m->prm.uni_cents != 0.0f) {
            int busy = 0, free_j = -1;
            for (int j = 0; j < m->prm.n_voices; ++j) {
                if (j == i) continue;
                if (m->v[j].st == V_ON) busy = 1;
                else if (m->v[j].st == V_FREE && free_j < 0) free_j = j;
            }
            if (!busy && free_j >= 0) {
                n6_voice_t *u = &m->v[free_j];
                *u = *v;                           /* copy of the master, retrig=1 */
                u->uni_slave = 1;
                u->logf0_tgt += (double)m->prm.uni_cents * (M_LN2 / 1200.0);
                u->logf0_cur = u->logf0_tgt;       /* detune with no glide */
            }
        }
        break; }
    case N6_EV_NOTE_OFF:
        /* with the pedal down the note keeps sounding, but is MARKED: the
         * keyboard will not send a second NoteOff, and without the mark the
         * voice would hang until it was stolen */
        for (int i = 0; i < m->prm.n_voices; ++i)
            if (m->v[i].st == V_ON && m->v[i].note == ev->d1) {
                if (m->sustain) m->v[i].held = 1;
                else { m->v[i].st = V_RELEASE; m->v[i].held = 0; }
            }
        break;
    case N6_EV_CC:
        if (ev->d1 == 1)  m->timbreB = (float)ev->d2 / 127.0f;
        if (ev->d1 == 64) {
            int on = ev->d2 >= 64;
            if (m->sustain && !on)               /* pedal released */
                for (int i = 0; i < m->prm.n_voices; ++i)
                    if (m->v[i].st == V_ON && m->v[i].held) {
                        m->v[i].st = V_RELEASE; m->v[i].held = 0;
                    }
            m->sustain = on;
        }
        /* emergency exit for a stuck note: 120 All Sound Off,
         * 123 All Notes Off (both go to release, without a click) */
        if (ev->d1 == 120 || ev->d1 == 123) {
            m->sustain = 0;
            for (int i = 0; i < m->prm.n_voices; ++i)
                if (m->v[i].st == V_ON) {
                    m->v[i].st = V_RELEASE; m->v[i].held = 0;
                }
        }
        break;
    case N6_EV_PB: {
        int val = ((int)ev->d2 << 7 | ev->d1) - 8192;
        m->pb_semitones = m->prm.pb_range_semitones * (float)val / 8192.0f;
        break; }
    case N6_EV_AT: m->timbreA = (float)ev->d1 / 127.0f; break;
    default: break;
    }
}

void n6_vm_tick(n6_voicemgr_t *m, n6_frame_t *frames)
{
    double glide_k = 1.0 - exp(-1.0 / (m->prm.glide_s * N6_FRAME_HZ + 1e-9));
    /* f0 micro-drift (D-19): an OU walk in the log domain on the frame grid.
     * Mean reversion k2=0.01 (τ≈0.4 s at 250 Hz); the step is picked for a
     * stationary std dev = drift_cents (σ_step = σ·sqrt(24·k2)/2); clamp 3σ.
     * Each voice has its own drift — a unison pair breathes with a mutual
     * detuning. */
    const float dr_sig = m->prm.drift_cents * (float)(M_LN2 / 1200.0);
    for (int i = 0; i < m->prm.n_voices; ++i) {
        n6_voice_t *v = &m->v[i];
        if (dr_sig != 0.0f) {
            float d = m->drift[i];
            d += n6_xs32_next_f32(&m->drift_rng) * dr_sig * 0.245f;
            d -= d * 0.01f;
            float lim = 3.0f * dr_sig;
            if (d > lim) d = lim; else if (d < -lim) d = -lim;
            m->drift[i] = d;
        }
        v->logf0_cur += (v->logf0_tgt - v->logf0_cur) * glide_k;
        if (v->st == V_ON)
            v->env += (1.0f - v->env) * m->attack_fr;
        else {
            v->env -= v->env * m->release_fr;
            if (v->env < 1e-4f && v->st == V_RELEASE)
                { v->st = V_FREE; v->env = 0.0f; }
        }
        frames[i].f0 = (float)exp(v->logf0_cur
                        + (double)m->pb_semitones * (M_LN2 / 12.0)
                        + (double)m->drift[i]);
        frames[i].amp = v->vel_amp;
        frames[i].tA = m->timbreA;
        frames[i].tB = m->timbreB;
        frames[i].gate = v->env;
    }
}
