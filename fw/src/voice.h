/* voice.h — the voice manager (spec §4.2, guide §6.3): allocation
 * free -> the oldest in release -> steal the oldest;
 * the 250 Hz tick emits control frames; the glide is exponential in log-f0. */
#ifndef N6_VOICE_H
#define N6_VOICE_H
#include "n6_config.h"
#include "midi.h"

#define N6_MAX_VOICES 8

typedef enum { V_FREE = 0, V_ON, V_RELEASE } n6_vstate_t;

typedef struct {
    n6_vstate_t st;
    uint8_t  note;
    uint32_t age;                        /* younger = larger */
    double   logf0_cur, logf0_tgt;
    float    vel_amp;
    float    env;                        /* simple AR gate envelope */
    int      retrig;                     /* NoteOn: clear the NPU slice (§8.2!) */
    int      held;                       /* NoteOff came under CC64 — waits for the pedal */
    int      uni_slave;                  /* D-19: unison duplicate, stolen first */
} n6_voice_t;

typedef struct {
    n6_params_t prm;
    n6_voice_t  v[N6_MAX_VOICES];
    uint32_t    age_ctr;
    float  pb_semitones;                 /* from PB */
    float  timbreA, timbreB;             /* AT / CC1 */
    int    sustain;                      /* CC64 */
    float  attack_fr, release_fr;        /* env increment per frame */
    /* D-19: f0 micro-drift (log domain), its own deterministic rng */
    float    drift[N6_MAX_VOICES];
    uint32_t drift_rng;
} n6_voicemgr_t;

void n6_vm_init(n6_voicemgr_t *m, const n6_params_t *prm);
void n6_vm_event(n6_voicemgr_t *m, const n6_midi_ev_t *ev);
/* The 250 Hz tick: frames for all voices (frames[n_voices]); the retrig flags
 * are read by the pipeline (NPU states are cleared AFTER DONE, R-3/§8.2). */
void n6_vm_tick(n6_voicemgr_t *m, n6_frame_t *frames);

#endif
