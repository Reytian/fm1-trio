/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's engine: the player plays chords (the FM-1's keys or a MIDI keyboard), the band plays a walking bass and
 * jazz drums under them.
 *
 *   the audio interrupt (tr_render, tr_midi)  the transport, the player's piano, the band's notes on time
 *   the main loop (tr_service, tr_key)        listening (what chord, when), the form, and the networks: the bass's
 *                                             next note a little before its beat, the drummer a few steps ahead
 *
 * A chord played up to TR_EARLY of a beat before the beat counts for that beat; one played later counts from the
 * next beat (the bass has already gone). Once the chords repeat (a form), the band knows what comes and the bass
 * lands with the player even when the player plays exactly on the beat. */
#pragma once
#include <stdint.h>
#include "trio_nn.h"

enum { TR_AUTO, TR_TWO, TR_FOUR };              /* feel: AUTO = in two until the form comes round, then in four */
enum { TR_STOP, TR_COUNT, TR_PLAY };

typedef struct {
    uint16_t bpm;                               /* 60..260 */
    uint8_t feel;                               /* TR_AUTO, TR_TWO, TR_FOUR */
    uint8_t freedom;                            /* 0..100: the bass's temperature */
    uint8_t energy;                             /* 0..100: the drummer */
    uint8_t swing;                              /* 50..75: the swung eighth, % of the beat */
    uint8_t piano;                              /* 1: TRIO sounds the player's notes; 0: a piano of its own does */
    uint8_t vol_piano, vol_bass, vol_drums, room;   /* 0..100 */
    uint8_t follow;                             /* 1: the form memory */
} tr_set_t;
extern tr_set_t tr_set;

void tr_init(void);                             /* power-on: voices, networks */
void tr_defaults(tr_set_t *s);
void tr_render(int32_t *lr, uint32_t n, uint32_t gain_q12);   /* audio interrupt: n stereo frames, 24-bit */
void tr_midi(uint32_t st, uint32_t d1, uint32_t d2);          /* audio interrupt: a MIDI message in */
void tr_key(int note, int vel);                 /* main loop: a panel key (vel 0: up) */
void tr_play(int on);                           /* main loop: start (a bar's count-in) or stop */
int tr_service(void);                           /* main loop, every pass: 1 = a network ran */
void tr_forget(void);                           /* the form and the chords so far forgotten */
void tr_panic(void);
void tr_seed(uint32_t s);                       /* the band's dice, from the next start (tests) */

/* TEACHING the bass. While teaching, TRIO's bass is quiet and the player is the bass: single notes from E1 to G3 (the
 * FM-1's keys or any MIDI in), sounded by the upright and not heard as chords; above them, chords as ever. Each beat
 * the band plays (over the form, or the player's chords) keeps the frozen BASS network's state there, fed the
 * player's notes, with the player's note on that beat (quantised: up to 0.2 of a beat early, up to 0.4 late; later,
 * a pickup) and the chord: the last 32 bars of it. Teaching off, or STOP: the main loop fine-tunes the network's
 * output layer on those beats (trio_nn.h), a slice a pass, and the bass walks more like the player from then on.
 * On top of what it learned before; tr_unlearn: as trained. Kept in a flash object (the app: tr_learned_obj) */
int tr_teach(int on);                           /* main loop. on: 1 teaching (0: refused, still learning the last
                                                 * line); off: the beats it learns (0: too few notes, dropped) */
int tr_teaching(void);
int tr_learning(void);                          /* the main loop learning a line */
int tr_taught(void);                            /* beats kept so far */
int tr_learned(void);                           /* lines learned into the bass (0: as trained) */
void tr_unlearn(void);                          /* as trained (a lesson or a learning under way dropped) */
extern volatile uint32_t tr_learn_count;        /* learnings and forgettings so far (the app: unsaved) */
const void *tr_learned_obj(uint32_t *len);      /* what it learned, as its flash object keeps it */
void *tr_learned_room(uint32_t *max);           /* to load one: its place, then tr_learned_check(bytes loaded) */
int tr_learned_check(int len);                  /* 0: in use; else as trained */

/* for the screen and the tests */
extern volatile uint8_t tr_state;
extern volatile int32_t tr_beat;                /* the beat sounding (the count-in's are -4..-1) */
extern volatile uint32_t tr_missed;             /* beats or steps the main loop was late for */
int tr_chord_now(int *root, int *qual);         /* the chord now (0: none yet) */
int tr_chord_next(int *root, int *qual, int *in_beats);   /* 0 none, 1 from the form, 2 from the network */
int tr_form_bars(void);                         /* the form's length in bars once it has come round (0: not yet) */
int tr_bass_note(void);                         /* the last bass note (-1: none) */
int tr_feel_now(void);                          /* TR_TWO or TR_FOUR */
uint32_t tr_nn_steps(void);                     /* network steps so far */
void tr_chord_name(int root, int qual, char *out);   /* "Bbm7", "F7", ... */
/* what the band played (the tests): the bass's token on each beat, the drums' pieces on each step */
int tr_log_bass(int32_t beat);                  /* -1: not decided */
uint16_t tr_log_drums(int32_t step);
#if defined(TNN_HOST) || defined(PLAT_HOST)
int tr_log_taught(int32_t beat);                /* what the player played on a beat taught (-1: not taught) */
int tr_log_skip(int32_t beat);                  /* the bass's pickup after a beat (-1: not decided) */
int tr_teach_span(int32_t *first, int32_t *last);   /* the last lesson's beats (0: none) */
int32_t tr_learned_at(void);                    /* the beat the last lesson came into use from */
int tr_agree(int32_t from, int32_t to, int lag);    /* % of the bass's notes the player's on that beat of the form */
const bass_in_t *tr_teach_in(int32_t beat, int *skip);   /* a beat's inputs while teaching (0: none), its pickup */
const float *tr_teach_h0(void);                 /* the state the last lesson started from */
int tr_teach_bytes(void);                       /* the RAM teaching keeps (the lesson's beats, the slots) */
#endif
