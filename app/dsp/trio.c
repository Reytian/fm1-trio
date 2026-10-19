/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's engine (trio.h). One compilation unit, -O2. */
#include "trio.h"
#include "voice.h"
#include "fastmath.h"
#if defined(TNN_HOST) || defined(PLAT_HOST)
#include <stdio.h>
#include <stdlib.h>
#endif

tr_set_t tr_set;
volatile uint8_t tr_state;
volatile int32_t tr_beat;
volatile uint32_t tr_missed;

#define SR 44100.0f
#define TR_EARLY 0.45f                          /* a chord this much of a beat early counts for the beat */
#define TR_LEAD_MS 40u                          /* the bass decides this long before its beat */
#define DRUM_AHEAD 6                            /* the drummer decides this many steps ahead */
#define HIST 1024                               /* beats of chords remembered (256 bars) */
#define NCOMMIT 64

#if defined(TNN_HOST) || defined(PLAT_HOST)    /* the tests, fm1-base's simulator and emulator */
#define TR_POOL
#else
#define TR_POOL __attribute__((section(".pool")))
#endif

/* TEACHING the bass (tr_teach): the line's register, how a note falls to a beat, what a lesson keeps, and the
 * learning's settings (tools/trio_finetune.py sweep) */
#define LINE_LO TNN_LO                          /* the player's line: E1 .. G3 */
#define LINE_HI TNN_LINE_HI
#define TEACH_EARLY 0.20f                       /* a note up to this much of a beat early is the beat's */
#define TEACH_PICKUP 0.40f                      /* from this far into the beat on, a note is a pickup before the next */
#define TEACH_MAX 128                           /* beats a lesson keeps: the line's last 32 bars */
#define TEACH_MIN 8                             /* beats with a note the bass could have played, at least */
#define LRN_RATE 0.03f                          /* the learning rate and the pull back toward the trained rows */
#define LRN_PULL 0.1f                           /* (MIN-1's); the passes: with the pull, 15 reach where 30 or 60 */
#define LRN_PASSES 15                           /* would, within 0.001 nats (tools/trio_finetune.py sweep) */
#define LRN_SLICE 6                             /* SGD steps a pass of the main loop (~1.5 ms on the FM-1) */

/* ------------------------------------------------------------------ shared: interrupt <-> main loop */
typedef struct {
    volatile int32_t beat;                      /* the beat this decision is for (INT32_MIN: none) */
    int8_t tok;
    uint8_t skip;
    float vel;
} bass_ev_t;
typedef struct {
    volatile int32_t step;
    uint16_t hits;
    uint8_t lvl[TNN_NPIECE];
} drum_ev_t;
static bass_ev_t bass_ev[8];
static drum_ev_t drum_ev[32];

typedef struct {
    uint8_t note, vel;                          /* (note | LINE_NOTE: the player's line's, while teaching) */
    uint32_t t;                                 /* the sample it came at */
} note_ev_t;
#define LINE_NOTE 0x80u
static note_ev_t midi_q[64];                    /* the interrupt's MIDI notes, for the main loop's listening */
static volatile uint32_t midi_qw, midi_qr;
static note_ev_t key_q[32];                     /* the panel's keys, for the interrupt's piano */
static volatile uint32_t key_qw, key_qr;
static volatile uint8_t teach_route;            /* teaching: the line's notes go to the bass (the main loop sets it) */

/* the transport, as the interrupt last published it (a block at a time) */
static volatile uint32_t pub_samples;           /* samples since power-on at the block's start */
static volatile int32_t pub_beat;
static volatile float pub_phase, pub_spb;
static volatile uint8_t want_play, want_stop;

/* ------------------------------------------------------------------ the interrupt's own */
static uint32_t samples;
static int32_t beat;                            /* the beat playing */
static float phase, spb;                        /* samples into it; samples a beat */
static int step_in_beat;                        /* 0, 1, 2: the triplet steps passed */
static uint8_t sustain, sus_held[128];
static int bass_last = -1;
static uint8_t to_line[128];                    /* notes that went to the bass (teaching): their note-offs go there too */
static int line_note = -1;                      /* the player's line's note sounding */

/* ------------------------------------------------------------------ the main loop's own */
typedef struct {
    int32_t beat;                               /* the beat it counts from */
    int8_t root, qual;
} commit_t;
static commit_t commits[NCOMMIT];
static int ncommit;
static int8_t hist_root[HIST], hist_qual[HIST]; /* the chord on each beat (as it finally was) */
static int32_t hist_upto = -1;                  /* the last beat written */
static uint8_t held[128];                       /* the player's notes held (velocity) */
static uint32_t pressed_at[128];                /* when each was last pressed */
static uint32_t last_on;                        /* the last note-on (samples); 0: listened */
static int8_t cur_root = -1, cur_qual;          /* the chord heard last (stopped or playing) */
static int32_t bass_next = 0;                   /* the next beat to decide */
static int32_t drum_next = 0;                   /* the next step to decide */
static float bass_h[BASS_H], drum_h[DRUM_H], harm_h[HARM_H];
static int8_t bass_ptok = TNN_START;
static uint8_t bass_pskip;
static int8_t band_root = -1, band_qual;        /* the chord the bass last played over */
static int32_t band_since;
static uint16_t drum_phit;
static uint8_t drum_pvel[TNN_NPIECE];
static uint32_t rng = 0x2545F491u, nn_steps, seed = 0x2545F491u;
static int8_t net_root = -1, net_qual;          /* the network's next chord, and when */
static int32_t net_at;
static float net_conf;
static float harm_lr[12], harm_lq[TNN_NQUAL];   /* its log-probabilities of the next chord's root (over this one's) and kind */
static uint8_t harm_ok;
#ifndef HARM_PRIOR
#define HARM_PRIOR 0.0f                         /* off: as a prior it took root-position recognition from 97 % to 66 % (Autumn Leaves, tests) */
#endif
static int32_t harm_last_beat;
static int8_t harm_prev_root = -1;
static int8_t next_root = -1, next_qual, next_how;   /* what the screen says comes next */
static int32_t next_in;
static int form_lag, form_len;                  /* the form: its lag in beats and how many beats matched */
static uint32_t form_drops;                     /* forms the player proved wrong (the tests) */
#if defined(TNN_HOST) || defined(PLAT_HOST)
#define LOGN 8192                               /* the tests read back whole runs */
#else
#define LOGN 256
#endif
static int8_t bass_log[LOGN], chord_log[LOGN];
static int32_t bass_log_beat[LOGN];
static uint16_t drum_log[2 * LOGN];
static int32_t drum_log_step[2 * LOGN];
#if defined(TNN_HOST) || defined(PLAT_HOST)
static uint8_t bass_log_skip[LOGN];
#endif

/* TEACHING the bass (tr_teach). While it is on, the bass is quiet and the player's notes from E1 to G3 are its line:
 * each falls to a beat (a slot a beat: its note, the one nearest the beat, and a pickup after it). On each beat the
 * band plays, the frozen network still takes its step, fed the player's note on the beat before as if the bass had
 * played it, and the step's state is kept with the beat's chord (pend); once the beat's windows have gone, it is
 * closed with what the player played on it, and kept (the ring) if it teaches something. Then (off, or STOP) the
 * main loop learns what was kept, a slice a pass, and puts it in use */
typedef struct {
    int32_t beat;                               /* the beat this slot hears (INT32_MIN: none) */
    int8_t note;                                /* its note (-1: none yet) */
    uint8_t skip;                               /* a pickup after it */
    float df;                                   /* the note's distance from the beat (beats) */
} slot_t;
static tnn_beat_t teach_ring[TEACH_MAX] TR_POOL;
static struct {
    uint8_t on, learning, started;              /* teaching; the main loop learning what it kept; a beat decided since */
    int n, w;                                   /* beats kept; the next to write */
    tnn_beat_t pend;                            /* the beat decided last while teaching, its target to come */
    int32_t pend_beat;                          /* (INT32_MIN: none) */
    slot_t slot[4];                             /* the beats the player's notes fall to, by beat & 3 */
    uint8_t down[LINE_HI + 1], key_line[128];   /* the line's notes held; the panel's keys that went to it */
    float on_x[LINE_HI + 1], off_x[LINE_HI + 1];   /* when each of its notes was last pressed and let go (beats) */
    int32_t learned_at;                         /* the beat the last lesson came into use from (tests) */
} tch TR_POOL;
volatile uint32_t tr_learn_count;
#if defined(TNN_HOST) || defined(PLAT_HOST)
static int8_t teach_log[LOGN];                  /* what the player played on each beat taught (the tests' agreement) */
static uint8_t teach_log_skip[LOGN];
static int32_t teach_log_beat[LOGN];
static int32_t teach_first = -1, teach_last = -1;
static bass_in_t teach_in[LOGN];                /* each beat's inputs while teaching, and the state before the first */
static int32_t teach_in_beat[LOGN];             /* (tools/trio_finetune.py replays a lesson from them) */
static float teach_h0[BASS_H];
#endif
static void line_in(int note, int vel, uint32_t t);

/* the demo's log (tools/demo_video.py; the simulator and the tests only, never the FM-1): TRIO_VIZ=FILE writes what
 * the networks decided and when (in samples since power-on), a line each. H the chord heard (its notes); B a beat of
 * the bass (its 35 tokens' chances as the engine draws them; teaching: what it would have played); T the player's
 * note on a beat taught; D a drum step; E teaching on or off; L the learning (its start; its passes' losses at its
 * end; forgotten); F the form learned or dropped (its bars) */
#if defined(TNN_HOST) || defined(PLAT_HOST)
static FILE *viz_file(void)
{
    static FILE *f;
    static int tried;
    if (!tried) {
        const char *p = getenv("TRIO_VIZ");
        tried = 1;
        f = p ? fopen(p, "w") : 0;
    }
    return f;
}
#define VIZ(...)                                                                                                  \
    do {                                                                                                          \
        FILE *vf_ = viz_file();                                                                                   \
        if (vf_) {                                                                                                \
            fprintf(vf_, __VA_ARGS__);                                                                            \
            fflush(vf_);                                                                                          \
        }                                                                                                         \
    } while (0)
#else
#define VIZ(...) ((void)0)
#endif

static inline float flr(float x)
{
    int32_t i = (int32_t)x;
    return (float)(x < (float)i ? i - 1 : i);
}

static const uint16_t QMASK[TNN_NQUAL] = {0x891, 0x491, 0x489, 0x449, 0x249, 0x4A1, 0x111};

static uint16_t chord_mask(int root, int qual)
{
    uint32_t m = QMASK[qual < 0 || qual >= TNN_NQUAL ? 0 : qual];
    int k = root % 12;
    return (uint16_t)(((m << k) | (m >> (12 - k))) & 0xFFFu);
}

void tr_chord_name(int root, int qual, char *out)
{
    static const char *const R[12] = {"C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"};
    static const char *const Q[TNN_NQUAL] = {"maj7", "7", "m7", "m7b5", "dim7", "7sus", "+"};
    const char *a = root >= 0 && root < 12 ? R[root] : "-", *b = qual >= 0 && qual < TNN_NQUAL ? Q[qual] : "";
    while (*a)
        *out++ = *a++;
    while (root >= 0 && *b)
        *out++ = *b++;
    *out = 0;
}

void tr_defaults(tr_set_t *s)
{
    s->bpm = 140;
    s->feel = TR_AUTO;
    s->freedom = 50;
    s->energy = 50;
    s->swing = 64;
    s->piano = 1;
    s->vol_piano = 75;
    s->vol_bass = 80;
    s->vol_drums = 70;
    s->room = 35;
    s->follow = 1;
}

void tr_init(void)
{
    int i;
    tnn_init();
    v_init();
    tr_defaults(&tr_set);
    for (i = 0; i < 8; i++)
        bass_ev[i].beat = INT32_MIN;
    for (i = 0; i < 32; i++)
        drum_ev[i].step = INT32_MIN;
    spb = SR * 60.0f / (float)tr_set.bpm;
    pub_spb = spb;
    tch.on = tch.learning = tch.started = 0;
    tch.n = tch.w = 0;
    tch.pend_beat = INT32_MIN;
    tch.learned_at = 0;
    for (i = 0; i < 4; i++)
        tch.slot[i].beat = INT32_MIN;
    for (i = 0; i <= LINE_HI; i++) {
        tch.down[i] = 0;
        tch.on_x[i] = tch.off_x[i] = -1e9f;
    }
    for (i = 0; i < 128; i++)
        tch.key_line[i] = to_line[i] = 0;
    teach_route = 0;
    line_note = -1;
}

/* ================================================================== the interrupt */
static void ep_on(int note, int vel)
{
    if (tr_set.piano)
        v_ep_on(note, (float)vel * (1.0f / 127.0f));
}
static void ep_off(int note)
{
    if (sustain)
        sus_held[note & 127] = 1;
    else
        v_ep_off(note);
}

static void line_on(int note, int vel)          /* the player's line, on the bass (teaching) */
{
    to_line[note & 127] = 1;
    line_note = note;
    bass_last = note;
    v_bass_on(note, (float)vel * (1.0f / 127.0f), 0);
}
static void line_off(int note)                  /* let go: the string stopped (unless another note has followed) */
{
    to_line[note & 127] = 0;
    if (line_note == note) {
        line_note = -1;
        v_bass_damp();
    }
}

void tr_midi(uint32_t st, uint32_t d1, uint32_t d2)
{
    uint32_t type = st & 0xF0u, w;
    if (type == 0x90u && d2) {
        int line = teach_route && d1 >= LINE_LO && d1 <= LINE_HI;
        if (line)
            line_on((int)d1, (int)d2);
        else
            ep_on((int)d1, (int)d2);
        w = midi_qw;
        if (w - midi_qr < 64u) {
            midi_q[w & 63u] = (note_ev_t){(uint8_t)(d1 | (line ? LINE_NOTE : 0u)), (uint8_t)d2, samples};
            midi_qw = w + 1;
        }
    } else if (type == 0x80u || type == 0x90u) {
        int line = to_line[d1 & 127u];
        if (line)
            line_off((int)d1);
        else
            ep_off((int)d1);
        w = midi_qw;
        if (w - midi_qr < 64u) {
            midi_q[w & 63u] = (note_ev_t){(uint8_t)(d1 | (line ? LINE_NOTE : 0u)), 0, samples};
            midi_qw = w + 1;
        }
    } else if (type == 0xB0u && d1 == 64u) {    /* the sustain pedal holds the piano's notes */
        sustain = d2 >= 64u;
        if (!sustain) {
            int k;
            for (k = 0; k < 128; k++)
                if (sus_held[k]) {
                    sus_held[k] = 0;
                    v_ep_off(k);
                }
        }
    } else if (type == 0xB0u && (d1 == 123u || d1 == 120u)) {
        v_ep_all_off();
        if (line_note >= 0)                     /* and the line's note */
            line_off(line_note);
    }
}

static float step_frac(int k)                   /* where step k of a beat falls: on it, the middle, the swung eighth */
{
    float s = (float)tr_set.swing * 0.01f;
    return k == 0 ? 0.0f : k == 1 ? 0.5f * s : s;
}

static void count_in(int32_t b, int k)          /* the count-in: the hi-hat's foot on each beat, 2 and 4 harder */
{
    if (k == 0)
        v_drum(P_HHF, (b & 1) ? 0.85f : 0.5f);
}

static void play_step(int32_t b, int k)        /* the band's notes for step k of beat b */
{
    int32_t g = b * 3 + k;
    drum_ev_t *d = &drum_ev[g & 31];
    int p;
    if (tr_state == TR_COUNT) {
        count_in(b, k);
        return;
    }
    if (d->step == g) {
        static const float LV[4] = {0.0f, 0.32f, 0.58f, 0.88f};
        for (p = 0; p < TNN_NPIECE; p++)
            if (d->hits >> p & 1u)
                v_drum(p, LV[d->lvl[p] & 3u] * (p == P_KICK ? 0.7f : 1.0f));
    } else
        tr_missed++;
    if (k == 0 || k == 2) {
        bass_ev_t *e = &bass_ev[b & 7];
        if (e->beat != b) {
            if (k == 0)
                tr_missed++;
            return;
        }
        if (k == 0) {
            if (e->tok >= 2) {
                bass_last = TNN_LO + e->tok - 2;
                v_bass_on(bass_last, e->vel, 0);
            } else if (e->tok == TNN_REST)
                v_bass_damp();
        } else if (e->skip && bass_last >= 0)  /* the pickup: a ghost on the swung eighth */
            v_bass_on(bass_last, 0.35f, 1);
    }
}

void tr_render(int32_t *lr, uint32_t n, uint32_t gain_q12)
{
    static float L[512], R[512];
    v_mix_t mix;
    uint32_t done = 0, i;
    float gain = (float)gain_q12 * (1.0f / 4096.0f) * 8300000.0f;
    while (key_qr != key_qw) {                  /* the panel's keys into the piano (or the line, teaching) */
        note_ev_t e = key_q[key_qr & 31u];
        int note = e.note & 127;
        if (e.vel && (e.note & LINE_NOTE))
            line_on(note, e.vel);
        else if (e.vel)
            ep_on(note, e.vel);
        else if (to_line[note])
            line_off(note);
        else
            ep_off(note);
        key_qr++;
    }
    if (want_stop) {
        want_stop = 0;
        tr_state = TR_STOP;
        v_bass_damp();
    }
    if (want_play) {                            /* a bar's count-in, then beat 0 */
        want_play = 0;
        beat = -4;
        phase = 0.0f;
        step_in_beat = 0;
        tr_state = TR_COUNT;
        play_step(beat, 0);
    }
    spb = SR * 60.0f / (float)tr_set.bpm;
    mix.piano = (float)tr_set.vol_piano * 0.01f;
    mix.bass = (float)tr_set.vol_bass * 0.01f;
    mix.drums = (float)tr_set.vol_drums * 0.01f;
    mix.room = (float)tr_set.room * 0.01f;
    if (n > 512u)
        n = 512u;
    while (done < n) {
        uint32_t chunk = n - done;
        if (tr_state != TR_STOP) {               /* render up to the next step, then play it */
            float next = step_in_beat < 2 ? step_frac(step_in_beat + 1) * spb : spb;
            float left = next - phase;
            uint32_t k = left <= 1.0f ? 1u : (uint32_t)left;
            if (k < chunk)
                chunk = k;
        }
        v_render(L + done, R + done, chunk, &mix);
        done += chunk;
        samples += chunk;
        if (tr_state != TR_STOP) {
            phase += (float)chunk;
            if (step_in_beat < 2 && phase >= step_frac(step_in_beat + 1) * spb - 0.5f) {
                step_in_beat++;
                play_step(beat, step_in_beat);
            } else if (phase >= spb - 0.5f) {
                phase -= spb;
                if (phase < 0.0f)
                    phase = 0.0f;
                beat++;
                step_in_beat = 0;
                if (beat == 0 && tr_state == TR_COUNT)
                    tr_state = TR_PLAY;
                tr_beat = beat;
                play_step(beat, 0);
            }
        }
    }
    for (i = 0; i < n; i++) {
        float l = L[i] * gain, r = R[i] * gain;
        l = l > 8388607.0f ? 8388607.0f : l < -8388607.0f ? -8388607.0f : l;
        r = r > 8388607.0f ? 8388607.0f : r < -8388607.0f ? -8388607.0f : r;
        lr[2 * i] = (int32_t)l;
        lr[2 * i + 1] = (int32_t)r;
    }
    pub_beat = beat;
    pub_phase = phase;
    pub_spb = spb;
    pub_samples = samples;
}

/* ================================================================== the main loop */
void tr_key(int note, int vel)
{
    uint32_t w = key_qw;
    int line = vel ? teach_route && note >= LINE_LO && note <= LINE_HI : tch.key_line[note & 127];
    if (w - key_qr < 32u) {
        key_q[w & 31u] = (note_ev_t){(uint8_t)(note | (line ? (int)LINE_NOTE : 0)), (uint8_t)vel, pub_samples};
        key_qw = w + 1;
    }
    if (line) {                                 /* the player's line (teaching), not a chord's note */
        tch.key_line[note & 127] = (uint8_t)(vel != 0);
        line_in(note, vel, pub_samples);
        return;
    }
    if (vel) {
        held[note & 127] = (uint8_t)vel;
        pressed_at[note & 127] = pub_samples ? pub_samples : 1u;
        last_on = pub_samples ? pub_samples : 1u;
    } else
        held[note & 127] = 0;
}

static void reset_band(void)
{
    int i;
    for (i = 0; i < 8; i++)
        bass_ev[i].beat = INT32_MIN;
    for (i = 0; i < 32; i++)
        drum_ev[i].step = INT32_MIN;
    tnn_zero(bass_h, BASS_H);
    tnn_zero(drum_h, DRUM_H);
    bass_ptok = TNN_START;
    bass_pskip = 0;
    band_root = -1;
    band_since = 0;
    drum_phit = 0;
    for (i = 0; i < TNN_NPIECE; i++)
        drum_pvel[i] = 0;
    bass_next = 0;
    drum_next = -12;
}

void tr_forget(void)
{
    int i;
    ncommit = 0;
    hist_upto = -1;
    for (i = 0; i < HIST; i++)
        hist_root[i] = -1;
    form_lag = form_len = 0;
    net_root = -1;
    tnn_zero(harm_h, HARM_H);
    harm_prev_root = -1;
    harm_ok = 0;
}

static void teach_fresh(void)                  /* a start: the line's beats from the new beat 0 */
{
    int i;
    tch.started = 0;
    tch.pend_beat = INT32_MIN;
    for (i = 0; i < 4; i++)
        tch.slot[i].beat = INT32_MIN;
    for (i = 0; i <= LINE_HI; i++)
        tch.on_x[i] = tch.off_x[i] = -1e9f;
}

static int teach_end(void);

void tr_play(int on)
{
    if (on && tr_state == TR_STOP) {
        reset_band();
        tr_forget();
        teach_fresh();
        if (cur_root >= 0) {                    /* the chord already held counts from beat 0 */
            commits[0] = (commit_t){0, cur_root, cur_qual};
            ncommit = 1;
        }
        rng = seed;
        tr_missed = 0;
        want_play = 1;
    } else if (!on && tr_state != TR_STOP) {
        if (tch.on)                             /* teaching: STOP learns the line */
            teach_end();
        want_stop = 1;
    }
}

void tr_seed(uint32_t s) { seed = s ? s : 1u; }

void tr_panic(void)
{
    want_stop = 1;
    v_panic();
}

/* the beat (fractional) at sample t, from what the interrupt published */
static float beat_at(uint32_t t)
{
    return (float)pub_beat + (pub_phase + (float)(int32_t)(t - pub_samples)) / pub_spb;
}

static int32_t commit_beat(uint32_t t)          /* the beat a chord played at sample t counts for */
{
    int32_t b = (int32_t)flr(beat_at(t) + TR_EARLY);
    if (b < 0)
        b = 0;
    if (b < bass_next - 1)                      /* too late for any beat decided: from the next */
        b = bass_next;
    return b;
}

/* ------------------------------------------------------------------ teaching the bass (the main loop's) */
static slot_t *slot_of(int32_t b)
{
    slot_t *s = &tch.slot[b & 3];
    if (s->beat != b) {
        s->beat = b;
        s->note = -1;
        s->skip = 0;
        s->df = 0.0f;
    }
    return s;
}

/* a note of the player's line, pressed (vel) or let go at sample t. While the band plays, a note falls to the beat it
 * is nearest after TEACH_EARLY: up to TEACH_PICKUP into it, the beat's note (the nearest the beat; two within 30 ms,
 * a double stop: the lower); later, or after the beat's note, a pickup before the next beat */
static void line_in(int note, int vel, uint32_t t)
{
    float x = beat_at(t), f, d, tol;
    int32_t b;
    slot_t *s;
    if (note < LINE_LO || note > LINE_HI)
        return;
    if (!vel) {
        tch.down[note] = 0;
        tch.off_x[note] = x;
        return;
    }
    tch.down[note] = 1;
    tch.on_x[note] = x;
    if (!tch.on || tr_state == TR_STOP)
        return;
    b = (int32_t)flr(x + TEACH_EARLY);
    if (b < 0)
        return;
    f = x - (float)b;
    d = f < 0.0f ? -f : f;
    tol = 0.03f * SR / pub_spb;
    s = slot_of(b);
    if (f >= TEACH_PICKUP)
        s->skip = 1;
    else if (s->note < 0 || d < s->df - tol) {   /* the first, or nearer the beat (the one before it: a grace note) */
        s->note = (int8_t)note;
        s->df = d;
    } else if (d <= s->df + tol) {
        if (note < s->note)
            s->note = (int8_t)note;
    } else
        s->skip = 1;
}

static int sounding_at(int32_t b)               /* a note of the line pressed before beat b, still held on it */
{
    int n;
    for (n = LINE_LO; n <= LINE_HI; n++)
        if (tch.on_x[n] < (float)b && (tch.down[n] || tch.off_x[n] > (float)b))
            return 1;
    return 0;
}

/* what the player played on beat b, its windows gone: its note's token, or a hold (a note still sounding), or a
 * rest; and a pickup after it */
static void teach_close(int32_t b, int8_t *tok, uint8_t *skip)
{
    const slot_t *s = &tch.slot[b & 3];
    int has = s->beat == b;
    *tok = (int8_t)(has && s->note >= 0 ? 2 + s->note - TNN_LO : sounding_at(b) ? TNN_HOLD : TNN_REST);
    *skip = (uint8_t)(has && s->skip);
}

static void teach_keep(int8_t tok, uint8_t skip)   /* the beat decided last, closed: kept if it teaches something */
{
    tch.pend.tok = tok;
    tch.pend.skip = skip;
    VIZ("T %d %d %d\n", (int)tch.pend_beat, tok, skip);
    if (bass_learnable(&tch.pend)) {
        teach_ring[tch.w] = tch.pend;
        tch.w = (tch.w + 1) % TEACH_MAX;
        if (tch.n < TEACH_MAX)
            tch.n++;
    }
#if defined(TNN_HOST) || defined(PLAT_HOST)
    teach_log[tch.pend_beat & (LOGN - 1)] = tok;
    teach_log_skip[tch.pend_beat & (LOGN - 1)] = skip;
    teach_log_beat[tch.pend_beat & (LOGN - 1)] = tch.pend_beat;
    teach_last = tch.pend_beat;
#endif
    tch.pend_beat = INT32_MIN;
}

/* teaching off: the last beat kept if its windows have gone (its note the bass's last either way), then what was
 * kept learned, if it has notes enough. The beats it learns (0: too few, dropped) */
static int teach_end(void)
{
    int k, notes = 0;
    teach_route = 0;
    tch.on = 0;
    if (tch.pend_beat != INT32_MIN) {
        int8_t pt;
        uint8_t ps;
        teach_close(tch.pend_beat, &pt, &ps);
        bass_ptok = pt;
        bass_pskip = ps;
        if (beat_at(pub_samples) >= (float)tch.pend_beat + 1.0f - TEACH_EARLY)
            teach_keep(pt, ps);
        tch.pend_beat = INT32_MIN;
    }
    for (k = 0; k < tch.n; k++)
        notes += bass_learnable(&teach_ring[(tch.w - tch.n + k + TEACH_MAX) % TEACH_MAX]) & 1;
    VIZ("E 0 %u %d\n", (unsigned)pub_samples, notes < TEACH_MIN ? 0 : tch.n);
    if (notes < TEACH_MIN) {
        tch.n = tch.w = 0;
        return 0;
    }
    bass_learn_begin(teach_ring, TEACH_MAX, (tch.w - tch.n + TEACH_MAX) % TEACH_MAX, tch.n, LRN_RATE, LRN_PULL,
                     LRN_PASSES);
    tch.learning = 1;
    VIZ("L 0 %u %d\n", (unsigned)pub_samples, tch.n);
    return tch.n;
}

int tr_teach(int on)
{
    if (!on)
        return tch.on ? teach_end() : 0;
    if (tch.on)
        return 1;
    if (tch.learning)                           /* still learning the last line */
        return 0;
    teach_fresh();
    tch.n = tch.w = 0;
    tch.on = 1;
    teach_route = 1;
    VIZ("E 1 %u\n", (unsigned)pub_samples);
    return 1;
}

/* the learning, a slice a pass of the main loop; done, in use from the next beat the bass decides */
static int learn_slice(void)
{
    int i;
    if (!tch.learning)
        return 0;
    for (i = 0; i < LRN_SLICE; i++)
        if (bass_learn_step() == TNN_LEARN_DONE) {
            bass_learn_commit();
            tch.learning = 0;
            tch.n = tch.w = 0;
            tch.learned_at = bass_next;
            tr_learn_count++;
#if defined(TNN_HOST) || defined(PLAT_HOST)
            if (viz_file()) {
                FILE *f = viz_file();
                int p;
                fprintf(f, "L 1 %u %d", (unsigned)pub_samples, LRN_PASSES);
                for (p = 0; p < LRN_PASSES; p++)
                    fprintf(f, " %.4f", (double)bass_learn_loss(p, 0));
                fprintf(f, "\n");
                fflush(f);
            }
#endif
            break;
        }
    return 1;
}

void tr_unlearn(void)                           /* the bass as trained: a lesson, a learning dropped too */
{
    bass_learn_cancel();
    tch.learning = 0;
    tch.on = 0;
    teach_route = 0;
    tch.n = tch.w = 0;
    tch.pend_beat = INT32_MIN;
    bass_unlearn();
    tr_learn_count++;
    VIZ("L 2 %u\n", (unsigned)pub_samples);
}

int tr_teaching(void) { return tch.on; }
int tr_learning(void) { return tch.learning; }
int tr_taught(void) { return tch.n; }
int tr_learned(void) { return bass_learned(); }
const void *tr_learned_obj(uint32_t *len) { return bass_learned_obj(len); }
void *tr_learned_room(uint32_t *max)
{
    bass_learn_cancel();
    tch.learning = 0;
    return bass_learned_room(max);
}
int tr_learned_check(int len) { return bass_learned_check(len); }

static int form_says(int32_t b, int8_t *root, int8_t *qual);
int tr_form_bars(void);

/* the chord at beat b as the band plays it: the player's last chord at or before b, until the form moves on (a change
 * in the form after that chord, which the player let go by: from there on, the form's). So the band keeps to the form
 * when the player stops or plays on the beat, and stays with a chord the player puts against it. 0: no chord yet */
static int chord_at(int32_t b, int8_t *root, int8_t *qual)
{
    int i, n;
    int32_t x;
    int8_t fr, fq, gr, gq;
    for (i = ncommit - 1; i >= 0 && commits[i].beat > b; i--)
        ;
    if (i < 0)
        return 0;
    *root = commits[i].root;
    *qual = commits[i].qual;
    if (commits[i].beat == b || !form_says(b, &fr, &fq))
        return 1;
    for (x = b, n = 0; x > commits[i].beat && n < form_lag; x--, n++)   /* (one round of the form is enough) */
        if (form_says(x - 1, &gr, &gq) && form_says(x, &fr, &fq) && (fr != gr || fq != gq)) {
            form_says(b, root, qual);
            return 1;
        }
    return 1;
}

static void commit(int8_t root, int8_t qual, uint32_t t)
{
    int32_t b;
    float h[12 + TNN_NQUAL + TNN_NDUR];
    harm_in_t hi;
    int k;
    cur_root = root;
    cur_qual = qual;
    if (tr_state == TR_STOP)
        return;
    b = commit_beat(t);
    {   /* a chord the form did not foresee (not its chord at b, nor the next beat's: played early) proves it wrong:
         * it is dropped, and the beats the band took from it since the player's last chord go back to that chord
         * in the history (the player held it), so that the same wrong form is not found again */
        int8_t fr, fq, gr, gq;
        if (form_says(b, &fr, &fq) && (fr != root || fq != qual) &&
            !(form_says(b + 1, &gr, &gq) && gr == root && gq == qual)) {
            int i;
            int32_t x;
            for (i = ncommit - 1; i >= 0 && commits[i].beat >= b; i--)
                ;
            if (i >= 0)
                for (x = commits[i].beat + 1; x < b && x <= hist_upto; x++)
                    if (hist_upto - x < HIST) {
                        hist_root[x % HIST] = commits[i].root;
                        hist_qual[x % HIST] = commits[i].qual;
                    }
            form_lag = form_len = 0;
            form_drops++;
            VIZ("F %d 0\n", (int)b);              /* (the demo's log: the form dropped) */
        }
    }
    if (ncommit && commits[ncommit - 1].beat >= b) {   /* replaces one for the same beat */
        commits[ncommit - 1] = (commit_t){b, root, qual};
    } else {
        if (ncommit == NCOMMIT) {
            for (k = 1; k < NCOMMIT; k++)
                commits[k - 1] = commits[k];
            ncommit--;
        }
        commits[ncommit++] = (commit_t){b, root, qual};
    }
    /* the network's guess of what comes next, and when */
    hi.ivl = (uint8_t)(harm_prev_root >= 0 ? (root - harm_prev_root + 12) % 12 : 0);
    hi.qual = qual;
    {
        int32_t d = b - harm_last_beat;
        static const int8_t DUR[TNN_NDUR] = {1, 2, 3, 4, 6, 8, 12, 16};
        int best = 3, bd = 1000;
        for (k = 0; k < TNN_NDUR; k++) {
            int e = d - DUR[k] < 0 ? DUR[k] - d : d - DUR[k];
            if (e < bd)
                bd = e, best = k;
        }
        hi.pdur = (uint8_t)best;
    }
    hi.start = (uint8_t)(b & 3);
    hi.first = harm_prev_root < 0;
    harm_step(harm_h, &hi, h, h + 12, h + 12 + TNN_NQUAL);
    nn_steps++;
    {
        static const int8_t DUR[TNN_NDUR] = {1, 2, 3, 4, 6, 8, 12, 16};
        int ri = tnn_argmax(h, 0, 12), qi = tnn_argmax(h + 12, 0, TNN_NQUAL), di = tnn_argmax(h + 12 + TNN_NQUAL, 0, TNN_NDUR);
        float mx = h[ri], sum = 0.0f;
        for (k = 0; k < 12; k++)
            sum += fm_exp2f((h[k] - mx) * FM_LOG2E);
        net_conf = 1.0f / sum;
        {                                       /* log-softmax of both, kept for listening */
            float m2 = h[12 + qi], s2 = 0.0f;
            for (k = 0; k < TNN_NQUAL; k++)
                s2 += fm_exp2f((h[12 + k] - m2) * FM_LOG2E);
            for (k = 0; k < 12; k++)
                harm_lr[k] = h[k] - mx - fm_logf(sum);
            for (k = 0; k < TNN_NQUAL; k++)
                harm_lq[k] = h[12 + k] - m2 - fm_logf(s2);
            harm_ok = 1;
        }
        net_root = (int8_t)((root + ri) % 12);
        net_qual = (int8_t)qi;
        net_at = b + DUR[di];
    }
    harm_prev_root = root;
    harm_last_beat = b;
}

/* listening: a chord from the notes pressed lately (held, or let go less than 120 ms ago), 25 ms after the last */
static void listen(void)
{
    chord_in_t in;
    float lg[12 * TNN_NQUAL], mx, sum = 0.0f;
    uint32_t now = pub_samples;
    int k, n = 0, low = 128, best, same;
    if (!last_on || (int32_t)(now - last_on) < (int32_t)(0.025f * SR))
        return;
    last_on = 0;
    for (k = 0; k < 12; k++)
        in.pc[k] = 0.0f;
    for (k = 0; k < 128; k++)
        if (held[k] || (pressed_at[k] && (int32_t)(now - pressed_at[k]) < (int32_t)(0.12f * SR))) {
            in.pc[k % 12] += held[k] ? (float)held[k] * (1.0f / 64.0f) : 0.8f;
            if (k < low)
                low = k;
            n++;
        }
    if (n < 2)
        return;
    in.low = (int8_t)(low % 12);
    in.proot = cur_root;
    in.pqual = cur_qual;
    chord_eval(&in, lg);
    nn_steps++;
    if (harm_ok && cur_root >= 0 && tr_state != TR_STOP)   /* what usually follows breaks ties between readings */
        for (k = 0; k < 12 * TNN_NQUAL; k++)
            lg[k] += HARM_PRIOR * (harm_lr[(k / TNN_NQUAL - cur_root + 12) % 12] + harm_lq[k % TNN_NQUAL]);
    best = tnn_argmax(lg, 0, 12 * TNN_NQUAL);
    mx = lg[best];
    for (k = 0; k < 12 * TNN_NQUAL; k++)
        sum += fm_exp2f((lg[k] - mx) * FM_LOG2E);
    same = best / TNN_NQUAL == cur_root && best % TNN_NQUAL == cur_qual;
    if (1.0f / sum < 0.2f)                      /* not sure enough: keep the chord we have */
        return;
    if (!same && cur_root >= 0) {               /* and clearly better than the chord we have */
        float pc = fm_exp2f((lg[cur_root * TNN_NQUAL + cur_qual] - mx) * FM_LOG2E) / sum;
        if (1.0f / sum < pc + 0.15f)
            return;
    }
    {
        uint32_t first = now;                   /* the chord's time: its earliest note's */
        int8_t r, q;
        for (k = 0; k < 128; k++)
            if (pressed_at[k] && (int32_t)(now - pressed_at[k]) < (int32_t)(0.25f * SR) &&
                (int32_t)(first - pressed_at[k]) > 0)
                first = pressed_at[k];
        /* the same chord again is news only when the band has gone on without it (the form moved): it stays */
        if (same && (tr_state == TR_STOP || !chord_at(commit_beat(first), &r, &q) || (r == cur_root && q == cur_qual)))
            return;
#if defined(TNN_HOST) || defined(PLAT_HOST)
        if (viz_file()) {
            FILE *f = viz_file();
            fprintf(f, "H %u %d %d %.3f", (unsigned)first, best / TNN_NQUAL, best % TNN_NQUAL, (double)(1.0f / sum));
            for (k = 0; k < 128; k++)
                if (held[k] || (pressed_at[k] && (int32_t)(now - pressed_at[k]) < (int32_t)(0.12f * SR)))
                    fprintf(f, " %d", k);
            fprintf(f, "\n");
            fflush(f);
        }
#endif
        commit((int8_t)(best / TNN_NQUAL), (int8_t)(best % TNN_NQUAL), first);
    }
}

static void notes_in(void)                      /* the interrupt's MIDI notes into the held set (or the line) */
{
    while (midi_qr != midi_qw) {
        note_ev_t e = midi_q[midi_qr & 63u];
        midi_qr++;
        if (e.note & LINE_NOTE)
            line_in(e.note & 127, e.vel, e.t);
        else if (e.vel) {
            held[e.note] = e.vel;
            pressed_at[e.note] = e.t ? e.t : 1u;
            last_on = e.t ? e.t : 1u;
        } else
            held[e.note] = 0;
    }
}

/* the form: the lag (whole bars) at which the chords before beat b repeat the longest */
static void find_form(int32_t b)
{
    int L, best = 0, blen = 0;
    for (L = 16; L <= 256 && L <= b; L += 4) {
        int m = 0;
        while (m < 64 && b - 1 - m - L >= 0 && hist_root[(b - 1 - m) % HIST] >= 0 &&
               hist_root[(b - 1 - m) % HIST] == hist_root[(b - 1 - m - L) % HIST] &&
               hist_qual[(b - 1 - m) % HIST] == hist_qual[(b - 1 - m - L) % HIST])
            m++;
        if (m > blen + 2 || (m > blen && L % 16 == 0)) {
            blen = m;
            best = L;
        }
    }
    form_lag = best;
    form_len = blen;
}

static int form_says(int32_t b, int8_t *root, int8_t *qual)   /* the form's chord at beat b (0: it can't say) */
{
    int32_t src;
    if (!tr_set.follow || form_lag == 0 || form_len < 8)
        return 0;
    src = b - form_lag;
    while (src > hist_upto && src >= 0)
        src -= form_lag;
    if (src < 0 || hist_root[src % HIST] < 0)
        return 0;
    *root = hist_root[src % HIST];
    *qual = hist_qual[src % HIST];
    return 1;
}

static int feel_at(int32_t b)
{
    if (tr_set.feel == TR_TWO)
        return TR_TWO;
    if (tr_set.feel == TR_FOUR)
        return TR_FOUR;
    return (form_len >= 8 && b >= form_lag) || b >= 64 ? TR_FOUR : TR_TWO;   /* AUTO: the head in two */
}

/* teaching, at beat b's decision: the beat decided before it closed (and kept), and the player's note on the beat
 * before b the bass's last, as if it had played it */
static void teach_step_in(int32_t b)
{
    int8_t pt;
    uint8_t ps;
    if (tch.pend_beat != INT32_MIN && tch.pend_beat < b) {
        teach_close(tch.pend_beat, &pt, &ps);
        teach_keep(pt, ps);
    }
    if (tch.started) {
        teach_close(b - 1, &pt, &ps);
        bass_ptok = pt;
        bass_pskip = ps;
    }
#if defined(TNN_HOST) || defined(PLAT_HOST)
    else {                                      /* a lesson's first beat: the state it starts from */
        int k;
        for (k = 0; k < BASS_H; k++)
            teach_h0[k] = bass_h[k];
        teach_first = b;
        teach_last = b - 1;
    }
#endif
    tch.started = 1;
}

#if defined(TNN_HOST) || defined(PLAT_HOST)
/* the demo's log of a bass beat: the tokens' chances as tnn_sample draws them (the logits shaped, at temperature) */
static void viz_bass(int32_t b, const bass_in_t *in, const float *lg, const uint8_t *allow, float temp, int tok,
                     int skip, int teaching)
{
    FILE *f = viz_file();
    float p[TNN_NTOK], mx = -1e30f, sum = 0.0f;
    int k;
    if (!f)
        return;
    for (k = 0; k < TNN_NTOK; k++)
        if (allow[k] && lg[k] > mx)
            mx = lg[k];
    for (k = 0; k < TNN_NTOK; k++) {
        p[k] = allow[k] ? fm_exp2f((lg[k] - mx) * (FM_LOG2E / temp)) : 0.0f;
        sum += p[k];
    }
    fprintf(f, "B %d %.0f %d %d %d %d %d %d %d %d %d %d", (int)b,
            (double)pub_samples + ((double)(b - pub_beat) * (double)pub_spb - (double)pub_phase), teaching, in->root,
            in->qual, in->nroot, next_how ? next_qual : -1, in->tochg, in->ptok, tok, skip, tr_set.freedom);
    for (k = 0; k < TNN_NTOK; k++)
        fprintf(f, " %.4f", (double)(p[k] / sum));
    fprintf(f, "\n");
    fflush(f);
}
#endif

static void decide_bass(int32_t b)
{
    bass_in_t in;
    float lg[TNN_NTOK], sk, temp;
    uint8_t allow[TNN_NTOK];
    int8_t r, q, fr, fq;
    int k, tok, two, change, flags;
    bass_ev_t *e = &bass_ev[b & 7];
    /* the chords so far, as the band played them, into the history (beat b-1 is final now: a chord played on it
     * counts, though the bass had gone; the form's own chords count where the player let it lead) */
    while (hist_upto < b - 1) {
        hist_upto++;
        if (chord_at(hist_upto, &r, &q)) {
            hist_root[hist_upto % HIST] = r;
            hist_qual[hist_upto % HIST] = q;
        } else
            hist_root[hist_upto % HIST] = -1;
    }
    if ((b & 3) == 0) {
#if defined(TNN_HOST) || defined(PLAT_HOST)
        int was = tr_form_bars();
#endif
        find_form(b);
#if defined(TNN_HOST) || defined(PLAT_HOST)
        if (tr_form_bars() != was)              /* (the demo's log: the form learned, or dropped) */
            VIZ("F %d %d\n", (int)b, tr_form_bars());
#endif
    }
    if (tch.on)
        teach_step_in(b);
    /* the chord for this beat: the player's, or the form's once it has moved on (a change the player has not
     * played yet: on the beat, or not at all) */
    if (!chord_at(b, &r, &q)) {
        e->tok = tch.on ? TNN_HOLD : TNN_REST;  /* (teaching: the player's note left alone) */
        e->skip = 0;
        e->vel = 0.0f;
        e->beat = b;
        return;
    }
    change = r != band_root || q != band_qual;
    band_since = change ? 0 : band_since + 1;
    band_root = r;
    band_qual = q;
    in.ptok = bass_ptok;
    in.root = r;
    in.qual = q;
    in.mask = chord_mask(r, q);
    in.since = (uint8_t)(band_since > 7 ? 7 : band_since);
    in.bib = (uint8_t)(b & 3);
    two = feel_at(b) == TR_TWO;
    in.feel = (uint8_t)two;
    in.pskip = bass_pskip;
    /* what comes next: the form's, else the network's guess if it is sure enough */
    in.nroot = -1;
    in.nmask = 0;
    in.tochg = 8;
    next_how = 0;
    for (k = 1; k <= 8; k++)
        if (form_says(b + k, &fr, &fq)) {
            if (fr != r || fq != q) {
                in.nroot = fr;
                in.nmask = chord_mask(fr, fq);
                in.tochg = (uint8_t)k;
                next_root = fr;
                next_qual = fq;
                next_in = k;
                next_how = 1;
                break;
            }
        } else
            break;
    if (!next_how && net_root >= 0 && net_conf > 0.35f && net_at > b) {
        int32_t d = net_at - b;
        in.nroot = net_root;
        in.nmask = chord_mask(net_root, net_qual);
        in.tochg = (uint8_t)(d > 8 ? 8 : d);
        next_root = net_root;
        next_qual = net_qual;
        next_in = d;
        next_how = 2;
    }
#if defined(TNN_HOST) || defined(PLAT_HOST)
    if (tch.on) {
        teach_in[b & (LOGN - 1)] = in;
        teach_in_beat[b & (LOGN - 1)] = b;
    }
#endif
    bass_step(bass_h, &in, lg, &sk);
    nn_steps++;
#ifdef TNN_HOST
    if (getenv("BASS_TRACE") && b >= atoi(getenv("BASS_TRACE")) && b < atoi(getenv("BASS_TRACE")) + 16)
        printf("    b%d ptok %d root %d qual %d since %d bib %d feel %d pskip %d next %d/%d in %d  change %d\n", (int)b,
               in.ptok, in.root, in.qual, in.since, in.bib, in.feel, in.pskip, in.nroot, in.nmask, in.tochg, change);
#endif
    flags = (change ? TNN_B_CHANGE : 0) | (two ? TNN_B_TWO : 0) | ((b & 1) ? TNN_B_ODD : 0);
    if (tch.on) {                               /* teaching: the bass quiet, the beat kept for the player's note on it */
        for (k = 0; k < BASS_H; k++) {
            float v = bass_h[k] * 32768.0f;
            int32_t x = (int32_t)(v + (v < 0.0f ? -0.5f : 0.5f));
            tch.pend.h[k] = (int16_t)(x > 32767 ? 32767 : x < -32768 ? -32768 : x);
        }
        tch.pend.root = r;
        tch.pend.mask = in.mask;
        tch.pend.flags = (uint8_t)flags;
        tch.pend.tok = -1;
        tch.pend.skip = 0;
        tch.pend_beat = b;
#if defined(TNN_HOST) || defined(PLAT_HOST)
        if (viz_file()) {                       /* what the network would have played here */
            float lg2[TNN_NTOK];
            uint8_t al2[TNN_NTOK];
            for (k = 0; k < TNN_NTOK; k++)
                lg2[k] = lg[k];
            bass_shape(lg2, al2, r, in.mask, flags);
            viz_bass(b, &in, lg2, al2, 0.30f + 0.012f * (float)tr_set.freedom, -1, 0, 1);
        }
#endif
        e->tok = TNN_HOLD;
        e->skip = 0;
        e->vel = 0.0f;
        e->beat = b;
        bass_log_beat[b & (LOGN - 1)] = INT32_MIN;   /* (the bass played nothing) */
        return;
    }
    /* what may be played (four: a note every beat, E1..G3; two: notes on 1 and 3, or a change, holds between) and
     * the nudges: the chord's notes on a change, the line kept low */
    bass_shape(lg, allow, r, in.mask, flags);
    temp = 0.30f + 0.012f * (float)tr_set.freedom;
    tok = tnn_sample(lg, allow, TNN_NTOK, temp, &rng);
    e->tok = (int8_t)tok;
    e->skip = (uint8_t)((two ? (b & 1) != 0 : tok >= 2) &&
                        tnn_unit(&rng) < tnn_sigmoid(sk) * (0.6f + 0.008f * (float)tr_set.energy));
    e->vel = 0.72f + ((b & 1) ? 0.08f : 0.0f) + 0.08f * tnn_unit(&rng);
#if defined(TNN_HOST) || defined(PLAT_HOST)
    viz_bass(b, &in, lg, allow, temp, tok, e->skip, 0);
#endif
    e->beat = b;                                /* (last: the interrupt reads the beat to know it's ready) */
    bass_ptok = (int8_t)tok;
    bass_pskip = e->skip;
    bass_log[b & (LOGN - 1)] = (int8_t)tok;
    chord_log[b & (LOGN - 1)] = (int8_t)(r * TNN_NQUAL + q);
#if defined(TNN_HOST) || defined(PLAT_HOST)
    bass_log_skip[b & (LOGN - 1)] = e->skip;
#endif
    bass_log_beat[b & (LOGN - 1)] = b;
}

static int sec_now(int32_t bar)
{
    int e = tr_set.energy;
    if (tr_set.feel == TR_AUTO && feel_at(bar * 4) == TR_TWO)
        return SEC_HEAD;
    return e < 35 ? SEC_BASS : e < 70 ? SEC_PIANO : SEC_HORN;
}

static void decide_drums(int32_t g)
{
    drum_in_t in;
    float hit[TNN_NPIECE], vel[3 * TNN_NPIECE];
    drum_ev_t *d = &drum_ev[g & 31];
    int32_t bar = (g >= 0 ? g : g - 11) / 12;
    int p;
    uint16_t hits = 0;
    in.phit = drum_phit;
    for (p = 0; p < TNN_NPIECE; p++)
        in.pvel[p] = drum_pvel[p];
    in.step = (uint8_t)(((g % 12) + 12) % 12);
    in.phr = (uint8_t)(((bar % 8) + 8) % 8);
    in.sec = (uint8_t)sec_now(bar);
    in.tempo = (uint8_t)(tr_set.bpm < 110 ? 0 : tr_set.bpm < 160 ? 1 : tr_set.bpm < 210 ? 2 : 3);
    in.energy = 0.45f + 0.006f * (float)tr_set.energy;
    drum_step(drum_h, &in, hit, vel);
    nn_steps++;
    for (p = 0; p < TNN_NPIECE; p++) {
        if (tnn_unit(&rng) < tnn_sigmoid(hit[p] * 1.1f)) {
            hits |= (uint16_t)(1u << p);
            d->lvl[p] = (uint8_t)(1 + tnn_sample(vel + 3 * p, 0, 3, 0.8f, &rng));
        } else
            d->lvl[p] = 0;
        drum_pvel[p] = d->lvl[p];
    }
    drum_phit = hits;
    d->hits = hits;
#if defined(TNN_HOST) || defined(PLAT_HOST)
    VIZ("D %d %.0f %u\n", (int)g, (double)pub_samples + (((double)(g / 3 - pub_beat) + (double)step_frac(g % 3)) *
                                       (double)pub_spb - (double)pub_phase), (unsigned)hits);
#endif
    d->step = g;                                /* (last) */
    drum_log[g & (2 * LOGN - 1)] = hits;
    drum_log_step[g & (2 * LOGN - 1)] = g;
}

int tr_service(void)
{
    int32_t pb;
    float ph, sp;
    notes_in();
    listen();
    if (tr_state == TR_STOP)
        return learn_slice();                   /* (stopped: the line learned meanwhile) */
    pb = pub_beat;
    ph = pub_phase;
    sp = pub_spb;
    /* the bass first (it has a deadline): beat b once it is less than TR_LEAD_MS away */
    {
        float until = ((float)(bass_next - pb)) * sp - ph;
        if (bass_next <= pb)
            bass_next = pb + 1;                 /* late: that beat has gone */
        else if (until < (float)TR_LEAD_MS * 0.001f * SR) {
            decide_bass(bass_next++);
            return 1;
        }
    }
    /* the drummer: a few steps ahead of the step sounding (from beat 0; the count-in is its own) */
    {
        int32_t cur = pb * 3 + (ph >= step_frac(2) * sp ? 2 : ph >= step_frac(1) * sp ? 1 : 0);
        if (drum_next < 0)
            drum_next = 0;
        if (drum_next <= cur)
            drum_next = cur + 1;
        if (drum_next <= cur + DRUM_AHEAD) {
            decide_drums(drum_next++);
            return 1;
        }
    }
    return learn_slice();                       /* nothing due: a slice of the line being learned */
}

/* ------------------------------------------------------------------ for the screen and the tests */
int tr_chord_now(int *root, int *qual)
{
    if (tr_state != TR_STOP && band_root >= 0) {
        *root = band_root;
        *qual = band_qual;
        return 1;
    }
    if (cur_root < 0)
        return 0;
    *root = cur_root;
    *qual = cur_qual;
    return 1;
}
int tr_chord_next(int *root, int *qual, int *in_beats)
{
    if (tr_state == TR_STOP || !next_how)
        return 0;
    *root = next_root;
    *qual = next_qual;
    *in_beats = next_in;
    return next_how;
}
int tr_form_bars(void) { return form_len >= 8 ? form_lag / 4 : 0; }
int tr_form_debug(int *lag, int *len)
{
    *lag = form_lag;
    *len = form_len;
    return (int)form_drops;
}
int tr_bass_note(void) { return bass_last; }
int tr_feel_now(void) { return feel_at(tr_beat < 0 ? 0 : tr_beat); }
uint32_t tr_nn_steps(void) { return nn_steps; }
int tr_log_chord(int32_t b) { return bass_log_beat[b & (LOGN - 1)] == b ? chord_log[b & (LOGN - 1)] : -1; }
int tr_log_bass(int32_t b) { return bass_log_beat[b & (LOGN - 1)] == b ? bass_log[b & (LOGN - 1)] : -1; }
uint16_t tr_log_drums(int32_t g)
{
    return drum_log_step[g & (2 * LOGN - 1)] == g ? drum_log[g & (2 * LOGN - 1)] : 0;
}

/* teaching, for the tests: what the player played on beat b (-1: not taught), the last lesson's beats, the beat it
 * came into use from; and how often the bass's notes on beats from..to-1 were the player's on the same beat of the
 * form (lag beats), in % of those that have a taught note there (-1: none) */
#if defined(TNN_HOST) || defined(PLAT_HOST)
int tr_log_taught(int32_t b)
{
    return teach_log_beat[b & (LOGN - 1)] == b ? teach_log[b & (LOGN - 1)] : -1;
}
int tr_teach_span(int32_t *first, int32_t *last)
{
    *first = teach_first;
    *last = teach_last;
    return teach_first >= 0 && teach_last >= teach_first;
}
int32_t tr_learned_at(void) { return tch.learned_at; }
int tr_log_skip(int32_t b) { return bass_log_beat[b & (LOGN - 1)] == b ? bass_log_skip[b & (LOGN - 1)] : -1; }
const bass_in_t *tr_teach_in(int32_t b, int *skip)
{
    *skip = teach_log_beat[b & (LOGN - 1)] == b ? teach_log_skip[b & (LOGN - 1)] : 0;
    return teach_in_beat[b & (LOGN - 1)] == b ? &teach_in[b & (LOGN - 1)] : 0;
}
const float *tr_teach_h0(void) { return teach_h0; }
int tr_teach_bytes(void) { return (int)(sizeof teach_ring + sizeof tch); }
int tr_agree(int32_t from, int32_t to, int lag)
{
    int32_t b, t;
    int n = 0, same = 0;
    if (lag <= 0 || teach_first < 0)
        return -1;
    for (b = from; b < to; b++) {
        int bt = tr_log_bass(b), want = -1;
        for (t = teach_first + (((b - teach_first) % lag) + lag) % lag; t <= teach_last; t += lag)
            if (tr_log_taught(t) >= 2)          /* the latest note taught on that beat of the form */
                want = tr_log_taught(t);
        if (want < 0 || bt < 0)
            continue;
        n++;
        same += bt == want;
    }
    return n ? (100 * same + n / 2) / n : -1;
}
#endif
