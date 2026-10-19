/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's four networks (trained by tools/trio_train.py, weights in trio_w.h):
 *
 *   CHORD  the player's notes -> the chord (12 roots x 7 kinds)       MLP 44-64-64-84, when the notes change
 *   HARM   a chord -> the next one's root and kind, this one's length  GRU 48, a chord a step
 *   BASS   the walking line: a note (or hold, rest) a beat, a pickup   GRU 96, a beat a step
 *   DRUM   the drummer: 9 pieces and how hard, a triplet eighth a step GRU 64, 12 steps a bar
 *
 * int8 weights, one scale a matrix, float maths (fastmath.h, no libm). Each model's input is a few one-hots and
 * numbers (the structs below), summed into its first layer column by column. Pure C99; runs in the main loop,
 * never in the audio interrupt. */
#pragma once
#include <stdint.h>

#define TNN_LO 28                           /* the bass's lowest token: E1; tokens are 0 rest, 1 hold, then 2 + pitch - TNN_LO */
#define TNN_HI 60                           /* C4 */
#define TNN_REST 0
#define TNN_HOLD 1
#define TNN_NTOK (2 + TNN_HI - TNN_LO + 1)  /* 35 */
#define TNN_START TNN_NTOK                  /* the token "before the first beat" */
#define TNN_NQUAL 7                         /* chord kinds */
enum { Q_MAJ, Q_DOM, Q_MIN, Q_HDIM, Q_DIM, Q_SUS, Q_AUG };
#define TNN_NPIECE 9                        /* drum pieces */
enum { P_RIDE, P_SNARE, P_KICK, P_HHF, P_HHO, P_XSTK, P_TOMH, P_TOML, P_CRASH };
enum { SEC_INTRO, SEC_HEAD, SEC_HORN, SEC_PIANO, SEC_BASS, SEC_DRUMS, SEC_OUTRO };
#define TNN_NDUR 8                          /* chord lengths: 1 2 3 4 6 8 12 16 beats */
#define BASS_H 96
#define DRUM_H 64
#define HARM_H 48

typedef struct {
    int8_t ptok;          /* the token played on the beat before (TNN_START at the start) */
    int8_t root;          /* the chord now: root 0..11 (-1 none), its pitch classes, its kind */
    uint16_t mask;
    int8_t qual;
    int8_t nroot;         /* the next chord (-1 not known) */
    uint16_t nmask;
    uint8_t tochg;        /* beats to it, 1..8 (8: eight or more, or not known) */
    uint8_t since;        /* beats since this chord came, 0..7 */
    uint8_t bib;          /* the beat in the bar, 0..3 */
    uint8_t feel;         /* 0 four, 1 two */
    uint8_t pskip;        /* a pickup was played before this beat */
} bass_in_t;

typedef struct {
    uint16_t phit;        /* the step before: its pieces (bit p), and their levels 1..3 */
    uint8_t pvel[TNN_NPIECE];
    uint8_t step;         /* 0..11 in the bar */
    uint8_t phr;          /* the bar in an 8-bar phrase */
    uint8_t sec;          /* SEC_* */
    uint8_t tempo;        /* 0 < 110, 1 < 160, 2 < 210, 3 faster */
    float energy;         /* the hits a bar it aims at, / 16 */
} drum_in_t;

typedef struct {
    uint8_t ivl;          /* this chord's root over the one before, 0..11 */
    int8_t qual;
    uint8_t pdur;         /* the one before: its length class */
    uint8_t start;        /* the beat this one starts on, 0..3 */
    uint8_t first;        /* nothing before it */
} harm_in_t;

typedef struct {
    float pc[12];         /* how much of each pitch class (any scale: the largest counts as 1) */
    int8_t low;           /* the lowest note's class */
    int8_t proot, pqual;  /* the chord before (-1: none) */
} chord_in_t;

void tnn_init(void);                        /* the weights into RAM (at power-on) */
void tnn_zero(float *h, int n);             /* a fresh state */
void bass_step(float *h, const bass_in_t *in, float *tok_logits, float *skip_logit);
void drum_step(float *h, const drum_in_t *in, float *hit_logits, float *vel_logits);
void harm_step(float *h, const harm_in_t *in, float *root_logits, float *qual_logits, float *dur_logits);
void chord_eval(const chord_in_t *in, float *logits);       /* 84: root * 7 + kind */

/* the same with the input as a dense vector (tests: the golden vectors are dense) */
void bass_step_x(float *h, const float *x, float *tok_logits, float *skip_logit);
void drum_step_x(float *h, const float *x, float *hit_logits, float *vel_logits);
void harm_step_x(float *h, const float *x, float *root_logits, float *qual_logits, float *dur_logits);
void chord_eval_x(const float *x, float *logits);
/* a struct's dense vector (tests: against Python's features) */
void bass_x(const bass_in_t *in, float *x);
void drum_x(const drum_in_t *in, float *x);

uint32_t tnn_rand(uint32_t *r);             /* xorshift32 */
float tnn_unit(uint32_t *r);                /* [0, 1) */
/* an index drawn from logits at temperature temp among those allowed (allow NULL: all) */
int tnn_sample(const float *logits, const uint8_t *allow, int n, float temp, uint32_t *r);
int tnn_argmax(const float *v, const uint8_t *allow, int n);
float tnn_sigmoid(float x);

/* what the bass may play on a beat, and the nudges the engine gives the network's logits there (trio.c
 * decide_bass, and the learning below): in four a note every beat, E1..G3; in two, on an odd beat without a change,
 * a hold only. On a change the chord's notes up (the root 2.5, the others 1.5), and above A2 each semitone 0.06 down,
 * an upright's register. lg in place (the nudges, in this order), allow[TNN_NTOK] */
enum { TNN_B_CHANGE = 1, TNN_B_TWO = 2, TNN_B_ODD = 4 };   /* a beat's flags: the chord changes on it; in two; beat 2 or 4 */
#define TNN_LINE_HI 55                      /* the bass's highest note: G3 */
void bass_allow(uint8_t *allow, int flags);
void bass_shape(float *lg, uint8_t *allow, int root, uint16_t mask, int flags);

/* LEARNING the player's line (trio.c tr_teach): the BASS network frozen but for its output layer, the 35 tokens'
 * rows and the pickup's (96 weights and a bias each). Each beat the player taught is kept as the frozen network had
 * it on that beat, fed the player's own notes: its state (Q15: a GRU's state is within -1..1), what the player played
 * (the token, and a pickup or none after it) and the chord, for bass_shape. Then passes over the beats in order, an
 * SGD step each: the cross-entropy of the player's token under the softmax the engine samples from (the tokens
 * allowed there, the nudges added), on a beat whose token the bass could have played there; and the pickup's (a
 * sigmoid, weighted 0.3 as in tools/trio_train.py) where one could follow (in two on an odd beat, in four after a
 * note); with an L2 pull of the rows it moves back toward the trained ones (tools/trio_finetune.py, the reference).
 * It learns on top of what it learned before, the older lessons fading. The work comes an SGD step at a time, for
 * the main loop. What it learned is kept as each row's change from the trained one (int8, in steps of its own
 * size) and its bias: the form a flash object keeps, in use as it is, so a restart has the network as it was */
#define TNN_LROWS (TNN_NTOK + 1)            /* the rows that learn: the tokens', then the pickup's */
#define TNN_LPASSES 64                      /* passes, at most */
typedef struct {
    int16_t h[BASS_H];                      /* the frozen network's state after the beat's step, x 32768 */
    int8_t tok;                             /* what the player played on the beat: a pitch's token, TNN_HOLD, TNN_REST */
    uint8_t skip;                           /* a pickup after it, before the next beat */
    int8_t root;                            /* the chord (bass_shape) */
    uint8_t flags;                          /* TNN_B_* */
    uint16_t mask;
} tnn_beat_t;
int bass_learnable(const tnn_beat_t *b);    /* what a beat teaches: 1 its token, 2 its pickup (0: nothing) */
enum { TNN_LEARN_DONE, TNN_LEARN_SGD };
/* the beats: n of them from ring[first], wrapping at cap (they stay the caller's until the learning is done) */
void bass_learn_begin(const tnn_beat_t *ring, int cap, int first, int n, float lr, float l2, int passes);
int bass_learn_step(void);                  /* one SGD step; TNN_LEARN_DONE once none is left */
void bass_learn_commit(void);               /* the rows learned in use, as they are kept */
void bass_learn_cancel(void);
/* a pass's mean losses, before its steps' updates: what 0 the tokens' cross-entropy, 1 the pickups' (tests) */
float bass_learn_loss(int pass, int what);
const float *bass_learn_rows(void);         /* the rows as learned, float [TNN_LROWS][BASS_H], then their biases (tests) */
int bass_learned(void);                     /* the lines learned into the rows in use (0: as trained) */
int bass_learn_bytes(void);                 /* the RAM learning keeps: the rows being learned, the kept (tests) */
void bass_unlearn(void);                    /* the trained rows again */
/* the flash object: where the rows in use are kept and its length (nothing learned: its 16-byte head alone); to load
 * one, its place and room (bass_learned_room), then bass_learned_check with what was loaded there (bytes, < 0: none):
 * 0, in use; else (none, damaged, another network's) the trained rows */
const void *bass_learned_obj(uint32_t *len);
void *bass_learned_room(uint32_t *max);
int bass_learned_check(int len);
