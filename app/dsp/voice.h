/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's sound: an electric piano for the player, an upright bass, a jazz kit, and a room. All synthesis, no
 * samples. The engine (trio.c) calls these from the audio interrupt only. */
#pragma once
#include <stdint.h>

#define V_SR 44100.0f
#define V_NEP 10                    /* piano voices */

void v_init(void);                  /* tables and buffers (at power-on) */
void v_ep_on(int note, float vel);  /* the player's piano */
void v_ep_off(int note);
void v_ep_all_off(void);
void v_bass_on(int note, float vel, int ghost);   /* the bass: a plucked string; ghost: a short, quiet pickup */
void v_bass_damp(void);             /* the hand on the strings */
void v_drum(int piece, float vel);  /* trio_nn.h P_* */
void v_panic(void);

typedef struct {
    float piano, bass, drums;       /* levels, 0..1 */
    float room;                     /* the reverb's send, 0..1 */
} v_mix_t;

/* n stereo frames into lr (float, about +-1), the mix as set */
void v_render(float *l, float *r, uint32_t n, const v_mix_t *mix);
