/* SPDX-License-Identifier: MPL-2.0 */
/* The bass alone, note by note (E1 .. A3, one each), for tools/bass_measure.py to compare with real upright notes:
 *
 *   cc -O2 -ffp-contract=off -std=c99 -DTNN_HOST -Iapp/dsp -Ibase/include -o app/build/tests/bass_notes \
 *       tests/host/bass_notes.c app/dsp/voice.c -lm
 *   app/build/tests/bass_notes OUT.wav NOTES.txt [SECONDS_A_NOTE (0.5)] [VELOCITY (0.8)] [GHOST (0; 1: a ghost note
 *       on each note's swung eighth)]
 *   python3 -I tools/bass_measure.py wav OUT.wav NOTES.txt OUT.npz
 *   python3 -I tools/bass_measure.py compare REF.npz OUT.npz
 *
 * OUT.wav: mono, 16-bit, at half the voice's level (no clipping); NOTES.txt: each note's onset (s) and pitch. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "voice.h"

static void w32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w16(FILE *f, uint16_t v) { fwrite(&v, 2, 1, f); }

int main(int argc, char **argv)
{
    float spn = argc > 3 ? (float)atof(argv[3]) : 0.5f, vel = argc > 4 ? (float)atof(argv[4]) : 0.8f, peak = 0.0f;
    int ghost = argc > 5 ? atoi(argv[5]) : 0, p, i;
    uint32_t n = (uint32_t)(spn * 44100.0f), total = 0;
    static float L[44100 * 4], R[44100 * 4];
    v_mix_t mix = {0.0f, 1.0f, 0.0f, 0.0f};
    FILE *w, *t;
    if (argc < 3 || n == 0 || n > 44100u * 4u) {
        fprintf(stderr, "usage: bass_notes OUT.wav NOTES.txt [seconds a note] [velocity] [ghost]\n");
        return 2;
    }
    w = fopen(argv[1], "wb");
    t = fopen(argv[2], "w");
    if (!w || !t)
        return 1;
    v_init();
    fwrite("RIFF\0\0\0\0WAVEfmt ", 1, 16, w);
    w32(w, 16);
    w16(w, 1);
    w16(w, 1);
    w32(w, 44100);
    w32(w, 88200);
    w16(w, 2);
    w16(w, 16);
    fwrite("data\0\0\0\0", 1, 8, w);
    for (p = 28; p <= 57; p++) {
        uint32_t k = 0;
        fprintf(t, "%.6f %d\n", (double)total / 44100.0, p);
        v_bass_on(p, vel, 0);
        if (ghost) {
            k = (uint32_t)((float)n * 0.66f);
            v_render(L, R, k, &mix);
            v_bass_on(p, 0.35f, 1);
        }
        v_render(L + k, R + k, n - k, &mix);
        for (i = 0; i < (int)n; i++) {
            float s = L[i] * 0.5f;
            if (fabsf(L[i]) > peak)
                peak = fabsf(L[i]);
            s = s > 1.0f ? 1.0f : s < -1.0f ? -1.0f : s;
            w16(w, (uint16_t)(int16_t)lrintf(s * 32767.0f));
        }
        total += n;
    }
    fseek(w, 4, SEEK_SET);
    w32(w, 36 + total * 2);
    fseek(w, 40, SEEK_SET);
    w32(w, total * 2);
    fclose(w);
    fclose(t);
    printf("%u notes, peak %.3f (1.0: full scale at MASTER's top)\n", 30u, (double)peak);
    return 0;
}
