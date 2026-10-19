/* SPDX-License-Identifier: MPL-2.0 */
/* The tunes the offline tests play (tests/host/trio_render.c, teach_test.c): their charts, a chorus or more of them
 * a beat at a time, and a jazz pianist's voicings. A header of static functions: each test includes it once. */
#pragma once
#include <string.h>
#include "trio_nn.h"

static const char *AUTUMN = "Cm7|F7|Bbmaj7|Ebmaj7|Am7b5|D7|Gm7|Gm7|Cm7|F7|Bbmaj7|Ebmaj7|Am7b5|D7|Gm7|Gm7|"
                            "Am7b5|D7|Gm7|Gm7|Cm7|F7|Bbmaj7|Ebmaj7|Am7b5|D7|Gm7 C7|Fm7 Bb7|Ebmaj7|Am7b5 D7|Gm7|Gm7";
static const char *BLUES = "F7|Bb7|F7|Cm7 F7|Bb7|Bdim7|F7|D7|Gm7|C7|F7 D7|Gm7 C7";
static const char *RHYTHM = "Bbmaj7 G7|Cm7 F7|Bbmaj7 G7|Cm7 F7|Fm7 Bb7|Ebmaj7 Ab7|Bbmaj7 G7|Cm7 F7|"
                            "Bbmaj7 G7|Cm7 F7|Bbmaj7 G7|Cm7 F7|Fm7 Bb7|Ebmaj7 Ab7|Cm7 F7|Bbmaj7|"
                            "D7|D7|G7|G7|C7|C7|F7|F7|"
                            "Bbmaj7 G7|Cm7 F7|Bbmaj7 G7|Cm7 F7|Fm7 Bb7|Ebmaj7 Ab7|Cm7 F7|Bbmaj7";

#define MAXB 2048
static int8_t ch_root[MAXB], ch_qual[MAXB];      /* the chart's chord on each beat */
static int nbeats;

static int parse_chord(const char *s, int *root, int *qual)
{
    static const int L[7] = {9, 11, 0, 2, 4, 5, 7};   /* A B C D E F G */
    int r;
    if (*s < 'A' || *s > 'G')
        return 0;
    r = L[*s - 'A'];
    s++;
    if (*s == 'b')
        r--, s++;
    else if (*s == '#')
        r++, s++;
    *root = (r + 12) % 12;
    if (!strncmp(s, "maj7", 4) || *s == 0 || *s == ' ' || *s == '|')
        *qual = Q_MAJ;
    else if (!strncmp(s, "m7b5", 4))
        *qual = Q_HDIM;
    else if (!strncmp(s, "m7", 2) || *s == 'm')
        *qual = Q_MIN;
    else if (!strncmp(s, "dim", 3))
        *qual = Q_DIM;
    else
        *qual = Q_DOM;
    return 1;
}

static void load_chart(const char *chart, int choruses)
{
    int c, b = 0;
    for (c = 0; c < choruses; c++) {
        const char *p = chart;
        while (*p) {
            int n = 0, k, roots[4], quals[4];
            while (*p && *p != '|') {
                while (*p == ' ')
                    p++;
                if (*p && *p != '|' && n < 4 && parse_chord(p, &roots[n], &quals[n]))
                    n++;
                while (*p && *p != ' ' && *p != '|')
                    p++;
            }
            if (*p == '|')
                p++;
            for (k = 0; k < 4 && b < MAXB; k++, b++) {
                int i = n == 0 ? 0 : k * n / 4;
                ch_root[b] = n ? (int8_t)roots[i] : -1;
                ch_qual[b] = n ? (int8_t)quals[i] : 0;
            }
        }
    }
    nbeats = b;
}

/* a jazz pianist's voicing: rootless (3rd, 5th or 13th, 7th, 9th) around middle C, or a root below a shell */
static int voicing(int root, int qual, int rooted, int *notes)
{
    static const int8_t RL[7][4] = {{4, 7, 11, 14}, {4, 9, 10, 14}, {3, 7, 10, 14}, {3, 6, 10, 12},
                                    {0, 3, 6, 9}, {5, 7, 10, 14}, {4, 8, 12, 16}};
    int n = 0, k, base = 55 + ((root - 55 % 12) + 24) % 12;   /* the 3rd near G3..F#4 */
    if (base > 62)
        base -= 12;
    if (rooted)
        notes[n++] = 40 + ((root - 4) + 24) % 12;            /* E2..D#3 */
    for (k = 0; k < 4; k++)
        notes[n++] = base + RL[qual][k];
    return n;
}
