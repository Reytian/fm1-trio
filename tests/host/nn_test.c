/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's networks in C against the quantized PyTorch models (tests/host/trio_golden.h from tools/trio_train.py),
 * and what a step costs (instructions, a scalar build, for tests/run_tests.sh's load estimate).
 *   cc -O2 -ffp-contract=off -DTNN_HOST -Iapp/dsp -Ibase/include tests/host/nn_test.c app/dsp/trio_nn.c */
#include <stdio.h>
#include <math.h>
#include "trio_nn.h"
#include "trio_w.h"
#include "trio_golden.h"
#ifdef __APPLE__
#include <libproc.h>
#include <unistd.h>
static unsigned long long insns(void)
{
    struct rusage_info_v4 ri;
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri);
    return ri.ri_instructions;
}
#else
static unsigned long long insns(void) { return 0; }
#endif

static int fails;
static float worst;
static void check(const char *what, const float *got, const float *want, int n)
{
    int i;
    float w = 0.0f;
    for (i = 0; i < n; i++) {
        float d = fabsf(got[i] - want[i]) / (1.0f + fabsf(want[i]));
        if (d > w)
            w = d;
    }
    if (w > worst)
        worst = w;
    if (w > 2e-4f) {
        printf("FAIL %s: off by %g\n", what, (double)w);
        fails++;
    }
}

int main(void)
{
    float h[BASS_H], out[128];
    int t, n;
    unsigned long long i0, i1;
    tnn_init();

    /* BASS: 16 beats of a held-out line */
    n = (int)(sizeof G_BASS_X / sizeof(float)) / BASS_NIN;
    tnn_zero(h, BASS_H);
    i0 = insns();
    for (t = 0; t < n; t++) {
        bass_step_x(h, G_BASS_X + t * BASS_NIN, out, out + TNN_NTOK);
        check("bass", out, G_BASS_Y + t * (TNN_NTOK + 1), TNN_NTOK + 1);
    }
    i1 = insns();
    printf("bass:  %d steps, %llu instructions a step\n", n, (i1 - i0) / (unsigned long long)n);

    /* DRUM: 24 steps (two bars) */
    n = (int)(sizeof G_DRUM_X / sizeof(float)) / DRUM_NIN;
    tnn_zero(h, DRUM_H);
    i0 = insns();
    for (t = 0; t < n; t++) {
        drum_step_x(h, G_DRUM_X + t * DRUM_NIN, out, out + TNN_NPIECE);
        check("drum", out, G_DRUM_Y + t * (4 * TNN_NPIECE), 4 * TNN_NPIECE);
    }
    i1 = insns();
    printf("drum:  %d steps, %llu instructions a step\n", n, (i1 - i0) / (unsigned long long)n);

    /* HARM: 12 chords */
    n = (int)(sizeof G_HARM_X / sizeof(float)) / HARM_NIN;
    tnn_zero(h, HARM_H);
    for (t = 0; t < n; t++) {
        harm_step_x(h, G_HARM_X + t * HARM_NIN, out, out + 12, out + 12 + TNN_NQUAL);
        check("harm", out, G_HARM_Y + t * (12 + TNN_NQUAL + TNN_NDUR), 12 + TNN_NQUAL + TNN_NDUR);
    }

    /* CHORD: 8 voicings */
    n = (int)(sizeof G_CHORD_X / sizeof(float)) / CHORD_NIN;
    i0 = insns();
    for (t = 0; t < n; t++) {
        chord_eval_x(G_CHORD_X + t * CHORD_NIN, out);
        check("chord", out, G_CHORD_Y + t * 12 * TNN_NQUAL, 12 * TNN_NQUAL);
    }
    i1 = insns();
    printf("chord: %d voicings, %llu instructions each\n", n, (i1 - i0) / (unsigned long long)n);
    printf("largest difference from PyTorch: %g (relative)\n", (double)worst);
    printf(fails ? "nn: FAILED\n" : "nn: all match\n");
    return fails != 0;
}
