/* SPDX-License-Identifier: MPL-2.0 */
/* TEACHING the bass (trio_nn.c bass_learn_*, bass_learned_*) against tools/trio_finetune.py, how what it learned is
 * kept, and what it costs. Bounds set before the first measurement (MIN-1's): the frozen states within 1e-4 of
 * PyTorch's, each pass's mean losses within 1e-4, the learned rows' logits on every beat after the passes within 2e-3
 * (hundreds of float SGD steps in another order); kept (each row's change int8, in steps of its own size) within
 * 0.02 of the float rows'; the kept form back from its bytes alone exactly (the memory scrubbed first); a head that is
 * not this network's or this form's, a length, a step or a bias that cannot be, refused (a damaged byte elsewhere is
 * the store's CRC's to find). The cost: instructions in a scalar build, as tests/host/trio_render.c counts them (one a
 * frame is ~0.030% of an FM-1 core).
 *   cc -O2 -ffp-contract=off -DTNN_HOST -Iapp/dsp -Ibase/include -Itests/host tests/host/ft_test.c app/dsp/trio_nn.c */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "trio_nn.h"
#include "ft_golden.h"
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

#define STORE_MAX 3840u                         /* plat.h PLAT_STORE_MAX: a flash object's bytes */
#define TEACH_MAX 128                           /* app/dsp/trio.c: beats a lesson keeps */
#define LESSON_PASSES 15                        /* app/dsp/trio.c LRN_PASSES */
static double fm1_ms(double insn) { return insn * 0.030 / 100.0 / 44100.0 * 1000.0; }

static int fails;
#define CHECK(c, ...)                                                                                              \
    do {                                                                                                           \
        if (!(c)) {                                                                                                \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                            \
            printf(__VA_ARGS__);                                                                                   \
            printf("\n");                                                                                          \
            fails++;                                                                                               \
        }                                                                                                          \
    } while (0)

static bass_in_t in[FTG_N];
static int8_t tok[FTG_N];
static uint8_t skip[FTG_N];
static tnn_beat_t ring[TEACH_MAX];

static int16_t q15(float v)                     /* as trio.c keeps a state */
{
    float x = v * 32768.0f;
    int32_t i = (int32_t)(x + (x < 0.0f ? -0.5f : 0.5f));
    return (int16_t)(i > 32767 ? 32767 : i < -32768 ? -32768 : i);
}

static int flags_of(const bass_in_t *b)
{
    return (b->since == 0 ? TNN_B_CHANGE : 0) | (b->feel ? TNN_B_TWO : 0) | (b->bib & 1 ? TNN_B_ODD : 0);
}

/* the logits (35 tokens and the pickup) on every beat of the lesson, the network teacher-forced from its state at the
 * lesson's start, with the rows in use */
static void lesson_logits(float out[FTG_N][TNN_LROWS])
{
    float h[BASS_H];
    int t;
    memcpy(h, FTG_H0, sizeof h);
    for (t = 0; t < FTG_N; t++)
        bass_step(h, &in[t], out[t], out[t] + TNN_NTOK);
}

static double max_diff(float a[FTG_N][TNN_LROWS], float b[FTG_N][TNN_LROWS])
{
    double d = 0.0;
    int t, k;
    for (t = 0; t < FTG_N; t++)
        for (k = 0; k < TNN_LROWS; k++)
            d = fmax(d, fabs((double)a[t][k] - (double)b[t][k]));
    return d;
}

/* the kept bytes loaded back as the app does (bass_learned_room, the bytes there, bass_learned_check), into memory
 * scrubbed first: nothing of what was in use is left to pass for them */
static int load(const void *b, int len)
{
    uint32_t max;
    void *p = bass_learned_room(&max);
    memset(p, 0xA5, max);
    if (len > 0)
        memcpy(p, b, (size_t)(len < (int)max ? len : (int)max));
    return bass_learned_check(len);
}

static void put_f32(uint8_t *b, float f) { memcpy(b, &f, 4); }

int main(void)
{
    static float trained[FTG_N][TNN_LROWS], lo[FTG_N][TNN_LROWS], kept[FTG_N][TNN_LROWS];
    static uint8_t obj[STORE_MAX], bad[STORE_MAX];
    float h[BASS_H], x[TNN_LROWS];
    double dh = 0.0, dl = 0.0, dz = 0.0, dq;
    unsigned long long i0, i1, i2, i3, i4;
    uint32_t len, hlen;
    int t, k, j, n = 0, steps = 0, olen, moved = -1;
    tnn_init();
    for (t = 0; t < FTG_N; t++) {
        const int16_t *f = FTG_IN + 13 * t;
        in[t].ptok = (int8_t)f[0];
        in[t].root = (int8_t)f[1];
        in[t].qual = (int8_t)f[2];
        in[t].mask = (uint16_t)f[3];
        in[t].nroot = (int8_t)f[4];
        in[t].nmask = (uint16_t)f[5];
        in[t].tochg = (uint8_t)f[6];
        in[t].since = (uint8_t)f[7];
        in[t].bib = (uint8_t)f[8];
        in[t].feel = (uint8_t)f[9];
        in[t].pskip = (uint8_t)f[10];
        tok[t] = (int8_t)f[11];
        skip[t] = (uint8_t)f[12];
    }
    lesson_logits(trained);
    /* the frozen states, and the lesson as trio.c keeps it: each beat that teaches something, its state Q15 */
    memcpy(h, FTG_H0, sizeof h);
    for (t = 0; t < FTG_N; t++) {
        tnn_beat_t *b = &ring[n];
        bass_step(h, &in[t], x, x + TNN_NTOK);
        for (j = 0; j < BASS_H; j++) {
            dh = fmax(dh, fabs((double)h[j] - (double)FTG_H[t * BASS_H + j]));
            b->h[j] = q15(h[j]);
        }
        b->tok = tok[t];
        b->skip = skip[t];
        b->root = in[t].root;
        b->mask = in[t].mask;
        b->flags = (uint8_t)flags_of(&in[t]);
        if (bass_learnable(b) && n < TEACH_MAX)
            n++;
    }
    CHECK(n == FTG_N, "every beat of the lesson teaches something (%d of %d)", n, FTG_N);
    /* the learning, its cost */
    bass_learn_begin(ring, TEACH_MAX, 0, n, FTG_LR, FTG_L2, FTG_PASSES);
    i0 = insns();
    while (bass_learn_step() == TNN_LEARN_SGD)
        steps++;
    i1 = insns();
    CHECK(steps == n * FTG_PASSES, "%d SGD steps (%d beats x %d passes)", steps, n, FTG_PASSES);
    CHECK(!bass_learned(), "learned, not yet committed: the trained rows in use");
    for (k = 0; k < FTG_PASSES; k++)
        for (j = 0; j < 2; j++)
            dl = fmax(dl, fabs((double)bass_learn_loss(k, j) - (double)FTG_LOSS[2 * k + j]));
    {   /* the learned rows (float) on every beat's state as PyTorch kept it (Q15) */
        const float *w = bass_learn_rows(), *b = w + TNN_LROWS * BASS_H;
        for (t = 0; t < FTG_N; t++)
            for (k = 0; k < TNN_LROWS; k++) {
                float z = 0.0f;
                for (j = 0; j < BASS_H; j++)
                    z += w[k * BASS_H + j] * ((float)q15(FTG_H[t * BASS_H + j]) * (1.0f / 32768.0f));
                z += b[k];
                dz = fmax(dz, fabs((double)z - (double)FTG_LOGITS[t * TNN_LROWS + k]));
            }
    }
    printf("parity: states within %.2e, pass losses within %.2e (tokens %.4f -> %.4f, pickups %.4f -> %.4f), learned "
           "logits within %.2e\n", dh, dl, (double)bass_learn_loss(0, 0), (double)bass_learn_loss(FTG_PASSES - 1, 0),
           (double)bass_learn_loss(0, 1), (double)bass_learn_loss(FTG_PASSES - 1, 1), dz);
    CHECK(dh <= 1e-4, "the frozen states %.2e off PyTorch's (bound 1e-4)", dh);
    CHECK(dl <= 1e-4, "a pass's loss %.2e off (bound 1e-4)", dl);
    CHECK(dz <= 2e-3, "a learned logit %.2e off (bound 2e-3)", dz);
    /* kept: each row's change int8; the logits as the network gives them, against the float rows' */
    bass_learn_commit();
    CHECK(bass_learned() == 1, "committed: one line learned (%d)", bass_learned());
    lesson_logits(kept);
    dq = 0.0;
    for (t = 0; t < FTG_N; t++)
        for (k = 0; k < TNN_LROWS; k++)
            dq = fmax(dq, fabs((double)kept[t][k] - (double)FTG_LOGITS[t * TNN_LROWS + k]));
    printf("kept: the logits within %.2e of the float rows' (bound 0.02)\n", dq);
    CHECK(dq <= 0.02, "kept logits %.2e off the float rows' (bound 0.02)", dq);
    /* its bytes: back exactly; nothing learned: the trained network exactly */
    {
        const void *p = bass_learned_obj(&len);
        CHECK(len == 16u + 8u * TNN_LROWS + TNN_LROWS * BASS_H && len <= STORE_MAX, "the object: %u bytes (at most %u)",
              len, STORE_MAX);
        memcpy(obj, p, len);
        olen = (int)len;
    }
    for (k = TNN_LROWS - 1; k >= 0; k--) {      /* a row that moved: its step, for the checks below */
        float st;
        memcpy(&st, obj + 16 + 4 * k, 4);
        if (st > 0.0f)
            moved = k;
    }
    CHECK(moved >= 0, "a row moved");
    bass_unlearn();
    lesson_logits(lo);
    CHECK(!bass_learned() && max_diff(lo, trained) == 0.0, "forgotten: the trained network exactly (%.2e)",
          max_diff(lo, trained));
    bass_learned_obj(&hlen);
    CHECK(hlen == 16u, "nothing learned: the object's head alone (%u)", hlen);
    CHECK(load(obj, olen) == 0 && bass_learned() == 1, "the object loaded: in use (%d)", bass_learned());
    lesson_logits(lo);
    CHECK(max_diff(lo, kept) == 0.0, "loaded: the logits exactly as kept (%.2e)", max_diff(lo, kept));
    /* refused, the trained rows then: another network's or form's, short, nothing learned, none (a damaged one the
     * store's CRC refuses first: a change's byte is still an object here) */
    if (moved >= 0) {
        /* the head: magic, the network, the rows, the columns, the learnings; a moved row's step's sign */
        const int AT[6] = {0, 4, 8, 10, 12, 16 + 4 * moved + 3};
        static const char *const WHAT[6] = {"magic", "network", "rows", "columns", "learnings", "step's sign"};
        int m;
        for (m = 0; m < 6; m++) {
            memcpy(bad, obj, (size_t)olen);
            if (m == 4)
                bad[12] = bad[13] = 0;          /* no learnings */
            else
                bad[AT[m]] ^= m == 5 ? 0x80u : 0x01u;
            CHECK(load(bad, olen) < 0 && !bass_learned(), "%s changed (byte %d): refused", WHAT[m], AT[m]);
        }
        memcpy(bad, obj, (size_t)olen);         /* a step of 1 or more, not a number; a bias not a number */
        put_f32(bad + 16 + 4 * moved, 1.0f);
        CHECK(load(bad, olen) < 0 && !bass_learned(), "a step of 1: refused");
        memcpy(bad, obj, (size_t)olen);
        put_f32(bad + 16 + 4 * moved, nanf(""));
        CHECK(load(bad, olen) < 0 && !bass_learned(), "a step not a number: refused");
        memcpy(bad, obj, (size_t)olen);
        put_f32(bad + 16 + 4 * TNN_LROWS + 4 * moved, nanf(""));
        CHECK(load(bad, olen) < 0 && !bass_learned(), "a bias not a number: refused");
        memcpy(bad, obj, (size_t)olen);
        bad[olen - 100] ^= 0x01u;
        CHECK(load(bad, olen) == 0, "a change's byte changed: still an object (the store's CRC finds that)");
        CHECK(load(obj, olen - 1) < 0 && !bass_learned(), "short: refused");
        CHECK(load(obj, 16) < 0 && !bass_learned(), "the head alone: as trained");
        CHECK(load(obj, -1) < 0 && !bass_learned(), "none: as trained");
        lesson_logits(lo);
        CHECK(max_diff(lo, trained) == 0.0, "refused: the trained network exactly");
    }
    /* on top: from the rows in use; no passes, the rows as they were */
    load(obj, olen);
    bass_learn_begin(ring, TEACH_MAX, 0, n, FTG_LR, FTG_L2, 0);
    while (bass_learn_step() != TNN_LEARN_DONE)
        ;
    bass_learn_commit();
    lesson_logits(lo);
    CHECK(bass_learned() == 2 && max_diff(lo, kept) <= 1e-5, "on top of what it learned: %d, the rows as they were (%.2e)",
          bass_learned(), max_diff(lo, kept));
    /* a step of the network with the rows learned and without */
    {
        int m;
        memcpy(h, FTG_H0, sizeof h);
        i2 = insns();
        for (m = 0; m < 200; m++)
            bass_step(h, &in[m % FTG_N], x, x + TNN_NTOK);
        i3 = insns();
        bass_unlearn();
        memcpy(h, FTG_H0, sizeof h);
        for (m = 0; m < 200; m++)
            bass_step(h, &in[m % FTG_N], x, x + TNN_NTOK);
        i4 = insns();
    }
    if (i1 > i0) {
        double step = (double)(i1 - i0) / steps, learned_step = (double)(i3 - i2) / 200.0,
               plain = (double)(i4 - i3) / 200.0;
        printf("cost: an SGD step %.0f insn (%.3f ms on the FM-1); a lesson of %d beats x %d passes %.0f ms of one core, "
               "of %d beats (the most a lesson keeps) %.0f ms\n", step, fm1_ms(step), n, FTG_PASSES,
               fm1_ms(step * n * FTG_PASSES), TEACH_MAX, fm1_ms(step * TEACH_MAX * LESSON_PASSES));
        printf("a step of the network: %.0f insn as trained, %.0f with the rows learned (%+.1f%%; %.2f ms)\n", plain,
               learned_step, 100.0 * (learned_step / plain - 1.0), fm1_ms(learned_step));
    }
    printf("memory, the pool: learning %d B (the rows being learned, their state, the kept rows: the flash object, "
           "%d B)\n", bass_learn_bytes(), olen);
    printf(fails ? "ft: %d FAILED\n" : "ft: ok\n", fails);
    return fails != 0;
}
