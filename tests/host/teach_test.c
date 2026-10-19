/* SPDX-License-Identifier: MPL-2.0 */
/* TEACHING the bass (trio.c tr_teach) offline, as on the FM-1. A pianist plays a tune's chords round twice, an
 * eighth early with the root lowest (the band learns the form), and lets go; the band plays a chorus alone: BEFORE.
 * Then the player teaches a walking line for two choruses, over the form (or, --chords, over the pianist's chords,
 * voiced from Ab3 up): MIDI notes from E1 to G3, on each chord R 5 8 then the half step above the next root on its
 * last beat (a chord of two beats: R and that approach), a ghost pickup into each change after a bar or more, each
 * note up to 25 ms off its beat, legato. Teaching off just after the last beat, the main loop learns the line while
 * the band plays on alone: AFTER, the two choruses after the next. The main loop gets, between audio blocks, only
 * the FM-1's time left (each call's instructions in FM-1 milliseconds by MIN-1's calibration, 0.030 % of a core per
 * instruction a frame; the render's own time and a screen's taken out: 7 ms a beat for the strips a beat redraws,
 * 0.2 ms a frame for the input), so a beat decided late is missed (LATE n on the screen).
 *
 *   teach_test [PROG] [BPM] [SEEDS] [--chords] [--fresh] [--two] [--teach CHORUSES] [--freedom N] [--untimed]
 *              [--dump FILE]
 *
 * Prints, a seed a line and their mean: how often the bass's note is the taught one on the same beat of the form
 * (all beats; on the changes and between them; its pitch class), before and after; its pickups after, where the
 * line had them and where not; the lesson's mean loss in its first pass and its last; the learning's cost; missed
 * beats. --dump writes the first seed's lesson for tools/trio_finetune.py: each beat's inputs as decide_bass built
 * them, what the player played, and the state the lesson started from. --untimed: the main loop works until it is
 * idle between blocks, as trio_render.c runs it (for a sanitizer's build: its instructions are not the FM-1's).
 * Exit 1 if a beat was missed, or the bass agrees with the line less than 15 points more often after than before.
 *
 * Build (tests/run_tests.sh): cc -O2 -ffp-contract=off -fno-vectorize -fno-slp-vectorize -DTNN_HOST -Iapp/dsp
 *   -Ibase/include tests/host/teach_test.c app/dsp/trio.c app/dsp/voice.c app/dsp/trio_nn.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "trio.h"
#ifdef __APPLE__
#include <libproc.h>
static unsigned long long insns(void)
{
    struct rusage_info_v4 ri;
    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri);
    return ri.ri_instructions;
}
#else
static unsigned long long insns(void) { return 0; }
#endif

#include "charts.h"

#define SRF 44100.0
#define BLOCK 256
static double fm1_ms(double insn) { return insn * 0.030 / 100.0 / SRF * 1000.0; }

/* the player's line, a chorus of it: a note a beat, and a ghost pickup after some */
static int8_t line_note[MAXB];
static uint8_t line_skip[MAXB], line_change[MAXB];
static int L;                                   /* beats a chorus */

static int low_root(int root) { return 31 + (root - 7 + 12) % 12; }   /* G1 .. F#2 */

static int two;                                 /* --two: the band in two, the line on 1 and 3 */

static void make_line(void)
{
    int b = 0;
    while (b < L) {
        int e = b, i, n, r = low_root(ch_root[b]), nr, fifth = ch_qual[b] == Q_HDIM || ch_qual[b] == Q_DIM ? 6
                                                                : ch_qual[b] == Q_AUG ? 8 : 7;
        while (e < L && ch_root[e] == ch_root[b] && ch_qual[e] == ch_qual[b])
            e++;
        n = e - b;
        nr = low_root(ch_root[e % L]);
        for (i = 0; i < n; i++) {
            static const int8_t UP[4] = {0, 0, 12, 0};
            line_note[b + i] = (int8_t)(n > 1 && i == n - 1 ? nr + 1 : r + (i % 2 ? fifth : UP[i % 4]));
            line_skip[b + i] = (uint8_t)(n >= 4 && i == n - 1);
            line_change[b + i] = (uint8_t)(i == 0);
        }
        b = e;
    }
    for (b = 0; two && b < L; b++)              /* in two: 2 and 4 held (unless the chord changes on them) */
        if ((b & 1) && !line_change[b]) {
            line_note[b] = -1;
            line_skip[b - 1] = 0;
        }
}

/* what happens when: MIDI in, teaching on and off */
enum { EV_MIDI, EV_TEACH_ON, EV_TEACH_OFF };
typedef struct {
    double t;                                   /* frames from PLAY */
    int what, st, d1, d2;
} ev_t;
static ev_t evs[65536];
static int nev;
static void add(double t, int what, int st, int d1, int d2)
{
    if (nev < (int)(sizeof evs / sizeof evs[0]))
        evs[nev++] = (ev_t){t, what, st, d1, d2};
}
static int by_time(const void *a, const void *b)
{
    const ev_t *x = a, *y = b;
    return x->t < y->t ? -1 : x->t > y->t ? 1 : (x->st & 0xF0) == 0x80 ? -1 : (y->st & 0xF0) == 0x80 ? 1 : 0;
}

static uint32_t jit = 12345u;
static double jitter(void)                      /* up to 25 ms either way (frames) */
{
    jit ^= jit << 13;
    jit ^= jit >> 17;
    jit ^= jit << 5;
    return ((double)(jit >> 8) / 16777216.0 * 2.0 - 1.0) * 0.025 * SRF;
}

/* a voicing from Ab3 up, root lowest (over the line, while teaching) */
static int high_voicing(int root, int qual, int *notes)
{
    static const int8_t IV[7][4] = {{0, 4, 7, 11}, {0, 4, 7, 10}, {0, 3, 7, 10}, {0, 3, 6, 10},
                                    {0, 3, 6, 9}, {0, 5, 7, 10}, {0, 4, 8, 12}};
    int k, r = 56 + (root - 8 + 12) % 12;
    for (k = 0; k < 4; k++)
        notes[k] = r + IV[qual][k];
    return 4;
}

typedef struct {
    double before, after, b_chg, a_chg, b_mid, a_mid, b_pc, a_pc, skip_yes, skip_no, loss0, loss1, learn_ms,
        slice_ms, learn_wall_ms;
    unsigned missed;
    int kept, kept_min;
} result_t;

static void agree(int32_t from, int32_t to, double *all, double *chg, double *mid, double *pc)
{
    int32_t b;
    int n = 0, s = 0, nc = 0, sc = 0, nm = 0, sm = 0, sp = 0;
    for (b = from; b < to; b++) {
        int tok = tr_log_bass(b), want = 2 + line_note[b % L] - TNN_LO;
        if (tok < 2 || line_note[b % L] < 0)
            continue;
        n++;
        s += tok == want;
        sp += (tok - 2 + TNN_LO) % 12 == line_note[b % L] % 12;
        if (line_change[b % L]) {
            nc++;
            sc += tok == want;
        } else {
            nm++;
            sm += tok == want;
        }
    }
    *all = n ? 100.0 * s / n : 0.0;
    *chg = nc ? 100.0 * sc / nc : 0.0;
    *mid = nm ? 100.0 * sm / nm : 0.0;
    *pc = n ? 100.0 * sp / n : 0.0;
}

static int fresh;                               /* --fresh: teaching on before PLAY, over the pianist's chords */
static int untimed;                             /* --untimed: the main loop until idle (no FM-1 clock) */
static int freedom = 50;                        /* --freedom: K2 FREEDOM, the bass's temperature (50, as TRIO starts) */

static void run(uint32_t seed, int bpm, int teach, int chords, const char *dump, result_t *res)
{
    const double spb = SRF * 60.0 / bpm;
    const int c_teach = fresh ? 0 : 3, c_after = c_teach + teach + 1, total = (c_after + 2) * L;
    int32_t lr[2 * BLOCK], last_beat = -99, b;
    int held[8], nheld = 0, k, ie = 0, was_learning = 0;
    double frames = 0.0, budget = 0.0, learn_from = -1.0;
    unsigned long long i0, lsum = 0, lmax = 0;
    nev = 0;
    jit = 12345u + seed * 7919u;
    /* the pianist: the chords an eighth early, two choruses (and, --chords, through the teaching) */
    for (b = 0; b < (chords ? (c_teach + teach) * L : 2 * L); b++) {
        int bl = b % L, notes[8], n;
        double at = ((double)b + 4.0 - 0.33) * spb;
        if (b > 0 && bl > 0 && ch_root[bl] == ch_root[bl - 1] && ch_qual[bl] == ch_qual[bl - 1])
            continue;
        if (b >= 2 * L && b < c_teach * L)      /* (the chorus before: the band alone) */
            continue;
        for (k = 0; k < nheld; k++)
            add(at - 1.0, EV_MIDI, 0x80, held[k], 0);
        n = b < 2 * L && !fresh ? voicing(ch_root[bl], ch_qual[bl], 1, notes)
                                : high_voicing(ch_root[bl], ch_qual[bl], notes);
        for (k = 0; k < n; k++)
            add(at, EV_MIDI, 0x90, notes[k], 70 + 7 * k);
        nheld = n;
        memcpy(held, notes, sizeof notes);
    }
    {   /* and lets go: after the second chorus, or the teaching */
        double at = ((double)(chords ? (c_teach + teach) * L : 2 * L) + 4.0 - 0.33) * spb;
        for (k = 0; k < nheld; k++)
            add(at, EV_MIDI, 0x80, held[k], 0);
    }
    /* the player: teaching on a beat before its first chorus (--fresh: before PLAY), the line, teaching off just
     * after its last beat */
    if (!fresh)
        add(((double)(c_teach * L) + 3.0) * spb, EV_TEACH_ON, 0, 0, 0);
    {
        int prev = -1;
        for (b = c_teach * L; b < (c_teach + teach) * L; b++) {
            int bl = b % L;
            double on = ((double)b + 4.0) * spb + jitter();
            if (line_note[bl] < 0)              /* (in two: held) */
                continue;
            if (prev >= 0)
                add(on + 0.010 * SRF, EV_MIDI, 0x80, prev, 0);   /* legato: the note before lets go just after */
            add(on, EV_MIDI, 0x90, line_note[bl], 84);
            prev = line_note[bl];
            if (line_skip[bl]) {                /* a ghost on the swung eighth, the same note */
                double g = ((double)b + 4.0 + 0.64) * spb + jitter() * 0.4;
                add(g - 0.004 * SRF, EV_MIDI, 0x80, prev, 0);
                add(g, EV_MIDI, 0x90, prev, 36);
            }
        }
        add(((double)((c_teach + teach) * L) + 4.0 - 0.1) * spb, EV_TEACH_OFF, 0, 0, 0);
        add(((double)((c_teach + teach) * L) + 4.0 - 0.1) * spb + 1.0, EV_MIDI, 0x80, prev, 0);
    }
    qsort(evs, (size_t)nev, sizeof evs[0], by_time);

    tr_init();
    tr_seed(seed);
    tr_set.bpm = (uint16_t)bpm;
    tr_set.freedom = (uint8_t)freedom;
    if (two)
        tr_set.feel = TR_TWO;
    if (fresh)
        tr_teach(1);
    tr_play(1);
    memset(res, 0, sizeof *res);
    while (frames < ((double)total + 4.0) * spb) {
        double block_ms = BLOCK * 1000.0 / SRF;
        while (ie < nev && evs[ie].t < frames + BLOCK) {
            const ev_t *e = &evs[ie++];
            if (e->what == EV_MIDI)
                tr_midi((uint32_t)e->st, (uint32_t)e->d1, (uint32_t)e->d2);
            else if (e->what == EV_TEACH_ON)
                tr_teach(1);
            else
                res->kept = tr_teach(0);
        }
        i0 = insns();
        tr_render(lr, BLOCK, 4096);
        budget += block_ms - fm1_ms((double)(insns() - i0)) - 0.2 * block_ms / (1000.0 / 60.0);
        if (tr_beat != last_beat) {             /* a beat: the screen's strips */
            last_beat = tr_beat;
            budget -= 7.0;
        }
        if (untimed)
            budget = 1e9;
        while (budget > 0.0) {                  /* the main loop: what time is left */
            int learning = tr_learning(), r;
            uint32_t steps = tr_nn_steps();
            i0 = insns();
            r = tr_service();
            i0 = insns() - i0;
            budget -= fm1_ms((double)i0);
            if (learning && tr_nn_steps() == steps) {   /* a slice of the learning */
                lsum += i0;
                lmax = i0 > lmax ? i0 : lmax;
                if (learn_from < 0.0)
                    learn_from = frames;
            }
            if (!r) {
                budget = 0.0;                   /* idle: nothing to carry */
                break;
            }
        }
        if (was_learning && !tr_learning() && learn_from >= 0.0)
            res->learn_wall_ms = (frames - learn_from) * 1000.0 / SRF;
        was_learning = tr_learning();
        frames += BLOCK;
    }
    if (!fresh)
        agree(2 * L, 3 * L, &res->before, &res->b_chg, &res->b_mid, &res->b_pc);
    agree(c_after * L, (c_after + 2) * L, &res->after, &res->a_chg, &res->a_mid, &res->a_pc);
    {
        int ny = 0, sy = 0, nn = 0, sn = 0;
        for (b = c_after * L; b < (c_after + 2) * L; b++) {
            if (tr_log_bass(b) < 2)
                continue;
            if (line_skip[b % L]) {
                ny++;
                sy += tr_log_skip(b) > 0;
            } else {
                nn++;
                sn += tr_log_skip(b) > 0;
            }
        }
        res->skip_yes = ny ? 100.0 * sy / ny : 0.0;
        res->skip_no = nn ? 100.0 * sn / nn : 0.0;
    }
    res->loss0 = (double)bass_learn_loss(0, 0);
    res->loss1 = (double)bass_learn_loss(14, 0);   /* (LRN_PASSES 15) */
    res->learn_ms = fm1_ms((double)lsum);
    res->slice_ms = fm1_ms((double)lmax);
    res->missed = tr_missed;
    if (dump) {                                 /* the lesson, for tools/trio_finetune.py */
        FILE *f = fopen(dump, "w");
        int32_t first, last;
        const float *h0 = tr_teach_h0();
        int n = 0;
        if (!f || !tr_teach_span(&first, &last))
            return;
        for (b = first; b <= last; b++) {
            int sk;
            n += tr_teach_in(b, &sk) && tr_log_taught(b) >= 0;
        }
        fprintf(f, "# TRIO: a lesson as tests/host/teach_test.c --dump wrote it (%d BPM, seed %u, %s): the state it\n"
                   "# starts from, then each beat decided while teaching, a line: beat, its inputs as decide_bass built\n"
                   "# them (ptok root qual mask nroot nmask tochg since bib feel pskip), what the player played on it\n"
                   "# (tok skip)\n", bpm, (unsigned)seed, chords ? "over the chords" : "over the form");
        fprintf(f, "h0");
        for (k = 0; k < BASS_H; k++)
            fprintf(f, " %.9g", (double)h0[k]);
        fprintf(f, "\nbeats %d\n", n);
        for (b = first; b <= last; b++) {
            int sk;
            const bass_in_t *in = tr_teach_in(b, &sk);
            if (!in || tr_log_taught(b) < 0)
                continue;
            fprintf(f, "%d %d %d %d %u %d %u %u %u %u %u %u %d %d\n", (int)b, in->ptok, in->root, in->qual,
                    (unsigned)in->mask, in->nroot, (unsigned)in->nmask, in->tochg, in->since, in->bib, in->feel,
                    in->pskip, tr_log_taught(b), sk);
        }
        fclose(f);
    }
}

int main(int argc, char **argv)
{
    const char *prog = "autumn", *dump = 0;
    int bpm = 160, seeds = 5, chords = 0, teach = 2, i, a, fails = 0;
    result_t r, m;
    for (i = 1, a = 0; i < argc; i++)
        if (!strcmp(argv[i], "--chords"))
            chords = 1;
        else if (!strcmp(argv[i], "--two"))
            two = 1;
        else if (!strcmp(argv[i], "--fresh"))
            fresh = chords = 1;
        else if (!strcmp(argv[i], "--untimed"))
            untimed = 1;
        else if (!strcmp(argv[i], "--freedom") && i + 1 < argc)
            freedom = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc)
            dump = argv[++i];
        else if (!strcmp(argv[i], "--teach") && i + 1 < argc)
            teach = atoi(argv[++i]);
        else if (a == 0 && ++a)
            prog = argv[i];
        else if (a == 1 && ++a)
            bpm = atoi(argv[i]);
        else if (a == 2 && ++a)
            seeds = atoi(argv[i]);
    load_chart(!strcmp(prog, "blues") ? BLUES : !strcmp(prog, "rhythm") ? RHYTHM : AUTUMN, 1);
    L = nbeats;
    make_line();
    printf("%s at %d BPM%s, FREEDOM %d: %d beats a chorus; taught %d chorus%s %s; the line R 5 8 and the half step above the "
           "next root%s, a pickup into each change after a bar or more\n", prog, bpm, two ? ", in two" : "", freedom, L,
           teach,
           teach > 1 ? "es" : "", fresh ? "from the first bar, over the pianist's chords (from Ab3 up), no form yet"
                                        : chords ? "over the pianist's chords (from Ab3 up)"
                                                 : "over the form (the pianist let go)",
           two ? " (in two: on 1 and 3, held through 2 and 4)" : "");
    memset(&m, 0, sizeof m);
    for (i = 1; i <= seeds; i++) {
        int fd[2];
        pid_t pid;
        /* each seed a fresh FM-1 (the engine's RAM as at power-on): a process of its own */
        if (pipe(fd) || (pid = fork()) < 0)
            return 2;
        if (!pid) {
            close(fd[0]);
            run(0x9E3779B9u * (uint32_t)i, bpm, teach, chords, i == 1 ? dump : 0, &r);
            if (write(fd[1], &r, sizeof r) != (ssize_t)sizeof r)
                _exit(2);
            _exit(0);
        }
        close(fd[1]);
        if (read(fd[0], &r, sizeof r) != (ssize_t)sizeof r)
            return 2;
        close(fd[0]);
        waitpid(pid, 0, 0);
        printf("seed %d: the taught note %4.1f%% -> %4.1f%% (changes %3.0f%% -> %3.0f%%, between %3.0f%% -> %3.0f%%; "
               "class %3.0f%% -> %3.0f%%); pickups after %3.0f%% where taught, %3.0f%% elsewhere; lesson %d beats, "
               "%.3f -> %.3f nats; learning %.0f ms of a core (slice %.2f ms), done in %.0f ms; missed %u\n",
               i, r.before, r.after, r.b_chg, r.a_chg, r.b_mid, r.a_mid, r.b_pc, r.a_pc, r.skip_yes, r.skip_no,
               r.kept, r.loss0, r.loss1, r.learn_ms, r.slice_ms, r.learn_wall_ms, r.missed);
        m.before += r.before / seeds;
        m.after += r.after / seeds;
        m.b_chg += r.b_chg / seeds;
        m.a_chg += r.a_chg / seeds;
        m.b_mid += r.b_mid / seeds;
        m.a_mid += r.a_mid / seeds;
        m.b_pc += r.b_pc / seeds;
        m.a_pc += r.a_pc / seeds;
        m.skip_yes += r.skip_yes / seeds;
        m.skip_no += r.skip_no / seeds;
        m.learn_ms = r.learn_ms > m.learn_ms ? r.learn_ms : m.learn_ms;
        m.slice_ms = r.slice_ms > m.slice_ms ? r.slice_ms : m.slice_ms;
        m.learn_wall_ms = r.learn_wall_ms > m.learn_wall_ms ? r.learn_wall_ms : m.learn_wall_ms;
        m.missed += r.missed;
        m.kept_min = i == 1 || r.kept < m.kept_min ? r.kept : m.kept_min;
    }
    printf("mean: the taught note %.1f%% -> %.1f%% (changes %.0f%% -> %.0f%%, between %.0f%% -> %.0f%%; class %.0f%% -> "
           "%.0f%%); pickups after %.0f%% where taught, %.0f%% elsewhere; missed %u\n", m.before, m.after, m.b_chg,
           m.a_chg, m.b_mid, m.a_mid, m.b_pc, m.a_pc, m.skip_yes, m.skip_no, m.missed);
    printf("cost: learning at most %.0f ms of a core (slices of at most %.2f ms), done at most %.0f ms after teaching "
           "off; RAM: teaching %d B, learning %d B (the pool)\n", m.learn_ms, m.slice_ms, m.learn_wall_ms,
           tr_teach_bytes(), bass_learn_bytes());
    if (m.missed) {
        printf("FAIL: %u beats missed\n", m.missed);
        fails++;
    }
    if (fresh && m.kept_min < (L * teach < 128 ? L * teach : 128) * 9 / 10) {   /* (no chorus before to compare) */
        printf("FAIL: a lesson kept %d beats\n", m.kept_min);
        fails++;
    } else if (!fresh && m.after < m.before + 15.0) {
        printf("FAIL: the bass agrees with the line %.1f%% after, %.1f%% before\n", m.after, m.before);
        fails++;
    }
    printf(fails ? "teach: FAILED\n" : "teach: ok\n");
    return fails != 0;
}
