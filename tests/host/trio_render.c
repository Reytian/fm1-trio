/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO without its screen: a pianist (a progression's chords as MIDI) into the engine, the main loop run between
 * audio blocks as on the FM-1, and out come a WAV, the band's beats, how the bass walked (the root when the chord
 * changes, the approach into it, steps), the drums a bar, and the cost (instructions -> an FM-1 core's load, by
 * the calibration in MIN-1's notes: 0.030 % per instruction a frame, from X0X's PERF TEST).
 *
 *   trio_render PROG BPM CHORUSES OUT.wav [early|ontime|late] [rooted|rootless] [CSV]
 *   PROG: autumn (G minor, AABC), blues (F), rhythm (Bb)
 *
 * Build (tests/run_tests.sh): cc -O2 -ffp-contract=off -fno-vectorize -fno-slp-vectorize -DTNN_HOST
 *   -Iapp/dsp -Ibase/include tests/host/trio_render.c app/dsp/trio.c app/dsp/voice.c app/dsp/trio_nn.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "trio.h"
int tr_form_debug(int *lag, int *len);
int tr_log_chord(int32_t b);
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

#include "charts.h"

static void wav_header(FILE *f, unsigned frames)
{
    unsigned data = frames * 4, v;
    fwrite("RIFF", 1, 4, f);
    v = 36 + data;
    fwrite(&v, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    v = 16;
    fwrite(&v, 4, 1, f);
    {
        unsigned short fmt[2] = {1, 2};
        unsigned rate = 44100, br = 44100 * 4;
        unsigned short ba[2] = {4, 16};
        fwrite(fmt, 2, 2, f);
        fwrite(&rate, 4, 1, f);
        fwrite(&br, 4, 1, f);
        fwrite(ba, 2, 2, f);
    }
    fwrite("data", 1, 4, f);
    fwrite(&data, 4, 1, f);
}

int main(int argc, char **argv)
{
    const char *prog = argc > 1 ? argv[1] : "autumn", *wav = argc > 4 ? argv[4] : "trio.wav";
    int bpm = argc > 2 ? atoi(argv[2]) : 160, choruses = argc > 3 ? atoi(argv[3]) : 3;
    int early = argc > 5 && !strcmp(argv[5], "early"), late = argc > 5 && !strcmp(argv[5], "late");
    int rooted = argc > 6 && !strcmp(argv[6], "rooted");
    const char *csv = argc > 7 ? argv[7] : NULL;
    int32_t lr[512];
    short pcm[512];
    float spb;
    unsigned frames = 0, total, b, k;
    unsigned long long ir = 0, is = 0, i0;
    int held[8], nheld = 0, ev_beat = 0, last_trace = -1;
    FILE *f;
    load_chart(!strcmp(prog, "blues") ? BLUES : !strcmp(prog, "rhythm") ? RHYTHM : AUTUMN, choruses);
    tr_init();
    if (getenv("TRIO_SEED"))
        tr_seed((uint32_t)atoi(getenv("TRIO_SEED")));
    tr_set.bpm = (uint16_t)bpm;
    spb = 44100.0f * 60.0f / (float)bpm;
    total = (unsigned)((float)(nbeats + 4 + 4) * spb);
    f = fopen(wav, "wb");
    if (!f)
        return 1;
    wav_header(f, total);
    tr_play(1);
    while (frames < total) {
        /* the pianist: the chart's chords as they change, on time, an eighth early, or a 16th late */
        while (ev_beat < nbeats) {
            float at = ((float)ev_beat + 4.0f + (early ? -0.33f : late ? 0.12f : 0.0f)) * spb;
            if (ev_beat > 0 && ch_root[ev_beat] == ch_root[ev_beat - 1] && ch_qual[ev_beat] == ch_qual[ev_beat - 1]) {
                ev_beat++;
                continue;
            }
            if (at > (float)(frames + 256u))
                break;
            for (k = 0; k < (unsigned)nheld; k++)
                tr_midi(0x80, (uint32_t)held[k], 0);
            nheld = voicing(ch_root[ev_beat], ch_qual[ev_beat], rooted, held);
            for (k = 0; k < (unsigned)nheld; k++)
                tr_midi(0x90, (uint32_t)held[k], 70u + (uint32_t)(k * 7u));
            ev_beat++;
        }
        i0 = insns();
        tr_render(lr, 256, 4096);
        ir += insns() - i0;
        i0 = insns();
        while (tr_service())
            ;
        is += insns() - i0;
        if (getenv("FORM_TRACE") && tr_beat >= 0 && tr_beat % 16 == 0 && tr_beat != last_trace) {
            int lag, len;
            int up = tr_form_debug(&lag, &len);
            last_trace = tr_beat;
            printf("  beat %d (bar %d): form lag %d beats, matched %d, drops so far %d\n", (int)tr_beat,
                   (int)tr_beat / 4, lag, len, up);
        }
        for (k = 0; k < 512; k++) {
            int32_t v = lr[k] >> 8;
            pcm[k] = (short)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
        }
        fwrite(pcm, 2, 512, f);
        frames += 256;
    }
    fclose(f);

    /* how the bass walked */
    {
        int changes = 0, roots = 0, tones = 0, pitched = 0, steps = 0, moves = 0, holds = 0, prev = -1;
        int app[25] = {0}, napp = 0, hits[TNN_NPIECE] = {0};
        FILE *c = csv ? fopen(csv, "w") : NULL;
        if (c)
            fprintf(c, "beat,root,qual,bass_tok,bass_pitch,drums,band\n");
        for (b = 0; b < (unsigned)nbeats; b++) {
            int tok = tr_log_bass((int32_t)b), p = tok >= 2 ? TNN_LO + tok - 2 : -1;
            int ch = b > 0 && (ch_root[b] != ch_root[b - 1] || ch_qual[b] != ch_qual[b - 1]);
            static const uint16_t QM[7] = {0x891, 0x491, 0x489, 0x449, 0x249, 0x4A1, 0x111};
            uint32_t m = QM[ch_qual[b]], mask = ((m << ch_root[b]) | (m >> (12 - ch_root[b]))) & 0xFFFu;
            if (tok == TNN_HOLD)
                holds++;
            if (p >= 0) {
                pitched++;
                tones += (int)(mask >> (p % 12) & 1u);
                if (ch) {
                    changes++;
                    roots += p % 12 == ch_root[b];
                    if (prev >= 0 && p - prev >= -12 && p - prev <= 12) {
                        app[p - prev + 12]++;
                        napp++;
                    }
                }
                if (prev >= 0) {
                    moves++;
                    steps += abs(p - prev) == 1 || abs(p - prev) == 2;
                }
                prev = p;
            }
            for (k = 0; k < 3; k++) {
                uint16_t h = tr_log_drums((int32_t)(b * 3 + k));
                int q;
                for (q = 0; q < TNN_NPIECE; q++)
                    hits[q] += h >> q & 1u;
            }
            if (c)
                fprintf(c, "%u,%d,%d,%d,%d,%u,%d\n", b, ch_root[b], ch_qual[b], tok, p,
                        tr_log_drums((int32_t)(b * 3)) | tr_log_drums((int32_t)(b * 3 + 1)) << 9,
                        tr_log_chord((int32_t)b));
        }
        if (c)
            fclose(c);
        printf("%s at %d bpm, %d choruses (%d beats), chords %s, %s\n", prog, bpm, choruses, nbeats,
               early ? "an eighth early" : late ? "a 16th late" : "on the beat", rooted ? "with roots" : "rootless");
        printf("bass: %d notes, %d holds; on a change the root %.0f%% (%d changes), chord tones on beats %.0f%%, steps %.0f%%\n",
               pitched, holds, 100.0 * roots / (changes ? changes : 1), changes, 100.0 * tones / (pitched ? pitched : 1),
               100.0 * steps / (moves ? moves : 1));
        printf("approach into a change: -1 %.0f%%  +1 %.0f%%  -2 %.0f%%  +2 %.0f%%  +5 %.0f%%  -7 %.0f%%  0 %.0f%%\n",
               100.0 * app[11] / (napp ? napp : 1), 100.0 * app[13] / (napp ? napp : 1),
               100.0 * app[10] / (napp ? napp : 1), 100.0 * app[14] / (napp ? napp : 1),
               100.0 * app[17] / (napp ? napp : 1), 100.0 * app[5] / (napp ? napp : 1),
               100.0 * app[12] / (napp ? napp : 1));
        printf("drums a bar:");
        {
            static const char *const N[TNN_NPIECE] = {"ride", "snare", "kick", "hh-foot", "hh", "xstick", "tom-hi",
                                                      "tom-lo", "crash"};
            for (k = 0; k < TNN_NPIECE; k++)
                printf(" %s %.2f", N[k], 4.0 * hits[k] / (double)nbeats);
        }
        printf("\nform: %d bars learned; missed %u; network steps %u\n", tr_form_bars(), (unsigned)tr_missed,
               (unsigned)tr_nn_steps());
        printf("load: render %.0f insn/frame (%.1f%%), networks %.0f insn/frame (%.1f%%), together %.1f%% of a core\n",
               (double)ir / frames, 0.030 * (double)ir / frames, (double)is / frames, 0.030 * (double)is / frames,
               0.030 * (double)(ir + is) / frames);
    }
    return 0;
}
