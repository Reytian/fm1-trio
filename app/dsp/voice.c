/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's sound (voice.h): every voice is synthesis.
 *   piano  an electric piano: two-operator FM with a tine's bright partial, 10 voices
 *   bass   an upright, plucked: its first twelve partials, each with the level and decay measured in FiloBass's
 *          recordings, and the finger's attack
 *   kit    ride, snare, kick, hi-hat (foot and stick), cross-stick, two toms, crash; the cymbals from a bank of six
 *          square waves at the 808's inharmonic ratios, filtered
 *   room   a four-line feedback delay network
 * One compilation unit, -O2. No doubles, no libm (fastmath.h). */
#include "voice.h"
#include "trio_nn.h"                            /* the drum pieces' numbers */
#include "fastmath.h"

#if defined(TNN_HOST) || defined(PLAT_HOST)
#define V_POOL
#else
#define V_POOL __attribute__((section(".pool")))
#endif

static uint32_t v_noise_s = 0x1234567u;
static inline float noise(void)                 /* white, -1..1 */
{
    uint32_t x = v_noise_s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    v_noise_s = x;
    return (float)(int32_t)x * (1.0f / 2147483648.0f);
}

static float sintab[1025] V_POOL;
static inline float sin_u(uint32_t ph)          /* sin(2 pi ph / 2^32), interpolated from the table */
{
    uint32_t i = ph >> 22;
    float f = (float)(ph & 0x3FFFFFu) * (1.0f / 4194304.0f);
    return sintab[i] + f * (sintab[i + 1] - sintab[i]);
}
static inline uint32_t inc_of(float hz) { return (uint32_t)(hz * (4294967296.0f / V_SR)); }
static inline float decay_of(float seconds) { return fm_expf(-1.0f / (seconds * V_SR)); }
static inline float mtof(float note) { return 440.0f * fm_exp2f((note - 69.0f) * (1.0f / 12.0f)); }

/* ------------------------------------------------------------------ filters */
typedef struct {                                /* a state-variable filter, topology-preserving (Zavalishin) */
    float g, k, a1, a2, a3, ic1, ic2;
} svf_t;
static void svf_set(svf_t *f, float hz, float q)
{
    float g = fm_sinf(FM_PI * hz / V_SR) / fm_cosf(FM_PI * hz / V_SR);
    f->g = g;
    f->k = 1.0f / q;
    f->a1 = 1.0f / (1.0f + g * (g + f->k));
    f->a2 = g * f->a1;
    f->a3 = g * f->a2;
}
/* one sample: *bp, *hp out (the lowpass is returned) */
static inline float svf(svf_t *f, float x, float *bp, float *hp)
{
    float v3 = x - f->ic2, v1 = f->a1 * f->ic1 + f->a2 * v3, v2 = f->ic2 + f->a2 * f->ic1 + f->a3 * v3;
    f->ic1 = 2.0f * v1 - f->ic1;
    f->ic2 = 2.0f * v2 - f->ic2;
    *bp = v1;
    *hp = x - f->k * v1 - v2;
    return v2;
}

/* ------------------------------------------------------------------ piano */
typedef struct {
    uint32_t pc, pm, pt, ic, im, it;            /* carrier, modulator, tine: phases and increments */
    float att, dec, dmul, idx, imul, tine, tmul, peak, gl, gr;
    int note;
    uint8_t held, on;
    uint32_t age;
} ep_t;
static ep_t ep[V_NEP];
static uint32_t ep_clock;
static const float ATT_K = 0.02f;               /* the attack's one-pole: ~1 ms */

void v_ep_on(int note, float vel)
{
    int i, pick = -1;
    ep_t *e;
    float d, f = mtof((float)note);
    for (i = 0; i < V_NEP; i++)                 /* the same note again: that voice */
        if (ep[i].on && ep[i].note == note)
            pick = i;
    for (i = 0; i < V_NEP && pick < 0; i++)
        if (!ep[i].on)
            pick = i;
    if (pick < 0) {                             /* steal: the oldest released, else the oldest */
        uint32_t best = 0xFFFFFFFFu;
        for (i = 0; i < V_NEP; i++)
            if (!ep[i].held && ep[i].age < best)
                best = ep[i].age, pick = i;
        for (i = 0; i < V_NEP && pick < 0; i++)
            if (ep[i].age < best)
                best = ep[i].age, pick = i;
    }
    e = &ep[pick];
    e->note = note;
    e->ic = inc_of(f);
    e->im = inc_of(f);
    e->it = inc_of(f * 14.0f < 9000.0f ? f * 14.0f : 9000.0f);
    e->pc = e->pm = e->pt = 0;
    d = 3.2f * fm_exp2f(-(float)(note - 48) * (1.0f / 30.0f));
    e->dmul = decay_of(d < 0.5f ? 0.5f : d > 6.0f ? 6.0f : d);
    e->att = e->on ? e->att * 0.5f : 0.0f;
    e->dec = 1.0f;
    e->idx = 0.9f + 1.6f * vel;
    e->imul = decay_of(0.35f);
    e->tine = 0.10f + 0.20f * vel;
    e->tmul = decay_of(0.03f);
    e->peak = 0.06f + 0.14f * vel;
    {
        float p = (float)(note - 60) * (1.0f / 48.0f);
        p = p < -0.6f ? -0.6f : p > 0.6f ? 0.6f : p;
        e->gl = 0.7f - 0.3f * p;
        e->gr = 0.7f + 0.3f * p;
    }
    e->held = 1;
    e->on = 1;
    e->age = ++ep_clock;
}

void v_ep_off(int note)
{
    int i;
    for (i = 0; i < V_NEP; i++)
        if (ep[i].on && ep[i].held && ep[i].note == note) {
            ep[i].held = 0;
            ep[i].dmul = decay_of(0.10f);
        }
}

void v_ep_all_off(void)
{
    int i;
    for (i = 0; i < V_NEP; i++)
        if (ep[i].on) {
            ep[i].held = 0;
            ep[i].dmul = decay_of(0.06f);
        }
}

/* ------------------------------------------------------------------ bass */
/* An upright, plucked, built from its partials: each of the first twelve harmonics a sine with its own level and
 * decay, as measured in FiloBass's recordings (walking lines from 48 tunes; tools/bass_measure.py), by register and
 * interpolated between registers. Then the finger: a rise of a few ms and a short dull thud of noise. The partials
 * all lie below 2.5 kHz, so they run at a quarter of the rate and are interpolated. No saturation: a real upright's
 * fundamental carries most of its energy, and that is what makes it boom rather than buzz. */
#define BK 20                                   /* partials */
#define BQ 4                                    /* their rate: V_SR / BQ */
#define BNB 5                                   /* the registers measured */
#include "bass_tab.h"                           /* B_PITCH[BNB], B_LEVEL[BNB][BK] (dB), B_DECAY[BNB][BK] (dB/s) */
typedef struct {
    uint32_t ph[BK], inc[BK];
    float amp[BK], mul[BK], xb[BK];             /* each partial's level, decay, and extra brightness at the attack */
    float rs, rsk, rf, ex;                      /* the rises (the fundamental's slower: the body takes a moment), the
                                                 * attack's brightness, fading */
    float rel, relmul;                          /* the release when the next note stops it */
    float thump, thmul, tlp, tlp2;              /* the finger's thud: its level, its decay, its lowpass (two poles) */
    int nk;
    uint8_t on;
} bv_t;
static bv_t bv[2];
static int bv_next;
static float b_y0, b_y1;                        /* the partials' last two samples (quarter rate), interpolated */
static int b_sub;
static float b_rfk, b_exmul;                    /* the upper partials' rise; the attack brightness's decay (quarter rate) */
#define B_RATE (V_SR / (float)BQ)
static inline float qdecay_of(float seconds) { return fm_expf(-1.0f / (seconds * B_RATE)); }
#ifndef B_GAIN
#define B_GAIN 0.56f                            /* the level: as loud as the string it replaced (K-weighted) */
#endif
#ifndef B_THUMP
#define B_THUMP 0.55f                           /* the thud, re. the note (its two-pole lowpass takes about half) */
#endif
#ifndef B_RISE_LO
#define B_RISE_LO 0.030f                        /* the fundamental's rise (s) at E1 ... */
#endif
#ifndef B_RISE_HI
#define B_RISE_HI 0.003f                        /* ... and from D3 up */
#endif
#ifndef B_RISE_UP
#define B_RISE_UP 0.003f                        /* the other partials' rise */
#endif
#ifndef B_XB
#define B_XB 1.8f                               /* the attack's extra brightness: partial k's level x (1 + B_XB k/11) */
#endif
#ifndef B_XT
#define B_XT 0.015f                             /* ... fading in this long (s) */
#endif

static void bass_release(bv_t *s, float seconds)
{
    if (s->on)
        s->relmul = qdecay_of(seconds);
}

void v_bass_on(int note, float vel, int ghost)
{
    bv_t *s;
    float f = mtof((float)note), x, w, g, rise;
    int k, b0;
    bass_release(&bv[bv_next], 0.012f);         /* the note before stops as this one starts */
    bass_release(&bv[bv_next ^ 1], 0.010f);
    s = &bv[bv_next ^ 1];
    bv_next ^= 1;
    x = ((float)note - B_PITCH[0]) / (B_PITCH[1] - B_PITCH[0]);   /* the register, between the measured ones */
    x = x < 0.0f ? 0.0f : x > (float)(BNB - 1) ? (float)(BNB - 1) : x;
    b0 = (int)x;
    if (b0 > BNB - 2)
        b0 = BNB - 2;
    w = x - (float)b0;
    g = B_GAIN * (0.55f + 0.45f * vel) * (ghost ? 0.40f : 1.0f);
    s->nk = 0;
    for (k = 0; k < BK; k++) {
        float hz = f * (float)(k + 1);
        float lv = B_LEVEL[b0][k] + w * (B_LEVEL[b0 + 1][k] - B_LEVEL[b0][k]);
        float dc = B_DECAY[b0][k] + w * (B_DECAY[b0 + 1][k] - B_DECAY[b0][k]);
        if (hz > 1600.0f)
            break;
        lv += (vel - 0.8f) * 10.0f * (float)k * (1.0f / (float)(BK - 1));   /* harder: brighter */
        if (ghost) {                            /* a dead note: the string muted, a short dull thud */
            lv -= (float)k * 1.5f;
            dc = -60.0f / 0.07f;
        }
        s->ph[k] = 0;
        s->inc[k] = (uint32_t)(hz * (4294967296.0f / B_RATE));
        s->amp[k] = g * fm_exp2f(lv * 0.16609640f);
        s->mul[k] = fm_exp2f(dc * 0.16609640f * (1.0f / B_RATE));
        s->xb[k] = B_XB * (float)k * (1.0f / (float)(BK - 1));
        s->nk = k + 1;
    }
    rise = B_RISE_LO + (B_RISE_HI - B_RISE_LO) * (((float)note - 28.0f) * (1.0f / 22.0f));
    rise = rise < B_RISE_HI ? B_RISE_HI : rise;
    s->rsk = 1.0f - fm_expf(-1.0f / (rise * B_RATE));
    s->rs = 0.0f;
    s->rf = 0.0f;
    s->ex = 1.0f;
    s->rel = 1.0f;
    s->relmul = 1.0f;
    s->thump = g * B_THUMP * (ghost ? 1.8f : 1.0f);
    s->thmul = decay_of(0.006f);
    s->tlp = s->tlp2 = 0.0f;
    s->on = 1;
}

void v_bass_damp(void)
{
    bass_release(&bv[0], 0.03f);
    bass_release(&bv[1], 0.03f);
}

static inline float bv_tick(bv_t *s)            /* a quarter-rate sample of a voice's partials */
{
    float y, z = 0.0f, ex = s->ex;
    int k;
    s->rs += s->rsk * (1.0f - s->rs);
    s->rf += b_rfk * (1.0f - s->rf);
    s->ph[0] += s->inc[0];
    y = s->amp[0] * s->rs * sin_u(s->ph[0]);
    s->amp[0] *= s->mul[0];
    for (k = 1; k < s->nk; k++) {
        s->ph[k] += s->inc[k];
        z += s->amp[k] * (1.0f + s->xb[k] * ex) * sin_u(s->ph[k]);
        s->amp[k] *= s->mul[k];
    }
    s->ex = ex * b_exmul;
    s->rel *= s->relmul;
    if (s->rel < 1e-4f || s->amp[0] < 2e-5f)
        s->on = 0;
    return (y + z * s->rf) * s->rel;
}

static inline float thump_tick(bv_t *s)         /* the finger's thud, full rate: noise under ~700 Hz, 12 dB/octave */
{
    s->tlp += 0.10f * (noise() - s->tlp);
    s->tlp2 += 0.10f * (s->tlp - s->tlp2);
    s->thump *= s->thmul;
    return s->tlp2 * s->thump;
}

/* ------------------------------------------------------------------ kit */
typedef struct {
    float e, mul, e2, mul2, v, pitch, pmul, f0;
    svf_t f, f2;
    uint32_t ph;
    uint8_t on;
} drum_t;
static drum_t dr[TNN_NPIECE];
static uint32_t mph[6], minc[6];                /* the metal bank */
static const float METAL_HZ[6] = {205.3f, 304.4f, 369.6f, 522.7f, 540.0f, 800.0f};
static const float PAN[TNN_NPIECE] = {0.35f, -0.12f, 0.0f, -0.35f, -0.35f, -0.12f, 0.25f, 0.4f, -0.45f};

void v_drum(int p, float vel)
{
    drum_t *d;
    if (p < 0 || p >= TNN_NPIECE)
        return;
    d = &dr[p];
    d->v = vel;
    d->on = 1;
    d->e = 1.0f;
    d->e2 = 1.0f;
    d->pitch = 1.0f;
    d->ph = 0;
    switch (p) {
    case P_RIDE:
        d->mul = decay_of(0.07f);               /* the stick's ping */
        d->mul2 = decay_of(1.3f);               /* the wash */
        break;
    case P_HHF:
        d->mul = decay_of(0.022f);
        d->mul2 = decay_of(0.012f);
        break;
    case P_HHO:
        d->mul = decay_of(0.20f);
        d->mul2 = decay_of(0.02f);
        break;
    case P_CRASH:
        d->mul = decay_of(1.6f);
        d->mul2 = decay_of(0.004f);             /* (the attack's swell) */
        d->e2 = 0.0f;
        break;
    case P_SNARE:
        d->mul = decay_of(0.12f);
        d->mul2 = decay_of(0.05f);
        d->pmul = decay_of(0.03f);
        d->f0 = 200.0f;
        break;
    case P_XSTK:
        d->mul = decay_of(0.012f);
        d->mul2 = decay_of(0.035f);
        d->f0 = 900.0f;
        d->pmul = 1.0f;
        break;
    case P_KICK:
        d->mul = decay_of(0.22f);
        d->mul2 = decay_of(0.006f);
        d->pmul = decay_of(0.025f);
        d->f0 = 52.0f;
        break;
    case P_TOMH:
    case P_TOML:
        d->mul = decay_of(0.28f);
        d->mul2 = decay_of(0.03f);
        d->pmul = decay_of(0.12f);
        d->f0 = p == P_TOMH ? 175.0f : 110.0f;
        break;
    }
}

/* one sample of the kit, stereo */
static inline void kit_tick(float *ol, float *or_)
{
    float metal = 0.0f, l = 0.0f, r = 0.0f, bp, hp, s;
    int k, any_metal = dr[P_RIDE].on | dr[P_HHF].on | dr[P_HHO].on | dr[P_CRASH].on;
    if (any_metal) {
        for (k = 0; k < 6; k++) {
            mph[k] += minc[k];
            metal += (mph[k] >> 31) ? 1.0f : -1.0f;
        }
        metal *= (1.0f / 6.0f);
    }
#define OUT(p, x)                                                                                                 \
    do {                                                                                                          \
        float x_ = (x);                                                                                           \
        l += x_ * (0.5f - 0.5f * PAN[p]);                                                                         \
        r += x_ * (0.5f + 0.5f * PAN[p]);                                                                         \
    } while (0)
    if (dr[P_RIDE].on) {
        drum_t *d = &dr[P_RIDE];
        svf(&d->f, 0.55f * metal + 0.45f * noise(), &bp, &hp);
        s = bp * (1.1f * d->e + 0.42f * d->e2) * d->v;
        d->e *= d->mul;
        d->e2 *= d->mul2;
        if (d->e2 < 1e-4f)
            d->on = 0;
        OUT(P_RIDE, 1.5f * s);
    }
    if (dr[P_HHF].on) {
        drum_t *d = &dr[P_HHF];
        svf(&d->f, metal, &bp, &hp);
        svf(&d->f2, noise(), &bp, &s);          /* (the foot's low chick: the bandpass) */
        s = (hp * d->e + 0.25f * bp * d->e2) * d->v;
        d->e *= d->mul;
        d->e2 *= d->mul2;
        if (d->e < 1e-4f)
            d->on = 0;
        OUT(P_HHF, 1.6f * s);
    }
    if (dr[P_HHO].on) {
        drum_t *d = &dr[P_HHO];
        svf(&d->f, metal, &bp, &hp);
        s = hp * d->e * d->v;
        d->e *= d->mul;
        if (d->e < 1e-4f)
            d->on = 0;
        OUT(P_HHO, 1.2f * s);
    }
    if (dr[P_CRASH].on) {
        drum_t *d = &dr[P_CRASH];
        svf(&d->f, 0.5f * metal + 0.5f * noise(), &bp, &hp);
        d->e2 += (1.0f - d->e2) * 0.05f;
        s = hp * d->e * d->e2 * d->v;
        d->e *= d->mul;
        if (d->e < 1e-4f)
            d->on = 0;
        OUT(P_CRASH, 1.0f * s);
    }
    if (dr[P_SNARE].on) {
        drum_t *d = &dr[P_SNARE];
        float f = d->f0 * (0.86f + 0.14f * d->pitch);
        svf(&d->f, noise(), &bp, &hp);
        d->ph += inc_of(f);
        s = (0.9f * bp * d->e + 0.45f * sin_u(d->ph) * d->e2) * d->v;
        d->e *= d->mul;
        d->e2 *= d->mul2;
        d->pitch *= d->pmul;
        if (d->e < 1e-4f)
            d->on = 0;
        OUT(P_SNARE, 1.4f * s);
    }
    if (dr[P_XSTK].on) {
        drum_t *d = &dr[P_XSTK];
        svf(&d->f, noise(), &bp, &hp);
        d->ph += inc_of(d->f0);
        s = (1.2f * bp * d->e + 0.6f * sin_u(d->ph) * d->e2) * d->v;
        d->e *= d->mul;
        d->e2 *= d->mul2;
        if (d->e2 < 1e-4f)
            d->on = 0;
        OUT(P_XSTK, 1.2f * s);
    }
    if (dr[P_KICK].on) {
        drum_t *d = &dr[P_KICK];
        d->ph += inc_of(d->f0 * (1.0f + 1.2f * d->pitch));
        s = (sin_u(d->ph) * d->e + 0.15f * noise() * d->e2) * d->v;
        d->e *= d->mul;
        d->e2 *= d->mul2;
        d->pitch *= d->pmul;
        if (d->e < 1e-4f)
            d->on = 0;
        OUT(P_KICK, 1.3f * s);
    }
    for (k = P_TOMH; k <= P_TOML; k++)
        if (dr[k].on) {
            drum_t *d = &dr[k];
            svf(&d->f, noise(), &bp, &hp);
            d->ph += inc_of(d->f0 * (0.9f + 0.1f * d->pitch));
            s = (sin_u(d->ph) * d->e + 0.3f * bp * d->e2) * d->v;
            d->e *= d->mul;
            d->e2 *= d->mul2;
            d->pitch *= d->pmul;
            if (d->e < 1e-4f)
                d->on = 0;
            OUT(k, 1.1f * s);
        }
#undef OUT
    *ol = l;
    *or_ = r;
}

/* ------------------------------------------------------------------ room */
#define RV_N 4
static const int RV_LEN[RV_N] = {1213, 1447, 1693, 1951};
static float rv_buf[1213 + 1447 + 1693 + 1951] V_POOL;
static int rv_w[RV_N], rv_base[RV_N];
static float rv_lp[RV_N];

static inline void room_tick(float in, float *l, float *r, float fb)
{
    float o[RV_N], m[RV_N];
    int i;
    for (i = 0; i < RV_N; i++) {
        float x = rv_buf[rv_base[i] + rv_w[i]];
        rv_lp[i] += 0.45f * (x - rv_lp[i]);    /* the walls take the highs */
        o[i] = rv_lp[i];
    }
    m[0] = 0.5f * (o[0] + o[1] + o[2] + o[3]);   /* Householder-like mixing (a 4x4 Hadamard / 2) */
    m[1] = 0.5f * (o[0] - o[1] + o[2] - o[3]);
    m[2] = 0.5f * (o[0] + o[1] - o[2] - o[3]);
    m[3] = 0.5f * (o[0] - o[1] - o[2] + o[3]);
    for (i = 0; i < RV_N; i++) {
        rv_buf[rv_base[i] + rv_w[i]] = in + fb * m[i];
        if (++rv_w[i] >= RV_LEN[i])
            rv_w[i] = 0;
    }
    *l = o[0] + o[2];
    *r = o[1] + o[3];
}

/* ------------------------------------------------------------------ */
void v_init(void)
{
    int i, b = 0;
    for (i = 0; i <= 1024; i++)
        sintab[i] = fm_sinf(2.0f * FM_PI * (float)i / 1024.0f);
    for (i = 0; i < 6; i++)
        minc[i] = inc_of(METAL_HZ[i] * 1.0f);
    svf_set(&dr[P_RIDE].f, 5200.0f, 0.9f);
    svf_set(&dr[P_HHF].f, 7800.0f, 0.7f);
    svf_set(&dr[P_HHF].f2, 900.0f, 1.2f);
    svf_set(&dr[P_HHO].f, 7000.0f, 0.7f);
    svf_set(&dr[P_CRASH].f, 4800.0f, 0.7f);
    svf_set(&dr[P_SNARE].f, 2800.0f, 0.8f);
    svf_set(&dr[P_XSTK].f, 1800.0f, 2.0f);
    svf_set(&dr[P_TOMH].f, 1300.0f, 1.0f);
    svf_set(&dr[P_TOML].f, 900.0f, 1.0f);
    b_rfk = 1.0f - fm_expf(-1.0f / (B_RISE_UP * B_RATE));
    b_exmul = qdecay_of(B_XT);
    for (i = 0; i < RV_N; i++) {
        rv_base[i] = b;
        b += RV_LEN[i];
    }
    v_panic();
}

void v_panic(void)
{
    int i, j;
    for (i = 0; i < V_NEP; i++)
        ep[i].on = 0;
    for (i = 0; i < 2; i++) {
        bv[i].on = 0;
        bv[i].thump = 0.0f;
    }
    b_y0 = b_y1 = 0.0f;
    for (i = 0; i < TNN_NPIECE; i++)
        dr[i].on = 0;
    for (j = 0; j < (int)(sizeof rv_buf / sizeof rv_buf[0]); j++)
        rv_buf[j] = 0.0f;
}

void v_render(float *lo, float *ro, uint32_t n, const v_mix_t *mix)
{
    uint32_t i;
    int k;
    float fb = 0.70f + 0.18f * mix->room;
    for (i = 0; i < n; i++) {
        float pl = 0.0f, pr = 0.0f, b = 0.0f, kl, kr, rl, rr, send;
        for (k = 0; k < V_NEP; k++) {
            ep_t *e = &ep[k];
            float m, c, a;
            if (!e->on)
                continue;
            e->pm += e->im;
            e->pc += e->ic;
            e->pt += e->it;
            e->att += ATT_K * (1.0f - e->att);
            m = sin_u(e->pm) * e->idx;
            c = sin_u(e->pc + (uint32_t)(int32_t)(m * 683565275.6f));
            a = e->att * e->dec * e->peak;
            c = (c + sin_u(e->pt) * e->tine) * a;
            e->dec *= e->dmul;
            e->idx = 0.25f + (e->idx - 0.25f) * e->imul;
            e->tine *= e->tmul;
            if (e->dec < 1e-4f)
                e->on = 0;
            pl += c * e->gl;
            pr += c * e->gr;
        }
        if (b_sub == 0) {                       /* the partials: a quarter-rate sample */
            float y = 0.0f;
            if (bv[0].on)
                y += bv_tick(&bv[0]);
            if (bv[1].on)
                y += bv_tick(&bv[1]);
            b_y0 = b_y1;
            b_y1 = y;
        }
        b = b_y0 + (b_y1 - b_y0) * ((float)b_sub * (1.0f / (float)BQ));
        b_sub = (b_sub + 1) & (BQ - 1);
        if (bv[0].thump > 1e-5f)
            b += thump_tick(&bv[0]);
        if (bv[1].thump > 1e-5f)
            b += thump_tick(&bv[1]);
        kit_tick(&kl, &kr);
        pl *= mix->piano;
        pr *= mix->piano;
        b *= mix->bass;
        kl *= mix->drums;
        kr *= mix->drums;
        send = (0.30f * (pl + pr) + 0.10f * b + 0.22f * (kl + kr)) * mix->room;
        room_tick(send, &rl, &rr, fb);
        lo[i] = pl + b + kl + 0.55f * rl;
        ro[i] = pr + b + kr + 0.55f * rr;
    }
}
