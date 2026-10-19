/* SPDX-License-Identifier: MPL-2.0 */
/* TRIO's networks (trio_nn.h). One compilation unit, -O2, pure C99. */
#include "trio_nn.h"
#include "trio_w.h"
#include "fastmath.h"

#if defined(TNN_HOST) || defined(PLAT_HOST)    /* the tests, fm1-base's simulator and emulator */
#define TNN_POOL
#else
#define TNN_POOL __attribute__((section(".pool")))
#endif

#define BASS_P 64
#define DRUM_P 48
#define HARM_P 32
#define CHORD_HID 64
#define CHORD_NOUT (12 * TNN_NQUAL)
#define NACT 64                                 /* the most non-zero inputs a step has */

/* the weights in RAM (a step reads them all): int8 matrices, float biases, one scale a matrix */
static int8_t bq[sizeof BASS_Q] TNN_POOL, dq[sizeof DRUM_Q] TNN_POOL, hq[sizeof HARM_Q] TNN_POOL,
    cq[sizeof CHORD_Q] TNN_POOL;
static float bf[sizeof BASS_F / sizeof(float)] TNN_POOL, df[sizeof DRUM_F / sizeof(float)] TNN_POOL,
    hf[sizeof HARM_F / sizeof(float)] TNN_POOL, cf[sizeof CHORD_F / sizeof(float)] TNN_POOL;

/* LEARNING (trio_nn.h): the rows learned as kept, a flash object's form: the network they belong to, each row's
 * change from the trained one (int8, in steps of its own size) and its bias */
#define LRN_MAGIC 0x314E4C42u                   /* "BLN1" */
static struct {
    uint32_t magic, net;                        /* "BLN1"; the trained BASS network's fingerprint */
    uint16_t rows, cols;                        /* TNN_LROWS, BASS_H (no rows: nothing learned) */
    uint16_t learnings, rsv;                    /* the lines learned into them */
    float step[TNN_LROWS];
    float bias[TNN_LROWS];
    int8_t q[TNN_LROWS * BASS_H];
} kept TNN_POOL;
typedef char tnn_kept_size_check[sizeof kept == 16u + 8u * TNN_LROWS + TNN_LROWS * BASS_H ? 1 : -1];   /* no padding */
static uint32_t net_id;                         /* the trained BASS network's fingerprint (FNV-1a of its weights) */
static uint8_t learned;                         /* kept's rows in use */
static void kept_head(int rows);

static void copy8(int8_t *d, const int8_t *s, unsigned n)
{
    unsigned i;
    for (i = 0; i < n; i++)
        d[i] = s[i];
}
static void copyf(float *d, const float *s, unsigned n)
{
    unsigned i;
    for (i = 0; i < n; i++)
        d[i] = s[i];
}

static uint32_t fnv(uint32_t h, const void *p, unsigned n)
{
    const uint8_t *b = p;
    while (n--)
        h = (h ^ *b++) * 16777619u;
    return h;
}

void tnn_init(void)
{
    copy8(bq, BASS_Q, sizeof bq);
    copy8(dq, DRUM_Q, sizeof dq);
    copy8(hq, HARM_Q, sizeof hq);
    copy8(cq, CHORD_Q, sizeof cq);
    copyf(bf, BASS_F, sizeof bf / sizeof(float));
    copyf(df, DRUM_F, sizeof df / sizeof(float));
    copyf(hf, HARM_F, sizeof hf / sizeof(float));
    copyf(cf, CHORD_F, sizeof cf / sizeof(float));
    net_id = fnv(fnv(fnv(2166136261u, BASS_Q, sizeof BASS_Q), BASS_F, sizeof BASS_F), BASS_S, sizeof BASS_S);
    learned = 0;                                /* (what it learned comes back from its flash object: bass_learned_check) */
    kept_head(0);
}

void tnn_zero(float *h, int n)
{
    int i;
    for (i = 0; i < n; i++)
        h[i] = 0.0f;
}

/* ------------------------------------------------------------------ kernels */
/* an int8 row by a float vector, n a multiple of 4: four sums, so an in-order FPU waits less on its adds */
static inline float dot8(const int8_t *w, const float *x, int n)
{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int j;
    for (j = 0; j < n; j += 4) {
        a0 += (float)w[j] * x[j];
        a1 += (float)w[j + 1] * x[j + 1];
        a2 += (float)w[j + 2] * x[j + 2];
        a3 += (float)w[j + 3] * x[j + 3];
    }
    return (a0 + a1) + (a2 + a3);
}

/* y = s W x + b, x given by its non-zero entries (column ix[k] with value v[k]) */
static void linear_sparse(const int8_t *w, float s, const float *b, int nin, int nout, const int16_t *ix,
                          const float *v, int n, float *y)
{
    int o, k;
    for (o = 0; o < nout; o++) {
        const int8_t *row = w + o * nin;
        float acc = 0.0f;
        for (k = 0; k < n; k++)
            acc += (float)row[ix[k]] * v[k];
        y[o] = s * acc + b[o];
    }
}

static void linear(const int8_t *w, float s, const float *b, const float *x, int nin, int nout, float *y)
{
    int o;
    for (o = 0; o < nout; o++)
        y[o] = s * dot8(w + o * nin, x, nin) + b[o];
}

float tnn_sigmoid(float x) { return 0.5f + 0.5f * fm_tanhf(0.5f * x); }

/* a GRU step as PyTorch's (gates r, z, n): r = s(Wir x + bir + Whr h + bhr), z likewise,
 * n = tanh(Win x + bin + r (Whn h + bhn)), h = (1 - z) n + z h. nh at most 96. */
static void gru(const int8_t *wi, float si, const int8_t *wh, float sh, const float *bi, const float *bh,
                const float *x, int nx, float *h, int nh)
{
    float gi[3 * 96], gh[3 * 96];
    int k;
    for (k = 0; k < 3 * nh; k++) {
        gi[k] = si * dot8(wi + k * nx, x, nx) + bi[k];
        gh[k] = sh * dot8(wh + k * nh, h, nh) + bh[k];
    }
    for (k = 0; k < nh; k++) {
        float r = tnn_sigmoid(gi[k] + gh[k]), z = tnn_sigmoid(gi[nh + k] + gh[nh + k]);
        float n = fm_tanhf(gi[2 * nh + k] + r * gh[2 * nh + k]);
        h[k] = (1.0f - z) * n + z * h[k];
    }
}

/* a dense vector's non-zero entries */
static int sparse_of(const float *x, int n, int16_t *ix, float *v)
{
    int i, m = 0;
    for (i = 0; i < n && m < NACT; i++)
        if (x[i] != 0.0f) {
            ix[m] = (int16_t)i;
            v[m++] = x[i];
        }
    return m;
}

/* ------------------------------------------------------------------ BASS */
#define ONE(base, i)                                                                                              \
    do {                                                                                                          \
        ix[n] = (int16_t)((base) + (i));                                                                          \
        v[n++] = 1.0f;                                                                                            \
    } while (0)

static int bass_sparse(const bass_in_t *in, int16_t *ix, float *v)
{
    int n = 0, k;
    ONE(BASS_IN_TOK, in->ptok);
    if (in->root >= 0) {
        ONE(BASS_IN_ROOT, in->root);
        for (k = 0; k < 12; k++)
            if (in->mask >> k & 1u)
                ONE(BASS_IN_MASK, k);
        ONE(BASS_IN_QUAL, in->qual);
    } else
        ONE(BASS_IN_NOCH, 0);
    if (in->nroot >= 0) {
        ONE(BASS_IN_NROOT, in->nroot);
        for (k = 0; k < 12; k++)
            if (in->nmask >> k & 1u)
                ONE(BASS_IN_NMASK, k);
    }
    ONE(BASS_IN_TOCHG, (in->tochg < 1 ? 1 : in->tochg > 8 ? 8 : in->tochg) - 1);
    ONE(BASS_IN_SINCE, in->since > 7 ? 7 : in->since);
    ONE(BASS_IN_BIB, in->bib & 3u);
    if (in->feel)
        ONE(BASS_IN_FEEL, 0);
    if (in->pskip)
        ONE(BASS_IN_PSKIP, 0);
    return n;
}

/* the trained output rows: row r of the tokens' head (r < TNN_NTOK), then the pickup's; a weight, a bias */
static const int8_t *w0_row(int r)
{
    return bq + (r < TNN_NTOK ? BASS_HEADS_0_WEIGHT + r * BASS_H : BASS_HEADS_1_WEIGHT);
}
static float w0_scale(int r) { return BASS_S[r < TNN_NTOK ? BASS_HEADS_0_WEIGHT_S : BASS_HEADS_1_WEIGHT_S]; }
static float b0(int r) { return bf[r < TNN_NTOK ? BASS_HEADS_0_BIAS + r : BASS_HEADS_1_BIAS]; }

/* the output layer: the trained rows, or (learned) each the trained row plus its change as kept */
static void bass_heads(const float *h, float *tok, float *skip)
{
    int r;
    for (r = 0; r < TNN_LROWS; r++) {
        float z = w0_scale(r) * dot8(w0_row(r), h, BASS_H);
        z = learned ? z + kept.step[r] * dot8(kept.q + r * BASS_H, h, BASS_H) + kept.bias[r] : z + b0(r);
        if (r < TNN_NTOK)
            tok[r] = z;
        else
            *skip = z;
    }
}

static void bass_run(float *h, const int16_t *ix, const float *v, int n, float *tok, float *skip)
{
    float p[BASS_P];
    linear_sparse(bq + BASS_PROJ_WEIGHT, BASS_S[BASS_PROJ_WEIGHT_S], bf + BASS_PROJ_BIAS, BASS_NIN, BASS_P, ix, v, n,
                  p);
    gru(bq + BASS_GRU_WEIGHT_IH_L0, BASS_S[BASS_GRU_WEIGHT_IH_L0_S], bq + BASS_GRU_WEIGHT_HH_L0,
        BASS_S[BASS_GRU_WEIGHT_HH_L0_S], bf + BASS_GRU_BIAS_IH_L0, bf + BASS_GRU_BIAS_HH_L0, p, BASS_P, h, BASS_H);
    bass_heads(h, tok, skip);
}

void bass_step(float *h, const bass_in_t *in, float *tok, float *skip)
{
    int16_t ix[NACT];
    float v[NACT];
    bass_run(h, ix, v, bass_sparse(in, ix, v), tok, skip);
}

void bass_step_x(float *h, const float *x, float *tok, float *skip)
{
    int16_t ix[NACT];
    float v[NACT];
    bass_run(h, ix, v, sparse_of(x, BASS_NIN, ix, v), tok, skip);
}

void bass_x(const bass_in_t *in, float *x)
{
    int16_t ix[NACT];
    float v[NACT];
    int i, n = bass_sparse(in, ix, v);
    for (i = 0; i < BASS_NIN; i++)
        x[i] = 0.0f;
    for (i = 0; i < n; i++)
        x[ix[i]] = v[i];
}

void bass_allow(uint8_t *allow, int flags)
{
    int k, hold = (flags & TNN_B_TWO) && (flags & TNN_B_ODD) && !(flags & TNN_B_CHANGE);
    for (k = 0; k < TNN_NTOK; k++)
        allow[k] = (uint8_t)(hold ? k == TNN_HOLD : k >= 2 && TNN_LO + k - 2 <= TNN_LINE_HI);
}

void bass_shape(float *lg, uint8_t *allow, int root, uint16_t mask, int flags)
{
    int k;
    bass_allow(allow, flags);
    for (k = 2; k < TNN_NTOK; k++) {
        int pitch = TNN_LO + k - 2;
        if ((flags & TNN_B_CHANGE) && (mask >> (pitch % 12) & 1u))   /* on a change, the chord's own notes, the root most */
            lg[k] += pitch % 12 == root ? 2.5f : 1.5f;
        if (pitch > 45)                         /* and the line kept low: an upright's register */
            lg[k] -= 0.06f * (float)(pitch - 45);
    }
}

/* ------------------------------------------------------------------ LEARNING the player's line (trio_nn.h) */
static float tw[TNN_LROWS * BASS_H + TNN_LROWS] TNN_POOL;   /* the rows being learned (float), then their biases */
static struct {
    const tnn_beat_t *ring;
    int cap, first, n, passes, pass, t;         /* the beats; the passes; where the work is */
    float lr, l2, sum[2];
    int cnt[2];
    float loss[TNN_LPASSES][2];
} L TNN_POOL;

static inline float dotf(const float *w, const float *x, int n)
{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int j;
    for (j = 0; j < n; j += 4) {
        a0 += w[j] * x[j];
        a1 += w[j + 1] * x[j + 1];
        a2 += w[j + 2] * x[j + 2];
        a3 += w[j + 3] * x[j + 3];
    }
    return (a0 + a1) + (a2 + a3);
}

int bass_learnable(const tnn_beat_t *b)
{
    uint8_t allow[TNN_NTOK];
    int two = (b->flags & TNN_B_TWO) != 0;
    if (b->tok < 0 || b->tok >= TNN_NTOK)
        return 0;
    bass_allow(allow, b->flags);
    return (allow[(int)b->tok] ? 1 : 0) | ((two ? (b->flags & TNN_B_ODD) != 0 : b->tok >= 2) ? 2 : 0);
}

int bass_learned(void) { return learned ? kept.learnings : 0; }
int bass_learn_bytes(void) { return (int)(sizeof tw + sizeof L + sizeof kept); }   /* (tests) */
float bass_learn_loss(int pass, int what) { return pass >= 0 && pass < L.passes ? L.loss[pass][what & 1] : 0.0f; }
const float *bass_learn_rows(void) { return tw; }

void bass_unlearn(void)
{
    learned = 0;
    kept_head(0);
}

void bass_learn_begin(const tnn_beat_t *ring, int cap, int first, int n, float lr, float l2, int passes)
{
    int r, j;
    L.ring = ring;
    L.cap = cap > 0 ? cap : 1;
    L.first = first;
    L.n = n < 0 ? 0 : n;
    L.passes = passes < 0 ? 0 : passes < TNN_LPASSES ? passes : TNN_LPASSES;
    L.lr = lr;
    L.l2 = l2;
    L.pass = L.t = 0;
    L.sum[0] = L.sum[1] = 0.0f;
    L.cnt[0] = L.cnt[1] = 0;
    for (r = 0; r < TNN_LROWS; r++) {           /* from the rows in use: on top of what it learned */
        const int8_t *w = w0_row(r);
        float s = w0_scale(r);
        for (j = 0; j < BASS_H; j++)
            tw[r * BASS_H + j] = learned ? s * (float)w[j] + kept.step[r] * (float)kept.q[r * BASS_H + j]
                                         : s * (float)w[j];
        tw[TNN_LROWS * BASS_H + r] = learned ? kept.bias[r] : b0(r);
    }
}

void bass_learn_cancel(void) { L.n = L.pass = L.passes = 0; }

/* row r's SGD step: its gradient g on the beat's state h, and the pull back toward the trained row */
static void row_step(int r, float g, const float *h)
{
    const int8_t *w0 = w0_row(r);
    const float s = w0_scale(r), lr = L.lr, l2 = L.l2;
    float *w = tw + r * BASS_H, *b = tw + TNN_LROWS * BASS_H + r;
    int j;
    for (j = 0; j < BASS_H; j++)
        w[j] -= lr * (g * h[j] + l2 * (w[j] - s * (float)w0[j]));
    *b -= lr * (g + l2 * (*b - b0(r)));
}

/* one SGD step on a beat: the token (the softmax over what the bass may play there, the nudges added) and the pickup */
static void learn_sgd(const tnn_beat_t *bt)
{
    const float *tb = tw + TNN_LROWS * BASS_H;
    float h[BASS_H], z[TNN_NTOK];
    uint8_t allow[TNN_NTOK];
    int what = bass_learnable(bt), y = bt->tok, k, j;
    for (j = 0; j < BASS_H; j++)
        h[j] = (float)bt->h[j] * (1.0f / 32768.0f);
    if (what & 1) {
        float m = -1e30f, S = 0.0f, zy;
        bass_allow(allow, bt->flags);
        for (k = 0; k < TNN_NTOK; k++)
            z[k] = allow[k] ? dotf(tw + k * BASS_H, h, BASS_H) + tb[k] : 0.0f;
        bass_shape(z, allow, bt->root, bt->mask, bt->flags);
        for (k = 0; k < TNN_NTOK; k++)
            if (allow[k] && z[k] > m)
                m = z[k];
        zy = z[y] - m;
        for (k = 0; k < TNN_NTOK; k++)
            if (allow[k]) {
                z[k] = fm_expf(z[k] - m);
                S += z[k];
            }
        L.sum[0] += -(zy - fm_logf(S));          /* the token's cross-entropy */
        L.cnt[0]++;
        for (k = 0; k < TNN_NTOK; k++)
            if (allow[k])
                row_step(k, z[k] / S - (k == y ? 1.0f : 0.0f), h);
    }
    if (what & 2) {                             /* the pickup: a sigmoid's cross-entropy, weighted as in training */
        float zs = dotf(tw + TNN_NTOK * BASS_H, h, BASS_H) + tb[TNN_NTOK], p = tnn_sigmoid(zs);
        float a = zs < 0.0f ? -zs : zs;
        L.sum[1] += (zs > 0.0f ? zs : 0.0f) + fm_logf(1.0f + fm_expf(-a)) - (float)bt->skip * zs;
        L.cnt[1]++;
        row_step(TNN_NTOK, 0.3f * (p - (float)bt->skip), h);
    }
}

int bass_learn_step(void)
{
    if (L.pass >= L.passes || !L.n)
        return TNN_LEARN_DONE;
    learn_sgd(&L.ring[(L.first + L.t) % L.cap]);
    if (++L.t >= L.n) {                          /* a pass done: its losses */
        int w;
        for (w = 0; w < 2; w++) {
            L.loss[L.pass][w] = L.cnt[w] ? L.sum[w] / (float)L.cnt[w] : 0.0f;
            L.sum[w] = 0.0f;
            L.cnt[w] = 0;
        }
        L.t = 0;
        L.pass++;
    }
    return TNN_LEARN_SGD;
}

static void kept_head(int rows)
{
    kept.magic = LRN_MAGIC;
    kept.net = net_id;
    kept.rows = (uint16_t)rows;
    kept.cols = BASS_H;
    kept.rsv = 0;
    if (!rows)
        kept.learnings = 0;
}

void bass_learn_commit(void)
{
    int r, j;
    for (r = 0; r < TNN_LROWS; r++) {           /* each row's change, int8: its largest 127 steps */
        const int8_t *w0 = w0_row(r);
        const float s = w0_scale(r);
        float mx = 0.0f, st;
        for (j = 0; j < BASS_H; j++) {
            float d = tw[r * BASS_H + j] - s * (float)w0[j];
            d = d < 0.0f ? -d : d;
            mx = d > mx ? d : mx;
        }
        st = mx * (1.0f / 127.0f);
        kept.step[r] = st;
        for (j = 0; j < BASS_H; j++) {
            float x = st > 0.0f ? (tw[r * BASS_H + j] - s * (float)w0[j]) / st : 0.0f;
            int v = (int)(x + (x < 0.0f ? -0.5f : 0.5f));
            kept.q[r * BASS_H + j] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
        kept.bias[r] = tw[TNN_LROWS * BASS_H + r];
    }
    kept.learnings = (uint16_t)(!learned ? 1u : kept.learnings < 0xFFFFu ? kept.learnings + 1u : 0xFFFFu);
    kept_head(TNN_LROWS);
    learned = 1;
}

const void *bass_learned_obj(uint32_t *len)
{
    *len = learned ? (uint32_t)sizeof kept : 16u;
    return &kept;
}

void *bass_learned_room(uint32_t *max)
{
    learned = 0;
    *max = (uint32_t)sizeof kept;
    return &kept;
}

static int finite(float x) { return x == x && x < 1e30f && x > -1e30f; }

int bass_learned_check(int len)
{
    int r, ok = len == (int)sizeof kept && kept.magic == LRN_MAGIC && kept.net == net_id && kept.rows == TNN_LROWS &&
                kept.cols == BASS_H && kept.learnings > 0;
    for (r = 0; ok && r < TNN_LROWS; r++)
        ok = finite(kept.step[r]) && kept.step[r] >= 0.0f && kept.step[r] < 1.0f && finite(kept.bias[r]);
    learned = (uint8_t)ok;
    if (!ok)
        kept_head(0);
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------ DRUM */
static int drum_sparse(const drum_in_t *in, int16_t *ix, float *v)
{
    int n = 0, p;
    for (p = 0; p < TNN_NPIECE; p++)
        if (in->phit >> p & 1u) {
            ONE(DRUM_IN_PHIT, p);
            ix[n] = (int16_t)(DRUM_IN_PVEL + p);
            v[n++] = (float)in->pvel[p] * (1.0f / 3.0f);
        }
    ONE(DRUM_IN_STEP, in->step % 12u);
    ONE(DRUM_IN_PHR, in->phr & 7u);
    ONE(DRUM_IN_SEC, in->sec < 7 ? in->sec : SEC_HEAD);
    ONE(DRUM_IN_TEMPO, in->tempo & 3u);
    if (in->energy != 0.0f) {
        ix[n] = (int16_t)DRUM_IN_ENERGY;
        v[n++] = in->energy;
    }
    return n;
}

static void drum_run(float *h, const int16_t *ix, const float *v, int n, float *hit, float *vel)
{
    float p[DRUM_P];
    linear_sparse(dq + DRUM_PROJ_WEIGHT, DRUM_S[DRUM_PROJ_WEIGHT_S], df + DRUM_PROJ_BIAS, DRUM_NIN, DRUM_P, ix, v, n,
                  p);
    gru(dq + DRUM_GRU_WEIGHT_IH_L0, DRUM_S[DRUM_GRU_WEIGHT_IH_L0_S], dq + DRUM_GRU_WEIGHT_HH_L0,
        DRUM_S[DRUM_GRU_WEIGHT_HH_L0_S], df + DRUM_GRU_BIAS_IH_L0, df + DRUM_GRU_BIAS_HH_L0, p, DRUM_P, h, DRUM_H);
    linear(dq + DRUM_HEADS_0_WEIGHT, DRUM_S[DRUM_HEADS_0_WEIGHT_S], df + DRUM_HEADS_0_BIAS, h, DRUM_H, TNN_NPIECE, hit);
    linear(dq + DRUM_HEADS_1_WEIGHT, DRUM_S[DRUM_HEADS_1_WEIGHT_S], df + DRUM_HEADS_1_BIAS, h, DRUM_H, 3 * TNN_NPIECE,
           vel);
}

void drum_step(float *h, const drum_in_t *in, float *hit, float *vel)
{
    int16_t ix[NACT];
    float v[NACT];
    drum_run(h, ix, v, drum_sparse(in, ix, v), hit, vel);
}

void drum_step_x(float *h, const float *x, float *hit, float *vel)
{
    int16_t ix[NACT];
    float v[NACT];
    drum_run(h, ix, v, sparse_of(x, DRUM_NIN, ix, v), hit, vel);
}

void drum_x(const drum_in_t *in, float *x)
{
    int16_t ix[NACT];
    float v[NACT];
    int i, n = drum_sparse(in, ix, v);
    for (i = 0; i < DRUM_NIN; i++)
        x[i] = 0.0f;
    for (i = 0; i < n; i++)
        x[ix[i]] = v[i];
}

/* ------------------------------------------------------------------ HARM */
static int harm_sparse(const harm_in_t *in, int16_t *ix, float *v)
{
    int n = 0;
    ONE(HARM_IN_IVL, in->ivl % 12u);
    ONE(HARM_IN_QUAL, in->qual >= 0 && in->qual < TNN_NQUAL ? in->qual : Q_MAJ);
    ONE(HARM_IN_PDUR, in->pdur & 7u);
    ONE(HARM_IN_START, in->start & 3u);
    if (in->first)
        ONE(HARM_IN_FIRST, 0);
    return n;
}

static void harm_run(float *h, const int16_t *ix, const float *v, int n, float *root, float *qual, float *dur)
{
    float p[HARM_P];
    linear_sparse(hq + HARM_PROJ_WEIGHT, HARM_S[HARM_PROJ_WEIGHT_S], hf + HARM_PROJ_BIAS, HARM_NIN, HARM_P, ix, v, n,
                  p);
    gru(hq + HARM_GRU_WEIGHT_IH_L0, HARM_S[HARM_GRU_WEIGHT_IH_L0_S], hq + HARM_GRU_WEIGHT_HH_L0,
        HARM_S[HARM_GRU_WEIGHT_HH_L0_S], hf + HARM_GRU_BIAS_IH_L0, hf + HARM_GRU_BIAS_HH_L0, p, HARM_P, h, HARM_H);
    linear(hq + HARM_HEADS_0_WEIGHT, HARM_S[HARM_HEADS_0_WEIGHT_S], hf + HARM_HEADS_0_BIAS, h, HARM_H, 12, root);
    linear(hq + HARM_HEADS_1_WEIGHT, HARM_S[HARM_HEADS_1_WEIGHT_S], hf + HARM_HEADS_1_BIAS, h, HARM_H, TNN_NQUAL, qual);
    linear(hq + HARM_HEADS_2_WEIGHT, HARM_S[HARM_HEADS_2_WEIGHT_S], hf + HARM_HEADS_2_BIAS, h, HARM_H, TNN_NDUR, dur);
}

void harm_step(float *h, const harm_in_t *in, float *root, float *qual, float *dur)
{
    int16_t ix[NACT];
    float v[NACT];
    harm_run(h, ix, v, harm_sparse(in, ix, v), root, qual, dur);
}

void harm_step_x(float *h, const float *x, float *root, float *qual, float *dur)
{
    int16_t ix[NACT];
    float v[NACT];
    harm_run(h, ix, v, sparse_of(x, HARM_NIN, ix, v), root, qual, dur);
}

/* ------------------------------------------------------------------ CHORD */
static void chord_run(const int16_t *ix, const float *v, int n, float *out)
{
    float a[CHORD_HID], b[CHORD_HID];
    int k;
    linear_sparse(cq + CHORD_L1_WEIGHT, CHORD_S[CHORD_L1_WEIGHT_S], cf + CHORD_L1_BIAS, CHORD_NIN, CHORD_HID, ix, v, n,
                  a);
    for (k = 0; k < CHORD_HID; k++)
        a[k] = a[k] > 0.0f ? a[k] : 0.0f;
    linear(cq + CHORD_L2_WEIGHT, CHORD_S[CHORD_L2_WEIGHT_S], cf + CHORD_L2_BIAS, a, CHORD_HID, CHORD_HID, b);
    for (k = 0; k < CHORD_HID; k++)
        b[k] = b[k] > 0.0f ? b[k] : 0.0f;
    linear(cq + CHORD_L3_WEIGHT, CHORD_S[CHORD_L3_WEIGHT_S], cf + CHORD_L3_BIAS, b, CHORD_HID, CHORD_NOUT, out);
}

void chord_eval(const chord_in_t *in, float *out)
{
    int16_t ix[NACT];
    float v[NACT], mx = 0.0f;
    int n = 0, k;
    for (k = 0; k < 12; k++)
        if (in->pc[k] > mx)
            mx = in->pc[k];
    for (k = 0; k < 12 && mx > 0.0f; k++)
        if (in->pc[k] > 0.0f) {
            ix[n] = (int16_t)(CHORD_IN_PC + k);
            v[n++] = in->pc[k] / mx;
        }
    ONE(CHORD_IN_LOW, (uint8_t)in->low % 12u);
    if (in->proot >= 0) {
        ONE(CHORD_IN_PROOT, in->proot);
        ONE(CHORD_IN_PQUAL, in->pqual >= 0 && in->pqual < TNN_NQUAL ? in->pqual : 0);
    } else
        ONE(CHORD_IN_NONE, 0);
    chord_run(ix, v, n, out);
}

void chord_eval_x(const float *x, float *out)
{
    int16_t ix[NACT];
    float v[NACT];
    chord_run(ix, v, sparse_of(x, CHORD_NIN, ix, v), out);
}

/* ------------------------------------------------------------------ sampling */
uint32_t tnn_rand(uint32_t *r)
{
    uint32_t x = *r ? *r : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *r = x;
    return x;
}

float tnn_unit(uint32_t *r) { return (float)(tnn_rand(r) >> 8) * (1.0f / 16777216.0f); }

int tnn_argmax(const float *v, const uint8_t *allow, int n)
{
    int i, best = -1;
    for (i = 0; i < n; i++)
        if ((!allow || allow[i]) && (best < 0 || v[i] > v[best]))
            best = i;
    return best < 0 ? 0 : best;
}

int tnn_sample(const float *logits, const uint8_t *allow, int n, float temp, uint32_t *r)
{
    float p[TNN_NTOK > 12 * TNN_NQUAL ? TNN_NTOK : 12 * TNN_NQUAL], mx = -1e30f, sum = 0.0f, target, run = 0.0f;
    int i, pick;
    if (temp <= 0.01f || n > (int)(sizeof p / sizeof p[0]))
        return tnn_argmax(logits, allow, n);
    for (i = 0; i < n; i++)
        if ((!allow || allow[i]) && logits[i] > mx)
            mx = logits[i];
    for (i = 0; i < n; i++) {
        p[i] = (!allow || allow[i]) ? fm_exp2f((logits[i] - mx) * (FM_LOG2E / temp)) : 0.0f;
        sum += p[i];
    }
    target = sum * tnn_unit(r);
    pick = tnn_argmax(logits, allow, n);               /* if the sums round short */
    for (i = 0; i < n; i++) {
        run += p[i];
        if (p[i] > 0.0f && run >= target) {
            pick = i;
            break;
        }
    }
    return pick;
}
