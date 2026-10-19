#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Train TRIO's four networks and write them for the firmware.

  CHORD  what chord the player's notes are (a voicing's pitch classes, its lowest note, the chord before)
  HARM   the next chord (its root from this one's, its kind) and how long this one lasts, a chord a step
  BASS   the walking line, a beat a step: the note (or rest, or hold) and whether a pickup goes before the next
  DRUM   the drummer, a triplet eighth a step: which of 9 pieces sound, and how hard

  python tools/trio_train.py --data ~/fm1/data/trio --out app [--quick]

Writes app/dsp/trio_w.h (int8 matrices with one scale each, float biases) and tests/host/trio_golden.h (inputs
and outputs of the quantized models, for the C kernels' test). The features here and in app/dsp/trio_nn.c must
stay the same: each group's offset is exported too.
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)
LO, HI, REST, HOLD = 28, 60, 0, 1
NTOK = 2 + HI - LO + 1
NQUAL, NPIECE = 7, 9
DUR_CLASSES = (1, 2, 3, 4, 6, 8, 12, 16)
SEC_DRUMS = 5


def dur_class(beats):
    return int(np.argmin([abs(beats - d) for d in DUR_CLASSES]))


def rot(mask, k):
    """A 12-bit pitch-class mask moved up k semitones."""
    k %= 12
    return ((mask << k) | (mask >> (12 - k))) & 0xFFF


def bits12(mask):
    return np.array([(int(mask) >> i) & 1 for i in range(12)], np.float32)


def fold(pitch):
    while pitch < LO:
        pitch += 12
    while pitch > HI:
        pitch -= 12
    return pitch


# ================================================================ feature layouts (mirrored in app/dsp/trio_nn.c)
class Layout:
    def __init__(self, groups):
        self.off, n = {}, 0
        for name, size in groups:
            self.off[name] = (n, size)
            n += size
        self.n = n


BASS_L = Layout([('tok', NTOK + 1), ('root', 12), ('mask', 12), ('qual', NQUAL), ('nroot', 12), ('nmask', 12),
                 ('tochg', 8), ('since', 8), ('bib', 4), ('feel', 1), ('pskip', 1), ('noch', 1)])
DRUM_L = Layout([('phit', NPIECE), ('pvel', NPIECE), ('step', 12), ('phr', 8), ('sec', 7), ('tempo', 4),
                 ('energy', 1)])
HARM_L = Layout([('ivl', 12), ('qual', NQUAL), ('pdur', len(DUR_CLASSES)), ('start', 4), ('first', 1)])
CHORD_L = Layout([('pc', 12), ('low', 12), ('proot', 12), ('pqual', NQUAL), ('none', 1)])


def put(x, L, name, idx=None, val=1.0, vec=None):
    o, n = L.off[name]
    if vec is not None:
        x[o:o + n] = vec
    elif idx is not None and 0 <= idx < n:
        x[o + idx] = val


# ---------------------------------------------------------------- BASS features
def bass_fields(tok, skip, root, qual, mask, bib, feel):
    """A line's raw inputs a beat (before any transposition): what the firmware knows at each beat."""
    n = len(tok)
    root = np.asarray(root, np.int64)
    mask = np.where(root >= 0, np.asarray(mask, np.int64) & 0xFFF, 0)
    f = {'ptok': np.concatenate([[NTOK], np.asarray(tok[:-1], np.int64)]), 'root': root, 'mask': mask,
         'qual': np.asarray(qual, np.int64), 'bib': np.asarray(bib, np.int64) % 4,
         'feel': np.asarray(feel, np.float32), 'pskip': np.concatenate([[0.0], np.asarray(skip[:-1], np.float32)]),
         'ytok': np.asarray(tok, np.int64), 'yskip': np.asarray(skip, np.float32)}
    nroot, nmask, tochg, since = np.full(n, -1), np.zeros(n, np.int64), np.full(n, 8), np.zeros(n, np.int64)
    key = root * 4096 + mask
    nxt_i = n
    for t in range(n - 1, -1, -1):           # the first later beat with another chord
        if t + 1 < n and key[t + 1] != key[t]:
            nxt_i = t + 1
        if nxt_i < n:
            nroot[t], nmask[t], tochg[t] = root[nxt_i], mask[nxt_i], min(nxt_i - t, 8)
        else:
            nroot[t], nmask[t], tochg[t] = -1, 0, 8
    for t in range(1, n):
        since[t] = 0 if key[t] != key[t - 1] else min(since[t - 1] + 1, 7)
    f.update(nroot=nroot, nmask=nmask, tochg=tochg, since=since)
    return f


def tok_map(shift):
    m = np.arange(NTOK + 1)
    for t in range(2, NTOK):
        m[t] = 2 + fold(t - 2 + LO + shift) - LO
    return m


TOK_MAPS = {k: tok_map(k) for k in range(-6, 7)}


def rot_v(mask, k):
    k %= 12
    return ((mask << k) | (mask >> (12 - k))) & 0xFFF


def bass_x(f, shift=0, drop=None):
    """The inputs (n x BASS_NIN) of a line's fields, moved up `shift` semitones; drop: beats whose next chord is hidden."""
    n = len(f['ptok'])
    X = np.zeros((n, BASS_L.n), np.float32)
    r = np.arange(n)
    o = lambda g: BASS_L.off[g][0]
    X[r, o('tok') + TOK_MAPS[shift][f['ptok']]] = 1
    has = f['root'] >= 0
    X[r[has], o('root') + (f['root'][has] + shift) % 12] = 1
    X[:, o('mask'):o('mask') + 12] = (rot_v(f['mask'], shift)[:, None] >> np.arange(12)) & 1
    X[r[has], o('qual') + f['qual'][has]] = 1
    X[~has, o('noch')] = 1
    nr, nm, tc = f['nroot'].copy(), f['nmask'].copy(), f['tochg'].copy()
    if drop is not None:
        nr[drop], nm[drop], tc[drop] = -1, 0, 8
    hn = nr >= 0
    X[r[hn], o('nroot') + (nr[hn] + shift) % 12] = 1
    X[:, o('nmask'):o('nmask') + 12] = (np.where(hn, rot_v(nm, shift), 0)[:, None] >> np.arange(12)) & 1
    X[r, o('tochg') + tc - 1] = 1
    X[r, o('since') + f['since']] = 1
    X[r, o('bib') + f['bib']] = 1
    X[:, o('feel')] = f['feel']
    X[:, o('pskip')] = f['pskip']
    return X, TOK_MAPS[shift][f['ytok']], f['yskip']


def bass_features(tok, skip, root, qual, mask, bib, feel, shift=0, drop_next=None):
    return bass_x(bass_fields(tok, skip, root, qual, mask, bib, feel), shift, drop_next)


# ---------------------------------------------------------------- DRUM features
def tempo_bucket(bpm):
    return 0 if bpm < 110 else 1 if bpm < 160 else 2 if bpm < 210 else 3


def drum_features(hits, vel, step, sec, phrase, bpm, energy):
    n = len(hits)
    X = np.zeros((n, DRUM_L.n), np.float32)
    o_ph, o_pv = DRUM_L.off['phit'][0], DRUM_L.off['pvel'][0]
    for t in range(n):
        x = X[t]
        if t > 0:
            for p in range(NPIECE):
                if hits[t - 1] >> p & 1:
                    x[o_ph + p] = 1.0
                    x[o_pv + p] = vel[t - 1, p] / 3.0
        put(x, DRUM_L, 'step', int(step[t]))
        put(x, DRUM_L, 'phr', int(phrase[t]) % 8)
        put(x, DRUM_L, 'sec', int(sec[t]))
        put(x, DRUM_L, 'tempo', tempo_bucket(bpm))
        x[DRUM_L.off['energy'][0]] = energy[t]
    Y = np.array([[(h >> p) & 1 for p in range(NPIECE)] for h in hits], np.float32)
    V = np.clip(vel.astype(np.int64) - 1, 0, 2)
    return X, Y, V


def section_energy(hits, sec):
    """Each step's energy: the hits a bar (every piece) over its section, /16."""
    n = len(hits)
    e = np.zeros(n, np.float32)
    counts = np.array([bin(int(h)).count('1') for h in hits], np.float32)
    t = 0
    while t < n:
        u = t
        while u < n and sec[u] == sec[t]:
            u += 1
        e[t:u] = counts[t:u].sum() / max((u - t) / 12, 1) / 16.0
        t = u
    return np.clip(e, 0, 1.5)


# ---------------------------------------------------------------- HARM features
def harm_features(root, qual, dur):
    n = len(root)
    X = np.zeros((n, HARM_L.n), np.float32)
    yroot = np.zeros(n, np.int64)
    yqual = np.zeros(n, np.int64)
    ydur = np.zeros(n, np.int64)
    pos = 0.0
    for i in range(n):
        x = X[i]
        put(x, HARM_L, 'ivl', (int(root[i]) - int(root[i - 1])) % 12 if i > 0 else 0)
        put(x, HARM_L, 'qual', int(qual[i]))
        put(x, HARM_L, 'pdur', dur_class(dur[i - 1]) if i > 0 else 3)
        put(x, HARM_L, 'start', int(round(pos)) % 4)
        x[HARM_L.off['first'][0]] = 1.0 if i == 0 else 0.0
        if i + 1 < n:
            yroot[i] = (int(root[i + 1]) - int(root[i])) % 12
            yqual[i] = int(qual[i + 1])
        else:
            yroot[i], yqual[i] = -100, -100
        ydur[i] = dur_class(dur[i])
        pos += dur[i]
    return X, yroot, yqual, ydur


# ---------------------------------------------------------------- CHORD features
def chord_features(pc, low, proot, pqual, shift, binar=False):
    x = np.zeros(CHORD_L.n, np.float32)
    p = np.roll(pc, shift)
    if binar:
        p = (p > 0.05 * p.max()).astype(np.float32)
    put(x, CHORD_L, 'pc', vec=p / max(p.max(), 1e-6))
    put(x, CHORD_L, 'low', (int(low) + shift) % 12)
    if proot >= 0:
        put(x, CHORD_L, 'proot', (int(proot) + shift) % 12)
        put(x, CHORD_L, 'pqual', int(pqual))
    else:
        x[CHORD_L.off['none'][0]] = 1.0
    return x


# ================================================================ models
class Seq(nn.Module):
    """input projection (linear) -> GRU -> heads"""

    def __init__(self, nin, proj, hid, heads):
        super().__init__()
        self.proj = nn.Linear(nin, proj)
        self.gru = nn.GRU(proj, hid, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hid, h) for h in heads])

    def forward(self, x, h=None):
        y, h = self.gru(self.proj(x), h)
        return [hd(y) for hd in self.heads], h


class MLP(nn.Module):
    def __init__(self, nin, hid, nout):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(nin, hid), nn.Linear(hid, hid), nn.Linear(hid, nout)

    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


def quantize_(model):
    """Weights to int8 with one scale per matrix (in place: the model then runs as the firmware will)."""
    scales = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2:
                s = p.abs().max().item() / 127.0 or 1.0
                p.copy_(torch.round(p / s).clamp(-127, 127) * s)
                scales[name] = s
    return scales


def train_loop(model, batches, loss_fn, steps, lr, eval_fn, label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        loss = loss_fn(model, *next(batches))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(steps // 5, 1) == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                ev = eval_fn(model)
            print(f'  {label} step {step}/{steps} train {loss.item():.3f} | ' +
                  ' '.join(f'{k} {v:.3f}' for k, v in ev.items()) + f' | {time.time() - t0:.0f}s', flush=True)
    return model


# ---------------------------------------------------------------- BASS
def load_seqs(path):
    d = np.load(path, allow_pickle=True)
    total, nseq = int(d['lens'].sum()), len(d['lens'])
    out = []
    for i, (st, n) in enumerate(zip(d['starts'], d['lens'])):
        q = {'src': str(d['src'][i])}
        for k in d.files:
            if k in ('starts', 'lens', 'src'):
                continue
            if d[k].shape[:1] == (total,):
                q[k] = d[k][st:st + n]
            elif d[k].shape[:1] == (nseq,):
                q[k] = d[k][i]
        out.append(q)
    return out


def split(seqs, held, rng, synth_held=30):
    real = sorted({q['src'].split('/')[0] for q in seqs if not q['src'].startswith('synth')})
    rng.shuffle(real)
    out_names = set(real[:held])
    syn = [i for i, q in enumerate(seqs) if q['src'].startswith('synth')]
    rng.shuffle(syn)
    out_syn = set(syn[:synth_held])
    va = [q for i, q in enumerate(seqs) if i in out_syn or
          (not q['src'].startswith('synth') and q['src'].split('/')[0] in out_names)]
    tr = [q for i, q in enumerate(seqs) if not (i in out_syn or
          (not q['src'].startswith('synth') and q['src'].split('/')[0] in out_names))]
    return tr, va


def bass_batches(seqs, rng, win=64, bs=96):
    fields = [bass_fields(q['tok'], q['skip'], q['root'], q['qual'], q['mask'], q['bib'], q['feel']) for q in seqs]
    w = np.array([len(q['tok']) * (3.0 if not q['src'].startswith('synth') else 1.0) for q in seqs])
    cw = np.cumsum(w).tolist()
    while True:
        Xs, Ts, Ss = [], [], []
        for i in rng.choices(range(len(seqs)), cum_weights=cw, k=bs):
            f = fields[i]
            n = len(f['ptok'])
            a = rng.randrange(max(n - win, 1))
            sub = {k: v[a:a + win] for k, v in f.items()}
            m = len(sub['ptok'])
            drop = np.array([rng.random() < 0.15 for _ in range(m)])
            X, yt, ys = bass_x(sub, rng.randrange(-6, 6), drop)
            if m < win:
                X = np.pad(X, ((0, win - m), (0, 0)))
                yt = np.pad(yt, (0, win - m), constant_values=-100)
                ys = np.pad(ys, (0, win - m), constant_values=-1)
            Xs.append(X), Ts.append(yt), Ss.append(ys)
        yield torch.tensor(np.stack(Xs)), torch.tensor(np.stack(Ts)), torch.tensor(np.stack(Ss))


def bass_loss(model, X, yt, ys):
    (lt, lsk), _ = model(X)
    ce = nn.functional.cross_entropy(lt.reshape(-1, NTOK), yt.reshape(-1), ignore_index=-100)
    m = ys.reshape(-1) >= 0
    bce = nn.functional.binary_cross_entropy_with_logits(lsk.reshape(-1)[m], ys.reshape(-1)[m])
    return ce + 0.3 * bce


def bass_eval_fn(val):
    data = [bass_features(s['tok'], s['skip'], s['root'], s['qual'], s['mask'], s['bib'], s['feel']) for s in val]

    def ev(model):
        nll = n = acc = 0.0
        for X, yt, ys in data:
            (lt, lsk), _ = model(torch.tensor(X)[None])
            nll += nn.functional.cross_entropy(lt[0], torch.tensor(yt), reduction='sum').item()
            acc += (lt[0].argmax(-1).numpy() == yt).sum()
            n += len(yt)
        return {'val_nll': nll / n, 'val_acc': acc / n}
    return ev


# ---------------------------------------------------------------- DRUM
def drum_batches(seqs, rng, win=96, bs=64):
    prepared = []
    for s in seqs:
        e = section_energy(s['hits'], s['sec'])
        X, Y, V = drum_features(s['hits'], s['vel'], s['step'], s['sec'], s['phrase'], float(s['bpm']), e)
        M = (s['sec'] != SEC_DRUMS).astype(np.float32)
        prepared.append((X, Y, V, M))
    while True:
        Xs, Ys, Vs, Ms = [], [], [], []
        for _ in range(bs):
            X, Y, V, M = prepared[rng.randrange(len(prepared))]
            a = rng.randrange(0, max(len(X) - win, 1) // 12 + 1) * 12
            Xs.append(X[a:a + win]), Ys.append(Y[a:a + win]), Vs.append(V[a:a + win]), Ms.append(M[a:a + win])
        L = min(len(x) for x in Xs)
        yield (torch.tensor(np.stack([x[:L] for x in Xs])), torch.tensor(np.stack([y[:L] for y in Ys])),
               torch.tensor(np.stack([v[:L] for v in Vs])), torch.tensor(np.stack([m[:L] for m in Ms])))


def drum_loss(model, X, Y, V, M):
    (lh, lv), _ = model(X)
    bce = nn.functional.binary_cross_entropy_with_logits(lh, Y, reduction='none').sum(-1)
    lv = lv.reshape(*lv.shape[:2], NPIECE, 3)
    ce = nn.functional.cross_entropy(lv.permute(0, 3, 1, 2), V, reduction='none')     # (B, T, P)
    ce = (ce * Y).sum(-1)
    return ((bce + ce) * M).sum() / M.sum()


def drum_eval_fn(val):
    data = []
    for s in val:
        e = section_energy(s['hits'], s['sec'])
        X, Y, V = drum_features(s['hits'], s['vel'], s['step'], s['sec'], s['phrase'], float(s['bpm']), e)
        data.append((X, Y, V, (s['sec'] != SEC_DRUMS).astype(np.float32)))

    def ev(model):
        tot = n = 0.0
        for X, Y, V, M in data:
            loss = drum_loss(model, torch.tensor(X)[None], torch.tensor(Y)[None], torch.tensor(V)[None],
                             torch.tensor(M)[None])
            tot += loss.item() * M.sum()
            n += M.sum()
        return {'val_loss': tot / n}
    return ev


# ---------------------------------------------------------------- HARM
def harm_seqs(harmony, bass):
    seqs = [(h['root'], h['qual'], h['dur'], h['src']) for h in harmony]
    for s in bass:                                       # FiloBass's charts too (not the synthetic lines)
        if s['src'].startswith('synth'):
            continue
        r, q = s['root'], s['qual']
        ev, t = [], 0
        while t < len(r):
            u = t
            while u < len(r) and r[u] == r[t] and q[u] == q[t]:
                u += 1
            if r[t] >= 0:
                ev.append((r[t], q[t], u - t))
            t = u
        if len(ev) > 8:
            seqs.append((np.array([e[0] for e in ev]), np.array([e[1] for e in ev]),
                         np.array([e[2] for e in ev], np.float32), 'fb/' + s['src']))
    return seqs


def harm_batches(seqs, rng, win=48, bs=64):
    prepared = [harm_features(r, q, d) for r, q, d, _ in seqs]
    while True:
        Xs, Rs, Qs, Ds = [], [], [], []
        for _ in range(bs):
            X, yr, yq, yd = prepared[rng.randrange(len(prepared))]
            a = rng.randrange(max(len(X) - win, 1))
            sl = slice(a, a + win)
            Xs.append(X[sl]), Rs.append(yr[sl]), Qs.append(yq[sl]), Ds.append(yd[sl])
        L = max(len(x) for x in Xs)
        pad = lambda arr, v: np.pad(arr, [(0, L - len(arr))] + [(0, 0)] * (arr.ndim - 1), constant_values=v)
        yield (torch.tensor(np.stack([pad(x, 0) for x in Xs])), torch.tensor(np.stack([pad(r, -100) for r in Rs])),
               torch.tensor(np.stack([pad(q, -100) for q in Qs])), torch.tensor(np.stack([pad(d, -100) for d in Ds])))


def harm_loss(model, X, yr, yq, yd):
    (lr, lq, ld), _ = model(X)
    f = nn.functional.cross_entropy
    return (f(lr.reshape(-1, 12), yr.reshape(-1), ignore_index=-100) +
            f(lq.reshape(-1, NQUAL), yq.reshape(-1), ignore_index=-100) +
            f(ld.reshape(-1, len(DUR_CLASSES)), yd.reshape(-1), ignore_index=-100))


def harm_eval_fn(val):
    data = [harm_features(r, q, d) for r, q, d, _ in val]

    def ev(model):
        accr = accq = accd = n = nd = 0.0
        for X, yr, yq, yd in data:
            (lr, lq, ld), _ = model(torch.tensor(X)[None])
            m = yr >= 0
            accr += (lr[0].argmax(-1).numpy()[m] == yr[m]).sum()
            accq += (lq[0].argmax(-1).numpy()[m] == yq[m]).sum()
            accd += (ld[0].argmax(-1).numpy() == yd).sum()
            n += m.sum()
            nd += len(yd)
        return {'next_root_acc': accr / n, 'next_qual_acc': accq / n, 'dur_acc': accd / nd}
    return ev


# ---------------------------------------------------------------- CHORD
TEMPLATES = {0: (0x891, 0x291, 0x091), 1: (0x491,), 2: (0x489, 0x289, 0x889, 0x089), 3: (0x449,), 4: (0x249,),
             5: (0x4A1,), 6: (0x111,)}


def template_guess(pc, low):
    """The baseline: the root and kind whose chord tones best match the notes (lowest note as the root breaks ties)."""
    best, bs = (0, 0), -1e9
    for r in range(12):
        for q, masks in TEMPLATES.items():
            for m in masks:
                tones = bits12(rot(m, r))
                s = (pc * (2 * tones - 1)).sum() + 0.3 * (low == r)
                if s > bs:
                    best, bs = (r, q), s
    return best


def chord_batches(v, idx, rng, bs=256):
    while True:
        Xs, Ys = [], []
        for _ in range(bs):
            i = idx[rng.randrange(len(idx))]
            sh = rng.randrange(12)
            prev = (v['proot'][i], v['pqual'][i]) if rng.random() < 0.8 else (-1, 0)
            Xs.append(chord_features(v['pc'][i], v['low'][i], prev[0], prev[1], sh, binar=rng.random() < 0.5))
            Ys.append(((int(v['root'][i]) + sh) % 12) * NQUAL + int(v['qual'][i]))
        yield torch.tensor(np.stack(Xs)), torch.tensor(np.array(Ys))


def chord_loss(model, X, Y):
    return nn.functional.cross_entropy(model(X), Y)


def chord_eval_fn(v, idx):
    X = np.stack([chord_features(v['pc'][i], v['low'][i], v['proot'][i], v['pqual'][i], 0, binar=True) for i in idx])
    Y = np.array([int(v['root'][i]) * NQUAL + int(v['qual'][i]) for i in idx])
    base = np.mean([template_guess((v['pc'][i] > 0.05 * v['pc'][i].max()).astype(np.float32), v['low'][i])
                    == (v['root'][i], v['qual'][i]) for i in idx])

    def ev(model):
        p = model(torch.tensor(X)).argmax(-1).numpy()
        return {'acc': (p == Y).mean(), 'root_acc': (p // NQUAL == Y // NQUAL).mean(), 'template_acc': base}
    return ev


# ================================================================ export
def c_array(name, data, ctype, per_line=24):
    vals = [str(int(v)) if ctype != 'float' else repr(float(np.float32(v))) + 'f' for v in data]
    lines = [', '.join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f'static const {ctype} {name}[{len(vals)}] = {{\n    ' + ',\n    '.join(lines) + '\n};\n'


def export(models, layouts, notes, path_w, path_g, goldens):
    out = ['/* SPDX-License-Identifier: MPL-2.0 */',
           "/* TRIO's networks (app/dsp/trio_nn.c): generated by tools/trio_train.py, do not edit.",
           ' * Trained on FiloBass (Riley & Dixon, ISMIR 2023, CC BY 4.0), JazzSAMBA (Long et al. 2026, MIT-style',
           ' * licence, "Copyright (c) 2026 Phillip Long and contributors") and synthetic lines from walking_model.py.']
    out += [' * ' + n for n in notes] + [' */', '#pragma once', '#include <stdint.h>', '']
    for mname, L in layouts.items():
        for g, (o, n) in L.off.items():
            out.append(f'#define {mname}_IN_{g.upper()} {o}')
        out.append(f'#define {mname}_NIN {L.n}')
        out.append('')
    for mname, (model, scales) in models.items():
        q8, fl, sc = [], [], []
        for pname, p in model.named_parameters():
            cname = f'{mname}_' + pname.replace('.', '_').upper()
            w = p.detach().cpu().numpy()
            if w.ndim == 2:
                out.append(f'#define {cname} {len(q8)}          /* {w.shape[0]} x {w.shape[1]} */')
                out.append(f'#define {cname}_S {len(sc)}')
                q8 += np.round(w / scales[pname]).astype(np.int64).reshape(-1).tolist()
                sc.append(scales[pname])
            else:
                out.append(f'#define {cname} {len(fl)}          /* {w.shape[0]} */')
                fl += w.reshape(-1).tolist()
        out.append(c_array(f'{mname}_Q', q8, 'int8_t'))
        out.append(c_array(f'{mname}_F', fl, 'float'))
        out.append(c_array(f'{mname}_S', sc, 'float'))
    open(path_w, 'w').write('\n'.join(out) + '\n')
    g = ['/* SPDX-License-Identifier: MPL-2.0 */',
         '/* Golden vectors for tests/host/nn_test.c: generated by tools/trio_train.py. */', '#pragma once', '']
    for name, arr in goldens.items():
        arr = np.asarray(arr)
        ctype = 'int32_t' if arr.dtype.kind in 'iu' else 'float'
        g.append(c_array(name, arr.reshape(-1), ctype, 12))
    open(path_g, 'w').write('\n'.join(g) + '\n')


def golden_seq(model, X):
    with torch.no_grad():
        outs, _ = model(torch.tensor(X)[None])
    return np.concatenate([o[0].numpy() for o in outs], axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True, help='the app directory (app/)')
    ap.add_argument('--tests', default='tests/host')
    ap.add_argument('--quick', action='store_true', help='few steps: a check of the pipeline')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    k = 0.05 if a.quick else 1.0
    notes = []

    # CHORD
    v = dict(np.load(os.path.join(a.data, 'voicings.npz'), allow_pickle=True))
    songs = sorted(set(v['src'].tolist()))
    rng.shuffle(songs)
    held = set(songs[:5])
    tr = [i for i in range(len(v['src'])) if v['src'][i] not in held]
    va = [i for i in range(len(v['src'])) if v['src'][i] in held]
    chord = MLP(CHORD_L.n, 64, 12 * NQUAL)
    print(f'CHORD: {len(tr)} voicings to learn, {len(va)} held out ({len(held)} songs)')
    train_loop(chord, chord_batches(v, tr, rng), chord_loss, int(4000 * k), 3e-3, chord_eval_fn(v, va), 'chord')
    cs = quantize_(chord)
    ev = chord_eval_fn(v, va)(chord)
    notes.append(f'CHORD: MLP {CHORD_L.n}-64-64-{12 * NQUAL}; held-out songs: {ev["acc"]:.3f} exact '
                 f'(root {ev["root_acc"]:.3f}), chord templates {ev["template_acc"]:.3f}')
    print('  int8:', ev)

    # HARM
    bass = load_seqs(os.path.join(a.data, 'bass.npz'))
    harmony = load_seqs(os.path.join(a.data, 'harmony.npz'))
    hs = harm_seqs(harmony, bass)
    rng.shuffle(hs)
    hv, ht = hs[:8], hs[8:]
    harm = Seq(HARM_L.n, 32, 48, (12, NQUAL, len(DUR_CLASSES)))
    print(f'HARM: {len(ht)} charts to learn, {len(hv)} held out')
    train_loop(harm, harm_batches(ht, rng), harm_loss, int(3000 * k), 3e-3, harm_eval_fn(hv), 'harm')
    hsc = quantize_(harm)
    ev = harm_eval_fn(hv)(harm)
    notes.append(f'HARM: {HARM_L.n} -> 32 -> GRU 48 -> 12+7+8; held-out charts: next root {ev["next_root_acc"]:.3f}, '
                 f'next kind {ev["next_qual_acc"]:.3f}, length {ev["dur_acc"]:.3f}')
    print('  int8:', ev)

    # BASS
    btr, bva = split(bass, 5, rng)
    bva = [s for s in bva if not s['src'].startswith('synth')] + [s for s in bva if s['src'].startswith('synth')][:30]
    bassm = Seq(BASS_L.n, 64, 96, (NTOK, 1))
    print(f'BASS: {len(btr)} lines to learn ({sum(not s["src"].startswith("synth") for s in btr)} FiloBass), '
          f'{len(bva)} held out')
    train_loop(bassm, bass_batches(btr, rng), bass_loss, int(6000 * k), 3e-3, bass_eval_fn(bva), 'bass')
    bsc = quantize_(bassm)
    ev = bass_eval_fn(bva)(bassm)
    notes.append(f'BASS: {BASS_L.n} -> 64 -> GRU 96 -> {NTOK}+1; held out: {ev["val_nll"]:.3f} nats a beat, '
                 f'{ev["val_acc"]:.3f} of beats the most likely note')
    print('  int8:', ev)

    # DRUM
    drums = load_seqs(os.path.join(a.data, 'drums.npz'))
    dtr, dva = split(drums, 4, rng)
    drum = Seq(DRUM_L.n, 48, 64, (NPIECE, NPIECE * 3))
    print(f'DRUM: {len(dtr)} takes to learn, {len(dva)} held out')
    train_loop(drum, drum_batches(dtr, rng), drum_loss, int(5000 * k), 3e-3, drum_eval_fn(dva), 'drum')
    dsc = quantize_(drum)
    ev = drum_eval_fn(dva)(drum)
    notes.append(f'DRUM: {DRUM_L.n} -> 48 -> GRU 64 -> 9+27; held out: {ev["val_loss"]:.3f} nats a step')
    print('  int8:', ev)

    # goldens: the quantized models on a few real inputs
    s = bva[0]
    Xb = bass_features(s['tok'][:16], s['skip'][:16], s['root'][:16], s['qual'][:16], s['mask'][:16],
                       s['bib'][:16], s['feel'][:16])[0]
    d0 = dva[0]
    Xd = drum_features(d0['hits'][:24], d0['vel'][:24], d0['step'][:24], d0['sec'][:24], d0['phrase'][:24],
                       float(d0['bpm']), section_energy(d0['hits'], d0['sec'])[:24])[0]
    Xh = harm_features(*hv[0][:3])[0][:12]
    Xc = np.stack([chord_features(v['pc'][i], v['low'][i], v['proot'][i], v['pqual'][i], 0, True) for i in va[:8]])
    with torch.no_grad():
        Yc = chord(torch.tensor(Xc)).numpy()
    goldens = {'G_BASS_X': Xb, 'G_BASS_Y': golden_seq(bassm, Xb), 'G_DRUM_X': Xd, 'G_DRUM_Y': golden_seq(drum, Xd),
               'G_HARM_X': Xh, 'G_HARM_Y': golden_seq(harm, Xh), 'G_CHORD_X': Xc, 'G_CHORD_Y': Yc}
    os.makedirs(os.path.join(a.out, 'dsp'), exist_ok=True)
    os.makedirs(a.tests, exist_ok=True)
    export({'CHORD': (chord, cs), 'HARM': (harm, hsc), 'BASS': (bassm, bsc), 'DRUM': (drum, dsc)},
           {'CHORD': CHORD_L, 'HARM': HARM_L, 'BASS': BASS_L, 'DRUM': DRUM_L}, notes,
           os.path.join(a.out, 'dsp', 'trio_w.h'), os.path.join(a.tests, 'trio_golden.h'), goldens)
    torch.save({'chord': chord.state_dict(), 'harm': harm.state_dict(), 'bass': bassm.state_dict(),
                'drum': drum.state_dict()}, os.path.join(a.data, 'trio_models.pt'))
    print('\n'.join(notes))
    print('wrote', os.path.join(a.out, 'dsp', 'trio_w.h'))


if __name__ == '__main__':
    main()
