#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""TRIO's training data: walking bass lines, jazz drumming, chord progressions and piano voicings.

Sources (see docs/DATA.md for where to get them and their licences):
  FiloBass  (Riley & Dixon, ISMIR 2023; CC BY 4.0): 48 walking lines by professional bassists, note by note
  JazzSAMBA (Long et al. 2026; MIT-style): 76 jazz standards: drums and piano played on electronic instruments
            (exact MIDI), chord charts aligned to the playing
  walking_model.py (this directory): synthetic walking lines over the JazzSAMBA and FiloBass progressions,
            for keys and progressions the recordings do not cover

  python3 -I tools/trio_data.py --filobass NOTE_DATA_CSV --jazzsamba DIR --out DIR [--synth N]

Writes to OUT: bass.npz (a beat a step), drums.npz (a triplet eighth a step), harmony.npz (a chord a step),
voicings.npz (one chord's piano notes a row). Everything is in absolute pitch classes; the trainer transposes.
"""
import argparse
import bisect
import csv
import glob
import math
import os
import random
import sys
from fractions import Fraction

import mido
import numpy as np

# ---------------------------------------------------------------- shared vocabulary (also in app/dsp/trio_nn.h)
LO, HI = 28, 60                       # the bass's tokens: E1 .. C4 (sounding)
REST, HOLD = 0, 1                     # then 2 + (pitch - LO)
NTOK = 2 + HI - LO + 1                # 35
MAJ, DOM, MIN, HDIM, DIM, SUS, AUG = range(7)
NQUAL = 7
QUAL_MASK = {MAJ: 0x891, DOM: 0x491, MIN: 0x489, HDIM: 0x449, DIM: 0x249, SUS: 0x4A1, AUG: 0x111}
#            maj7 {0,4,7,11}  7 {0,4,7,10}  m7 {0,3,7,10}  m7b5 {0,3,6,10}  dim7 {0,3,6,9}  7sus4 {0,5,7,10}  aug {0,4,8}
SIX_MAJ, SIX_MIN, MMAJ7, TRIAD_MAJ, TRIAD_MIN, B9 = 0x291, 0x289, 0x889, 0x091, 0x089, 0x002
# drum kit pieces (JazzSAMBA's e-kit notes -> TRIO's pieces; mapped from where each note falls in the bar)
RIDE, SNARE, KICK, HHF, HHO, XSTK, TOMH, TOML, CRASH = range(9)
NPIECE = 9
KIT = {45: RIDE, 38: SNARE, 36: KICK, 44: HHF, 46: HHO, 42: HHO, 26: HHO, 22: HHO, 40: XSTK, 48: TOMH, 50: TOMH,
       47: TOMH, 41: TOML, 43: TOML, 58: TOML, 55: CRASH, 59: CRASH, 51: CRASH, 49: CRASH, 57: CRASH, 52: CRASH}
SEC_INTRO, SEC_HEAD, SEC_HORN, SEC_PIANO, SEC_BASS, SEC_DRUMS, SEC_OUTRO = range(7)

LETTER = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def pc_of(name):
    """'Bb', 'B-', 'Cb', 'F#', 'Bbb' -> pitch class."""
    pc = LETTER[name[0]]
    for ch in name[1:]:
        pc += {'#': 1, 'b': -1, '-': -1}.get(ch, 0)
    return pc % 12


def tok_of(pitch):
    while pitch < LO:
        pitch += 12
    while pitch > HI:
        pitch -= 12
    return 2 + pitch - LO


def mask_rel(mask, root):
    """A mask of intervals over the root -> the mask of pitch classes."""
    return sum(1 << ((root + i) % 12) for i in range(12) if mask >> i & 1)


# ---------------------------------------------------------------- FiloBass: bass lines (a beat a step)
FB_QUAL = {'7': (DOM, QUAL_MASK[DOM]), '7b9': (DOM, QUAL_MASK[DOM] | B9), '7#11': (DOM, QUAL_MASK[DOM]),
           'm7': (MIN, QUAL_MASK[MIN]), 'minmaj7': (MIN, MMAJ7), 'maj7': (MAJ, QUAL_MASK[MAJ]),
           'maj7#11': (MAJ, QUAL_MASK[MAJ]), '6': (MAJ, SIX_MAJ), '': (MAJ, TRIAD_MAJ),
           'm7b5': (HDIM, QUAL_MASK[HDIM]), 'dim': (DIM, QUAL_MASK[DIM])}


def filobass(path):
    """One sequence per piece: tok, skip, root, qual, mask, bib (beat in bar), feel."""
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r['title'], []).append(r)
    seqs = []
    for title, rs in by.items():
        notes, chords = [], {}
        for r in rs:
            on, du = Fraction(r['note_global_offset']), Fraction(r['note_duration'])
            if r['type'] == 'Note':
                notes.append((on, du, int(r['note_midi']) - 12))          # written bass clef -> sounding
            if r['chord_root']:
                chords[r['chord_id']] = (Fraction(r['chord_global_offset']), Fraction(r['chord_duration']),
                                         pc_of(r['chord_root']), r['chord_kind'])
        notes.sort()
        chords = sorted(chords.values())
        n = int(math.ceil(max(on + du for on, du, _ in notes)))
        onsets = {on: p for on, du, p in notes}
        starts = [on for on, _, _ in notes]
        seq = {k: np.zeros(n, np.int16) for k in ('tok', 'skip', 'root', 'qual', 'mask', 'bib', 'feel')}
        cstarts = [c[0] for c in chords]
        for b in range(n):
            fb = Fraction(b)
            if fb in onsets:
                seq['tok'][b] = tok_of(onsets[fb])
            else:
                i = bisect.bisect_left(starts, fb) - 1
                seq['tok'][b] = HOLD if i >= 0 and starts[i] + notes[i][1] > fb else REST
            j = bisect.bisect_left(starts, fb + Fraction(1, 1000))
            seq['skip'][b] = int(j < len(starts) and starts[j] < fb + 1)
            k = bisect.bisect_right(cstarts, fb) - 1
            if k >= 0 and fb < chords[k][0] + chords[k][1] and chords[k][3] in FB_QUAL:
                q, m = FB_QUAL[chords[k][3]]
                seq['root'][b], seq['qual'][b], seq['mask'][b] = chords[k][2], q, mask_rel(m, chords[k][2])
            else:
                seq['root'][b], seq['qual'][b], seq['mask'][b] = -1, 0, 0
            seq['bib'][b] = b % 4
        seq['src'] = title
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------- JazzSAMBA
def js_quality(r):
    t, s = r['triad'], r['seventh']
    if t == 'maj' and s == 'dom':
        return DOM, QUAL_MASK[DOM] | (B9 if r['alt_b9'] == 'True' else 0)
    if t == 'aug':
        return AUG, QUAL_MASK[AUG]
    if t in ('maj', 'power'):
        if s == 'maj':
            return MAJ, QUAL_MASK[MAJ]
        return MAJ, SIX_MAJ if r['has_6'] == 'True' else TRIAD_MAJ
    if t == 'min':
        if s == 'min':
            return MIN, QUAL_MASK[MIN]
        if s == 'maj':
            return MIN, MMAJ7
        return MIN, SIX_MIN if r['has_6'] == 'True' else TRIAD_MIN
    if t == 'dim':
        return (DIM, QUAL_MASK[DIM]) if s in ('dim', 'none') else (HDIM, QUAL_MASK[HDIM])
    if t == 'sus':
        return SUS, QUAL_MASK[SUS]
    return None


def js_takes(root, protocols=('async',)):
    songs = {r['song_subdir_basename']: r for r in csv.DictReader(open(os.path.join(root, 'songs.csv')))}
    for proto in protocols:
        for d in sorted(glob.glob(os.path.join(root, proto, 'songs', '*', '*'))):
            name, take = d.split(os.sep)[-2], d.split(os.sep)[-1]
            if name in songs and os.path.exists(os.path.join(d, 'annotations', 'bars.csv')):
                yield songs[name], proto, take, d


def js_chords(d, num):
    out = []
    for r in csv.DictReader(open(os.path.join(d, 'annotations', 'chords.csv'))):
        if r['is_no_chord'] == 'True' or r['root'] in ('NA', ''):
            continue
        q = js_quality(r)
        if q is None:
            continue
        s = (int(r['start_bar']) + float(r['start_fraction'])) * num
        e = (int(r['end_bar']) + float(r['end_fraction'])) * num
        if e - s > 1e-6:
            out.append((s, e, pc_of(r['root']), q[0], mask_rel(q[1], pc_of(r['root'])),
                        float(r['start_time']), float(r['end_time'])))
    return sorted(out)


def js_sections(d):
    secs = []
    for r in csv.DictReader(open(os.path.join(d, 'annotations', 'sections.csv'))):
        top, solo = r['top_level_section'], r['soloist']
        if top == 'intro':
            c = SEC_INTRO
        elif top == 'outro':
            c = SEC_OUTRO
        elif top.startswith('head'):
            c = SEC_HEAD
        elif solo == 'piano':
            c = SEC_PIANO
        elif solo == 'bass':
            c = SEC_BASS
        elif solo == 'drums':
            c = SEC_DRUMS
        else:
            c = SEC_HORN
        secs.append((int(r['start_bar']), int(r['end_bar']), c))
    return secs


def midi_notes(path):
    """[(on_s, off_s, note, vel)] from a MIDI file, in seconds."""
    t, held, out = 0.0, {}, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            held.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type in ('note_off', 'note_on'):
            if held.get(msg.note):
                on, v = held[msg.note].pop(0)
                out.append((on, t, msg.note, v))
    return sorted(out)


def js_drums(root):
    """One sequence per 4/4 swing take: hits (bits), vel (levels 0..3 per piece), step, bar, phrase, sec, bpm."""
    seqs = []
    for song, proto, take, d in js_takes(root):
        if song['time_signature'] != '4/4' or song['is_swung'] != 'True' or not os.path.exists(d + '/drums.mid'):
            continue
        bars = list(csv.DictReader(open(d + '/annotations/bars.csv')))
        bs = [float(b['start_time']) for b in bars]
        be = [float(b['end_time']) for b in bars]
        n = len(bars) * 12
        hits = np.zeros(n, np.int32)
        vel = np.zeros((n, NPIECE), np.int8)
        for on, off, note, v in midi_notes(d + '/drums.mid'):
            if note not in KIT:
                continue
            i = bisect.bisect_right(bs, on) - 1
            if i < 0 or on >= be[i] + 0.2:
                continue
            k = i * 12 + int(round((on - bs[i]) / (be[i] - bs[i]) * 12))
            if k >= n:
                continue
            p = KIT[note]
            hits[k] |= 1 << p
            vel[k, p] = max(vel[k, p], 1 if v < 45 else 2 if v < 90 else 3)
        sec = np.full(n, SEC_HEAD, np.int8)
        phrase = np.zeros(n, np.int8)
        for s, e, c in js_sections(d):
            for b in range(max(s, 0), min(e, len(bars))):
                sec[b * 12:(b + 1) * 12] = c
                phrase[b * 12:(b + 1) * 12] = (b - s) % 8
        seqs.append({'hits': hits, 'vel': vel, 'step': np.arange(n) % 12, 'sec': sec, 'phrase': phrase,
                     'bpm': float(song['bpm']), 'genre': song['genre'], 'src': f"{song['song_subdir_basename']}/{take}"})
    return seqs


def js_harmony(root):
    """Chord sequences (one per take, both protocols, the preferred takes): root, qual, mask, dur (beats)."""
    seqs = []
    for song, proto, take, d in js_takes(root, ('async', 'sync')):
        if take != 'preferred':
            continue
        num = int(song['time_signature_numerator'])
        ch = js_chords(d, num)
        merged = []
        for s, e, r, q, m, _, _ in ch:                       # the same chord restated: one longer chord
            if merged and merged[-1][2] == r and merged[-1][3] == q and abs(merged[-1][1] - s) < 1e-6:
                merged[-1] = (merged[-1][0], e, r, q, m)
            else:
                merged.append((s, e, r, q, m))
        if len(merged) > 8:
            seqs.append({'root': np.array([c[2] for c in merged], np.int8),
                         'qual': np.array([c[3] for c in merged], np.int8),
                         'mask': np.array([c[4] for c in merged], np.int16),
                         'dur': np.array([c[1] - c[0] for c in merged], np.float32),
                         'num': num, 'src': song['song_subdir_basename']})
    return seqs


def js_voicings(root):
    """Each charted chord with the piano's notes under it (comping only: no piano solos), async takes."""
    rows = []
    for song, proto, take, d in js_takes(root):
        if not os.path.exists(d + '/piano.mid'):
            continue
        num = int(song['time_signature_numerator'])
        notes = midi_notes(d + '/piano.mid')
        ons = [n[0] for n in notes]
        bars = list(csv.DictReader(open(d + '/annotations/bars.csv')))
        secs = js_sections(d)
        prev = (-1, 0)
        for s, e, r, q, m, ts, te in js_chords(d, num):
            bar = int(s // num)
            sec = next((c for a, b, c in secs if a <= bar < b), SEC_HEAD)
            if sec == SEC_PIANO or te - ts < 0.25:
                prev = (r, q)
                continue
            pcw = np.zeros(12, np.float32)
            low, n_on = 999, 0
            i = bisect.bisect_left(ons, ts - 4.0)
            for on, off, note, v in notes[i:]:
                if on >= te:
                    break
                ov = min(off, te) - max(on, ts - 0.12)       # a chord played a little early counts (comping)
                if ov <= 0:
                    continue
                pcw[note % 12] += ov * (v / 64.0)
                if on >= ts - 0.12:
                    n_on += 1
                    low = min(low, note)
            if n_on >= 2 and pcw.sum() > 0:
                rows.append((pcw / pcw.sum(), low % 12, r, q, prev[0], prev[1], song['song_subdir_basename']))
            prev = (r, q)
    return rows


# ---------------------------------------------------------------- synthetic lines (walking_model.py)
def synth(charts, n_seq, seed):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import walking_model as wm
    style = wm.Style()
    wm.fit(style, [(wm.parse_progression(wm.AUTUMN_A), 'Gm'), (wm.parse_progression(wm.F_BLUES), 'F')])
    sym = {MAJ: 'maj7', DOM: '7', MIN: 'm7', HDIM: 'm7b5', DIM: 'dim7', SUS: '7', AUG: ''}
    names = 'C Db D Eb E F Gb G Ab A Bb B'.split()
    rng = random.Random(seed)
    out = []
    pool = [c for c in charts if c['num'] == 4 and all(abs(d * 2 - round(d * 2)) < 1e-3 for d in c['dur'])]
    while len(out) < n_seq:
        c = rng.choice(pool)
        shift = rng.randrange(12)
        # 16 to 32 bars from a random start: the chart's chords cut into 4/4 bars, two beats the shortest
        beats, i = [], rng.randrange(len(c['root']))
        while len(beats) < 4 * rng.choice((16, 24, 32)):
            r, q, d = (int(c['root'][i]) + shift) % 12, int(c['qual'][i]), int(round(c['dur'][i]))
            beats += [(r, q)] * max(d, 1)
            i = (i + 1) % len(c['root'])
        beats = beats[:len(beats) // 4 * 4]
        bars = []
        for b in range(0, len(beats), 4):
            chs, offs = [], []
            for k in range(4):
                if k == 0 or beats[b + k] != beats[b + k - 1]:
                    offs.append(k)
                    chs.append(beats[b + k])
            bars.append(wm.Bar(4, [(o * wm.TPB, wm.parse_chord(names[r] + sym[q])) for o, (r, q) in zip(offs, chs)]))
        weights = {}
        for bar in bars:
            for _, ch in bar.chords:
                weights[ch] = weights.get(ch, 0) + 1
        key = wm.guess_key(weights)
        feel = 'two' if rng.random() < 0.2 else 'four'
        st = wm.replace(style, feel=feel, skip_prob=0.3, theta=dict(style.theta))
        events = wm.build_events(bars, key, feel, random.Random(rng.random()), st.skip_prob, st.phrase_bars)
        try:
            chain = wm.model_chain(events, st)
            pitches = chain.sample(random.Random(rng.random()), rng.uniform(0.6, 1.3))
        except ValueError:
            continue
        wm.realize(events, pitches)
        n = len(beats)
        seq = {k: np.zeros(n, np.int16) for k in ('tok', 'skip', 'root', 'qual', 'mask', 'bib', 'feel')}
        seq['tok'][:] = HOLD
        for e in events:
            if e.rest or e.pitch is None:
                continue
            b, frac = divmod(e.start, wm.TPB)
            if frac == 0 and not e.ghost:
                seq['tok'][b] = tok_of(e.pitch)
            else:
                seq['skip'][b] = 1
        for b, (r, q) in enumerate(beats):
            m = {MAJ: QUAL_MASK[MAJ], DOM: QUAL_MASK[DOM], MIN: QUAL_MASK[MIN], HDIM: QUAL_MASK[HDIM],
                 DIM: QUAL_MASK[DIM], SUS: QUAL_MASK[SUS], AUG: QUAL_MASK[AUG]}[q]
            seq['root'][b], seq['qual'][b], seq['mask'][b], seq['bib'][b] = r, q, mask_rel(m, r), b % 4
        seq['feel'][:] = 1 if feel == 'two' else 0
        seq['src'] = f"synth/{c['src']}"
        out.append(seq)
    return out


def save(path, seqs):
    """Sequences of equal-length arrays -> one npz: the arrays concatenated, with their starts."""
    keys = [k for k in seqs[0] if isinstance(seqs[0][k], np.ndarray)]
    lens = np.array([len(s[keys[0]]) for s in seqs])
    data = {k: np.concatenate([s[k] for s in seqs]) for k in keys}
    data['starts'] = np.concatenate([[0], np.cumsum(lens)[:-1]])
    data['lens'] = lens
    data['src'] = np.array([s['src'] for s in seqs])
    for k in seqs[0]:
        if k not in keys and k != 'src':
            data[k] = np.array([s[k] for s in seqs])
    np.savez_compressed(path, **data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--filobass', required=True, help='FiloBass notebooks_and_scripts/note_data.csv')
    ap.add_argument('--jazzsamba', required=True, help='JazzSAMBA root (songs.csv, async/, sync/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--synth', type=int, default=1500, help='synthetic lines (16-32 bars each)')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fb = filobass(a.filobass)
    print(f'FiloBass: {len(fb)} lines, {sum(len(s["tok"]) for s in fb)} beats')
    harm = js_harmony(a.jazzsamba)
    print(f'JazzSAMBA harmony: {len(harm)} charts, {sum(len(s["root"]) for s in harm)} chords')
    syn = synth(harm, a.synth, a.seed)
    print(f'synthetic: {len(syn)} lines, {sum(len(s["tok"]) for s in syn)} beats')
    save(os.path.join(a.out, 'bass.npz'), fb + syn)
    dr = js_drums(a.jazzsamba)
    print(f'JazzSAMBA drums: {len(dr)} takes, {sum(len(s["hits"]) for s in dr) // 12} bars')
    save(os.path.join(a.out, 'drums.npz'), dr)
    save(os.path.join(a.out, 'harmony.npz'), harm)
    vo = js_voicings(a.jazzsamba)
    print(f'JazzSAMBA voicings: {len(vo)} chords under the piano')
    np.savez_compressed(os.path.join(a.out, 'voicings.npz'), pc=np.array([v[0] for v in vo]),
                        low=np.array([v[1] for v in vo], np.int8), root=np.array([v[2] for v in vo], np.int8),
                        qual=np.array([v[3] for v in vo], np.int8), proot=np.array([v[4] for v in vo], np.int8),
                        pqual=np.array([v[5] for v in vo], np.int8), src=np.array([v[6] for v in vo]))


if __name__ == '__main__':
    main()
