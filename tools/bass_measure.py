#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""TRIO's upright bass, measured: the partials of real pizzicato notes in FiloBass's audio (walking lines in 48 tunes),
and app/dsp/bass_tab.h from them; the same analysis on TRIO's own notes, to compare.

For each note (onset, pitch from FiloBass's aligned MIDI): each harmonic's amplitude by weighted least squares at k*f0
(f0 known; each track's tuning found first), in Hann-weighted frames every 10 ms from the onset; the RMS envelope at
2 ms; the pitch over the first 120 ms. Notes followed by another sooner than 0.25 s are left out (no decay to see).

  bass_measure.py stems FILOBASS_DIR REF.npz [MAX_TRACKS]   FiloBass ISMIR Publication/ (audio_bass_stems,
                                                              midi_fully_aligned); about 7 minutes for all 48
  bass_measure.py fit REF.npz app/dsp/bass_tab.h            the tables: per register, each partial's level and decay
  bass_measure.py wav NOTES.wav NOTES.txt OUT.npz           TRIO's notes (tests/host/bass_notes.c writes both files)
  bass_measure.py compare REF.npz OTHER.npz [...]           tables side by side: spectra, decays, attack, pitch

Needs NumPy, mido and ffmpeg. Run Python with -I on downloaded data.
"""
import json
import subprocess
import sys

import numpy as np

SR = 44100
K = 40                     # harmonics analysed (those above 3 kHz left out)
KFIT = 20                  # partials in the synth's tables (app/dsp/voice.c BK)
FMAX = 3000.0
HOP = 0.010
NFR = 31                   # frames: 0 .. 300 ms after the onset
PB = [(28, 33), (34, 39), (40, 45), (46, 51), (52, 57)]   # registers (MIDI)
KS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]


def decode(path):
    raw = subprocess.run(['ffmpeg', '-v', 'quiet', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def midi_notes(path):
    import mido
    t = 0.0; on = {}; notes = []
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            notes.append((on.pop(msg.note), msg.note))
    return sorted(notes)


def tuning(x, notes):
    """the track's tuning in cents (harmonic-sum search over the longest notes)"""
    cents = []
    for i, (t0, p) in enumerate(notes):
        nxt = notes[i + 1][0] if i + 1 < len(notes) else t0 + 2
        if nxt - t0 < 0.6 or p < 28 or p > 57:
            continue
        a = int((t0 + 0.1) * SR)
        seg = x[a:a + 16384]
        if len(seg) < 16384:
            continue
        S = np.abs(np.fft.rfft(seg * np.hanning(16384), 65536))
        f = np.fft.rfftfreq(65536, 1 / SR)
        f0 = 440 * 2 ** ((p - 69) / 12)
        cand = f0 * 2 ** (np.linspace(-40, 40, 81) / 1200)
        sc = [sum(np.interp(k * c, f, S) for k in range(1, 7)) for c in cand]
        cents.append(np.linspace(-40, 40, 81)[int(np.argmax(sc))])
        if len(cents) >= 40:
            break
    return float(np.median(cents)) if cents else 0.0


def analyze_note(x, t0, f0, tnext, want_pitch):
    """harmonic amplitudes [frame, k], RMS envelope [2 ms steps, 0..100 ms], pitch [frames 0..120 ms] in cents"""
    nk = min(K, int(FMAX / f0))
    win = int(max(1024, 2.5 * SR / f0))
    n = np.arange(win)
    w = np.hanning(win)
    B = np.concatenate([np.cos(2 * np.pi * f0 * k * n / SR)[:, None] for k in range(1, nk + 1)] +
                       [np.sin(2 * np.pi * f0 * k * n / SR)[:, None] for k in range(1, nk + 1)], axis=1)
    P = np.linalg.pinv(B * w[:, None])
    amps = np.full((NFR, K), np.nan)
    for j in range(NFR):
        a = int((t0 + j * HOP) * SR)
        if (a + win) / SR > tnext - 0.005 or a + win > len(x):
            break
        c = P @ (x[a:a + win] * w)
        amps[j, :nk] = np.hypot(c[:nk], c[nk:])              # each partial's amplitude
    env = np.zeros(50)
    for j in range(50):
        a = int((t0 - 0.010 + j * 0.002) * SR)
        if a >= 0 and a + 88 <= len(x):
            env[j] = np.sqrt(np.mean(x[a:a + 88] ** 2))
    # pitch: harmonics 2..5 in 40 ms frames, a fine search +-60 cents
    pitch = np.full(9, np.nan)
    pw = int(0.040 * SR)
    for j in range(9 if want_pitch else 0):
        a = int((t0 + 0.005 + j * 0.0125) * SR)
        if (a + pw) / SR > tnext - 0.005 or a + pw > len(x):
            break
        seg = x[a:a + pw] * np.hanning(pw)
        cs = np.linspace(-60, 60, 61)
        best, bc = -1, 0
        for c in cs:
            f = f0 * 2 ** (c / 1200)
            s = 0.0
            for k in range(2, 6):
                ph = 2 * np.pi * f * k * np.arange(pw) / SR
                s += np.hypot(seg @ np.cos(ph), seg @ np.sin(ph))
            if s > best:
                best, bc = s, c
        pitch[j] = bc
    return amps, env, pitch


def notes_of(x, notes, tune, recs, glide_every=6):
    for i, (t0, p) in enumerate(notes):
        if p < 28 or p > 57:
            continue
        tnext = notes[i + 1][0] if i + 1 < len(notes) else t0 + 1.0
        if tnext - t0 < 0.25:
            continue
        f0 = 440 * 2 ** ((p - 69 + tune / 100) / 12)
        amps, env, pitch = analyze_note(x, t0, f0, tnext, i % glide_every == 0)
        if np.isnan(amps[3, 0]):
            continue
        recs.append((p, tnext - t0, amps, env, pitch))


def save(recs, out):
    P = np.array([r[0] for r in recs]); D = np.array([r[1] for r in recs])
    A = np.stack([r[2] for r in recs]); E = np.stack([r[3] for r in recs]); G = np.stack([r[4] for r in recs])
    np.savez_compressed(out, pitch=P, dur=D, amps=A, env=E, glide=G)
    print(f'{out}: {len(recs)} notes')


def stats(path):
    z = np.load(path)
    P, A, E, G, D = z['pitch'], z['amps'], z['env'], z['glide'], z['dur']
    out = {'n': len(P), 'bins': []}
    with np.errstate(divide='ignore', invalid='ignore'):
        for lo, hi in PB:
            m = (P >= lo) & (P <= hi)
            if m.sum() < 3:
                out['bins'].append(None)
                continue
            a = A[m]
            tot3 = np.sqrt(np.nansum(a[:, 3, :] ** 2, axis=1))
            rel3 = 20 * np.log10(a[:, 3, :] / tot3[:, None])
            rel20 = 20 * np.log10(a[:, 20, :] / np.sqrt(np.nansum(a[:, 20, :] ** 2, axis=1))[:, None])
            rel0 = 20 * np.log10(a[:, 0, :] / np.sqrt(np.nansum(a[:, 0, :] ** 2, axis=1))[:, None])
            # the attack: each partial in frames 0..3 re. its own level at frame 3 (dB)
            att = [np.nanmedian(20 * np.log10(a[:, fr, :] / a[:, 3, :]), axis=0).tolist() for fr in range(4)]
            # decay per k: dB/s over frames 4..25 (40-250 ms), notes with >= 10 frames
            dec = np.full((a.shape[0], a.shape[2]), np.nan)
            t = np.arange(a.shape[1]) * HOP
            for i in range(a.shape[0]):
                for k in range(a.shape[2]):
                    y = a[i, 4:26, k]
                    ok = np.isfinite(y) & (y > 0)
                    if ok.sum() >= 10:
                        dec[i, k] = np.polyfit(t[4:26][ok], 20 * np.log10(y[ok]), 1)[0]
            # total level over time, re. frame 3
            tot = 20 * np.log10(np.sqrt(np.nansum(a ** 2, axis=2)) / tot3[:, None])
            pitch = (lo + hi) / 2
            f0 = 440 * 2 ** ((pitch - 69) / 12)
            cen = []
            for fr in (0, 3, 10, 20, 30):
                w = a[:, fr, :] ** 2
                k = np.arange(1, a.shape[2] + 1) * f0
                cen.append(float(np.nanmedian(np.nansum(w * k, axis=1) / np.nansum(w, axis=1))))
            out['bins'].append({
                'lo': lo, 'hi': hi, 'n': int(m.sum()),
                'rel3': np.nanmedian(rel3, axis=0).tolist(), 'rel20': np.nanmedian(rel20, axis=0).tolist(),
                'rel0': np.nanmedian(rel0, axis=0).tolist(),
                'decay': np.nanmedian(dec, axis=0).tolist(),
                'tot': np.nanmedian(np.where(np.isfinite(tot), tot, np.nan), axis=0).tolist(),
                'att': att,
                'centroid': cen,
            })
        en = E / np.maximum(E[:, :].max(axis=1, keepdims=True), 1e-12)
        out['env'] = np.median(en, axis=0).tolist()
        out['peak_ms'] = float(np.median(np.argmax(E, axis=1) * 2 - 10))
        g = G[np.isfinite(G[:, 0])]
        if len(g):
            g = g - np.nanmedian(g[:, 6:9], axis=1, keepdims=True)
            out['glide'] = np.nanmedian(g, axis=0).tolist()
            out['glide_n'] = int(len(g))
    return out


def show(sts, names):
    print('notes:', ', '.join(f'{nm} {s["n"]}' for nm, s in zip(names, sts)))
    for b in range(len(PB)):
        lo, hi = PB[b]
        print(f'\n=== MIDI {lo}-{hi} (f0 {440*2**((lo-69)/12):.0f}-{440*2**((hi-69)/12):.0f} Hz)')
        hdr = 'k        ' + ''.join(f'{k:>6d}' for k in KS)
        print(hdr)
        for nm, s in zip(names, sts):
            x = s['bins'][b]
            if x is None:
                continue
            for key, lab in (('rel0', 'dB@0ms'), ('rel3', 'dB@30ms'), ('rel20', 'dB@200'), ('decay', 'dB/s')):
                v = x[key]
                print(f'{nm[:4]:4s} {lab:8s}' + ''.join(f'{v[k-1]:6.0f}' if k - 1 < len(v) and np.isfinite(v[k-1]) else '     -' for k in KS))
            for fr in (0, 1, 2):
                v = x['att'][fr]
                print(f'{nm[:4]:4s} {"f%d-f3" % fr:8s}' + ''.join(f'{v[k-1]:6.1f}' if k - 1 < len(v) and np.isfinite(v[k-1]) else '     -' for k in KS))
            print(f'{nm[:4]:4s} level re 30 ms at 0/50/100/200/300 ms: ' +
                  ' '.join(f'{x["tot"][i]:5.1f}' if np.isfinite(x["tot"][i]) else '    -' for i in (0, 5, 10, 20, 30)) +
                  f'   centroid Hz at 0/30/100/200/300 ms: ' + ' '.join(f'{c:5.0f}' for c in x['centroid']) + f'  (n {x["n"]})')
    print('\nattack: RMS (2 ms) re. its peak, at -10,-4,0,2,4,6,10,20,40,60,88 ms')
    for nm, s in zip(names, sts):
        e = s['env']
        print(f'{nm[:4]:4s} ' + ' '.join(f'{e[i]:.2f}' for i in (0, 3, 5, 6, 7, 8, 10, 15, 25, 35, 49)) + f'   peak at {s["peak_ms"]:.0f} ms')
    print('\npitch (cents re. 80-120 ms) at 25, 37, 50, 62, 75, 87 ms (frame centres)')
    for nm, s in zip(names, sts):
        if 'glide' in s:
            print(f'{nm[:4]:4s} ' + ' '.join(f'{v:+5.1f}' for v in s['glide'][:6]) + f'  (n {s["glide_n"]})')


def arr(a):
    return '{' + ', '.join(f'{v:.1f}f' for v in a) + '}'


def fit(ref_npz, out_h):
    K = KFIT
    st = stats(ref_npz)
    bins = [b for b in st['bins'] if b]
    P = np.array([(b['lo'] + b['hi']) / 2 for b in bins])
    L = np.array([b['rel3'][:K] for b in bins], dtype=float)
    D = np.array([b['decay'][:K] for b in bins], dtype=float)
    N = [b['n'] for b in bins]

    # levels: the first three as measured; above, a 3-point smoothing along k (the measurement's own wobble)
    Ls = L.copy()
    for i in range(len(bins)):
        for k in range(3, K):
            lo, hi = max(3, k - 1), min(K, k + 2)
            Ls[i, k] = np.nanmean(L[i, lo:hi])
    # decays: the first four as measured; above, a straight line in k fitted to k = 3..12 (noisy where faint)
    Ds = D.copy()
    for i in range(len(bins)):
        k = np.arange(K)
        ok = np.isfinite(D[i]) & (k >= 2)
        a, b = np.polyfit(k[ok], D[i, ok], 1)
        for j in range(4, K):
            Ds[i, j] = a * j + b
    # partials above what a register's analysis reached (3 kHz): continued from the last two (they are cut at 1.6 kHz
    # when played anyway)
    for i in range(len(bins)):
        for k in range(K):
            if not np.isfinite(Ls[i, k]):
                Ls[i, k] = Ls[i, k - 1] + min(-2.0, Ls[i, k - 1] - Ls[i, k - 2])
            if not np.isfinite(Ds[i, k]):
                Ds[i, k] = Ds[i, k - 1]
    Ds = np.clip(Ds, -400.0, -2.0)            # every partial dies, none grows
    # no partial outlasts the fundamental by much: at most 1.5x its T60
    for i in range(len(bins)):
        Ds[i, 1:] = np.minimum(Ds[i, 1:], Ds[i, 0] / 1.5)


    with open(out_h, 'w') as f:
        f.write('/* SPDX-License-Identifier: MPL-2.0 */\n')
        f.write('/* The upright\'s partials, measured in FiloBass (Riley and Dixon, ISMIR 2023; CC BY 4.0): %d notes of walking\n'
                ' * lines in 48 tunes, by register (tools/bass_measure.py). Level: dB re. the note\'s total, 30-80 ms in; decay: dB\n'
                ' * a second over 40-250 ms. Lightly smoothed above the fourth partial. */\n' % sum(N))
        f.write(f'static const float B_PITCH[BNB] = {arr(P)};\n')
        f.write('static const float B_LEVEL[BNB][BK] = {\n' + ',\n'.join('    ' + arr(r) for r in Ls) + '};\n')
        f.write('static const float B_DECAY[BNB][BK] = {\n' + ',\n'.join('    ' + arr(r) for r in Ds) + '};\n')
    print(open(out_h).read())



if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'stems':
        import glob
        import os
        d = sys.argv[2]
        names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, 'midi_fully_aligned', '*.mid')))
        if len(sys.argv) > 4:
            names = names[::max(1, len(names) // int(sys.argv[4]))][:int(sys.argv[4])]
        recs = []
        for nm in names:
            x = decode(os.path.join(d, 'audio_bass_stems', nm + '.mp3'))
            notes = midi_notes(os.path.join(d, 'midi_fully_aligned', nm + '.mid'))
            tune = tuning(x, notes)
            n0 = len(recs)
            notes_of(x, notes, tune, recs)
            print(nm, len(notes), 'notes,', len(recs) - n0, 'analysed, tuning %+.0f cents' % tune, flush=True)
        save(recs, sys.argv[3])
    elif cmd == 'wav':
        recs = []
        x = decode(sys.argv[2])
        notes = [(float(a), int(b)) for a, b in (l.split() for l in open(sys.argv[3]) if l.strip())]
        notes_of(x, notes, 0.0, recs, glide_every=1)
        save(recs, sys.argv[4])
    elif cmd == 'fit':
        fit(sys.argv[2], sys.argv[3])
    elif cmd == 'compare':
        show([stats(p) for p in sys.argv[2:]], [p.split('/')[-1].replace('.npz', '') for p in sys.argv[2:]])
    else:
        sys.exit(__doc__)
