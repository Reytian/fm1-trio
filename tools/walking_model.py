#!/usr/bin/env python3
"""Walking bass as mathematics: a constrained random walk on the bass's pitch grid.

A walking line is a sequence of pitches p_0 .. p_{n-1} (MIDI 28-55, E1-G3) on the note
onsets of a chord progression. Every line gets an energy

    E(p) = sum_s U_s(p_s) + sum_s B_s(p_s, p_{s+1}) - theta_turn * (number of direction changes)

and a probability  P(p) = exp(-E(p) / T) / Z.

* U_s  scores one note: chord tone on every chord change (hard rule), roots preferred,
       scale and chromatic notes cost more, a soft register preference.
* B_s  scores a move: interval size, and for the note before a chord change the
       approach interval (half step, whole step, fifth ...). A chromatic note must move
       by a half step (hard rule).
* T    is the temperature. T -> 0 gives the single most typical line (Viterbi);
       larger T gives freer, more surprising lines.

The theta weights are fitted by maximum entropy (moment matching) so that the model's
averages equal the FiloBass corpus of professional jazz bass lines (Riley & Dixon,
ISMIR 2023). Because a line is a chain, everything is exact dynamic programming: the
best line, random lines, marginals, entropy in bits per bar, and counting valid lines.

Usage
    python walking_model.py demo
    python walking_model.py generate "Cm7 | F7 | Bbmaj7 | Ebmaj7" --key Bb --temp 1 --seed 3
    python walking_model.py analyze "../Walking Bass 示例.gp5"
Needs:  pip install numpy pyguitarpro
"""
from __future__ import annotations

import argparse
import bisect
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LOW, HIGH = 28, 55                                  # E1 .. G3 on a 4-string bass
PITCHES = np.arange(LOW, HIGH + 1)
NP = len(PITCHES)
PCS = PITCHES % 12
DELTA = PITCHES[None, :] - PITCHES[:, None]         # DELTA[p, q] = q - p
DIST = np.abs(DELTA)
SIGN = np.sign(DELTA)
STEP = (DIST == 1) | (DIST == 2)
TPB = 6                                             # ticks per beat: eighth = 3, triplet eighth = 2
INF = np.inf
NAMES = 'C Db D Eb E F Gb G Ab A Bb B'.split()
K = 5                                               # move classes: -2 down leap, -1 down step, 0 none,
MOVE = np.where(DELTA == 0, 0, SIGN * np.where(DIST <= 2, 1, 2))   # +1 up step, +2 up leap
STATE_DIR = np.arange(K * NP) % K - 2               # state = K * pitch index + (class + 2)
_SD = np.sign(STATE_DIR)
BOTH_MOVE = (_SD[:, None] != 0) & (_SD[None, :] != 0)
TURN = BOTH_MOVE & (_SD[:, None] != _SD[None, :])
ZIGZAG = TURN & (np.abs(STATE_DIR)[:, None] == 2) & (np.abs(STATE_DIR)[None, :] == 2)

# FiloBass (Riley & Dixon 2023): 48 Aebersold tracks played by Ron Carter, Rufus Reid,
# Steve Gilmore, Tyrone Wheeler and others. Approach = interval from the note before a
# chord change to the note on it (target minus approach note).
# The approach shares are counted from FiloBass's own note_data.csv (the authors' note_interval column at
# chord changes, 15,795 events). The paper's Table 4 has the same counts with the +-1 and +-2 labels swapped.
FILOBASS = {'root_at_change': 0.679, 'root_all': 0.327, 'step_all': 0.475, 'turn': 1 / 2.46,
            'quarter_bars': 0.6281,
            'approach': {'-1': 0.2144, '+1': 0.2753, '+2': 0.0401, '+5': 0.1000, '0': 0.0593,
                         '-7': 0.0436, '-2': 0.1216, 'other': 0.1457}}
CATS = list(FILOBASS['approach'])
CAT_OF = {-1: 0, 1: 1, 2: 2, 5: 3, 0: 4, -7: 5, -2: 6}
CAT_INDEX = np.vectorize(lambda d: CAT_OF.get(int(d), 7))(DELTA)
INTERVAL_COST = np.array([1.5, 0.2, 0.0, 0.5, 0.6, 0.7, 1.6, 1.0, 2.0, 2.0, 2.2, 2.4, 1.6])  # by |move|
APPROACH_LABEL = {-1: 'chrom (above)', 1: 'chrom (below)', -2: 'scale (above)', 2: 'scale (below)',
                  5: 'dominant (5th)', -7: 'dominant (5th)', 0: 'common tone'}

# ---------------------------------------------------------------- harmony
LETTER = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
QUALITY = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'maj7': (0, 4, 7, 11),
           'm7': (0, 3, 7, 10), '6': (0, 4, 7, 9), 'm6': (0, 3, 7, 9), 'm7b5': (0, 3, 6, 10),
           'dim7': (0, 3, 6, 9), '7b9': (0, 1, 4, 7, 10)}
MODES = {'ionian': (0, 2, 4, 5, 7, 9, 11), 'dorian': (0, 2, 3, 5, 7, 9, 10),
         'phrygian': (0, 1, 3, 5, 7, 8, 10), 'lydian': (0, 2, 4, 6, 7, 9, 11),
         'mixolydian': (0, 2, 4, 5, 7, 9, 10), 'aeolian': (0, 2, 3, 5, 7, 8, 10),
         'locrian': (0, 1, 3, 5, 6, 8, 10), 'melodic minor': (0, 2, 3, 5, 7, 9, 11),
         'lydian dominant': (0, 2, 4, 6, 7, 9, 10), 'phrygian dominant': (0, 1, 4, 5, 7, 8, 10),
         'locrian #2': (0, 2, 3, 5, 6, 8, 10), 'whole-half': (0, 2, 3, 5, 6, 8, 9, 11),
         'half-whole': (0, 1, 3, 4, 6, 7, 9, 10)}
CANDIDATE_MODES = {'': ('ionian', 'lydian'), 'maj7': ('ionian', 'lydian'), '6': ('ionian', 'lydian'),
                   '7': ('mixolydian', 'lydian dominant', 'phrygian dominant'),
                   '7b9': ('phrygian dominant', 'half-whole'), 'm': ('aeolian', 'dorian', 'phrygian'),
                   'm7': ('dorian', 'aeolian', 'phrygian'), 'm6': ('melodic minor', 'dorian'),
                   'm7b5': ('locrian', 'locrian #2'), 'dim7': ('whole-half',)}
MAJOR = (0, 2, 4, 5, 7, 9, 11)
CHORD_RE = r'([A-G][#b]?)(maj7|m7b5|m7|m6|7b9|dim7|m|7|6|)(?:/([A-G][#b]?))?'


def pitch_class(name):
    return (LETTER[name[0]] + name[1:].count('#') - name[1:].count('b')) % 12


@dataclass(frozen=True)
class Chord:
    symbol: str
    root: int              # pitch class of the root
    quality: str
    tones: frozenset
    bass: int              # the note the bass is asked for: root, or the slash note


def parse_chord(symbol):
    m = re.fullmatch(CHORD_RE, symbol)
    if not m:
        raise ValueError(f'unknown chord {symbol!r}')
    root_name, quality, slash = m.groups()
    root = pitch_class(root_name)
    bass = pitch_class(slash) if slash else root
    return Chord(symbol, root, quality, frozenset((root + t) % 12 for t in QUALITY[quality]) | {bass}, bass)


def key_scale(key):
    """'Bb' -> B-flat major scale; 'Gm' -> its relative major. Returns (pitch classes, major tonic)."""
    minor = key.endswith('m')
    tonic = pitch_class(key[:-1] if minor else key)
    major = (tonic + 3) % 12 if minor else tonic
    return frozenset((major + i) % 12 for i in MAJOR), major


def chord_scale(chord, key_pcs):
    """Chord-scale theory: among the usual modes for this chord, the one closest to the key."""
    options = CANDIDATE_MODES[chord.quality]
    best = max(options, key=lambda m: (len({(chord.root + i) % 12 for i in MODES[m]} & key_pcs),
                                       -options.index(m)))
    return frozenset((chord.root + i) % 12 for i in MODES[best]) | chord.tones


def guess_key(weighted_chords):
    """Major key whose scale holds the most chord tones (weighted by duration)."""
    def fit(t):
        scale = {(t + i) % 12 for i in MAJOR}
        return sum(len(c.tones & scale) * w for c, w in weighted_chords.items())
    return NAMES[max(range(12), key=fit)]


# ---------------------------------------------------------------- time
@dataclass
class Bar:
    beats: int
    chords: list           # [(tick offset inside the bar, Chord)]


def parse_progression(text, beats=4):
    """'Cm7 | F7 | Gm7 C7' -> bars; several chords in a bar share it evenly."""
    bars = []
    for chunk in text.split('|'):
        symbols = chunk.split()
        if symbols:
            step = beats * TPB // len(symbols)
            bars.append(Bar(beats, [(i * step, parse_chord(s)) for i, s in enumerate(symbols)]))
    return bars


@dataclass
class Event:
    start: int                     # absolute tick
    dur: int
    rest: bool = False
    ghost: bool = False            # quiet pickup ("skip")
    dead: bool = False
    pitch: int | None = None
    chord: Chord | None = None
    scale: frozenset = frozenset()
    target: bool = False           # first note at or after a chord change: must be a chord tone
    strong: bool = False           # downbeat that is not a target
    approach: bool = False         # the note right before a target
    change: bool = False           # a chord symbol starts exactly here
    bar: int = 0
    string: int = 0
    fret: int = 0


# Broken-feel bars (ticks; negative = rest; 2-tick notes are triplet eighths).
BROKEN_BARS = [([12, 6, 6], 3), ([9, 3, 12], 3), ([6, -3, 3, 6, 6], 2), ([6, 2, 2, 2, 12], 2),
               ([-3, 3, 6, 12], 2), ([6, 6, 9, 3], 2), ([18, 6], 1), ([12, 2, 2, 2, 6], 2),
               ([6, 6, 6, 6], 2)]


def bar_rhythm(feel, beats, rng, skip_prob):
    """[(ticks, 'note' | 'rest' | 'ghost')] for one bar."""
    note = lambda d: (d, 'note')
    if feel == 'four':
        return [note(TPB)] * beats
    if feel == 'boogie':
        return [note(TPB // 2)] * (2 * beats)
    if feel == 'two':
        if beats % 2:
            return [note(beats * TPB)]
        out = []
        for _ in range(beats // 2):
            out += [note(9), (3, 'ghost')] if rng.random() < skip_prob else [note(12)]
        return out
    if feel == 'broken':
        patterns, weights = zip(*BROKEN_BARS)
        return [(abs(d), 'rest' if d < 0 else 'note') for d in rng.choices(patterns, weights)[0]]
    raise ValueError(f'unknown feel {feel!r}')


def annotate(events, bar_starts, target_ticks, change_ticks, key_pcs, chord_at=None, loop=True):
    """Give every event its chord, chord-scale and role (target / approach / strong beat)."""
    target_ticks, bar_set = sorted(target_ticks), set(bar_starts)
    previous, pitched = -1, []
    for e in events:
        if chord_at:
            e.chord = chord_at(e.start)
        e.scale = chord_scale(e.chord, key_pcs)
        e.bar = bisect.bisect_right(bar_starts, e.start) - 1
        e.change = e.start in change_ticks
        if e.rest or e.dead:
            continue
        i = bisect.bisect_right(target_ticks, previous)
        e.target = i < len(target_ticks) and target_ticks[i] <= e.start
        e.strong = e.start in bar_set and not e.target
        previous = e.start
        pitched.append(e)
    for a, b in zip(pitched, pitched[1:]):
        a.approach = b.target
    if loop and pitched:
        pitched[-1].approach = True        # the line loops back to its first chord
    return events


def build_events(bars, key, feel='four', rng=None, skip_prob=0.25, phrase_bars=2, loop=True):
    """Onsets of a progression in a feel, annotated with harmony and roles."""
    rng = rng or random.Random(0)
    bar_starts, changes, t = [], [], 0
    for bar in bars:
        bar_starts.append(t)
        changes += [(t + offset, chord) for offset, chord in bar.chords]
        t += bar.beats * TPB
    events = []
    if feel == 'dotted':                   # dotted quarters: 3 notes against every 2 beats... of 4/4
        events = [Event(s, min(9, t - s)) for s in range(0, t, 9)]
    else:
        for start, bar in zip(bar_starts, bars):
            for dur, kind in bar_rhythm(feel, bar.beats, rng, skip_prob):
                events.append(Event(start, dur, rest=kind == 'rest', ghost=kind == 'ghost'))
                start += dur
    change_ticks = {tick for tick, _ in changes}
    target_ticks, since = set(change_ticks), 0
    for start in bar_starts:               # long chords: a phrase target every few bars
        since = 0 if start in change_ticks else since + 1
        if since and since % phrase_bars == 0:
            target_ticks.add(start)
    chord_at = lambda tick: [c for t0, c in changes if t0 <= tick][-1]
    return annotate(events, bar_starts, target_ticks, change_ticks, key_scale(key)[0], chord_at, loop)


# ---------------------------------------------------------------- model
@dataclass
class Style:
    name: str = 'swing'
    feel: str = 'four'
    swing: bool = True
    drums: str = 'swing'
    scale_cost: float = 1.0
    chrom_cost: float = 2.0
    strong_cost: float = 1.0
    reg_center: float = 39.0
    reg_weight: float = 0.15
    repeat_cost: float = 2.5
    zigzag_cost: float = 1.0               # leap one way, then leap straight back
    temp: float = 1.0
    skip_prob: float = 0.25
    phrase_bars: int = 2
    hard: bool = True                      # False: rule breaks cost `penalty` instead of being impossible
    penalty: float = 12.0
    theta: dict = field(default_factory=lambda: {
        'root': 1.0, 'root_all': 0.0, 'step': 0.0, 'turn': 0.0,
        **{'app' + c: math.log(p / 0.2753) for c, p in FILOBASS['approach'].items()}})


def unary(e, style):
    """U_s(p) for every pitch p (numpy row)."""
    th, bad = style.theta, (INF if style.hard else style.penalty)
    tone = np.isin(PCS, list(e.chord.tones))
    scale = np.isin(PCS, list(e.scale))
    root = PCS == e.chord.bass
    u = style.reg_weight * ((PITCHES - style.reg_center) / 6.0) ** 2
    if e.target:
        u = u + np.where(tone, 0.0, bad) - th['root'] * root
    else:
        u = u + np.where(tone, 0.0, np.where(scale, style.scale_cost, style.chrom_cost))
        if e.strong:
            u = u + np.where(tone, 0.0, style.strong_cost)
    return u - th['root_all'] * root


def binary(a, b, style):
    """B_s(p, q) for every pair of pitches (numpy matrix)."""
    th, bad = style.theta, (INF if style.hard else style.penalty)
    chromatic = ~np.isin(PCS, list(a.scale | a.chord.tones))
    if b.target:
        cost = -np.array([th['app' + c] for c in CATS])[CAT_INDEX] + 0.15 * DIST * (CAT_INDEX == 7)
    else:
        cost = INTERVAL_COST[np.minimum(DIST, 12)] + np.where(DELTA == -5, 0.3, 0.0)
        cost = np.where(DIST == 0, 0.0 if b.ghost else style.repeat_cost, cost)
        cost = cost - th['step'] * STEP
    # A chromatic note moves on by a half step, or by a whole step when it starts an enclosure.
    cost = cost + np.where(chromatic[:, None] & (DIST == 2), 1.0, 0.0)
    broken_rule = (DIST > 12) | (chromatic[:, None] & ((DIST > 2) | (DIST == 0)))
    return np.where(broken_rule, bad, cost)


# Hard rule sets for practice stages and for counting how many lines exist.
RULES = {'A': dict(target='root', middle=('chord',), approach=None),
         'B': dict(target='root', middle=('chord',), approach=(-1, 1)),
         'C': dict(target='root', middle=('chord', 'scale'), approach=(-1, 1, -2, 2, 5, -7)),
         'D': dict(target='chord', middle=('chord', 'scale', 'chrom'), approach=(-1, 1, -2, 2, 5, -7))}


def rule_unary(rule):
    def u(e):
        tone, scale = np.isin(PCS, list(e.chord.tones)), np.isin(PCS, list(e.scale))
        if e.target:
            ok = (PCS == e.chord.bass) if rule['target'] == 'root' else tone
        elif e.approach and rule['approach']:
            ok = np.ones(NP, bool)                       # judged by its interval to the target
        else:
            ok = tone | (scale if 'scale' in rule['middle'] else False) | ('chrom' in rule['middle'])
        return np.where(ok, 0.0, INF)
    return u


def rule_binary(rule, max_leap=7):
    def b(first, second):
        chromatic = ~np.isin(PCS, list(first.scale | first.chord.tones))
        ok = (DIST >= 1) & (DIST <= max_leap) & ~(chromatic[:, None] & (DIST != 1))
        if second.target and rule['approach']:
            ok &= np.isin(DELTA, rule['approach'])
        return np.where(ok, 0.0, INF)
    return b


def _draw(rng, weights):
    return int(np.searchsorted(np.cumsum(weights), rng.random() * weights.sum(), side='right'))


class Chain:
    """Exact inference on one line. A state is (pitch, class of the last move)."""

    def __init__(self, events, unary_fn, binary_fn, turn_bonus=0.0, loop=True, zigzag_cost=0.0):
        self.ev = [e for e in events if not e.rest and not e.dead]
        self.U = np.array([unary_fn(e) for e in self.ev])
        self.B = [binary_fn(a, b) for a, b in zip(self.ev, self.ev[1:])]
        # Closing move: the last note approaches the first one again (any octave).
        self.G = binary_fn(self.ev[-1], self.ev[0]) + self.U[0][None, :] if loop else None
        self.turn, self.zig = turn_bonus, zigzag_cost
        self._cache = {}

    def _matrices(self, temp):
        mats = []
        with np.errstate(over='ignore'):
            for B in self.B:
                M = np.zeros((K * NP, K * NP))
                for d in range(-2, 3):
                    for dn in range(-2, 3):
                        allowed = ((MOVE == dn) & (DELTA != 0)) | ((DELTA == 0) & (dn == d))
                        turned = d * dn < 0
                        cost = B - (self.turn if turned else 0.0) + (
                            self.zig if turned and abs(d) == 2 and abs(dn) == 2 else 0.0)
                        M[d + 2::K, dn + 2::K] = np.where(allowed, np.exp(-cost / temp), 0.0)
                mats.append(M)
        return mats

    def _final(self, temp):
        return np.ones(NP) if self.G is None else np.exp(-self.G / temp).sum(axis=1)

    def forward(self, temp):
        if temp in self._cache:
            return self._cache[temp]
        mats, phis = self._matrices(temp), np.exp(-self.U / temp)
        alpha, log_c = np.zeros((len(self.ev), K * NP)), 0.0
        a = np.zeros(K * NP)
        a[2::K] = phis[0]
        for s in range(len(self.ev)):
            if s:
                a = (alpha[s - 1] @ mats[s - 1]) * np.repeat(phis[s], K)
            c = a.sum()
            if c <= 0:
                raise ValueError('no line satisfies these rules')
            alpha[s], log_c = a / c, log_c + math.log(c)
        g = np.repeat(self._final(temp), K)
        log_z = log_c + math.log((alpha[-1] * g).sum())
        self._cache[temp] = (mats, phis, alpha, g, log_z)
        return self._cache[temp]

    def sample(self, rng, temp=1.0):
        mats, _, alpha, g, _ = self.forward(temp)
        states = [_draw(rng, alpha[-1] * g)]
        for s in range(len(self.ev) - 2, -1, -1):
            states.append(_draw(rng, alpha[s] * mats[s][:, states[-1]]))
        return [LOW + x // K for x in reversed(states)]

    def best(self):
        """Most probable line (T -> 0), by Viterbi."""
        with np.errstate(divide='ignore'):
            logs = [np.log(M) for M in self._matrices(1.0)]
            final = np.log(np.exp(-self.G).max(axis=1)) if self.G is not None else np.zeros(NP)
        delta = np.full(K * NP, -INF)
        delta[2::K] = -self.U[0]
        back = []
        for s in range(1, len(self.ev)):
            scores = delta[:, None] + logs[s - 1]
            back.append(scores.argmax(axis=0))
            delta = scores.max(axis=0) + np.repeat(-self.U[s], K)
        x = int((delta + np.repeat(final, K)).argmax())
        path = [x]
        for arg in reversed(back):
            x = int(arg[x])
            path.append(x)
        return [LOW + x // K for x in reversed(path)]

    def marginals(self, temp=1.0):
        mats, phis, alpha, g, log_z = self.forward(temp)
        n = len(self.ev)
        beta = np.zeros_like(alpha)
        beta[-1] = g / g.sum()
        for s in range(n - 2, -1, -1):
            b = mats[s] @ (np.repeat(phis[s + 1], K) * beta[s + 1])
            beta[s] = b / b.sum()
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xis = []
        for s in range(n - 1):
            xi = alpha[s][:, None] * mats[s] * (np.repeat(phis[s + 1], K) * beta[s + 1])[None, :]
            xis.append(xi / xi.sum())
        return gamma, xis, log_z

    def entropy_bits(self, temp=1.0):
        """Exact entropy H(P) of the line distribution, via the chain rule."""
        gamma, xis, _ = self.marginals(temp)
        h = -np.sum(gamma[0][gamma[0] > 0] * np.log2(gamma[0][gamma[0] > 0]))
        for xi in xis:
            rows = xi.sum(axis=1, keepdims=True)
            mask = xi > 0
            h -= np.sum(xi[mask] * np.log2((xi / np.where(rows > 0, rows, 1))[mask]))
        return float(h)

    def pitch_entropy_by_slot(self, temp=1.0):
        gamma, _, _ = self.marginals(temp)
        p = gamma.reshape(len(self.ev), NP, K).sum(axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)

    def log_prob(self, pitches, temp=1.0):
        """Natural-log probability of a given line (pitches of the pitched events)."""
        idx = [p - LOW for p in pitches]
        energy = sum(self.U[s][i] for s, i in enumerate(idx))
        energy += sum(B[i, j] for B, i, j in zip(self.B, idx, idx[1:]))
        prev, turns, zigs = 0, 0, 0
        for a, b in zip(pitches, pitches[1:]):
            d = b - a
            c = prev if d == 0 else int(np.sign(d)) * (1 if abs(d) <= 2 else 2)
            if prev * c < 0:
                turns += 1
                zigs += abs(prev) == 2 and abs(c) == 2
            prev = c
        energy += -self.turn * turns + self.zig * zigs
        log_final = math.log(self._final(temp)[idx[-1]])
        return -energy / temp + log_final - self.forward(temp)[4]

    def expected_stats(self, temp=1.0):
        gamma, xis, _ = self.marginals(temp)
        n = len(self.ev)
        p = gamma.reshape(n, NP, K).sum(axis=2)
        st = defaultdict(float)
        for e, row in zip(self.ev, p):
            root = row[PCS == e.chord.bass].sum()
            st['n'] += 1
            st['root'] += root
            st['chromatic'] += row[~np.isin(PCS, list(e.scale | e.chord.tones))].sum()
            if e.target:
                st['n_t'] += 1
                st['root_t'] += root
        for s, xi in enumerate(xis):
            pq = xi.reshape(NP, K, NP, K).sum(axis=(1, 3))
            st['moves'] += 1
            st['step'] += pq[STEP].sum()
            st['turn'] += xi[TURN].sum()
            st['turn_opp'] += xi[BOTH_MOVE].sum()
            if self.ev[s + 1].target:
                st['n_app'] += 1
                for k, c in enumerate(CATS):
                    st['app' + c] += pq[CAT_INDEX == k].sum()
            else:
                st['n_free'] += 1
                st['step_free'] += pq[STEP].sum()
        return st

    def conditional_entropy_next(self, temp=1.0):
        """H(note s | note s+1) in bits: how open a note still is once you know the next one."""
        gamma, xis, _ = self.marginals(temp)
        out = []
        for xi in xis:
            pq = xi.reshape(NP, K, NP, K).sum(axis=(1, 3))
            q = pq.sum(axis=0)
            joint = -np.sum(pq[pq > 0] * np.log2(pq[pq > 0]))
            out.append(joint + np.sum(q[q > 0] * np.log2(q[q > 0])))
        return out


def model_chain(events, style, loop=True):
    return Chain(events, lambda e: unary(e, style), lambda a, b: binary(a, b, style),
                 style.theta['turn'], loop, style.zigzag_cost)


def logit(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def fit(style, progressions, iters=80, eta=0.6):
    """Maximum-entropy fit: move theta until expected features equal the FiloBass rates."""
    th, f = style.theta, FILOBASS
    for _ in range(iters):
        st = defaultdict(float)
        for bars, key in progressions:
            chain = model_chain(build_events(bars, key, style.feel, phrase_bars=style.phrase_bars), style)
            for k, v in chain.expected_stats().items():
                st[k] += v
        share = st['n_app'] / st['moves']   # moves into chord changes; the rest are free moves
        app_steps = sum(f['approach'][c] for c in ('-1', '+1', '+2', '-2'))
        free_step_goal = (f['step_all'] - share * app_steps) / (1 - share)
        rates = {'root': st['root_t'] / st['n_t'], 'root_all': st['root'] / st['n'],
                 'step': st['step_free'] / st['n_free'], 'turn': st['turn'] / st['turn_opp']}
        goals = {'root': f['root_at_change'], 'root_all': f['root_all'], 'step': free_step_goal,
                 'turn': f['turn']}
        for k in rates:
            th[k] += eta * (logit(goals[k]) - logit(rates[k]))
        for c in CATS:
            rate = max(st['app' + c] / st['n_app'], 1e-9)
            th['app' + c] += eta * (math.log(f['approach'][c]) - math.log(rate))
        top = max(th['app' + c] for c in CATS)
        for c in CATS:
            th['app' + c] -= top
    model = {**rates, 'step_all': st['step'] / st['moves'],
             **{'app' + c: st['app' + c] / st['n_app'] for c in CATS}}
    return model


# ---------------------------------------------------------------- styles that are functions
def register(pc, low=LOW):
    """The lowest pitch of this pitch class at or above `low`."""
    return low + (pc - low) % 12


def boogie_line(events):
    """Boogie-woogie: a fixed eighth-note figure on every chord. Zero choices = 0 bits."""
    pitched = [e for e in events if not e.rest]
    first_root = register(pitched[0].chord.bass)
    for i, e in enumerate(pitched):
        minor = e.chord.quality.startswith('m')
        figure = (0, 3, 7, 9, 10, 9, 7, 3) if minor else (0, 4, 7, 9, 10, 9, 7, 4)
        slot = (e.start % (4 * TPB)) // (TPB // 2)
        e.pitch = register(e.chord.bass) + figure[slot]
    pitched[-1].pitch = first_root + 1        # turnaround: half step above the first root
    return events


def country_events(bars, key):
    """Country two-beat: root and fifth, with a two-note walk into every new chord."""
    key_pcs, _ = key_scale(key)
    events, t, bar_starts = [], 0, []
    roots = [register(bar.chords[0][1].bass, 31) if register(bar.chords[0][1].bass, 31) <= 42
             else register(bar.chords[0][1].bass) for bar in bars]
    for i, bar in enumerate(bars):
        bar_starts.append(t)
        chord, root = bar.chords[0][1], roots[i]
        nxt = roots[(i + 1) % len(bars)]
        if bars[(i + 1) % len(bars)].chords[0][1].root == chord.root:
            fifth = root - 5 if root - 5 >= LOW else root + 7
            events += [Event(t, 12, pitch=root), Event(t + 12, 12, pitch=fifth)]
        else:
            target = min((nxt + k for k in (-12, 0, 12) if LOW <= nxt + k <= HIGH), key=lambda q: abs(q - root))
            way = 1 if target > root else -1

            def cost(x1, x2):
                return (x1 % 12 not in key_pcs) + 0.4 * (x2 % 12 not in key_pcs) + 0.05 * abs(x1 - root)
            walks = [(x1, x2) for x2 in range(target - 2 * way, target, way) if (target - x2) * way in (1, 2)
                     for x1 in (x2 - way, x2 - 2 * way) if (x1 - root) * way > 0]
            x1, x2 = min(walks, key=lambda w: cost(*w))
            events += [Event(t, 12, pitch=root), Event(t + 12, 6, pitch=x1), Event(t + 18, 6, pitch=x2)]
        t += bar.beats * TPB
    changes = [(s + off, c) for s, bar in zip(bar_starts, bars) for off, c in bar.chords]
    chord_at = lambda tick: [c for t0, c in changes if t0 <= tick][-1]
    ticks = {tick for tick, _ in changes}
    return annotate(events, bar_starts, ticks, ticks, key_pcs, chord_at)


# ---------------------------------------------------------------- measuring a line
def features(events):
    notes = [e for e in events if not e.rest and not e.dead and e.pitch is not None]
    moves = [b.pitch - a.pitch for a, b in zip(notes, notes[1:])]
    into = [b.pitch - a.pitch for a, b in zip(notes, notes[1:]) if b.target]
    targets = [e for e in notes if e.target]
    runs, run, sign = [], 0, 0
    for m in moves:
        if not m:
            continue
        s = 1 if m > 0 else -1
        if s == sign:
            run += 1
        else:
            if run:
                runs.append(run)
            run, sign = 1, s
    runs.append(run)
    bars = defaultdict(list)
    for e in events:
        bars[e.bar].append(e)
    four = [all(e.dur == TPB and not e.rest for e in b) and len(b) == 4 for b in bars.values()]
    beats = sum(e.dur for e in events) / TPB
    share = lambda xs, cond: sum(1 for x in xs if cond(x)) / max(len(xs), 1)
    return {
        'notes/beat': len([e for e in events if not e.rest]) / beats,
        'root at change': share(targets, lambda e: e.pitch % 12 == e.chord.bass),
        'half-step approach': share(into, lambda d: abs(d) == 1),
        'whole-step approach': share(into, lambda d: abs(d) == 2),
        '5th approach': share(into, lambda d: d in (5, -7)),
        'steps': share(moves, lambda d: abs(d) in (1, 2)),
        'leaps': share(moves, lambda d: abs(d) >= 3),
        'repeats': share(moves, lambda d: d == 0),
        'chromatic notes': share(notes, lambda e: e.pitch % 12 not in (e.scale | e.chord.tones)),
        'run length': sum(runs) / max(len(runs), 1),
        'range': (max(e.pitch for e in notes) - min(e.pitch for e in notes)) if notes else 0,
        'plain 4-quarter bars': sum(four) / max(len(four), 1)}


def surprisal(events, style):
    """(bits, rule breaks): how surprising a whole line is under a (fitted) style.
    Rule breaks are counted and charged a finite penalty instead of making P = 0."""
    soft = replace(style, hard=False, theta=dict(style.theta))
    chain = model_chain(events, soft, loop=False)
    pitches = [e.pitch for e in chain.ev]
    breaks = sum(1 for s, i in enumerate(pitches) if not np.isfinite(model_chain_term(chain, s, i, style)))
    return -chain.log_prob(pitches) / math.log(2), breaks


def model_chain_term(chain, s, pitch, style):
    """Hard-rule check for one note and the move out of it (used to count rule breaks)."""
    e = chain.ev[s]
    u = unary(e, style)[pitch - LOW]
    if s + 1 < len(chain.ev):
        u += binary(e, chain.ev[s + 1], style)[pitch - LOW, chain.ev[s + 1].pitch - LOW]
    return u


# ---------------------------------------------------------------- Guitar Pro
OPEN = {1: 43, 2: 38, 3: 33, 4: 28}


def finger(notes):
    """String/fret by dynamic programming: one-finger-per-fret box, few shifts, low positions."""
    def candidates(pitch):
        return [(s, pitch - OPEN[s]) for s in OPEN if 0 <= pitch - OPEN[s] <= 15]
    layers = [{}]
    for s, f in candidates(notes[0].pitch):
        for hand in ([1] if f == 0 else range(max(1, f - 3), f + 1)):
            layers[0][(s, f, hand)] = (0.03 * hand, None)
    for note in notes[1:]:
        layer = {}
        for s, f in candidates(note.pitch):
            for (ps, pf, hand), (cost, _) in layers[-1].items():
                options = [(hand, 0.1)] if f == 0 else (
                    [(hand, 0.0)] if hand <= f <= hand + 3 else
                    [(h, 1.0 + 0.15 * abs(h - hand)) for h in range(max(1, f - 3), f + 1)])
                for new_hand, move in options:
                    total = cost + move + 0.05 * abs(s - ps) + 0.03 * new_hand + (1.0 if f > 12 else 0)
                    if (s, f, new_hand) not in layer or total < layer[(s, f, new_hand)][0]:
                        layer[(s, f, new_hand)] = (total, (ps, pf, hand))
        layers.append(layer)
    state = min(layers[-1], key=lambda k: layers[-1][k][0])
    for note, layer in zip(reversed(notes), reversed(layers)):
        note.string, note.fret = state[0], state[1]
        state = layer[state][1]


def read_gp(path):
    """Sections (split at markers) of the first track: [(title, events, bar count)]."""
    import guitarpro as gp
    song = gp.parse(str(path))
    sections, chord, t = [], None, 0
    for measure in song.tracks[0].measures:
        header = measure.header
        if header.marker or not sections:
            sections.append({'title': header.marker.title if header.marker else Path(path).stem,
                             'events': [], 'bar_starts': [], 'changes': set()})
        sec = sections[-1]
        sec['bar_starts'].append(t)
        for beat in measure.voices[0].beats:
            dur = beat.duration.time * TPB // 960
            m = re.match(CHORD_RE + r'(?=\s|$)', beat.text or '')
            if m:
                chord = parse_chord(m.group(0))
                sec['changes'].add(t)
            notes = [n for n in beat.notes]
            if notes and notes[0].type == gp.NoteType.tie and sec['events']:
                sec['events'][-1].dur += dur
            elif chord is not None:
                n = notes[0] if notes else None
                dead = bool(n) and n.type == gp.NoteType.dead
                sec['events'].append(Event(t, dur, rest=not notes or beat.status == gp.BeatStatus.rest,
                                           ghost=bool(n) and n.effect.ghostNote, dead=dead,
                                           pitch=None if not n or dead else n.realValue, chord=chord))
            t += dur
    out = []
    for sec in sections:
        events = sec['events']
        if not any(e.pitch for e in events):
            continue
        weights = Counter()
        for e in events:
            weights[e.chord] += e.dur
        key_pcs = key_scale(guess_key(weights))[0]
        annotate(events, sec['bar_starts'], sec['changes'], sec['changes'], key_pcs, None, loop=False)
        out.append((sec['title'], events, len(sec['bar_starts'])))
    return out


def key_signature(key):
    from guitarpro.models import KeySignature
    _, major = key_scale(key)
    fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}[major]
    return KeySignature((fifths, 1 if key.endswith('m') else 0))


def drum_bar(style, beats):
    ride, foot, hat, kick, snare = 51, 44, 42, 36, 38
    if style == 'swing':
        out = []
        for _ in range(beats // 2):
            out += [(6, [(ride, 1, 95)]), (3, [(ride, 1, 100), (foot, 5, 85)]), (3, [(ride, 1, 70)])]
        if beats % 2:
            out.append((6, [(ride, 1, 95)] + ([(foot, 5, 80)] if beats == 3 else [])))
        return out
    out = []
    for b in range(beats):
        accent = (kick, 6, 105) if b % 2 == 0 else (snare, 3, 100)
        out += [(3, [(hat, 2, 90), accent]), (3, [(hat, 2, 65)])]
    return out


def write_gp(sections, path, tempo=120):
    """sections: dicts with title, events (pitched), bars, key, swing, drums."""
    import guitarpro as gp
    from guitarpro.models import (Beat, BeatStatus, Color, Duration, GuitarString, Marker,
                                  MeasureHeader, MidiChannel, Note, NoteType, Song, Track,
                                  TripletFeel, Tuplet)
    gp_dur = {24: (1, False, False), 18: (2, True, False), 12: (2, False, False), 9: (4, True, False),
              6: (4, False, False), 3: (8, False, False), 2: (8, False, True)}

    def pieces(n):
        out = []
        for size in (24, 18, 12, 9, 6, 3, 2):
            while n >= size:
                out.append(size)
                n -= size
        return out

    song = Song(title='Walking Bass Model', subtitle='Lines generated by walking_model.py', tempo=tempo)
    headers, plans, start = [], [], Duration.quarterTime
    for sec in sections:
        events = sec['events']
        finger([e for e in events if not e.rest])
        pitched = [e for e in events if not e.rest]
        labels = {id(a): APPROACH_LABEL.get(b.pitch - a.pitch) for a, b in zip(pitched, pitched[1:]) if b.target}
        bar_starts, t = [], 0
        for bar in sec['bars']:
            bar_starts.append(t)
            t += bar.beats * TPB
        change_at = {s + off: c.symbol for s, bar in zip(bar_starts, sec['bars']) for off, c in bar.chords}
        per_bar = defaultdict(list)
        for e in events:
            s, end, first = e.start, e.start + e.dur, True
            while s < end:
                b = bisect.bisect_right(bar_starts, s) - 1
                seg_end = min(end, bar_starts[b] + sec['bars'][b].beats * TPB)
                for size in pieces(seg_end - s):
                    per_bar[b].append((e, s, size, not first))
                    s += size
                    first = False
        for b, bar in enumerate(sec['bars']):
            header = MeasureHeader(number=len(headers) + 1, start=start, keySignature=key_signature(sec['key']),
                                   tripletFeel=TripletFeel.eighth if sec['swing'] else TripletFeel.none)
            header.timeSignature.numerator = bar.beats
            if b == 0:
                header.marker = Marker(title=sec['title'], color=Color(200, 40, 40))
            headers.append(header)
            start += header.length
            plans.append((sec, per_bar[b], change_at, labels))
    song.measureHeaders = headers
    bass = Track(song, number=1, name='Bass', strings=[GuitarString(s, OPEN[s]) for s in OPEN],
                 channel=MidiChannel(channel=0, effectChannel=1, instrument=33, volume=120))
    drums = Track(song, number=2, name='Drums', isPercussionTrack=True,
                  strings=[GuitarString(s, 0) for s in range(1, 7)],
                  channel=MidiChannel(channel=9, effectChannel=9, instrument=0, volume=85))
    song.tracks = [bass, drums]
    for measure, (sec, segments, change_at, labels) in zip(bass.measures, plans):
        voice, offset = measure.voices[0], measure.start
        for e, s, size, cont in segments:
            value, dotted, triplet = gp_dur[size]
            beat = Beat(voice, duration=Duration(value=value, isDotted=dotted,
                                                 tuplet=Tuplet(3, 2) if triplet else Tuplet()),
                        start=offset, status=BeatStatus.rest if e.rest else BeatStatus.normal)
            texts = [change_at[s]] if s in change_at else []
            if not cont and labels.get(id(e)):
                texts.append(labels[id(e)])
            if texts:
                beat.text = ' | '.join(texts)
            if not e.rest:
                note = Note(beat, value=e.fret, string=e.string, velocity=60 if e.ghost else 95,
                            type=NoteType.tie if cont else NoteType.normal)
                note.effect.ghostNote = e.ghost and not cont
                beat.notes.append(note)
            voice.beats.append(beat)
            offset += beat.duration.time
    for measure, (sec, _, _, _) in zip(drums.measures, plans):
        voice, offset = measure.voices[0], measure.start
        for size, hits in drum_bar(sec['drums'], measure.header.timeSignature.numerator):
            value, dotted, triplet = gp_dur[size]
            beat = Beat(voice, duration=Duration(value=value, isDotted=dotted), start=offset,
                        status=BeatStatus.normal)
            for midi, string, velocity in hits:
                beat.notes.append(Note(beat, value=midi, string=string, type=NoteType.normal, velocity=velocity))
            voice.beats.append(beat)
            offset += beat.duration.time
    gp.write(song, str(path))


# ---------------------------------------------------------------- demo
AUTUMN_A = 'Cm7 | F7 | Bbmaj7 | Ebmaj7 | Am7b5 | D7 | Gm7 | Gm7'
F_BLUES = 'F7 | Bb7 | F7 | Cm7 F7 | Bb7 | Bdim7 | F7 | D7 | Gm7 | C7 | F7 D7 | Gm7 C7'
MODAL = 'Dm7 | Dm7 | Dm7 | Dm7 | Dm7 | Dm7 | Dm7 | Dm7'
A_BLUES = 'A7 | A7 | A7 | A7 | D7 | D7 | A7 | A7 | E7 | D7 | A7 | E7'
POP_E = 'E | C#m | A | B7 | E | C#m | F#m7 | B7'
COUNTRY_G = 'G | G | C | C | G | G | D | D'


LETTERS = 'CDEFGAB'
DEGREE_STEPS = (0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6)    # semitones above the root -> letter steps


def spell_pc(pc, letter):
    acc = (pc - LETTER[letter] + 6) % 12 - 6
    if acc not in (-1, 0, 1) or (letter, acc) in (('E', 1), ('B', 1), ('F', -1), ('C', -1)):
        return NAMES[pc].replace('b', '♭')
    return letter + {0: '', 1: '♯', -1: '♭'}[acc]


def spell_line(events):
    """Names that read like a chart: chord and scale tones spelled from the chord root,
    chromatic approach notes from the note they lead into (C♯→D, D♭→C)."""
    notes = [e for e in events if not e.rest and not e.dead and e.pitch is not None]
    letters = {}
    for e in notes:
        interval = (e.pitch - e.chord.root) % 12
        steps = DEGREE_STEPS[interval]
        if interval == 6 and e.chord.quality in ('m7b5', 'dim7'):
            steps = 4                                   # flat five, not sharp eleven
        if interval == 9 and e.chord.quality == 'dim7':
            steps = 6                                   # diminished seventh
        letters[id(e)] = LETTERS[(LETTERS.index(e.chord.symbol[0]) + steps) % 7]
    pairs = list(zip(notes, notes[1:]))
    if notes and notes[-1].approach:
        pairs.append((notes[-1], notes[0]))             # a looping line leads back to its start
    for a, b in reversed(pairs):
        chromatic = a.pitch % 12 not in (a.scale | a.chord.tones)
        if (b.pitch - a.pitch) % 12 in (1, 11) and (chromatic or letters[id(a)] == letters[id(b)]):
            i = LETTERS.index(letters[id(b)])
            up = (b.pitch - a.pitch) % 12 == 1
            letters[id(a)] = LETTERS[(i - 1) % 7] if up else LETTERS[(i + 1) % 7]
    return {id(e): spell_pc(e.pitch % 12, letters[id(e)]) for e in notes}


def pretty(symbol):
    return re.sub(r'(?<=[A-G])b', '♭', symbol).replace('#', '♯').replace('7b5', '7♭5').replace('7b9', '7♭9')


def line_text(events):
    """'Cm7: C B♭ A G | F7: ...' for a realized line. (x) = quiet pickup, - = rest."""
    spelled, out, bar, chord = spell_line(events), [], None, None
    for e in events:
        if e.bar != bar:
            out.append('|')
            bar = e.bar
        if e.change or (chord and e.chord.symbol != chord):
            out.append(pretty(e.chord.symbol) + ':')
        chord = e.chord.symbol
        name = '-' if e.rest else spelled[id(e)]
        out.append(f'({name})' if e.ghost else name)
    return ' '.join(out[1:])


def realize(events, pitches):
    for e, p in zip([e for e in events if not e.rest], pitches):
        e.pitch = p
    return events


def table(rows, cols):
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for r in rows:
        print('| ' + ' | '.join(r) + ' |')


def pct(x):
    return f'{100 * x:.0f}%'


def demo(out_dir=HERE):
    print('# walking_model.py demo\n')
    swing = Style()
    train = [(parse_progression(AUTUMN_A), 'Gm'), (parse_progression(F_BLUES), 'F')]
    model = fit(swing, train)
    f = FILOBASS
    print('## 1. Max-entropy fit to FiloBass (Autumn Leaves A + F blues, T = 1)\n')
    rows = [('root on chord change', f['root_at_change'], model['root']), ('root, all notes', f['root_all'], model['root_all']),
            ('steps (1-2 semitones), all moves', f['step_all'], model['step_all']),
            ('direction change per move (1/run)', f['turn'], model['turn'])]
    rows += [(f'approach {c}', f['approach'][c], model['app' + c]) for c in CATS]
    table([(a, pct(b), pct(c)) for a, b, c in rows], ['feature', 'FiloBass', 'fitted model'])
    print('\ntheta:', {k: round(v, 2) for k, v in swing.theta.items()})

    autumn = parse_progression(AUTUMN_A)
    events = build_events(autumn, 'Gm')
    chain = model_chain(events, swing)
    ent, cond = chain.pitch_entropy_by_slot(), chain.conditional_entropy_next()
    by_beat = defaultdict(lambda: ([], []))
    for s, e in enumerate(chain.ev):
        beat = (e.start % (4 * TPB)) // TPB + 1
        by_beat[beat][0].append(ent[s])
        if s < len(cond):
            by_beat[beat][1].append(cond[s])
    print('\n## 2. Information per beat, Autumn Leaves A, fitted model\n')
    table([(f'beat {b}', f'{np.mean(h):.2f}', f'{np.mean(c):.2f}') for b, (h, c) in sorted(by_beat.items())],
          ['beat', 'bits (how open the note is)', 'bits once the next note is known'])
    print(f'\nwhole line: {chain.entropy_bits():.1f} bits = {chain.entropy_bits() / 8:.1f} bits per bar')

    print('\n## 3. How many lines obey the rules?\n')
    one_bar = build_events(parse_progression('Cm7 | F7'), 'Gm', loop=False)[:5]
    rows = []
    for name, rule in RULES.items():
        bar_chain = Chain(one_bar, rule_unary(rule), rule_binary(rule), loop=False)
        full = Chain(build_events(autumn, 'Gm', loop=False), rule_unary(rule), rule_binary(rule), loop=False)
        n_bar = math.exp(bar_chain.forward(1.0)[4])
        bits = full.forward(1.0)[4] / math.log(2)
        rows.append((name, f'{n_bar:.0f}', f'{bits:.1f}', f'{bits / 8:.1f}'))
    table(rows, ['rule set', 'lines for one bar Cm7 -> F', 'bits, 8 bars', 'bits per bar'])

    sections, style_rows = [], []
    rng = random.Random(2026)

    def add(title, events, bars, key, style, pitches=None):
        if pitches is not None:
            realize(events, pitches)
        sections.append(dict(title=title, events=events, bars=bars, key=key, swing=style.swing, drums=style.drums))
        print(f'\n**{title}**\n`{line_text(events)}`')

    print('\n## 4. Practice stages (uniform random line among all legal ones)')
    for name, rule in RULES.items():
        ev = build_events(autumn, 'Gm')
        c = Chain(ev, rule_unary(rule), rule_binary(rule))
        add(f'Stage {name} ({", ".join(rule["middle"])}; approach {rule["approach"] or "free"})',
            ev, autumn, 'Gm', swing, c.sample(rng))

    print('\n## 5. Temperature sweep, Autumn Leaves A')
    sweep = []
    for temp in (0, 0.6, 1.0, 1.6):
        ev = build_events(autumn, 'Gm')
        c = model_chain(ev, swing)
        pitches = c.best() if temp == 0 else c.sample(rng, temp)
        add(f'T = {temp} ({"Viterbi: most typical line" if temp == 0 else "one random line"})',
            ev, autumn, 'Gm', swing, pitches)
        if temp == 0:
            x = features(ev)
            sweep.append(('0 (best line)', 0.0, x['root at change'], x['half-step approach'], x['steps'],
                          x['chromatic notes'], x['run length']))
        else:
            st = c.expected_stats(temp)
            sweep.append((str(temp), c.entropy_bits(temp) / 8, st['root_t'] / st['n_t'],
                          (st['app-1'] + st['app+1']) / st['n_app'], st['step'] / st['moves'],
                          st['chromatic'] / st['n'], st['turn_opp'] / st['turn']))
    print('\nExpected values (exact, from the marginals), except T = 0 which is the single best line:\n')
    table([(t, f'{b:.1f}', pct(r), pct(h), pct(s), pct(ch), f'{run:.2f}') for t, b, r, h, s, ch, run in sweep],
          ['T', 'bits/bar', 'root on change', 'half-step approach', 'steps', 'chromatic', 'run length'])

    print('\n## 6. Styles as parameter settings')
    styles = [
        ('Two-feel with skips', AUTUMN_A, 'Gm', replace(swing, name='two', feel='two', skip_prob=0.35,
                                                         theta={**swing.theta, 'root': swing.theta['root'] + 1.5})),
        ('Four-feel, F blues (fitted)', F_BLUES, 'F', swing),
        ('Bebop: more chromatic', AUTUMN_A, 'Gm', replace(swing, name='bebop', chrom_cost=0.8, theta={
            **swing.theta, 'app-1': swing.theta['app-1'] + 0.6, 'app+1': swing.theta['app+1'] + 0.6})),
        ('Modal: 8 bars of Dm7, phrase targets every 2 bars', MODAL, 'C', replace(swing, name='modal', scale_cost=0.3,
                                                                                    theta={**swing.theta, 'turn': swing.theta['turn'] - 1.0})),
        ('Broken feel', AUTUMN_A, 'Gm', replace(swing, name='broken', feel='broken', theta=dict(swing.theta))),
        ('5/4 walking', 'Cm7 | F7 | Bbmaj7 | Ebmaj7', 'Gm', replace(swing, name='5/4', theta=dict(swing.theta))),
        ('3 over 4: dotted quarters across the bar line', 'Cm7 | F7 | Bbmaj7 | Ebmaj7 | Am7b5 | D7', 'Gm',
         replace(swing, name='dotted', feel='dotted', theta=dict(swing.theta))),
        ('Pop / rock: straight, mostly in the key', POP_E, 'E', replace(swing, name='pop', swing=False, drums='straight',
                                                                         chrom_cost=4.0, scale_cost=0.5, theta={
            **swing.theta, 'app+2': swing.theta['app+2'] + 0.8, 'app-2': swing.theta['app-2'] + 0.8})),
    ]
    for title, prog, key, st in styles:
        bars = parse_progression(prog, beats=5 if st.name == '5/4' else 4)
        ev = build_events(bars, key, st.feel, rng, st.skip_prob, st.phrase_bars)
        c = model_chain(ev, st)
        add(title, ev, bars, key, st, c.sample(rng, st.temp))
        samples = []
        for k in range(200):
            ev2 = build_events(bars, key, st.feel, random.Random(k), st.skip_prob, st.phrase_bars)
            c2 = model_chain(ev2, st)
            samples.append(features(realize(ev2, c2.sample(random.Random(1000 + k)))))
        mean = {k: np.mean([s[k] for s in samples]) for k in samples[0]}
        style_rows.append((title, c.entropy_bits() / len(bars), mean))

    boogie_bars = parse_progression(A_BLUES)
    ev = boogie_line(build_events(boogie_bars, 'A', 'boogie'))
    add('Boogie / shuffle: a function of the chord (0 bits)', ev, boogie_bars, 'A',
        replace(swing, drums='straight'))
    style_rows.append(('Boogie / shuffle (function)', 0.0, features(ev)))
    country_bars = parse_progression(COUNTRY_G)
    ev = country_events(country_bars, 'G')
    add('Country two-beat: a function of the chords (0 bits)', ev, country_bars, 'G',
        replace(swing, swing=False, drums='straight'))
    style_rows.append(('Country two-beat (function)', 0.0, features(ev)))

    print('\n## 7. What each style looks like in numbers (stochastic styles: mean of 200 lines)\n')
    keys = ['notes/beat', 'root at change', 'half-step approach', 'steps', 'leaps', 'chromatic notes', 'run length']
    table([(t, f'{b:.1f}', f"{m['notes/beat']:.2f}", *[pct(m[k]) for k in keys[1:6]], f"{m['run length']:.2f}")
           for t, b, m in style_rows], ['style', 'bits/bar'] + keys)

    print('\n## 8. Measuring hand-written lines\n')
    rows = []
    sources = [HERE.parent / 'Walking Bass 示例.gp5',
               HERE.parents[2] / 'Songs' / 'New Light - John Mayer' / 'New Light - Bass (Jazz Walking).gp5']
    for src in sources:
        if not src.exists():
            continue
        for title, ev, n_bars in read_gp(src):
            if sum(1 for e in ev if e.target) < 3:
                continue
            x = features(ev)
            bits, breaks = surprisal(ev, swing)
            n_notes = sum(1 for e in ev if not e.rest and not e.dead)
            rows.append((title[:46], f'{bits / n_bars:.1f}', f'{bits / n_notes:.2f}', str(breaks),
                         pct(x['root at change']), pct(x['half-step approach']), pct(x['5th approach']),
                         pct(x['steps']), pct(x['chromatic notes']), f"{x['run length']:.2f}"))
    table(rows, ['line', 'surprise bits/bar', 'bits/note', 'rule breaks', 'root on change',
                 'half-step approach', '5th approach', 'steps', 'chromatic', 'run length'])

    path = out_dir / 'Walking Bass 模型生成.gp5'
    write_gp(sections, path)
    print(f'\nWrote {path} ({sum(len(s["bars"]) for s in sections)} bars)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('demo')
    g = sub.add_parser('generate')
    g.add_argument('progression')
    g.add_argument('--key', required=True)
    g.add_argument('--feel', default='four', choices=['four', 'two', 'broken', 'dotted'])
    g.add_argument('--temp', type=float, default=1.0)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--beats', type=int, default=4)
    g.add_argument('--out')
    a = sub.add_parser('analyze')
    a.add_argument('file')
    args = ap.parse_args()
    if args.cmd == 'demo':
        demo()
        return
    style = Style()
    fit(style, [(parse_progression(AUTUMN_A), 'Gm'), (parse_progression(F_BLUES), 'F')])
    if args.cmd == 'generate':
        bars = parse_progression(args.progression, args.beats)
        rng = random.Random(args.seed)
        events = build_events(bars, args.key, args.feel, rng)
        chain = model_chain(events, style)
        realize(events, chain.best() if args.temp == 0 else chain.sample(rng, args.temp))
        print(line_text(events))
        if args.out:
            write_gp([dict(title=args.progression, events=events, bars=bars, key=args.key, swing=True,
                           drums='swing')], Path(args.out))
            print('Wrote', args.out)
    else:
        for title, events, n_bars in read_gp(args.file):
            bits, breaks = surprisal(events, style)
            print(f'\n{title}: {bits / n_bars:.1f} bits/bar, {breaks} rule breaks')
            print('  ' + line_text(events))
            for k, v in features(events).items():
                print(f'  {k}: {v:.2f}')


if __name__ == '__main__':
    main()
