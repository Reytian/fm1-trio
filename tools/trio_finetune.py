#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""TEACHING the bass (TRIO's teach mode, app/dsp/trio.c tr_teach): the reference for app/dsp/trio_nn.c's learning,
and the numbers its settings come from.

  tools/trio_finetune.py sweep LESSON...    the table: learning rates, passes, the pull back toward the trained rows
  tools/trio_finetune.py golden LESSON      tests/host/ft_golden.h for tests/host/ft_test.c
  tools/trio_finetune.py check LESSON       one setting (the firmware's), its numbers

Run it with the PyTorch venv (~/fm1/torch-venv). A LESSON is what tests/host/teach_test.c --dump writes: the BASS
network's state where the teaching began, then each beat decided while teaching, its inputs as decide_bass built them
(the player's note on the beat before as the bass's last) and what the player played on it.

The network is read from app/dsp/trio_w.h, as the FM-1 runs it: each int8 matrix times its scale. It is frozen but
for its output layer: the 35 tokens' rows and the pickup's (96 weights and a bias each), float32. The lesson goes
through the frozen GRU once from its state; each beat's state is kept as the C keeps it (Q15). A beat teaches its
token when the bass could have played that token there (bass_allow), and its pickup where one could follow (in two
on beats 2 and 4, in four after a note); the last 128 such beats are the lesson (the C's ring). Then passes over
them in order, an SGD step each, float32 in the C's order:
  the token: z = W h + b on the tokens allowed there, the engine's nudges added (bass_shape: on a change the chord's
             notes up, the root most; above A2 down), the softmax, g = p - onehot(token);
  the pickup: s = w h + b, g = 0.3 (sigmoid(s) - pickup) (trio_train.py weights its loss 0.3);
  each row moved: W -= lr (g h + l2 (W - W0)), b -= lr (g + l2 (b - b0)).
The sweep scores each setting by the lesson's cross-entropy after it (teacher-forced), by how often the bass, playing
on its own over the lesson's chords (sampled at FREEDOM 50, as decide_bass), plays the taught note, before and after,
and by the cross-entropy of FiloBass's 48 lines (what the network does apart from the player's line: the side
effect), with the trained rows and the learned. app/dsp/trio.c's LRN_RATE, LRN_PULL, LRN_PASSES (0.03, 0.1, 15): with
the pull, the passes reach a fixed point, 15 within 0.001 nats of 30 or 60. Autumn Leaves taught twice (the R 5 8
line of teach_test.c): the lesson from 1.587 to 0.315 nats a beat; the bass on its own plays the taught note 22% ->
51% of the beats (FREEDOM 0: 23% -> 92%); FiloBass 1.535 -> 1.707 nats (no pull: the line 97%, FiloBass +0.84).

The lessons:  tests/host/teach_test.c (tests/run_tests.sh builds it) autumn 160 1 --dump L.txt; golden: blues 160 1
--teach 1 --dump L.txt. The FiloBass lines come from ~/fm1/data/trio/bass.npz (tools/trio_data.py, docs/DATA.md).
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
WEIGHTS = HERE / "app" / "dsp" / "trio_w.h"
GOLDEN = HERE / "tests" / "host" / "ft_golden.h"
LO, HI, REST, HOLD = 28, 60, 0, 1
NTOK = 2 + HI - LO + 1                      # 35
H = 96
LINE_HI = 55
ROWS = NTOK + 1                             # the tokens', then the pickup's
TEACH_MAX = 128                             # app/dsp/trio.c: beats a lesson keeps
QMASK = [0x891, 0x491, 0x489, 0x449, 0x249, 0x4A1, 0x111]
CHANGE, TWO, ODD = 1, 2, 4
f32 = np.float32


# ---------------------------------------------------------------- the network, as trio_w.h has it
def read_weights():
    text = WEIGHTS.read_text()
    defs = {m.group(1): int(m.group(2)) for m in re.finditer(r"#define (BASS_\w+) (\d+)", text)}

    def arr(name, typ):
        m = re.search(r"static const %s %s\[(\d+)\] = \{(.*?)\};" % (typ, name), text, re.S)
        vals = [v.strip().rstrip("f") for v in m.group(2).replace("\n", " ").split(",") if v.strip()]
        out = np.array([float(v) for v in vals], dtype=f32 if typ == "float" else np.int64)
        assert len(out) == int(m.group(1))
        return out

    return defs, arr("BASS_Q", "int8_t"), arr("BASS_F", "float"), arr("BASS_S", "float")


class Bass:
    """BASS as the FM-1 runs it: 114 -> 64 -> GRU 96 -> 35 + 1, each matrix int8 times its scale"""

    def __init__(self):
        d, q, fl, s = read_weights()
        self.nin = d["BASS_NIN"]
        self.off = {k: v for k, v in d.items() if k.startswith("BASS_IN_")}

        def mat(name, rows, cols):
            o = d[name]
            return q[o:o + rows * cols].reshape(rows, cols), s[d[name + "_S"]]

        def vec(name, n):
            return fl[d[name]:d[name] + n].astype(f32)

        self.Wp_q, self.sp = mat("BASS_PROJ_WEIGHT", 64, self.nin)
        self.bp = vec("BASS_PROJ_BIAS", 64)
        self.Wi_q, self.si = mat("BASS_GRU_WEIGHT_IH_L0", 3 * H, 64)
        self.Wh_q, self.sh = mat("BASS_GRU_WEIGHT_HH_L0", 3 * H, H)
        self.bi, self.bh = vec("BASS_GRU_BIAS_IH_L0", 3 * H), vec("BASS_GRU_BIAS_HH_L0", 3 * H)
        W0q, s0 = mat("BASS_HEADS_0_WEIGHT", NTOK, H)
        W1q, s1 = mat("BASS_HEADS_1_WEIGHT", 1, H)
        # the trained rows as the C computes them: s * (float) q, a float32 product each
        self.W0 = np.concatenate([W0q.astype(f32) * f32(s0), W1q.astype(f32) * f32(s1)]).astype(f32)
        self.b0 = np.concatenate([vec("BASS_HEADS_0_BIAS", NTOK), vec("BASS_HEADS_1_BIAS", 1)]).astype(f32)
        self.gru = torch.nn.GRU(64, H, batch_first=True)
        self.proj = torch.nn.Linear(self.nin, 64)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor(self.Wp_q * float(self.sp), dtype=torch.float32))
            self.proj.bias.copy_(torch.tensor(self.bp))
            self.gru.weight_ih_l0.copy_(torch.tensor(self.Wi_q * float(self.si), dtype=torch.float32))
            self.gru.weight_hh_l0.copy_(torch.tensor(self.Wh_q * float(self.sh), dtype=torch.float32))
            self.gru.bias_ih_l0.copy_(torch.tensor(self.bi))
            self.gru.bias_hh_l0.copy_(torch.tensor(self.bh))

    def x(self, b):
        """a beat's inputs (the fields decide_bass fills) as the dense vector trio_nn.c bass_sparse makes"""
        x = np.zeros(self.nin, f32)
        o = self.off
        x[o["BASS_IN_TOK"] + b["ptok"]] = 1
        if b["root"] >= 0:
            x[o["BASS_IN_ROOT"] + b["root"]] = 1
            for k in range(12):
                if b["mask"] >> k & 1:
                    x[o["BASS_IN_MASK"] + k] = 1
            x[o["BASS_IN_QUAL"] + b["qual"]] = 1
        else:
            x[o["BASS_IN_NOCH"]] = 1
        if b["nroot"] >= 0:
            x[o["BASS_IN_NROOT"] + b["nroot"]] = 1
            for k in range(12):
                if b["nmask"] >> k & 1:
                    x[o["BASS_IN_NMASK"] + k] = 1
        x[o["BASS_IN_TOCHG"] + min(max(b["tochg"], 1), 8) - 1] = 1
        x[o["BASS_IN_SINCE"] + min(b["since"], 7)] = 1
        x[o["BASS_IN_BIB"] + (b["bib"] & 3)] = 1
        if b["feel"]:
            x[o["BASS_IN_FEEL"]] = 1
        if b["pskip"]:
            x[o["BASS_IN_PSKIP"]] = 1
        return x

    def states(self, h0, X):
        """the GRU through the inputs X [n, nin] from h0: each step's state [n, 96]"""
        with torch.no_grad():
            y, _ = self.gru(self.proj(torch.tensor(X)[None]), torch.tensor(h0)[None, None])
        return y[0].numpy().astype(f32)


def q15(h):
    """a state as the C keeps it: x 32768, rounded half away from zero, clamped; back to float"""
    v = (h * f32(32768.0)).astype(f32)
    r = np.where(v < 0, np.trunc(v - f32(0.5)), np.trunc(v + f32(0.5)))
    return (np.clip(r, -32768, 32767).astype(f32) * f32(1.0 / 32768.0)).astype(f32)


# ---------------------------------------------------------------- a lesson
def load_lesson(path):
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    h0 = np.array([float(v) for v in lines[0][1:]], f32)
    n = int(lines[1][1])
    keys = ("beat", "ptok", "root", "qual", "mask", "nroot", "nmask", "tochg", "since", "bib", "feel", "pskip",
            "tok", "skip")
    beats = [dict(zip(keys, (int(v) for v in ln))) for ln in lines[2:2 + n]]
    for b in beats:
        b["flags"] = (CHANGE if b["since"] == 0 else 0) | (TWO if b["feel"] else 0) | (ODD if b["bib"] & 1 else 0)
    return h0, beats


def allow_of(flags):
    hold = (flags & TWO) and (flags & ODD) and not (flags & CHANGE)
    a = np.zeros(NTOK, bool)
    if hold:
        a[HOLD] = True
    else:
        a[2:2 + LINE_HI - LO + 1] = True
    return a


def shape(z, b):
    """bass_shape's nudges, in place, float32 in the C's order"""
    for k in range(2, NTOK):
        p = LO + k - 2
        if (b["flags"] & CHANGE) and (b["mask"] >> (p % 12) & 1):
            z[k] = f32(z[k] + f32(2.5 if p % 12 == b["root"] else 1.5))
        if p > 45:
            z[k] = f32(z[k] - f32(f32(0.06) * f32(p - 45)))
    return z


def learnable(b):
    if b["tok"] < 0 or b["tok"] >= NTOK:
        return 0
    two = b["flags"] & TWO
    return (1 if allow_of(b["flags"])[b["tok"]] else 0) | (2 if ((b["flags"] & ODD) if two else b["tok"] >= 2) else 0)


def prepare(net, h0, beats):
    """the states through the lesson (teacher-forced), and the lesson as the C's ring keeps it: the last TEACH_MAX
    beats that teach something, their states Q15"""
    X = np.stack([net.x(b) for b in beats])
    hs = net.states(h0, X)
    kept = [i for i, b in enumerate(beats) if learnable(b)][-TEACH_MAX:]
    return hs, kept


class Head:
    def __init__(self, net):
        self.W0, self.b0 = net.W0, net.b0
        self.W, self.b = net.W0.copy(), net.b0.copy()


def sigmoid(x):
    return f32(1.0) / (f32(1.0) + np.exp(-x, dtype=f32))


def learn(head, hq, beats, kept, lr, l2, passes):
    """the passes over the kept beats; each pass's mean losses (the tokens', the pickups'), before its steps' updates"""
    lr, l2, out = f32(lr), f32(l2), []
    for _ in range(passes):
        tot, cnt = [0.0, 0.0], [0, 0]
        for i in kept:
            b, h = beats[i], hq[i]
            what = learnable(b)
            if what & 1:
                a = np.nonzero(allow_of(b["flags"]))[0]
                z = np.zeros(NTOK, f32)
                z[a] = (head.W[a] @ h + head.b[a]).astype(f32)
                shape(z, b)
                za = z[a]
                m = za.max()
                e = np.exp(za - m).astype(f32)
                S = f32(e.sum())
                y = int(np.nonzero(a == b["tok"])[0][0])
                tot[0] += float(-(za[y] - m - np.log(S)))
                cnt[0] += 1
                g = (e / S).astype(f32)
                g[y] -= f32(1.0)
                head.W[a] -= lr * (np.outer(g, h).astype(f32) + l2 * (head.W[a] - head.W0[a]))
                head.b[a] -= lr * (g + l2 * (head.b[a] - head.b0[a]))
            if what & 2:
                r = NTOK
                s = f32(head.W[r] @ h + head.b[r])
                tot[1] += float(max(s, 0.0) + np.log1p(np.exp(-abs(s))) - b["skip"] * s)
                cnt[1] += 1
                g = f32(f32(0.3) * (sigmoid(s) - f32(b["skip"])))
                head.W[r] -= lr * (g * h + l2 * (head.W[r] - head.W0[r]))
                head.b[r] -= lr * (g + l2 * (head.b[r] - head.b0[r]))
        out.append((tot[0] / max(cnt[0], 1), tot[1] / max(cnt[1], 1)))
    return out


def lesson_nll(head, hq, beats, kept):
    """the kept beats' token cross-entropy (as the engine samples: allowed tokens, nudges), and how often the taught
    token is the most likely"""
    nll, hit, n = 0.0, 0, 0
    for i in kept:
        b = beats[i]
        if not learnable(b) & 1:
            continue
        a = allow_of(b["flags"])
        z = np.full(NTOK, -1e30, f32)
        z[a] = shape((head.W[:NTOK] @ hq[i] + head.b[:NTOK]).astype(f32), b)[a]
        m = z.max()
        nll += float(np.log(np.exp(z - m).sum()) + m - z[b["tok"]])
        hit += int(np.argmax(z) == b["tok"])
        n += 1
    return nll / max(n, 1), hit / max(n, 1)


def free_run(net, head, h_start, beats, runs=200, temp=0.9, seed=1):
    """the bass on its own over the lesson's last 128 beats' chords (two passes of them), from the state at the
    lesson's end: as decide_bass samples (FREEDOM 50: temperature 0.9; DRUMS 50: pickups at the sigmoid's own rate),
    its own notes and pickups fed back. How often it plays the taught note, on the second pass"""
    g = torch.Generator().manual_seed(seed)
    seg = beats[-TEACH_MAX:]
    seq = seg + seg
    n = len(seq)
    h = torch.tensor(h_start)[None].repeat(runs, 1)
    ptok = torch.full((runs,), seq[0]["ptok"], dtype=torch.long)
    pskip = torch.zeros(runs)
    W, b = torch.tensor(head.W), torch.tensor(head.b)
    same = cnt = 0
    with torch.no_grad():
        for t, bt in enumerate(seq):
            base = net.x(bt)
            X = torch.tensor(base)[None].repeat(runs, 1)
            o = net.off
            X[:, o["BASS_IN_TOK"]:o["BASS_IN_TOK"] + NTOK + 1] = 0
            X[torch.arange(runs), o["BASS_IN_TOK"] + ptok] = 1
            X[:, o["BASS_IN_PSKIP"]] = pskip
            y, hn = net.gru(net.proj(X)[:, None], h[None])
            h = hn[0]
            z = h @ W[:NTOK].T + b[:NTOK]
            nud = torch.tensor(shape(np.zeros(NTOK, f32), bt))
            a = torch.tensor(allow_of(bt["flags"]))
            lg = torch.where(a, (z + nud) / temp, torch.tensor(-1e30))
            tok = torch.multinomial(torch.softmax(lg, -1), 1, generator=g)[:, 0]
            sk = h @ W[NTOK] + b[NTOK]
            can = (tok >= 2) if not (bt["flags"] & TWO) else torch.full((runs,), bool(bt["flags"] & ODD))
            pskip = (can & (torch.rand(runs, generator=g) < torch.sigmoid(sk))).float()
            ptok = tok
            if t >= n // 2 and bt["tok"] >= 2:
                same += int((tok == bt["tok"]).sum())
                cnt += runs
    return same / max(cnt, 1)


_FB = None


def filobass_states(net):
    """FiloBass's 48 lines through the frozen network (trio_train.py's features; teacher-forced from zero), once"""
    global _FB
    if _FB is None:
        sys.path.insert(0, str(HERE / "tools"))
        import trio_train as T
        seqs = [s for s in T.load_seqs(str(Path.home() / "fm1" / "data" / "trio" / "bass.npz"))
                if not s["src"].startswith("synth")]
        hs, ys = [], []
        for s in seqs:
            X, yt, _ = T.bass_features(s["tok"], s["skip"], s["root"], s["qual"], s["mask"], s["bib"], s["feel"])
            hs.append(net.states(np.zeros(H, f32), X.astype(f32)))
            ys.append(yt)
        _FB = (np.concatenate(hs), np.concatenate(ys))
    return _FB


def filobass_nll(head, net):
    h, y = filobass_states(net)
    z = h @ head.W[:NTOK].T + head.b[:NTOK]
    m = z.max(1, keepdims=True)
    return float((np.log(np.exp(z - m).sum(1)) + m[:, 0] - z[np.arange(len(y)), y]).mean())


def report(net, path, settings):
    h0, beats = load_lesson(path)
    hs, kept = prepare(net, h0, beats)
    hq = q15(hs)
    base = Head(net)
    tn, ta = lesson_nll(base, hq, beats, kept)
    fb0 = filobass_nll(base, net)
    fr0 = free_run(net, base, hs[-1], beats)
    print(f"{Path(path).name}: {len(beats)} beats taught, {len(kept)} kept; trained: lesson {tn:.3f} nats "
          f"(most likely {100 * ta:.0f}%), on its own {100 * fr0:.1f}% the taught note; FiloBass {fb0:.4f} nats")
    print(f"{'lr':>5s} {'pass':>4s} {'l2':>5s} | {'lesson':>6s} {'top':>4s} | {'own':>5s} | {'FiloBass':>8s} {'change':>7s}")
    for lr, passes, l2 in settings:
        hd = Head(net)
        learn(hd, hq, beats, kept, lr, l2, passes)
        n1, a1 = lesson_nll(hd, hq, beats, kept)
        fr = free_run(net, hd, hs[-1], beats)
        fb = filobass_nll(hd, net)
        print(f"{lr:5.2f} {passes:4d} {l2:5.2f} | {n1:6.3f} {100 * a1:3.0f}% | {100 * fr:4.1f}% | {fb:8.4f} "
              f"{fb - fb0:+7.4f}", flush=True)


def c_array(name, ctype, vals, per=8):
    out = [f"static const {ctype} {name}[{len(vals)}] = {{"]
    for i in range(0, len(vals), per):
        out.append("    " + ", ".join(vals[i:i + per]) + ",")
    out.append("};")
    return "\n".join(out)


def ff(v):
    return "%.9gf" % float(f32(v))


def golden(path, lr=0.03, passes=15, l2=0.1):
    """the C's check: a lesson's beats (their inputs and what was played), the frozen states, each pass's losses, the
    learned rows' logits on every kept beat at the end"""
    net = Bass()
    h0, beats = load_lesson(path)
    hs, kept = prepare(net, h0, beats)
    hq = q15(hs)
    hd = Head(net)
    losses = learn(hd, hq, beats, kept, lr, l2, passes)
    lo = (hq @ hd.W.T + hd.b).astype(f32)                  # [n, 36]
    keys = ("ptok", "root", "qual", "mask", "nroot", "nmask", "tochg", "since", "bib", "feel", "pskip", "tok", "skip")
    out = ["/* SPDX-License-Identifier: MPL-2.0 */",
           "/* Golden vectors for tests/host/ft_test.c (teaching the bass): made by tools/trio_finetune.py golden, do",
           " * not edit. The lesson (tests/host/teach_test.c blues 160 1 --teach 1 --dump): each beat's inputs as",
           " * decide_bass built them and what the player played (ptok root qual mask nroot nmask tochg since bib feel",
           " * pskip tok skip); the state it started from; the frozen network's state after each beat; each pass's",
           " * mean losses (the tokens', the pickups'); the learned rows' logits (35 tokens, the pickup) on every beat",
           " * after the passes. */",
           "#pragma once",
           f"#define FTG_N {len(beats)}",
           f"#define FTG_LR {lr}f",
           f"#define FTG_L2 {l2}f",
           f"#define FTG_PASSES {passes}",
           c_array("FTG_IN", "int16_t", [str(b[k]) for b in beats for k in keys], per=13),
           c_array("FTG_H0", "float", [ff(v) for v in h0]),
           c_array("FTG_H", "float", [ff(v) for v in hs.flatten()]),
           c_array("FTG_LOSS", "float", [ff(v) for p in losses for v in p]),
           c_array("FTG_LOGITS", "float", [ff(v) for v in lo.flatten()])]
    GOLDEN.write_text("\n".join(out) + "\n")
    print(f"golden: {GOLDEN.relative_to(HERE)} ({len(beats)} beats, {len(kept)} kept; lr {lr}, {passes} passes, "
          f"l2 {l2}): token loss {losses[0][0]:.4f} -> {losses[-1][0]:.4f}, pickup {losses[0][1]:.4f} -> "
          f"{losses[-1][1]:.4f}")


if __name__ == "__main__":
    torch.set_num_threads(8)
    if len(sys.argv) > 2 and sys.argv[1] == "golden":
        golden(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] in ("sweep", "check"):
        net = Bass()
        grid = [(0.03, 15, 0.1)] if sys.argv[1] == "check" else \
            [(lr, p, l2) for lr in (0.01, 0.03, 0.1) for p in (5, 10, 15, 30, 60) for l2 in (0.0, 0.1, 0.3)]
        for p in sys.argv[2:]:
            report(net, p, grid)
    else:
        sys.exit(__doc__)
