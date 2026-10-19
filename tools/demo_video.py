#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Demo videos of TRIO's networks at work (Chinese captions). The host simulator plays a scripted session, and each
frame shows the FM-1's screen beside what the networks decide: the chord the CHORD network heard, the 35 notes'
chances the BASS network gives each beat (and the note drawn), the DRUM network's steps.

  PYTHON=~/fm1/venv/bin/python ./build.sh --host
  ~/fm1/venv/bin/python tools/demo_video.py teach     app/build/demo/trio-demo.mp4
  ~/fm1/venv/bin/python tools/demo_video.py autumn    app/build/demo/trio-autumn-leaves.mp4
  ~/fm1/venv/bin/python tools/demo_video.py MODE b12  a still of bar 12 (app/build/demo/video-MODE/still-b12.png)

teach: Autumn Leaves' A section at 120 in four; the chords twelve bars (the form learned), the band alone eight, a
walking line taught sixteen with EDIT (R 5 8 and the half step above the next root, a pickup into the changes after
a bar or more; as tests/host/teach_test.c), then the band alone with the bass as it learned: eight bars at FREEDOM 50,
eight at FREEDOM 0.
autumn: the whole of Autumn Leaves' changes (32 bars, A A B C, tests/host/charts.h) three times at 140, the chords
alone (rooted voicings, an eighth early, as the manual asks); in AUTO feel; the third time with FREEDOM and DRUMS at
80. Below the panels, the bass line as played, a bar a cell; at the end each chorus's numbers. The melody is not
played: only the chords, which is what TRIO listens to.

The simulator runs the firmware's own code (app/dsp/trio.c and the rest); TRIO_VIZ=FILE makes the engine log what
it decided (trio.c viz_*, VIZ). Needs Pillow and ffmpeg (libx264, aac).
"""
import bisect
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "app" / "build" / "demo"
HOST = HERE / "app" / "build" / "host" / "trio_host"
FPS = 20
SHOT_MS = 50
W, H = 1280, 720
LEAD = 150.0
SR = 44.1                                            # samples a ms
BEAT = 500.0                                         # (set by the session: 60000 / BPM)

Q_MAJ, Q_DOM, Q_MIN, Q_HDIM, Q_DIM, Q_SUS, Q_AUG = range(7)
QMASK = [0x891, 0x491, 0x489, 0x449, 0x249, 0x4A1, 0x111]
NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
QN = ["maj7", "7", "m7", "m7b5", "dim7", "7sus", "+"]
# the A section of Autumn Leaves (tools/make_scenarios.py): name, root lowest voicing, root, kind
A = [("Cm7", [48, 51, 55, 58], 0, Q_MIN), ("F7", [53, 57, 60, 63], 5, Q_DOM),
     ("Bbmaj7", [46, 50, 53, 57], 10, Q_MAJ), ("Ebmaj7", [51, 55, 58, 62], 3, Q_MAJ),
     ("Am7b5", [45, 48, 51, 55], 9, Q_HDIM), ("D7", [50, 54, 57, 60], 2, Q_DOM),
     ("Gm7", [43, 46, 50, 53], 7, Q_MIN), ("Gm7", [43, 46, 50, 53], 7, Q_MIN)]
# the whole tune, as tests/host/charts.h has it
AUTUMN = ("Cm7|F7|Bbmaj7|Ebmaj7|Am7b5|D7|Gm7|Gm7|Cm7|F7|Bbmaj7|Ebmaj7|Am7b5|D7|Gm7|Gm7|"
          "Am7b5|D7|Gm7|Gm7|Cm7|F7|Bbmaj7|Ebmaj7|Am7b5|D7|Gm7 C7|Fm7 Bb7|Ebmaj7|Am7b5 D7|Gm7|Gm7")
# the teach session's bars: chords, the band alone, taught, after (twice)
CH_BARS, ALONE, TEACH, AFTER = 12, 8, 16, 8
T_BAR0 = CH_BARS + ALONE
A_BAR0 = T_BAR0 + TEACH
END_BAR = A_BAR0 + 2 * AFTER
CHORUSES = 3                                        # the autumn session's


def chord_name(r, q):
    return NAMES[r % 12] + QN[q] if r >= 0 and 0 <= q < 7 else "-"


def note_name(n):
    return f"{NAMES[n % 12]}{n // 12 - 1}"


def parse_chart(chart):
    """a chart (bars '|', chords in a bar ' ') -> a (root, kind) a beat, 4/4"""
    letters = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    beats = []
    for bar in chart.split("|"):
        chords = []
        for c in bar.split():
            r, rest = letters[c[0]], c[1:]
            if rest[:1] in ("b", "#"):
                r += -1 if rest[0] == "b" else 1
                rest = rest[1:]
            q = (Q_MAJ if rest.startswith("maj7") or rest == "" else Q_HDIM if rest.startswith("m7b5") else
                 Q_MIN if rest.startswith("m") else Q_DIM if rest.startswith("dim") else Q_DOM)
            chords.append((r % 12, q))
        for k in range(4):
            beats.append(chords[k * len(chords) // 4])
    return beats


def voicing(root, qual):
    """a jazz pianist's rooted voicing (tests/host/charts.h): the root from E2 to D#3, then 3 5 7 9 above it"""
    RL = [[4, 7, 11, 14], [4, 9, 10, 14], [3, 7, 10, 14], [3, 6, 10, 12], [0, 3, 6, 9], [5, 7, 10, 14], [4, 8, 12, 16]]
    base = 55 + ((root - 55 % 12) + 24) % 12
    if base > 62:
        base -= 12
    return [40 + ((root - 4) + 24) % 12] + [base + i for i in RL[qual]]


def line(chart):
    """R 5 8 and the half step above the next root on a chord's last beat; a pickup into each change after a bar or
    more (as tests/host/teach_test.c and tools/make_scenarios.py)"""
    n_b, out, b = len(chart), [], 0
    while b < n_b:
        e = b
        while e < n_b and chart[e] == chart[b]:
            e += 1
        n = e - b
        r, nr = 31 + (chart[b][0] - 7) % 12, 31 + (chart[e % n_b][0] - 7) % 12
        fifth = 6 if chart[b][1] == Q_HDIM else 7
        for i in range(n):
            note = nr + 1 if n > 1 and i == n - 1 else r + (fifth if i % 2 else 12 if i % 4 == 2 else 0)
            out.append((note, n >= 4 and i == n - 1))
        b = e
    return out


# ---------------------------------------------------------------- the sessions, for the simulator
class Script:
    def __init__(self):
        self.t = 0.0
        self.lines = []
        self.next_shot = 0.0
        self.events = []                            # what the player did, when: ("midi", (st, d1, d2)), ("ctl", label)

    def say(self, s):
        self.lines.append(s)

    def _shots_until(self, t):
        while self.next_shot <= t:
            ms = int(round(self.next_shot - self.t))
            if ms > 0:
                self.say(f"wait {ms}")
                self.t += ms
            self.say(f"shot s{int(round(self.next_shot)):07d}.png")
            self.next_shot += SHOT_MS

    def wait_until(self, t):
        self._shots_until(t)
        ms = int(round(t - self.t))
        if ms > 0:
            self.say(f"wait {ms}")
            self.t += ms

    def cmd(self, s, ms, label=None):
        if label:                                   # a button or a knob: what it did, for the video
            self.events.append((self.t, "ctl", label))
        self.say(s)
        self.t += ms

    def midi(self, *b):
        self.events.append((self.t, "midi", b))
        self.cmd("midi " + " ".join(f"{x:02X}" for x in b), 2)


def start(s, bpm, feel_taps):
    """boot, the tempo, the feel; PLAY at 5 s (an intro card before it). The beat clock from PLAY's press"""
    global BEAT
    BEAT = 60000.0 / bpm
    s.cmd("wav demo.wav", 0)
    s.wait_until(600)
    if bpm != 140:
        s.cmd(f"turn K1 {bpm - 140} 10", 10 * abs(bpm - 140), f"转 K1：TEMPO {bpm}")
    for i in range(feel_taps):
        s.cmd("tap ARP", 80, "按 ARP：FEEL " + ("TWO", "FOUR", "AUTO")[i % 3])
    s.wait_until(5000)
    t_play = s.t
    s.cmd("tap PLAY", 80, "按 PLAY：开始")
    return t_play, lambda beat: t_play + 4.0 + (4 + beat) * BEAT


def play_chords(s, at, chart_beats, b0, b1, held):
    """the pianist: each change's voicing LEAD ms before its beat, the chord before let go just before"""
    for b in range(b0, b1):
        c = chart_beats[b % len(chart_beats)]
        if b > 0 and c == chart_beats[(b - 1) % len(chart_beats)] and b != b0:
            continue
        notes = voicing(*c) if len(c) == 2 else c[2]
        on = at(b) - LEAD
        s.wait_until(on - 2 * len(held) - 4)
        for n in held:
            s.midi(0x80, n, 0)
        s.wait_until(on)
        for i, n in enumerate(notes):
            s.midi(0x90, n, 72 + 6 * i)
        held = notes
    return held


def session_teach():
    s = Script()
    t_play, at = start(s, 120, 2)                    # FOUR
    chart = [(c[2], c[3]) for c in A for _ in range(4)]
    taught = line(chart)
    jit = [((i * 7919) % 41) - 20 for i in range(512)]
    held = play_chords(s, at, [(c[2], c[3], c[1]) for c in A for _ in range(4)], 0, 4 * CH_BARS, [])
    s.wait_until(at(4 * CH_BARS) - LEAD)
    for n in held:
        s.midi(0x80, n, 0)
    s.wait_until(at(4 * T_BAR0 - 1) - 40)            # a beat before: EDIT (it acts as it is let go)
    s.cmd("tap EDIT", 80, "按 EDIT：开始教贝斯")
    prev = None
    for b in range(4 * T_BAR0, 4 * A_BAR0):
        note, ghost = taught[b % 32]
        s.wait_until(at(b) + jit[b % 512])
        s.midi(0x90, note, 84)
        if prev is not None:
            s.midi(0x80, prev, 0)
        prev = note
        if ghost:
            s.wait_until(at(b + 0.64) - 4)
            s.midi(0x80, note, 0)
            s.midi(0x90, note, 36)
    s.wait_until(at(4 * A_BAR0) - 0.1 * BEAT - 40)   # just after the last beat: EDIT, it learns
    s.cmd("tap EDIT", 80, "按 EDIT：开始学")
    s.midi(0x80, prev, 0)
    s.wait_until(at(4 * (A_BAR0 + AFTER)) - 300)     # FREEDOM to 0 for the last eight bars
    s.cmd("turn K2 -25 10", 250, "转 K2：FREEDOM 0")
    s.wait_until(at(4 * END_BAR) + 100)
    s.cmd("tap PLAY", 80, "按 PLAY：停")
    s.wait_until(at(4 * END_BAR) + 2500)
    s.cmd("wavstop", 0)
    return s, t_play, at, at(4 * END_BAR) + 600


def session_autumn():
    s = Script()
    t_play, at = start(s, 140, 0)                    # AUTO: in two until the form comes round (or 16 bars)
    beats = parse_chart(AUTUMN)
    n = len(beats)
    held = play_chords(s, at, beats, 0, 2 * n, [])
    s.wait_until(at(2 * n) - LEAD - 700)             # the third time: FREEDOM and DRUMS up
    s.cmd("turn K2 15 10", 150, "转 K2：FREEDOM 80")
    s.cmd("turn K3 15 10", 150, "转 K3：DRUMS 80")
    held = play_chords(s, at, beats, 2 * n, CHORUSES * n, held)
    s.wait_until(at(CHORUSES * n) - 60)              # the end: the band stops after the last bar, Gm7 rings on
    s.cmd("tap PLAY", 80, "按 PLAY：停")
    s.wait_until(at(CHORUSES * n) + 1500)
    for k in held:
        s.midi(0x80, k, 0)
    s.wait_until(at(CHORUSES * n) + 3500)
    s.cmd("wavstop", 0)
    return s, t_play, at, at(CHORUSES * n) + 2600


# ---------------------------------------------------------------- what the engine logged
def load_log(path):
    log = {"B": {}, "T": {}, "D": [], "H": [], "E": [], "L": [], "F": []}
    for ln in Path(path).read_text().splitlines():
        f = ln.split()
        if not f:
            continue
        k = f[0]
        if k == "B":
            b = int(f[1])
            log["B"][b] = {
                "beat": b, "t": float(f[2]) / SR, "teach": int(f[3]), "root": int(f[4]), "qual": int(f[5]),
                "nroot": int(f[6]), "nqual": int(f[7]), "tochg": int(f[8]), "ptok": int(f[9]), "tok": int(f[10]),
                "skip": int(f[11]), "freedom": int(f[12]), "p": [float(x) for x in f[13:13 + 35]]}
        elif k == "T":
            log["T"][int(f[1])] = (int(f[2]), int(f[3]))
        elif k == "D":
            log["D"].append((int(f[1]), float(f[2]) / SR, int(f[3])))
        elif k == "H":
            log["H"].append((float(f[1]) / SR, int(f[2]), int(f[3]), float(f[4]), [int(x) for x in f[5:]]))
        elif k == "E":
            log["E"].append((int(f[1]), float(f[2]) / SR, int(f[3]) if len(f) > 3 else 0))
        elif k == "L":
            log["L"].append((int(f[1]), float(f[2]) / SR, [float(x) for x in f[4:]] if f[1] == "1" else []))
        elif k == "F":
            log["F"].append((int(f[1]), int(f[2])))
    return log


# ---------------------------------------------------------------- drawing
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=2 if bold else 0)


BG = (14, 17, 21)
PANEL = (26, 31, 38)
LINE = (52, 60, 70)
TXT = (226, 230, 236)
DIM = (138, 146, 158)
ORANGE = (240, 164, 72)
CYAN = (86, 200, 232)
GRAY = (92, 100, 112)
F_TITLE, F_CAP, F_H, F_S, F_XS, F_BIG = font(30, True), font(26, True), font(19, True), font(17), font(13), font(40, True)
F_T, F_SUB = font(10), font(20, True)
PIECES = ["ride", "军鼓", "底鼓", "踩镲", "开镲", "边击", "高通", "低通", "吊镲"]


def text(d, xy, s, f, fill=TXT, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def panel(d, box, title):
    d.rounded_rectangle(box, 10, fill=PANEL)
    text(d, (box[0] + 16, box[1] + 10), title, F_H, DIM)


def draw_card(d, title, rows, foot, x_what=390):
    text(d, (W // 2, 70), title, F_TITLE, TXT, "ma")
    y = 150
    for name, what, when, col in rows:
        d.rounded_rectangle((120, y, W - 120, y + 78), 10, fill=PANEL)
        text(d, (150, y + 14), name, F_BIG if len(name) < 8 else F_H, col)
        text(d, (x_what, y + 14), what, F_H, TXT)
        text(d, (x_what, y + 46), when, F_S, DIM)
        y += 92
    text(d, (W // 2, y + 18), foot, F_S, DIM, "ma")


def draw_frame(now, ctx):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if now < ctx["t_play"]:
        ctx["intro"](d)
        return im
    if now >= ctx["t_end"]:
        ctx["end"](d)
        return im
    part = [p for p in ctx["parts"] if p[0] <= now]
    if part:
        text(d, (32, 10), part[-1][1], F_TITLE)
    where = ctx["where"](now)
    if where:
        text(d, (W - 32, 18), where, F_H, DIM, "ra")
    c = [x for x in ctx["captions"] if x[0] <= now < x[1]]
    if c:
        text(d, (32, 52), c[-1][2], F_SUB, CYAN)
    shot = ctx["shot_at"](now)                     # the FM-1's screen, 1.5 times
    d.rounded_rectangle((24, 84, 408, 468), 16, fill=(40, 40, 44))
    if shot:
        im.paste(shot.resize((360, 360), Image.LANCZOS), (36, 96))
    text(d, (216, 476), "FM-1 的屏幕（模拟器，和 FM-1 跑的是同一份代码）", F_XS, DIM, "ma")
    draw_chord(d, now, ctx)
    draw_bass(d, now, ctx)
    draw_drums(d, now, ctx)
    draw_strip(d, now, ctx)
    draw_keys(d, now, ctx)
    learn = ctx["learning"]
    if learn and learn[0] <= now < learn[0] + 3800:
        draw_learning(d, now - learn[0], learn[1])
    return im


def draw_chord(d, now, ctx):
    log = ctx["log"]
    box = (424, 84, 1256, 144)
    panel(d, box, "CHORD 网络：从你按着的音认出和弦")
    hs = [h for h in log["H"] if h[0] <= now]
    if not hs:
        text(d, (440, 116), "（还没有和弦）", F_S, DIM)
        return
    t, r, q, conf, notes = hs[-1]
    if now - t > 4000 and now >= ctx["chords_until"]:
        text(d, (440, 116), "你没在弹和弦：和弦来自 TRIO 记住的曲式", F_S, DIM)
        return
    fresh = now - t < 700
    text(d, (440, 116), "听到  " + " ".join(note_name(n) for n in notes), F_S, TXT if fresh else DIM)
    text(d, (1080, 104), "→", F_H, DIM)
    text(d, (1112, 92 if fresh else 102), chord_name(r, q), F_BIG if fresh else F_H, ORANGE if fresh else TXT)


def bass_now(now, ctx):
    log = ctx["log"]["B"]
    i = bisect.bisect_right(ctx["b_times"], now + 40) - 1
    return log[ctx["b_keys"][i]] if i >= 0 else None


def draw_bass(d, now, ctx):
    log = ctx["log"]
    box = (424, 152, 1256, 344)
    panel(d, box, "BASS 网络：每拍之前 40 毫秒，给每个音一个可能性")
    e = bass_now(now, ctx)
    if not e:
        return
    b = e["beat"]
    taught = ctx["taught_at"](b)
    nxt = chord_name(e["nroot"], e["nqual"]) if e["nroot"] >= 0 and e["nqual"] >= 0 else "?"
    inp = f"和弦 {chord_name(e['root'], e['qual'])}   下一个 {nxt}（{e['tochg']} 拍后）   第 {b % 4 + 1} 拍   上一个音 " + \
          (note_name(28 + e["ptok"] - 2) if 2 <= e["ptok"] < 35 else "-")
    text(d, (440, 184), inp, F_S, TXT)
    x0, y0, bw, hmax = 444, 326, 26, 88
    d.line((x0, y0 - hmax, x0 + 28 * bw - 4, y0 - hmax), fill=LINE, width=1)
    d.line((x0, y0 - hmax // 2, x0 + 28 * bw - 4, y0 - hmax // 2), fill=LINE, width=1)
    text(d, (x0 + 28 * bw + 4, y0 - hmax - 8), "100%", F_XS, DIM)
    text(d, (x0 + 28 * bw + 4, y0 - hmax // 2 - 8), "50%", F_XS, DIM)
    hold = e["p"][1] > 0.5                          # in two, beats 2 and 4: a hold, no note to choose
    for k in range(2, 30):                          # E1 .. G3
        x = x0 + (k - 2) * bw
        p = e["p"][k]
        hgt = max(int(hmax * p), 1)
        player = e["teach"] and log["T"].get(b, (-1, 0))[0] == k
        col = GRAY if e["teach"] else (ORANGE if e["tok"] == k else (88, 108, 128))
        d.rectangle((x, y0 - hgt, x + bw - 5, y0), fill=col)
        if player:
            d.rectangle((x - 1, y0 - hmax - 4, x + bw - 4, y0), outline=CYAN, width=3)
        elif taught == k:
            d.rectangle((x - 1, y0 - hmax - 4, x + bw - 4, y0), outline=CYAN, width=2)
        n = 28 + k - 2
        if n % 12 in (0, 4, 7):
            text(d, (x + (bw - 5) // 2, y0 + 3), note_name(n), F_XS, DIM, "ma")
    if e["teach"]:
        pt = log["T"].get(b, (-1, 0))[0]
        msg = "你弹的：" + (note_name(28 + pt - 2) if pt >= 2 else "（延音）") + "      灰色：网络自己会怎么弹"
        text(d, (440, 208), msg, F_H, CYAN)
    elif hold:
        text(d, (440, 208), "two-feel 的第 2、4 拍：上一个音延长，这一拍不弹新音", F_H, ORANGE)
    else:
        played = note_name(28 + e["tok"] - 2) if e["tok"] >= 2 else "延音" if e["tok"] == 1 else "休止"
        p = e["p"][e["tok"]] if 0 <= e["tok"] < 35 else 0.0
        msg = f"抽到：{played}（可能性 {round(100 * p)}%）    FREEDOM {e['freedom']}"
        if e["skip"]:
            msg += "    + 拍后加花"
        text(d, (440, 208), msg, F_H, ORANGE)
        if taught >= 2:
            text(d, (x0 + 28 * bw - 8, y0 - hmax + 4), "蓝框：你在这个位置教的音", F_XS, CYAN, "ra")
    score = ctx["score_at"](now)
    if score:
        text(d, (1240, 212), score, F_S, CYAN, "ra")


def draw_drums(d, now, ctx):
    box = (424, 352, 1256, 496)
    panel(d, box, "DRUM 网络：提前半小节，决定每个三连音步打哪几件鼓（竖线隔开四拍）")
    ds = ctx["drums"]
    cur = ctx["step_at"](now)
    if cur is None:
        return
    bar = cur // 12
    x0, y0, cw, ch = 500, 384, 62, 12
    for i, nm in enumerate(PIECES):
        text(d, (492, y0 + i * ch + 5), nm, F_T, DIM, "rm")
    for k in (3, 6, 9):                             # the beats
        d.line((x0 + k * cw - 7, y0 - 2, x0 + k * cw - 7, y0 + 9 * ch), fill=DIM, width=1)
    for k in range(12):
        g = bar * 12 + k
        x = x0 + k * cw
        if g == cur:
            d.rectangle((x - 3, y0 - 3, x + cw - 7, y0 + 9 * ch), outline=TXT, width=1)
        hits = ds.get(g)
        for i in range(9):
            on = hits is not None and hits[1] >> i & 1
            ahead = g > cur
            if on and (g <= cur or g <= cur + 6):
                col = ORANGE if not ahead else (150, 110, 60)
            else:
                col = LINE
            d.rectangle((x, y0 + i * ch, x + cw - 12, y0 + i * ch + ch - 3), fill=col)


def draw_strip(d, now, ctx):
    """the bass line as played: the last bars, a cell each, the chord over its notes (gray: not the chord's);
    teaching: the player's line, in blue"""
    log = ctx["log"]
    e = bass_now(now, ctx)
    d.rounded_rectangle((24, 504, 1256, 556), 10, fill=PANEL)
    text(d, (36, 520), "贝斯线", F_S, DIM)
    if not e:
        return
    cur = e["beat"]
    bar = cur // 4
    x_l, cw = 104, (1256 - 8 - 104) / 5
    for i, bb in enumerate(range(bar - 4, bar + 1)):
        if bb < 0:
            continue
        x = x_l + i * cw
        chords = []
        for k in range(4):
            b = 4 * bb + k
            x_n = x + 6 + k * (cw - 12) / 4
            be = log["B"].get(b)
            if not be or b > cur:
                continue
            c = (be["root"], be["qual"])
            if not chords or c != chords[-1][1]:
                chords.append((x_n, c))
            if be["teach"]:                         # the player's note, once it was played
                t = log["T"].get(b, (-1, 0))[0]
                col_in, col_out = CYAN, (60, 140, 165)
            else:
                t = be["tok"]
                col_in, col_out = TXT, DIM
            if t >= 2:
                n = 28 + t - 2
                inside = QMASK[be["qual"]] >> ((n - be["root"]) % 12) & 1
                col = ORANGE if b == cur else col_in if inside else col_out
                skip = log["T"].get(b, (0, 0))[1] if be["teach"] else be["skip"]
                text(d, (x_n, 532), note_name(n) + ("+" if skip else ""), F_S, col)
            elif t == 1:
                text(d, (x_n + 6, 532), "—", F_S, ORANGE if b == cur else DIM)
        for x_c, c in chords:
            text(d, (x_c, 508), chord_name(*c), F_H, CYAN)
        d.line((x + cw - 3, 510, x + cw - 3, 550), fill=LINE, width=1)


# the keyboard: E1 .. C6, the white keys' positions
KEY_LO, KEY_HI = 28, 84
WHITE = [n for n in range(KEY_LO, KEY_HI + 1) if n % 12 in (0, 2, 4, 5, 7, 9, 11)]
KW = 35
KX0 = 24 + (1232 - KW * len(WHITE)) // 2
KY0, KWH, KBH = 594, 112, 68
GREEN = (110, 214, 140)


def key_box(n):
    """a key's rectangle and whether it is black"""
    if n % 12 in (0, 2, 4, 5, 7, 9, 11):
        i = WHITE.index(n)
        return (KX0 + i * KW, KY0, KX0 + (i + 1) * KW - 2, KY0 + KWH), False
    i = WHITE.index(n - 1)
    x = KX0 + (i + 1) * KW - 11
    return (x, KY0, x + 20, KY0 + KBH), True


class Keys:
    """the keys held at a time, from the session's MIDI (replayed in order; from the start again for a still)"""

    def __init__(self, events, teach_spans):
        self.ev = [(t, b) for t, k, b in events if k == "midi"]
        self.ctl = [(t, b) for t, k, b in events if k == "ctl"]
        self.spans = teach_spans
        self.i, self.now, self.held = 0, -1.0, {}

    def at(self, now):
        if now < self.now:
            self.i, self.held = 0, {}
        while self.i < len(self.ev) and self.ev[self.i][0] <= now:
            t, (st, n, v) = self.ev[self.i]
            if st & 0xF0 == 0x90 and v:
                line = any(a <= t < z for a, z in self.spans) and n <= 55
                self.held[n] = (v, line)
            else:
                self.held.pop(n, None)
            self.i += 1
        self.now = now
        return self.held

    def control(self, now):
        c = [lab for t, lab in self.ctl if t <= now < t + 1800]
        return c[-1] if c else ""


def sounding_bass(now, ctx):
    """the note the network's bass sounds now (None: none; teaching: the player is the bass)"""
    log = ctx["log"]["B"]
    i = bisect.bisect_right(ctx["b_times"], now) - 1
    while i >= 0:
        e = log[ctx["b_keys"][i]]
        if e["teach"] or now > ctx["t_stop"]:
            return None
        if e["tok"] >= 2:
            return 28 + e["tok"] - 2
        if e["tok"] == 0:
            return None
        i -= 1
    return None


def draw_keys(d, now, ctx):
    d.rounded_rectangle((24, 564, 1256, 712), 10, fill=PANEL)
    text(d, (40, 570), "琴键：你按的", F_H, DIM)
    x = 40 + text_w("琴键：你按的", F_H) + 16
    d.rectangle((x, 576, x + 14, 588), fill=ORANGE)
    text(d, (x + 20, 572), "和弦", F_S, TXT)
    x += 20 + text_w("和弦", F_S) + 16
    if ctx["teaching_shown"]:
        d.rectangle((x, 576, x + 14, 588), fill=CYAN)
        text(d, (x + 20, 572), "你弹的贝斯线", F_S, TXT)
        x += 20 + text_w("你弹的贝斯线", F_S) + 16
    d.ellipse((x, 575, x + 13, 588), fill=GREEN)
    text(d, (x + 20, 572), "网络弹的贝斯（不是你按的）", F_S, DIM)
    ctl = ctx["keys"].control(now)
    if ctl:
        w = text_w(ctl, F_H) + 28
        d.rounded_rectangle((1244 - w, 568, 1244, 592), 8, fill=(30, 52, 62), outline=CYAN, width=2)
        text(d, (1244 - w // 2, 580), ctl, F_H, CYAN, "mm")
    held = ctx["keys"].at(now)
    bass = sounding_bass(now, ctx)
    for black in (False, True):
        for n in range(KEY_LO, KEY_HI + 1):
            box, is_black = key_box(n)
            if is_black != black:
                continue
            h = held.get(n)
            if h:
                col = (CYAN if h[0] > 50 else (130, 190, 210)) if h[1] else ORANGE
                if black:
                    col = tuple(int(c * 0.8) for c in col)
            else:
                col = (24, 26, 30) if black else (226, 228, 232)
            d.rectangle(box, fill=col, outline=(40, 44, 50))
            if n == bass:
                cx, cy = (box[0] + box[2]) // 2, box[3] - 12
                d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=GREEN, outline=(20, 60, 30))
            if not black and n % 12 == 0:
                text(d, ((box[0] + box[2]) // 2, KY0 + KWH - 30), note_name(n), F_XS, (90, 96, 106), "ma")


def text_w(s, f):
    return int(f.getlength(s))


def draw_learning(d, dt, losses):
    box = (436, 172, 1244, 340)
    d.rounded_rectangle(box, 12, fill=(20, 26, 34), outline=CYAN, width=2)
    text(d, (456, 184), "学习中：只改 BASS 网络的最后一层（36 行 × 96 个数）", F_H, CYAN)
    text(d, (456, 210), "把存下的每一拍过 15 遍；网络的其他部分不动。FM-1 上大约 1 秒，乐队一直在弹", F_S, TXT)
    if not losses:
        return
    frac = min(dt / 3000.0, 1.0)
    n = max(1, int(round(frac * len(losses))))
    pr = [100.0 * math.exp(-x) for x in losses]     # the player's note's chance, its geometric mean over the beats
    x0, y0, w, h = 500, 328, 700, 78
    pts = [(x0 + i * w / (len(pr) - 1), y0 - h * pr[i] / 100.0) for i in range(n)]
    d.line((x0, y0, x0 + w, y0), fill=LINE, width=1)
    d.line((x0, y0 - h, x0 + w, y0 - h), fill=LINE, width=1)
    text(d, (x0 - 6, y0 - h - 8), "100%", F_XS, DIM, "ra")
    text(d, (x0 - 6, y0 - 8), "0", F_XS, DIM, "ra")
    if len(pts) > 1:
        d.line(pts, fill=ORANGE, width=3)
    for x, y in pts:
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=ORANGE)
    text(d, (x0 + w, y0 - h + 6), f"第 {n} 遍：网络给你弹的音的可能性 {pr[n - 1]:.0f}%", F_H, ORANGE, "ra")
    text(d, (x0 + 8, y0 - h + 6), f"第 1 遍 {pr[0]:.0f}%", F_S, DIM)


# ---------------------------------------------------------------- each mode's words
def teach_words(ctx, at, t_play, log):
    B = lambda bar: at(4 * bar)                      # noqa: E731
    taught = {b: v[0] for b, v in log["T"].items()}

    def taught_at(b):                               # after teaching: the note taught on this beat of the form
        if b < 4 * A_BAR0:
            return -1
        for t in range(b - 32, 4 * T_BAR0 - 1, -32):
            if t in taught and taught[t] >= 2:
                return taught[t]
        return -1

    def score_at(ms):                               # how many beats of this round so far were the taught note
        b = int(ctx["beat_at"](ms) + 0.1)
        if b < 4 * CH_BARS or 4 * T_BAR0 <= b < 4 * A_BAR0:
            return ""
        r0 = (b // 32) * 32
        same = n = 0
        for x in range(r0, b + 1):
            e = log["B"].get(x)
            want = -1
            for t in range(x + 32 * 8, 4 * T_BAR0 - 1, -32):
                if 4 * T_BAR0 <= t < 4 * A_BAR0 and t in taught and taught[t] >= 2:
                    want = taught[t]
                    break
            if e and not e["teach"] and want >= 2:
                n += 1
                same += e["tok"] == want
        if not n:
            return ""
        label = "这一遍和你教的一样" if b >= 4 * A_BAR0 else "（教之前）这一遍和后来教的一样"
        return f"{label}：{same}/{n} 拍"

    def intro(d):
        draw_card(d, "TRIO：FM-1 芯片上的四个小神经网络", [
            ("CHORD", "认和弦：从你按着的音认出是什么和弦", "你按下音之后 25 毫秒", ORANGE),
            ("HARM", "猜下一个和弦，和它什么时候来", "每换一个和弦（曲式学会以前，屏幕上的 NEXT?）", ORANGE),
            ("BASS", "贝斯：每一拍弹哪个音，要不要加花", "每拍之前 40 毫秒算好", ORANGE),
            ("DRUM", "鼓手：每个三连音步打哪几件鼓、多重", "提前半小节算好", ORANGE)],
            "它们在主循环里一步一步算（一步贝斯约 3 毫秒）；声音由音频中断准时放出来。下面是模拟器里的一次演奏")

    def end(d):
        draw_card(d, "小结", [
            ("平时", "四个网络按拍子一步一步算，网络本身不变", "权重在 Mac 上训练好，开机拷进内存", ORANGE),
            ("教贝斯", "按 EDIT 你弹贝斯：每拍存下网络的状态和你的音", "最多存最后 32 小节", CYAN),
            ("学习", "再按 EDIT：只改贝斯网络的最后一层，15 遍", "FM-1 上大约 1 秒；SAVE 存进 flash", CYAN),
            ("效果", "同样的和弦，贝斯更常弹你的音", "FREEDOM 50 大约一半的拍，FREEDOM 0 几乎照弹", CYAN)],
            "按住 EDIT 1.5 秒：回到训练好的样子")

    l0 = [x for x in log["L"] if x[0] == 0]
    l1 = [x for x in log["L"] if x[0] == 1]
    def where(ms):
        b = ctx["beat_at"](ms)
        return f"第 {int(b // 4) + 1} 小节" if 0 <= b < 4 * END_BAR else ""

    ctx.update(taught_at=taught_at, score_at=score_at, intro=intro, end=end, chords_until=at(4 * CH_BARS),
               where=where, teaching_shown=True, learning=(l0[-1][1], l1[-1][2]) if l0 and l1 else None)
    ctx["parts"] = [(t_play, "① 平时：你弹和弦，网络配贝斯和鼓"),
                    (B(T_BAR0) - BEAT, "② 教贝斯：按 EDIT，你来弹贝斯"),
                    (B(A_BAR0), "③ 学完：同样的和弦，贝斯开始像你")]
    ctx["captions"] = [
        (t_play, B(0) - 200, "按 PLAY：先数一小节拍子"),
        (B(0) - 200, B(1.5), "你弹一个和弦，CHORD 网络大约 25 毫秒后认出它"),
        (B(1.5), B(3), "HARM 网络猜下一个和弦和它什么时候来（屏幕上的 NEXT?）"),
        (B(3), B(5), "每拍之前 40 毫秒，BASS 网络看和弦、下一个和弦、上一个音，给 E1–G3 每个音打分"),
        (B(5), B(7.5), "柱子越高越可能；橙色是这一拍真正弹出的音：按可能性抽签，FREEDOM 越高越敢冒险"),
        (B(7.5), B(10), "DRUM 网络提前半小节决定每步打哪几件鼓：亮的已经打了，暗的已经定好"),
        (B(10), B(CH_BARS), "和弦转回来了：TRIO 记住了 8 小节的曲式（屏幕上 NEXT 不再带问号）"),
        (B(CH_BARS), B(T_BAR0) - BEAT, "你停手，乐队照着曲式继续走。这一遍是 TRIO 自己的走法"),
        (B(T_BAR0) - BEAT, B(T_BAR0 + 3), "按 EDIT：TRIO 的贝斯不弹了，你在 E1–G3 弹的音就是贝斯"),
        (B(T_BAR0 + 3), B(T_BAR0 + 9), "灰色：网络自己会怎么弹；蓝框：你实际弹的。每一拍都把网络的状态和你的音存下来"),
        (B(T_BAR0 + 9), B(A_BAR0), "这条线：每个和弦弹根音、五度、八度，最后一拍弹下一个根音上方半音"),
        (B(A_BAR0), B(A_BAR0) + 3800, "再按 EDIT：开始学。只改贝斯网络的最后一层，别的网络不动"),
        (B(A_BAR0) + 3800, B(A_BAR0 + AFTER), "蓝框是你在这个位置教的音：橙色（贝斯弹的）经常落在上面"),
        (B(A_BAR0 + AFTER), B(END_BAR), "FREEDOM 调到 0：贝斯几乎照着你教的弹"),
    ]


def chorus_stats(log, beats, c):
    """a chorus of the autumn session: the root on the changes, the chord's notes, half-step approaches, pickups,
    holds; the drums a bar"""
    n = len(beats)
    ch = roots = tones = notes = app = napp = skips = holds = 0
    prev = None
    for b in range(c * n, (c + 1) * n):
        e = log["B"].get(b)
        if not e:
            continue
        p = 28 + e["tok"] - 2 if e["tok"] >= 2 else None
        change = b == 0 or beats[b % n] != beats[(b - 1) % n]
        if e["tok"] == 1:
            holds += 1
        if p is not None:
            notes += 1
            tones += QMASK[e["qual"]] >> ((p - e["root"]) % 12) & 1
            if change:
                ch += 1
                roots += p % 12 == e["root"]
                if prev is not None:
                    napp += 1
                    app += abs(p - prev) == 1
            prev = p
        skips += e["skip"]
    hits = [0] * 9
    for g, t, h in log["D"]:
        if c * n * 3 <= g < (c + 1) * n * 3:
            for i in range(9):
                hits[i] += h >> i & 1
    bars = n / 4
    return dict(changes=ch, roots=roots, tones=tones, notes=notes, app=app, napp=napp, skips=skips, holds=holds,
                ride=hits[0] / bars, snare=hits[1] / bars, kick=hits[2] / bars, hh=hits[3] / bars)


def autumn_words(ctx, at, t_play, log):
    beats = parse_chart(AUTUMN)
    n = len(beats)
    B = lambda bar: at(4 * bar)                      # noqa: E731
    SECT = ["A", "A", "B", "C"]

    def where(ms):
        b = int(ctx["beat_at"](ms) + 0.02)
        if b < 0 or b >= CHORUSES * n:
            return ""
        bar = (b % n) // 4
        return f"第 {b // n + 1} 遍 · {SECT[bar // 8]} 段 · 第 {bar + 1} 小节"

    stats = [chorus_stats(log, beats, c) for c in range(CHORUSES)]

    def intro(d):
        draw_card(d, "Autumn Leaves：神经网络配出什么", [
            ("曲子", "Autumn Leaves，G 小调，32 小节（A A B C），每分钟 140 拍", "三遍；只弹和弦（根音在最低，比拍子早一点）", ORANGE),
            ("你", "弹和弦：下面的琴键会亮；CHORD 网络认出它，HARM 网络猜下一个", "钢琴的声音也是 TRIO 的", ORANGE),
            ("贝斯", "BASS 网络每拍写一个音：下面一行是它真正弹的", "灰色的音不在和弦里：经过音、接近音", ORANGE),
            ("鼓", "DRUM 网络提前半小节决定每一步", "第一遍 two-feel，之后 four-feel；第三遍更热闹", ORANGE)],
            "没有事先写好的贝斯线和鼓谱：每一个音都是网络当场算出来的", x_what=300)

    def end(d):
        rows = []
        for c, s in enumerate(stats):
            rows.append((f"第 {c + 1} 遍",
                         f"换和弦落根音 {s['roots']}/{s['changes']} · 和弦内音 {round(100 * s['tones'] / max(s['notes'], 1))}%"
                         f" · 半音进入 {round(100 * s['app'] / max(s['napp'], 1))}% · 加花 {s['skips']} 次",
                         f"贝斯 {s['notes']} 个音、{s['holds']} 个延音；鼓每小节：ride {s['ride']:.1f}  军鼓 {s['snare']:.1f}"
                         f"  底鼓 {s['kick']:.1f}  踩镲 {s['hh']:.1f}", ORANGE if c < 2 else CYAN))
        rows.append(("对照", "FiloBass（48 条真人 walking）：换和弦落根音 75% · 半音进入 51%",
                     "TRIO 的贝斯网络就是从这些线学的", DIM))
        draw_card(d, "这三遍贝斯和鼓弹了什么", rows, "第一遍：曲式还没学会，前 16 小节 two-feel；第三遍：FREEDOM 80、DRUMS 80",
                  x_what=330)

    forms = log["F"]
    first = next((f for f in forms if f[1]), None)  # the form: its first guess, the first drop, the whole tune
    drop = next((f for f in forms if not f[1] and first and f[0] > first[0]), None)
    whole = next((f for f in forms if f[1] == n // 4), None)
    f_bar = first[0] / 4 if first else 10.0
    d_bar = max(drop[0] / 4, 19.0) if drop else 19.0
    w_bar = whole[0] / 4 if whole else 41.0
    s2, s3 = stats[1], stats[2]
    pct = lambda a, b: round(100 * a / max(b, 1))   # noqa: E731
    caps = [
        (t_play, B(0) - 200, "按 PLAY：数一小节；然后你在每小节前一点点弹和弦"),
        (B(0) - 200, B(4), "第一遍：TRIO 还不认识这首曲子。AUTO：先 two-feel（两拍一个音），像乐队演奏主题"),
        (B(4), B(f_bar), "屏幕上的 NEXT? 是 HARM 网络猜的下一个和弦；猜对了，贝斯就能提前走向它"),
        (B(16), B(19), "满 16 小节：转成 four-feel，每拍一个音，这就是 walking bass"),
        (B(25), B(32), "C 段第 3、4 小节两拍一换（Gm7 C7 | Fm7 Bb7）：贝斯也两拍换一个和弦"),
        (B(32), B(w_bar), "第二遍：和弦从头再来，TRIO 把你弹的和它记下的一拍一拍对照"),
        (B(w_bar + 6), B(w_bar + 14), "柱子：这一拍每个音的可能性；橙色是抽中的音。可能性分散的地方，每一遍都弹得不一样"),
        (B(w_bar + 14), B(64) - 700, "三遍里没有一小节是重复的：同样的和弦，网络每一次都重新算"),
        (B(64) - 700, B(72), "第三遍：FREEDOM 调到 80（贝斯更大胆），DRUMS 调到 80（鼓手更热闹）"),
        (B(72), B(CHORUSES * 32) - 60,
         f"这一遍半音进入 {pct(s3['app'], s3['napp'])}%（第二遍 {pct(s2['app'], s2['napp'])}%），加花 {s3['skips']} 次"
         f"（第二遍 {s2['skips']} 次），鼓也打得更多"),
        (B(CHORUSES * 32) - 60, B(CHORUSES * 32) + 2600, "最后停在 Gm7"),
    ]
    if first:
        caps.append((B(f_bar), B(min(f_bar + 4, 16)),
                     f"A 段转回来了：TRIO 先以为曲式是 {first[1]} 小节（屏幕上 FORM {first[1]} BARS）"))
        caps.append((B(min(f_bar + 4, 16) - 0.001) + 1, B(16), "下面一行是贝斯线：每格一小节，蓝色是和弦，灰色的音不在和弦里"))
    if drop:
        caps.append((B(d_bar), B(25), "B 段和它记的不一样：TRIO 放掉这个曲式，一边听一边接着找"))
    if whole:
        tried = []
        for b, bars in forms:
            if bars and bars != whole[1] and bars not in tried and b < whole[0]:
                tried.append(bars)
        caps.append((B(w_bar), B(w_bar + 6),
                     f"TRIO 记住了整首 {whole[1]} 小节（先后试过 {'、'.join(map(str, tried))} 小节）：贝斯提前知道下一个和弦"))
    caps.sort(key=lambda c: c[0])
    ctx.update(taught_at=lambda b: -1, score_at=lambda ms: "", intro=intro, end=end, where=where, teaching_shown=False,
               chords_until=at(CHORUSES * n), learning=None, captions=caps,
               parts=[(t_play, "Autumn Leaves：你弹和弦，TRIO 配贝斯和鼓")])


# ---------------------------------------------------------------- the video
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("teach", "autumn") else None
    if not mode:
        sys.exit(__doc__)
    work = OUT / f"video-{mode}"
    work.mkdir(parents=True, exist_ok=True)
    s, t_play, at, t_end = session_teach() if mode == "teach" else session_autumn()
    (work / "demo.txt").write_text(f"# TRIO's demo video, {mode} (tools/demo_video.py: generated)\n" +
                                   "\n".join(s.lines) + "\n")
    stills = sys.argv[2:]
    if not stills or not (work / "viz.txt").exists():
        for old in work.glob("s[0-9]*.png"):
            old.unlink()
        r = subprocess.run([str(HOST), str(work / "demo.txt"), str(work)], env={"TRIO_VIZ": str(work / "viz.txt")},
                           capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr)
    log = load_log(work / "viz.txt")
    shots = sorted(work.glob("s[0-9]*.png"))
    shot_ms = [int(p.stem[1:]) for p in shots]
    cache = {}

    def shot_at(ms):
        i = bisect.bisect_right(shot_ms, ms) - 1
        if i < 0:
            return None
        if i not in cache:
            cache.clear()
            cache[i] = Image.open(shots[i]).convert("RGB")
        return cache[i]

    def beat_at(ms):
        return (ms - at(0)) / BEAT

    def step_at(ms):
        b = beat_at(ms)
        if b < 0:
            return None
        k = int(b)
        f = b - k
        return 3 * k + (2 if f >= 0.64 else 1 if f >= 0.32 else 0)

    b_keys = sorted(log["B"])
    ons = [t for k, t, _ in log["E"] if k == 1]
    offs = [t for k, t, _ in log["E"] if k == 0]
    stops = [t for t, k, lab in s.events if k == "ctl" and "停" in lab]
    ctx = {"log": log, "at": at, "shot_at": shot_at, "beat_at": beat_at, "step_at": step_at, "t_play": t_play,
           "t_end": t_end, "b_keys": b_keys, "b_times": [log["B"][b]["t"] for b in b_keys],
           "drums": {g: (t, h) for g, t, h in log["D"]}, "keys": Keys(s.events, list(zip(ons, offs))),
           "t_stop": stops[-1] + 30 if stops else t_end}
    (teach_words if mode == "teach" else autumn_words)(ctx, at, t_play, log)
    t0, t1 = 0.0, t_end + 6000
    if stills:                                      # stills: MS..., or bN for bar N's start + 300 ms, or "end"
        for a in stills:
            ms = t_end + 1000 if a == "end" else at(4 * float(a[1:])) + 300 if a.startswith("b") else float(a)
            draw_frame(ms, ctx).save(work / f"still-{a}.png")
            print(work / f"still-{a}.png")
        return 0
    out = OUT / ("trio-demo.mp4" if mode == "teach" else "trio-autumn-leaves.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-ss", f"{t0 / 1000:.3f}", "-i", str(work / "demo.wav"),
                           "-map", "0:v", "-map", "1:a", "-af", "volume=8dB,apad", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                           "-t", f"{(t1 - t0) / 1000:.3f}", str(out)],
                          stdin=subprocess.PIPE)
    n = int((t1 - t0) / 1000 * FPS)
    for i in range(n):
        ff.stdin.write(draw_frame(t0 + i * 1000.0 / FPS, ctx).tobytes())
        if i % (FPS * 20) == 0:
            print(f"frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
