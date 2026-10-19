# TRIO's training data

The networks' weights (`app/dsp/trio_w.h`) and the bass's measured partials (`app/dsp/bass_tab.h`) are in this
repository. To make them again, you need the two datasets below. Neither is redistributed here.

| Dataset | What TRIO takes from it | Licence | Where |
| --- | --- | --- | --- |
| FiloBass (Riley and Dixon, ISMIR 2023) | 48 walking-bass lines by professional bassists, note by note, with their chords: `notebooks_and_scripts/note_data.csv`. For the bass's sound: the audio stems (`audio_bass_stems/`) and the MIDI aligned to them (`midi_fully_aligned/`) | CC BY 4.0 | <https://github.com/aim-qmul/FiloBass> (release v1.0.0), <https://aim-qmul.github.io/FiloBass/> |
| JazzSAMBA (Long et al., 2026) | 76 jazz standards played on electronic instruments: the drums' MIDI (54 takes), the piano's (its voicings) and the chord charts aligned to the playing | MIT-style: "Copyright (c) 2026 Phillip Long and contributors" | <https://github.com/pnlong/jazzsamba>, Zenodo DOI 10.5281/zenodo.22943962 |

Two things to know about FiloBass:

- `note_midi` in `note_data.csv` is the written pitch, an octave above sounding: the bass is a transposing
  instrument in notation. `trio_data.py` subtracts 12.
- The paper's Table 4 has the labels of the ±1 and ±2 approach intervals swapped. Recounted from `note_interval`
  in `note_data.csv`, the approaches into a chord change, by the approach note's distance below the target, are:

  | Target − approach (semitones) | Share |
  | --- | --- |
  | +1 (a half step from below) | 27.5% |
  | −1 (a half step from above) | 21.4% |
  | −2 (a whole step from above) | 12.2% |
  | +5 (a fourth from below) | 10.0% |
  | 0 (the same note) | 5.9% |
  | −7 (a fifth from above) | 4.4% |
  | +2 (a whole step from below) | 4.0% |
  | others | 14.6% |

  `tools/walking_model.py` uses these corrected shares for its synthetic lines.

## Training

The tools need Python 3 with PyTorch, NumPy and mido. Run Python with `-I` on downloaded data: it loads no code from
the data's directory.

    python3 -I tools/trio_data.py --filobass FILOBASS/notebooks_and_scripts/note_data.csv \
        --jazzsamba JAZZSAMBA --out DATA --synth 1500
    python3 -I tools/trio_train.py --data DATA --out app --tests tests/host

`trio_data.py` writes `bass.npz` (a beat a step: 48 FiloBass lines and 1,500 synthetic ones), `drums.npz` (a swung
triplet eighth a step: 54 takes, 12,006 bars), `harmony.npz` (76 charts, 21,120 chords) and `voicings.npz` (23,265
voicings). `trio_train.py` trains the four networks, writes `app/dsp/trio_w.h` (8-bit weights, a scale per matrix)
and `tests/host/trio_golden.h` (inputs and outputs that `tests/host/nn_test.c` checks the C code against). It takes
a few minutes on a laptop's CPU. `--quick` checks the pipeline in seconds. The harmony network learns from
JazzSAMBA's 76 charts and FiloBass's 48 (not from the synthetic lines).

## The bass's sound

`tools/bass_measure.py` measures real upright pizzicato in FiloBass's audio: 42,758 notes, each partial's level and
decay by register, the attack, the pitch. It writes `app/dsp/bass_tab.h`, the voice's tables. It needs NumPy, mido and
ffmpeg, and the analysis takes about seven minutes.

    python3 -I tools/bass_measure.py stems "FILOBASS/FiloBass ISMIR Publication" REF.npz
    python3 -I tools/bass_measure.py fit REF.npz app/dsp/bass_tab.h

To compare TRIO's own notes with the measurement, render them with `tests/host/bass_notes.c` (its comment has the
commands), then run `bass_measure.py wav` and `bass_measure.py compare REF.npz NOTES.npz`.
