# Credits

## The platform

TRIO is built on [fm1-base](https://github.com/charlesvestal/fm1-base) by Charles Vestal (MPL-2.0), and on
[felucca-bsp](https://github.com/hugelton/felucca-bsp) by Leo Kuroshita, Hügelton Instruments (MPL-2.0), from
Felucca. fm1-base's own credits (the fonts, the screen reader, the tools) are in `base/CREDITS.md`. `app/app.c` and
`app/app.h` started from fm1-base's template.

For Bluetooth MIDI, the firmware links JieLi's Bluetooth libraries from the AC79 SDK, Copyright (c) JIELI (Zhuhai
Jieli Technology Co., Ltd.), under the Apache License 2.0 (`base/LICENSES/Apache-2.0.txt`). A build copies them
from the SDK; they aren't in this repository. For details, see `base/LICENSING.md`.

## What the networks learned from

The weights in `app/dsp/trio_w.h` were trained on these datasets. Neither dataset is included here.

**FiloBass**: Xavier Riley and Simon Dixon, "FiloBass: A Dataset and Corpus Based Study of Jazz Basslines",
ISMIR 2023. <https://github.com/aim-qmul/FiloBass>, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). TRIO's bass network learned from its 48 note-level
walking-bass lines (`note_data.csv`). The harmony network learned from their chords too. Its statistics also set
the approach-note shares of `tools/walking_model.py`. Changes: the lines were transposed to sounding pitch and cut
into beats. The upright's sound was measured from its audio stems: each partial's level and decay by register, in
42,758 notes (`tools/bass_measure.py`, giving `app/dsp/bass_tab.h`). None of the audio itself is in TRIO.

**JazzSAMBA**: Phillip Long and contributors, 2026. <https://github.com/pnlong/jazzsamba>, Zenodo
DOI 10.5281/zenodo.22943962. TRIO's drum network learned from its drum MIDI. The chord network learned from its
piano voicings, and the harmony network from its chord charts. Its licence:

    JazzSAMBA original recordings and annotations
    Copyright (c) 2026 Phillip Long and contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this dataset (original audio, ground-truth MIDI, and annotations) to use,
    copy, modify, merge, publish, and distribute, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the dataset.

    Derived files (automatic music transcription MIDI and debleeded audio) are
    estimates, not ground truth, and are provided as-is.

    THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

## Methods

- The walking-bass line model in `tools/walking_model.py` follows the usual teaching: chord tones on the strong
  beats, approach notes into each change, a pull back to the middle of the register.
- The upright bass is additive: twenty partials a note, with the levels and decays measured in FiloBass's
  recordings, a finger's attack and a short thud. The cymbals are a bank of six square waves at the TR-808's metal
  frequencies, filtered. The room is a four-line feedback delay network.
