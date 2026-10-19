// SPDX-License-Identifier: MPL-2.0
// TRIO's part of the emulator's worklet: the page's MIDI (app.js) into the app as USB-MIDI packets (cable 0).
globalThis.fm1App = {
  message(fm1, m) {
    if (!m.midi) return;
    const [st, d1, d2] = m.midi;
    fm1.ex.web_midi_in(0, ((st >> 4) | (st << 8) | (d1 << 16) | (d2 << 24)) >>> 0);
  },
};
