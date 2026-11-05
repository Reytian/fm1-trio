// SPDX-License-Identifier: MPL-2.0
// TRIO's part of the emulator page: a MIDI keyboard on this computer (Web MIDI) plays TRIO as a keyboard on the
// FM-1's USB does (its notes, the sustain pedal). Without one, the panel's keys and the computer's (Q..I, A..K, the
// digits) play chords too.
export function attach({ node, status }) {
  globalThis.trio = { midi: (st, d1, d2) => node.port.postMessage({ type: "app", midi: [st, d1, d2] }) };   // (tests)
  if (!navigator.requestMIDIAccess) {
    status("No Web MIDI in this browser: play the chords on the panel's keys, or the computer's (Q..I, A..K, 1..0).");
    return;
  }
  navigator.requestMIDIAccess().then((midi) => {
    const hook = () => {
      const names = [];
      for (const input of midi.inputs.values()) {
        input.onmidimessage = (e) => {
          const [st, d1 = 0, d2 = 0] = e.data;
          if (st >= 0x80 && st < 0xf0) node.port.postMessage({ type: "app", midi: [st, d1 & 0x7f, d2 & 0x7f] });
        };
        names.push(input.name);
      }
      status(names.length ? "MIDI in: " + names.join(", ") + ". Press PLAY (or P), then play chords."
                          : "No MIDI keyboard found: plug one in, or play the panel's keys. PLAY (or P) starts the band.");
    };
    hook();
    midi.onstatechange = hook;
  }).catch(() => status("MIDI not allowed: play the chords on the panel's keys. PLAY (or P) starts the band."));
}
