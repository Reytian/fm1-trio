# TRIO

TRIO turns the M-VAVE FM-1 into a jazz rhythm section that listens. Play chords on the FM-1's keys or a MIDI
keyboard or digital piano, by cable or Bluetooth, and a walking bass and a drummer play along with you. Both are
small neural networks running on the FM-1 itself. TRIO hears each chord, learns the tune's form once it comes round,
and walks toward the next change before it arrives. Stop playing and the band keeps the form going. Play something
new and it follows you. Press EDIT and play the bass yourself for a chorus or two: the FM-1 learns your line, and
the bass walks more the way you do.

![TRIO playing: F7 in large type, NEXT Bbmaj7 IN 3, FORM 8 BARS, BASS F2, FEEL AUTO (IN FOUR)](manual/img/screen-playing.png "TRIO on the FM-1's screen")

Version {{VERSION}}. Installing is at your own risk; the installer can put M-VAVE's own firmware back.

- [Try it in the browser](emu/): the same code the FM-1 runs, with sound. Play it with a MIDI keyboard on your
  computer. No FM-1 needed.
- [Install](install/): from Chrome or Edge, with the FM-1 connected by USB.
- [Manual](manual/): playing with the band, the form, the feel, the mix.
- [Download {{PKG}}](firmware/{{PKG}}): for the command-line installer, `python3 tools/fm1_install.py {{PKG}}`.
- [Source](https://github.com/Reytian/fm1-trio): MPL-2.0. Built on fm1-base and felucca-bsp. The bass learned from
  FiloBass (CC BY 4.0) and the drummer from JazzSAMBA.

M-VAVE and FM-1 are trademarks of their owners. TRIO is not affiliated with M-VAVE.
