#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
# Build TRIO (fm1-base tools/build.py on app/):
#   ./build.sh                 app/build/trio.fwsc (the JieLi toolchain, in Docker on macOS)
#   ./build.sh --host          the host simulator: app/build/host/trio_host
#   ./build.sh --web           the browser emulator: app/build/emu/
#   ./build.sh --release X.Y   app/build/trio-X.Y.fwsc
#   JIELI_TOOLCHAIN  JieLi Linux toolchain (default: ~/.jieli/toolchain)
#   AC79_SDK         JieLi AC79 SDK checkout (default: ~/fw-AC79_AIoT_SDK)
#   PYTHON           a Python 3 with Pillow (default: python3)
set -e
cd "$(dirname "$0")"
export JIELI_TOOLCHAIN="${JIELI_TOOLCHAIN:-$HOME/.jieli/toolchain}"
export AC79_SDK="${AC79_SDK:-$HOME/fw-AC79_AIoT_SDK}"
PY="${PYTHON:-python3}"
"$PY" -c 'import PIL' 2>/dev/null || { echo "build.sh: $PY has no Pillow (pip3 install Pillow, or PYTHON=a python with it)"; exit 1; }
exec "$PY" base/tools/build.py app "$@"
