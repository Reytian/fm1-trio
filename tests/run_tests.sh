#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
# TRIO's tests: the networks in C against PyTorch's own numbers, the bass's learning against its PyTorch reference
# (tools/trio_finetune.py), the engine offline (three tunes: the form learned, no beat missed; a walking line taught:
# the bass plays it more often after, no beat missed), then the app in the host simulator (tests/basic.txt, band.txt,
# teach.txt). Exit 1 on any failure.
set -e
cd "$(dirname "$0")/.."
mkdir -p app/build/tests
CC="${CC:-cc}"
FLAGS="-O2 -ffp-contract=off -fno-vectorize -fno-slp-vectorize -std=c99 -Wall -Wextra -Wdouble-promotion -Werror
       -DTNN_HOST -Iapp/dsp -Ibase/include"
# shellcheck disable=SC2086
$CC $FLAGS -Itests/host -o app/build/tests/nn_test tests/host/nn_test.c app/dsp/trio_nn.c -lm
app/build/tests/nn_test
# shellcheck disable=SC2086
$CC $FLAGS -Itests/host -o app/build/tests/ft_test tests/host/ft_test.c app/dsp/trio_nn.c -lm
app/build/tests/ft_test
# shellcheck disable=SC2086
$CC $FLAGS -o app/build/tests/trio_render tests/host/trio_render.c app/dsp/trio.c app/dsp/voice.c app/dsp/trio_nn.c -lm
for tune in autumn:32 blues:12 rhythm:32; do
    out=$(app/build/tests/trio_render "${tune%%:*}" 160 3 app/build/tests/"${tune%%:*}".wav early rooted)
    echo "$out" | grep -E '^(bass|form):'
    echo "$out" | grep -q "form: ${tune##*:} bars learned; missed 0;" || { echo "FAIL ${tune%%:*}"; exit 1; }
done
# shellcheck disable=SC2086
$CC $FLAGS -Itests/host -o app/build/tests/teach_test tests/host/teach_test.c app/dsp/trio.c app/dsp/voice.c \
    app/dsp/trio_nn.c -lm
for args in "autumn 160 5" "blues 200 3 --two" "rhythm 160 3 --fresh"; do   # (each exits 1 on a missed beat)
    # shellcheck disable=SC2086
    out=$(app/build/tests/teach_test $args) || { echo "$out" | grep -E '^(mean|FAIL)'; echo "FAIL teach $args"; exit 1; }
    echo "teach $args: $(echo "$out" | grep '^mean' | sed 's/^mean: //')"
done
./build.sh --host >/dev/null
for s in basic band teach; do           # (a failed check exits 1: the scenario's FAIL lines, then stop)
    out=$(app/build/host/trio_host app/tests/$s.txt app/build/host) ||
        { echo "$out" | grep -E '^(FAIL|line )' || true; echo "FAIL $s"; exit 1; }
    echo "$out" | tail -1
done
echo "all passed"
