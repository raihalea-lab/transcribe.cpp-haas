#!/usr/bin/env bash
# test-transcribe.sh <transcribe.cpp dir>: build transcribe.cpp (CLI, quantize
# tool, unit tests), run its tests, and, when a Haas GGUF is at hand, compare
# its transcripts of the Japanese samples with the ones saved the first time
# ($HAAS_HOME/golden/). A changed transcript stops the update: check it by
# ear, and delete the golden file to accept it.
#
# HAAS_MODEL overrides the model (default $HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf).
set -euo pipefail
. "$(dirname "$0")/common.sh"

tc=$(cd "${1:?usage: test-transcribe.sh <transcribe.cpp dir>}" && pwd)
build="$tc/build"
jobs=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)

say "building transcribe.cpp $(git -C "$tc" rev-parse --short HEAD)"
log="$build.log"
configure=(-DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON)
targets=()
if is_windows; then
  # Only the CLI: upstream runs its C++ tests on Linux and macOS, not under
  # MSVC. ARM64 needs clang-cl (ggml's ARM CPU code refuses MSVC).
  targets=(--target transcribe-cli)
  configure+=(-DTRANSCRIBE_BUILD_TESTS=OFF -DGGML_NATIVE=OFF)
  if [ "$(windows_arch)" = aarch64 ]; then
    need clang-cl ninja
    configure+=(-G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DGGML_OPENMP=OFF)
  fi
fi
{ cmake -S "$tc" -B "$build" "${configure[@]}" &&
  cmake --build "$build" --config Release -j "$jobs" ${targets[@]+"${targets[@]}"}; } >"$log" 2>&1 ||
  { tail -40 "$log" >&2; die "build failed (log: $log)"; }
if is_windows; then
  say "skipping transcribe.cpp's unit tests on Windows"
else
  say "running transcribe.cpp's tests"
  (cd "$build" && ctest -C Release -j "$jobs" --output-on-failure) >"$log" 2>&1 ||
    { tail -60 "$log" >&2; die "transcribe.cpp tests failed (log: $log)"; }
fi
cli="$build/bin/transcribe-cli"
[ -x "$cli" ] || [ -x "$cli.exe" ] || cli="$build/bin/Release/transcribe-cli"  # Visual Studio generator

model=${HAAS_MODEL:-$HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf}
if [ ! -f "$model" ]; then
  say "no Haas model at $model: skipping the transcript check (run haas/convert.sh)"
  exit 0
fi
golden="$HAAS_HOME/golden"
mkdir -p "$golden"
status=0
for sample in ja.wav ja-short.wav fleurs-ja.wav; do
  [ -f "$tc/samples/$sample" ] || continue
  out=$(mktemp)
  "$cli" -q --timestamps none -m "$model" -o "$out" "$tc/samples/$sample" >"$log" 2>&1 ||
    { cat "$log" >&2; die "transcribe-cli failed on $sample"; }
  ref="$golden/$(basename "$model" .gguf).${sample%.wav}.txt"
  if [ ! -f "$ref" ]; then
    cp "$out" "$ref"
    say "saved $ref:"
    cat "$ref" >&2
  elif ! cmp -s "$out" "$ref"; then
    echo "transcript of $sample changed:" >&2
    diff "$ref" "$out" >&2 || true
    status=1
  else
    say "$sample: transcript unchanged"
  fi
  rm -f "$out"
done
[ "$status" = 0 ] || die "transcripts changed; listen and, if the new ones are fine, delete the files above and rerun"
