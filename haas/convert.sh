#!/usr/bin/env bash
# convert.sh [<Haas .nemo | HF repo id>]: convert Haas-ja-v1.0 into
#   $HAAS_HOME/models/Haas-ja-v1.0/Haas-ja-v1.0-F32.gguf   (reference)
#   $HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf               (for Hex)
# The default source is third-intelligence/Haas-ja-v1.0 on Hugging Face.
# Runs on macOS, Linux or Windows (Git Bash) with uv; needs neither NeMo nor a GPU.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/hex/common.sh"
repo=$(git -C "$here" rev-parse --show-toplevel)
src=${1:-third-intelligence/Haas-ja-v1.0}

work="$HAAS_HOME/models/Haas-ja-v1.0"  # the converter picks the profile from this name
f32="$work/Haas-ja-v1.0-F32.gguf"
q8="$HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf"
mkdir -p "$work"
if is_windows; then
  need git uv cmake  # MSVC comes from VS Build Tools; CMake finds it
else
  need git uv cmake c++
fi

index=()
[ "$(uname)" = Linux ] && index=(--index https://download.pytorch.org/whl/cpu)  # CPU torch, not CUDA's
if [ -f "$f32" ] && [ "${1:-}" = "" ]; then
  say "using the existing $f32 (delete it to convert again)"
else
  say "converting $src"
  uv run --no-project --python 3.12 ${index[@]+"${index[@]}"} \
    --with torch --with numpy --with gguf --with sentencepiece --with omegaconf --with pyyaml --with huggingface_hub \
    python "$repo/scripts/convert-parakeet.py" "$src" "$f32.part"
  mv "$f32.part" "$f32"
fi

# The tool's path: build/bin/ (Makefiles, Ninja) or build/bin/Release/
# (the Visual Studio generator), with .exe on Windows.
quantize_tool() {
  local t
  for t in "$repo/build/bin/transcribe-quantize" "$repo/build/bin/Release/transcribe-quantize"; do
    if [ -x "$t" ] || [ -x "$t.exe" ]; then echo "$t"; return 0; fi
  done
  return 1
}
if ! tool=$(quantize_tool); then
  say "building transcribe-quantize"
  configure=(-DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON)
  if is_windows; then
    configure+=(-DTRANSCRIBE_BUILD_TESTS=OFF -DGGML_NATIVE=OFF)
    if [ "$(windows_arch)" = aarch64 ]; then
      need clang-cl ninja
      configure+=(-G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DGGML_OPENMP=OFF)
    fi
  fi
  mkdir -p "$repo/build"
  # A bare -j is unlimited under Makefiles and runs out of memory.
  jobs=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  log="$repo/build/haas-convert.log"  # inside the ignored build/, not beside it
  { cmake -S "$repo" -B "$repo/build" "${configure[@]}" &&
    cmake --build "$repo/build" --config Release -j "$jobs" --target transcribe-quantize transcribe-cli; } >"$log" 2>&1 ||
    { tail -40 "$log" >&2; die "build failed (log: $log)"; }
  tool=$(quantize_tool) || die "built, but no transcribe-quantize under $repo/build/bin"
fi
"$tool" "$f32" "$q8" --quant Q8_0 >/dev/null
say "wrote $f32"
say "wrote $q8 (add this one in Hex)"
