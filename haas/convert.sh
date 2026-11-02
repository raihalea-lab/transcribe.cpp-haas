#!/usr/bin/env bash
# convert.sh [<Haas .nemo | HF repo id>]: convert Haas-ja-v1.0 into
#   $HAAS_HOME/models/Haas-ja-v1.0/Haas-ja-v1.0-F32.gguf   (reference)
#   $HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf               (for Hex)
# The default source is third-intelligence/Haas-ja-v1.0 on Hugging Face.
# Runs on macOS or Linux with uv; needs neither NeMo nor a GPU.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/hex/common.sh"
repo=$(git -C "$here" rev-parse --show-toplevel)
src=${1:-third-intelligence/Haas-ja-v1.0}

work="$HAAS_HOME/models/Haas-ja-v1.0"  # the converter picks the profile from this name
f32="$work/Haas-ja-v1.0-F32.gguf"
q8="$HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf"
mkdir -p "$work"
need git uv cmake c++

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

tool="$repo/build/bin/transcribe-quantize"
if [ ! -x "$tool" ]; then
  say "building transcribe-quantize"
  cmake -S "$repo" -B "$repo/build" -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON >/dev/null
  cmake --build "$repo/build" -j --target transcribe-quantize transcribe-cli >/dev/null
fi
"$tool" "$f32" "$q8" --quant Q8_0 >/dev/null
say "wrote $f32"
say "wrote $q8 (add this one in Hex)"
