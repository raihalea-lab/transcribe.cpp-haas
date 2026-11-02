#!/usr/bin/env bash
# parity.sh [<Haas .nemo | HF repo id>] [<gguf>]: compare the C++ KDA predictor
# and TDT decoding with the PyTorch reference (haas/check_parity.py) on the
# real weights. Linux only: the reference imports flash-linear-attention,
# which needs Triton (no GPU needed). Expects a Haas-ASR checkout in
# $HAAS_ASR_SRC (default: ../Haas-ASR/src next to this repo).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/hex/common.sh"
[ "$(uname)" = Linux ] ||
  die "parity.sh runs on Linux only (fla needs Triton, which has no macOS build). On a Mac, check the model by ear instead: build/bin/transcribe-cli -m $HAAS_HOME/models/Haas-ja-v1.0-Q8_0.gguf samples/ja.wav"
repo=$(git -C "$here" rev-parse --show-toplevel)
src=${1:-third-intelligence/Haas-ja-v1.0}
gguf=${2:-$HAAS_HOME/models/Haas-ja-v1.0/Haas-ja-v1.0-F32.gguf}
[ -f "$gguf" ] || die "no GGUF at $gguf: run haas/convert.sh first"
haas_src=${HAAS_ASR_SRC:-$(dirname "$repo")/Haas-ASR/src}
[ -f "$haas_src/haas_asr/prednet.py" ] ||
  die "no Haas-ASR at $haas_src (git clone https://github.com/third-intelligence/Haas-ASR, or set HAAS_ASR_SRC)"

[ -x "$repo/build/bin/transcribe-cli" ] || {
  cmake -S "$repo" -B "$repo/build" -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON >/dev/null
  cmake --build "$repo/build" -j --target transcribe-cli transcribe-quantize >/dev/null
}
"$here/build_parity.sh" "$repo/build" >/dev/null

tol=1e-5
case "$gguf" in *Q8_0*) tol=2e-2 ;; esac
# Versions from Haas-ASR's uv.lock; fla 0.5.2 needs this Triton.
cd "$here"
uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  --with torch==2.14.0 --with triton==3.8.0 --with flash-linear-attention==0.5.2 --with transformers==5.20.2 \
  --with einops --with numpy --with pyyaml --with huggingface_hub \
  python check_parity.py "$src" "$gguf" --haas-src "$haas_src" --parity-bin "$repo/build/bin/haas-kda-parity" --tol "$tol"
