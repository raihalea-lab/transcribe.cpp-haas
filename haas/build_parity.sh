#!/usr/bin/env bash
# Build haas-kda-parity against an existing transcribe.cpp CPU build
# (default: build/, made with `cmake -B build && cmake --build build --target transcribe-cli`).
# Linux (the VM) or macOS. Kept out of the upstream CMake files on purpose.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$root/build}"
out="$build/bin/haas-kda-parity"
libs=("$build/src/libtranscribe.a")
for l in ggml ggml-cpu ggml-base ggml-metal ggml-blas; do
  f="$(find "$build/ggml" -name "lib$l.a" | head -1)"
  [[ -n "$f" ]] && libs+=("$f")
done
extra=(-lpthread)
if [[ "$(uname)" == Darwin ]]; then
  extra+=(-framework Foundation -framework Metal -framework MetalKit -framework Accelerate)
else
  extra+=(-ldl)
fi
mkdir -p "$(dirname "$out")"
"${CXX:-c++}" -std=c++17 -O2 -I"$root/include" -I"$root/src" -I"$root/ggml/include" \
  "$root/haas/kda_parity.cpp" -Wl,--start-group "${libs[@]}" -Wl,--end-group "${extra[@]}" -o "$out" 2>/dev/null \
|| "${CXX:-c++}" -std=c++17 -O2 -I"$root/include" -I"$root/src" -I"$root/ggml/include" \
  "$root/haas/kda_parity.cpp" "${libs[@]}" "${libs[@]}" "${extra[@]}" -o "$out"
echo "$out"
