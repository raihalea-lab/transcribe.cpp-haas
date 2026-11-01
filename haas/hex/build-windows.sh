#!/usr/bin/env bash
# build-windows.sh: from Git Bash on Windows, build the Hex prepared in
# $HAAS_HOME/hex (prepare-hex.sh) with Hex's scripts/build-windows.sh, as a
# local build (HEX_LOCAL_BUILD=1: updates off). Prints the .exe.
#
# Not yet tried on a Windows machine. Needs what Hex's Windows build needs
# (Rust MSVC toolchain, Visual Studio Build Tools, CMake, jq, Vulkan SDK for
# x86_64); the KDA predictor itself runs on the CPU.
set -euo pipefail
. "$(dirname "$0")/common.sh"

case "$(uname -s)" in
  MINGW64* | MSYS*) ;;
  *) die "build-windows.sh runs from Git Bash on Windows" ;;
esac
[ -d "$HEX_DIR/.git" ] || die "no Hex checkout in $HEX_DIR; run prepare-hex.sh"
cd "$HEX_DIR"
HEX_LOCAL_BUILD=1 HEX_COMMIT=$(git rev-parse --short=7 HEAD) scripts/build-windows.sh >&2
ls -t dist/hex-*-windows-*.exe | head -1
