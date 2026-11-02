#!/usr/bin/env bash
# update.sh [--ref <hex ref>] [--skip-tests] [--no-install] [--push]
#
# Build Hex with the Haas transcribe.cpp and install it:
#   1. pick the Hex release (newest vX.Y.Z tag, or --ref: a tag, a branch
#      such as dev, or a commit) and read the transcribe.cpp commit it pins;
#   2. put this fork's Haas commits on that commit (sync-transcribe.sh;
#      rebases on a new branch haas-<pin7> when the pin moved);
#   3. build and test that transcribe.cpp (test-transcribe.sh);
#   4. check Hex out with its transcribe.cpp patch pointed at it
#      (prepare-hex.sh, one local commit on branch haas-build);
#   5. build, sign and install the app (build-macos.sh), or the Windows
#      executable from Git Bash (build-windows.sh);
#   6. with --push, push the rebased Haas branch to origin as `haas`.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/common.sh"

ref="" tests=1 install=1 push=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) ref=${2:?--ref needs a value}; shift ;;
    --skip-tests) tests=0 ;;
    --no-install) install=0 ;;
    --push) push=1 ;;
    -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

need git cmake cargo jq c++
[ "$(uname)" = Darwin ] && need xcrun
fetch_hex
[ -n "$ref" ] || ref=$(latest_hex_tag)
if git -C "$HEX_DIR" rev-parse -q --verify "refs/tags/$ref^{commit}" >/dev/null; then
  hex_rev=$(git -C "$HEX_DIR" rev-parse "refs/tags/$ref^{commit}")
elif git -C "$HEX_DIR" rev-parse -q --verify "refs/remotes/origin/$ref^{commit}" >/dev/null; then
  hex_rev=$(git -C "$HEX_DIR" rev-parse "refs/remotes/origin/$ref^{commit}")
else
  hex_rev=$(git -C "$HEX_DIR" rev-parse --verify "$ref^{commit}") || die "unknown Hex ref $ref"
fi
pin=$(hex_pinned_rev <(git -C "$HEX_DIR" show "$hex_rev:Cargo.toml"))
[ -n "$pin" ] || die "Hex $ref ($hex_rev) does not pin anomalyco/transcribe.cpp by commit"
say "Hex $ref = ${hex_rev:0:8}, transcribe.cpp pin ${pin:0:7}"
if ! git -C "$HEX_DIR" cat-file -e "$hex_rev:src/transcription/custom_models.rs" 2>/dev/null; then
  echo "warning: Hex $ref has no Labs > Use your own speech model, so Haas cannot be added in it." >&2
  echo "         Build --ref dev until a release includes it (anomalyco/hex#759)." >&2
fi

target=$("$here/sync-transcribe.sh" "$pin")
say "transcribe.cpp for Hex: ${target:0:8} in $TC_BUILD_DIR"
if [ "$tests" = 1 ]; then
  "$here/test-transcribe.sh" "$TC_BUILD_DIR"
fi
"$here/prepare-hex.sh" "$hex_rev" "$TC_BUILD_DIR"

case "$(uname -s)" in
  Darwin)
    if [ "$install" = 1 ]; then "$here/build-macos.sh"; else "$here/build-macos.sh" --no-install; fi ;;
  MINGW64* | MSYS*)
    "$here/build-windows.sh" ;;
  *)
    say "Hex is prepared in $HEX_DIR; build it there with Hex's own scripts" ;;
esac

if [ "$push" = 1 ]; then
  repo=$(git -C "$here" rev-parse --show-toplevel)
  say "pushing ${target:0:8} to origin as haas"
  git -C "$repo" push --force-with-lease origin "$target:refs/heads/haas"
  if [ "$(git -C "$repo" rev-parse HEAD)" != "$target" ]; then
    echo "note: this checkout is still at $(git -C "$repo" rev-parse --short HEAD);" \
      "update it with: git -C $repo fetch origin && git -C $repo reset --hard origin/haas" >&2
  fi
fi
