#!/usr/bin/env bash
# prepare-hex.sh <hex ref> <transcribe.cpp dir>: check out Hex at <hex ref>
# in $HAAS_HOME/hex on the local branch haas-build, with one commit on top
# that points Cargo's [patch.crates-io] transcribe-cpp(-sys) at
# <transcribe.cpp dir> instead of anomalyco's pinned commit.
set -euo pipefail
. "$(dirname "$0")/common.sh"

ref=${1:?usage: prepare-hex.sh <hex ref> <transcribe.cpp dir>}
tc=$(cd "${2:?usage: prepare-hex.sh <hex ref> <transcribe.cpp dir>}" && pwd)

fetch_hex
git -C "$HEX_DIR" checkout --quiet --force -B haas-build "$ref"
manifest="$HEX_DIR/Cargo.toml"
pin=$(hex_pinned_rev "$manifest")
[ -n "$pin" ] || die "Hex $ref does not pin anomalyco/transcribe.cpp by commit in [patch.crates-io]; this setup needs that."
[ "$(git -C "$tc" show HEAD:haas/BASE)" = "$pin" ] ||
  die "$tc is based on $(git -C "$tc" show HEAD:haas/BASE | cut -c1-7), Hex $ref pins ${pin:0:7}: run sync-transcribe.sh $pin first"

path=$(native_path "$tc")
src='https://github.com/anomalyco/transcribe.cpp'
sed -i.bak \
  -e "s|^transcribe-cpp = { git = \"$src\", rev = \"$pin\" }\$|transcribe-cpp = { path = \"$path/bindings/rust/transcribe-cpp\" }|" \
  -e "s|^transcribe-cpp-sys = { git = \"$src\", rev = \"$pin\" }\$|transcribe-cpp-sys = { path = \"$path\" }|" \
  "$manifest"
rm -f "$manifest.bak"
[ "$(grep -c "^transcribe-cpp\(-sys\)\? = { path = " "$manifest")" = 2 ] || die "could not rewrite the transcribe.cpp patch in $manifest"

say "updating Cargo.lock for the local transcribe.cpp"
cargo update --quiet --manifest-path "$manifest" -p transcribe-cpp -p transcribe-cpp-sys
git -C "$HEX_DIR" commit --quiet -am "Build with Haas transcribe.cpp $(git -C "$tc" rev-parse --short HEAD)

Points [patch.crates-io] transcribe-cpp(-sys) at a local checkout of the
Haas fork (KDA prediction network) based on ${pin:0:7}."
say "Hex $ref ($(git -C "$HEX_DIR" rev-parse --short "$ref^{commit}")) + Haas transcribe.cpp $(git -C "$tc" rev-parse --short HEAD) on branch haas-build"
