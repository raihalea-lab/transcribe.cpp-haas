# Shared helpers for the Haas Hex scripts (sourced, bash).
#
# Layout (override with HAAS_HOME):
#   $HAAS_HOME/hex                 upstream Hex clone, built on a local branch
#   $HAAS_HOME/transcribe.cpp-build  worktree of this fork that Hex compiles
#   $HAAS_HOME/models              converted GGUFs
HEX_UPSTREAM=${HEX_UPSTREAM:-https://github.com/anomalyco/hex}
TC_UPSTREAM=${TC_UPSTREAM:-https://github.com/anomalyco/transcribe.cpp}
HAAS_HOME=${HAAS_HOME:-$HOME/haas-hex}
HEX_DIR=$HAAS_HOME/hex
TC_BUILD_DIR=$HAAS_HOME/transcribe.cpp-build
SIGN_IDENTITY=${HAAS_SIGN_IDENTITY:-Hex Local}

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*" >&2; }

# The newest Hex release tag (vX.Y.Z; no prereleases).
latest_hex_tag() {
  git ls-remote --tags --refs "$HEX_UPSTREAM" 'v*' \
    | sed -n 's|.*refs/tags/\(v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$|\1|p' \
    | sort -t. -k1,1.2n -k2,2n -k3,3n | tail -1
}

# Clone or fetch Hex into $HEX_DIR.
fetch_hex() {
  if [ ! -d "$HEX_DIR/.git" ]; then
    mkdir -p "$HAAS_HOME"
    git clone --quiet "$HEX_UPSTREAM" "$HEX_DIR"
  fi
  git -C "$HEX_DIR" fetch --quiet --tags --force origin
}

# The transcribe.cpp commit a Hex Cargo.toml pins in [patch.crates-io].
hex_pinned_rev() { # <Cargo.toml>
  sed -n 's|^transcribe-cpp-sys = { git = "https://github.com/anomalyco/transcribe.cpp", rev = "\([0-9a-f]\{40\}\)" }$|\1|p' "$1"
}

# A path usable in Cargo.toml (Windows Git Bash needs C:/...).
native_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}
