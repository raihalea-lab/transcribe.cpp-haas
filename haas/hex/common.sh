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

# Git Bash on Windows (MSYS2).
is_windows() { case "$(uname -s)" in MINGW* | MSYS*) return 0 ;; *) return 1 ;; esac; }

# x86_64 or aarch64 on Windows (HEX_WINDOWS_ARCH wins). Git Bash itself may
# run emulated on ARM64, so ask Windows about the processor.
windows_arch() {
  if [ -n "${HEX_WINDOWS_ARCH:-}" ]; then
    echo "$HEX_WINDOWS_ARCH"
    return
  fi
  case "$(powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).Architecture' 2>/dev/null | tr -d '\r')" in
    12) echo aarch64 ;;
    *) echo x86_64 ;;
  esac
}

# need <command>...: stop with install hints when any is missing.
need() {
  local missing=() c
  for c in "$@"; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
  [ ${#missing[@]} = 0 ] && return 0
  echo "error: missing ${missing[*]}. Install with:" >&2
  if is_windows; then
    for c in "${missing[@]}"; do
      case "$c" in
        jq) echo "  winget install --id jqlang.jq -e" >&2 ;;
        cmake) echo "  winget install --id Kitware.CMake -e" >&2 ;;
        cargo) echo "  winget install --id Rustlang.Rustup -e   (then, in a new terminal: rustup default stable)" >&2 ;;
        clang-cl | ninja) echo "  run this from \"Developer PowerShell for VS 2022\" (bash -l), with VS Build Tools' clang (see haas/README.md)" >&2 ;;
        *) echo "  $c" >&2 ;;
      esac
    done
    echo "(open a new Git Bash afterwards so PATH picks them up)" >&2
    exit 1
  fi
  for c in "${missing[@]}"; do
    case "$c" in
      cmake | jq) echo "  brew install $c        (or: uv tool install $c)" >&2 ;;
      cargo) echo "  curl https://sh.rustup.rs -sSf | sh   (Rust via rustup)" >&2 ;;
      uv) echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2 ;;
      c++ | xcrun) echo "  xcode-select --install   (Xcode for the Hex app build)" >&2 ;;
      *) echo "  $c" >&2 ;;
    esac
  done
  exit 1
}

# The newest Hex release tag (vX.Y.Z; no prereleases).
latest_hex_tag() {
  git ls-remote --tags --refs "$HEX_UPSTREAM" 'v*' \
    | sed -n 's|.*refs/tags/\(v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$|\1|p' \
    | sort -t. -k1,1.2n -k2,2n -k3,3n | tail -1
}

# Clone or fetch Hex into $HEX_DIR.
fetch_hex() {
  # LF checkouts: Hex's shell scripts must run in Git Bash, which a
  # Windows-style (CRLF) checkout would break.
  if [ ! -d "$HEX_DIR/.git" ]; then
    mkdir -p "$HAAS_HOME"
    git clone --quiet -c core.autocrlf=false "$HEX_UPSTREAM" "$HEX_DIR"
  elif [ "$(git -C "$HEX_DIR" config core.autocrlf || true)" != false ]; then
    git -C "$HEX_DIR" config core.autocrlf false
    git -C "$HEX_DIR" rm -r -q --cached .
    git -C "$HEX_DIR" reset -q --hard
  fi
  git -C "$HEX_DIR" fetch --quiet --tags --force origin
}

# The transcribe.cpp commit a Hex Cargo.toml pins in [patch.crates-io].
hex_pinned_rev() { # <Cargo.toml>
  sed -n 's|^transcribe-cpp-sys = { git = "https://github.com/anomalyco/transcribe.cpp", rev = "\([0-9a-f]\{40\}\)" }.*|\1|p' "$1"
}

# A path usable in Cargo.toml (Windows Git Bash needs C:/...).
native_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}
