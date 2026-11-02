#!/usr/bin/env bash
# build-windows.sh [--no-install]: from Git Bash on Windows, build the Hex
# prepared in $HAAS_HOME/hex (prepare-hex.sh) with Hex's own
# scripts/build-windows.sh as a local build (HEX_LOCAL_BUILD=1: updates off),
# then install it for the current user:
#
#   %LOCALAPPDATA%\Programs\Hex-Haas\hex.exe     the app (stable path, so
#                                                Launch at login keeps working)
#   Start menu "Hex (Haas)"                       runs `hex.exe app`
#
# Any running Hex (this one or the official app) is asked to quit first: Hex
# allows one instance, and a second launch would only show the first. The
# official installer can't install a local build (it downloads releases), so
# this replaces it; settings, history and models in %APPDATA%\hex are shared.
#
# x64 builds ggml's Vulkan backend (Vulkan SDK to build, a GPU driver with
# vulkan-1.dll to run); ARM64 runs on the CPU and builds with clang-cl + Ninja
# from a "Developer PowerShell for VS 2022" (run `bash -l` there).
set -euo pipefail
. "$(dirname "$0")/common.sh"

install=1
[ "${1:-}" = --no-install ] && install=0

is_windows || die "build-windows.sh runs from Git Bash on Windows"
[ -d "$HEX_DIR/.git" ] || die "no Hex checkout in $HEX_DIR; run prepare-hex.sh"
arch=$(windows_arch)
[ "$arch" = aarch64 ] && need clang-cl ninja

say "building Hex for $arch ($(git -C "$HEX_DIR" log -1 --format='%h %s'))"
exe=$(cd "$HEX_DIR" && HEX_WINDOWS_ARCH=$arch HEX_LOCAL_BUILD=1 HEX_COMMIT=$(git rev-parse --short=7 HEAD) \
  scripts/build-windows.sh | tail -1)
[ -f "$exe" ] || die "Hex's build-windows.sh did not produce an executable"
say "built $exe"
[ "$install" = 1 ] || { echo "$exe"; exit 0; }

if [ -f "$(cygpath -u "$APPDATA")/hex/install.json" ]; then
  echo "note: the official Hex is installed too. Uninstall it in Settings > Apps > Hex" \
    "(that keeps settings, history and models), or it will start at login and update itself." >&2
fi

say "quitting the running Hex, if any"
"$exe" quit --force >/dev/null 2>&1 || true
for _ in $(seq 30); do
  tasklist //FI "IMAGENAME eq hex.exe" 2>/dev/null | grep -qi '^hex.exe' || break
  sleep 1
done
if tasklist //FI "IMAGENAME eq hex.exe" 2>/dev/null | grep -qi '^hex.exe'; then
  taskkill //F //IM hex.exe >/dev/null 2>&1 || true
  sleep 1
fi

dir="$(cygpath -u "$LOCALAPPDATA")/Programs/Hex-Haas"
installed="$dir/hex.exe"
say "installing $(cygpath -w "$installed")"
mkdir -p "$dir"
cp "$exe" "$installed"

win_exe=$(cygpath -w "$installed")
powershell -NoProfile -Command "
  \$link = Join-Path ([Environment]::GetFolderPath('Programs')) 'Hex (Haas).lnk'
  \$s = (New-Object -ComObject WScript.Shell).CreateShortcut(\$link)
  \$s.TargetPath = '$win_exe'
  \$s.Arguments = 'app'
  \$s.WorkingDirectory = '$(cygpath -w "$dir")'
  \$s.Save()" || echo "note: could not create the Start menu shortcut" >&2

say "starting Hex"
powershell -NoProfile -Command "Start-Process -FilePath '$win_exe' -ArgumentList 'app'"
echo "$win_exe"
