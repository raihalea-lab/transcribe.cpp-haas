#!/usr/bin/env bash
# build-macos.sh [--no-install]: build Hex from $HAAS_HOME/hex (prepared by
# prepare-hex.sh), sign it with the self-signed identity "$SIGN_IDENTITY",
# and replace /Applications/Hex.app with it.
#
# The app is built as Hex releases are (HEX_RELEASE_UNSIGNED=1: name Hex,
# bundle id anomaly.hex, ad-hoc signed) and marked as a local build
# (HEX_LOCAL_BUILD=1: "Local build · updates off", never downloads an
# update over this one). It is then re-signed inside-out in the order of
# Hex's .github/release/sign-macos.sh, with the same entitlements but no
# timestamp or notarization. Signing every build with the same identity
# keeps macOS's microphone / accessibility permissions across rebuilds.
set -euo pipefail
. "$(dirname "$0")/common.sh"

install=1
[ "${1:-}" = --no-install ] && install=0

[ "$(uname)" = Darwin ] || die "build-macos.sh runs on macOS"
[ -d "$HEX_DIR/.git" ] || die "no Hex checkout in $HEX_DIR; run prepare-hex.sh"
security find-identity -p codesigning | grep -qF "\"$SIGN_IDENTITY\"" ||
  die "no code signing identity \"$SIGN_IDENTITY\"; run haas/hex/create-signing-identity.sh"

say "building Hex ($(git -C "$HEX_DIR" log -1 --format='%h %s'))"
bundle=$(cd "$HEX_DIR" && HEX_RELEASE_UNSIGNED=1 HEX_LOCAL_BUILD=1 scripts/build-app.sh | tail -1)
[ -d "$bundle" ] || die "build-app.sh did not produce an app bundle"

say "signing $bundle as \"$SIGN_IDENTITY\""
sign() { codesign --force --options runtime --sign "$SIGN_IDENTITY" "$@"; }
# Any bundled dylib first (a build may still ship libonnxruntime).
if [ -d "$bundle/Contents/Frameworks" ]; then
  find "$bundle/Contents/Frameworks" -type f -name '*.dylib' -print0 |
    while IFS= read -r -d '' lib; do sign "$lib"; done
fi
sign "$bundle/Contents/Library/HexMicrophone.driver"
sign --identifier anomaly.hex.scrawl-helper "$bundle/Contents/MacOS/scrawl-helper"
sign --entitlements "$HEX_DIR/.github/release/hex.entitlements" "$bundle"
codesign --verify --deep --strict --verbose=2 "$bundle"

if [ "$install" = 0 ]; then
  echo "$bundle"
  exit 0
fi

installed=/Applications/Hex.app
if [ -d "$installed" ]; then
  id=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$installed/Contents/Info.plist" 2>/dev/null || true)
  case "$id" in
    anomaly.hex | com.kitlangton.hex2) ;;
    *) die "refusing to replace $installed (bundle id \"$id\" is not Hex)" ;;
  esac
  say "quitting the running Hex"
  osascript -e 'tell application id "anomaly.hex" to quit' >/dev/null 2>&1 || true
  for _ in $(seq 30); do
    pgrep -f "$installed/Contents/MacOS/hex" >/dev/null || break
    sleep 1
  done
  pkill -TERM -f "$installed/Contents/MacOS/hex" 2>/dev/null || true
fi
say "installing $installed"
rm -rf "$installed"
/usr/bin/ditto "$bundle" "$installed"
/usr/bin/open "$installed"
echo "$installed"
