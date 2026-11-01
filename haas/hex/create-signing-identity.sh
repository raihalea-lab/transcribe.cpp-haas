#!/usr/bin/env bash
# create-signing-identity.sh ["Hex Local"]: create a self-signed code signing
# certificate + key in the login keychain, once, for build-macos.sh.
#
# macOS ties an app's privacy permissions (microphone, accessibility, input
# monitoring) to its signature. Signing every local build with this one
# identity keeps them across rebuilds; ad-hoc signatures change every build.
# macOS asks for your login password once, to trust the certificate for
# code signing. The first codesign run may ask to use the key: choose
# "Always Allow".
set -euo pipefail
. "$(dirname "$0")/common.sh"

name=${1:-$SIGN_IDENTITY}
keychain="$HOME/Library/Keychains/login.keychain-db"
[ "$(uname)" = Darwin ] || die "create-signing-identity.sh runs on macOS"

if security find-identity -p codesigning | grep -qF "\"$name\""; then
  say "\"$name\" already exists:"
  security find-identity -p codesigning | grep -F "\"$name\""
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/req.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $name
[ext]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
subjectKeyIdentifier = hash
CNF
# /usr/bin/openssl is LibreSSL, whose PKCS#12 defaults `security import` reads.
/usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$tmp/req.cnf" \
  -keyout "$tmp/key.pem" -out "$tmp/cert.pem" 2>/dev/null
pass=$(uuidgen)
/usr/bin/openssl pkcs12 -export -name "$name" -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
  -out "$tmp/identity.p12" -passout "pass:$pass"

say "importing \"$name\" into the login keychain"
security import "$tmp/identity.p12" -k "$keychain" -P "$pass" -T /usr/bin/codesign >/dev/null
say "trusting it for code signing (macOS asks for your password)"
security add-trusted-cert -r trustRoot -p codeSign -k "$keychain" "$tmp/cert.pem"

security find-identity -v -p codesigning | grep -F "\"$name\"" ||
  die "\"$name\" was imported but is not a valid code signing identity"
