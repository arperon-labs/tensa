#!/usr/bin/env bash
# finalize-license.sh — download the canonical AGPL-3.0 text and verify it
#
# Run this once after cloning the licensing template into your TENSA repo.
# It replaces the placeholder in LICENSE with the real AGPL-3.0 text from
# the GNU Foundation.

set -euo pipefail

AGPL_URL="https://www.gnu.org/licenses/agpl-3.0.txt"
AGPL_FILE="LICENSE-AGPL-3.0.txt"
# Expected SHA-256 of the upstream file. This can change if FSF reformats
# whitespace; verify against an independent source before trusting.
EXPECTED_SHA256="73ce9e0785beb10ae6cbcea4fba3d4bc18edab9fed88d19eac3dbb0b5f83ea96"

if [[ ! -f NOTICE ]]; then
  echo "ERROR: run this from the repo root (NOTICE file not found)." >&2
  exit 1
fi

echo "Downloading AGPL-3.0 canonical text..."
curl -fsSL "$AGPL_URL" -o "$AGPL_FILE"

echo "Verifying checksum..."
ACTUAL_SHA256=$(sha256sum "$AGPL_FILE" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "WARNING: SHA-256 mismatch."
  echo "  expected: $EXPECTED_SHA256"
  echo "  actual:   $ACTUAL_SHA256"
  echo "The FSF may have reformatted the file. Review manually and update"
  echo "EXPECTED_SHA256 in this script if the content is still correct."
fi

# Replace the [Paste ...] placeholder in LICENSE with the real text
if grep -q "BEGIN AGPL-3.0 CANONICAL TEXT" LICENSE; then
  python3 - "$AGPL_FILE" <<'PY'
import sys, pathlib
agpl = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
license_path = pathlib.Path("LICENSE")
text = license_path.read_text(encoding="utf-8")
start = "================================================================\nBEGIN AGPL-3.0 CANONICAL TEXT\n================================================================\n"
end = "\n================================================================\nEND AGPL-3.0 CANONICAL TEXT\n================================================================\n"
if start not in text or end not in text:
    raise SystemExit("LICENSE markers not found; nothing replaced.")
before = text.split(start)[0] + start + "\n"
after = end + text.split(end)[1]
license_path.write_text(before + agpl + after, encoding="utf-8")
print("LICENSE updated with canonical AGPL-3.0 text.")
PY
else
  echo "LICENSE already has the AGPL text or a different structure; skipping in-place replacement."
fi

echo "Done."
