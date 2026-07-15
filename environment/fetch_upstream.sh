#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$ROOT/upstream/source"
ARCHIVE_URL='https://tudatalib.ulb.tu-darmstadt.de/bitstreams/ddf7dced-8274-4063-8a9f-eae2c1891cad/download'
ARCHIVE_SHA='1aa7a2f033a0d20c23ab5b7f014e77013a71b05593310cec28eaf2bc6091c6cb'
COMMIT='82db40b5744297617446da888cd73d5e26f57239'
BRANCH='Original-Order'
PUBLIC_REMOTE='https://git.rwth-aachen.de/alexander.botz/megaminx-solver-v2.0'

if [[ -d "$DEST/.git" ]]; then
  test "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$DEST" status --porcelain)"
  echo "upstream already pinned at $COMMIT"
  exit 0
fi

command -v curl >/dev/null
command -v 7z >/dev/null
command -v git >/dev/null

mkdir -p "$ROOT/upstream/downloads"
archive="$ROOT/upstream/downloads/Projekt.7z"
tmp=$(mktemp -d "$ROOT/upstream/.extract.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

if [[ ! -f "$archive" ]] || ! echo "$ARCHIVE_SHA  $archive" | sha256sum -c - >/dev/null 2>&1; then
  partial="$archive.partial"
  rm -f "$partial"
  curl --fail --location --silent --show-error "$ARCHIVE_URL" -o "$partial"
  echo "$ARCHIVE_SHA  $partial" | sha256sum -c - >/dev/null
  mv "$partial" "$archive"
fi

7z x -y -o"$tmp" "$archive" >/dev/null
source_repo="$tmp/Projekt - Kopie"
test -d "$source_repo/.git"
test "$(git -C "$source_repo" rev-parse "$COMMIT^{commit}")" = "$COMMIT"

rm -rf "$DEST"
git clone --quiet --no-hardlinks --branch "$BRANCH" "$source_repo" "$DEST"
git -C "$DEST" remote set-url origin "$PUBLIC_REMOTE"
git -C "$DEST" checkout --quiet --detach "$COMMIT"
test -z "$(git -C "$DEST" status --porcelain)"
echo "fetched upstream commit $COMMIT"
