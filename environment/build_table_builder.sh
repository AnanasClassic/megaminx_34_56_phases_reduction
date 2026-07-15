#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM="$ROOT/upstream/source/Original Order"
STAGE="$ROOT/build/table-builder-src"
OUT="$ROOT/build/mdr-table"
IMAGE=${GO_IMAGE:-golang:1.22.3-bookworm@sha256:f43c6f049f04cbbaeb28f0aad3eea15274a7d0a7899a617d0037aec48d7ab010}

rm -rf "$STAGE"
mkdir -p "$STAGE"
for source in "$UPSTREAM"/*.go; do
  if [[ $(basename "$source") != main.go ]]; then cp "$source" "$STAGE/"; fi
done
cp "$UPSTREAM/go.mod" "$STAGE/go.mod"
cp "$ROOT/builder/go/main.go" "$STAGE/mdr_table_main.go"

docker run --rm -v "$ROOT:/work" -w /work/build/table-builder-src "$IMAGE" \
  go build -buildvcs=false -trimpath -o /work/build/mdr-table.partial .
mv "$OUT.partial" "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
echo "built $OUT"
