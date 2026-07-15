#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="$ROOT/upstream/source/Original Order"
OUT="$ROOT/build/upstream-bin"
IMAGE=${GO_IMAGE:-golang:1.22.3-bookworm@sha256:f43c6f049f04cbbaeb28f0aad3eea15274a7d0a7899a617d0037aec48d7ab010}

test -d "$SOURCE"
mkdir -p "$ROOT/build"

docker run --rm \
  -v "$ROOT:/work" \
  -w '/work/upstream/source/Original Order' \
  "$IMAGE" \
  go test -buildvcs=false ./...

docker run --rm \
  -v "$ROOT:/work" \
  -w '/work/upstream/source/Original Order' \
  "$IMAGE" \
  go build -buildvcs=false -trimpath -o /work/build/upstream-bin.tmp .

mv "$OUT.tmp" "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
echo "built $OUT"
