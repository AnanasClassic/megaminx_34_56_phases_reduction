#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"pair34": ROOT / "phase_3_4", "pair56": ROOT / "phase_5_6"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and unpack a publication certificate database.")
    parser.add_argument("pair", choices=sorted(DIRECTORIES))
    args = parser.parse_args()
    directory = DIRECTORIES[args.pair]
    manifest = json.loads((directory / "certificates/manifest.json").read_text(encoding="utf-8"))
    archive = directory / "certificates" / manifest["archive"]["file"]
    output = ROOT / "certificates" / args.pair / "beam-cascade.sqlite3"
    temporary = output.with_suffix(".sqlite3.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    size = 0
    with lzma.open(archive, "rb") as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            target.write(chunk)
            checksum.update(chunk)
            size += len(chunk)
    expected = manifest["database"]
    if checksum.hexdigest() != expected["sha256"] or size != expected["bytes"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"error: {args.pair} database checksum mismatch")
    temporary.replace(output)
    print(json.dumps({"pair": args.pair, "database": str(output.relative_to(ROOT)), "bytes": size, "sha256": checksum.hexdigest(), "valid": True}, sort_keys=True))


if __name__ == "__main__":
    main()
