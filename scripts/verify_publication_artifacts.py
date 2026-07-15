#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import lzma
import sqlite3
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> tuple[str, int]:
    checksum = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(chunk)
            size += len(chunk)
    return checksum.hexdigest(), size


def verify_pair(directory: Path) -> dict[str, object]:
    manifest_path = directory / "certificates/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest_path.parent / manifest["archive"]["file"]
    observed_archive_sha, observed_archive_bytes = digest(archive)
    if observed_archive_sha != manifest["archive"]["sha256"] or observed_archive_bytes != manifest["archive"]["bytes"]:
        raise ValueError(f"{manifest['pair']}: certificate archive drift")

    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        checksum = hashlib.sha256()
        database_bytes = 0
        with lzma.open(archive, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                checksum.update(chunk)
                database_bytes += len(chunk)
                temporary.write(chunk)
        temporary.flush()
        if checksum.hexdigest() != manifest["database"]["sha256"] or database_bytes != manifest["database"]["bytes"]:
            raise ValueError(f"{manifest['pair']}: uncompressed database drift")
        connection = sqlite3.connect(f"file:{temporary.name}?mode=ro", uri=True)
        try:
            total, maximum = connection.execute(
                "SELECT COUNT(*), MAX(solution_length) FROM certificates"
            ).fetchone()
            beams = {
                str(beam): count for beam, count in connection.execute(
                    "SELECT beam_width, COUNT(*) FROM certificates GROUP BY beam_width"
                )
            }
            checkpoints = {
                (bytes(checkpoint).hex(), epoch): count
                for checkpoint, epoch, count in connection.execute(
                    "SELECT checkpoint_sha256, checkpoint_epoch, COUNT(*) "
                    "FROM certificates GROUP BY checkpoint_sha256, checkpoint_epoch"
                )
            }
        finally:
            connection.close()
    if total != manifest["database"]["certificates"] or maximum != manifest["database"]["maximum_solution_length"]:
        raise ValueError(f"{manifest['pair']}: database coverage drift")
    if beams != manifest["certificates_by_beam_width"]:
        raise ValueError(f"{manifest['pair']}: beam histogram drift")

    expected_checkpoints = {}
    for record in manifest["checkpoints"]:
        model = (manifest_path.parent / record["file"]).resolve()
        model_sha, _ = digest(model)
        if model_sha != record["sha256"]:
            raise ValueError(f"{manifest['pair']}: checkpoint drift: {model.name}")
        expected_checkpoints[(model_sha, record["epoch"])] = record["certificates"]
    if checkpoints != expected_checkpoints:
        raise ValueError(f"{manifest['pair']}: checkpoint provenance drift")
    if manifest["exact_reductions"] + total != manifest["raw_compositions"]:
        raise ValueError(f"{manifest['pair']}: exhaustive total drift")
    return {
        "pair": manifest["pair"],
        "raw_compositions": manifest["raw_compositions"],
        "exact_reductions": manifest["exact_reductions"],
        "direct_certificates": total,
        "maximum_solution_length": maximum,
        "checkpoints": len(expected_checkpoints),
        "valid": True,
    }


def main() -> None:
    results = [verify_pair(ROOT / "phase_3_4"), verify_pair(ROOT / "phase_5_6")]
    print(json.dumps({"valid": True, "pairs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
