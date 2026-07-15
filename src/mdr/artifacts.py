from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .config import ConfigError


SHA64 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(ValueError):
    pass


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_table_metadata(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {
        "schema_version", "phase", "transition", "repository_commit",
        "upstream_commit", "metric", "generators", "state_count", "diameter",
        "antipode_count", "payloads", "complete",
    }
    missing = required - set(data)
    if missing:
        raise ArtifactError(f"missing table metadata fields: {sorted(missing)}")
    if data["schema_version"] != 1 or data["phase"] not in (3, 4, 5, 6):
        raise ArtifactError("unsupported table metadata version or phase")
    if data["metric"] != "FTM" or data["complete"] is not True:
        raise ArtifactError("table is not a complete FTM artifact")
    if not isinstance(data["state_count"], int) or data["state_count"] <= 0:
        raise ArtifactError("invalid state_count")
    if not isinstance(data["diameter"], int) or data["diameter"] < 0:
        raise ArtifactError("invalid diameter")
    if not isinstance(data["antipode_count"], int) or data["antipode_count"] <= 0:
        raise ArtifactError("invalid antipode_count")
    if not isinstance(data["payloads"], dict) or not data["payloads"]:
        raise ArtifactError("payload manifest is empty")
    return data


def compare_table_controls(metadata: dict[str, Any], config: dict[str, Any]) -> list[str]:
    phase = str(metadata["phase"])
    if phase not in config["phases"]:
        raise ConfigError(f"phase {phase} is not configured")
    expected = config["phases"][phase]
    face_count = {"3": 7, "4": 6, "5": 5, "6": 4}[phase]
    expected_generators = [
        f"{face}{power}"
        for face in config["metric"]["faces"][:face_count]
        for power in config["metric"]["powers"]
    ]
    mismatches: list[str] = []
    checks = {
        "transition": expected["transition"],
        "repository_commit": config["upstream"]["commit"],
        "upstream_commit": config["upstream"]["commit"],
        "generators": expected_generators,
        "state_count": expected["state_count"],
        "diameter": expected["expected_diameter"],
        "antipode_count": expected["expected_antipodes"],
    }
    for field, control in checks.items():
        if metadata.get(field) != control:
            mismatches.append(f"{field}: generated={metadata.get(field)!r} control={control!r}")
    return mismatches


def verify_payloads(metadata_path: Path | str, metadata: dict[str, Any]) -> None:
    base = Path(metadata_path).resolve().parent
    for name, spec in metadata["payloads"].items():
        if not isinstance(spec, dict) or not SHA64.fullmatch(str(spec.get("sha256", ""))):
            raise ArtifactError(f"invalid checksum declaration for payload {name}")
        payload_path = (base / name).resolve()
        if payload_path.parent != base:
            raise ArtifactError(f"payload path escapes artifact directory: {name}")
        if not payload_path.is_file():
            raise ArtifactError(f"missing payload: {name}")
        actual = sha256_file(payload_path)
        if actual != spec["sha256"]:
            raise ArtifactError(f"checksum mismatch for payload {name}")
        if "bytes" in spec and payload_path.stat().st_size != spec["bytes"]:
            raise ArtifactError(f"size mismatch for payload {name}")


def compare_published_histogram(metadata_path: Path | str, phase: int) -> list[str]:
    controls_path = Path(__file__).resolve().parents[2] / "controls" / "published_histograms.json"
    with controls_path.open("r", encoding="utf-8") as handle:
        controls = json.load(handle)
    expected = controls["phases"][str(phase)]
    histogram_path = Path(metadata_path).resolve().parent / "histogram.csv"
    with histogram_path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed: list[int] = []
    for expected_depth, row in enumerate(rows):
        try:
            depth = int(row["depth"])
            count = int(row["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"malformed histogram row {row!r}") from exc
        if depth != expected_depth:
            raise ArtifactError(f"non-contiguous histogram at depth {expected_depth}: got {depth}")
        observed.append(count)
    mismatches: list[str] = []
    for depth in range(max(len(expected), len(observed))):
        control = expected[depth] if depth < len(expected) else None
        generated = observed[depth] if depth < len(observed) else None
        if generated != control:
            mismatches.append(f"histogram depth {depth}: generated={generated!r} control={control!r}")
    return mismatches
