from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "project.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    pass


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_config(data)
    return data


def config_sha256(path: Path | str = DEFAULT_CONFIG) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(data: dict[str, Any]) -> None:
    _require(data.get("schema_version") == 1, "unsupported schema_version")

    metric = data.get("metric", {})
    _require(metric.get("name") == "FTM", "metric must be FTM")
    expected_faces = ["U", "R", "F", "L", "BR", "BL", "FR", "FL", "DR", "DL", "B", "D"]
    _require(metric.get("faces") == expected_faces, "face order differs from pinned chain")
    _require(metric.get("powers") == [1, 2, 3, 4], "FTM powers must be 1..4")
    _require(metric.get("cost_per_non_identity_face_turn") == 1, "every FTM turn must cost one")

    upstream = data.get("upstream", {})
    _require(SHA40.fullmatch(str(upstream.get("commit", ""))) is not None, "invalid upstream commit")
    _require(SHA64.fullmatch(str(upstream.get("archive_sha256", ""))) is not None, "invalid archive checksum")
    _require(upstream.get("branch") == "Original-Order", "unexpected upstream branch")

    expected = {
        "3": ("G5->G6", "7Gen", 208099584, 14, 212),
        "4": ("G6->G7", "6Gen", 68400, 8, 2531),
        "5": ("G7->G8", "5Gen", 64157184, 13, 3484),
        "6": ("G8->G9", "4Gen", 25945920, 13, 117),
    }
    phases = data.get("phases", {})
    _require(set(phases) == set(expected), "phases must be exactly 3,4,5,6")
    for phase, values in expected.items():
        row = phases[phase]
        observed = (
            row.get("transition"),
            row.get("upstream_table"),
            row.get("state_count"),
            row.get("expected_diameter"),
            row.get("expected_antipodes"),
        )
        _require(observed == values, f"phase {phase} controls differ from pinned baseline")

    pairs = data.get("pairs", {})
    _require(set(pairs) == {"pair34", "pair56"}, "unexpected pair set")
    for name, phase_ids, max_length in (
        ("pair34", ("3", "4"), 21),
        ("pair56", ("5", "6"), 25),
    ):
        row = pairs[name]
        expected_product = phases[phase_ids[0]]["expected_antipodes"] * phases[phase_ids[1]]["expected_antipodes"]
        _require(row.get("phases") == [int(phase_ids[0]), int(phase_ids[1])], f"{name} phase order mismatch")
        _require(row.get("expected_raw_pairs") == expected_product, f"{name} raw product mismatch")
        _require(row.get("max_length") == max_length, f"{name} bound mismatch")

    policy = data.get("resource_policy", {})
    _require(isinstance(policy.get("minimum_free_gib"), int) and policy["minimum_free_gib"] > 0, "invalid free-space reserve")
    _require(policy.get("maximum_project_gib") == 10, "project size cap must be 10 GiB")
    _require(policy.get("maximum_workers") == 10, "worker cap must be 10")
    _require(isinstance(policy.get("cpu_nice"), int) and 0 <= policy["cpu_nice"] <= 19, "invalid CPU nice value")
    _require(policy.get("io_priority_class") == "best-effort", "I/O class must be best-effort")
    _require(isinstance(policy.get("io_priority_level"), int) and 0 <= policy["io_priority_level"] <= 7, "invalid I/O priority")
    _require(policy.get("atomic_publication") is True, "atomic publication must be enabled")
    _require(policy.get("single_writer") is True, "single-writer policy must be enabled")
