from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .artifacts import (
    compare_published_histogram,
    compare_table_controls,
    load_table_metadata,
    sha256_file,
    verify_payloads,
)
from .compositions import verify_compositions
from .config import DEFAULT_CONFIG, ROOT, load_config
from .hard_states import verify_hard
from .pair34_certificate_verify import verify as verify_pair34_certificates
from .pair56_certificate_verify import verify as verify_pair56_certificates
from .reduction_verify import verify as verify_reductions


class ProofError(ValueError):
    pass


PAIR_CONTROLS = {
    "pair34": {
        "label": "3+4",
        "phases": (3, 4),
        "raw": 536_572,
        "bound": 21,
        "database": Path("certificates/pair34/beam-cascade.sqlite3"),
        "problem": Path("training/pair34/problem.json"),
        "certificate_verifier": verify_pair34_certificates,
    },
    "pair56": {
        "label": "5+6",
        "phases": (5, 6),
        "raw": 407_628,
        "bound": 25,
        "database": Path("certificates/pair56/beam-cascade.sqlite3"),
        "problem": Path("training/pair56/problem.json"),
        "certificate_verifier": verify_pair56_certificates,
    },
}


def _require_equal(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ProofError(f"{name}: observed={observed!r}, expected={expected!r}")


def _verify_table_and_hard(
    *, root: Path, config: dict[str, object], phase: int,
) -> tuple[dict[str, object], dict[str, object]]:
    table = root / "tables" / f"phase{phase}"
    metadata_path = table / "metadata.json"
    metadata = load_table_metadata(metadata_path)
    verify_payloads(metadata_path, metadata)
    mismatches = compare_table_controls(metadata, config)
    mismatches.extend(compare_published_histogram(metadata_path, phase))
    if mismatches:
        raise ProofError(f"phase {phase} control mismatch: {'; '.join(mismatches)}")

    hard_path = root / "hard_states" / f"phase{phase}_depth{metadata['diameter']}.bin"
    hard = verify_hard(hard_path, table)
    _require_equal(f"phase {phase} hard depth", hard["depth"], metadata["diameter"])
    _require_equal(f"phase {phase} hard count", hard["count"], metadata["antipode_count"])
    return {
        "phase": phase,
        "state_count": metadata["state_count"],
        "diameter": metadata["diameter"],
        "antipode_count": metadata["antipode_count"],
        "metadata_sha256": sha256_file(metadata_path),
        "hard_states_sha256": sha256_file(hard_path),
        "hard_states_metadata_sha256": sha256_file(Path(str(hard_path) + ".metadata.json")),
    }, hard


def _coverage_summary(
    *, pair: str, reduction: dict[str, object], certificates: dict[str, object],
) -> dict[str, object]:
    controls = PAIR_CONTROLS[pair]
    raw = int(controls["raw"])
    bound = int(controls["bound"])
    _require_equal(f"{pair} reduction raw count", reduction["raw_pairs"], raw)
    _require_equal(
        f"{pair} certificate source count",
        certificates["remaining_representatives_total"], reduction["remaining"],
    )
    _require_equal(
        f"{pair} certificate coverage",
        certificates["verified_certificates"], reduction["remaining"],
    )
    certificate_maximum = certificates["maximum_solution_length"]
    if certificate_maximum is None or int(certificate_maximum) > bound:
        raise ProofError(
            f"{pair} maximum certificate length is {certificate_maximum!r}, bound is {bound}"
        )
    covered = int(reduction["closed"]) + int(certificates["verified_certificates"])
    _require_equal(f"{pair} exhaustive coverage", covered, raw)
    return {
        "raw_states": raw,
        "covered_states": covered,
        "closed_by_exact_reduction": reduction["closed"],
        "covered_by_direct_certificate": certificates["verified_certificates"],
        "maximum_verified_solution_length": int(certificate_maximum),
        "required_maximum_length": bound,
        "missing_certificates": 0,
        "invalid_certificates": 0,
    }


def certify_pair(
    pair: str, *, root: Path = ROOT, config_path: Path = DEFAULT_CONFIG,
    database: Path | None = None, max_length: int | None = None,
) -> dict[str, object]:
    if pair not in PAIR_CONTROLS:
        raise ProofError(f"unknown pair: {pair}")
    controls = PAIR_CONTROLS[pair]
    bound = int(controls["bound"])
    if max_length is not None:
        _require_equal(f"{pair} requested max length", max_length, bound)
    root = root.resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)
    phases = tuple(int(value) for value in controls["phases"])

    tables: list[dict[str, object]] = []
    hard_results: list[dict[str, object]] = []
    for phase in phases:
        table, hard = _verify_table_and_hard(root=root, config=config, phase=phase)
        tables.append(table)
        hard_results.append(hard)

    composition_root = root / "compositions" / pair
    composition = verify_compositions(
        composition_root,
        root / "hard_states" / f"phase{phases[0]}_depth{tables[0]['diameter']}.bin",
        root / "hard_states" / f"phase{phases[1]}_depth{tables[1]['diameter']}.bin",
    )
    _require_equal(f"{pair} Cartesian product", composition["raw_pairs"], controls["raw"])

    reduction_root = root / "reductions" / pair
    reduction = verify_reductions(reduction_root, composition_root)
    database_path = (database or root / controls["database"]).resolve()
    certificate_verifier: Callable[..., dict[str, object]] = controls["certificate_verifier"]
    certificates = certificate_verifier(
        database=database_path,
        problem_path=(root / controls["problem"]).resolve(),
        hard_ids_path=(reduction_root / "remaining_ids.bin").resolve(),
        states_path=(composition_root / "unique_states.bin").resolve(),
    )
    coverage = _coverage_summary(
        pair=pair, reduction=reduction, certificates=certificates,
    )

    checksums = {
        "config": sha256_file(config_path),
        "composition_metadata": sha256_file(composition_root / "metadata.json"),
        "reduction_metadata": sha256_file(reduction_root / "metadata.json"),
        "certificate_database": sha256_file(database_path),
        "problem": sha256_file(root / controls["problem"]),
        "checker_source": sha256_file(Path(__file__)),
    }
    result: dict[str, object] = {
        "valid": True,
        "pair": pair,
        "phase_pair": controls["label"],
        **coverage,
        "tables": tables,
        "hard_states": hard_results,
        "composition": composition,
        "reduction": reduction,
        "certificates": certificates,
        "checksums": checksums,
    }
    proof_payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["proof_summary_sha256"] = hashlib.sha256(proof_payload).hexdigest()
    return result


def certify_all(
    *, root: Path = ROOT, config_path: Path = DEFAULT_CONFIG,
    pair34_database: Path | None = None, pair56_database: Path | None = None,
) -> dict[str, object]:
    results = {
        "pair34": certify_pair(
            "pair34", root=root, config_path=config_path, database=pair34_database,
        ),
        "pair56": certify_pair(
            "pair56", root=root, config_path=config_path, database=pair56_database,
        ),
    }
    payload = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    return {
        "valid": True,
        "conditional_megaminx_upper_bound": 112,
        "external_dependencies": [
            "the published 114-move FTM bound",
            "compatibility with the surrounding published phase chain",
        ],
        "pairs": results,
        "proof_bundle_sha256": hashlib.sha256(payload).hexdigest(),
    }
