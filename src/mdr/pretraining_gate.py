from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import compare_published_histogram, compare_table_controls, load_table_metadata, verify_payloads
from .compositions import verify_compositions
from .config import DEFAULT_CONFIG, ROOT, config_sha256, load_config
from .dual_verify import independent_result
from .hard_states import verify_hard
from .reduction_verify import verify as verify_reductions
from .resources import GIB, check_disk
from .state import FullState, Move, format_word, invert_word


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_smoke() -> dict[str, object]:
    primary = ROOT / "build" / "mdr-verify"
    table_builder = ROOT / "build" / "mdr-table"
    if not primary.is_file() or not table_builder.is_file():
        raise GateError("verifier or table builder is absent")
    self_test = subprocess.run([str(table_builder), "self-test"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if self_test.returncode != 0:
        raise GateError(f"table-builder self-test failed: {self_test.stderr.strip()}")
    word = (Move("U", 1), Move("R", 2), Move("FR", 3), Move("D", 4))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        state_path = root / "state.bin"
        solution_path = root / "solution.txt"
        state_path.write_bytes(FullState.solved().apply(word).encode())
        solution_path.write_bytes(format_word(invert_word(word)))
        expected = independent_result(state_path, solution_path, "solved", len(word))
        command = [str(primary), "verify", "--state", str(state_path), "--solution", str(solution_path), "--target", "solved", "--max-length", str(len(word))]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 or json.loads(completed.stdout) != expected:
            raise GateError("primary and independent verifier smoke test disagree")
    return {
        "primary_sha256": sha256(primary),
        "table_builder_sha256": sha256(table_builder),
        "table_builder_self_test": self_test.stdout.strip().splitlines(),
        "dual_smoke": expected,
    }


def ensure_no_training() -> None:
    for directory in (ROOT / "models", ROOT / "datasets", ROOT / "searches"):
        for path in directory.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                raise GateError(f"pre-training gate found training/search artifact: {path}")


def run(projected_gib: float) -> dict[str, object]:
    config = load_config()
    policy = config["resource_policy"]
    budget = check_disk(
        ROOT,
        policy["minimum_free_gib"],
        projected_gib,
        project_root=ROOT,
        maximum_project_gib=policy["maximum_project_gib"],
    )
    ensure_no_training()
    verifier = primary_smoke()
    table_results: dict[str, object] = {}
    hard_results: dict[str, object] = {}
    hard_paths = {3: "phase3_depth14.bin", 4: "phase4_depth8.bin", 5: "phase5_depth13.bin", 6: "phase6_depth13.bin"}
    for phase in (3, 4, 5, 6):
        metadata_path = ROOT / "tables" / f"phase{phase}" / "metadata.json"
        metadata = load_table_metadata(metadata_path)
        verify_payloads(metadata_path, metadata)
        mismatches = compare_table_controls(metadata, config)
        mismatches.extend(compare_published_histogram(metadata_path, phase))
        if mismatches:
            raise GateError(f"phase {phase} control mismatch: {mismatches}")
        table_results[str(phase)] = {
            "metadata_sha256": sha256(metadata_path),
            "state_count": metadata["state_count"],
            "diameter": metadata["diameter"],
            "antipode_count": metadata["antipode_count"],
        }
        hard_results[str(phase)] = verify_hard(
            ROOT / "hard_states" / hard_paths[phase], ROOT / "tables" / f"phase{phase}"
        )
    pair_specs = {
        "pair34": (hard_paths[3], hard_paths[4]),
        "pair56": (hard_paths[5], hard_paths[6]),
    }
    composition_results: dict[str, object] = {}
    reduction_results: dict[str, object] = {}
    for pair, (hard_a, hard_b) in pair_specs.items():
        composition_results[pair] = verify_compositions(
            ROOT / "compositions" / pair,
            ROOT / "hard_states" / hard_a,
            ROOT / "hard_states" / hard_b,
        )
        reduction_results[pair] = verify_reductions(
            ROOT / "reductions" / pair, ROOT / "compositions" / pair
        )
    upstream = ROOT / "upstream" / "source"
    observed_commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).stdout.strip()
    if observed_commit != config["upstream"]["commit"]:
        raise GateError("upstream checkout drifted")
    pipeline_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).stdout.strip() or None
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready_for_training": True,
        "proof_status": "NOT_PROVED",
        "scope": "M1-M4 complete; model training has not started",
        "config_sha256": config_sha256(DEFAULT_CONFIG),
        "upstream_commit": observed_commit,
        "pipeline_commit": pipeline_commit,
        "verifier": verifier,
        "tables": table_results,
        "hard_states": hard_results,
        "compositions": composition_results,
        "reductions": reduction_results,
        "resources": {
            "project_gib": round(budget.project_bytes / GIB, 3),
            "projected_additional_gib": projected_gib,
            "project_cap_gib": policy["maximum_project_gib"],
            "free_gib": round(budget.free_bytes / GIB, 2),
            "reserve_gib": policy["minimum_free_gib"],
            "maximum_workers": policy["maximum_workers"],
        },
        "complete": True,
    }
    output = ROOT / "reports" / "pretraining-gate.json"
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected-gib", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        report = run(args.projected_gib)
        print(json.dumps({
            "ready_for_training": report["ready_for_training"],
            "proof_status": report["proof_status"],
            "project_gib": report["resources"]["project_gib"],
            "report": str(ROOT / "reports" / "pretraining-gate.json"),
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, GateError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
