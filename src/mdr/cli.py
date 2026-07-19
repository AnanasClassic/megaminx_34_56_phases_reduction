from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from .artifacts import ArtifactError, compare_published_histogram, compare_table_controls, load_table_metadata, verify_payloads
from .config import DEFAULT_CONFIG, ROOT, ConfigError, config_sha256, load_config
from .proof_certify import ProofError, certify_all, certify_pair
from .resources import GIB, ResourceError, check_disk


NOT_IMPLEMENTED = {
    "verify": "use scripts/verify for the implemented dual verifier",
    "build-tables": "use scripts/build_tables for the implemented M2 builder",
    "extract-antipodes": "use scripts/extract_antipodes for the implemented M2 extractor",
    "compose-pairs": "use scripts/compose_pairs for the implemented M3 composer",
    "canonicalize": "use scripts/canonicalize for the implemented M4 reducer",
    "pretraining-gate": "use scripts/pretraining_gate for the implemented exhaustive gate",
    "train": "use scripts/train_pair56 for the implemented combined 5+6 Q-MLP trainer",
    "solve": "model-guided search is not implemented and cannot provide proof coverage",
}


def _command_version(command: str, args: list[str]) -> str | None:
    if shutil.which(command) is None:
        return None
    result = subprocess.run([command, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else f"exit={result.returncode}"


def cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(json.dumps({"valid": True, "config_sha256": config_sha256(args.config), "phases": sorted(config["phases"])}, sort_keys=True))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source = ROOT / "upstream" / "source"
    expected_commit = config["upstream"]["commit"]
    observed_commit = None
    if (source / ".git").is_dir():
        observed_commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.strip()
    policy = config["resource_policy"]
    budget = check_disk(
        ROOT, policy["minimum_free_gib"], project_root=ROOT,
        maximum_project_gib=policy["maximum_project_gib"],
    )
    report = {
        "config_valid": True,
        "config_sha256": config_sha256(args.config),
        "commands": {
            "docker": _command_version("docker", ["--version"]),
            "git": _command_version("git", ["--version"]),
            "python3": _command_version("python3", ["--version"]),
            "7z": _command_version("7z", ["i"]),
        },
        "upstream_expected_commit": expected_commit,
        "upstream_observed_commit": observed_commit,
        "upstream_ready": observed_commit == expected_commit,
        "disk_free_gib": round(budget.free_bytes / GIB, 2),
        "disk_reserve_gib": round(budget.reserve_bytes / GIB, 2),
        "project_size_gib": round(budget.project_bytes / GIB, 3),
        "project_size_cap_gib": round(budget.maximum_project_bytes / GIB, 2),
        "maximum_workers": policy["maximum_workers"],
        "proof_status": "NOT_PROVED",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["upstream_ready"] or any(value is None for value in report["commands"].values()):
        return 1
    return 0


def cmd_check_disk(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    minimum = args.minimum_free_gib
    if minimum is None:
        minimum = config["resource_policy"]["minimum_free_gib"]
    policy = config["resource_policy"]
    budget = check_disk(
        args.path, minimum, args.projected_gib, project_root=ROOT,
        maximum_project_gib=policy["maximum_project_gib"],
    )
    print(json.dumps({
        "ok": True,
        "path": str(budget.path),
        "free_gib": round(budget.free_bytes / GIB, 2),
        "projected_gib": args.projected_gib,
        "reserve_gib": minimum,
        "project_gib": round(budget.project_bytes / GIB, 3),
        "projected_project_gib": round(budget.projected_project_bytes / GIB, 3),
        "project_cap_gib": policy["maximum_project_gib"],
    }, sort_keys=True))
    return 0


def cmd_validate_table(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    metadata = load_table_metadata(args.metadata)
    verify_payloads(args.metadata, metadata)
    mismatches = compare_table_controls(metadata, config)
    mismatches.extend(compare_published_histogram(args.metadata, metadata["phase"]))
    print(json.dumps({"valid": True, "control_mismatches": mismatches}, indent=2, sort_keys=True))
    return 3 if mismatches else 0


def cmd_unavailable(args: argparse.Namespace) -> int:
    print(NOT_IMPLEMENTED[args.command], file=sys.stderr)
    return 2


def cmd_certify(args: argparse.Namespace) -> int:
    if args.pair == "all":
        if args.max_length is not None or args.database is not None:
            raise ProofError("--max-length and --database require pair34 or pair56")
        result = certify_all(
            root=args.root, config_path=args.config,
            pair34_database=args.pair34_database,
            pair56_database=args.pair56_database,
            go_verifier=args.go_verifier,
        )
    else:
        if args.pair34_database is not None or args.pair56_database is not None:
            raise ProofError("pair-specific database flags require 'certify all'")
        result = certify_pair(
            args.pair, root=args.root, config_path=args.config,
            database=args.database, max_length=args.max_length,
            go_verifier=args.go_verifier,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="mdr")
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config").set_defaults(func=cmd_validate_config)
    commands.add_parser("doctor").set_defaults(func=cmd_doctor)
    disk = commands.add_parser("check-disk")
    disk.add_argument("--path", type=Path, default=ROOT)
    disk.add_argument("--minimum-free-gib", type=float)
    disk.add_argument("--projected-gib", type=float, default=0)
    disk.set_defaults(func=cmd_check_disk)
    table = commands.add_parser("validate-table")
    table.add_argument("metadata", type=Path)
    table.set_defaults(func=cmd_validate_table)
    certify = commands.add_parser("certify")
    certify.add_argument("pair", choices=("pair34", "pair56", "all"))
    certify.add_argument("--root", type=Path, default=ROOT)
    certify.add_argument("--max-length", type=int)
    certify.add_argument("--database", type=Path)
    certify.add_argument("--pair34-database", type=Path)
    certify.add_argument("--pair56-database", type=Path)
    certify.add_argument(
        "--go-verifier", type=Path,
        help="also stream every direct word through the independent Go FullStateV1 verifier",
    )
    certify.add_argument("--report", type=Path)
    certify.set_defaults(func=cmd_certify)
    for name in NOT_IMPLEMENTED:
        unavailable = commands.add_parser(name)
        unavailable.add_argument("arguments", nargs=argparse.REMAINDER)
        unavailable.set_defaults(func=cmd_unavailable)
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args, unknown = argument_parser.parse_known_args(argv)
    if unknown and args.command not in NOT_IMPLEMENTED:
        argument_parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    try:
        return args.func(args)
    except (
        ConfigError, ArtifactError, ProofError, ResourceError, OSError,
        ValueError, sqlite3.Error, json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
