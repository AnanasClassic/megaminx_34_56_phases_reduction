from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ROOT
from .pair34_problem import load_problem as load_pair34_problem
from .pair56_problem import load_problem as load_pair56_problem
from .state import FACE_ORDER


class PhaseGroupAuditError(ValueError):
    pass


CONTROLS = {
    "pair34": {
        "groups": (("G5", 7), ("G6", 6), ("G7", 5)),
        "factors": (208_099_584, 68_400),
        "loader": load_pair34_problem,
    },
    "pair56": {
        "groups": (("G7", 5), ("G8", 4), ("G9", 3)),
        "factors": (64_157_184, 25_945_920),
        "loader": load_pair56_problem,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_problem(pair: str, problem_path: Path) -> dict[str, Any]:
    try:
        control = CONTROLS[pair]
    except KeyError as exc:
        raise PhaseGroupAuditError(f"unsupported pair {pair!r}") from exc
    try:
        import sympy
        from sympy.combinatorics import Permutation, PermutationGroup
    except ImportError as exc:
        raise PhaseGroupAuditError(
            "SymPy is required for the exact phase-group audit; install requirements-proof.txt"
        ) from exc

    problem_path = problem_path.resolve()
    problem = control["loader"](problem_path)
    groups = control["groups"]
    factors = control["factors"]
    source_face_count = groups[0][1]
    target_face_count = groups[-1][1]
    expected_source_faces = list(FACE_ORDER[:source_face_count])
    expected_target_faces = list(FACE_ORDER[:target_face_count])
    if problem.get("source_faces") != expected_source_faces:
        raise PhaseGroupAuditError(f"{pair} source-face prefix drifted")
    if problem.get("target_faces") != expected_target_faces:
        raise PhaseGroupAuditError(f"{pair} target-face prefix drifted")
    expected_names = [
        f"{face}{power}"
        for face in expected_source_faces
        for power in range(1, 5)
    ]
    if problem.get("names") != expected_names:
        raise PhaseGroupAuditError(f"{pair} action order drifted")
    actions = problem["actions"]
    state_size = problem["state_size"]
    identity = list(range(state_size))
    if len(actions) != 4 * source_face_count or any(
        len(action) != state_size or sorted(action) != identity
        for action in actions
    ):
        raise PhaseGroupAuditError(f"{pair} action payload is not a permutation family")

    base_generators = [Permutation(actions[4 * face]) for face in range(source_face_count)]
    exact_groups = {
        name: PermutationGroup(base_generators[:face_count])
        for name, face_count in groups
    }
    orders = {name: int(exact_groups[name].order()) for name, _ in groups}
    observed_factors: list[int] = []
    transitions: list[dict[str, Any]] = []
    for index, expected_factor in enumerate(factors):
        source_name = groups[index][0]
        target_name = groups[index + 1][0]
        source_order = orders[source_name]
        target_order = orders[target_name]
        if source_order % target_order:
            raise PhaseGroupAuditError(
                f"{pair} {target_name} order does not divide {source_name} order"
            )
        observed_factor = source_order // target_order
        observed_factors.append(observed_factor)
        if observed_factor != expected_factor:
            raise PhaseGroupAuditError(
                f"{pair} {source_name}/{target_name} index {observed_factor} != {expected_factor}"
            )
        transitions.append({
            "transition": f"{source_name}->{target_name}",
            "source_order": source_order,
            "target_order": target_order,
            "index": observed_factor,
            "expected_index": expected_factor,
        })

    source_name = groups[0][0]
    target_name = groups[-1][0]
    source_order = orders[source_name]
    target_order = orders[target_name]
    combined_index = source_order // target_order
    expected_combined = factors[0] * factors[1]
    if combined_index != expected_combined:
        raise PhaseGroupAuditError(f"{pair} combined subgroup index drifted")
    if problem.get("phase_space_factors") != list(factors):
        raise PhaseGroupAuditError(f"{pair} stored phase factors drifted")
    if problem.get("expected_space_size") != expected_combined:
        raise PhaseGroupAuditError(f"{pair} stored combined index drifted")
    if problem.get("source_group_order") != source_order:
        raise PhaseGroupAuditError(f"{pair} stored source-group order drifted")
    if problem.get("target_group_order") != target_order:
        raise PhaseGroupAuditError(f"{pair} stored target-group order drifted")

    fixed_positions = problem.get("fixed_positions")
    if not isinstance(fixed_positions, list) or any(
        not isinstance(position, int) or position < 0 or position >= state_size
        for position in fixed_positions
    ) or len(set(fixed_positions)) != len(fixed_positions):
        raise PhaseGroupAuditError(f"{pair} fixed-position set is malformed")
    stabilizer_order = int(
        exact_groups[source_name].pointwise_stabilizer(fixed_positions).order()
    )
    if stabilizer_order != target_order:
        raise PhaseGroupAuditError(
            f"{pair} target coloring has a larger pointwise stabilizer than {target_name}"
        )
    if problem.get("target_stabilizer_order") != stabilizer_order:
        raise PhaseGroupAuditError(f"{pair} stored target-stabilizer order drifted")

    return {
        "schema_version": 1,
        "pair": pair,
        "problem": f"training/{pair}/problem.json",
        "problem_sha256": _sha256(problem_path),
        "sympy_version": sympy.__version__,
        "groups": {
            name: {"face_count": face_count, "order": orders[name]}
            for name, face_count in groups
        },
        "transitions": transitions,
        "combined_index": combined_index,
        "expected_combined_index": expected_combined,
        "target_pointwise_stabilizer_order": stabilizer_order,
        "target_group_order": target_order,
        "full_state_conjugacy": "validated for every committed FTM action",
        "valid": True,
    }


def audit_all(root: Path = ROOT) -> dict[str, Any]:
    reports = {
        pair: audit_problem(pair, root / "training" / pair / "problem.json")
        for pair in CONTROLS
    }
    pair34_g7 = reports["pair34"]["groups"]["G7"]["order"]
    pair56_g7 = reports["pair56"]["groups"]["G7"]["order"]
    if pair34_g7 != pair56_g7:
        raise PhaseGroupAuditError("pair34 and pair56 disagree on the shared G7 order")
    return {
        "schema_version": 1,
        "pairs": reports,
        "shared_G7_order": pair34_g7,
        "valid": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=("pair34", "pair56", "all"), default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.pair == "all":
            result = audit_all(args.root)
        else:
            result = audit_problem(
                args.pair, args.root / "training" / args.pair / "problem.json"
            )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, PhaseGroupAuditError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
