from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pair56_problem import (
    PINNED_P900_SHA256,
    STATE_SIZE,
    _compose_gather,
    _inverse,
    _power,
    _sha256,
    _verifier_conjugacy,
)
from .state import FACE_ORDER


K_MAX = 22
SOURCE_FACES = ("U", "R", "F", "L", "BR", "BL", "FR")
TARGET_FACES = ("U", "R", "F", "L", "BR")
EXPECTED_SPACE_SIZE = 208_099_584 * 68_400


class Pair34ProblemError(ValueError):
    pass


def build_problem(source_generator: Path) -> dict[str, Any]:
    source_generator = source_generator.resolve()
    observed_sha256 = _sha256(source_generator)
    if observed_sha256 != PINNED_P900_SHA256:
        raise Pair34ProblemError(f"p900 generator checksum mismatch: {observed_sha256}")
    source = json.loads(source_generator.read_text(encoding="utf-8"))
    try:
        by_name = dict(zip(source["names"], source["actions"], strict=True))
    except (KeyError, TypeError, ValueError) as exc:
        raise Pair34ProblemError("malformed p900 generator") from exc

    identity = list(range(STATE_SIZE))
    for name, action in by_name.items():
        if len(action) != STATE_SIZE or sorted(action) != identity:
            raise Pair34ProblemError(f"invalid source action {name}")
    conjugacy = _verifier_conjugacy(by_name)

    names: list[str] = []
    actions: list[list[int]] = []
    face_ids: list[int] = []
    for face_id, face in enumerate(SOURCE_FACES):
        for verifier_power in range(1, 5):
            names.append(f"{face}{verifier_power}")
            actions.append(_power(by_name[face], 5 - verifier_power))
            face_ids.append(face_id)

    moved_by_target: set[int] = set()
    for face in TARGET_FACES:
        moved_by_target.update(
            position for position, source_position in enumerate(by_name[face])
            if position != source_position
        )
    fixed_positions = [position for position in identity if position not in moved_by_target]
    black_class = len(fixed_positions)
    token_class_map = [black_class] * STATE_SIZE
    for class_id, token_id in enumerate(fixed_positions):
        token_class_map[token_id] = class_id
    target = token_class_map.copy()

    try:
        from sympy.combinatorics import Permutation, PermutationGroup
    except ImportError as exc:
        raise Pair34ProblemError("sympy is required to certify the pair34 quotient") from exc
    source_group = PermutationGroup([Permutation(by_name[face]) for face in SOURCE_FACES])
    target_group = PermutationGroup([Permutation(by_name[face]) for face in TARGET_FACES])
    source_order = int(source_group.order())
    target_order = int(target_group.order())
    stabilizer_order = int(source_group.pointwise_stabilizer(fixed_positions).order())
    if stabilizer_order != target_order:
        raise Pair34ProblemError("collapsed target stabilizer is larger than G7")
    if source_order // target_order != EXPECTED_SPACE_SIZE:
        raise Pair34ProblemError("G5/G7 subgroup index disagrees with regenerated phase tables")

    inverse_actions: list[int] = []
    for face_id in range(len(SOURCE_FACES)):
        start = face_id * 4
        inverse_actions.extend((start + 3, start + 2, start + 1, start))

    problem: dict[str, Any] = {
        "schema_version": 1,
        "pair": "pair34",
        "transition": "G5->G7",
        "metric": "FTM",
        "state_encoding": "p900-S120 with all G7-movable stickers collapsed",
        "state_size": STATE_SIZE,
        "num_classes": black_class + 1,
        "black_class": black_class,
        "source_faces": list(SOURCE_FACES),
        "target_faces": list(TARGET_FACES),
        "names": names,
        "actions": actions,
        "face_ids": face_ids,
        "inverse_actions": inverse_actions,
        "target": target,
        "token_class_map": token_class_map,
        "verifier_conjugacy": conjugacy,
        "p900_power_for_verifier_power": [4, 3, 2, 1],
        "fixed_positions": fixed_positions,
        "black_positions": sorted(moved_by_target),
        "expected_space_size": EXPECTED_SPACE_SIZE,
        "phase_space_factors": [208_099_584, 68_400],
        "source_group_order": source_order,
        "target_group_order": target_order,
        "target_stabilizer_order": stabilizer_order,
        "quotient_certification": "Schreier-Sims orders and pointwise target stabilizer equality",
        "K_max": K_MAX,
        "symmetry_policy": "identity only; the phase-chain rotation stabilizer is trivial",
        "source_generator_sha256": observed_sha256,
        "source_generator_provenance": "megaminx-transformer/generators/p900.json",
        "complete": True,
    }
    validate_problem(problem)
    return problem


def validate_problem(problem: dict[str, Any]) -> None:
    if problem.get("schema_version") != 1 or problem.get("pair") != "pair34":
        raise Pair34ProblemError("unsupported pair34 problem schema")
    if problem.get("metric") != "FTM" or problem.get("K_max") != K_MAX:
        raise Pair34ProblemError("pair34 metric or fixed K_max drifted")
    if problem.get("state_size") != STATE_SIZE or problem.get("expected_space_size") != EXPECTED_SPACE_SIZE:
        raise Pair34ProblemError("pair34 state shape or space size drifted")
    if problem.get("target_stabilizer_order") != problem.get("target_group_order"):
        raise Pair34ProblemError("pair34 target stabilizer is invalid")
    source_order = problem.get("source_group_order")
    target_order = problem.get("target_group_order")
    if not isinstance(source_order, int) or not isinstance(target_order, int):
        raise Pair34ProblemError("pair34 subgroup orders are absent")
    if source_order % target_order or source_order // target_order != EXPECTED_SPACE_SIZE:
        raise Pair34ProblemError("pair34 subgroup index is invalid")
    names = problem.get("names")
    actions = problem.get("actions")
    face_ids = problem.get("face_ids")
    inverses = problem.get("inverse_actions")
    target = problem.get("target")
    if not all(isinstance(value, list) for value in (names, actions, face_ids, inverses, target)):
        raise Pair34ProblemError("pair34 problem arrays are absent")
    if len(names) != 28 or len(actions) != 28 or len(face_ids) != 28 or len(inverses) != 28:
        raise Pair34ProblemError("pair34 must have exactly 28 FTM actions")
    identity = list(range(STATE_SIZE))
    for index, action in enumerate(actions):
        if len(action) != STATE_SIZE or sorted(action) != identity:
            raise Pair34ProblemError(f"invalid pair34 action {index}")
        if _compose_gather(action, actions[inverses[index]]) != identity:
            raise Pair34ProblemError(f"invalid pair34 inverse {index}")
    if len(target) != STATE_SIZE or len(set(target)) != problem.get("num_classes"):
        raise Pair34ProblemError("invalid pair34 target")
    conjugacy = problem.get("verifier_conjugacy")
    if not isinstance(conjugacy, list) or sorted(conjugacy) != identity:
        raise Pair34ProblemError("FullStateV1 conjugacy is absent")
    target_actions = len(TARGET_FACES) * 4
    for action in actions[:target_actions]:
        if [target[index] for index in action] != target:
            raise Pair34ProblemError("target is not invariant under G7")
    if all([target[index] for index in action] == target for action in actions[target_actions:]):
        raise Pair34ProblemError("source generators do not leave the G7 quotient")


def load_problem(path: Path) -> dict[str, Any]:
    problem = json.loads(path.read_text(encoding="utf-8"))
    validate_problem(problem)
    return problem


def write_problem(source_generator: Path, output: Path) -> dict[str, Any]:
    problem = build_problem(source_generator)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(problem, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return problem
