from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .state import FACE_ORDER, FullState, Move


STATE_SIZE = 120
K_MAX = 26
SOURCE_FACES = ("U", "R", "F", "L", "BR")
TARGET_FACES = ("U", "R", "F")
EXPECTED_SPACE_SIZE = 1_664_617_163_489_280
PINNED_P900_SHA256 = "3e38e75ee4f3387c33917393068b2fadf7959b3490f86d6e924f266960f45dbd"


class Pair56ProblemError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_gather(first: list[int], second: list[int]) -> list[int]:
    """Permutation for applying ``first`` and then ``second`` to a gather-state."""
    return [first[index] for index in second]


def _power(base: list[int], power: int) -> list[int]:
    result = list(range(len(base)))
    for _ in range(power):
        result = _compose_gather(result, base)
    return result


def _inverse(action: list[int]) -> list[int]:
    result = [0] * len(action)
    for source, destination in enumerate(action):
        result[destination] = source
    return result


def _full_sticker_action(face: str) -> list[int]:
    state = FullState.solved().apply((Move(face, 1),))
    corners = [
        3 * piece + (slot + orientation) % 3
        for piece, orientation in zip(state.corner_pieces, state.corner_orientations)
        for slot in range(3)
    ]
    edges = [
        60 + 2 * piece + (slot + orientation) % 2
        for piece, orientation in zip(state.edge_pieces, state.edge_orientations)
        for slot in range(2)
    ]
    return corners + edges


def full_state_to_pair56(state: FullState, problem: dict[str, Any]) -> list[int]:
    """Convert a physical FullStateV1 representative to the certified G7/G9 coloring."""
    full_tokens = [
        3 * piece + (slot + orientation) % 3
        for piece, orientation in zip(state.corner_pieces, state.corner_orientations)
        for slot in range(3)
    ] + [
        60 + 2 * piece + (slot + orientation) % 2
        for piece, orientation in zip(state.edge_pieces, state.edge_orientations)
        for slot in range(2)
    ]
    conjugacy = problem["verifier_conjugacy"]
    token_classes = problem["token_class_map"]
    result = [problem["black_class"]] * STATE_SIZE
    for full_position, full_token in enumerate(full_tokens):
        p900_position = conjugacy[full_position]
        p900_token = conjugacy[full_token]
        result[p900_position] = token_classes[p900_token]
    return result


def _propagate_conjugacy(
    generators: list[tuple[list[int], list[int]]], seed: int, target: int,
) -> dict[int, int] | None:
    mapping = {seed: target}
    reverse = {target: seed}
    queue = [seed]
    while queue:
        source = queue.pop()
        image = mapping[source]
        for left, right in generators:
            next_source = left[source]
            next_image = right[image]
            if next_source in mapping and mapping[next_source] != next_image:
                return None
            if next_image in reverse and reverse[next_image] != next_source:
                return None
            if next_source not in mapping:
                mapping[next_source] = next_image
                reverse[next_image] = next_source
                queue.append(next_source)
    return mapping


def _verifier_conjugacy(by_name: dict[str, list[int]]) -> list[int]:
    generator_pairs: list[tuple[list[int], list[int]]] = []
    for face in FACE_ORDER:
        full = _full_sticker_action(face)
        # The pinned p900 clockwise convention is the inverse of FullStateV1.
        p900_for_verifier = _inverse(by_name[face])
        generator_pairs.extend(((full, p900_for_verifier), (_inverse(full), by_name[face])))
    combined: dict[int, int] = {}
    for seed, candidates in ((0, range(60)), (60, range(60, 120))):
        orbit_mapping = None
        for candidate in candidates:
            observed = _propagate_conjugacy(generator_pairs, seed, candidate)
            if observed is not None and len(observed) == 60:
                orbit_mapping = observed
                break
        if orbit_mapping is None:
            raise Pair56ProblemError("p900 moves are not conjugate to FullStateV1 moves")
        combined.update(orbit_mapping)
    if sorted(combined) != list(range(STATE_SIZE)) or sorted(combined.values()) != list(range(STATE_SIZE)):
        raise Pair56ProblemError("invalid p900/FullStateV1 conjugacy")
    return [combined[index] for index in range(STATE_SIZE)]


def build_problem(source_generator: Path) -> dict[str, Any]:
    source_generator = source_generator.resolve()
    observed_sha256 = _sha256(source_generator)
    if observed_sha256 != PINNED_P900_SHA256:
        raise Pair56ProblemError(
            f"p900 generator checksum mismatch: {observed_sha256}"
        )
    source = json.loads(source_generator.read_text(encoding="utf-8"))
    try:
        by_name = dict(zip(source["names"], source["actions"], strict=True))
    except (KeyError, TypeError, ValueError) as exc:
        raise Pair56ProblemError("malformed p900 generator") from exc

    identity = list(range(STATE_SIZE))
    for name, action in by_name.items():
        if len(action) != STATE_SIZE or sorted(action) != identity:
            raise Pair56ProblemError(f"invalid source action {name}")

    verifier_conjugacy = _verifier_conjugacy(by_name)
    names: list[str] = []
    actions: list[list[int]] = []
    face_ids: list[int] = []
    for face_id, face in enumerate(SOURCE_FACES):
        if face not in by_name:
            raise Pair56ProblemError(f"source generator lacks {face}")
        base = by_name[face]
        for verifier_power in range(1, 5):
            names.append(f"{face}{verifier_power}")
            actions.append(_power(base, 5 - verifier_power))
            face_ids.append(face_id)

    moved_by_target: set[int] = set()
    for face in TARGET_FACES:
        moved_by_target.update(
            index for index, source_index in enumerate(by_name[face])
            if index != source_index
        )
    fixed_positions = [index for index in identity if index not in moved_by_target]
    black_class = len(fixed_positions)
    token_class_map = [black_class] * STATE_SIZE
    for class_id, token_id in enumerate(fixed_positions):
        token_class_map[token_id] = class_id
    target = token_class_map.copy()

    try:
        from sympy.combinatorics import Permutation, PermutationGroup
    except ImportError as exc:
        raise Pair56ProblemError("sympy is required to certify the pair56 quotient") from exc
    source_group = PermutationGroup([Permutation(by_name[face]) for face in SOURCE_FACES])
    target_group = PermutationGroup([Permutation(by_name[face]) for face in TARGET_FACES])
    source_group_order = int(source_group.order())
    target_group_order = int(target_group.order())
    target_stabilizer_order = int(source_group.pointwise_stabilizer(fixed_positions).order())
    if target_stabilizer_order != target_group_order:
        raise Pair56ProblemError("collapsed target stabilizer is larger than G9")
    if source_group_order // target_group_order != EXPECTED_SPACE_SIZE:
        raise Pair56ProblemError("p900 subgroup index disagrees with regenerated phase tables")

    inverse_actions = []
    for face_id in range(len(SOURCE_FACES)):
        start = face_id * 4
        inverse_actions.extend((start + 3, start + 2, start + 1, start))

    problem: dict[str, Any] = {
        "schema_version": 1,
        "pair": "pair56",
        "transition": "G7->G9",
        "metric": "FTM",
        "state_encoding": "p900-S120 with all G9-movable stickers collapsed",
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
        "verifier_conjugacy": verifier_conjugacy,
        "p900_power_for_verifier_power": [4, 3, 2, 1],
        "fixed_positions": fixed_positions,
        "black_positions": sorted(moved_by_target),
        "expected_space_size": EXPECTED_SPACE_SIZE,
        "phase_space_factors": [64_157_184, 25_945_920],
        "source_group_order": source_group_order,
        "target_group_order": target_group_order,
        "target_stabilizer_order": target_stabilizer_order,
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
    if problem.get("schema_version") != 1 or problem.get("pair") != "pair56":
        raise Pair56ProblemError("unsupported pair56 problem schema")
    if problem.get("metric") != "FTM" or problem.get("K_max") != K_MAX:
        raise Pair56ProblemError("pair56 metric or fixed K_max drifted")
    if problem.get("state_size") != STATE_SIZE or problem.get("num_classes") != 67:
        raise Pair56ProblemError("pair56 state shape drifted")
    if problem.get("expected_space_size") != EXPECTED_SPACE_SIZE:
        raise Pair56ProblemError("pair56 space size drifted")
    source_order = problem.get("source_group_order")
    target_order = problem.get("target_group_order")
    if not isinstance(source_order, int) or not isinstance(target_order, int):
        raise Pair56ProblemError("pair56 subgroup orders are absent")
    if source_order // target_order != EXPECTED_SPACE_SIZE or source_order % target_order:
        raise Pair56ProblemError("pair56 subgroup index is invalid")
    if problem.get("target_stabilizer_order") != target_order:
        raise Pair56ProblemError("pair56 mask has the wrong stabilizer")
    names = problem.get("names")
    actions = problem.get("actions")
    face_ids = problem.get("face_ids")
    inverses = problem.get("inverse_actions")
    target = problem.get("target")
    if not all(isinstance(value, list) for value in (names, actions, face_ids, inverses, target)):
        raise Pair56ProblemError("pair56 problem arrays are absent")
    if len(names) != 20 or len(actions) != 20 or len(face_ids) != 20 or len(inverses) != 20:
        raise Pair56ProblemError("pair56 must have exactly 20 FTM actions")
    identity = list(range(STATE_SIZE))
    for index, action in enumerate(actions):
        if len(action) != STATE_SIZE or sorted(action) != identity:
            raise Pair56ProblemError(f"invalid action {index}")
        inverse = actions[inverses[index]]
        if _compose_gather(action, inverse) != identity:
            raise Pair56ProblemError(f"invalid inverse for action {index}")
    if len(target) != STATE_SIZE or min(target) != 0 or max(target) != 66:
        raise Pair56ProblemError("invalid pair56 target")
    if len(set(target)) != 67 or len(problem.get("black_positions", [])) != 54:
        raise Pair56ProblemError("pair56 target class collapse drifted")
    conjugacy = problem.get("verifier_conjugacy")
    if not isinstance(conjugacy, list) or sorted(conjugacy) != identity:
        raise Pair56ProblemError("FullStateV1 conjugacy is absent")
    for face_id, face in enumerate(SOURCE_FACES):
        full_action = _full_sticker_action(face)
        verifier_action = actions[face_id * 4]
        if any(conjugacy[full_action[index]] != verifier_action[conjugacy[index]] for index in identity):
            raise Pair56ProblemError(f"action {face}1 disagrees with FullStateV1")
    target_face_count = len(problem.get("target_faces", []))
    for action in actions[: target_face_count * 4]:
        if [target[index] for index in action] != target:
            raise Pair56ProblemError("target is not invariant under G9")
    if all([target[index] for index in action] == target for action in actions[target_face_count * 4 :]):
        raise Pair56ProblemError("source generators do not leave the G9 quotient")


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
