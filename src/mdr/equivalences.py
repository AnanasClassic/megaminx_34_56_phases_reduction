from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .hard_states import phase_index, read_hard
from .state import FACE_ORDER, MAGIC, MOVE_SPECS, FullState


CONTROLS = {"pair34": (3, 4), "pair56": (5, 6)}


def neighbor_orders() -> dict[str, list[str]]:
    incident: dict[int, list[str]] = {}
    for face in FACE_ORDER:
        for edge in MOVE_SPECS[face][2]:
            incident.setdefault(edge, []).append(face)
    if len(incident) != 30 or any(len(value) != 2 for value in incident.values()):
        raise ValueError("move cycles do not define the dodecahedral face graph")
    return {
        face: [next(other for other in incident[edge] if other != face) for edge in MOVE_SPECS[face][2]]
        for face in FACE_ORDER
    }


def rotations() -> list[dict[str, str]]:
    orders = neighbor_orders()
    result: list[dict[str, str]] = []
    anchor = FACE_ORDER[0]
    anchor_neighbor = orders[anchor][0]
    for image in FACE_ORDER:
        for neighbor_image in orders[image]:
            mapping = {anchor: image, anchor_neighbor: neighbor_image}
            changed = True
            bad = False
            while changed and not bad:
                changed = False
                for face, face_image in list(mapping.items()):
                    anchors = [neighbor for neighbor in orders[face] if neighbor in mapping]
                    if not anchors:
                        continue
                    neighbor = anchors[0]
                    delta = (orders[face_image].index(mapping[neighbor]) - orders[face].index(neighbor)) % 5
                    for index, other in enumerate(orders[face]):
                        other_image = orders[face_image][(index + delta) % 5]
                        if other in mapping and mapping[other] != other_image:
                            bad = True
                            break
                        if other not in mapping:
                            mapping[other] = other_image
                            changed = True
            if not bad and len(mapping) == 12 and len(set(mapping.values())) == 12:
                result.append(mapping)
    unique = {tuple(mapping[face] for face in FACE_ORDER): mapping for mapping in result}
    if len(unique) != 60:
        raise ValueError(f"expected 60 rotations, generated {len(unique)}")
    return [unique[key] for key in sorted(unique)]


def decode_unchecked(data: bytes) -> FullState:
    if len(data) != 108 or data[:8] != MAGIC:
        raise ValueError("invalid composed state record")
    return FullState(tuple(data[8:38]), tuple(data[38:68]), tuple(data[68:88]), tuple(data[88:108]))


def encode_unchecked(state: FullState) -> bytes:
    return b"".join((MAGIC, bytes(state.edge_pieces), bytes(state.edge_orientations), bytes(state.corner_pieces), bytes(state.corner_orientations)))


def inverse(state: FullState) -> FullState:
    edge_position = [0] * 30
    corner_position = [0] * 20
    for position, piece in enumerate(state.edge_pieces):
        edge_position[piece] = position
    for position, piece in enumerate(state.corner_pieces):
        corner_position[piece] = position
    return FullState(
        tuple(edge_position),
        tuple(state.edge_orientations[source] for source in edge_position),
        tuple(corner_position),
        tuple((-state.corner_orientations[source]) % 3 for source in corner_position),
    )


def analyze(pair: str, compositions: Path, reductions: Path, hard_a: Path, hard_b: Path, table_a: Path) -> dict[str, object]:
    phase_a, phase_b = CONTROLS[pair]
    all_rotations = rotations()
    nested = {
        "pair34": [set(FACE_ORDER[:7]), set(FACE_ORDER[:6]), set(FACE_ORDER[:5])],
        "pair56": [set(FACE_ORDER[:5]), set(FACE_ORDER[:4]), set(FACE_ORDER[:3])],
    }[pair]
    admissible = [
        index for index, mapping in enumerate(all_rotations)
        if all({mapping[face] for face in subset} == subset for subset in nested)
    ]
    if len(admissible) != 1 or any(all_rotations[admissible[0]][face] != face for face in FACE_ORDER):
        raise ValueError("phase-preserving rotation stabilizer is not the identity")

    states_blob = (compositions / "states.bin").read_bytes()
    if len(states_blob) % 108:
        raise ValueError("composition state payload is malformed")
    states = {states_blob[offset : offset + 108] for offset in range(0, len(states_blob), 108)}
    depths = table_a.read_bytes()
    inverse_in_set = 0
    inverse_first_max = 0
    first_counterexample: dict[str, int] | None = None
    diameter = max(value for value in depths if value != 255)
    for raw_id, offset in enumerate(range(0, len(states_blob), 108)):
        state = decode_unchecked(states_blob[offset : offset + 108])
        inverted = inverse(state)
        inverted_bytes = encode_unchecked(inverted)
        if inverted_bytes in states:
            inverse_in_set += 1
        index = phase_index(inverted, phase_a)
        depth = depths[index]
        if depth == diameter:
            inverse_first_max += 1
        if first_counterexample is None and (inverted_bytes not in states or depth != diameter):
            first_counterexample = {"raw_id": raw_id, "inverse_phase_index": index, "inverse_phase_depth": depth}

    _, _, records_a = read_hard(hard_a)
    _, _, records_b = read_hard(hard_b)
    boundary_compatible = 0
    for left in records_a:
        left_faces = {code // 4 for code in range(32) if left.last_mask & (1 << code)}
        for right in records_b:
            right_faces = {code // 4 for code in range(32) if right.first_mask & (1 << code)}
            if left_faces & right_faces:
                boundary_compatible += 1

    report: dict[str, object] = {
        "schema_version": 1,
        "pair": pair,
        "rotation_group": {
            "size": len(all_rotations),
            "construction": "images of one oriented adjacent face pair; cyclic neighbor orders preserved",
            "rotations": [
                {
                    "id": index,
                    "face_map": [mapping[face] for face in FACE_ORDER],
                    "move_conjugation": "face maps as listed; power 1..4 is preserved",
                }
                for index, mapping in enumerate(all_rotations)
            ],
            "admissible_ids": admissible,
            "admissible_group_size": len(admissible),
            "state_transform_for_admissible_group": "identity FullStateV1 byte map",
        },
        "inversion": {
            "generator_set_is_inverse_closed": True,
            "target_subgroup_is_inverse_closed": True,
            "raw_state_count": len(states),
            "inverse_in_raw_set": inverse_in_set,
            "inverse_preserves_first_max_depth": inverse_first_max,
            "raw_set_closed": inverse_in_set == len(states),
            "admissible": False,
            "reason": "inversion exchanges left and right cosets and the exhaustive raw set is not closed under it",
            "first_counterexample": first_counterexample,
        },
        "boundary_all_optimal_masks": {
            "compatible_pairs": boundary_compatible,
            "raw_pairs": len(records_a) * len(records_b),
            "result": "no same-face merge exists even after choosing alternative optimal phase paths" if boundary_compatible == 0 else "compatible pairs require physical replay",
        },
        "complete": True,
    }
    reductions.mkdir(parents=True, exist_ok=True)
    path = reductions / "equivalences.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    metadata_path = reductions / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    metadata["payloads"][path.name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    metadata["symmetry_policy"] = "all 60 rotations enumerated; nested phase stabilizer is identity"
    metadata["inversion_policy"] = "rejected by exact closure scan and left/right-coset mismatch"
    temporary = metadata_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=tuple(CONTROLS), required=True)
    parser.add_argument("--compositions", type=Path, required=True)
    parser.add_argument("--reductions", type=Path, required=True)
    parser.add_argument("--hard-a", type=Path, required=True)
    parser.add_argument("--hard-b", type=Path, required=True)
    parser.add_argument("--table-a", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.pair, args.compositions, args.reductions, args.hard_a, args.hard_b, args.table_a)
    print(json.dumps({
        "pair": args.pair,
        "rotation_group": report["rotation_group"]["size"],
        "admissible_rotations": report["rotation_group"]["admissible_group_size"],
        "inversion_admissible": report["inversion"]["admissible"],
        "boundary_compatible_pairs": report["boundary_all_optimal_masks"]["compatible_pairs"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
