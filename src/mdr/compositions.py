from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from .hard_states import HardStateError, apply_code, phase_index, read_hard
from .state import FullState


class CompositionError(ValueError):
    pass


CONTROLS = {"pair34": (3, 4, 536572), "pair56": (5, 6, 407628)}


def compose(left: FullState, right: FullState) -> FullState:
    """Apply the group element right after left, using position records."""
    edge_pieces = tuple(left.edge_pieces[source] for source in right.edge_pieces)
    edge_orientations = tuple(
        left.edge_orientations[source] ^ right.edge_orientations[destination]
        for destination, source in enumerate(right.edge_pieces)
    )
    corner_pieces = tuple(left.corner_pieces[source] for source in right.corner_pieces)
    corner_orientations = tuple(
        (left.corner_orientations[source] + right.corner_orientations[destination]) % 3
        for destination, source in enumerate(right.corner_pieces)
    )
    return FullState(edge_pieces, edge_orientations, corner_pieces, corner_orientations)


def _verify_payloads(root: Path, metadata: dict[str, object]) -> None:
    for name, spec in metadata["payloads"].items():
        path = root / name
        data = path.read_bytes()
        if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise CompositionError(f"payload mismatch: {name}")


def verify_compositions(root: Path, hard_a: Path, hard_b: Path) -> dict[str, int | bool | str]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    pair = metadata.get("pair")
    if pair not in CONTROLS or metadata.get("complete") is not True:
        raise CompositionError("incomplete or unknown composition manifest")
    phase_a, phase_b, expected_raw = CONTROLS[pair]
    observed_phase_a, _, records_a = read_hard(hard_a)
    observed_phase_b, _, records_b = read_hard(hard_b)
    if (observed_phase_a, observed_phase_b) != (phase_a, phase_b):
        raise CompositionError("hard-state phase order mismatch")
    raw_count = len(records_a) * len(records_b)
    if raw_count != expected_raw or metadata.get("raw_pairs") != expected_raw:
        raise CompositionError("raw Cartesian product mismatch")
    _verify_payloads(root, metadata)
    states = (root / "states.bin").read_bytes()
    pairs = (root / "pair_indices.bin").read_bytes()
    mapping = (root / "raw_to_unique.bin").read_bytes()
    unique = (root / "unique_states.bin").read_bytes()
    if len(states) != raw_count * 108 or len(pairs) != raw_count * 8 or len(mapping) != raw_count * 4 or len(unique) % 108:
        raise CompositionError("composition payload length mismatch")
    multiplicity = [0] * (len(unique) // 108)
    seen_unique: set[bytes] = set()
    for unique_id in range(len(multiplicity)):
        state = unique[unique_id * 108 : (unique_id + 1) * 108]
        if state in seen_unique:
            raise CompositionError(f"duplicate unique state {unique_id}")
        seen_unique.add(state)
    raw_id = 0
    for record_a in records_a:
        for record_b in records_b:
            expected = compose(record_b.state, record_a.state)
            encoded = expected.encode()
            observed = states[raw_id * 108 : (raw_id + 1) * 108]
            if observed != encoded:
                raise CompositionError(f"physical composition mismatch at raw pair {raw_id}")
            index_a, index_b = struct.unpack_from("<II", pairs, raw_id * 8)
            if (index_a, index_b) != (record_a.index, record_b.index):
                raise CompositionError(f"coordinate provenance mismatch at raw pair {raw_id}")
            if phase_index(expected, phase_a) != record_a.index:
                raise CompositionError(f"first coordinate round-trip mismatch at raw pair {raw_id}")
            after_a = expected
            for code in record_a.solution:
                after_a = apply_code(after_a, code)
            if after_a != record_b.state or phase_index(after_a, phase_b) != record_b.index:
                raise CompositionError(f"second coordinate round-trip mismatch at raw pair {raw_id}")
            unique_id = struct.unpack_from("<I", mapping, raw_id * 4)[0]
            if unique_id >= len(multiplicity) or unique[unique_id * 108 : (unique_id + 1) * 108] != observed:
                raise CompositionError(f"raw-to-unique mapping mismatch at raw pair {raw_id}")
            multiplicity[unique_id] += 1
            raw_id += 1
    statistics = json.loads((root / "statistics.json").read_text(encoding="utf-8"))
    observed_stats = {
        "raw_pairs": raw_count,
        "valid_pairs": raw_count,
        "unique_full_states": len(multiplicity),
        "duplicate_pairs": raw_count - len(multiplicity),
        "maximum_multiplicity": max(multiplicity),
    }
    for key, value in observed_stats.items():
        if statistics.get(key) != value or metadata.get(key) != value and key in metadata:
            raise CompositionError(f"statistics mismatch for {key}")
    return {"valid": True, "pair": pair, **observed_stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hard-a", type=Path, required=True)
    parser.add_argument("--hard-b", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_compositions(args.root, args.hard_a, args.hard_b), sort_keys=True))
        return 0
    except (OSError, KeyError, json.JSONDecodeError, HardStateError, CompositionError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
