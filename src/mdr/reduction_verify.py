from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from .equivalences import decode_unchecked
from .hard_states import apply_code
from .state import FullState


class ReductionError(ValueError):
    pass


CONTROLS = {"pair34": (536572, 21), "pair56": (407628, 25)}


def verify(root: Path, compositions: Path) -> dict[str, int | bool | str]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    pair = metadata.get("pair")
    if pair not in CONTROLS or metadata.get("complete") is not True:
        raise ReductionError("unknown or incomplete reduction manifest")
    raw_count, bound = CONTROLS[pair]
    if metadata.get("raw_pairs") != raw_count or metadata.get("max_length") != bound:
        raise ReductionError("reduction controls mismatch")
    for name, spec in metadata["payloads"].items():
        data = (root / name).read_bytes()
        if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ReductionError(f"payload mismatch: {name}")
    mapping = (root / "mapping.bin").read_bytes()
    witnesses = (root / "witnesses.bin").read_bytes()
    remaining_blob = (root / "remaining_ids.bin").read_bytes()
    states = (compositions / "states.bin").read_bytes()
    raw_to_unique = (compositions / "raw_to_unique.bin").read_bytes()
    if len(mapping) != raw_count * 16 or len(states) != raw_count * 108 or len(raw_to_unique) != raw_count * 4 or len(remaining_blob) % 4:
        raise ReductionError("reduction payload length mismatch")
    remaining: list[int] = []
    boundary = local = 0
    intervals: list[tuple[int, int]] = []
    max_length = 0
    for raw_id in range(raw_count):
        record = mapping[raw_id * 16 : (raw_id + 1) * 16]
        unique_id = struct.unpack_from("<I", record, 0)[0]
        status, length = record[4], record[5]
        representative = struct.unpack_from("<I", record, 8)[0]
        offset = struct.unpack_from("<I", record, 12)[0]
        if unique_id != struct.unpack_from("<I", raw_to_unique, raw_id * 4)[0]:
            raise ReductionError(f"unique mapping mismatch at raw {raw_id}")
        if status == 0:
            if length != 0 or representative != unique_id:
                raise ReductionError(f"invalid remaining mapping at raw {raw_id}")
            remaining.append(representative)
            continue
        if status not in (1, 2) or representative != 0xFFFFFFFF or length > bound or offset + length > len(witnesses):
            raise ReductionError(f"invalid closed mapping at raw {raw_id}")
        intervals.append((offset, offset + length))
        word = witnesses[offset : offset + length]
        state = decode_unchecked(states[raw_id * 108 : (raw_id + 1) * 108])
        for code in word:
            state = apply_code(state, code)
        if state != FullState.solved():
            raise ReductionError(f"witness replay failed at raw {raw_id}")
        max_length = max(max_length, length)
        if status == 1:
            boundary += 1
        else:
            local += 1
    declared_remaining = list(struct.iter_unpack("<I", remaining_blob))
    if remaining != [value[0] for value in declared_remaining]:
        raise ReductionError("remaining representative list mismatch")
    if intervals:
        intervals.sort()
        if intervals[0][0] != 0 or intervals[-1][1] != len(witnesses) or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
            raise ReductionError("witness payload has overlaps or gaps")
    elif witnesses:
        raise ReductionError("unreferenced witness bytes")
    stats = json.loads((root / "statistics.json").read_text(encoding="utf-8"))
    expected = {
        "closed_by_boundary_merge": boundary,
        "closed_by_local_rewrite": local,
        "remaining_canonical_representatives": len(remaining),
        "maximum_verified_witness_length": max_length,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ReductionError(f"statistics mismatch: {key}")
    equivalences = json.loads((root / "equivalences.json").read_text(encoding="utf-8"))
    if equivalences.get("complete") is not True or equivalences["rotation_group"]["admissible_group_size"] != 1 or equivalences["inversion"]["admissible"] is not False:
        raise ReductionError("equivalence analysis is absent or unsafe")
    if boundary + local + len(remaining) != raw_count:
        raise ReductionError("reduction coverage is not total")
    return {"valid": True, "pair": pair, "raw_pairs": raw_count, "closed": boundary + local, "remaining": len(remaining)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compositions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.root, args.compositions), sort_keys=True))
        return 0
    except (OSError, KeyError, json.JSONDecodeError, ReductionError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
