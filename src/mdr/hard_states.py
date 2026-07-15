from __future__ import annotations

import argparse
import hashlib
import json
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path

from .state import FACE_ORDER, MOVE_SPECS, FullState, Move


MAGIC = b"MDRHSV1\0"
HEADER = struct.Struct("<8sBBHI")
RECORD_BYTES = 172
DENSE = {
    3: ((tuple(range(17)), (15, 16)), (tuple(range(22)) + (25,), (20, 21, 25))),
    5: (((*range(13), 14), (8, 12)), ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19), (8, 13, 17))),
    6: (((0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14), (7, 14)), ((0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16, 19), (7, 11, 19))),
}
FACE_COUNTS = {3: 7, 4: 6, 5: 5, 6: 4}


class HardStateError(ValueError):
    pass


def _positions(piece_at_position: tuple[int, ...]) -> list[int]:
    result = [0] * len(piece_at_position)
    for position, piece in enumerate(piece_at_position):
        result[piece] = position
    return result


def _ordered_rank(values: list[int], size: int) -> int:
    result = 0
    scale = 1
    for index, value in enumerate(values):
        reduced = value - sum(previous < value for previous in values[:index])
        result += reduced * scale
        scale *= size - index
    return result


def phase_index(state: FullState, phase: int) -> int:
    edge_positions = _positions(state.edge_pieces)
    corner_positions = _positions(state.corner_pieces)
    if phase == 4:
        corner = corner_positions[13]
        edges = [edge_positions[12], edge_positions[18]]
        modified = [value - 4 if value > 21 else value for value in edges]
        if modified[0] < modified[1]:
            modified[1] -= 1
        value = state.corner_orientations[corner] + 3 * corner + 45 * modified[0] + 900 * modified[1]
        for piece in (12, 18):
            value = (value << 1) | state.edge_orientations[edge_positions[piece]]
        return value
    try:
        (corner_allowed, corner_pieces), (edge_allowed, edge_pieces) = DENSE[phase]
    except KeyError as exc:
        raise HardStateError(f"unsupported phase {phase}") from exc
    corner_ord = {position: index for index, position in enumerate(corner_allowed)}
    edge_ord = {position: index for index, position in enumerate(edge_allowed)}
    cp = [corner_positions[piece] for piece in corner_pieces]
    ep = [edge_positions[piece] for piece in edge_pieces]
    corner_rank = (
        state.corner_orientations[cp[0]]
        + 3 * state.corner_orientations[cp[1]]
        + 9 * _ordered_rank([corner_ord[p] for p in cp], len(corner_allowed))
    )
    edge_orientation = sum(
        state.edge_orientations[position] << shift for position, shift in zip(ep, (2, 1, 0))
    )
    edge_rank = edge_orientation + 8 * _ordered_rank([edge_ord[p] for p in ep], len(edge_allowed))
    corner_count = len(corner_allowed) * (len(corner_allowed) - 1) * 9
    return corner_rank + corner_count * edge_rank


def code_move(code: int) -> Move:
    return Move(FACE_ORDER[code // 4], code % 4 + 1)


def apply_code(state: FullState, code: int) -> FullState:
    for _ in range(code % 4 + 1):
        state = state.apply_base(FACE_ORDER[code // 4])
    return state


class DenseTransitions:
    def __init__(self, phase: int):
        (self.corner_allowed, _), (self.edge_allowed, _) = DENSE[phase]
        self.face_count = FACE_COUNTS[phase]
        self.corner_count = len(self.corner_allowed) * (len(self.corner_allowed) - 1) * 9
        self.edge_count = len(self.edge_allowed) * (len(self.edge_allowed) - 1) * (len(self.edge_allowed) - 2) * 8
        self.corner_reps: list[tuple[tuple[int, ...], tuple[int, ...]] | None] = [None] * self.corner_count
        self.edge_reps: list[tuple[tuple[int, ...], tuple[int, ...]] | None] = [None] * self.edge_count
        for a in self.corner_allowed:
            for b in self.corner_allowed:
                if a == b:
                    continue
                for orientation in range(9):
                    rep = ((a, b), (orientation % 3, orientation // 3))
                    self.corner_reps[self._corner_rank(*rep)] = rep
        for a in self.edge_allowed:
            for b in self.edge_allowed:
                if a == b:
                    continue
                for c in self.edge_allowed:
                    if c in (a, b):
                        continue
                    for orientation in range(8):
                        rep = ((a, b, c), ((orientation >> 2) & 1, (orientation >> 1) & 1, orientation & 1))
                        self.edge_reps[self._edge_rank(*rep)] = rep
        if any(rep is None for rep in self.corner_reps + self.edge_reps):
            raise HardStateError("dense coordinate is not bijective")
        self.corner_transitions: list[array] = []
        self.edge_transitions: list[array] = []
        for code in range(self.face_count * 4):
            self.corner_transitions.append(array("I", (
                self._corner_rank(*self._apply(rep, code, True)) for rep in self.corner_reps if rep is not None
            )))
            self.edge_transitions.append(array("I", (
                self._edge_rank(*self._apply(rep, code, False)) for rep in self.edge_reps if rep is not None
            )))

    def _corner_rank(self, positions: tuple[int, ...], orientations: tuple[int, ...]) -> int:
        ordinal = {value: index for index, value in enumerate(self.corner_allowed)}
        return orientations[0] + 3 * orientations[1] + 9 * _ordered_rank([ordinal[p] for p in positions], len(ordinal))

    def _edge_rank(self, positions: tuple[int, ...], orientations: tuple[int, ...]) -> int:
        ordinal = {value: index for index, value in enumerate(self.edge_allowed)}
        bits = orientations[0] * 4 + orientations[1] * 2 + orientations[2]
        return bits + 8 * _ordered_rank([ordinal[p] for p in positions], len(ordinal))

    @staticmethod
    def _apply(rep: tuple[tuple[int, ...], tuple[int, ...]], code: int, corner: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
        positions, orientations = map(list, rep)
        face = FACE_ORDER[code // 4]
        kind, corner_cycle, edge_cycle = MOVE_SPECS[face]
        cycle = corner_cycle if corner else edge_cycle
        for _ in range(code % 4 + 1):
            for piece, position in enumerate(positions):
                if position not in cycle:
                    continue
                offset = cycle.index(position)
                if kind == "r":
                    if corner:
                        orientations[piece] = (orientations[piece] + (1 if offset == 0 else 2)) % 3
                    elif offset in (2, 4):
                        orientations[piece] ^= 1
                positions[piece] = cycle[(offset + 1) % 5]
        return tuple(positions), tuple(orientations)

    def next(self, index: int, code: int) -> int:
        corner = index % self.corner_count
        edge = index // self.corner_count
        return self.corner_transitions[code][corner] + self.corner_count * self.edge_transitions[code][edge]


@dataclass(frozen=True)
class HardRecord:
    index: int
    depth: int
    solution: tuple[int, ...]
    state: FullState
    first_mask: int
    last_mask: int


def read_hard(path: Path) -> tuple[int, int, list[HardRecord]]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise HardStateError("truncated hard-state header")
    magic, phase, diameter, record_size, count = HEADER.unpack_from(data)
    if magic != MAGIC or record_size != RECORD_BYTES:
        raise HardStateError("unsupported hard-state format")
    if len(data) != HEADER.size + count * RECORD_BYTES:
        raise HardStateError("hard-state file length mismatch")
    records: list[HardRecord] = []
    offset = HEADER.size
    for _ in range(count):
        body = data[offset : offset + 140]
        digest = data[offset + 140 : offset + RECORD_BYTES]
        if hashlib.sha256(body).digest() != digest:
            raise HardStateError(f"record checksum mismatch at byte {offset}")
        index = struct.unpack_from("<I", body, 0)[0]
        depth, length = body[4], body[5]
        if depth != diameter or length > 16 or any(value != 255 for value in body[116 + length : 132]):
            raise HardStateError(f"invalid solution record for index {index}")
        records.append(HardRecord(
            index=index,
            depth=depth,
            state=FullState.decode(body[8:116]),
            solution=tuple(body[116 : 116 + length]),
            first_mask=struct.unpack_from("<I", body, 132)[0],
            last_mask=struct.unpack_from("<I", body, 136)[0],
        ))
        offset += RECORD_BYTES
    return phase, diameter, records


def verify_hard(path: Path, table: Path) -> dict[str, int | bool]:
    phase, diameter, records = read_hard(path)
    depth = (table / "depths.bin").read_bytes()
    face_count = FACE_COUNTS[phase]
    allowed_mask = (1 << (face_count * 4)) - 1
    memo: dict[int, int] = {}
    dense = DenseTransitions(phase) if phase != 4 else None

    def lowering_indices(index: int) -> list[tuple[int, int]]:
        assert dense is not None
        result = []
        for code in range(face_count * 4):
            next_index = dense.next(index, code)
            if depth[next_index] != 255 and depth[next_index] + 1 == depth[index]:
                result.append((code, next_index))
        return result

    def lowering(state: FullState, index: int) -> list[tuple[int, FullState, int]]:
        result = []
        for code in range(face_count * 4):
            moved = apply_code(state, code)
            next_index = phase_index(moved, phase)
            if depth[next_index] != 255 and depth[next_index] + 1 == depth[index]:
                result.append((code, moved, next_index))
        return result

    def last_moves(state: FullState, index: int) -> int:
        if index in memo:
            return memo[index]
        result = 0
        if dense is not None:
            for code, next_index in lowering_indices(index):
                result |= 1 << code if depth[index] == 1 else last_moves(state, next_index)
        else:
            for code, moved, next_index in lowering(state, index):
                result |= 1 << code if depth[index] == 1 else last_moves(moved, next_index)
        memo[index] = result
        return result

    seen: set[int] = set()
    for record in records:
        if record.index in seen:
            raise HardStateError(f"duplicate hard index {record.index}")
        seen.add(record.index)
        if record.index >= len(depth) or depth[record.index] != diameter:
            raise HardStateError(f"index {record.index} is not at depth {diameter}")
        if phase_index(record.state, phase) != record.index:
            raise HardStateError(f"coordinate mismatch for index {record.index}")
        solved = record.state
        for code in record.solution:
            solved = apply_code(solved, code)
        if solved != FullState.solved():
            raise HardStateError(f"solution replay failed for index {record.index}")
        if dense is not None:
            first_mask = sum(1 << code for code, _ in lowering_indices(record.index))
        else:
            first_mask = sum(1 << code for code, _, _ in lowering(record.state, record.index))
        if record.first_mask != first_mask or record.first_mask & ~allowed_mask:
            raise HardStateError(f"first-move mask mismatch for index {record.index}")
        if record.last_mask != last_moves(record.state, record.index) or record.last_mask & ~allowed_mask:
            raise HardStateError(f"last-move mask mismatch for index {record.index}")
        if not record.solution or not (record.first_mask & (1 << record.solution[0])) or not (record.last_mask & (1 << record.solution[-1])):
            raise HardStateError(f"stored solution is absent from masks for index {record.index}")
    metadata_path = Path(str(path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("complete") is not True or metadata.get("count") != len(records):
        raise HardStateError("hard-state metadata mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["payload"]["sha256"]:
        raise HardStateError("hard-state payload checksum mismatch")
    return {"valid": True, "phase": phase, "depth": diameter, "count": len(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_hard(args.hard, args.table), sort_keys=True))
        return 0
    except (OSError, KeyError, HardStateError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
