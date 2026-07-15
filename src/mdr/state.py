from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAGIC = b"MDRFSV1\x00"
STATE_SIZE = 108
FACE_ORDER = ("U", "R", "F", "L", "BR", "BL", "FR", "FL", "DR", "DL", "B", "D")
MOVE_SPECS = {
    "U": ("u", (0, 1, 2, 3, 4), (0, 1, 2, 3, 4)),
    "R": ("r", (0, 4, 9, 11, 5), (4, 9, 14, 16, 5)),
    "F": ("r", (1, 0, 5, 10, 6), (0, 5, 10, 15, 6)),
    "L": ("r", (2, 1, 6, 14, 7), (1, 6, 11, 19, 7)),
    "BL": ("r", (3, 2, 7, 13, 8), (2, 7, 12, 18, 8)),
    "BR": ("r", (4, 3, 8, 12, 9), (3, 8, 13, 17, 9)),
    "D": ("u", (15, 16, 17, 18, 19), (25, 26, 27, 28, 29)),
    "FL": ("r", (15, 19, 14, 6, 10), (29, 24, 11, 15, 20)),
    "FR": ("r", (16, 15, 10, 5, 11), (25, 20, 10, 16, 21)),
    "DR": ("r", (17, 16, 11, 9, 12), (26, 21, 14, 17, 22)),
    "B": ("r", (18, 17, 12, 8, 13), (27, 22, 13, 18, 23)),
    "DL": ("r", (19, 18, 13, 7, 14), (28, 23, 12, 19, 24)),
}


class StateError(ValueError):
    pass


@dataclass(frozen=True)
class Move:
    face: str
    power: int

    @property
    def token(self) -> str:
        return f"{self.face}{self.power}"

    def inverse(self) -> "Move":
        return Move(self.face, 5 - self.power)


def _parity(values: tuple[int, ...], size: int) -> int:
    if len(values) != size or sorted(values) != list(range(size)):
        raise StateError(f"expected a permutation of 0..{size - 1}")
    return sum(values[i] > values[j] for i in range(size) for j in range(i + 1, size)) % 2


@dataclass(frozen=True)
class FullState:
    edge_pieces: tuple[int, ...]
    edge_orientations: tuple[int, ...]
    corner_pieces: tuple[int, ...]
    corner_orientations: tuple[int, ...]

    @classmethod
    def solved(cls) -> "FullState":
        return cls(tuple(range(30)), (0,) * 30, tuple(range(20)), (0,) * 20)

    def validate(self) -> None:
        if _parity(self.edge_pieces, 30) != 0:
            raise StateError("edge permutation must be even")
        if _parity(self.corner_pieces, 20) != 0:
            raise StateError("corner permutation must be even")
        if len(self.edge_orientations) != 30 or any(value not in (0, 1) for value in self.edge_orientations):
            raise StateError("edge orientations must contain 30 bits")
        if len(self.corner_orientations) != 20 or any(value not in (0, 1, 2) for value in self.corner_orientations):
            raise StateError("corner orientations must contain 20 trits")
        if sum(self.edge_orientations) % 2:
            raise StateError("edge orientation sum must be even")
        if sum(self.corner_orientations) % 3:
            raise StateError("corner orientation sum must vanish modulo 3")

    def encode(self) -> bytes:
        self.validate()
        return b"".join((
            MAGIC,
            bytes(self.edge_pieces),
            bytes(self.edge_orientations),
            bytes(self.corner_pieces),
            bytes(self.corner_orientations),
        ))

    @classmethod
    def decode(cls, data: bytes) -> "FullState":
        if len(data) != STATE_SIZE:
            raise StateError(f"FullStateV1 must be {STATE_SIZE} bytes, got {len(data)}")
        if data[:8] != MAGIC:
            raise StateError("invalid FullStateV1 magic")
        result = cls(
            tuple(data[8:38]),
            tuple(data[38:68]),
            tuple(data[68:88]),
            tuple(data[88:108]),
        )
        result.validate()
        return result

    @classmethod
    def read(cls, path: Path | str) -> "FullState":
        return cls.decode(Path(path).read_bytes())

    def write(self, path: Path | str) -> None:
        target = Path(path)
        temporary = target.with_name(target.name + ".partial")
        temporary.write_bytes(self.encode())
        temporary.replace(target)

    def apply_base(self, face: str) -> "FullState":
        try:
            kind, corner_cycle, edge_cycle = MOVE_SPECS[face]
        except KeyError as exc:
            raise StateError(f"unknown face {face!r}") from exc
        edge_records = list(zip(self.edge_pieces, self.edge_orientations))
        corner_records = list(zip(self.corner_pieces, self.corner_orientations))
        if kind == "r":
            for offset, position in enumerate(corner_cycle):
                piece, orientation = corner_records[position]
                corner_records[position] = (piece, (orientation + (1 if offset == 0 else 2)) % 3)
            for position in (edge_cycle[2], edge_cycle[4]):
                piece, orientation = edge_records[position]
                edge_records[position] = (piece, orientation ^ 1)
        old_corners = [corner_records[position] for position in corner_cycle]
        old_edges = [edge_records[position] for position in edge_cycle]
        for offset, position in enumerate(corner_cycle):
            corner_records[position] = old_corners[(offset - 1) % 5]
        for offset, position in enumerate(edge_cycle):
            edge_records[position] = old_edges[(offset - 1) % 5]
        return FullState(
            tuple(record[0] for record in edge_records),
            tuple(record[1] for record in edge_records),
            tuple(record[0] for record in corner_records),
            tuple(record[1] for record in corner_records),
        )

    def apply(self, word: Iterable[Move]) -> "FullState":
        state = self
        for move in word:
            for _ in range(move.power):
                state = state.apply_base(move.face)
        state.validate()
        return state


def parse_word(data: bytes) -> tuple[Move, ...]:
    if b"\r" in data:
        raise StateError("CR is forbidden in canonical move words")
    if data.endswith(b"\n"):
        data = data[:-1]
    if b"\n" in data:
        raise StateError("move word must occupy one line")
    if not data:
        return ()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise StateError("move word must be ASCII") from exc
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        raise StateError("tokens must be separated by one ASCII space")
    result = []
    for token in text.split(" "):
        if len(token) < 2 or token[:-1] not in FACE_ORDER or token[-1] not in "1234":
            raise StateError(f"invalid move token {token!r}")
        result.append(Move(token[:-1], int(token[-1])))
    return tuple(result)


def format_word(word: Iterable[Move]) -> bytes:
    return " ".join(move.token for move in word).encode("ascii") + b"\n"


def invert_word(word: Iterable[Move]) -> tuple[Move, ...]:
    return tuple(move.inverse() for move in reversed(tuple(word)))


def in_target(state: FullState, target: str) -> bool:
    target = target.lower()
    if target == "solved":
        return state == FullState.solved()
    counts = {"g5": 7, "g6": 6, "g7": 5, "g8": 4, "g9": 3}
    if target not in counts:
        raise StateError(f"unknown target {target!r}")
    mobile_edges: set[int] = set()
    mobile_corners: set[int] = set()
    solved = FullState.solved()
    for face in FACE_ORDER[:counts[target]]:
        moved = solved.apply_base(face)
        mobile_edges.update(i for i in range(30) if moved.edge_pieces[i] != i or moved.edge_orientations[i])
        mobile_corners.update(i for i in range(20) if moved.corner_pieces[i] != i or moved.corner_orientations[i])
    return all(
        position in mobile_edges or (state.edge_pieces[position] == position and state.edge_orientations[position] == 0)
        for position in range(30)
    ) and all(
        position in mobile_corners or (state.corner_pieces[position] == position and state.corner_orientations[position] == 0)
        for position in range(20)
    )
