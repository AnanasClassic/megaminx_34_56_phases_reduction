from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import ROOT
from .pair34_certificates import PROOF_MAX_LENGTH, SCHEMA, verification_sha256
from .pair56_certificates import file_sha256
from .pair34_problem import load_problem
from .pair34_training import DEFAULT_PROBLEM
from .pair56_problem import full_state_to_pair56
from .state import FullState, Move, in_target


DEFAULT_IDS = ROOT / "reductions" / "pair34" / "remaining_ids.bin"
DEFAULT_STATES = ROOT / "compositions" / "pair34" / "unique_states.bin"


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _open_checked_database(
    database: Path, problem_path: Path, hard_ids_path: Path, states_path: Path,
) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    expected = {
        "schema": SCHEMA,
        "pair": "pair34",
        "metric": "FTM",
        "target": "G7",
        "proof_max_length": str(PROOF_MAX_LENGTH),
        "problem_sha256": file_sha256(problem_path),
        "hard_ids_sha256": file_sha256(hard_ids_path),
        "composition_states_sha256": file_sha256(states_path),
    }
    observed = dict(connection.execute("SELECT key, value FROM metadata"))
    if observed != expected:
        connection.close()
        raise ValueError("pair34 certificate database metadata mismatch")
    return connection


def verify(
    *, database: Path, problem_path: Path, hard_ids_path: Path, states_path: Path,
) -> dict[str, object]:
    if not database.is_file():
        raise ValueError(f"certificate database does not exist: {database}")
    problem = load_problem(problem_path)
    states_blob = states_path.read_bytes()
    hard_ids_blob = hard_ids_path.read_bytes()
    if len(states_blob) % 108 or len(hard_ids_blob) % 4:
        raise ValueError("malformed pair34 certificate source artifacts")
    allowed_ids = {
        int.from_bytes(hard_ids_blob[offset : offset + 4], "little")
        for offset in range(0, len(hard_ids_blob), 4)
    }
    names = problem["names"]
    actions = problem["actions"]
    target = problem["target"]
    verified = 0
    maximum = 0
    beams: dict[str, int] = {}
    with closing(_open_checked_database(
        database, problem_path, hard_ids_path, states_path,
    )) as connection:
        rows = connection.execute(
            """
            SELECT state_id, state_sha256, solution, solution_length, beam_width,
                   checkpoint_sha256, checkpoint_epoch, verification_sha256
            FROM certificates ORDER BY state_id
            """
        )
        for (
            state_id, state_digest, solution, length, beam_width,
            checkpoint_digest, checkpoint_epoch, verification_digest,
        ) in rows:
            if state_id not in allowed_ids or state_id * 108 + 108 > len(states_blob):
                raise ValueError(f"certificate references invalid hard state {state_id}")
            state_bytes = states_blob[state_id * 108 : (state_id + 1) * 108]
            if hashlib.sha256(state_bytes).digest() != state_digest:
                raise ValueError(f"physical state checksum mismatch at {state_id}")
            if len(solution) != length or length > 21 or any(move >= 28 for move in solution):
                raise ValueError(f"invalid solution encoding at {state_id}")
            if verification_sha256(
                state_id=state_id, state_bytes=state_bytes, solution=solution,
                beam_width=beam_width, checkpoint_sha256=checkpoint_digest,
                checkpoint_epoch=checkpoint_epoch,
            ) != verification_digest:
                raise ValueError(f"verification checksum mismatch at {state_id}")

            state = FullState.decode(state_bytes)
            colored = full_state_to_pair56(state, problem)
            word = []
            for move in solution:
                colored = [colored[index] for index in actions[move]]
                name = names[move]
                word.append(Move(name[:-1], int(name[-1])))
            if colored != target:
                raise ValueError(f"quotient replay failed at {state_id}")
            if not in_target(state.apply(word), "g7"):
                raise ValueError(f"FullStateV1 replay failed at {state_id}")
            verified += 1
            maximum = max(maximum, length)
            beams[str(beam_width)] = beams.get(str(beam_width), 0) + 1
    return {
        "valid": True,
        "pair": "pair34",
        "database": _portable_path(database),
        "verified_certificates": verified,
        "maximum_solution_length": maximum if verified else None,
        "certificates_by_beam_width": beams,
        "remaining_representatives_total": len(allowed_ids),
        "coverage": verified / len(allowed_ids),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Independently replay pair34 certificates.")
    result.add_argument("--database", type=Path, required=True)
    result.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM)
    result.add_argument("--hard-ids", type=Path, default=DEFAULT_IDS)
    result.add_argument("--composition-states", type=Path, default=DEFAULT_STATES)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        print(json.dumps(verify(
            database=args.database.resolve(), problem_path=args.problem.resolve(),
            hard_ids_path=args.hard_ids.resolve(), states_path=args.composition_states.resolve(),
        ), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
