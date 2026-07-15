from __future__ import annotations

import hashlib
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA = "mdr-pair56-certificates-v1"
PROOF_MAX_LENGTH = 25


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verification_sha256(
    *, state_id: int, state_bytes: bytes, solution: bytes, beam_width: int,
    checkpoint_sha256: bytes, checkpoint_epoch: int,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"MDRP56CERT1\x00")
    digest.update(struct.pack("<III", state_id, beam_width, checkpoint_epoch))
    digest.update(checkpoint_sha256)
    digest.update(state_bytes)
    digest.update(solution)
    return digest.digest()


class Pair56CertificateStore:
    def __init__(
        self, path: Path, *, problem_path: Path, hard_ids_path: Path,
        composition_states_path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS certificates (
                state_id INTEGER PRIMARY KEY,
                state_sha256 BLOB NOT NULL CHECK(length(state_sha256) = 32),
                solution BLOB NOT NULL CHECK(length(solution) = solution_length),
                solution_length INTEGER NOT NULL CHECK(solution_length BETWEEN 0 AND 25),
                beam_width INTEGER NOT NULL CHECK(beam_width > 0),
                checkpoint_sha256 BLOB NOT NULL CHECK(length(checkpoint_sha256) = 32),
                checkpoint_epoch INTEGER NOT NULL CHECK(checkpoint_epoch >= 0),
                verification_sha256 BLOB NOT NULL CHECK(length(verification_sha256) = 32),
                created_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS certificates_beam_width
                ON certificates(beam_width);
            """
        )
        expected = {
            "schema": SCHEMA,
            "pair": "pair56",
            "metric": "FTM",
            "target": "G9",
            "proof_max_length": str(PROOF_MAX_LENGTH),
            "problem_sha256": file_sha256(problem_path),
            "hard_ids_sha256": file_sha256(hard_ids_path),
            "composition_states_sha256": file_sha256(composition_states_path),
        }
        observed = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if observed:
            if observed != expected:
                raise ValueError("pair56 certificate database metadata mismatch")
        else:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES(?, ?)", expected.items()
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Pair56CertificateStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def certified_ids(self, state_ids: Iterable[int]) -> set[int]:
        values = list(state_ids)
        result: set[int] = set()
        for start in range(0, len(values), 900):
            chunk = values[start : start + 900]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            result.update(row[0] for row in self.connection.execute(
                f"SELECT state_id FROM certificates WHERE state_id IN ({placeholders})", chunk
            ))
        return result

    def add(
        self, *, state_id: int, state_bytes: bytes, solution: tuple[int, ...],
        beam_width: int, checkpoint_sha256_hex: str, checkpoint_epoch: int,
    ) -> bool:
        solution_bytes = bytes(solution)
        if len(solution_bytes) > PROOF_MAX_LENGTH or any(move >= 20 for move in solution_bytes):
            raise ValueError("invalid pair56 proof solution")
        state_digest = hashlib.sha256(state_bytes).digest()
        checkpoint_digest = bytes.fromhex(checkpoint_sha256_hex)
        verification = verification_sha256(
            state_id=state_id, state_bytes=state_bytes, solution=solution_bytes,
            beam_width=beam_width, checkpoint_sha256=checkpoint_digest,
            checkpoint_epoch=checkpoint_epoch,
        )
        before = self.connection.total_changes
        self.connection.execute(
            """
            INSERT INTO certificates(
                state_id, state_sha256, solution, solution_length, beam_width,
                checkpoint_sha256, checkpoint_epoch, verification_sha256, created_utc
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                state_sha256=excluded.state_sha256,
                solution=excluded.solution,
                solution_length=excluded.solution_length,
                beam_width=excluded.beam_width,
                checkpoint_sha256=excluded.checkpoint_sha256,
                checkpoint_epoch=excluded.checkpoint_epoch,
                verification_sha256=excluded.verification_sha256,
                created_utc=excluded.created_utc
            WHERE excluded.solution_length < certificates.solution_length
               OR (excluded.solution_length = certificates.solution_length
                   AND excluded.beam_width < certificates.beam_width)
            """,
            (
                state_id, state_digest, solution_bytes, len(solution_bytes), beam_width,
                checkpoint_digest, checkpoint_epoch, verification,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return self.connection.total_changes > before

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM certificates").fetchone()[0])

    def statistics(self) -> dict[str, object]:
        count, maximum = self.connection.execute(
            "SELECT count(*), max(solution_length) FROM certificates"
        ).fetchone()
        beams = {
            str(beam): amount for beam, amount in self.connection.execute(
                "SELECT beam_width, count(*) FROM certificates GROUP BY beam_width ORDER BY beam_width"
            )
        }
        return {
            "database": str(self.path.resolve()),
            "certificates": int(count),
            "maximum_solution_length": int(maximum) if maximum is not None else None,
            "certificates_by_beam_width": beams,
        }
