from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


class GoBulkReplayError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class GoBulkReplay:
    """Stream FullStateV1 records to one persistent Go verifier process.

    The certificate database and quotient encoding remain Python concerns.  The
    protocol deliberately sends only a monotonically increasing record ID, the
    physical state bytes, and the canonical move word.  The Go implementation
    therefore replays the physical claim without importing the Python state
    transition code.
    """

    def __init__(
        self, verifier: Path, *, target: str, max_length: int,
        timeout_seconds: float = 7200,
    ) -> None:
        self.verifier = verifier.resolve()
        self.target = target.lower()
        self.max_length = max_length
        if not self.verifier.is_file():
            raise GoBulkReplayError(f"Go verifier does not exist: {self.verifier}")
        if max_length < 0:
            raise GoBulkReplayError("Go bulk replay length bound must be nonnegative")
        if timeout_seconds <= 0:
            raise GoBulkReplayError("Go bulk replay timeout must be positive")
        self.timeout_seconds = timeout_seconds
        try:
            self.process = subprocess.Popen(
                [
                    str(self.verifier), "verify-batch",
                    "--target", self.target,
                    "--max-length", str(self.max_length),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise GoBulkReplayError(f"cannot start Go verifier {self.verifier}: {exc}") from exc
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.process.kill()
            raise GoBulkReplayError("cannot open Go verifier pipes")
        self.sent = 0
        self.first_state_id: int | None = None
        self.previous_state_id = -1
        self.transcript = hashlib.sha256()
        self.finished = False

    def feed(self, *, state_id: int, state_bytes: bytes, word: str) -> None:
        if self.finished:
            raise GoBulkReplayError("cannot feed a finished Go bulk replay")
        if state_id <= self.previous_state_id:
            raise GoBulkReplayError("Go bulk replay IDs must be strictly increasing")
        if any(character in word for character in "\t\r\n"):
            raise GoBulkReplayError("canonical Go bulk replay words cannot contain tabs or newlines")
        record = f"{state_id}\t{state_bytes.hex()}\t{word}\n".encode("ascii")
        try:
            assert self.process.stdin is not None
            written = self.process.stdin.write(record)
            if written != len(record):
                raise OSError(f"short pipe write: {written} of {len(record)} bytes")
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.abort()
            raise GoBulkReplayError("Go verifier terminated while receiving records") from exc
        if self.first_state_id is None:
            self.first_state_id = state_id
        self.previous_state_id = state_id
        self.transcript.update(record)
        self.sent += 1

    def finish(
        self, *, expected_count: int, expected_maximum_length: int | None,
    ) -> dict[str, Any]:
        if self.finished:
            raise GoBulkReplayError("Go bulk replay was already finished")
        self.finished = True
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        try:
            self.process.stdin.close()
            # communicate() drains stdout and stderr concurrently.  Setting the
            # closed input pipe to None prevents communicate() from trying to
            # flush it again.
            self.process.stdin = None
            stdout, stderr = self.process.communicate(timeout=self.timeout_seconds)
            returncode = self.process.returncode
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            stdout, stderr = self.process.communicate()
            raise GoBulkReplayError(
                f"Go bulk replay exceeded its {self.timeout_seconds:g}-second timeout"
            ) from exc
        except OSError as exc:
            self.abort()
            raise GoBulkReplayError(f"cannot finish Go bulk replay: {exc}") from exc
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise GoBulkReplayError(
                f"Go bulk replay failed with exit {returncode}: {detail or 'no diagnostic'}"
            )
        try:
            report = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GoBulkReplayError("Go bulk replay emitted malformed JSON") from exc
        expected = {
            "valid": True,
            "records": expected_count,
            "maximum_solution_length": expected_maximum_length,
            "target": self.target,
            "max_length": self.max_length,
            "first_state_id": self.first_state_id,
            "last_state_id": self.previous_state_id if self.sent else None,
            "transcript_sha256": self.transcript.hexdigest(),
        }
        if self.sent != expected_count:
            raise GoBulkReplayError(
                f"Go bulk replay received {self.sent} records, expected {expected_count}"
            )
        if report != expected:
            raise GoBulkReplayError(
                f"Go bulk replay report mismatch: observed={report!r}, expected={expected!r}"
            )
        return {
            **report,
            "implementation": "verifier/go/main.go verify-batch",
            "verifier": str(self.verifier),
            "verifier_sha256": _sha256(self.verifier),
        }

    def abort(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass
        try:
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
        self.finished = True
