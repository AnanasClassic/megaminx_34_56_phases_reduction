import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mdr.go_bulk_verify import GoBulkReplay, GoBulkReplayError
from mdr.state import FullState


FAKE_VERIFIER = """#!/usr/bin/env python3
import hashlib
import json
import sys

target = sys.argv[sys.argv.index("--target") + 1]
limit = int(sys.argv[sys.argv.index("--max-length") + 1])
records = []
transcript = hashlib.sha256()
for line in sys.stdin.buffer:
    transcript.update(line)
    fields = line.rstrip(b"\\n").decode("ascii").split("\\t")
    if len(fields) != 3:
        raise SystemExit(2)
    records.append(fields)
maximum = max((len(fields[2].split()) for fields in records), default=None)
print(json.dumps({
    "valid": True,
    "records": len(records),
    "maximum_solution_length": maximum,
    "target": target,
    "max_length": limit,
    "first_state_id": int(records[0][0]) if records else None,
    "last_state_id": int(records[-1][0]) if records else None,
    "transcript_sha256": transcript.hexdigest(),
}))
"""


class GoBulkReplayTests(unittest.TestCase):
    def make_verifier(self, root: Path, source: str = FAKE_VERIFIER) -> Path:
        verifier = root / "fake-verifier"
        verifier.write_text(source, encoding="utf-8")
        verifier.chmod(0o755)
        return verifier

    def test_streams_all_records_and_checks_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory))
            replay = GoBulkReplay(verifier, target="G7", max_length=21)
            state = FullState.solved().encode()
            replay.feed(state_id=2, state_bytes=state, word="")
            replay.feed(state_id=7, state_bytes=state, word="U1 R2")
            report = replay.finish(expected_count=2, expected_maximum_length=2)
            self.assertTrue(report["valid"])
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["target"], "g7")
            expected_transcript = (
                f"2\t{state.hex()}\t\n7\t{state.hex()}\tU1 R2\n".encode("ascii")
            )
            self.assertEqual(
                report["transcript_sha256"], hashlib.sha256(expected_transcript).hexdigest(),
            )
            self.assertEqual(report["first_state_id"], 2)
            self.assertEqual(report["last_state_id"], 7)
            self.assertEqual(len(report["verifier_sha256"]), 64)

    def test_rejects_non_increasing_ids_before_sending_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory))
            replay = GoBulkReplay(verifier, target="g9", max_length=25)
            replay.feed(state_id=3, state_bytes=b"state", word="U1")
            with self.assertRaises(GoBulkReplayError):
                replay.feed(state_id=3, state_bytes=b"state", word="U1")
            replay.abort()

    def test_empty_transcript_has_null_endpoints_and_standard_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory))
            replay = GoBulkReplay(verifier, target="g9", max_length=25)
            report = replay.finish(expected_count=0, expected_maximum_length=None)
            self.assertIsNone(report["first_state_id"])
            self.assertIsNone(report["last_state_id"])
            self.assertEqual(
                report["transcript_sha256"], hashlib.sha256(b"").hexdigest(),
            )

    def test_fails_closed_on_report_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory))
            replay = GoBulkReplay(verifier, target="g9", max_length=25)
            replay.feed(state_id=1, state_bytes=b"state", word="F1")
            with self.assertRaises(GoBulkReplayError):
                replay.finish(expected_count=2, expected_maximum_length=1)

    def test_fails_closed_on_transcript_digest_mismatch(self) -> None:
        source = FAKE_VERIFIER.replace(
            '"transcript_sha256": transcript.hexdigest(),',
            '"transcript_sha256": "0" * 64,',
        )
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory), source)
            replay = GoBulkReplay(verifier, target="g7", max_length=21)
            replay.feed(state_id=9, state_bytes=b"state", word="U1")
            with self.assertRaisesRegex(GoBulkReplayError, "report mismatch"):
                replay.finish(expected_count=1, expected_maximum_length=1)

    def test_fails_closed_on_first_or_last_id_mismatch(self) -> None:
        source = FAKE_VERIFIER.replace(
            '"first_state_id": int(records[0][0]) if records else None,',
            '"first_state_id": 0 if records else None,',
        )
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory), source)
            replay = GoBulkReplay(verifier, target="g9", max_length=25)
            replay.feed(state_id=4, state_bytes=b"state", word="")
            with self.assertRaisesRegex(GoBulkReplayError, "report mismatch"):
                replay.finish(expected_count=1, expected_maximum_length=0)

    def test_communicate_drains_large_stderr_without_deadlock(self) -> None:
        source = FAKE_VERIFIER.replace(
            "print(json.dumps({",
            'sys.stderr.write("x" * (2 * 1024 * 1024))\nprint(json.dumps({',
        )
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory), source)
            replay = GoBulkReplay(
                verifier, target="g7", max_length=21, timeout_seconds=5,
            )
            replay.feed(state_id=1, state_bytes=b"state", word="")
            report = replay.finish(expected_count=1, expected_maximum_length=0)
            self.assertEqual(report["records"], 1)

    def test_timeout_terminates_verifier(self) -> None:
        source = """#!/usr/bin/env python3
import sys
import time
for _line in sys.stdin:
    pass
time.sleep(10)
"""
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory), source)
            replay = GoBulkReplay(
                verifier, target="g7", max_length=21, timeout_seconds=0.05,
            )
            replay.feed(state_id=1, state_bytes=b"state", word="")
            with self.assertRaisesRegex(GoBulkReplayError, "timeout"):
                replay.finish(expected_count=1, expected_maximum_length=0)

    def test_rejects_nonpositive_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier = self.make_verifier(Path(directory))
            with self.assertRaisesRegex(GoBulkReplayError, "timeout must be positive"):
                GoBulkReplay(verifier, target="g7", max_length=21, timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
