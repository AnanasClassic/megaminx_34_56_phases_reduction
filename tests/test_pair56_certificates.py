import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mdr.config import ROOT
from mdr.pair56_certificate_verify import _open_checked_database, verify
from mdr.pair56_certificates import Pair56CertificateStore
from mdr.state import FullState


FAKE_GO_VERIFIER = """#!/usr/bin/env python3
import hashlib
import json
import sys
target = sys.argv[sys.argv.index("--target") + 1]
limit = int(sys.argv[sys.argv.index("--max-length") + 1])
transcript = hashlib.sha256()
lines = list(sys.stdin.buffer)
for line in lines:
    transcript.update(line)
rows = [line.rstrip(b"\\n").decode("ascii").split("\\t") for line in lines]
maximum = max((len(row[2].split()) for row in rows), default=None)
print(json.dumps({"valid": True, "records": len(rows),
                  "maximum_solution_length": maximum,
                  "target": target, "max_length": limit,
                  "first_state_id": int(rows[0][0]) if rows else None,
                  "last_state_id": int(rows[-1][0]) if rows else None,
                  "transcript_sha256": transcript.hexdigest()}))
"""


class Pair56CertificateTests(unittest.TestCase):
    def test_store_resumes_and_keeps_better_beam_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = root / "problem.json"
            ids = root / "ids.bin"
            states = root / "states.bin"
            database = root / "certificates.sqlite3"
            problem.write_bytes(b"problem")
            ids.write_bytes((7).to_bytes(4, "little"))
            states.write_bytes(b"state-artifact")
            checkpoint = "11" * 32
            with Pair56CertificateStore(
                database, problem_path=problem, hard_ids_path=ids,
                composition_states_path=states,
            ) as store:
                self.assertTrue(store.add(
                    state_id=7, state_bytes=b"physical-state", solution=(1, 2, 3),
                    beam_width=256, checkpoint_sha256_hex=checkpoint, checkpoint_epoch=10,
                ))
                self.assertFalse(store.add(
                    state_id=7, state_bytes=b"physical-state", solution=(1, 2, 3),
                    beam_width=512, checkpoint_sha256_hex=checkpoint, checkpoint_epoch=11,
                ))
                self.assertTrue(store.add(
                    state_id=7, state_bytes=b"physical-state", solution=(1, 2, 3),
                    beam_width=64, checkpoint_sha256_hex=checkpoint, checkpoint_epoch=12,
                ))
                store.commit()
            with Pair56CertificateStore(
                database, problem_path=problem, hard_ids_path=ids,
                composition_states_path=states,
            ) as store:
                self.assertEqual(store.certified_ids((7, 8)), {7})
                row = store.connection.execute(
                    "SELECT solution, solution_length, beam_width, checkpoint_epoch FROM certificates"
                ).fetchone()
                self.assertEqual(row, (b"\x01\x02\x03", 3, 64, 12))

    def test_metadata_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("problem", "ids", "states")]
            for path in paths:
                path.write_bytes(path.name.encode())
            database = root / "certificates.sqlite3"
            with Pair56CertificateStore(
                database, problem_path=paths[0], hard_ids_path=paths[1],
                composition_states_path=paths[2],
            ):
                pass
            paths[2].write_bytes(b"changed")
            with self.assertRaises(ValueError):
                Pair56CertificateStore(
                    database, problem_path=paths[0], hard_ids_path=paths[1],
                    composition_states_path=paths[2],
                )

    def test_verifier_opens_database_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("problem", "ids", "states")]
            for path in paths:
                path.write_bytes(path.name.encode())
            database = root / "certificates.sqlite3"
            with Pair56CertificateStore(
                database, problem_path=paths[0], hard_ids_path=paths[1],
                composition_states_path=paths[2],
            ):
                pass
            before = hashlib.sha256(database.read_bytes()).digest()
            connection = _open_checked_database(database, paths[0], paths[1], paths[2])
            connection.close()
            after = hashlib.sha256(database.read_bytes()).digest()
            self.assertEqual(after, before)

    def test_bulk_go_replay_is_bound_to_the_same_certificate_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = ROOT / "training" / "pair56" / "problem.json"
            ids = root / "ids.bin"
            states = root / "states.bin"
            database = root / "certificates.sqlite3"
            go_verifier = root / "mdr-verify"
            state = FullState.solved().encode()
            ids.write_bytes((0).to_bytes(4, "little"))
            states.write_bytes(state)
            go_verifier.write_text(FAKE_GO_VERIFIER, encoding="utf-8")
            go_verifier.chmod(0o755)
            with Pair56CertificateStore(
                database, problem_path=problem, hard_ids_path=ids,
                composition_states_path=states,
            ) as store:
                store.add(
                    state_id=0, state_bytes=state, solution=(), beam_width=1,
                    checkpoint_sha256_hex="11" * 32, checkpoint_epoch=0,
                )
                store.commit()
            result = verify(
                database=database, problem_path=problem, hard_ids_path=ids,
                states_path=states, go_verifier=go_verifier,
            )
            self.assertEqual(result["verified_certificates"], 1)
            self.assertEqual(result["replay"]["go_full_state"]["records"], 1)


if __name__ == "__main__":
    unittest.main()
