import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mdr.pair56_certificate_verify import _open_checked_database
from mdr.pair56_certificates import Pair56CertificateStore


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


if __name__ == "__main__":
    unittest.main()
