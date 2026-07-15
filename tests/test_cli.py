import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from mdr.cli import main
from mdr.resources import GIB, ResourceError, check_disk, directory_size


class CliTests(unittest.TestCase):
    def test_proof_commands_fail_closed(self) -> None:
        for command in ("verify", "build-tables", "pretraining-gate"):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([command]), 2)

    def test_certify_rejects_bound_drift_before_reading_artifacts(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["certify", "pair34", "--max-length", "20"]), 1)
            self.assertEqual(main(["certify", "pair56", "--max-length", "26"]), 1)

    def test_unimplemented_verifier_rejects_realistic_arguments(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(["verify", "--state", "state.bin", "--solution", "solution.txt", "--max-length", "21"]),
                2,
            )

    def test_validate_config(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["validate-config"]), 0)

    def test_impossible_disk_reserve_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ResourceError):
                check_disk(Path(directory), minimum_free_gib=10 ** 9)

    def test_project_cap_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload").write_bytes(b"x" * 1024)
            self.assertGreaterEqual(directory_size(root), 1024)
            with self.assertRaises(ResourceError):
                check_disk(
                    root, minimum_free_gib=0, projected_gib=1,
                    project_root=root, maximum_project_gib=0.5,
                )
            budget = check_disk(
                root, minimum_free_gib=0, projected_gib=0,
                project_root=root, maximum_project_gib=0.5,
            )
            self.assertEqual(budget.project_bytes, 1024)
            self.assertEqual(budget.maximum_project_bytes, int(0.5 * GIB))


if __name__ == "__main__":
    unittest.main()
