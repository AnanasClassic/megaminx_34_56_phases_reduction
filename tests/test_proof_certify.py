import unittest
from pathlib import Path

from mdr.config import ROOT
from mdr.pair34_certificate_verify import _portable_path
from mdr.proof_certify import ProofError, _coverage_summary


class ProofCertifyTests(unittest.TestCase):
    def test_repository_database_paths_are_portable(self) -> None:
        path = ROOT / "certificates" / "pair34" / "beam-cascade.sqlite3"
        self.assertEqual(_portable_path(path), "certificates/pair34/beam-cascade.sqlite3")
        external = Path("/tmp/mdr-external.sqlite3")
        self.assertEqual(_portable_path(external), str(external))

    def test_exact_coverage_contract(self) -> None:
        result = _coverage_summary(
            pair="pair34",
            reduction={"raw_pairs": 536572, "closed": 203, "remaining": 536369},
            certificates={
                "remaining_representatives_total": 536369,
                "verified_certificates": 536369,
                "maximum_solution_length": 21,
            },
        )
        self.assertEqual(result["covered_states"], 536572)
        self.assertEqual(result["missing_certificates"], 0)

    def test_partial_certificate_set_fails_closed(self) -> None:
        with self.assertRaises(ProofError):
            _coverage_summary(
                pair="pair56",
                reduction={"raw_pairs": 407628, "closed": 8461, "remaining": 399167},
                certificates={
                    "remaining_representatives_total": 399167,
                    "verified_certificates": 399166,
                    "maximum_solution_length": 25,
                },
            )

    def test_excess_length_fails_closed(self) -> None:
        with self.assertRaises(ProofError):
            _coverage_summary(
                pair="pair34",
                reduction={"raw_pairs": 536572, "closed": 203, "remaining": 536369},
                certificates={
                    "remaining_representatives_total": 536369,
                    "verified_certificates": 536369,
                    "maximum_solution_length": 22,
                },
            )


if __name__ == "__main__":
    unittest.main()
