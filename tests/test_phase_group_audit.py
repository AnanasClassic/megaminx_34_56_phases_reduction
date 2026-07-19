import copy
import importlib.util
import json
import unittest

from mdr.config import ROOT
from mdr.hard_states import phase_index
from mdr.pair34_problem import Pair34ProblemError, validate_problem as validate_pair34
from mdr.pair56_problem import Pair56ProblemError, validate_problem as validate_pair56
from mdr.phase_group_audit import CONTROLS, audit_all
from mdr.state import FACE_ORDER, FullState, Move


class PhaseGroupAuditTests(unittest.TestCase):
    def test_chain_contract_contains_all_four_individual_phases(self) -> None:
        self.assertEqual(
            CONTROLS["pair34"]["groups"],
            (("G5", 7), ("G6", 6), ("G7", 5)),
        )
        self.assertEqual(
            CONTROLS["pair56"]["groups"],
            (("G7", 5), ("G8", 4), ("G9", 3)),
        )
        self.assertEqual(
            CONTROLS["pair34"]["factors"], (208_099_584, 68_400)
        )
        self.assertEqual(
            CONTROLS["pair56"]["factors"], (64_157_184, 25_945_920)
        )

    def test_committed_actions_are_bound_to_full_state_conjugacy(self) -> None:
        for pair, validate, error in (
            ("pair34", validate_pair34, Pair34ProblemError),
            ("pair56", validate_pair56, Pair56ProblemError),
        ):
            with self.subTest(pair=pair):
                problem = json.loads(
                    (ROOT / "training" / pair / "problem.json").read_text(
                        encoding="utf-8"
                    )
                )
                corrupted = copy.deepcopy(problem)
                corrupted["actions"][:4], corrupted["actions"][4:8] = (
                    corrupted["actions"][4:8],
                    corrupted["actions"][:4],
                )
                with self.assertRaisesRegex(error, "FullStateV1"):
                    validate(corrupted)

    def test_phase_coordinates_have_exact_generator_boundary_contract(self) -> None:
        solved = FullState.solved()
        for phase, source_face_count in ((3, 7), (4, 6), (5, 5), (6, 4)):
            with self.subTest(phase=phase):
                root = phase_index(solved, phase)
                for face in FACE_ORDER[: source_face_count - 1]:
                    for power in range(1, 5):
                        self.assertEqual(
                            phase_index(solved.apply((Move(face, power),)), phase),
                            root,
                        )
                removed_face = FACE_ORDER[source_face_count - 1]
                for power in range(1, 5):
                    self.assertNotEqual(
                        phase_index(
                            solved.apply((Move(removed_face, power),)), phase
                        ),
                        root,
                    )

    @unittest.skipUnless(
        importlib.util.find_spec("sympy") is not None,
        "SymPy proof dependency is not installed",
    )
    def test_pinned_manifests_pass_exact_schreier_sims_audit(self) -> None:
        result = audit_all(ROOT)
        self.assertTrue(result["valid"])
        self.assertEqual(
            result["pairs"]["pair34"]["transitions"][0]["index"],
            208_099_584,
        )
        self.assertEqual(
            result["pairs"]["pair56"]["transitions"][1]["index"],
            25_945_920,
        )


if __name__ == "__main__":
    unittest.main()
