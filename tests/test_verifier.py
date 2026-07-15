import os
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

from mdr.config import ROOT
from mdr.dual_verify import main as dual_verify_main
from mdr.state import FACE_ORDER, FullState, Move, StateError, format_word, in_target, invert_word, parse_word


PRIMARY = ROOT / "build" / "mdr-verify"


class IndependentStateTests(unittest.TestCase):
    def test_every_generator_has_order_five(self) -> None:
        solved = FullState.solved()
        for face in FACE_ORDER:
            with self.subTest(face=face):
                self.assertEqual(solved.apply([Move(face, 1)] * 5), solved)

    def test_words_and_inverses(self) -> None:
        rng = random.Random(20260714)
        solved = FullState.solved()
        for _ in range(100):
            word = tuple(Move(rng.choice(FACE_ORDER), rng.randrange(1, 5)) for _ in range(rng.randrange(0, 80)))
            self.assertEqual(solved.apply(word).apply(invert_word(word)), solved)

    def test_serialization_round_trip(self) -> None:
        state = FullState.solved().apply(parse_word(b"U1 R2 FR3 D4\n"))
        self.assertEqual(FullState.decode(state.encode()), state)
        self.assertEqual(len(state.encode()), 108)

    def test_strict_word_rejections(self) -> None:
        for malformed in (b"U", b"U0", b"U5", b"X1", b" U1", b"U1 ", b"U1  R1", b"U1\nR1\n", b"U1\r\n"):
            with self.subTest(malformed=malformed), self.assertRaises(StateError):
                parse_word(malformed)

    def test_subgroup_predicates(self) -> None:
        solved = FullState.solved()
        for target, allowed, removed in (
            ("g5", "FR", "FL"),
            ("g6", "BL", "FR"),
            ("g7", "BR", "BL"),
            ("g8", "L", "BR"),
            ("g9", "F", "L"),
        ):
            with self.subTest(target=target):
                self.assertTrue(in_target(solved.apply([Move(allowed, 1)]), target))
                self.assertFalse(in_target(solved.apply([Move(removed, 1)]), target))

    def test_corrupt_states_are_rejected(self) -> None:
        good = bytearray(FullState.solved().encode())
        cases = []
        bad_magic = good.copy(); bad_magic[0] ^= 1; cases.append(bad_magic)
        duplicate_edge = good.copy(); duplicate_edge[8] = duplicate_edge[9]; cases.append(duplicate_edge)
        odd_edge_orientation = good.copy(); odd_edge_orientation[38] = 1; cases.append(odd_edge_orientation)
        bad_corner_orientation = good.copy(); bad_corner_orientation[88] = 3; cases.append(bad_corner_orientation)
        for data in cases:
            with self.assertRaises(StateError):
                FullState.decode(bytes(data))


@unittest.skipUnless(PRIMARY.is_file() or os.environ.get("MDR_REQUIRE_GO_VERIFIER") != "1", "primary verifier is required")
class PrimaryCrossCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("MDR_REQUIRE_GO_VERIFIER") == "1" and not PRIMARY.is_file():
            raise RuntimeError("primary verifier is absent")
        if not PRIMARY.is_file():
            raise unittest.SkipTest("primary verifier is not built")

    def go_state(self, word: tuple[Move, ...], directory: Path) -> bytes:
        word_path = directory / "word.txt"
        state_path = directory / "state.bin"
        word_path.write_bytes(format_word(word))
        subprocess.run([str(PRIMARY), "make-state", "--moves", str(word_path), "--out", str(state_path)], check=True)
        return state_path.read_bytes()

    def test_all_moves_and_random_words_match(self) -> None:
        rng = random.Random(314159)
        words = [(Move(face, power),) for face in FACE_ORDER for power in range(1, 5)]
        words.extend(tuple(Move(rng.choice(FACE_ORDER), rng.randrange(1, 5)) for _ in range(rng.randrange(0, 100))) for _ in range(200))
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            for index, word in enumerate(words):
                with self.subTest(index=index):
                    expected = FullState.solved().apply(word).encode()
                    self.assertEqual(self.go_state(word, directory), expected)

    def test_dual_verifier_accepts_valid_solution_and_rejects_bound(self) -> None:
        word = parse_word(b"U1 R2 FR3 D4\n")
        inverse = invert_word(word)
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            state = directory / "state.bin"
            solution = directory / "solution.txt"
            state.write_bytes(FullState.solved().apply(word).encode())
            solution.write_bytes(format_word(inverse))
            self.assertEqual(dual_verify_main(["--state", str(state), "--solution", str(solution), "--max-length", str(len(inverse))]), 0)
            self.assertEqual(dual_verify_main(["--state", str(state), "--solution", str(solution), "--max-length", str(len(inverse) - 1)]), 2)

    def test_upstream_coordinates_are_left_cosets_and_match_targets(self) -> None:
        rng = random.Random(20260714)
        phase_rows = {3: (7, "g6"), 4: (6, "g7"), 5: (5, "g8"), 6: (4, "g9")}
        with tempfile.TemporaryDirectory() as directory_text:
            state_path = Path(directory_text) / "state.bin"

            def phase_index(state: FullState, phase: int) -> dict[str, int]:
                state.write(state_path)
                result = subprocess.run(
                    [str(PRIMARY), "phase-index", "--state", str(state_path), "--phase", str(phase)],
                    check=True, text=True, stdout=subprocess.PIPE,
                )
                return json.loads(result.stdout)

            for phase, (source_face_count, target) in phase_rows.items():
                target_faces = FACE_ORDER[:source_face_count - 1]
                source_faces = FACE_ORDER[:source_face_count]
                solved_index = phase_index(FullState.solved(), phase)["solved_index"]
                append_mismatches = 0
                for _ in range(30):
                    prefix = tuple(Move(rng.choice(source_faces), rng.randrange(1, 5)) for _ in range(12))
                    suffix = tuple(Move(rng.choice(target_faces), rng.randrange(1, 5)) for _ in range(12))
                    state = FullState.solved().apply(prefix)
                    self.assertEqual(
                        phase_index(state, phase)["index"],
                        phase_index(FullState.solved().apply(suffix + prefix), phase)["index"],
                    )
                    append_mismatches += (
                        phase_index(state, phase)["index"]
                        != phase_index(state.apply(suffix), phase)["index"]
                    )
                    inside = FullState.solved().apply(suffix)
                    self.assertEqual(
                        phase_index(inside, phase)["index"] == solved_index,
                        in_target(inside, target),
                    )
                self.assertGreater(append_mismatches, 0, f"phase {phase} did not distinguish coset side")


if __name__ == "__main__":
    unittest.main()
