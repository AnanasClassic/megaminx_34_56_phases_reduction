import tempfile
import unittest
from pathlib import Path

from mdr.compositions import compose
from mdr.equivalences import inverse, rotations
from mdr.hard_states import HardStateError, read_hard
from mdr.state import FullState, parse_word


class GroupSemanticsTests(unittest.TestCase):
    def test_full_state_composition_matches_word_concatenation(self) -> None:
        left_word = parse_word(b"U1 R2 FR3")
        right_word = parse_word(b"L4 BL1 F2")
        left = FullState.solved().apply(left_word)
        right = FullState.solved().apply(right_word)
        self.assertEqual(compose(left, right), FullState.solved().apply(left_word + right_word))

    def test_inversion_is_exact_group_inverse(self) -> None:
        state = FullState.solved().apply(parse_word(b"U1 R2 FR3 D4 BL2"))
        self.assertEqual(compose(state, inverse(state)), FullState.solved())
        self.assertEqual(compose(inverse(state), state), FullState.solved())

    def test_rotation_enumerator_is_the_full_orientation_preserving_group(self) -> None:
        generated = rotations()
        self.assertEqual(len(generated), 60)
        self.assertEqual(len({tuple(row[face] for face in row) for row in generated}), 60)

    def test_hard_record_corruption_is_rejected_when_artifact_exists(self) -> None:
        source = Path(__file__).resolve().parents[1] / "hard_states" / "phase6_depth13.bin"
        if not source.is_file():
            self.skipTest("generated M2 artifact is absent")
        corrupted = bytearray(source.read_bytes())
        corrupted[24] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard.bin"
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(HardStateError, "checksum"):
                read_hard(path)


if __name__ == "__main__":
    unittest.main()
