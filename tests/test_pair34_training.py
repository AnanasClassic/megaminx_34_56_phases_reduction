import unittest

import torch

from mdr.config import ROOT
from mdr.pair34_problem import K_MAX, load_problem
from mdr.pair56_problem import full_state_to_pair56
from mdr.pair56_training import build_allowed_moves, make_generator, sample_rw_middle_batch, sparse_q_metrics
from mdr.qmlp import PairQMLP, count_parameters
from mdr.state import FullState, Move


class Pair34TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = load_problem(ROOT / "training" / "pair34" / "problem.json")

    def test_exact_problem_contract(self) -> None:
        self.assertEqual(K_MAX, 22)
        self.assertEqual(self.problem["K_max"], 22)
        self.assertEqual(self.problem["transition"], "G5->G7")
        self.assertEqual(len(self.problem["actions"]), 28)
        self.assertEqual(self.problem["num_classes"], 43)
        self.assertEqual(len(self.problem["fixed_positions"]), 42)
        self.assertEqual(len(self.problem["black_positions"]), 78)
        self.assertEqual(self.problem["expected_space_size"], 14_234_011_545_600)
        self.assertEqual(self.problem["target_stabilizer_order"], self.problem["target_group_order"])
        self.assertEqual(
            self.problem["source_group_order"] // self.problem["target_group_order"],
            self.problem["expected_space_size"],
        )

    def test_model_shape_and_size(self) -> None:
        model = PairQMLP(state_size=120, num_classes=43, actions=28)
        model.eval()
        output = model(torch.tensor(self.problem["target"]).repeat(3, 1))
        self.assertEqual(tuple(output.shape), (3, 28))
        self.assertEqual(count_parameters(model), 619_996)

    def test_all_moves_intertwine_full_state_and_model_coordinate(self) -> None:
        solved = FullState.solved()
        target = full_state_to_pair56(solved, self.problem)
        self.assertEqual(target, self.problem["target"])
        for move_id, name in enumerate(self.problem["names"]):
            moved = solved.apply((Move(name[:-1], int(name[-1])),))
            observed = full_state_to_pair56(moved, self.problem)
            expected = [target[index] for index in self.problem["actions"][move_id]]
            self.assertEqual(observed, expected, name)

    def test_sampler_uses_nontrivial_g5_g7_edges(self) -> None:
        actions = torch.tensor(self.problem["actions"], dtype=torch.int64)
        target = torch.tensor(self.problem["target"], dtype=torch.int64)
        inverses = torch.tensor(self.problem["inverse_actions"], dtype=torch.int64)
        face_ids = torch.tensor(self.problem["face_ids"], dtype=torch.int64)
        allowed = build_allowed_moves(face_ids)
        states, pivots, previous, following = sample_rw_middle_batch(
            batch_size=64, K_min=2, K_max=22, target=target, actions=actions,
            inverse_actions=inverses, face_ids=face_ids, allowed_moves=allowed,
            generator=make_generator(torch.device("cpu"), 321), debug=True,
        )
        previous_states = torch.gather(states, 1, actions[previous])
        following_states = torch.gather(states, 1, actions[following])
        self.assertTrue(bool((previous_states != states).any(dim=1).all()))
        self.assertTrue(bool((following_states != states).any(dim=1).all()))
        self.assertTrue(bool((face_ids[previous] != face_ids[following]).all()))
        loss, accuracy = sparse_q_metrics(torch.zeros((64, 28)), pivots, previous, following)
        self.assertGreater(float(loss), 0)
        self.assertEqual(float(accuracy), 0)


if __name__ == "__main__":
    unittest.main()
