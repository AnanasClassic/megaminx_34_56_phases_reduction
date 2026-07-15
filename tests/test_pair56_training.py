import unittest
from pathlib import Path

import torch

from mdr.config import ROOT
from mdr.pair56_problem import K_MAX, full_state_to_pair56, load_problem
from mdr.pair56_training import (
    build_allowed_moves,
    make_generator,
    sample_rw_middle_batch,
    sparse_q_metrics,
)
from mdr.pair56_testing import batched_beam_solve
from mdr.qmlp import PairQMLP, count_parameters
from mdr.state import FullState, Move


class Pair56TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem_path = ROOT / "training" / "pair56" / "problem.json"
        cls.problem = load_problem(cls.problem_path)

    def test_problem_contract(self) -> None:
        self.assertEqual(self.problem["K_max"], 26)
        self.assertEqual(K_MAX, 26)
        self.assertEqual(self.problem["num_classes"], 67)
        self.assertEqual(len(self.problem["actions"]), 20)
        self.assertEqual(len(self.problem["black_positions"]), 54)
        self.assertEqual(len(self.problem["fixed_positions"]), 66)
        self.assertEqual(self.problem["p900_power_for_verifier_power"], [4, 3, 2, 1])
        self.assertEqual(sorted(self.problem["verifier_conjugacy"]), list(range(120)))
        self.assertEqual(self.problem["expected_space_size"], 1_664_617_163_489_280)
        self.assertEqual(self.problem["target_stabilizer_order"], self.problem["target_group_order"])
        self.assertEqual(
            self.problem["source_group_order"] // self.problem["target_group_order"],
            self.problem["expected_space_size"],
        )

    def test_small_model_shape_and_size(self) -> None:
        model = PairQMLP(
            state_size=120, num_classes=67, actions=20,
            hd1=64, hd2=256, residual_blocks=2,
        )
        model.eval()
        output = model(torch.tensor(self.problem["target"]).repeat(3, 1))
        self.assertEqual(tuple(output.shape), (3, 20))
        self.assertEqual(count_parameters(model), 802_260)

    def test_full_state_conversion_intertwines_all_pair56_moves(self) -> None:
        solved = FullState.solved()
        target = full_state_to_pair56(solved, self.problem)
        self.assertEqual(target, self.problem["target"])
        for move_id, name in enumerate(self.problem["names"]):
            moved = solved.apply((Move(name[:-1], int(name[-1])),))
            observed = full_state_to_pair56(moved, self.problem)
            expected = [target[index] for index in self.problem["actions"][move_id]]
            self.assertEqual(observed, expected, name)

    def test_sampler_uses_real_nontrivial_coset_steps(self) -> None:
        actions = torch.tensor(self.problem["actions"], dtype=torch.int64)
        target = torch.tensor(self.problem["target"], dtype=torch.int64)
        inverses = torch.tensor(self.problem["inverse_actions"], dtype=torch.int64)
        face_ids = torch.tensor(self.problem["face_ids"], dtype=torch.int64)
        allowed = build_allowed_moves(face_ids)
        batch = sample_rw_middle_batch(
            batch_size=64, K_min=2, target=target, actions=actions,
            inverse_actions=inverses, face_ids=face_ids, allowed_moves=allowed,
            generator=make_generator(torch.device("cpu"), 123), debug=True,
        )
        states, pivots, previous, following = batch
        rows = torch.arange(states.size(0))
        previous_states = torch.gather(states, 1, actions[previous])
        following_states = torch.gather(states, 1, actions[following])
        self.assertTrue(bool((previous_states != states).any(dim=1).all()))
        self.assertTrue(bool((following_states != states).any(dim=1).all()))
        self.assertTrue(bool((face_ids[previous] != face_ids[following]).all()))
        predictions = torch.zeros((states.size(0), 20))
        loss, accuracy = sparse_q_metrics(predictions, pivots, previous, following)
        self.assertGreater(float(loss), 0)
        self.assertEqual(float(accuracy), 0)

    def test_batched_beam_replays_one_move_states(self) -> None:
        actions = torch.tensor(self.problem["actions"], dtype=torch.int64)
        target = torch.tensor(self.problem["target"], dtype=torch.int64)
        face_ids = torch.tensor(self.problem["face_ids"], dtype=torch.int64)
        model = PairQMLP(state_size=120, num_classes=67, actions=20)
        model.eval()
        moved = target.index_select(0, actions[12])
        solutions = batched_beam_solve(
            model=model, initials=torch.stack((target, moved)), target=target,
            actions=actions, face_ids=face_ids, beam_width=20, max_steps=1,
            amp="fp32",
        )
        self.assertEqual(solutions[0], ())
        self.assertIsNotNone(solutions[1])
        replayed = moved.index_select(0, actions[solutions[1][0]])
        self.assertTrue(torch.equal(replayed, target))


if __name__ == "__main__":
    unittest.main()
