import copy
import unittest

from mdr.config import ConfigError, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_pinned_config_is_valid(self) -> None:
        config = load_config()
        self.assertEqual(config["pairs"]["pair34"]["expected_raw_pairs"], 212 * 2531)
        self.assertEqual(config["pairs"]["pair56"]["expected_raw_pairs"], 3484 * 117)

    def test_raw_count_cannot_drift(self) -> None:
        config = load_config()
        broken = copy.deepcopy(config)
        broken["pairs"]["pair34"]["expected_raw_pairs"] += 1
        with self.assertRaisesRegex(ConfigError, "raw product mismatch"):
            validate_config(broken)

    def test_metric_cannot_drift(self) -> None:
        config = load_config()
        broken = copy.deepcopy(config)
        broken["metric"]["cost_per_non_identity_face_turn"] = 2
        with self.assertRaises(ConfigError):
            validate_config(broken)


if __name__ == "__main__":
    unittest.main()
