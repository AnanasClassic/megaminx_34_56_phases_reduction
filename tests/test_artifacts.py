import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mdr.artifacts import ArtifactError, atomic_write_json, load_table_metadata, verify_payloads
from mdr.config import ROOT, load_config


class ArtifactTests(unittest.TestCase):
    def test_published_histogram_controls_have_expected_totals(self) -> None:
        controls = json.loads((ROOT / "controls" / "published_histograms.json").read_text())
        expected = {3: 208099584, 4: 68400, 5: 64157184, 6: 25945920}
        for phase, total in expected.items():
            self.assertEqual(sum(controls["phases"][str(phase)]), total)

    def test_atomic_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})
            self.assertEqual(list(Path(directory).glob("*.partial")), [])

    def test_payload_checksum_is_enforced(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "depths.bin"
            payload.write_bytes(b"depths")
            metadata = {
                "schema_version": 1,
                "phase": 4,
                "transition": "G6->G7",
                "repository_commit": config["upstream"]["commit"],
                "upstream_commit": config["upstream"]["commit"],
                "metric": "FTM",
                "generators": [
                    f"{face}{power}"
                    for face in config["metric"]["faces"][:6]
                    for power in config["metric"]["powers"]
                ],
                "state_count": 68400,
                "diameter": 8,
                "antipode_count": 2531,
                "payloads": {
                    "depths.bin": {
                        "sha256": hashlib.sha256(b"depths").hexdigest(),
                        "bytes": 6,
                    }
                },
                "complete": True,
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata))
            loaded = load_table_metadata(metadata_path)
            verify_payloads(metadata_path, loaded)
            payload.write_bytes(b"changed")
            with self.assertRaisesRegex(ArtifactError, "checksum mismatch"):
                verify_payloads(metadata_path, loaded)

    def test_incomplete_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps({"schema_version": 1}))
            with self.assertRaisesRegex(ArtifactError, "missing"):
                load_table_metadata(path)


if __name__ == "__main__":
    unittest.main()
