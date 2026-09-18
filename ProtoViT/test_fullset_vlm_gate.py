import tempfile
import unittest
from pathlib import Path

from PIL import Image

import fullset_vlm_gate as gate
import vlm_eval


class FullsetVLMGateTest(unittest.TestCase):
    def test_prompt_is_batched_explicit_and_label_blind(self):
        record = {
            "before_prediction": "Bird A",
            "after_prediction": "Bird B",
            "before_msp": 0.4,
            "after_msp": 0.7,
        }
        prompt = gate.build_batch_prompt("full_reasoning", [("S01", record)])
        self.assertIn("BEFORE is the frozen source model", prompt)
        self.assertIn("AFTER accumulated online updates", prompt)
        self.assertIn("Bird A", prompt)
        self.assertIn("Bird B", prompt)
        self.assertIn("paired reasoning canvas", prompt)
        self.assertNotIn("Hidden Ground Truth", prompt)

    def test_batch_response_validation(self):
        payload = {
            "decisions": [
                {"id": "S01", "action": "accept", "adaptation_score": 3, "rationale": "Better evidence."},
                {"id": "S02", "action": "ROLLBACK", "adaptation_score": -2, "rationale": "Evidence degraded."},
            ]
        }
        result = gate.validate_batch_response(payload, ["S01", "S02"])
        self.assertEqual(result["S01"]["action"], "ACCEPT")
        self.assertEqual(result["S02"]["adaptation_score"], -2)
        with self.assertRaises(ValueError):
            gate.validate_batch_response({"decisions": payload["decisions"][:1]}, ["S01", "S02"])

    def test_nested_json_extraction_keeps_outer_batch(self):
        raw = 'preface {"decisions": [{"id": "S01", "action": "ACCEPT"}]} suffix'
        parsed = vlm_eval.extract_json_fragment(raw)
        self.assertEqual(parsed["decisions"][0]["id"], "S01")

    def test_compact_canvas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.png"
            Image.new("RGB", (224, 224), "#778899").save(raw)
            proto_dir = root / "prototypes"
            proto_dir.mkdir()
            for index, color in enumerate(("red", "green", "blue", "orange")):
                Image.new("RGB", (120, 120), color).save(proto_dir / f"prototype-img-original{index}.png")
            prototype = lambda index, cls: {
                "proto_idx": index,
                "class_index": cls,
                "class_name": f"Bird {cls}",
                "contribution": 2.0,
                "patch_locations": [[5, 6, 5, 6], [7, 7, 8, 8]],
                "slots": [1, 1, 1, 1],
            }
            record = {
                "before_prediction": "Bird 0",
                "after_prediction": "Bird 1",
                "before_prediction_index": 0,
                "after_prediction_index": 1,
                "before_msp": 0.5,
                "after_msp": 0.6,
            }
            evidence = {
                "before_predicted": [prototype(0, 0)],
                "before_any": [prototype(1, 1), prototype(2, 2), prototype(3, 3)],
                "after_predicted": [prototype(1, 1)],
                "after_any": [prototype(0, 0), prototype(2, 2), prototype(3, 3)],
            }
            destination = root / "canvas.jpg"
            gate.compact_evidence_canvas(raw, record, evidence, proto_dir, destination)
            with Image.open(destination) as image:
                self.assertEqual(image.size, (1744, 1030))

    def test_development_manifest_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subset.json"
            gate.write_json(
                path,
                {"samples": [{"corruption_type": "fog", "image_path": "001/bird.jpg"}]},
            )
            self.assertEqual(gate.development_keys(path), {"fog::001/bird.jpg"})


if __name__ == "__main__":
    unittest.main()
