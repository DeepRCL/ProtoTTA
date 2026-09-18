import tempfile
import unittest
from pathlib import Path

import paired_adaptation_supervision as paired


class PairedAdaptationSupervisionTest(unittest.TestCase):
    def test_prompt_has_predictions_but_not_ground_truth(self):
        prompt = paired.build_prompt("full_reasoning", "Predicted A", "Predicted B")
        self.assertIn("Predicted A", prompt)
        self.assertIn("Predicted B", prompt)
        self.assertNotIn("Hidden Ground Truth", prompt)
        self.assertIn("randomly ordered", prompt)

    def test_state_order_is_deterministic_and_balanced(self):
        first = paired.state_order("tent", "sample_1_noise")
        self.assertEqual(first, paired.state_order("tent", "sample_1_noise"))
        orders = {paired.state_order("tent", f"sample_{index}") for index in range(30)}
        self.assertEqual(orders, {("before", "after"), ("after", "before")})

    def test_changed_pairs_and_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = [
                ("improved", False, True, 1, 2),
                ("harmed", True, False, 2, 1),
                ("same_wrong", False, False, 3, 4),
                ("unchanged_prediction", True, True, 2, 2),
            ]
            for sample_id, before_correct, after_correct, before_pred, after_pred in examples:
                for method, correct, prediction in (
                    ("unadapted", before_correct, before_pred),
                    ("tent", after_correct, after_pred),
                ):
                    sample_dir = root / method / "samples" / sample_id
                    sample_dir.mkdir(parents=True)
                    paired.write_json(
                        sample_dir / "03_meta.json",
                        {
                            "sample_idx": 0,
                            "predicted_index": prediction,
                            "predicted_class": f"Bird {prediction}",
                            "ground_truth_class": "Bird 2",
                            "is_correct": correct,
                        },
                    )
            pairs = paired.changed_pairs(root, "tent")
            self.assertEqual(len(pairs), 3)
            outcomes = {item["sample_id"]: paired.outcome(item["before"], item["after"]) for item in pairs}
            self.assertEqual(outcomes["improved"], "improved")
            self.assertEqual(outcomes["harmed"], "harmed")
            self.assertEqual(outcomes["same_wrong"], "neither_correct")


if __name__ == "__main__":
    unittest.main()
