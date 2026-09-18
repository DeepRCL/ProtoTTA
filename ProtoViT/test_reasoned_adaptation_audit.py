import tempfile
import unittest
from pathlib import Path

import reasoned_adaptation_audit as audit


class ReasonedAdaptationAuditTest(unittest.TestCase):
    def test_prompt_is_explicit_continuous_and_label_blind(self):
        prompt = audit.build_prompt("prototta", "full_reasoning", "Before Bird", "After Bird")
        self.assertIn("BEFORE is the frozen, unadapted source model", prompt)
        self.assertIn("AFTER is ProtoTTA after online CONTINUOUS adaptation", prompt)
        self.assertIn("not reset for this image", prompt)
        self.assertIn("Before Bird", prompt)
        self.assertIn("After Bird", prompt)
        self.assertNotIn("Secret Ground Truth", prompt)
        self.assertIn("ACCEPT", prompt)
        self.assertIn("ROLLBACK", prompt)

    def test_condition_prompts_are_matched_except_evidence(self):
        prompts = {
            condition: audit.build_prompt("tent", condition, "Bird One", "Bird Two")
            for condition in audit.CONDITION_ORDER
        }
        for prompt in prompts.values():
            self.assertIn("BEFORE prediction: Bird One", prompt)
            self.assertIn("AFTER prediction: Bird Two", prompt)
            self.assertIn('"adaptation_score"', prompt)
        self.assertNotIn("prototype evidence", prompts["predictions_only"].split("CONTROL CONDITION:")[0])
        self.assertIn("side-by-side evidence canvas", prompts["full_reasoning"])

    def test_audit_validation(self):
        valid = {
            "before_evidence_quality": 2,
            "after_evidence_quality": 4,
            "adaptation_score": 3,
            "adaptation_quality": "improved",
            "recommended_action": "accept",
            "focus_change": "Focus moved to the head.",
            "prototype_change": "Prototype matches became coherent.",
            "comparative_analysis": "The after state has stronger visible support.",
            "confidence": 4,
        }
        cleaned = audit.validate_audit(valid)
        self.assertEqual(cleaned["recommended_action"], "ACCEPT")
        self.assertEqual(cleaned["adaptation_quality"], "IMPROVED")
        with self.assertRaises(ValueError):
            audit.validate_audit({**valid, "adaptation_score": 6})

    def test_decisive_pair_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = [
                ("improved", False, True, 1, 2),
                ("harmed", True, False, 2, 1),
                ("both_wrong", False, False, 3, 4),
                ("both_correct", True, True, 2, 2),
            ]
            for sample_id, before_correct, after_correct, before_pred, after_pred in examples:
                for method, correct, prediction in (
                    ("unadapted", before_correct, before_pred),
                    ("tent", after_correct, after_pred),
                ):
                    sample_dir = root / method / "samples" / sample_id
                    sample_dir.mkdir(parents=True)
                    audit.write_json(
                        sample_dir / "03_meta.json",
                        {
                            "predicted_index": prediction,
                            "predicted_class": f"Bird {prediction}",
                            "ground_truth_class": "Bird 2",
                            "is_correct": correct,
                        },
                    )
            pairs = audit.decisive_pairs(root, "tent")
            self.assertEqual({pair["sample_id"] for pair in pairs}, {"improved", "harmed"})


if __name__ == "__main__":
    unittest.main()
