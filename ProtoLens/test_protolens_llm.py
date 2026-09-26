import unittest

from protolens_llm_common import (
    build_prompt,
    parse_decisions,
    reject_label_fields,
    select_prediction,
    task_for_index,
)


class ProtoLensLLMTest(unittest.TestCase):
    def test_task_mapping_covers_corruption_and_severity(self):
        self.assertEqual(task_for_index(0), ("qwerty", 20))
        self.assertEqual(task_for_index(19), ("aggressive", 80))
        with self.assertRaises(ValueError):
            task_for_index(20)

    def test_label_fields_are_rejected(self):
        reject_label_fields({"prediction": "positive", "score": 0.8})
        with self.assertRaises(ValueError):
            reject_label_fields({"ground_truth": "positive"})

    def test_prompt_and_response_are_label_blind_and_strict(self):
        record = {
            "corrupted_review": "The service was unexpectedly good.",
            "before_prediction": "negative",
            "after_prediction": "positive",
            "before_msp": 0.6,
            "after_msp": 0.8,
        }
        prompt = build_prompt([record], "output_only")
        self.assertIn("Ground truth and correctness are withheld", prompt)
        decisions = parse_decisions(
            '{"decisions":[{"id":"S01","action":"ACCEPT",'
            '"adaptation_score":4,"rationale":"The review supports positive."}]}',
            ["S01"],
        )
        self.assertEqual(decisions[0]["adaptation_score"], 4)

    def test_prediction_selector(self):
        self.assertEqual(select_prediction(0, 1, -5), 0)
        self.assertEqual(select_prediction(0, 1, 2), 1)
        self.assertEqual(select_prediction(1, 1, -5), 1)


if __name__ == "__main__":
    unittest.main()
