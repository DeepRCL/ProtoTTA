import unittest

import vlm_gate_diagnostics as diagnostic


class VLMGateDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "before_prediction": "Bird A",
            "after_prediction": "Bird B",
            "before_msp": 0.4,
            "after_msp": 0.8,
        }

    def test_order_is_deterministic_and_reverse_is_exact(self):
        order = diagnostic.deterministic_order("fog", "fog::bird.jpg")
        self.assertEqual(order[::-1], diagnostic.deterministic_order("fog", "fog::bird.jpg", reverse=True))
        self.assertEqual(set(order), {"before", "after"})

    def test_nomsp_prompt_is_label_blind_and_omits_confidence(self):
        prompt = diagnostic.build_prompt(self.record, ("after", "before"), "image_nomsp")
        self.assertIn("STATE A is AFTER continuous ProtoTTA", prompt)
        self.assertIn("STATE B is BEFORE frozen source", prompt)
        self.assertNotIn("MSP", prompt)
        self.assertNotIn("ground truth class", prompt.lower())

    def test_msp_prompt_includes_matched_confidence(self):
        prompt = diagnostic.build_prompt(self.record, ("after", "before"), "image_msp")
        self.assertIn("MSP 0.800", prompt)
        self.assertIn("MSP 0.400", prompt)

    def test_candidate_gallery_prompt_warns_against_self_confirmation(self):
        prompt = diagnostic.build_prompt(
            self.record, ("before", "after"), "candidate_gallery_nomsp"
        )
        self.assertIn("candidate-aligned comparison", prompt)
        self.assertIn("not proof", prompt)
        self.assertNotIn("MSP", prompt)

    def test_reasoned_prompt_has_explicit_roles_and_comparative_schema(self):
        prompt = diagnostic.build_reasoned_prompt(
            self.record, ("after", "before"), "reasoned_rich_nomsp"
        )
        self.assertIn("STATE A is AFTER continuous ProtoTTA", prompt)
        self.assertIn("STATE B is BEFORE frozen source", prompt)
        self.assertIn("adaptation_score", prompt)
        self.assertIn("3-6", prompt)

    def test_reasoned_validation(self):
        parsed = diagnostic.validate_reasoned(
            {
                "chosen_state": "b",
                "adaptation_score": -2,
                "state_a_evidence_quality": 2,
                "state_b_evidence_quality": 4,
                "focus_comparison": "B focuses more clearly on the head.",
                "prototype_comparison": "B retrieves anatomically closer examples.",
                "comparative_analysis": "B has more coherent visible evidence than A.",
                "confidence": 4,
            }
        )
        self.assertEqual(parsed["chosen_state"], "B")
        self.assertEqual(parsed["adaptation_score"], -2)

    def test_skeptical_prompt_locks_image_decision(self):
        prior = {"chosen_state": "B", "comparative_analysis": "B fits the visible bird."}
        prompt = diagnostic.build_skeptical_prompt(
            self.record, ("before", "after"), prior
        )
        self.assertIn("LOCKED IMAGE-ONLY DECISION: STATE B", prompt)
        self.assertIn("Override", prompt)
        self.assertIn("BACKGROUND_FOCUS", prompt)

    def test_skeptical_override_requires_flag(self):
        payload = {
            "chosen_state": "A",
            "adaptation_score": -2,
            "override_image_decision": True,
            "failure_flags": ["BACKGROUND_FOCUS"],
            "state_a_evidence_quality": 4,
            "state_b_evidence_quality": 2,
            "focus_comparison": "B focuses on background.",
            "prototype_comparison": "A has a closer visual match.",
            "comparative_analysis": "B's focus is visibly outside the bird.",
            "confidence": 4,
        }
        parsed = diagnostic.validate_skeptical(payload, "B")
        self.assertTrue(parsed["override_image_decision"])
        self.assertEqual(parsed["failure_flags"], ["BACKGROUND_FOCUS"])


if __name__ == "__main__":
    unittest.main()
