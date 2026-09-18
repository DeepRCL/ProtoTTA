import tempfile
import unittest
from pathlib import Path

from PIL import Image

import failure_detection as fd


class FailureDetectionTest(unittest.TestCase):
    def test_prompt_contains_prediction_but_not_hidden_label(self):
        prompt = fd.build_prompt("Predicted Finch")
        self.assertIn("Predicted Finch", prompt)
        self.assertNotIn("Hidden Sparrow", prompt)

    def test_sanitize_board_covers_label_title(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            destination = Path(temporary) / "blind.png"
            image = Image.new("RGB", (400, 300), "blue")
            for x in range(400):
                for y in range(30):
                    image.putpixel((x, y), (255, 0, 0))
            image.save(source)
            fd.sanitize_board(source, destination, "test board")
            with Image.open(destination) as blinded:
                self.assertNotEqual(blinded.getpixel((5, 5)), (255, 0, 0))
                self.assertEqual(blinded.getpixel((5, 200)), (0, 0, 255))

    def test_collect_and_summarize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = [0, 0, 1, 1, 0, 1]
            for index, is_wrong in enumerate(labels):
                sample_dir = root / "unadapted" / "samples" / f"sample_{index}_noise"
                sample_dir.mkdir(parents=True)
                fd.write_json(
                    sample_dir / "03_meta.json",
                    {
                        "sample_idx": index,
                        "corruption_type": "noise",
                        "predicted_class": "Bird A",
                        "ground_truth_class": "Bird B" if is_wrong else "Bird A",
                        "is_correct": not is_wrong,
                        "max_softmax_probability": 0.2 if is_wrong else 0.9,
                        "msp_prediction_matches_saved": True,
                    },
                )
                fd.write_json(
                    sample_dir / fd.SCORE_NAME,
                    {
                        "label_blind": True,
                        "overall_quality_score": 1 if is_wrong else 5,
                        "failure_rationale": "Synthetic test rationale.",
                    },
                )

            records, missing = fd.collect_records(root, ["unadapted"])
            self.assertEqual(len(records), len(labels))
            self.assertEqual(sum(missing.values()), 0)
            summary = fd.summarize_group(records, resamples=30, seed=1)
            self.assertEqual(summary["n"], len(labels))
            self.assertAlmostEqual(summary["metrics"]["MSP"]["auroc"], 1.0)
            self.assertAlmostEqual(summary["metrics"]["VLM quality"]["aupr"], 1.0)


if __name__ == "__main__":
    unittest.main()
