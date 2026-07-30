import unittest

from scripts.run_local_vlm_track_labeling import (
    compact_to_agent,
    extract_json,
    local_prompt,
    safe_unknown,
)


class LocalVlmTrackLabelingTest(unittest.TestCase):
    def test_extract_json_accepts_fenced_model_output(self):
        self.assertEqual(
            extract_json('```json\n{"label":"team_a"}\n```'),
            {"label": "team_a"},
        )

    def test_local_prompt_is_compact(self):
        seeds = [{"reviewed_label": "team_a"}, {"reviewed_label": "team_b"}]
        prompt = local_prompt(
            {
                "key": "fixture-h1-0001:1",
                "track_id": 1,
                "clip_id": "fixture-h1-0001",
            },
            seeds,
        )
        self.assertIn("1=team_a, 2=team_b", prompt)
        self.assertIn('"matched_reference":0', prompt)
        self.assertLess(len(prompt), 1000)

    def test_compact_team_requires_same_label_reference(self):
        seeds = [{"reviewed_label": "team_a"}, {"reviewed_label": "team_b"}]
        valid = compact_to_agent(
            {
                "label": "team_a",
                "matched_reference": 1,
                "official_evidence": False,
                "reason": "Uniform matches reference one.",
            },
            seeds,
        )
        self.assertEqual(valid["label"], "team_a")
        with self.assertRaisesRegex(ValueError, "same-team"):
            compact_to_agent(
                {
                    "label": "team_a",
                    "matched_reference": 2,
                    "official_evidence": False,
                    "reason": "Uniform matches reference two.",
                },
                seeds,
            )

    def test_persistent_invalid_response_falls_back_to_unknown(self):
        response = safe_unknown(
            "other requires positive official/non-participant evidence"
        )
        self.assertEqual(response["label"], "unknown")
        self.assertTrue(response["abstain"])
        self.assertFalse(response["official_evidence_visible"])
