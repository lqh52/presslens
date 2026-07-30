from __future__ import annotations

import unittest

import numpy as np

from scripts.fixture_identity_recovery import (
    classify,
    fixture_id,
    medoid,
    nearest_other,
    other_veto,
    signal_distance,
    stable_holdout,
)


class FixtureIdentityRecoveryTest(unittest.TestCase):
    def test_fixture_id_removes_published_alias(self):
        self.assertEqual(
            fixture_id("ars-che-20160924-h2-0058-published"),
            "ars-che-20160924",
        )

    def test_weighted_distance_uses_only_dino_and_colour(self):
        left = (np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0]))
        right = (np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0]))
        self.assertAlmostEqual(
            signal_distance(left, right, dino_weight=0.8), 0.8
        )

    def test_medoid_is_robust_to_one_outlier(self):
        vectors = {
            "a": (np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])),
            "b": (np.asarray([0.99, 0.01]), np.asarray([1.0, 0.0])),
            "outlier": (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
        }
        self.assertIn(
            medoid(list(vectors), vectors, dino_weight=0.8), {"a", "b"}
        )

    def test_classification_reports_margin(self):
        target = (np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0]))
        prototypes = {
            "team_a": target,
            "team_b": (
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
            ),
        }
        result = classify(target, prototypes, dino_weight=0.8)
        self.assertEqual(result["label"], "team_a")
        self.assertGreater(result["margin"], 0.9)
        self.assertTrue(result["signals_agree"])

    def test_independent_signals_can_disagree(self):
        prototypes = {
            "team_a": (np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])),
            "team_b": (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
        }
        target = (np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))
        result = classify(target, prototypes, dino_weight=0.8)
        self.assertFalse(result["signals_agree"])

    def test_nearest_other_vetoes_a_closer_negative(self):
        prototypes = {
            "team_a": (np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])),
            "team_b": (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
        }
        unit = np.asarray([0.9, 0.1])
        unit = unit / np.linalg.norm(unit)
        target = (unit, unit)
        result = classify(target, prototypes, dino_weight=0.8)
        distances = nearest_other(target, [target])
        self.assertTrue(other_veto(result, distances))

    def test_holdout_is_stable(self):
        first = stable_holdout("clip:12", 0.25)
        self.assertEqual(first, stable_holdout("clip:12", 0.25))
        self.assertFalse(stable_holdout("clip:12", 0.0))
        self.assertTrue(stable_holdout("clip:12", 1.0))


if __name__ == "__main__":
    unittest.main()
