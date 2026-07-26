from __future__ import annotations

import unittest

from scripts.derive_weak_tactical_labels import (
    LABELS,
    direction_is_trusted,
    gate_direction_dependent_label,
)


def direction_metadata(**overrides):
    row = {
        "direction_confident": True,
        "direction_status": "calibrated",
        "direction_source": "pitch_spatial_calibration",
        "attacking_direction_raw": 1,
        "attacking_direction_label": "left_to_right",
    }
    row.update(overrides)
    return row


class WeakTacticalDirectionGateTest(unittest.TestCase):
    def test_direction_dependent_weak_labels_abstain(self):
        for label_name in (
            "high_press",
            "central_screen",
            "trap_left",
            "trap_right",
        ):
            with self.subTest(label=label_name):
                label = LABELS.index(label_name)
                self.assertEqual(
                    gate_direction_dependent_label(
                        label,
                        0.91,
                        "directional_rule",
                        direction_trusted=False,
                    ),
                    (-1, 0.0, "abstain_direction"),
                )

    def test_unstructured_weak_label_is_direction_invariant(self):
        label = LABELS.index("unstructured")
        self.assertEqual(
            gate_direction_dependent_label(
                label,
                0.72,
                "no_local_pressure",
                direction_trusted=False,
            ),
            (label, 0.72, "no_local_pressure"),
        )

    def test_direction_requires_explicit_consistent_provenance(self):
        self.assertTrue(direction_is_trusted(direction_metadata()))
        self.assertFalse(direction_is_trusted({}))
        self.assertFalse(
            direction_is_trusted(
                direction_metadata(direction_confident=False)
            )
        )
        self.assertFalse(
            direction_is_trusted(
                direction_metadata(direction_status="abstained_ambiguous")
            )
        )
        self.assertFalse(
            direction_is_trusted(
                direction_metadata(
                    attacking_direction_raw=-1,
                    attacking_direction_label="left_to_right",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
