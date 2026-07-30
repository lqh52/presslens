from __future__ import annotations

import unittest

import numpy as np

from scripts.classify_track_identities import (
    appearance_features,
    fixture_id,
    normalized_rows,
    torso_crop,
)


class TrackIdentityTest(unittest.TestCase):
    def test_torso_crop_uses_upper_central_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[12:55, 16:84] = (20, 40, 180)
        crop = torso_crop(image, [0, 0, 100, 100])
        self.assertIsNotNone(crop)
        self.assertLess(crop.shape[0], 60)
        self.assertLess(crop.shape[1], 80)

    def test_appearance_feature_shape_and_geometry(self):
        image = np.full((200, 300, 3), (40, 100, 180), dtype=np.uint8)
        features = appearance_features(image, [30, 20, 90, 180])
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (82,))
        self.assertAlmostEqual(float(features[-4]), 0.2)
        self.assertAlmostEqual(float(features[-3]), 0.8)

    def test_fixture_id_joins_clips_from_same_match(self):
        self.assertEqual(fixture_id("bur-ars-20150411-h1-0128"), "bur-ars-20150411")
        self.assertEqual(
            fixture_id("lei-ars-20150926-h1-0093-published"),
            "lei-ars-20150926",
        )

    def test_normalized_rows_have_unit_length(self):
        matrix = normalized_rows(np.asarray([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
