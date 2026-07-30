import unittest

import numpy as np

from scripts.build_reviewed_web_demo import align_canonical_orientation


class CanonicalVideoOrientationTests(unittest.TestCase):
    def test_opposite_frame_direction_is_rotated_into_locked_view(self):
        features = np.zeros((2, 13), dtype=np.float32)
        features[0, :4] = [0.8, 0.25, 0.1, -0.2]
        features[1, :2] = [0.7, 0.4]

        aligned = align_canonical_orientation(features, -1, 1)

        np.testing.assert_allclose(
            aligned[0, :4],
            [0.2, 0.75, -0.1, 0.2],
        )
        np.testing.assert_allclose(aligned[1, :2], [0.3, 0.6])
        np.testing.assert_allclose(
            features[0, :4],
            [0.8, 0.25, 0.1, -0.2],
        )

    def test_matching_direction_is_not_copied_or_changed(self):
        features = np.zeros((1, 13), dtype=np.float32)
        self.assertIs(
            align_canonical_orientation(features, 1, 1),
            features,
        )


if __name__ == "__main__":
    unittest.main()
