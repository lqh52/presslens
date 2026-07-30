import unittest

import torch

from scripts.infer_skillcorner_pressing_stg import temporal_windows
from scripts.train_skillcorner_pressing_stg import SpatiotemporalGraphNet


class SkillCornerPressingSTGTest(unittest.TestCase):
    def test_model_accepts_video_graph_schema(self):
        model = SpatiotemporalGraphNet(13, 3, width=16)
        features = torch.zeros(2, 5, 23, 13)
        masks = torch.ones(2, 5, 23, dtype=torch.bool)
        self.assertEqual(model(features, masks).shape, (2, 3))

    def test_temporal_windows_span_two_seconds_when_available(self):
        rows = [{"frame": frame} for frame in range(100)]
        windows = temporal_windows(rows, list(range(100)))
        self.assertGreater(len(windows), 1)
        source_frames = [rows[index]["frame"] for index in windows[0]]
        self.assertEqual(source_frames, [0, 13, 25, 37, 50])


if __name__ == "__main__":
    unittest.main()
