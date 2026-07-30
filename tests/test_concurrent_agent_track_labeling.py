import unittest

from scripts.run_concurrent_agent_track_labeling import split_round_robin


class ConcurrentAgentTrackLabelingTest(unittest.TestCase):
    def test_round_robin_preserves_every_key(self):
        keys = [f"key-{index}" for index in range(10)]
        shards = split_round_robin(keys, 3)
        self.assertEqual([len(shard) for shard in shards], [4, 3, 3])
        self.assertEqual(sorted(key for shard in shards for key in shard), keys)
