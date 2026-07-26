import unittest

import numpy as np

from text_embeddings import bm25_scores, hybrid_scores


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "Tactical situation: High press. Dense pressure around the build up.",
            "Tactical situation: Central screen. Central passing lanes are screened.",
            "Tactical situation: No local pressure. Limited coordinated pressure.",
        ]

    def test_bm25_distinguishes_central_press_from_high_press(self):
        central = bm25_scores("central press", self.documents)
        high = bm25_scores("high press", self.documents)

        self.assertEqual(int(np.argmax(central)), 1)
        self.assertEqual(int(np.argmax(high)), 0)

    def test_hybrid_score_uses_lexical_signal_to_break_semantic_tie(self):
        cosine = np.asarray([0.7, 0.7, 0.7], dtype=np.float32)
        lexical = bm25_scores("central press", self.documents)

        scores = hybrid_scores(cosine, lexical)

        self.assertEqual(int(np.argmax(scores)), 1)
        self.assertTrue(np.all((0 <= scores) & (scores <= 1)))


if __name__ == "__main__":
    unittest.main()
