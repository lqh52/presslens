from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_candidates import (
    create_candidates,
    overlaps_stoppage,
    resolve_game_slug,
)
from scripts.rank_video_candidates import (
    DEFAULT_QUERY_SPECS,
    QuerySpec,
    make_ranked_item,
    parse_query_specs,
    select_balanced,
)


class CandidateIngestionTest(unittest.TestCase):
    def test_existing_match_keeps_historic_auto_slug(self):
        game = "2015-04-11 - 19-30 Burnley 0 - 1 Arsenal"
        self.assertEqual(resolve_game_slug(game), "burnley-arsenal")

    def test_auto_slug_handles_club_names_containing_digits(self):
        game = "2017-01-29 - 19-30 1. FSV Mainz 05 1 - 1 Dortmund"
        self.assertEqual(resolve_game_slug(game), "1-fsv-mainz-05-dortmund")

    def test_explicit_slug_is_normalized(self):
        self.assertEqual(
            resolve_game_slug("ignored", "Arsenal Chelsea 2016"),
            "arsenal-chelsea-2016",
        )

    def test_explicit_slug_is_used_in_candidate_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            game = Path(raw) / "2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea"
            game.mkdir()
            (game / "Labels-v2.json").write_text(
                json.dumps({"annotations": []})
            )
            (game / "1_224p.mkv").touch()
            (game / "2_224p.mkv").touch()
            with patch(
                "scripts.build_candidates.subprocess.check_output",
                return_value="30.0\n",
            ):
                candidates = create_candidates(
                    game,
                    window_seconds=12,
                    stride_seconds=15,
                    margin_seconds=3,
                    game_slug="arsenal-chelsea-2016",
                )
        self.assertEqual(
            [candidate.id for candidate in candidates],
            [
                "arsenal-chelsea-2016-h1-0000",
                "arsenal-chelsea-2016-h2-0000",
            ],
        )

    def test_throw_ins_and_restarts_are_excluded(self):
        for label in (
            "Throw-in",
            "Direct free-kick",
            "Indirect free-kick",
            "Kick-off",
            "Penalty",
        ):
            self.assertTrue(
                overlaps_stoppage(10, 22, [(16, label)], margin=3),
                label,
            )

    def test_focused_default_queries_have_stable_ids(self):
        self.assertEqual(
            [query.id for query in parse_query_specs(None)],
            [
                "high_press",
                "left_touchline_trap",
                "right_touchline_trap",
            ],
        )
        self.assertEqual(parse_query_specs(None), DEFAULT_QUERY_SPECS)

    def test_named_and_legacy_queries_are_supported(self):
        specs = parse_query_specs(
            ["wide_press=press toward the wing", "a bare legacy prompt"]
        )
        self.assertEqual(
            specs,
            [
                QuerySpec("wide_press", "press toward the wing"),
                QuerySpec("query_2", "a bare legacy prompt"),
            ],
        )

    def test_ranked_item_retains_every_query_score(self):
        specs = [QuerySpec("left", "left trap"), QuerySpec("right", "right trap")]
        item = make_ranked_item({"id": "clip-1"}, [0.1, 0.4], specs)
        self.assertEqual(item["matched_query_id"], "right")
        self.assertEqual(item["matched_query"], "right trap")
        self.assertEqual(item["query_scores"], {"left": 0.1, "right": 0.4})

    def test_balanced_selection_is_unique_and_near_equal(self):
        specs = [
            QuerySpec("high", "high press"),
            QuerySpec("left", "left trap"),
            QuerySpec("right", "right trap"),
        ]
        ranked = []
        for index in range(9):
            ranked.append(
                {
                    "id": f"clip-{index}",
                    "query_score": 1.0 - index / 100,
                    "query_scores": {
                        "high": 1.0 - index / 100,
                        "left": 0.9 - abs(index - 4) / 100,
                        "right": 0.8 + index / 100,
                    },
                }
            )
        selected = select_balanced(ranked, specs, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item["id"] for item in selected}), 8)
        counts = {
            query.id: sum(
                item["selected_for_query_id"] == query.id for item in selected
            )
            for query in specs
        }
        self.assertEqual(counts, {"high": 3, "left": 3, "right": 2})


if __name__ == "__main__":
    unittest.main()
