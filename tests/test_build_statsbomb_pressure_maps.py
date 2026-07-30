import unittest

from scripts.build_statsbomb_pressure_maps import point


class StatsBombPressureMapsTest(unittest.TestCase):
    def test_point_maps_120x80_to_105x68(self):
        self.assertEqual(point([0, 0]), [0.0, 0.0])
        self.assertEqual(point([60, 40]), [52.5, 34.0])
        self.assertEqual(point([120, 80]), [105.0, 68.0])
