from datetime import date
import unittest

import numpy as np

from evaluate_baselines import metrics, score_methods


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.actual = np.array([0.10, 0.20, 0.30, 0.40])
        self.early = np.array([0.00, 0.10, 0.20, 0.30])
        self.late = np.array([0.20, 0.30, 0.40, 0.50])
        self.clear = np.ones(4, dtype=bool)

    def test_linear_interpolation_uses_time_weights(self):
        result = score_methods(self.actual, self.actual.copy(), self.clear,
                               [(date(2024, 1, 9), self.late, self.clear), (date(2024, 1, 1), self.early, self.clear)],
                               date(2024, 1, 5))
        scores = result["common_mask"]
        self.assertEqual(result["interpolation_weight_later"], 0.5)
        self.assertAlmostEqual(scores["linear_interpolation"]["rmse"], 0.0)
        self.assertAlmostEqual(scores["starfm"]["rmse"], 0.0)
        self.assertAlmostEqual(scores["persistence_earlier"]["bias"], -0.1)
        self.assertAlmostEqual(scores["persistence_later"]["bias"], 0.1)
        # Equal distance: the earlier pair is used for nearest persistence.
        self.assertAlmostEqual(scores["persistence_nearest"]["bias"], -0.1)

    def test_common_mask_excludes_cloudy_or_invalid_pair_pixels(self):
        cloudy = np.array([True, False, True, True])
        late = self.late.copy()
        late[3] = 1.5
        result = score_methods(self.actual, self.actual, self.clear,
                               [(date(2024, 1, 1), self.early, cloudy), (date(2024, 1, 9), late, self.clear)],
                               date(2024, 1, 3))
        self.assertEqual(result["common_pixels"], 2)
        self.assertEqual(result["starfm_on_saved_mask"]["n"], 4)
        self.assertEqual(result["common_mask"]["starfm"]["n"], 2)
        self.assertAlmostEqual(result["interpolation_weight_later"], 0.25)

    def test_unbracketed_target_and_empty_overlap_are_rejected(self):
        pairs = [(date(2024, 1, 1), self.early, self.clear), (date(2024, 1, 9), self.late, self.clear)]
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            score_methods(self.actual, self.actual, self.clear, pairs, date(2024, 1, 10))
        with self.assertRaisesRegex(RuntimeError, "No pixels"):
            score_methods(self.actual, self.actual, np.zeros(4, dtype=bool), pairs, date(2024, 1, 5))

    def test_metrics(self):
        result = metrics(np.array([0.1, 0.3]), np.array([0.2, 0.2]))
        self.assertAlmostEqual(result["rmse"], 0.1)
        self.assertAlmostEqual(result["bias"], 0.0)
        self.assertAlmostEqual(result["r2"], 0.0)


if __name__ == "__main__":
    unittest.main()
