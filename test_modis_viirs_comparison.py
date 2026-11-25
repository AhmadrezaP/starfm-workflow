import unittest
from datetime import date

from run_modis_viirs_comparison import (
    COMPARABLE_BANDS,
    _paired_rows,
    _pipeline_command,
    _selected_bands,
    build_parser,
)


class ModisViirsComparisonTests(unittest.TestCase):
    def test_all_bands_uses_only_bands_shared_by_both_sensors(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42",
                "--all-bands", "--comparison-id", "sensor_compare",
            ]
        )
        self.assertEqual(COMPARABLE_BANDS, ("B2", "B3", "B4", "B5", "B7"))
        self.assertEqual(_selected_bands(args), COMPARABLE_BANDS)

    def test_b1_is_rejected_for_a_sensor_comparison(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42",
                "--band", "B1", "--comparison-id", "sensor_compare",
            ]
        )
        with self.assertRaises(ValueError):
            _selected_bands(args)

    def test_command_has_identical_pair_dates_and_heldout_validation_for_both_sensors(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42",
                "--band", "B3", "--comparison-id", "sensor_compare",
                "--pair-dates", "2024-03-11", "2024-03-27", "--remove-unnecessary-files",
            ]
        )
        modis = _pipeline_command(args, "MODIS", "B3", "sensor_compare_modis_b3")
        viirs = _pipeline_command(args, "VIIRS", "B3", "sensor_compare_viirs_b3")
        self.assertIn("--validate-held-out-landsat", modis)
        self.assertEqual(modis[modis.index("--pair-dates") + 1:modis.index("--pair-dates") + 3], [
            date(2024, 3, 11).isoformat(), date(2024, 3, 27).isoformat(),
        ])
        self.assertEqual(
            modis[modis.index("--pair-dates") + 1:modis.index("--pair-dates") + 3],
            viirs[viirs.index("--pair-dates") + 1:viirs.index("--pair-dates") + 3],
        )
        self.assertIn("--discard-intermediate-outputs-after-validation", modis)

    def test_paired_table_calculates_viirs_minus_modis(self):
        rows = [
            {
                "coarse_sensor": "MODIS", "predicted_landsat_band": "B3", "test_id": "m",
                "prediction_date": "2024-03-19", "pair_dates": "2024-03-11;2024-03-27",
                "coregistration_method": "moving-window", "rmse_surface_reflectance": 0.02,
                "mean_difference_predicted_minus_actual": 0.01, "relative_error_percent": 10,
                "valid_comparison_percent": 80, "coregistration_quality_percent": 50,
            },
            {
                "coarse_sensor": "VIIRS", "predicted_landsat_band": "B3", "test_id": "v",
                "prediction_date": "2024-03-19", "pair_dates": "2024-03-11;2024-03-27",
                "coregistration_method": "moving-window", "rmse_surface_reflectance": 0.015,
                "mean_difference_predicted_minus_actual": 0.005, "relative_error_percent": 8,
                "valid_comparison_percent": 82, "coregistration_quality_percent": 55,
            },
        ]
        paired = _paired_rows(rows, ("B3",))
        self.assertAlmostEqual(paired[0]["viirs_minus_modis_rmse_surface_reflectance"], -0.005)
        self.assertEqual(paired[0]["viirs_minus_modis_valid_comparison_percent"], 2.0)


if __name__ == "__main__":
    unittest.main()
