import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date

from run_search_distance_sweep import _distances, _pipeline_command, _write_aggregate_results, build_parser
from run_uncertainty_sweep import _write_results


class SearchDistanceSweepTests(unittest.TestCase):
    def test_resumed_sweeps_accept_old_and_new_metric_columns(self):
        rows = [{"test_id": "old", "rmse_surface_reflectance": 0.1},
                {"test_id": "new", "rmse_surface_reflectance": 0.09,
                 "mae_surface_reflectance": 0.08, "r2_surface_reflectance": -0.2}]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for write, filename in [(_write_aggregate_results, "search_distance_sweep_results.csv"),
                                    (_write_results, "uncertainty_sweep_results.csv")]:
                write(root, rows)
                with (root / filename).open() as stream:
                    saved = list(csv.DictReader(stream))
                self.assertEqual(saved[0]["mae_surface_reflectance"], "")
                self.assertEqual(float(saved[1]["r2_surface_reflectance"]), -0.2)

    def test_default_sweep_is_five_selected_distances(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42",
                "--band", "B3", "--sweep-id", "default_sweep",
            ]
        )
        self.assertEqual(
            _distances(args.start_distance, args.stop_distance, args.step_distance),
            [550, 650, 750, 850, 950],
        )

    def test_command_uses_named_test_and_preserves_shared_nbar(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B3",
                "--sweep-id", "search_sweep", "--pair-dates", "2024-03-11",
            ]
        )
        command = _pipeline_command(args, "B3", 500, "search_sweep_b3_d0500")
        self.assertIn("--reuse-existing-nbar", command)
        self.assertIn("--discard-intermediate-outputs-after-validation", command)
        self.assertEqual(command[command.index("--test-id") + 1], "search_sweep_b3_d0500")
        self.assertEqual(command[command.index("--pair-dates") + 1], date(2024, 3, 11).isoformat())

    def test_all_bands_selection_is_limited_to_the_selected_sensor(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42",
                "--sweep-id", "all_bands", "--coarse-sensor", "MODIS", "--all-bands",
            ]
        )
        from run_search_distance_sweep import _selected_bands
        self.assertEqual(_selected_bands(args), ("B2", "B3", "B4", "B5", "B7"))

    def test_explicit_cleanup_option_is_available(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B3",
                "--sweep-id", "cleanup_sweep", "--remove-unnecessary-files",
            ]
        )
        command = _pipeline_command(args, "B3", 500, "cleanup_sweep_b3_d0500")
        self.assertIn("--discard-intermediate-outputs-after-validation", command)
