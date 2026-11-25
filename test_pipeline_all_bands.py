import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from run_nearest_pair_pipeline import main


class AllBandsTests(unittest.TestCase):
    def test_method_specific_matching_band_and_override(self):
        for method, override, expected in [
            ("arosics", [], "B5"),
            ("moving-window", [], None),
            ("moving-window", ["--coregistration-band", "B5"], None),
            ("arosics", ["--coregistration-band", "B4"], "B4"),
            ("none", [], None),
            ("none", ["--coregistration-band", "B5"], None),
        ]:
            for selection in (["--band", "B1"], ["--all-bands"]):
                with patch("run_nearest_pair_pipeline._run_single_band") as run:
                    main(["--prediction-date", "2023-09-18", "--path", "29", "--row", "32",
                          "--coregistration-method", method, *selection, *override])
                    self.assertTrue(run.called)
                    for call in run.call_args_list:
                        self.assertEqual(call.args[0].coregistration_band, expected or call.args[0].band)

    def test_sensor_bands_and_forwarded_options(self):
        for sensor, expected in [
            ("VIIRS", ["B1", "B2", "B3", "B4", "B5", "B7"]),
            ("MODIS", ["B2", "B3", "B4", "B5", "B7"]),
        ]:
            with TemporaryDirectory() as root, patch("run_nearest_pair_pipeline._run_single_band") as run:
                main(["--prediction-date", "2023-09-18", "--path", "29", "--row", "32",
                      "--all-bands", "--coarse-sensor", sensor, "--test-id", "example",
                      "--output-dir", root, "--pair-dates", "2023-09-02", "2023-09-26",
                      "--validate-held-out-landsat"])
                self.assertEqual([call.args[0].band for call in run.call_args_list], expected)
                for call in run.call_args_list:
                    args = call.args[0]
                    self.assertEqual(args.test_id, f"example_{args.band.lower()}")
                    self.assertEqual(args.coregistration_band, args.band)
                    self.assertTrue(args.validate_held_out_landsat)
                    self.assertEqual(len(args.pair_dates), 2)

    def test_checks_all_destinations_before_processing(self):
        with TemporaryDirectory() as root, patch("run_nearest_pair_pipeline._run_single_band") as run:
            (Path(root) / "OUTPUTS" / "TESTS" / "example_b7").mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                main(["--prediction-date", "2023-09-18", "--path", "29", "--row", "32",
                      "--all-bands", "--test-id", "example", "--output-dir", root])
            run.assert_not_called()

    def test_stops_on_failure(self):
        with patch("run_nearest_pair_pipeline._run_single_band", side_effect=RuntimeError("failed")) as run:
            with self.assertRaises(RuntimeError):
                main(["--prediction-date", "2023-09-18", "--path", "29", "--row", "32", "--all-bands"])
            self.assertEqual(run.call_count, 1)
