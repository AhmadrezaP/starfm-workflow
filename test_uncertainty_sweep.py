import unittest
from datetime import date

from run_uncertainty_sweep import (
    DEFAULT_UNCERTAINTIES,
    _pipeline_command,
    _selected_bands,
    _uncertainties,
    build_parser,
)


class UncertaintySweepTests(unittest.TestCase):
    def test_default_values(self):
        args = build_parser().parse_args([
            "--prediction-date", "2023-09-18", "--path", "29", "--row", "32",
            "--band", "B1", "--sweep-id", "uncertainty_test",
        ])
        self.assertEqual(_uncertainties(args.uncertainties), [20, 35, 50, 65, 80])
        self.assertEqual(tuple(args.uncertainties), DEFAULT_UNCERTAINTIES)

    def test_command_fixes_distance_and_varies_uncertainty(self):
        args = build_parser().parse_args([
            "--prediction-date", "2023-09-18", "--path", "29", "--row", "32",
            "--band", "B1", "--sweep-id", "uncertainty_test",
            "--pair-dates", "2023-09-02", "2023-09-26",
        ])
        command = _pipeline_command(args, "B1", 35, "uncertainty_test_b1_u0035")
        self.assertEqual(command[command.index("--starfm-max-search-distance") + 1], "750")
        self.assertEqual(command[command.index("--starfm-uncertainty") + 1], "35")
        self.assertEqual(command[command.index("--coregistration-band") + 1], "B1")
        self.assertEqual(
            command[command.index("--pair-dates") + 1:command.index("--pair-dates") + 3],
            [date(2023, 9, 2).isoformat(), date(2023, 9, 26).isoformat()],
        )

    def test_all_bands_respects_sensor_support(self):
        common = [
            "--prediction-date", "2023-09-18", "--path", "29", "--row", "32",
            "--all-bands", "--sweep-id", "uncertainty_test",
        ]
        viirs = build_parser().parse_args(common + ["--coarse-sensor", "VIIRS"])
        modis = build_parser().parse_args(common + ["--coarse-sensor", "MODIS"])
        self.assertEqual(_selected_bands(viirs), ("B1", "B2", "B3", "B4", "B5", "B7"))
        self.assertEqual(_selected_bands(modis), ("B2", "B3", "B4", "B5", "B7"))
