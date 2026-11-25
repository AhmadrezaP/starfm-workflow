import unittest

from run_modis_nbar_hdf import MODIS_BANDS, build_parser


class ModisNbarParserTests(unittest.TestCase):
    def test_supports_modis_reflective_bands(self):
        self.assertEqual(MODIS_BANDS, (1, 2, 3, 4, 5, 6, 7))
        args = build_parser().parse_args(
            [
                "--mod09ga", "daily.hdf",
                "--mcd43a1", "parameters.hdf",
                "--mcd43a2", "quality.hdf",
                "--land-cover", "land_cover.hdf",
                "--band", "1",
                "--output", "nbar.tif",
            ]
        )
        self.assertEqual(args.band, 1)
        self.assertEqual(args.valid_quality_values, {0, 1})


if __name__ == "__main__":
    unittest.main()
