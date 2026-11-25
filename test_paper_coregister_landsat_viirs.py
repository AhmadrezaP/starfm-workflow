import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

from paper_coregister_landsat_viirs import apply_integer_shift, main, search_integer_shift


class PaperCoregistrationTests(unittest.TestCase):
    def test_recovers_known_translation(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(80, 90))
        moving = apply_integer_shift(reference, -3, 2)

        result = search_integer_shift(
            reference,
            moving,
            max_shift_pixels=5,
            minimum_samples=100,
        )

        self.assertEqual((result.x_pixels, result.y_pixels), (3, -2))
        self.assertAlmostEqual(result.correlation, 1.0, places=12)

    def test_invalid_pixels_are_ignored(self):
        reference = np.random.default_rng(7).normal(size=(40, 40))
        moving = apply_integer_shift(reference, -1, 1)
        reference[:3, :3] = np.nan
        moving[30:, 30:] = np.nan

        result = search_integer_shift(
            reference,
            moving,
            max_shift_pixels=2,
            minimum_samples=200,
        )

        self.assertEqual((result.x_pixels, result.y_pixels), (1, -1))
        self.assertAlmostEqual(result.correlation, 1.0, places=12)

    def test_cli_writes_shifted_landsat_grid_and_report(self):
        def write_raster(path: Path, values: np.ndarray, projection: str) -> None:
            dataset = gdal.GetDriverByName("GTiff").Create(
                str(path), values.shape[1], values.shape[0], 1, gdal.GDT_Float32
            )
            dataset.SetGeoTransform((500000.0, 30.0, 0.0, 3000000.0, 0.0, -30.0))
            dataset.SetProjection(projection)
            dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
            dataset.GetRasterBand(1).WriteArray(values.astype(np.float32))
            dataset.FlushCache()

        reference_physical = 0.2 + np.random.default_rng(9).normal(scale=0.01, size=(40, 40))
        viirs_physical = apply_integer_shift(reference_physical, -2, 1)
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(32643)
        projection = spatial_reference.ExportToWkt()

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            landsat_path = directory_path / "landsat_B4.tif"
            viirs_path = directory_path / "viirs_M5_nbar.tif"
            viirs_nir_path = directory_path / "viirs_M7_nbar.tif"
            output_directory = directory_path / "MOVING_WINDOW"
            output_path = output_directory / "viirs_M5_nbar_coregistered.tif"
            output_nir_path = directory_path / "viirs_M7_registered.tif"
            write_raster(landsat_path, (reference_physical + 0.2) / 0.0000275, projection)
            write_raster(viirs_path, viirs_physical, projection)
            write_raster(viirs_nir_path, viirs_physical * 1.2, projection)

            main(
                [
                    "--landsat", str(landsat_path),
                    "--viirs-nbar", str(viirs_path),
                    "--output-dir", str(output_directory),
                    "--max-shift-pixels", "4",
                    "--minimum-samples", "100",
                    "--window-size", "40",
                    "--apply-viirs", str(viirs_nir_path), str(output_nir_path),
                ]
            )

            report = json.loads(output_path.with_suffix(".coregistration.json").read_text())
            self.assertEqual((report["x_shift_pixels"], report["y_shift_pixels"]), (2, -1))
            self.assertGreater(report["correlation_improvement"], 0.5)
            output = gdal.Open(str(output_path))
            self.assertEqual((output.RasterXSize, output.RasterYSize), (40, 40))
            self.assertEqual(output.GetGeoTransform(), (500000.0, 30.0, 0.0, 3000000.0, 0.0, -30.0))
            self.assertTrue(output_nir_path.exists())
            self.assertEqual(len(report["outputs"]), 2)


if __name__ == "__main__":
    unittest.main()
