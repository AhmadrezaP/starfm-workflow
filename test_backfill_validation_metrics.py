import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from osgeo import gdal, osr

from backfill_validation_metrics import calculate, process


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.test = self.root / 'OUTPUTS/TESTS/example'
        self.validation = self.test / 'VALIDATION'
        self.validation.mkdir(parents=True)
        self.actual = self.validation / 'actual.tif'
        self.pred = self.test / 'predicted.tif'
        self.mask = self.validation / 'mask.tif'
        self.raster(self.actual, [[10000, 12000], [14000, 16000]])
        self.raster(self.pred, [[11000, 11000], [15000, 0]])
        self.raster(self.mask, [[1, 1], [1, 0]])

    def tearDown(self):
        self.temp.cleanup()

    def raster(self, path, values):
        d = gdal.GetDriverByName('GTiff').Create(str(path), 2, 2, 1, gdal.GDT_UInt16)
        d.SetGeoTransform((0, 30, 0, 60, 0, -30))
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(32614)
        d.SetProjection(sr.ExportToWkt())
        d.GetRasterBand(1).WriteArray(np.array(values))
        d = None

    def test_block_metrics(self):
        m = calculate(self.pred, self.actual, self.mask, block_rows=1)
        self.assertAlmostEqual(m['mae_surface_reflectance'], .0275)
        self.assertAlmostEqual(m['r2_surface_reflectance'], .625)
        self.assertEqual(m['valid_comparison_pixels'], 3)

    def test_constant_reference(self):
        self.raster(self.actual, [[10000, 10000], [10000, 10000]])
        self.assertIsNone(calculate(self.pred, self.actual, self.mask, 1)['r2_surface_reflectance'])

    def test_negative_r2(self):
        self.raster(self.pred, [[20000, 20000], [20000, 20000]])
        self.assertLess(calculate(self.pred, self.actual, self.mask)['r2_surface_reflectance'], 0)

    def prepare_reports(self):
        m = calculate(self.pred, self.actual, self.mask)
        m.pop('mae_surface_reflectance')
        m.pop('r2_surface_reflectance')
        summary = dict(m, final_output_geotiff='/old/COREGISTRATION_FILES/OUTPUTS/TESTS/example/predicted.tif')
        report = dict(m, actual_landsat_on_prediction_grid=str(self.actual), comparison_mask=str(self.mask))
        (self.test / 'test_summary.json').write_text(json.dumps(summary))
        (self.validation / 'heldout_validation_metrics.json').write_text(json.dumps(report))
        (self.test / 'run_manifest.json').write_text(json.dumps({'outputs': {'heldout_landsat_validation': report}}))

    def test_preview_apply_and_backup(self):
        self.prepare_reports()
        path = self.test / 'test_summary.json'
        before = path.read_bytes()
        process(self.test, self.root)
        self.assertEqual(path.read_bytes(), before)
        process(self.test, self.root, True)
        self.assertAlmostEqual(json.loads(path.read_text())['mae_surface_reflectance'], .0275)
        backup = path.with_name(path.name + '.before_metrics_backfill')
        self.assertEqual(backup.read_bytes(), before)
        process(self.test, self.root, True)
        self.assertEqual(backup.read_bytes(), before)
        self.assertTrue(path.with_suffix('.csv').exists())

    def test_mismatch_refuses_update(self):
        self.prepare_reports()
        path = self.test / 'test_summary.json'
        data = json.loads(path.read_text())
        data['valid_comparison_pixels'] = 99
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            process(self.test, self.root, True)
        self.assertFalse(path.with_name(path.name + '.before_metrics_backfill').exists())

    def test_grid_mismatch(self):
        d = gdal.Open(str(self.mask), gdal.GA_Update)
        d.SetGeoTransform((30, 30, 0, 60, 0, -30))
        d = None
        with self.assertRaises(ValueError):
            calculate(self.pred, self.actual, self.mask)


if __name__ == '__main__':
    unittest.main()
