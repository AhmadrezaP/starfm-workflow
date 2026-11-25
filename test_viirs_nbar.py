import unittest

import numpy as np

from viirs_nbar import normalize_to_nbar
from run_viirs_nbar_hdf import fill_brdf_parameters_by_land_cover


class NormalizeToNbarTests(unittest.TestCase):
    def test_isotropic_surface_is_unchanged(self):
        reflectance = np.array([[0.12, 0.31]])
        result = normalize_to_nbar(
            reflectance,
            np.array([[25.0, 40.0]]),
            np.array([[35.0, 10.0]]),
            np.array([[100.0, 120.0]]),
            np.array([[220.0, 20.0]]),
            np.ones((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        )
        np.testing.assert_allclose(result, reflectance)

    def test_nadir_observation_is_unchanged(self):
        reflectance = np.array([[0.2]])
        result = normalize_to_nbar(
            reflectance,
            np.array([[35.0]]),
            np.array([[0.0]]),
            np.array([[130.0]]),
            np.array([[130.0]]),
            np.array([[0.1]]),
            np.array([[0.03]]),
            np.array([[0.02]]),
        )
        np.testing.assert_allclose(result, reflectance)

    def test_invalid_mask_produces_nan(self):
        result = normalize_to_nbar(
            np.array([[0.2]]),
            np.array([[35.0]]),
            np.array([[10.0]]),
            np.array([[130.0]]),
            np.array([[160.0]]),
            np.array([[0.1]]),
            np.array([[0.03]]),
            np.array([[0.02]]),
            valid_mask=np.array([[False]]),
        )
        self.assertTrue(np.isnan(result[0, 0]))


class LandCoverGapFillTests(unittest.TestCase):
    def test_uses_class_lut_then_global_fallback(self):
        iso, vol, geo, usable, source = fill_brdf_parameters_by_land_cover(
            f_iso=np.array([[0.10, 0.30, np.nan, np.nan]]),
            f_vol=np.array([[0.01, 0.03, np.nan, np.nan]]),
            f_geo=np.array([[0.02, 0.04, np.nan, np.nan]]),
            quality=np.array([[0, 0, 1, 2]], dtype=np.uint8),
            land_cover=np.array([[1, 1, 1, 2]], dtype=np.uint8),
            trusted_quality_values={0},
            fill_quality_values={1, 2},
            minimum_class_samples=2,
        )
        np.testing.assert_allclose(iso, [[0.10, 0.30, 0.20, 0.20]])
        np.testing.assert_allclose(vol, [[0.01, 0.03, 0.02, 0.02]])
        np.testing.assert_allclose(geo, [[0.02, 0.04, 0.03, 0.03]])
        np.testing.assert_array_equal(usable, [[True, True, True, True]])
        np.testing.assert_array_equal(source, [[0, 0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
