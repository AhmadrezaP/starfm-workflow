import unittest

import numpy as np

from build_coregistration_mask import landsat_clear_mask, viirs_m5_clear_mask


class CoregistrationMaskTests(unittest.TestCase):
    def test_viirs_rejects_cloud_bad_m5_and_adjacent_cloud(self):
        qf1 = np.array([[0b00000000, 0b00001000, 0b00001100, 0b00000000]], dtype=np.uint8)
        qf5 = np.array([[0, 0, 0, 0b01000000]], dtype=np.uint8)
        qf7 = np.array([[0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(
            viirs_m5_clear_mask(qf1, qf5, qf7), [[True, False, False, False]]
        )
        np.testing.assert_array_equal(
            viirs_m5_clear_mask(np.array([[0]], dtype=np.uint8), np.array([[0]], dtype=np.uint8), np.array([[2]], dtype=np.uint8)),
            [[False]],
        )

    def test_landsat_rejects_documented_cloud_related_bits(self):
        qa = np.array([[0, 1, 2, 4, 8, 16, 32]], dtype=np.uint16)
        np.testing.assert_array_equal(landsat_clear_mask(qa), [[True, False, False, False, False, False, False]])


if __name__ == "__main__":
    unittest.main()
