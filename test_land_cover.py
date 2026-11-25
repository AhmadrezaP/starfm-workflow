from datetime import date
import unittest
from unittest.mock import patch

from run_nearest_pair_pipeline import LAND_COVER_MAX_FALLBACK_YEARS, _land_cover_entries


def missing(year):
    return RuntimeError(f"CMR returned no MCD12Q1 file labelled A{year}001. Nearby results: []")


class LandCoverFallbackTests(unittest.TestCase):
    def lookup(self, available):
        def entries(short_name, version, day, bbox=None):
            if day.year in available:
                return [{"title": f"MCD12Q1.A{day.year}001"}]
            raise missing(day.year)
        return patch("run_nearest_pair_pipeline._cmr_entries_for_filename_date", side_effect=entries)

    def test_same_year_used_when_published(self):
        with self.lookup({2024, 2023}):
            self.assertEqual(_land_cover_entries(date(2024, 6, 1))[1], 2024)

    def test_unpublished_year_falls_back_to_latest(self):
        with self.lookup({2024, 2023}):
            entries, year = _land_cover_entries(date(2026, 3, 5))
        self.assertEqual(year, 2024)
        self.assertEqual(entries, [{"title": "MCD12Q1.A2024001"}])

    def test_fallback_is_bounded(self):
        with self.lookup({2026 - LAND_COVER_MAX_FALLBACK_YEARS - 1}):
            with self.assertRaisesRegex(RuntimeError, "No MCD12Q1 land cover"):
                _land_cover_entries(date(2026, 1, 1))

    def test_other_errors_are_not_masked(self):
        with patch("run_nearest_pair_pipeline._cmr_entries_for_filename_date", side_effect=RuntimeError("CMR unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CMR unavailable"):
                _land_cover_entries(date(2025, 1, 1))


if __name__ == "__main__":
    unittest.main()
