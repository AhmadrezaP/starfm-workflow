import unittest
import csv
import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from osgeo import gdal

from run_nearest_pair_pipeline import (
    BAND_INFO,
    STARFM_FILL_VALUE,
    STARFM_HEADER_FILENAME_BUF,
    _coregistration_quality_metadata,
    _evaluate_heldout_landsat,
    _reflectance_error_metrics,
    _normalize_explicit_pair_dates,
    _should_keep_intermediate_outputs,
    _shift_boolean_mask,
    _starfm_cwd_relative,
    _starfm_int16,
    _test_id,
    _validate_starfm_output_filename,
    _write_starfm_parameter_file,
    _write_prediction_geotiff,
    build_parser,
    select_bracketing_landsat_items,
    select_nearest_landsat_item,
)


class NearestPairSelectionTests(unittest.TestCase):
    def test_reflectance_metrics_and_undefined_r2(self):
        actual = np.array([0.1, 0.2, 0.3])
        perfect = _reflectance_error_metrics(actual, actual)
        self.assertEqual(perfect["r2_surface_reflectance"], 1.0)
        self.assertEqual(perfect["mae_surface_reflectance"], 0.0)
        shifted = _reflectance_error_metrics(actual, actual + 0.2)
        self.assertAlmostEqual(shifted["r2_surface_reflectance"], -5.0)
        self.assertAlmostEqual(shifted["mae_surface_reflectance"], 0.2)
        mixed = _reflectance_error_metrics(actual, np.array([0.2, 0.1, 0.3]))
        self.assertAlmostEqual(mixed["mae_surface_reflectance"], 0.2 / 3)
        for values in ([0.2], [0.2, 0.2], [0.2] * 3):
            self.assertIsNone(_reflectance_error_metrics(values, values)["r2_surface_reflectance"])
        with self.assertRaises(ValueError):
            _reflectance_error_metrics([], [])
        with self.assertRaises(ValueError):
            _reflectance_error_metrics([np.nan], [0.2])

    def test_selects_nearest_date_then_lower_cloud_on_same_date_tie(self):
        items = [
            {"id": "LC09_L2SP_143042_20240410_20240411_02_T1", "properties": {"datetime": "2024-04-10T05:00:00Z", "eo:cloud_cover": 2}},
            {"id": "LC09_L2SP_143042_20240421_20240422_02_T1", "properties": {"datetime": "2024-04-21T05:00:00Z", "eo:cloud_cover": 20}},
            {"id": "LC08_L2SP_143042_20240421_20240422_02_T1", "properties": {"datetime": "2024-04-21T05:00:00Z", "eo:cloud_cover": 5}},
        ]
        selected = select_nearest_landsat_item(items, date(2024, 4, 20), 143, 42, 30)
        self.assertEqual(selected["id"], "LC08_L2SP_143042_20240421_20240422_02_T1")

    def test_one_pair_selection_excludes_a_landsat_scene_on_the_prediction_date(self):
        items = [
            {"id": "LC08_L2SP_029032_20220907_20220914_02_T1", "properties": {"datetime": "2022-09-07T17:13:09Z", "eo:cloud_cover": 1}},
            {"id": "LC09_L2SP_029032_20220908_20220915_02_T1", "properties": {"datetime": "2022-09-08T17:13:21Z", "eo:cloud_cover": 0}},
            {"id": "LC09_L2SP_029032_20220915_20230329_02_T1", "properties": {"datetime": "2022-09-15T17:13:21Z", "eo:cloud_cover": 2}},
        ]
        selected = select_nearest_landsat_item(
            items, date(2022, 9, 8), 29, 32, 10, exclude_target_date=True
        )
        self.assertEqual(selected["id"], "LC08_L2SP_029032_20220907_20220914_02_T1")

    def test_band_mapping_includes_coregistration_red_band(self):
        self.assertEqual(BAND_INFO["B4"].viirs_band, "M5")
        self.assertEqual(BAND_INFO["B4"].modis_band, 1)
        self.assertEqual(BAND_INFO["B3"].modis_band, 4)
        self.assertIsNone(BAND_INFO["B1"].modis_band)
        self.assertEqual(BAND_INFO["B7"].parameter_products, ("VNP43D25", "VNP43D26", "VNP43D27"))

    def test_accepts_l2sr_scene_identifier(self):
        item = {
            "id": "LC08_L2SR_029032_20220907_20220914_02_T1_SR",
            "properties": {"datetime": "2022-09-07T17:13:09Z", "eo:cloud_cover": 0.02},
        }
        self.assertEqual(
            select_nearest_landsat_item([item], date(2022, 9, 7), 29, 32, 10)["id"],
            item["id"],
        )

    def test_starfm_inputs_are_signed_int16_reflectance_times_10000(self):
        encoded, valid = _starfm_int16(
            np.array([[-0.01, 0.1234, 1.0, 1.01, np.nan]]),
            np.ones((1, 5), dtype=bool),
        )
        self.assertEqual(encoded.dtype, np.dtype("<i2"))
        np.testing.assert_array_equal(
            encoded,
            np.array([[STARFM_FILL_VALUE, 1234, 10000, STARFM_FILL_VALUE, STARFM_FILL_VALUE]], dtype="<i2"),
        )
        np.testing.assert_array_equal(valid, np.array([[False, True, True, False, False]]))

    def test_mask_shift_marks_exposed_edges_invalid(self):
        source = np.array([[True, False, True], [False, True, False]])
        np.testing.assert_array_equal(
            _shift_boolean_mask(source, 1, -1),
            np.array([[False, False, True], [False, False, False]]),
        )

    def test_starfm_mask_parameters_are_absent_when_disabled(self):
        with TemporaryDirectory() as temporary_directory:
            parameters = Path(temporary_directory) / "starfm_parameters.txt"
            _write_starfm_parameter_file(
                parameters,
                pair_landsat=[Path("pair_landsat_before.bin"), Path("pair_landsat_after.bin")],
                pair_landsat_mask=[Path("pair_landsat_before.mask"), Path("pair_landsat_after.mask")],
                pair_viirs=[Path("pair_viirs_before.bin"), Path("pair_viirs_after.bin")],
                pair_viirs_mask=[Path("pair_viirs_before.mask"), Path("pair_viirs_after.mask")],
                prediction_viirs=Path("prediction_viirs.bin"),
                prediction_viirs_mask=Path("prediction_viirs.mask"),
                prediction_output=Path("prediction.bin"),
                rows=2,
                columns=3,
                resolution=30,
                max_search_distance=750,
                uncertainty=50,
                use_mask_files=False,
            )
            content = parameters.read_text(encoding="utf-8")
            self.assertNotIn("_MASK", content)
            self.assertIn("NUM_IN_PAIRS = 2", content)
            self.assertIn("pair_landsat_before.bin pair_landsat_after.bin", content)

    def test_starfm_parameter_paths_stay_short_and_cwd_relative(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / ("nested_" + "x" * 80) / "STARFM"
            inputs = output_directory / "INPUTS"
            inputs.mkdir(parents=True)
            landsat = inputs / "landsat_20230902_B1.bin"
            viirs = inputs / "viirs_20230902_B1.bin"
            prediction = inputs / "viirs_20230918_B1.bin"
            for path in (landsat, viirs, prediction):
                path.write_bytes(b"")
            parameters = output_directory / "starfm_parameters.txt"
            _write_starfm_parameter_file(
                parameters,
                pair_landsat=[_starfm_cwd_relative(landsat, output_directory)],
                pair_landsat_mask=[],
                pair_viirs=[_starfm_cwd_relative(viirs, output_directory)],
                pair_viirs_mask=[],
                prediction_viirs=_starfm_cwd_relative(prediction, output_directory),
                prediction_viirs_mask=Path("unused.mask"),
                prediction_output=Path("predicted_B1_20230918.bin"),
                rows=2,
                columns=3,
                resolution=30,
                max_search_distance=750,
                uncertainty=50,
                use_mask_files=False,
            )
            content = parameters.read_text(encoding="utf-8")
            self.assertIn("OUT_PDAY_LANDSAT_FNAME = predicted_B1_20230918.bin", content)
            self.assertIn("IN_PAIR_LANDSAT_FNAME = INPUTS/landsat_20230902_B1.bin", content)
            self.assertNotIn(str(output_directory), content)
            self.assertLess(
                len("predicted_B1_20230918.bin.hdr".encode("utf-8")),
                STARFM_HEADER_FILENAME_BUF,
            )
            with self.assertRaises(ValueError):
                _validate_starfm_output_filename(Path("a" * STARFM_HEADER_FILENAME_BUF))

    def test_selects_nearest_clear_scenes_on_both_sides_of_prediction(self):
        items = [
            {"id": "LC08_L2SP_029032_20220907_20220914_02_T1", "properties": {"datetime": "2022-09-07T17:13:09Z", "eo:cloud_cover": 1}},
            {"id": "LC09_L2SP_029032_20220915_20230329_02_T1", "properties": {"datetime": "2022-09-15T17:13:21Z", "eo:cloud_cover": 2}},
            {"id": "LC08_L2SP_029032_20220822_20220829_02_T1", "properties": {"datetime": "2022-08-22T17:13:09Z", "eo:cloud_cover": 0}},
        ]
        before, after = select_bracketing_landsat_items(items, date(2022, 9, 8), 29, 32, 10)
        self.assertEqual(before["id"], "LC08_L2SP_029032_20220907_20220914_02_T1")
        self.assertEqual(after["id"], "LC09_L2SP_029032_20220915_20230329_02_T1")

    def test_prediction_output_uses_landsat_c2_stored_value_scale(self):
        with TemporaryDirectory() as temporary_directory:
            reference_path = Path(temporary_directory) / "reference.tif"
            output_path = Path(temporary_directory) / "prediction.tif"
            reference = gdal.GetDriverByName("GTiff").Create(
                str(reference_path), 2, 2, 1, gdal.GDT_UInt16
            )
            reference.GetRasterBand(1).SetNoDataValue(0)
            reference.GetRasterBand(1).WriteArray(np.full((2, 2), 10000, dtype=np.uint16))
            _write_prediction_geotiff(
                output_path,
                np.array([[0, 1234], [STARFM_FILL_VALUE, 6341]], dtype="<i2"),
                reference,
            )
            reference = None
            output = gdal.Open(str(output_path))
            band = output.GetRasterBand(1)
            np.testing.assert_array_equal(
                band.ReadAsArray(),
                np.array([[7273, 11760], [0, 30331]], dtype=np.uint16),
            )
            self.assertEqual(band.GetNoDataValue(), 0)
            self.assertIn(band.GetScale(), (None, 1.0))
            self.assertIn(band.GetOffset(), (None, 0.0))

    def test_quality_mask_is_disabled_by_default(self):
        args = build_parser().parse_args(
            ["--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4"]
        )
        self.assertFalse(args.use_quality_mask)
        self.assertFalse(args.keep_intermediate_outputs)
        self.assertEqual(args.pair_count, 2)

    def test_parser_accepts_one_pair_mode(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4",
                "--pair-count", "1",
            ]
        )
        self.assertEqual(args.pair_count, 1)

    def test_explicit_pair_dates_are_sorted_and_cannot_leak_prediction_date(self):
        normalized = _normalize_explicit_pair_dates(
            [date(2024, 3, 27), date(2024, 3, 11)], date(2024, 3, 19)
        )
        self.assertEqual(normalized, (date(2024, 3, 11), date(2024, 3, 27)))
        with self.assertRaises(ValueError):
            _normalize_explicit_pair_dates([date(2024, 3, 19)], date(2024, 3, 19))

    def test_parser_accepts_one_or_two_explicit_pair_dates(self):
        one_pair = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B3",
                "--pair-dates", "2024-03-11",
            ]
        )
        two_pairs = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B3",
                "--pair-dates", "2024-02-24", "2024-03-11",
            ]
        )
        self.assertEqual(one_pair.pair_dates, [date(2024, 3, 11)])
        self.assertEqual(two_pairs.pair_dates, [date(2024, 2, 24), date(2024, 3, 11)])

    def test_test_id_is_portable_and_parser_accepts_it(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B3",
                "--test-id", "ag_stable_01",
            ]
        )
        self.assertEqual(args.test_id, "ag_stable_01")
        with self.assertRaises(Exception):
            _test_id("../unsafe")

    def test_parser_accepts_modis_sensor_selection(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B4",
                "--coarse-sensor", "MODIS",
            ]
        )
        self.assertEqual(args.coarse_sensor, "MODIS")

    def test_coregistration_method_defaults_to_moving_window_and_accepts_alternatives(self):
        default = build_parser().parse_args(
            ["--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4"]
        )
        arosics = build_parser().parse_args(
            [
                "--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4",
                "--coregistration-method", "arosics",
            ]
        )
        no_coregistration = build_parser().parse_args(
            [
                "--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4",
                "--coregistration-method", "none",
            ]
        )
        self.assertEqual(default.coregistration_method, "moving-window")
        self.assertEqual(arosics.coregistration_method, "arosics")
        self.assertEqual(no_coregistration.coregistration_method, "none")

    def test_coregistration_band_is_independent_from_the_requested_output_band(self):
        args = build_parser().parse_args(
            [
                "--prediction-date", "2024-03-19", "--path", "143", "--row", "42", "--band", "B3",
                "--coarse-sensor", "MODIS", "--coregistration-band", "B5",
            ]
        )
        self.assertEqual(args.band, "B3")
        self.assertEqual(args.coregistration_band, "B5")

    def test_coregistration_quality_metadata_uses_the_method_specific_percentage(self):
        moving_metric, moving_quality = _coregistration_quality_metadata(
            "moving-window", {"best_correlation": 0.295951}
        )
        self.assertEqual(moving_metric, "best_pearson_correlation_percent")
        self.assertAlmostEqual(moving_quality, 29.5951)
        arosics_metric, arosics_quality = _coregistration_quality_metadata(
            "arosics", {"shift_reliability_percent": 98.5}
        )
        self.assertEqual(arosics_metric, "arosics_shift_reliability_percent")
        self.assertEqual(arosics_quality, 98.5)
        none_metric, none_quality = _coregistration_quality_metadata("none", {})
        self.assertEqual(none_metric, "not_applicable_no_coregistration")
        self.assertIsNone(none_quality)

    def test_parser_enables_heldout_landsat_validation_only_when_requested(self):
        default = build_parser().parse_args(
            ["--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4"]
        )
        enabled = build_parser().parse_args(
            [
                "--prediction-date", "2022-09-08", "--path", "29", "--row", "32", "--band", "B4",
                "--validate-held-out-landsat",
            ]
        )
        self.assertFalse(default.validate_held_out_landsat)
        self.assertTrue(enabled.validate_held_out_landsat)

    def test_heldout_validation_preserves_intermediate_outputs(self):
        self.assertFalse(_should_keep_intermediate_outputs(False, False))
        self.assertTrue(_should_keep_intermediate_outputs(True, False))
        self.assertTrue(_should_keep_intermediate_outputs(False, True))

    def test_heldout_validation_uses_clear_target_pixels_and_physical_reflectance(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predicted_path = root / "predicted.tif"
            actual_path = root / "actual.tif"
            qa_path = root / "actual_qa.tif"
            transform = (500000, 30, 0, 4500000, 0, -30)
            projection = "EPSG:32615"

            def create(path, values, data_type):
                dataset = gdal.GetDriverByName("GTiff").Create(
                    str(path), 2, 2, 1, data_type
                )
                dataset.SetGeoTransform(transform)
                dataset.SetProjection(projection)
                dataset.GetRasterBand(1).SetNoDataValue(0)
                dataset.GetRasterBand(1).WriteArray(values)
                dataset = None

            create(predicted_path, np.array([[11000, 22000], [0, 29000]], dtype=np.uint16), gdal.GDT_UInt16)
            create(actual_path, np.array([[10000, 20000], [0, 30000]], dtype=np.uint16), gdal.GDT_UInt16)
            create(qa_path, np.array([[0, 0], [0, 1 << 3]], dtype=np.uint16), gdal.GDT_UInt16)
            metrics = _evaluate_heldout_landsat(
                predicted_landsat=predicted_path,
                heldout_landsat=actual_path,
                heldout_landsat_qa=qa_path,
                output_directory=root / "validation",
                band_name="B4",
            )
            self.assertEqual(metrics["valid_comparison_pixels"], 2)
            self.assertAlmostEqual(metrics["valid_comparison_percent"], 50.0)
            self.assertAlmostEqual(metrics["actual_mean_surface_reflectance"], 0.2125)
            self.assertAlmostEqual(metrics["predicted_mean_surface_reflectance"], 0.25375)
            self.assertAlmostEqual(metrics["mean_difference_predicted_minus_actual"], 0.04125)
            self.assertAlmostEqual(metrics["mae_surface_reflectance"], 0.04125)
            self.assertAlmostEqual(metrics["rmse_surface_reflectance"], np.sqrt((0.0275 ** 2 + 0.055 ** 2) / 2))
            self.assertAlmostEqual(metrics["r2_surface_reflectance"], 0.9)
            saved = json.loads(Path(metrics["metrics_json"]).read_text())
            self.assertEqual(saved["r2_surface_reflectance"], metrics["r2_surface_reflectance"])
            with Path(metrics["metrics_csv"]).open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["mae_surface_reflectance"]), 0.04125)
            self.assertAlmostEqual(metrics["relative_error_percent"], 19.4117647059)
            self.assertTrue(Path(metrics["metrics_csv"]).is_file())
            self.assertTrue(Path(metrics["predicted_minus_actual"]).is_file())


if __name__ == "__main__":
    unittest.main()
