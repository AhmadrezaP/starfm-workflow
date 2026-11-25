"""Download and preprocess one or two Landsat--coarse-sensor pairs for STARFM.

Given a *prediction date*, WRS-2 path/row, and a supported Landsat band, this
script selects either the nearest clear Landsat scene or one clear scene on
each side of the prediction:

1. find one (``--pair-count 1``) or two (``--pair-count 2``) Landsat 8/9
   Collection 2 Level-2 SR scenes, or use explicitly supplied ``--pair-dates``;
2. download and align every Landsat observation to one common fine grid;
3. download the required VIIRS or MODIS daily reflectance, BRDF, and MCD12Q1 inputs;
4. create daily coarse-sensor NBAR and run moving-window, AROSICS, or a
   grid-alignment-only no-coregistration baseline, then apply that treatment
   consistently to every coarse-sensor date; and
5. prepare the requested original-STARFM pairs, run STARFM, and export its prediction as a
   georeferenced GeoTIFF.

NASA downloads require an Earthdata Login token. Credentials can be kept in
the ignored ``config.yaml`` file, or supplied through the environment-variable
names accepted by the credential command-line options. They are read at runtime
and are never written to the manifest or command line. Landsat discovery uses
the public USGS LandsatLook STAC service, while its individual-file download
uses a USGS EROS M2M application token.

VIIRS mode supports B1, B2, B3, B4, B5, and B7. MODIS mode supports B2, B3,
B4, B5, and B7; MODIS has no coastal-aerosol counterpart for Landsat B1.
Thermal B6 is deliberately rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
import yaml
from osgeo import gdal
from shared_cache import cached_download, cached_product, cache_key, file_lock


USGS_STAC_SEARCH = "https://landsatlook.usgs.gov/stac-server/search"
USGS_M2M_API = "https://m2m.cr.usgs.gov/api/api/json/stable/"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
REQUEST_TIMEOUT_SECONDS = 90
# MCD12Q1 is published long after its year ends; recent dates use the newest
# earlier year available, at most this many years back.
LAND_COVER_MAX_FALLBACK_YEARS = 3
# USGS and NASA services can answer bursts of requests with 5xx; retry those
# with jittered backoff.
REQUEST_RETRY_ATTEMPTS = 5
# M2M also limits how often one user may log in. Space logins out and wait
# minutes, not seconds, when that limit is reported.
M2M_LOGIN_SPACING_SECONDS = 5
M2M_RATE_LIMIT_WAIT_SECONDS = (60, 180)


@dataclass(frozen=True)
class BandInfo:
    landsat_band: str
    viirs_band: str
    parameter_products: tuple[str, str, str]
    modis_band: int | None


@dataclass(frozen=True)
class StarfmPair:
    """One fine/coarse observation pair on the common STARFM grid."""

    acquisition_date: date
    landsat: Path
    landsat_qa: Path
    viirs: Path
    valid_mask: Path | None


# VNP43D01--D27 store fiso/fvol/fgeo consecutively for M1--M5, M7, M8,
# M10, and M11.  These are the spectral correspondences used by myStarfm.py.
BAND_INFO: dict[str, BandInfo] = {
    "B1": BandInfo("B1", "M2", ("VNP43D04", "VNP43D05", "VNP43D06"), None),
    "B2": BandInfo("B2", "M3", ("VNP43D07", "VNP43D08", "VNP43D09"), 3),
    "B3": BandInfo("B3", "M4", ("VNP43D10", "VNP43D11", "VNP43D12"), 4),
    "B4": BandInfo("B4", "M5", ("VNP43D13", "VNP43D14", "VNP43D15"), 1),
    "B5": BandInfo("B5", "M7", ("VNP43D16", "VNP43D17", "VNP43D18"), 2),
    "B7": BandInfo("B7", "M11", ("VNP43D25", "VNP43D26", "VNP43D27"), 7),
}
STARFM_SCALE_FACTOR = 10_000
STARFM_FILL_VALUE = -9999
STARFM_DATA_MIN = 0
STARFM_DATA_MAX = 10_000
# Original StarFM_util.c writeHeader() uses char tmpname[200] for "<out>.hdr".
STARFM_HEADER_FILENAME_BUF = 200
LANDSAT_C2_SR_SCALE_FACTOR = 0.0000275
LANDSAT_C2_SR_OFFSET = -0.2


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Date must be YYYY-MM-DD.") from error


def _normalize_explicit_pair_dates(
    pair_dates: Sequence[date] | None,
    prediction_date: date,
) -> tuple[date, ...] | None:
    """Validate and chronologically order explicit STARFM pair dates."""
    if pair_dates is None:
        return None
    if len(pair_dates) not in (1, 2):
        raise ValueError("--pair-dates needs exactly one or two YYYY-MM-DD dates.")
    if len(set(pair_dates)) != len(pair_dates):
        raise ValueError("--pair-dates cannot contain the same date more than once.")
    if prediction_date in pair_dates:
        raise ValueError(
            "--pair-dates cannot include --prediction-date; that would leak target-date Landsat into STARFM."
        )
    return tuple(sorted(pair_dates))


def _test_id(value: str) -> str:
    """Accept a portable, human-readable identifier for one experiment run."""
    normalized = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", normalized):
        raise argparse.ArgumentTypeError(
            "--test-id must be 1--80 characters using letters, digits, underscores, or hyphens."
        )
    return normalized


def _write_test_summary(
    output_directory: Path,
    manifest: dict[str, Any],
    coregistration_report: dict[str, Any],
) -> dict[str, str]:
    """Write a compact, spreadsheet-friendly record of one named experiment."""
    validation = manifest["outputs"]["heldout_landsat_validation"] or {}
    coregistration_quality_metric, coregistration_quality_percent = _coregistration_quality_metadata(
        manifest["coregistration_method"], coregistration_report
    )
    summary = {
        "test_id": manifest["test_id"],
        "prediction_date": manifest["prediction_date"],
        "pair_dates": ";".join(manifest["selected_pair_dates"]),
        "pair_count": manifest["pair_count"],
        "selection_method": manifest["selection_method"],
        "coarse_sensor": manifest["coarse_sensor"],
        "predicted_landsat_band": manifest["requested_landsat_band"],
        "coregistration_method": manifest["coregistration_method"],
        "coregistration_landsat_band": manifest["coregistration_landsat_band"],
        "coregistration_quality_metric": coregistration_quality_metric,
        "coregistration_quality_percent": coregistration_quality_percent,
        "coregistration_x_shift_pixels": coregistration_report.get("x_shift_pixels"),
        "coregistration_y_shift_pixels": coregistration_report.get("y_shift_pixels"),
        "starfm_max_search_distance_m": manifest["starfm_max_search_distance_m"],
        "starfm_uncertainty_scaled": manifest["starfm_uncertainty_scaled"],
        "max_cloud_percent": manifest["max_cloud_percent"],
        "rmse_surface_reflectance": validation.get("rmse_surface_reflectance"),
        "mae_surface_reflectance": validation.get("mae_surface_reflectance"),
        "r2_surface_reflectance": validation.get("r2_surface_reflectance"),
        "mean_difference_predicted_minus_actual": validation.get("mean_difference_predicted_minus_actual"),
        "relative_error_percent": validation.get("relative_error_percent"),
        "valid_comparison_percent": validation.get("valid_comparison_percent"),
        "valid_comparison_pixels": validation.get("valid_comparison_pixels"),
        "total_prediction_grid_pixels": validation.get("total_prediction_grid_pixels"),
        "final_output_geotiff": manifest["outputs"]["final_output_geotiff"],
        "coregistration_report": manifest["coregistration_report"],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "test_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    json_path = output_directory / "test_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path.absolute()), "json": str(json_path.absolute())}


def _band(value: str) -> str:
    normalized = value.upper().replace("SR_", "")
    if normalized not in BAND_INFO:
        raise argparse.ArgumentTypeError(
            f"Unsupported band {value!r}. Choose one of: {', '.join(BAND_INFO)}."
        )
    return normalized


def _iso_interval(day: date) -> str:
    return f"{day.isoformat()}T00:00:00Z,{day.isoformat()}T23:59:59Z"


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Download URL has no filename: {url}")
    return name


def _viirs_daily_id(path: Path) -> str:
    """Return a stable product/date/tile identifier without archive timestamps."""
    match = re.match(r"(VNP09GA\.A\d{7}\.h\d{2}v\d{2})", path.name)
    if not match:
        raise RuntimeError(f"Could not derive a standard VIIRS daily identifier from: {path.name}")
    return match.group(1)


def _modis_daily_id(path: Path) -> str:
    """Return a stable MOD09GA product/date/tile identifier."""
    match = re.match(r"(MOD09GA\.A\d{7}\.h\d{2}v\d{2})", path.name)
    if not match:
        raise RuntimeError(f"Could not derive a standard MODIS daily identifier from: {path.name}")
    return match.group(1)


def _retryable_status(status_code: int) -> bool:
    return status_code >= 500 or status_code == 429


def _retry_pause(label: str, attempt: int, reason: str) -> None:
    delay = random.uniform(5, 15) * 2 ** (attempt - 1)
    print(f"{label} attempt {attempt} failed ({reason}); retrying in {delay:.0f} s.", flush=True)
    time.sleep(delay)


def _json_with_retry(send: Any, label: str) -> Any:
    """Return ``send()``'s JSON, retrying only server errors, throttling and dropped connections."""
    for attempt in range(1, REQUEST_RETRY_ATTEMPTS + 1):
        last = attempt == REQUEST_RETRY_ATTEMPTS
        try:
            response = send()
            if last or not _retryable_status(response.status_code):
                response.raise_for_status()
                return response.json()
            reason = f"HTTP {response.status_code}"
        except (requests.ConnectionError, requests.Timeout) as error:
            if last:
                raise
            reason = type(error).__name__
        _retry_pause(label, attempt, reason)
    raise AssertionError("unreachable")


def _request_json(url: str, *, params: dict[str, Any] | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is not None:
        send = lambda: requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    else:
        send = lambda: requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    parsed = urlparse(url)
    return _json_with_retry(send, f"{parsed.netloc}{parsed.path}")


def _download(
    url: str,
    destination: Path,
    token: str | None = None,
    m2m_token: str | None = None,
) -> Path:
    return cached_download(destination, _valid_download,
                           lambda staged: _download_unlocked(url, staged, token, m2m_token))


def _download_unlocked(url, destination, token=None, m2m_token=None) -> Path:
    """Download one asset atomically, resuming by filename when it exists."""
    if destination.exists() and destination.stat().st_size > 0 and _valid_download(destination):
        return destination
    destination.unlink(missing_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if m2m_token:
        headers["X-Auth-Token"] = m2m_token
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Server errors, throttling and dropped or truncated transfers are retried;
    # client errors and wrong content fail at once.
    for attempt in range(1, REQUEST_RETRY_ATTEMPTS + 1):
        last = attempt == REQUEST_RETRY_ATTEMPTS
        reason = None
        try:
            with requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True) as response:
                if not last and _retryable_status(response.status_code):
                    reason = f"HTTP {response.status_code}"
                else:
                    response.raise_for_status()
                    response_content_type = response.headers.get("Content-Type", "unknown")
                    response_host = urlparse(response.url).netloc
                    with temporary.open("wb") as stream:
                        received = 0
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                                received += len(chunk)
                    expected_length = response.headers.get('Content-Length')
                    if expected_length and not response.headers.get('Content-Encoding') and received != int(expected_length):
                        if last:
                            raise RuntimeError(f"Incomplete download for {destination.name}: {received}/{expected_length} bytes")
                        reason = f"incomplete {received}/{expected_length} bytes"
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as error:
            if last:
                raise
            reason = type(error).__name__
        if reason is None:
            break
        _retry_pause(f"Download {destination.name}", attempt, reason)
    temporary.replace(destination)
    if not _valid_download(destination):
        with destination.open("rb") as stream:
            signature = stream.read(16).hex()
        size = destination.stat().st_size
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {destination.name}, but it is not the expected binary data file. "
            f"Response host={response_host}, content-type={response_content_type}, "
            f"size={size} bytes, signature={signature}."
        )
    return destination


def _valid_download(path: Path) -> bool:
    """Reject an HTML login/error response saved under an imagery filename."""
    with path.open("rb") as stream:
        header = stream.read(16)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return header.startswith((
            bytes.fromhex("49492a00"),  # classic little-endian TIFF
            bytes.fromhex("4d4d002a"),  # classic big-endian TIFF
            bytes.fromhex("49492b00"),  # little-endian BigTIFF
        ))
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return header.startswith(bytes.fromhex("894844460d0a1a0a"))
    if path.suffix.lower() == ".hdf":
        return header.startswith(bytes.fromhex("0e031301"))
    return bool(header)


def _item_datetime(item: dict[str, Any]) -> datetime:
    raw = item["properties"].get("datetime") or item["properties"].get("start_datetime")
    if raw is None:
        raise ValueError("Landsat STAC item has no datetime.")
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def select_nearest_landsat_item(
    items: Iterable[dict[str, Any]],
    target_date: date,
    path: int,
    row: int,
    max_cloud_percent: float,
    *,
    exclude_target_date: bool = False,
) -> dict[str, Any]:
    """Choose nearest eligible date; cloud cover breaks an equal-date tie."""
    target = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    usable: list[dict[str, Any]] = []
    path_row_pattern = re.compile(rf"^LC0[89]_L2S[PR]_{path:03d}{row:03d}_")
    for item in items:
        properties = item.get("properties", {})
        identifier = item.get("id", "")
        cloud = float(properties.get("eo:cloud_cover", 100.0))
        if (
            path_row_pattern.match(identifier)
            and cloud <= max_cloud_percent
            and (not exclude_target_date or _item_datetime(item).date() != target_date)
        ):
            usable.append(item)
    if not usable:
        raise RuntimeError(
            f"No Landsat 8/9 L2SP scene for WRS-2 {path:03d}/{row:03d} met the "
            f"{max_cloud_percent:g}% scene-cloud limit in the search range."
        )
    return min(
        usable,
        key=lambda item: (
            abs(_item_datetime(item) - target),
            float(item["properties"].get("eo:cloud_cover", 100.0)),
            item["id"],
        ),
    )


def select_bracketing_landsat_items(
    items: Iterable[dict[str, Any]],
    target_date: date,
    path: int,
    row: int,
    max_cloud_percent: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the nearest eligible Landsat scene before and after prediction."""
    path_row_pattern = re.compile(rf"^LC0[89]_L2S[PR]_{path:03d}{row:03d}_")
    eligible = [
        item
        for item in items
        if path_row_pattern.match(str(item.get("id", "")))
        and float(item.get("properties", {}).get("eo:cloud_cover", 100.0)) <= max_cloud_percent
    ]
    before = [item for item in eligible if _item_datetime(item).date() < target_date]
    after = [item for item in eligible if _item_datetime(item).date() > target_date]
    if not before or not after:
        missing = "before" if not before else "after"
        raise RuntimeError(
            f"No Landsat 8/9 L2SP scene {missing} {target_date.isoformat()} for WRS-2 "
            f"{path:03d}/{row:03d} met the {max_cloud_percent:g}% scene-cloud limit. "
            "Two-pair STARFM requires one eligible scene on each side of the prediction date."
        )
    key = lambda item: (
        abs((_item_datetime(item).date() - target_date).days),
        float(item["properties"].get("eo:cloud_cover", 100.0)),
        item["id"],
    )
    return min(before, key=key), min(after, key=key)


def find_nearest_landsat(
    target_date: date,
    path: int,
    row: int,
    maximum_days: int,
    max_cloud_percent: float,
    *,
    exclude_target_date: bool = False,
) -> dict[str, Any]:
    start = target_date - timedelta(days=maximum_days)
    end = target_date + timedelta(days=maximum_days)
    payload = {
        "collections": ["landsat-c2l2-sr"],
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": 100,
        "query": {
            # LandsatLook STAC represents WRS-2 values as zero-padded strings,
            # e.g. "029" and "032", rather than JSON numbers.
            "landsat:wrs_path": {"eq": f"{path:03d}"},
            "landsat:wrs_row": {"eq": f"{row:03d}"},
        },
    }
    response = _request_json(USGS_STAC_SEARCH, payload=payload)
    return select_nearest_landsat_item(
        response.get("features", []),
        target_date,
        path,
        row,
        max_cloud_percent,
        exclude_target_date=exclude_target_date,
    )


def find_landsat_on_date(
    target_date: date, path: int, row: int, max_cloud_percent: float
) -> dict[str, Any]:
    """Find the eligible Landsat 8/9 L2 surface-reflectance scene on one date."""
    payload = {
        "collections": ["landsat-c2l2-sr"],
        "datetime": f"{target_date.isoformat()}T00:00:00Z/{target_date.isoformat()}T23:59:59Z",
        "limit": 100,
        "query": {
            "landsat:wrs_path": {"eq": f"{path:03d}"},
            "landsat:wrs_row": {"eq": f"{row:03d}"},
        },
    }
    response = _request_json(USGS_STAC_SEARCH, payload=payload)
    exact_date_items = [
        item
        for item in response.get("features", [])
        if _item_datetime(item).date() == target_date
    ]
    if not exact_date_items:
        raise RuntimeError(
            f"No Landsat 8/9 L2 surface-reflectance scene exists on {target_date.isoformat()} "
            f"for WRS-2 {path:03d}/{row:03d}."
        )
    return select_nearest_landsat_item(
        exact_date_items, target_date, path, row, max_cloud_percent
    )


def find_bracketing_landsat(
    target_date: date, path: int, row: int, maximum_days: int, max_cloud_percent: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = target_date - timedelta(days=maximum_days)
    end = target_date + timedelta(days=maximum_days)
    payload = {
        "collections": ["landsat-c2l2-sr"],
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": 100,
        "query": {
            "landsat:wrs_path": {"eq": f"{path:03d}"},
            "landsat:wrs_row": {"eq": f"{row:03d}"},
        },
    }
    response = _request_json(USGS_STAC_SEARCH, payload=payload)
    return select_bracketing_landsat_items(
        response.get("features", []), target_date, path, row, max_cloud_percent
    )


def _asset_url(item: dict[str, Any], asset_key: str) -> str:
    assets = item.get("assets", {})
    if asset_key in assets and "href" in assets[asset_key]:
        return str(assets[asset_key]["href"])
    # LandsatLook has used both semantic asset keys and source filename keys.
    suffix = {"B1": "SR_B1.TIF", "B2": "SR_B2.TIF", "B3": "SR_B3.TIF", "B4": "SR_B4.TIF", "B5": "SR_B5.TIF", "B7": "SR_B7.TIF", "QA_PIXEL": "QA_PIXEL.TIF"}[asset_key]
    for asset in assets.values():
        href = str(asset.get("href", ""))
        if href.upper().endswith(suffix):
            return href
    semantic = {"B1": "coastal", "B2": "blue", "B3": "green", "B4": "red", "B5": "nir08", "B7": "swir22", "QA_PIXEL": "qa_pixel"}[asset_key]
    if semantic in assets and "href" in assets[semantic]:
        return str(assets[semantic]["href"])
    raise RuntimeError(f"Landsat STAC item lacks the required {asset_key} asset.")


def _usgs_post(endpoint: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    return _json_with_retry(
        lambda: requests.post(f"{USGS_M2M_API}{endpoint}", headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS),
        f"USGS M2M {endpoint}",
    )


def _usgs_api(endpoint: str, *, token: str, body: dict[str, Any]) -> dict[str, Any]:
    result = _usgs_post(endpoint, body, headers={"X-Auth-Token": token})
    if result.get("errorCode"):
        raise RuntimeError(f"USGS M2M {endpoint} failed: {result.get('errorCode')} - {result.get('errorMessage')}")
    return result


def _read_credentials_config(path: Path) -> dict[str, str]:
    """Load optional local credentials without exposing their values in outputs."""
    if not path.exists():
        return {}
    try:
        contents = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise RuntimeError(f"Could not parse credentials config {path}: {error}") from error
    if not isinstance(contents, dict):
        raise RuntimeError(f"Credentials config {path} must contain a YAML mapping.")
    credentials = contents.get("credentials", contents)
    if not isinstance(credentials, dict):
        raise RuntimeError(f"The credentials section in {path} must be a YAML mapping.")
    return {str(key): str(value).strip() for key, value in credentials.items() if value is not None and str(value).strip()}


def _credential(config: dict[str, str], key: str, environment_name: str) -> str | None:
    """Prefer an explicit environment override, then use the ignored local config."""
    return os.environ.get(environment_name) or config.get(key)


def _m2m_login(body: dict[str, str]) -> dict[str, Any]:
    """Log in, waiting out M2M's per-user authentication rate limit."""
    for attempt in range(1, REQUEST_RETRY_ATTEMPTS + 1):
        result = _usgs_post("login-token", body)
        if not str(result.get("errorCode") or "").startswith("RATE_LIMIT") or attempt == REQUEST_RETRY_ATTEMPTS:
            return result
        delay = random.uniform(*M2M_RATE_LIMIT_WAIT_SECONDS) * attempt
        print(f"USGS M2M login rate-limited ({result['errorCode']}); retrying in {delay:.0f} s.", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _usgs_m2m_token(
    api_key: str | None, username: str | None, application_token: str | None, lock_directory: Path | None = None,
) -> str:
    """Use an existing M2M session key, or exchange an ERS application token.

    Concurrent logins for one account can fail with HTTP 500, so jobs sharing
    ``lock_directory`` log in one at a time.
    """
    if api_key:
        return api_key
    if not username or not application_token:
        raise RuntimeError(
            "Set usgs_m2m_api_key, or both usgs_username and usgs_m2m_token, in config.yaml "
            "(or use the corresponding environment variables) before downloading Landsat. "
            "USGS M2M now requires an ERS application token; an account password is not accepted by its API."
        )
    body = {"username": username, "token": application_token}
    if lock_directory is None:
        result = _m2m_login(body)
    else:
        # Waiting jobs queue on the lock, so a rate-limited login holds them back too.
        with file_lock(lock_directory / ".m2m_login.lock"):
            result = _m2m_login(body)
            time.sleep(M2M_LOGIN_SPACING_SECONDS)
    if result.get("errorCode") or not result.get("data"):
        raise RuntimeError(f"USGS M2M token login failed: {result.get('errorCode')} - {result.get('errorMessage')}")
    return str(result["data"])


def _usgs_scene_entity_id(
    landsat_id: str,
    scene_bbox: tuple[float, float, float, float],
    acquisition_date: date,
    m2m_token: str,
) -> str:
    """Resolve a public Landsat product ID to M2M's internal scene entity ID."""
    min_x, min_y, max_x, max_y = scene_bbox
    search = _usgs_api(
        "scene-search",
        token=m2m_token,
        body={
            "datasetName": "landsat_ot_c2_l2",
            "sceneFilter": {
                "acquisitionFilter": {
                    "start": acquisition_date.isoformat(),
                    "end": acquisition_date.isoformat(),
                },
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {"latitude": min_y, "longitude": min_x},
                    "upperRight": {"latitude": max_y, "longitude": max_x},
                },
            },
        },
    )
    target_display_id = re.sub(r"_SR$", "", landsat_id)
    results = (search.get("data") or {}).get("results") or []
    for scene in results:
        if str(scene.get("displayId", "")) == target_display_id and scene.get("entityId"):
            return str(scene["entityId"])
    available = [str(scene.get("displayId", "")) for scene in results]
    raise RuntimeError(
        "USGS M2M could not resolve the selected Landsat scene to an internal entity ID. "
        f"Wanted: {target_display_id}. M2M returned: {available}"
    )


def _download_landsat_assets(
    landsat_id: str,
    needed_assets: Iterable[str],
    destination: Path,
    m2m_token: str,
    scene_bbox: tuple[float, float, float, float],
    acquisition_date: date,
) -> dict[str, Path]:
    """Request verified individual C2 files through the official USGS M2M API."""
    scene_entity_id = re.sub(r"_SR$", "", landsat_id)
    m2m_entity_id = _usgs_scene_entity_id(landsat_id, scene_bbox, acquisition_date, m2m_token)
    options = _usgs_api(
        "download-options", token=m2m_token,
        body={"entityIds": m2m_entity_id, "datasetName": "landsat_ot_c2_l2"},
    )
    required_suffixes = {
        "B1": "_SR_B1.TIF", "B2": "_SR_B2.TIF", "B3": "_SR_B3.TIF",
        "B4": "_SR_B4.TIF", "B5": "_SR_B5.TIF", "B7": "_SR_B7.TIF",
        "QA_PIXEL": "_QA_PIXEL.TIF",
    }
    requested = {asset: required_suffixes[asset] for asset in needed_assets}
    # M2M entity IDs are not guaranteed to include the file extension, whereas
    # their band marker is stable in both the legacy and current API responses.
    requested_markers = {asset: suffix.removesuffix(".TIF") for asset, suffix in requested.items()}
    downloads: list[dict[str, str]] = []
    option_data = options.get("data") or []
    for option in option_data:
        # Older M2M responses nest individual files in ``secondaryDownloads``.
        # Current responses may instead return one top-level entry per file.
        candidates = [option, *(option.get("secondaryDownloads") or [])]
        for item in candidates:
            entity_id = str(item.get("entityId", ""))
            for marker in requested_markers.values():
                if marker in entity_id.upper() and item.get("id"):
                    if not any(download["entityId"] == entity_id for download in downloads):
                        downloads.append({"entityId": entity_id, "productId": str(item["id"])})
                    break
    if len(downloads) != len(requested):
        found = [item["entityId"] for item in downloads]
        product_names = [str(option.get("productName", "")) for option in option_data]
        raise RuntimeError(
            "USGS M2M did not provide every requested Landsat asset. "
            f"Found: {found}; product groups: {product_names}"
        )
    request = _usgs_api(
        "download-request", token=m2m_token,
        # USGS validates this against registered M2M download applications.
        body={"downloadApplication": "EE", "downloads": downloads},
    )
    available = request.get("data", {}).get("availableDownloads", [])
    by_asset: dict[str, str] = {}
    for item in available:
        entity_id, url = str(item.get("entityId", "")), item.get("url")
        if url:
            for asset, marker in requested_markers.items():
                if marker in entity_id.upper():
                    by_asset[asset] = str(url)
    missing = set(requested) - set(by_asset)
    if missing:
        raise RuntimeError(f"USGS M2M did not immediately provide download URLs for: {sorted(missing)}")
    return {
        asset: _download(
            by_asset[asset],
            destination / f"{scene_entity_id}{suffix}",
            m2m_token=m2m_token,
        )
        for asset, suffix in requested.items()
    }


def _cmr_entries(short_name: str, version: str, day: date, bbox: tuple[float, float, float, float] | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "short_name": short_name,
        "version": version,
        "temporal": _iso_interval(day),
        "page_size": 200,
        "downloadable": "true",
    }
    if bbox is not None:
        params["bounding_box"] = ",".join(str(value) for value in bbox)
    result = _request_json(CMR_GRANULES, params=params)
    return list(result.get("feed", {}).get("entry", []))


def _cmr_entry_for_filename_date(
    short_name: str, version: str, acquisition_date: date, bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Return the first file whose AYYYYDDD token exactly matches the requested day."""
    return _cmr_entries_for_filename_date(short_name, version, acquisition_date, bbox)[0]


def _cmr_entries_for_filename_date(
    short_name: str,
    version: str,
    acquisition_date: date,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Find every CMR granule whose filename date exactly matches the requested day."""
    expected = f"A{acquisition_date.year}{acquisition_date.timetuple().tm_yday:03d}"
    candidates: list[dict[str, Any]] = []
    for offset in (-1, 0, 1):
        candidates.extend(_cmr_entries(short_name, version, acquisition_date + timedelta(days=offset), bbox))
    matching = [entry for entry in candidates if f".{expected}." in _entry_name(entry)]
    if matching:
        return matching
    names = sorted({_entry_name(entry) for entry in candidates})
    raise RuntimeError(f"CMR returned no {short_name} file labelled {expected}. Nearby results: {names}")


def _land_cover_entries(
    acquisition_date: date, bbox: tuple[float, float, float, float] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return MCD12Q1 entries and the year used, falling back to earlier years."""
    error: RuntimeError | None = None
    for year in range(acquisition_date.year, acquisition_date.year - LAND_COVER_MAX_FALLBACK_YEARS - 1, -1):
        try:
            return _cmr_entries_for_filename_date("MCD12Q1", "061", date(year, 1, 1), bbox), year
        except RuntimeError as missing:
            if "CMR returned no MCD12Q1" not in str(missing):
                raise
            error = missing
    raise RuntimeError(
        f"No MCD12Q1 land cover for {acquisition_date.year} or the "
        f"{LAND_COVER_MAX_FALLBACK_YEARS} preceding years."
    ) from error


def _cmr_download_url(entry: dict[str, Any]) -> str:
    links = entry.get("links", [])
    candidates = [link.get("href") for link in links if link.get("href") and not link.get("inherited")]
    for href in candidates:
        if str(href).endswith((".h5", ".hdf", ".HDF")):
            return str(href)
    if candidates:
        return str(candidates[0])
    raise RuntimeError(f"CMR granule has no downloadable link: {entry.get('producer_granule_id')}")


def _entry_name(entry: dict[str, Any]) -> str:
    return str(entry.get("producer_granule_id") or entry.get("title") or _filename_from_url(_cmr_download_url(entry)))


def _tile_from_name(name: str) -> str:
    match = re.search(r"\.h(\d{2})v(\d{2})\.", name)
    if not match:
        raise RuntimeError(f"Could not infer a VIIRS tile from granule name: {name}")
    return f"h{match.group(1)}v{match.group(2)}"


def _overlap_area(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    min_x, min_y = max(left[0], right[0]), max(left[1], right[1])
    max_x, max_y = min(left[2], right[2]), min(left[3], right[3])
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


def _entry_bbox(entry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    boxes = entry.get("boxes") or []
    if not boxes:
        return None
    values = [float(value) for value in str(boxes[0]).split()]
    if len(values) != 4:
        return None
    return values[0], values[1], values[2], values[3]


def select_viirs_tile(entries: Iterable[dict[str, Any]], landsat_bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    candidates = list(entries)
    if not candidates:
        raise RuntimeError("CMR returned no VNP09GA tiles intersecting the selected Landsat scene.")
    return max(
        candidates,
        key=lambda entry: (_overlap_area(_entry_bbox(entry), landsat_bbox) if _entry_bbox(entry) else 0.0, _entry_name(entry)),
    )


def select_named_tile(entries: Iterable[dict[str, Any]], tile: str, product: str) -> dict[str, Any]:
    for entry in entries:
        if f".{tile}." in _entry_name(entry):
            return entry
    raise RuntimeError(f"CMR returned no {product} granule for tile {tile}.")


def _viirs_entries_for_scene(
    acquisition_date: date, landsat_bbox: tuple[float, float, float, float]
) -> list[dict[str, Any]]:
    """Return one VNP09GA entry for each tile intersecting the Landsat scene."""
    entries = _cmr_entries_for_filename_date("VNP09GA", "002", acquisition_date, landsat_bbox)
    by_tile: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_bbox = _entry_bbox(entry)
        if entry_bbox is not None and _overlap_area(entry_bbox, landsat_bbox) <= 0.0:
            continue
        tile = _tile_from_name(_entry_name(entry))
        by_tile.setdefault(tile, entry)
    if not by_tile:
        raise RuntimeError(
            f"CMR returned no VNP09GA tiles intersecting the Landsat scene for {acquisition_date.isoformat()}."
        )
    return [by_tile[tile] for tile in sorted(by_tile)]


def _modis_entries_for_scene(
    acquisition_date: date, landsat_bbox: tuple[float, float, float, float]
) -> list[dict[str, Any]]:
    """Return one MOD09GA tile for each MODIS tile intersecting the Landsat scene."""
    entries = _cmr_entries_for_filename_date("MOD09GA", "061", acquisition_date, landsat_bbox)
    by_tile: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_bbox = _entry_bbox(entry)
        if entry_bbox is not None and _overlap_area(entry_bbox, landsat_bbox) <= 0.0:
            continue
        tile = _tile_from_name(_entry_name(entry))
        by_tile.setdefault(tile, entry)
    if not by_tile:
        raise RuntimeError(
            f"CMR returned no MOD09GA tiles intersecting the Landsat scene for {acquisition_date.isoformat()}."
        )
    return [by_tile[tile] for tile in sorted(by_tile)]


def _mosaic_viirs_nbar_tiles(inputs: Iterable[Path], output: Path) -> Path:
    """Mosaic adjacent native-grid VIIRS NBAR tiles into one Float32 GeoTIFF."""
    sources = [str(path) for path in inputs]
    if not sources:
        raise ValueError("At least one VIIRS NBAR tile is required for mosaicking.")
    output.parent.mkdir(parents=True, exist_ok=True)
    vrt_path = f"/vsimem/{output.stem}_mosaic.vrt"
    vrt = gdal.BuildVRT(
        vrt_path,
        sources,
        options=gdal.BuildVRTOptions(srcNodata=-9999.0, VRTNodata=-9999.0, resolution="highest"),
    )
    if vrt is None:
        raise RuntimeError("GDAL could not build the VIIRS NBAR tile mosaic.")
    translated = gdal.Translate(
        str(output),
        vrt,
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
        noData=-9999.0,
    )
    vrt = None
    gdal.Unlink(vrt_path)
    if translated is None:
        raise RuntimeError(f"GDAL could not write the VIIRS NBAR mosaic: {output}")
    translated.FlushCache()
    translated = None
    return output


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def _remove_intermediate_outputs(files: Iterable[Path], directories: Iterable[Path] = ()) -> int:
    """Remove generated intermediates only after a final output is available."""
    removed = 0
    for path in {Path(path) for path in files}:
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
        header = path.with_suffix(path.suffix + ".hdr")
        if header.is_file() or header.is_symlink():
            header.unlink()
            removed += 1
    for directory in {Path(path) for path in directories}:
        if directory.is_dir():
            removed += sum(1 for path in directory.rglob("*") if path.is_file() or path.is_symlink())
            shutil.rmtree(directory)
    return removed


def _should_keep_intermediate_outputs(
    keep_intermediate_outputs: bool, heldout_landsat_validation: bool
) -> bool:
    """Keep diagnostic inputs automatically when a held-out result is scored."""
    return keep_intermediate_outputs or heldout_landsat_validation


def _best_window(mask_path: Path, window_size: int, minimum_valid_fraction: float) -> tuple[int, int, int, int]:
    dataset = gdal.Open(str(mask_path))
    if dataset is None:
        raise ValueError(f"GDAL could not open mask: {mask_path}")
    mask = dataset.GetRasterBand(1).ReadAsArray().astype(bool)
    size = min(window_size, mask.shape[0], mask.shape[1])
    stride = max(1, size // 2)
    best = (-1, 0, 0)
    for y_offset in range(0, mask.shape[0] - size + 1, stride):
        for x_offset in range(0, mask.shape[1] - size + 1, stride):
            valid = int(mask[y_offset:y_offset + size, x_offset:x_offset + size].sum())
            if valid > best[0]:
                best = (valid, x_offset, y_offset)
    fraction = best[0] / float(size * size)
    if fraction < minimum_valid_fraction:
        raise RuntimeError(
            f"No {size}x{size} matching window reached the required valid fraction "
            f"({fraction:.2%} found; {minimum_valid_fraction:.2%} required)."
        )
    return best[1], best[2], size, size


def _open_single_band(path: Path) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    if dataset is None or dataset.RasterCount != 1:
        raise ValueError(f"GDAL could not open a single-band raster: {path}")
    return dataset


def _align_landsat_to_reference(source: Path, reference: Path, output: Path) -> Path:
    """Resample a second Landsat observation to the first pair's exact grid."""
    reference_dataset = _open_single_band(reference)
    geotransform = reference_dataset.GetGeoTransform()
    min_x, max_y = geotransform[0], geotransform[3]
    max_x = min_x + reference_dataset.RasterXSize * geotransform[1]
    min_y = max_y + reference_dataset.RasterYSize * geotransform[5]
    source_dataset = _open_single_band(source)
    nodata = source_dataset.GetRasterBand(1).GetNoDataValue()
    if output.exists():
        existing = _open_single_band(output)
        if (
            existing.RasterXSize == reference_dataset.RasterXSize
            and existing.RasterYSize == reference_dataset.RasterYSize
            and existing.GetGeoTransform() == reference_dataset.GetGeoTransform()
            and existing.GetProjection() == reference_dataset.GetProjection()
        ):
            return output
    output.parent.mkdir(parents=True, exist_ok=True)
    translated = gdal.Warp(
        str(output),
        source_dataset,
        format="GTiff",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=reference_dataset.RasterXSize,
        height=reference_dataset.RasterYSize,
        dstSRS=reference_dataset.GetProjection(),
        resampleAlg="near",
        srcNodata=nodata,
        dstNodata=nodata,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )
    if translated is None:
        raise RuntimeError(f"GDAL could not align {source} to the reference Landsat grid.")
    translated.FlushCache()
    translated = None
    return output


def _read_float_raster(path: Path) -> tuple[np.ndarray, gdal.Dataset]:
    dataset = _open_single_band(path)
    band = dataset.GetRasterBand(1)
    raw = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None and np.isfinite(nodata):
        raw[np.isclose(raw, nodata)] = np.nan
    raw[~np.isfinite(raw)] = np.nan
    return raw, dataset


def _read_boolean_mask(path: Path) -> tuple[np.ndarray, gdal.Dataset]:
    """Read an explicit 0/1 mask without interpreting zero as missing data."""
    dataset = _open_single_band(path)
    return dataset.GetRasterBand(1).ReadAsArray() != 0, dataset


def _shift_boolean_mask(values: np.ndarray, x_pixels: int, y_pixels: int) -> np.ndarray:
    """Apply the registration translation to a 0/1 mask, exposing invalid edges."""
    source = np.asarray(values, dtype=bool)
    shifted = np.zeros(source.shape, dtype=bool)
    height, width = source.shape
    out_y0, out_y1 = max(0, y_pixels), min(height, height + y_pixels)
    out_x0, out_x1 = max(0, x_pixels), min(width, width + x_pixels)
    if out_y0 < out_y1 and out_x0 < out_x1:
        shifted[out_y0:out_y1, out_x0:out_x1] = source[
            out_y0 - y_pixels : out_y1 - y_pixels,
            out_x0 - x_pixels : out_x1 - x_pixels,
        ]
    return shifted


def _starfm_int16(reflectance: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode physical reflectance as STARFM's signed-int16 scale-10000 input."""
    usable = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(reflectance)
        & (reflectance >= 0.0)
        & (reflectance <= 1.0)
    )
    encoded = np.full(reflectance.shape, STARFM_FILL_VALUE, dtype="<i2")
    encoded[usable] = np.rint(reflectance[usable] * STARFM_SCALE_FACTOR).astype("<i2")
    return encoded, usable


def _write_envi_header(path: Path, rows: int, columns: int) -> None:
    path.with_suffix(path.suffix + ".hdr").write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {columns}",
                f"lines = {rows}",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 2",
                "interleave = bsq",
                "byte order = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_starfm_input(path: Path, values: np.ndarray, rows: int, columns: int) -> None:
    if values.shape != (rows, columns):
        raise ValueError(f"STARFM input shape {values.shape} does not match expected {(rows, columns)}.")
    values.astype("<i2", copy=False).tofile(path)
    _write_envi_header(path, rows, columns)


def _write_starfm_mask(path: Path, values: np.ndarray, rows: int, columns: int) -> None:
    """Write STARFM's optional byte mask format (0 = invalid, 1 = valid)."""
    if values.shape != (rows, columns):
        raise ValueError(f"STARFM mask shape {values.shape} does not match expected {(rows, columns)}.")
    values.astype(np.uint8, copy=False).tofile(path)
    path.with_suffix(path.suffix + ".hdr").write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {columns}",
                f"lines = {rows}",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 1",
                "interleave = bsq",
                "byte order = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _starfm_cwd_relative(path: Path, output_directory: Path) -> Path:
    """Return a posix-relative path for STARFM when cwd is ``output_directory``."""
    return Path(path.resolve().relative_to(output_directory.resolve()).as_posix())


def _validate_starfm_output_filename(prediction_output: Path) -> None:
    """Reject output names that overflow StarFM's 200-byte ``"<name>.hdr"`` buffer."""
    header_name = f"{prediction_output.as_posix()}.hdr"
    if len(header_name.encode("utf-8")) >= STARFM_HEADER_FILENAME_BUF:
        raise ValueError(
            f"STARFM output name is too long for the original writeHeader buffer "
            f"({len(header_name)} >= {STARFM_HEADER_FILENAME_BUF} bytes including .hdr): "
            f"{prediction_output}"
        )


def _write_starfm_parameter_file(
    path: Path,
    *,
    pair_landsat: Sequence[Path],
    pair_landsat_mask: Sequence[Path],
    pair_viirs: Sequence[Path],
    pair_viirs_mask: Sequence[Path],
    prediction_viirs: Path,
    prediction_viirs_mask: Path,
    prediction_output: Path,
    rows: int,
    columns: int,
    resolution: float,
    max_search_distance: int,
    uncertainty: int,
    use_mask_files: bool,
) -> None:
    if not pair_landsat or len(pair_landsat) != len(pair_viirs):
        raise ValueError("STARFM needs an equal, non-empty number of Landsat and VIIRS pairs.")
    if use_mask_files and (
        len(pair_landsat_mask) != len(pair_landsat)
        or len(pair_viirs_mask) != len(pair_viirs)
    ):
        raise ValueError("STARFM needs one optional mask per input pair.")
    _validate_starfm_output_filename(Path(prediction_output))
    pair_viirs_s = " ".join(Path(p).as_posix() for p in pair_viirs)
    pair_landsat_s = " ".join(Path(p).as_posix() for p in pair_landsat)
    prediction_viirs_s = Path(prediction_viirs).as_posix()
    prediction_output_s = Path(prediction_output).as_posix()
    lines = [
        "STARFM_PARAMETER_START",
        f"NUM_IN_PAIRS = {len(pair_landsat)}",
        f"IN_PAIR_MODIS_FNAME = {pair_viirs_s}",
    ]
    if use_mask_files:
        pair_viirs_mask_s = " ".join(Path(p).as_posix() for p in pair_viirs_mask)
        lines.append(f"IN_PAIR_MODIS_MASK = {pair_viirs_mask_s}")
    lines.append(f"IN_PAIR_LANDSAT_FNAME = {pair_landsat_s}")
    if use_mask_files:
        pair_landsat_mask_s = " ".join(Path(p).as_posix() for p in pair_landsat_mask)
        lines.append(f"IN_PAIR_LANDSAT_MASK = {pair_landsat_mask_s}")
    lines.append(f"IN_PDAY_MODIS_FNAME = {prediction_viirs_s}")
    if use_mask_files:
        lines.append(f"IN_PDAY_MODIS_MASK = {Path(prediction_viirs_mask).as_posix()}")
    lines.extend(
        [
            f"OUT_PDAY_LANDSAT_FNAME = {prediction_output_s}",
            f"NROWS = {rows}",
            f"NCOLS = {columns}",
            f"RESOLUTION = {resolution:g}",
            f"SCALE_FACTOR = {STARFM_SCALE_FACTOR}",
            f"LANDSAT_FILLV = {STARFM_FILL_VALUE}",
            f"LANDSAT_DATA_RANGE = {STARFM_DATA_MIN}, {STARFM_DATA_MAX}",
            f"LANDSAT_UNCERTAINTY = {uncertainty}",
            f"MODIS_FILLV = {STARFM_FILL_VALUE}",
            f"MODIS_DATA_RANGE = {STARFM_DATA_MIN}, {STARFM_DATA_MAX}",
            f"MODIS_UNCERTAINTY = {uncertainty}",
            "USE_SPATIAL_FLAG = 1",
            f"MAX_SEARCH_DISTANCE = {max_search_distance}",
            "NUM_SLICE_PURE_TEST = 40",
            "PREDICTION_METHOD = 1",
            "USE_LUT_IMPROVEMENT = 0",
            "STARFM_PARAMETER_END",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_prediction_geotiff(path: Path, encoded: np.ndarray, reference: gdal.Dataset) -> None:
    """Write prediction as raw Landsat-style stored DN (display scale 1, offset 0)."""
    reference_band = reference.GetRasterBand(1)
    output_type = reference_band.DataType
    if output_type != gdal.GDT_UInt16:
        output_type = gdal.GDT_UInt16
    output_nodata = reference_band.GetNoDataValue()
    if output_nodata is None:
        output_nodata = 0
    output = gdal.GetDriverByName("GTiff").Create(
        str(path),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        output_type,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"GDAL could not create STARFM GeoTIFF: {path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(output_nodata)
    # Landsat's delivered SR GeoTIFF displays stored DN values directly.  The
    # physical C2 conversion remains in this pipeline, but is deliberately
    # not attached as output metadata: viewers should use scale 1 / offset 0.
    band.SetScale(1.0)
    band.SetOffset(0.0)
    valid = encoded != STARFM_FILL_VALUE
    reflectance = encoded.astype(np.float64) / STARFM_SCALE_FACTOR
    stored = np.full(encoded.shape, output_nodata, dtype=np.uint16)
    converted = np.rint(
        (reflectance[valid] - LANDSAT_C2_SR_OFFSET) / LANDSAT_C2_SR_SCALE_FACTOR
    )
    stored[valid] = np.clip(converted, 1, np.iinfo(np.uint16).max).astype(np.uint16)
    band.WriteArray(stored)
    band.FlushCache()
    output.FlushCache()


def _write_validation_mask(path: Path, values: np.ndarray, reference: gdal.Dataset) -> None:
    """Write a 0/1 held-out comparison mask on the prediction grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(path), reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"GDAL could not create validation mask: {path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    band = output.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(np.asarray(values, dtype=np.uint8))
    band.FlushCache()
    output.FlushCache()


def _write_difference_geotiff(path: Path, values: np.ndarray, reference: gdal.Dataset) -> None:
    """Write predicted-minus-actual physical reflectance, retaining invalid cells as nodata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0
    output = gdal.GetDriverByName("GTiff").Create(
        str(path), reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if output is None:
        raise RuntimeError(f"GDAL could not create validation difference raster: {path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    stored = np.where(np.isfinite(values), values, nodata).astype(np.float32)
    band = output.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(stored)
    band.FlushCache()
    output.FlushCache()


def _coregistration_quality_metadata(
    method: str,
    report: dict[str, Any],
) -> tuple[str, float | None]:
    """Return the method-specific co-registration percentage for validation."""
    if method == "moving-window":
        value = report.get("best_correlation")
        metric = "best_pearson_correlation_percent"
        value = float(value) * 100.0 if value is not None else None
    elif method == "arosics":
        value = report.get("shift_reliability_percent")
        metric = "arosics_shift_reliability_percent"
        value = float(value) if value is not None else None
    elif method == "none":
        value = None
        metric = "not_applicable_no_coregistration"
    else:
        raise ValueError(f"Unsupported coregistration method: {method}")
    return metric, value if value is not None and np.isfinite(value) else None


def _reflectance_error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Score already-masked physical reflectances; R² is 1-SSE/SST, not correlation².

    Undefined R² (fewer than two observations or constant reference) is None,
    serialized as JSON null / an empty CSV cell. Negative R² is retained.
    """
    actual = np.asarray(actual, dtype=np.float64).ravel()
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Reflectance metrics require nonempty, equally sized arrays.")
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(predicted))):
        raise ValueError("Reflectance metrics require finite, already-masked values.")
    errors = predicted - actual
    squared_error = float(np.sum(errors ** 2))
    total_variation = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = (
        float(1.0 - squared_error / total_variation)
        if actual.size >= 2 and np.any(actual != actual[0]) and total_variation > 0.0
        else None
    )
    return {
        "rmse_surface_reflectance": float(np.sqrt(squared_error / actual.size)),
        "mae_surface_reflectance": float(np.mean(np.abs(errors))),
        "r2_surface_reflectance": r2,
    }


def _evaluate_heldout_landsat(
    *,
    predicted_landsat: Path,
    heldout_landsat: Path,
    heldout_landsat_qa: Path,
    output_directory: Path,
    band_name: str,
    coregistration_method: str | None = None,
    coregistration_quality_metric: str | None = None,
    coregistration_quality_percent: float | None = None,
    coregistration_report_path: Path | None = None,
) -> dict[str, Any]:
    """Compare a STARFM prediction with a target-date Landsat scene.

    The target Landsat image is first aligned to the prediction grid. Statistics
    use only finite, physically plausible pixels that are clear in target-date
    QA_PIXEL.  Thus the target image is a reference only; it never enters the
    STARFM input pairs or its registration step.
    """
    from build_coregistration_mask import landsat_clear_mask

    output_directory.mkdir(parents=True, exist_ok=True)
    aligned_landsat = output_directory / f"heldout_actual_{band_name}_on_prediction_grid.tif"
    aligned_qa = output_directory / "heldout_QA_PIXEL_on_prediction_grid.tif"
    _align_landsat_to_reference(heldout_landsat, predicted_landsat, aligned_landsat)
    _align_landsat_to_reference(heldout_landsat_qa, predicted_landsat, aligned_qa)

    predicted_raw, predicted_dataset = _read_float_raster(predicted_landsat)
    actual_raw, actual_dataset = _read_float_raster(aligned_landsat)
    qa_dataset = _open_single_band(aligned_qa)
    if actual_raw.shape != predicted_raw.shape or qa_dataset.RasterYSize != predicted_raw.shape[0] or qa_dataset.RasterXSize != predicted_raw.shape[1]:
        raise ValueError("Held-out Landsat, its QA raster, and the prediction must share one grid.")

    actual_reflectance = actual_raw * LANDSAT_C2_SR_SCALE_FACTOR + LANDSAT_C2_SR_OFFSET
    predicted_reflectance = predicted_raw * LANDSAT_C2_SR_SCALE_FACTOR + LANDSAT_C2_SR_OFFSET
    actual_clear = landsat_clear_mask(qa_dataset.GetRasterBand(1).ReadAsArray())
    valid = (
        actual_clear
        & np.isfinite(actual_reflectance)
        & np.isfinite(predicted_reflectance)
        & (actual_reflectance >= 0.0)
        & (actual_reflectance <= 1.0)
        & (predicted_reflectance >= 0.0)
        & (predicted_reflectance <= 1.0)
    )
    if not np.any(valid):
        raise RuntimeError("Held-out validation has no clear, valid pixels shared by prediction and target Landsat.")

    difference = np.full(predicted_reflectance.shape, np.nan, dtype=np.float64)
    difference[valid] = predicted_reflectance[valid] - actual_reflectance[valid]
    actual_values = actual_reflectance[valid]
    predicted_values = predicted_reflectance[valid]
    difference_values = difference[valid]
    actual_mean = float(np.mean(actual_values))
    predicted_mean = float(np.mean(predicted_values))
    mean_difference = float(np.mean(difference_values))
    relative_error_percent = (
        float(100.0 * mean_difference / actual_mean) if not np.isclose(actual_mean, 0.0) else None
    )
    error_metrics = _reflectance_error_metrics(actual_values, predicted_values)

    comparison_mask = output_directory / f"heldout_comparison_mask_{band_name}.tif"
    difference_output = output_directory / f"predicted_minus_actual_{band_name}.tif"
    _write_validation_mask(comparison_mask, valid, predicted_dataset)
    _write_difference_geotiff(difference_output, difference, predicted_dataset)
    metrics = {
        "band": band_name,
        "coregistration_method": coregistration_method,
        "coregistration_quality_metric": coregistration_quality_metric,
        "coregistration_quality_percent": coregistration_quality_percent,
        "coregistration_report": (
            str(coregistration_report_path.absolute())
            if coregistration_report_path is not None
            else None
        ),
        "actual_mean_surface_reflectance": actual_mean,
        "predicted_mean_surface_reflectance": predicted_mean,
        "mean_difference_predicted_minus_actual": mean_difference,
        "relative_error_percent": relative_error_percent,
        **error_metrics,
        "valid_comparison_pixels": int(np.count_nonzero(valid)),
        "total_prediction_grid_pixels": int(valid.size),
        "valid_comparison_percent": float(100.0 * np.mean(valid)),
        "comparison_mask": str(comparison_mask.absolute()),
        "actual_landsat_on_prediction_grid": str(aligned_landsat.absolute()),
        "actual_landsat_qa_on_prediction_grid": str(aligned_qa.absolute()),
        "predicted_minus_actual": str(difference_output.absolute()),
    }
    csv_path = output_directory / "heldout_validation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    json_path = output_directory / "heldout_validation_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_csv"] = str(csv_path.absolute())
    metrics["metrics_json"] = str(json_path.absolute())
    return metrics


def _run_original_starfm(
    *,
    executable: Path,
    pairs: Sequence[StarfmPair],
    prediction_viirs: Path,
    prediction_mask: Path | None,
    shift_x_pixels: int,
    shift_y_pixels: int,
    output_directory: Path,
    prediction_date: date,
    band_name: str,
    max_search_distance: int,
    uncertainty: int,
    use_mask_files: bool,
) -> dict[str, Any]:
    """Prepare common-grid, multi-pair original-STARFM inputs and run STARFM."""
    if not executable.is_file():
        raise RuntimeError(f"Original STARFM executable does not exist: {executable}")
    if not os.access(executable, os.X_OK):
        raise RuntimeError(f"Original STARFM executable is not executable: {executable}")
    if not pairs:
        raise ValueError("STARFM needs at least one input pair.")

    from build_coregistration_mask import landsat_clear_mask

    landsat_dataset = _open_single_band(pairs[0].landsat)
    rows, columns = landsat_dataset.RasterYSize, landsat_dataset.RasterXSize
    prediction_reflectance, prediction_dataset = _read_float_raster(prediction_viirs)
    if prediction_reflectance.shape != (rows, columns) or prediction_dataset.GetGeoTransform() != landsat_dataset.GetGeoTransform():
        raise ValueError("Prediction VIIRS input must use the exact common Landsat grid.")
    if prediction_mask is None:
        prediction_valid = np.isfinite(prediction_reflectance)
    else:
        prediction_valid_unshifted, prediction_mask_dataset = _read_boolean_mask(prediction_mask)
        if prediction_valid_unshifted.shape != (rows, columns) or prediction_mask_dataset.GetGeoTransform() != landsat_dataset.GetGeoTransform():
            raise ValueError("Prediction validity mask must use the exact common Landsat grid.")
        prediction_valid = _shift_boolean_mask(prediction_valid_unshifted, shift_x_pixels, shift_y_pixels)
    prediction_values, prediction_valid = _starfm_int16(prediction_reflectance, prediction_valid)

    output_directory.mkdir(parents=True, exist_ok=True)
    input_directory = output_directory / "INPUTS"
    input_directory.mkdir(exist_ok=True)
    prediction_viirs_path = input_directory / f"viirs_{prediction_date:%Y%m%d}_{band_name}.bin"
    prediction_viirs_mask_path = input_directory / f"viirs_{prediction_date:%Y%m%d}_{band_name}.mask"
    output_binary = output_directory / f"predicted_{band_name}_{prediction_date:%Y%m%d}.bin"
    parameter_file = output_directory / "starfm_parameters.txt"
    stdout_log = output_directory / "starfm.stdout.log"
    stderr_log = output_directory / "starfm.stderr.log"
    output_tif = output_directory / f"predicted_{band_name}_{prediction_date:%Y%m%d}.tif"

    pair_landsat_paths: list[Path] = []
    pair_viirs_paths: list[Path] = []
    pair_landsat_mask_paths: list[Path] = []
    pair_viirs_mask_paths: list[Path] = []
    pair_valid_pixel_counts: dict[str, int] = {}
    for pair in pairs:
        pair_landsat_dataset = _open_single_band(pair.landsat)
        pair_reflectance, pair_dataset = _read_float_raster(pair.viirs)
        if (
            pair_reflectance.shape != (rows, columns)
            or pair_landsat_dataset.GetGeoTransform() != landsat_dataset.GetGeoTransform()
            or pair_dataset.GetGeoTransform() != landsat_dataset.GetGeoTransform()
        ):
            raise ValueError(f"Pair {pair.acquisition_date:%Y-%m-%d} is not on the common STARFM grid.")
        raw = pair_landsat_dataset.GetRasterBand(1).ReadAsArray().astype(np.float64)
        nodata = pair_landsat_dataset.GetRasterBand(1).GetNoDataValue()
        if nodata is not None:
            raw[np.isclose(raw, nodata)] = np.nan
        if pair.valid_mask is None:
            pair_valid = np.isfinite(raw) & np.isfinite(pair_reflectance)
        else:
            pair_valid_unshifted, pair_mask_dataset = _read_boolean_mask(pair.valid_mask)
            if (
                pair_valid_unshifted.shape != (rows, columns)
                or pair_mask_dataset.GetGeoTransform() != landsat_dataset.GetGeoTransform()
            ):
                raise ValueError(f"Pair validity mask does not match {pair.acquisition_date:%Y-%m-%d}.")
            qa_dataset = _open_single_band(pair.landsat_qa)
            landsat_valid = landsat_clear_mask(qa_dataset.GetRasterBand(1).ReadAsArray())
            if landsat_valid.shape != (rows, columns):
                raise ValueError(f"Landsat QA does not match pair {pair.acquisition_date:%Y-%m-%d}.")
            pair_valid = landsat_valid & _shift_boolean_mask(pair_valid_unshifted, shift_x_pixels, shift_y_pixels)
        fine_values, pair_valid = _starfm_int16(
            raw * LANDSAT_C2_SR_SCALE_FACTOR + LANDSAT_C2_SR_OFFSET, pair_valid
        )
        pair_values, pair_valid = _starfm_int16(pair_reflectance, pair_valid)
        fine_values, pair_valid = _starfm_int16(
            raw * LANDSAT_C2_SR_SCALE_FACTOR + LANDSAT_C2_SR_OFFSET, pair_valid
        )
        landsat_path = input_directory / f"landsat_{pair.acquisition_date:%Y%m%d}_{band_name}.bin"
        viirs_path = input_directory / f"viirs_{pair.acquisition_date:%Y%m%d}_{band_name}.bin"
        landsat_mask_path = input_directory / f"landsat_{pair.acquisition_date:%Y%m%d}_{band_name}.mask"
        viirs_mask_path = input_directory / f"viirs_{pair.acquisition_date:%Y%m%d}_{band_name}.mask"
        _write_starfm_input(landsat_path, fine_values, rows, columns)
        _write_starfm_input(viirs_path, pair_values, rows, columns)
        if use_mask_files:
            _write_starfm_mask(landsat_mask_path, pair_valid, rows, columns)
            _write_starfm_mask(viirs_mask_path, pair_valid, rows, columns)
        pair_landsat_paths.append(landsat_path)
        pair_viirs_paths.append(viirs_path)
        pair_landsat_mask_paths.append(landsat_mask_path)
        pair_viirs_mask_paths.append(viirs_mask_path)
        pair_valid_pixel_counts[pair.acquisition_date.isoformat()] = int(np.count_nonzero(pair_valid))

    _write_starfm_input(prediction_viirs_path, prediction_values, rows, columns)
    # The supplied macOS executable crashes when any optional external mask
    # parameter is present.  By default, invalid observations are represented
    # directly in the three reflectance inputs with the configured fill value.
    if use_mask_files:
        _write_starfm_mask(prediction_viirs_mask_path, prediction_valid, rows, columns)
    # STARFM runs with cwd=output_directory. Use short cwd-relative names so the
    # original writeHeader() 200-byte "<out>.hdr" buffer cannot overflow on long
    # Swan output trees (crash previously appeared later in cleanup()).
    _write_starfm_parameter_file(
        parameter_file,
        pair_landsat=[_starfm_cwd_relative(path, output_directory) for path in pair_landsat_paths],
        pair_landsat_mask=[
            _starfm_cwd_relative(path, output_directory) for path in pair_landsat_mask_paths
        ],
        pair_viirs=[_starfm_cwd_relative(path, output_directory) for path in pair_viirs_paths],
        pair_viirs_mask=[
            _starfm_cwd_relative(path, output_directory) for path in pair_viirs_mask_paths
        ],
        prediction_viirs=_starfm_cwd_relative(prediction_viirs_path, output_directory),
        prediction_viirs_mask=_starfm_cwd_relative(prediction_viirs_mask_path, output_directory),
        prediction_output=Path(output_binary.name),
        rows=rows,
        columns=columns,
        resolution=abs(landsat_dataset.GetGeoTransform()[1]),
        max_search_distance=max_search_distance,
        uncertainty=uncertainty,
        use_mask_files=use_mask_files,
    )
    completed = subprocess.run(
        [str(executable.resolve()), parameter_file.name],
        cwd=output_directory,
        capture_output=True,
        text=True,
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Original STARFM failed with exit code {completed.returncode}. "
            f"See {stdout_log} and {stderr_log}."
        )
    expected_size = rows * columns * np.dtype("<i2").itemsize
    if not output_binary.is_file() or output_binary.stat().st_size != expected_size:
        actual_size = output_binary.stat().st_size if output_binary.is_file() else 0
        raise RuntimeError(
            f"Original STARFM did not create the expected {expected_size}-byte output; got {actual_size} bytes."
        )
    predicted_values = np.fromfile(output_binary, dtype="<i2").reshape(rows, columns)
    _write_prediction_geotiff(output_tif, predicted_values, landsat_dataset)
    return {
        "run_directory": output_directory,
        "parameter_file": parameter_file,
        "pair_landsat_binaries": [str(path.absolute()) for path in pair_landsat_paths],
        "pair_viirs_binaries": [str(path.absolute()) for path in pair_viirs_paths],
        "prediction_viirs_binary": prediction_viirs_path,
        "output_binary": output_binary,
        "output_geotiff": output_tif,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "pair_valid_pixels": pair_valid_pixel_counts,
        "prediction_valid_pixels": int(np.count_nonzero(prediction_valid)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and preprocess one or two Landsat--VIIRS pairs.")
    parser.add_argument("--prediction-date", type=_parse_date, required=True, help="VIIRS prediction date, YYYY-MM-DD.")
    parser.add_argument("--path", type=int, required=True, help="WRS-2 path.")
    parser.add_argument("--row", type=int, required=True, help="WRS-2 row.")
    bands = parser.add_mutually_exclusive_group(required=True)
    bands.add_argument("--band", type=_band, help="Requested Landsat reflective band, e.g. B4.")
    bands.add_argument("--all-bands", action="store_true", help="Run all supported bands: VIIRS B1/B2/B3/B4/B5/B7; MODIS B2/B3/B4/B5/B7.")
    parser.add_argument(
        "--coarse-sensor",
        choices=("VIIRS", "MODIS"),
        type=str.upper,
        default="VIIRS",
        help="Daily coarse sensor to use: VIIRS (default) or MODIS Terra.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("COREGISTRATION_FILES"), help="Standard data root for downloaded inputs, outputs, and manifests.")
    parser.add_argument("--maximum-days", type=int, default=32, help="Search this many days before/after the prediction date; default: 32.")
    parser.add_argument("--max-cloud-percent", type=float, default=10.0, help="Maximum scene-level Landsat cloud cover eligible for nearest-date selection; default: 10.")
    parser.add_argument(
        "--pair-count",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Number of Landsat/VIIRS input pairs: 1 selects the nearest clear scene "
            "other than the prediction date; 2 selects one clear scene before and one after. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--pair-dates",
        type=_parse_date,
        nargs="+",
        metavar="YYYY-MM-DD",
        help=(
            "Explicit one or two Landsat pair dates. Overrides --pair-count and automatic "
            "nearest/bracketing selection; each date must meet --max-cloud-percent and must "
            "differ from --prediction-date."
        ),
    )
    parser.add_argument(
        "--test-id",
        type=_test_id,
        help=(
            "Optional unique experiment label. When set, run-specific outputs, validation, "
            "manifest, and a one-row CSV/JSON analysis summary are stored under "
            "OUTPUTS/TESTS/<test-id>/."
        ),
    )
    parser.add_argument(
        "--validate-held-out-landsat",
        action="store_true",
        help=(
            "Download the eligible Landsat scene on --prediction-date only as a hidden "
            "reference, then write paper-style accuracy metrics after STARFM finishes. "
            "The target-date scene is never used as a STARFM pair."
        ),
    )
    parser.add_argument("--earthdata-token-env", default="EARTHDATA_TOKEN", help="Environment-variable name containing an Earthdata bearer token.")
    parser.add_argument("--usgs-api-key-env", default="USGS_M2M_API_KEY", help="Environment-variable name containing an already active USGS M2M API key.")
    parser.add_argument("--usgs-username-env", default="USGS_USERNAME", help="USGS EROS username environment-variable name, used with an M2M application token.")
    parser.add_argument("--usgs-application-token-env", default="USGS_M2M_TOKEN", help="Environment-variable name containing a USGS EROS M2M application token.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Ignored local YAML file holding credentials; default: config.yaml. Environment variables override it.",
    )
    parser.add_argument("--window-size", type=int, default=1024, help="Moving-window size on the Landsat grid; default: 1024.")
    parser.add_argument(
        "--coregistration-method",
        choices=("moving-window", "arosics", "none"),
        default="moving-window",
        help=(
            "Coarse-sensor-to-Landsat registration method: moving-window uses the "
            "paper-style exhaustive Pearson-correlation search (default); arosics "
            "uses AROSICS global COREG; none matches the Landsat grid without "
            "estimating or applying a spatial shift."
        ),
    )
    parser.add_argument(
        "--coregistration-band",
        type=_band,
        default=None,
        help=(
            "AROSICS matching band (default: B5). Ignored for moving-window and none; "
            "those modes process each prediction band using that same band."
        ),
    )
    parser.add_argument("--minimum-window-valid-fraction", type=float, default=0.30, help="Minimum valid fraction for automatic matching-window selection; default: 0.30.")
    parser.add_argument(
        "--use-quality-mask",
        action="store_true",
        help=(
            "Enable conservative Landsat and VIIRS cloud/quality masking for coregistration "
            "and STARFM. Disabled by default to match the older myStarfm workflow."
        ),
    )
    parser.add_argument(
        "--starfm-executable",
        type=Path,
        default=Path("StarFM_original/source/StarFM.exe"),
        help="Path to the original STARFM executable (build with: make -C StarFM_original/source).",
    )
    parser.add_argument(
        "--starfm-max-search-distance",
        type=int,
        default=750,
        help="Original STARFM spatial search distance in metres; default: 750.",
    )
    parser.add_argument(
        "--starfm-uncertainty",
        type=int,
        default=50,
        help="Original STARFM uncertainty in reflectance×10000 units; default: 50.",
    )
    parser.add_argument(
        "--starfm-use-mask-files",
        action="store_true",
        help=(
            "Pass optional external mask files to STARFM. Disabled by default because "
            "the supplied macOS executable crashes when they are used."
        ),
    )
    parser.add_argument(
        "--keep-intermediate-outputs",
        action="store_true",
        help=(
            "Keep generated NBAR, co-registration, temporary STARFM inputs, and logs. "
            "By default they are removed after a successful final GeoTIFF is written, except "
            "when --validate-held-out-landsat is enabled."
        ),
    )
    parser.add_argument(
        "--discard-intermediate-outputs-after-validation",
        action="store_true",
        help=(
            "After held-out validation, remove generated NBAR, co-registration, and STARFM "
            "input/log intermediates while retaining the final prediction, validation products, "
            "manifest, and test summary. Useful for parameter sweeps."
        ),
    )
    parser.add_argument(
        "--reuse-existing-nbar",
        action="store_true",
        help=(
            "Reuse a complete existing daily coarse-sensor NBAR mosaic instead of rebuilding it. "
            "Useful when repeated tests differ only in STARFM or coregistration settings."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.all_bands:
        selected_bands = [
            band for band, info in BAND_INFO.items()
            if args.coarse_sensor == "VIIRS" or info.modis_band is not None
        ]
        runs = []
        for band in selected_bands:
            run = argparse.Namespace(**vars(args))
            run.band = band
            run.all_bands = False
            run.coregistration_band = (
                band if run.coregistration_method != "arosics"
                else (args.coregistration_band or "B5")
            )
            if args.test_id:
                run.test_id = _test_id(f"{args.test_id}_{band.lower()}")
                destination = args.output_dir / "OUTPUTS" / "TESTS" / run.test_id
                if destination.exists():
                    raise RuntimeError(f"Refusing to overwrite existing test results: {destination}")
            runs.append(run)
        for index, run in enumerate(runs, start=1):
            print(f"Running band {run.band} ({index}/{len(runs)}) with {args.coarse_sensor}", flush=True)
            _run_single_band(run)
        return
    args.coregistration_band = (
        args.band if args.coregistration_method != "arosics"
        else (args.coregistration_band or "B5")
    )
    _run_single_band(args)


def _run_single_band(args: argparse.Namespace) -> None:
    if not (1 <= args.path <= 251 and 1 <= args.row <= 248):
        raise ValueError("WRS-2 path must be 1--251 and row must be 1--248.")
    if args.maximum_days < 0 or not (0.0 <= args.max_cloud_percent <= 100.0):
        raise ValueError("--maximum-days must be non-negative and --max-cloud-percent must be 0--100.")
    if not (0.0 < args.minimum_window_valid_fraction <= 1.0):
        raise ValueError("--minimum-window-valid-fraction must be in (0, 1].")
    if args.starfm_max_search_distance <= 0 or args.starfm_uncertainty < 0:
        raise ValueError("--starfm-max-search-distance must be positive and --starfm-uncertainty non-negative.")
    credentials = _read_credentials_config(args.config)
    token = _credential(credentials, "earthdata_token", args.earthdata_token_env)
    if not token:
        raise RuntimeError(
            "Set earthdata_token in config.yaml (or "
            f"{args.earthdata_token_env} in the environment) before downloading NASA inputs."
        )

    requested = BAND_INFO[args.band]
    coregistration_band = BAND_INFO[args.coregistration_band]
    if args.coarse_sensor == "MODIS" and requested.modis_band is None:
        raise ValueError(
            "MODIS mode does not support Landsat B1: MOD09GA has no coastal-aerosol counterpart. "
            "Choose B2, B3, B4, B5, or B7."
        )
    if args.coarse_sensor == "MODIS" and coregistration_band.modis_band is None:
        raise ValueError(
            "MODIS mode does not support Landsat B1 for --coregistration-band: "
            "MOD09GA has no coastal-aerosol counterpart. Choose B2, B3, B4, B5, or B7."
        )
    if args.coarse_sensor == "MODIS" and args.use_quality_mask:
        raise ValueError(
            "--use-quality-mask is currently implemented for VIIRS only. "
            "MODIS mode runs with the default quality-control setting disabled."
        )
    if args.coregistration_method == "arosics" and args.use_quality_mask:
        raise ValueError(
            "--use-quality-mask is not yet supported with --coregistration-method arosics. "
            "Disable it (the default) or use --coregistration-method moving-window."
        )
    data_root = args.output_dir
    test_results_root = (
        data_root / "OUTPUTS" / "TESTS" / args.test_id
        if args.test_id is not None
        else None
    )
    if test_results_root is not None and test_results_root.exists():
        raise RuntimeError(
            f"Test output directory already exists: {test_results_root}. "
            "Choose a new --test-id to prevent results from being overwritten."
        )
    run_outputs_root = test_results_root or (data_root / "OUTPUTS")

    explicit_pair_dates = _normalize_explicit_pair_dates(args.pair_dates, args.prediction_date)
    if explicit_pair_dates is not None:
        landsat_items = tuple(
            find_landsat_on_date(pair_date, args.path, args.row, args.max_cloud_percent)
            for pair_date in explicit_pair_dates
        )
        pair_count = len(explicit_pair_dates)
        selection_method = "explicit_pair_dates"
    elif args.pair_count == 1:
        landsat_items = (
            find_nearest_landsat(
                args.prediction_date,
                args.path,
                args.row,
                args.maximum_days,
                args.max_cloud_percent,
                # A one-pair run is also valid for held-out Landsat validation:
                # never let the target-date scene become STARFM input.
                exclude_target_date=True,
            ),
        )
        pair_count = 1
        selection_method = "nearest_clear_landsat_excluding_prediction_date"
    else:
        before_item, after_item = find_bracketing_landsat(
            args.prediction_date, args.path, args.row, args.maximum_days, args.max_cloud_percent
        )
        landsat_items = (before_item, after_item)
        pair_count = 2
        selection_method = "nearest_clear_landsat_before_and_after_prediction"
    pair_dates = tuple(_item_datetime(item).date() for item in landsat_items)
    reference_date = pair_dates[0]
    landsat_bbox = tuple(float(value) for value in landsat_items[0]["bbox"])
    if len(landsat_bbox) != 4:
        raise RuntimeError("Selected reference Landsat STAC item lacks a usable bounding box.")
    path_row_id = f"P{args.path:03d}_R{args.row:03d}"
    landsat_inputs = data_root / "INPUTS" / "LANDSAT" / path_row_id
    # B4 is also retained for the optional existing VIIRS quality-mask workflow.
    needed_landsat_assets = sorted({args.band, args.coregistration_band, "B4", "QA_PIXEL"})
    m2m_token = _usgs_m2m_token(
        _credential(credentials, "usgs_m2m_api_key", args.usgs_api_key_env),
        _credential(credentials, "usgs_username", args.usgs_username_env),
        _credential(credentials, "usgs_m2m_token", args.usgs_application_token_env),
        lock_directory=landsat_inputs.parent,
    )
    downloaded_landsat: list[dict[str, Path]] = []
    for item, pair_date in zip(landsat_items, pair_dates):
        item_bbox = tuple(float(value) for value in item["bbox"])
        downloaded_landsat.append(
            _download_landsat_assets(
                str(item["id"]), needed_landsat_assets, landsat_inputs, m2m_token, item_bbox, pair_date
            )
        )

    heldout_item: dict[str, Any] | None = None
    heldout_landsat_files: dict[str, Path] | None = None
    if args.validate_held_out_landsat:
        heldout_item = find_landsat_on_date(
            args.prediction_date, args.path, args.row, args.max_cloud_percent
        )
        if any(str(heldout_item["id"]) == str(item["id"]) for item in landsat_items):
            raise RuntimeError(
                "Held-out Landsat validation selected a target-date scene as a STARFM pair. "
                "Choose a prediction date different from every input-pair date."
            )
        heldout_landsat_files = _download_landsat_assets(
            str(heldout_item["id"]),
            sorted({args.band, "QA_PIXEL"}),
            landsat_inputs,
            m2m_token,
            tuple(float(value) for value in heldout_item["bbox"]),
            args.prediction_date,
        )

    # STARFM needs every fine/coarse input on exactly the same grid. Keep the
    # first selected Landsat scene as that grid and align any further scenes
    # with nearest-neighbour resampling so QA bits stay categorical.
    aligned_outputs = run_outputs_root / "ALIGNED_LANDSAT" / args.coarse_sensor / path_row_id
    pair_landsat_files: list[dict[str, Path]] = [downloaded_landsat[0]]
    intermediate_files: set[Path] = set()
    for downloaded_scene in downloaded_landsat[1:]:
        aligned_scene: dict[str, Path] = {}
        for asset, source in downloaded_scene.items():
            aligned_scene[asset] = _align_landsat_to_reference(
                source,
                downloaded_landsat[0][asset],
                aligned_outputs / f"{source.stem}_on_{reference_date:%Y%m%d}_grid.TIF",
            )
        pair_landsat_files.append(aligned_scene)
        intermediate_files.update(aligned_scene.values())

    coarse_inputs = data_root / "INPUTS" / args.coarse_sensor / path_row_id
    # Non-reused intermediates are private to the named run. Reused products
    # below are immutable, versioned cache entries and never cleanup targets.
    nbar_outputs = run_outputs_root / "NBAR" / args.coarse_sensor / path_row_id
    coregistration_method_directory = {
        "moving-window": "MOVING_WINDOW",
        "arosics": "AROSICS",
        "none": "NO_COREGISTRATION",
    }[args.coregistration_method]
    coregistration_band_directory = args.coregistration_band
    coregistration_outputs = (
        run_outputs_root
        / "COREGISTRATION"
        / args.coarse_sensor
        / coregistration_method_directory
        / coregistration_band_directory
        / path_row_id
    )
    mask_outputs = run_outputs_root / "COREGISTRATION" / args.coarse_sensor / "MASKS" / path_row_id
    viirs_product_ids = {
        "VNP43D40",
        *coregistration_band.parameter_products,
        *requested.parameter_products,
    }

    def download_daily_coarse_inputs(
        acquisition_date: date,
    ) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
        """Download a daily coarse image, BRDF inputs, and land cover for every tile."""
        daily_by_tile: dict[str, Path] = {}
        entries = (
            _viirs_entries_for_scene(acquisition_date, landsat_bbox)
            if args.coarse_sensor == "VIIRS"
            else _modis_entries_for_scene(acquisition_date, landsat_bbox)
        )
        for entry in entries:
            tile_name = _tile_from_name(_entry_name(entry))
            url = _cmr_download_url(entry)
            daily_by_tile[tile_name] = _download(
                url, coarse_inputs / _filename_from_url(url), token
            )

        brdf_files: dict[str, Path] = {}
        if args.coarse_sensor == "VIIRS":
            for product_id in sorted(viirs_product_ids):
                entry = _cmr_entry_for_filename_date(product_id, "002", acquisition_date)
                url = _cmr_download_url(entry)
                brdf_files[product_id] = _download(
                    url, coarse_inputs / _filename_from_url(url), token
                )
        else:
            for product_id in ("MCD43A1", "MCD43A2"):
                candidates = _cmr_entries_for_filename_date(
                    product_id, "061", acquisition_date, landsat_bbox
                )
                for tile_name in daily_by_tile:
                    entry = select_named_tile(candidates, tile_name, product_id)
                    url = _cmr_download_url(entry)
                    brdf_files[f"{product_id}:{tile_name}"] = _download(
                        url, coarse_inputs / _filename_from_url(url), token
                    )

        land_cover_entries, land_cover_year = _land_cover_entries(acquisition_date, landsat_bbox)
        land_cover_years[acquisition_date] = land_cover_year
        if land_cover_year != acquisition_date.year:
            print(f"MCD12Q1 {acquisition_date.year} is unavailable; using {land_cover_year} land cover for {acquisition_date}.")
        land_cover_by_tile: dict[str, Path] = {}
        for tile_name in daily_by_tile:
            entry = select_named_tile(land_cover_entries, tile_name, "MCD12Q1")
            url = _cmr_download_url(entry)
            land_cover_by_tile[tile_name] = _download(
                url, coarse_inputs / _filename_from_url(url), token
            )
        return daily_by_tile, brdf_files, land_cover_by_tile

    land_cover_years: dict[date, int] = {}
    daily_inputs = {
        acquisition_date: download_daily_coarse_inputs(acquisition_date)
        for acquisition_date in (*pair_dates, args.prediction_date)
    }
    prediction_daily, prediction_product_files, prediction_land_cover = daily_inputs[args.prediction_date]
    daily_product_name = "VNP09GA" if args.coarse_sensor == "VIIRS" else "MOD09GA"
    prediction_id = f"{path_row_id}_{args.prediction_date:%Y%m%d}_{daily_product_name}_mosaic"

    scripts = Path(__file__).resolve().parent

    def nbar_command(
        info: BandInfo,
        output: Path,
        daily_coarse: Path,
        daily_products: dict[str, Path],
        daily_land_cover: Path,
        tile_name: str,
    ) -> list[str]:
        if args.coarse_sensor == "VIIRS":
            return [
                sys.executable, str(scripts / "run_viirs_nbar_hdf.py"),
                "--vnp09ga", str(daily_coarse),
                "--fiso", str(daily_products[info.parameter_products[0]]),
                "--fvol", str(daily_products[info.parameter_products[1]]),
                "--fgeo", str(daily_products[info.parameter_products[2]]),
                "--quality", str(daily_products["VNP43D40"]),
                "--land-cover", str(daily_land_cover),
                "--band", info.viirs_band,
                "--output", str(output),
                "--brdf-source-output", str(output.with_name(f"{output.stem}_BRDF_source.tif")),
            ]
        if info.modis_band is None:
            raise ValueError(f"No MODIS counterpart is defined for {info.landsat_band}.")
        return [
            sys.executable, str(scripts / "run_modis_nbar_hdf.py"),
            "--mod09ga", str(daily_coarse),
            "--mcd43a1", str(daily_products[f"MCD43A1:{tile_name}"]),
            "--mcd43a2", str(daily_products[f"MCD43A2:{tile_name}"]),
            "--land-cover", str(daily_land_cover),
            "--band", str(info.modis_band),
            "--output", str(output),
            "--brdf-source-output", str(output.with_name(f"{output.stem}_BRDF_source.tif")),
        ]

    def build_daily_nbar_mosaic(
        info: BandInfo,
        acquisition_date: date,
        daily_coarse: dict[str, Path],
        daily_products: dict[str, Path],
        daily_land_cover: dict[str, Path],
        output_root: Path,
    ) -> Path:
        daily_token = f"{daily_product_name}.A{acquisition_date.year}{acquisition_date.timetuple().tm_yday:03d}_mosaic"
        sensor_band = info.viirs_band if args.coarse_sensor == "VIIRS" else f"B{info.modis_band}"
        mosaic_output = output_root / f"{daily_token}_{sensor_band}_NBAR_gapfilled.tif"
        tile_outputs: list[Path] = []
        for tile_name, daily_path in daily_coarse.items():
            source_id = _viirs_daily_id(daily_path) if args.coarse_sensor == "VIIRS" else _modis_daily_id(daily_path)
            tile_output = (
                output_root
                / "TILES"
                / f"{source_id}_{sensor_band}_NBAR_gapfilled.tif"
            )
            _run(
                nbar_command(
                    info,
                    tile_output,
                    daily_path,
                    daily_products,
                    daily_land_cover[tile_name],
                    tile_name,
                )
            )
            tile_outputs.append(tile_output)
            if not args.reuse_existing_nbar:
                intermediate_files.add(tile_output)
                intermediate_files.add(tile_output.with_name(f"{tile_output.stem}_BRDF_source.tif"))
        if not args.reuse_existing_nbar:
            intermediate_files.add(mosaic_output)
        return _mosaic_viirs_nbar_tiles(tile_outputs, mosaic_output)

    def daily_nbar_mosaic(info, acquisition_date, daily_coarse, daily_products, daily_land_cover):
        def build(directory):
            return build_daily_nbar_mosaic(info, acquisition_date, daily_coarse,
                                          daily_products, daily_land_cover, directory)

        def validate(path):
            dataset = gdal.Open(str(path))
            if dataset is None or dataset.RasterCount != 1 or dataset.RasterXSize < 1 or dataset.RasterYSize < 1:
                raise RuntimeError(f"Invalid cached NBAR raster: {path}")
            if dataset.GetRasterBand(1).ReadAsArray(0, 0, 1, 1) is None:
                raise RuntimeError(f"Unreadable cached NBAR raster: {path}")

        if not args.reuse_existing_nbar:
            return build(nbar_outputs)
        # Include source versions, tile membership, band and preprocessing code.
        code_files = sorted(p for p in scripts.glob('*.py') if not p.name.startswith('test_'))
        key = cache_key([*daily_coarse.values(), *daily_products.values(), *daily_land_cover.values()],
                        {'sensor': args.coarse_sensor, 'band': info.landsat_band,
                         'date': acquisition_date.isoformat(), 'tiles': sorted(daily_coarse),
                         'gdal': gdal.VersionInfo(), 'numpy': np.__version__}, code_files)
        return cached_product(data_root / 'OUTPUTS' / 'NBAR_CACHE_V2', key, build, validate)

    pair_coreg_nbar: dict[date, Path] = {}
    pair_requested_nbar: dict[date, Path] = {}
    for pair_date in pair_dates:
        pair_daily, product_files, land_cover = daily_inputs[pair_date]
        pair_coreg_nbar[pair_date] = daily_nbar_mosaic(
            coregistration_band, pair_date, pair_daily, product_files, land_cover
        )
        pair_requested_nbar[pair_date] = (
            pair_coreg_nbar[pair_date]
            if (
                requested.viirs_band == coregistration_band.viirs_band
                if args.coarse_sensor == "VIIRS"
                else requested.modis_band == coregistration_band.modis_band
            )
            else daily_nbar_mosaic(requested, pair_date, pair_daily, product_files, land_cover)
        )
    prediction_requested_nbar = daily_nbar_mosaic(
        requested, args.prediction_date, prediction_daily, prediction_product_files, prediction_land_cover
    )

    pair_masks: dict[date, Path | None] = {pair_date: None for pair_date in pair_dates}
    prediction_mask: Path | None = None
    window: tuple[int, int, int, int] | None = None
    # Remove stale masks from a previous quality-masked run as well, even when
    # the current default run does not create fresh ones.
    intermediate_files.update(
        {
            *(mask_outputs / f"{path_row_id}_{pair_date:%Y%m%d}_VNP09GA_mosaic_valid_mask_M5_B4.tif" for pair_date in pair_dates),
            mask_outputs / f"{prediction_id}_valid_mask_M5_B4.tif",
        }
    )
    if args.use_quality_mask:
        for pair_date, files in zip(pair_dates, pair_landsat_files):
            pair_vnp09 = daily_inputs[pair_date][0]
            mask = mask_outputs / f"{path_row_id}_{pair_date:%Y%m%d}_VNP09GA_mosaic_valid_mask_M5_B4.tif"
            _run([
                sys.executable, str(scripts / "build_coregistration_mask.py"),
                "--landsat-reference", str(files["B4"]),
                "--landsat-qa", str(files["QA_PIXEL"]),
                *sum((["--vnp09ga", str(path)] for path in pair_vnp09.values()), []),
                "--output", str(mask),
            ])
            pair_masks[pair_date] = mask
            intermediate_files.add(mask)
        prediction_mask = mask_outputs / f"{prediction_id}_valid_mask_M5_B4.tif"
        _run([
            sys.executable, str(scripts / "build_coregistration_mask.py"),
            "--landsat-reference", str(pair_landsat_files[0]["B4"]),
            "--landsat-qa", str(pair_landsat_files[0]["QA_PIXEL"]),
            *sum((["--vnp09ga", str(path)] for path in prediction_daily.values()), []),
            "--output", str(prediction_mask),
        ])
        intermediate_files.add(prediction_mask)
        reference_mask = pair_masks[reference_date]
        if reference_mask is None:
            raise RuntimeError("Quality-mask mode did not create the reference mask.")
        window = _best_window(reference_mask, args.window_size, args.minimum_window_valid_fraction)
    registered_m5 = coregistration_outputs / f"{pair_coreg_nbar[reference_date].stem}_coregistered.tif"
    coregistration_script = {
        "moving-window": "paper_coregister_landsat_viirs.py",
        "arosics": "arosics_coregister_landsat_viirs.py",
        "none": "no_coregister_landsat_coarse.py",
    }[args.coregistration_method]
    coregistration_command = [
        sys.executable, str(scripts / coregistration_script),
        "--landsat", str(pair_landsat_files[0][args.coregistration_band]),
        "--viirs-nbar", str(pair_coreg_nbar[reference_date]),
        "--output", str(registered_m5),
    ]
    if args.coregistration_method == "arosics":
        coregistration_command.extend(["--window-size", str(args.window_size)])
    if args.use_quality_mask and args.coregistration_method != "none":
        reference_mask = pair_masks[reference_date]
        if reference_mask is None or window is None:
            raise RuntimeError("Quality-mask mode did not prepare a matching window.")
        coregistration_command[8:8] = [
            "--valid-mask", str(reference_mask),
            "--window", ",".join(str(value) for value in window),
            "--window-size", str(args.window_size),
        ]
    registered_pair_requested: dict[date, Path] = {}
    for pair_date in pair_dates:
        source = pair_requested_nbar[pair_date]
        if pair_date == reference_date and source == pair_coreg_nbar[reference_date]:
            registered_pair_requested[pair_date] = registered_m5
            continue
        destination = coregistration_outputs / f"{source.stem}_coregistered_on_{reference_date:%Y%m%d}.tif"
        coregistration_command.extend(["--apply-viirs", str(source), str(destination)])
        registered_pair_requested[pair_date] = destination
    registered_prediction = coregistration_outputs / f"{prediction_requested_nbar.stem}_coregistered.tif"
    coregistration_command.extend(
        ["--apply-viirs", str(prediction_requested_nbar), str(registered_prediction)]
    )
    _run(coregistration_command)

    coregistration_report_path = registered_m5.with_suffix({
        "moving-window": ".coregistration.json",
        "arosics": ".arosics.json",
        "none": ".no_coregistration.json",
    }[args.coregistration_method])
    intermediate_files.update(
        {
            registered_m5,
            coregistration_report_path,
            registered_prediction,
            *registered_pair_requested.values(),
        }
    )
    if args.coregistration_method == "moving-window":
        intermediate_files.add(registered_m5.with_suffix(".correlation_scores.csv"))
    elif args.coregistration_method == "arosics":
        # AROSICS creates these Landsat-grid warps before applying the one
        # reference-pair correction to every remaining date.
        arosics_applied_sources = [
            (pair_coreg_nbar[reference_date], registered_m5),
            *(
                (pair_requested_nbar[pair_date], registered_pair_requested[pair_date])
                for pair_date in pair_dates
                if pair_date != reference_date
                or pair_requested_nbar[pair_date] != pair_coreg_nbar[reference_date]
            ),
            (prediction_requested_nbar, registered_prediction),
        ]
        intermediate_files.update(
            output.parent / f"{source.stem}_on_landsat_grid.tif"
            for source, output in arosics_applied_sources
        )
    coregistration_report = json.loads(coregistration_report_path.read_text(encoding="utf-8"))
    pair_suffix = "_and_".join(pair_date.strftime("%Y%m%d") for pair_date in pair_dates)
    starfm_run_directory = (
        run_outputs_root
        / "STARFM"
        / args.coarse_sensor
        / coregistration_method_directory
        / coregistration_band_directory
        / path_row_id
        / f"{args.prediction_date:%Y%m%d}_from_{pair_suffix}_{args.band}"
    )
    starfm_outputs = _run_original_starfm(
        executable=args.starfm_executable,
        pairs=[
            StarfmPair(pair_date, files[args.band], files["QA_PIXEL"], registered_pair_requested[pair_date], pair_masks[pair_date])
            for pair_date, files in zip(pair_dates, pair_landsat_files)
        ],
        prediction_viirs=registered_prediction,
        prediction_mask=prediction_mask,
        # AROSICS has already geometrically corrected the rasters themselves.
        # Its shifts may be sub-pixel; quality masks are currently disallowed
        # in that mode, so STARFM does not need an integer mask displacement.
        shift_x_pixels=(
            int(coregistration_report["x_shift_pixels"])
            if args.coregistration_method == "moving-window"
            else 0
        ),
        shift_y_pixels=(
            int(coregistration_report["y_shift_pixels"])
            if args.coregistration_method == "moving-window"
            else 0
        ),
        output_directory=starfm_run_directory,
        prediction_date=args.prediction_date,
        band_name=args.band,
        max_search_distance=args.starfm_max_search_distance,
        uncertainty=args.starfm_uncertainty,
        use_mask_files=args.starfm_use_mask_files,
    )
    final_output = Path(starfm_outputs["output_geotiff"])
    if not final_output.is_file():
        raise RuntimeError("STARFM reported success but the final GeoTIFF is missing; intermediates were retained.")
    validation_metrics: dict[str, Any] | None = None
    if heldout_landsat_files is not None:
        coregistration_quality_metric, coregistration_quality_percent = (
            _coregistration_quality_metadata(args.coregistration_method, coregistration_report)
        )
        validation_directory = (
            run_outputs_root
            / "VALIDATION"
            / args.coarse_sensor
            / coregistration_method_directory
            / coregistration_band_directory
            / path_row_id
            / f"{args.prediction_date:%Y%m%d}_from_{pair_suffix}_{args.band}"
        )
        validation_metrics = _evaluate_heldout_landsat(
            predicted_landsat=final_output,
            heldout_landsat=heldout_landsat_files[args.band],
            heldout_landsat_qa=heldout_landsat_files["QA_PIXEL"],
            output_directory=validation_directory,
            band_name=args.band,
            coregistration_method=args.coregistration_method,
            coregistration_quality_metric=coregistration_quality_metric,
            coregistration_quality_percent=coregistration_quality_percent,
            coregistration_report_path=coregistration_report_path,
        )
    retaining_intermediate_outputs = (
        _should_keep_intermediate_outputs(
            args.keep_intermediate_outputs, args.validate_held_out_landsat
        )
        and not args.discard_intermediate_outputs_after_validation
    )
    removed_intermediate_files = 0
    if not retaining_intermediate_outputs:
        starfm_files = [
            path for path in starfm_run_directory.iterdir()
            if path.is_file() and path != final_output
        ]
        removed_intermediate_files = _remove_intermediate_outputs(
            intermediate_files | set(starfm_files),
            [starfm_run_directory / "INPUTS"],
        )

    manifest = {
        "test_id": args.test_id,
        "selection_method": selection_method,
        "pair_count": pair_count,
        "requested_explicit_pair_dates": (
            [pair_date.isoformat() for pair_date in explicit_pair_dates]
            if explicit_pair_dates is not None
            else None
        ),
        "prediction_date": args.prediction_date.isoformat(),
        "max_cloud_percent": args.max_cloud_percent,
        "selected_pair_dates": [pair_date.isoformat() for pair_date in pair_dates],
        "reference_grid_pair_date": reference_date.isoformat(),
        "landsat_pairs": [
            {
                "date": pair_date.isoformat(),
                "product_id": str(item["id"]),
                "scene_cloud_percent": item["properties"].get("eo:cloud_cover"),
            }
            for item, pair_date in zip(landsat_items, pair_dates)
        ],
        "wrs2_path": args.path,
        "wrs2_row": args.row,
        "requested_landsat_band": requested.landsat_band,
        "coregistration_landsat_band": coregistration_band.landsat_band,
        "coregistration_viirs_band": (
            coregistration_band.viirs_band if args.coarse_sensor == "VIIRS" else None
        ),
        "coregistration_modis_band": (
            coregistration_band.modis_band if args.coarse_sensor == "MODIS" else None
        ),
        "coarse_sensor": args.coarse_sensor,
        "coregistration_method": args.coregistration_method,
        "coregistration_report": str(coregistration_report_path.absolute()),
        "requested_viirs_band": requested.viirs_band if args.coarse_sensor == "VIIRS" else None,
        "requested_modis_band": requested.modis_band if args.coarse_sensor == "MODIS" else None,
        "quality_control_enabled": args.use_quality_mask,
        "starfm_external_mask_files_enabled": args.starfm_use_mask_files,
        "starfm_max_search_distance_m": args.starfm_max_search_distance,
        "starfm_uncertainty_scaled": args.starfm_uncertainty,
        "heldout_landsat_validation_enabled": args.validate_held_out_landsat,
        "land_cover_years": {
            acquisition_date.isoformat(): {"acquisition_year": acquisition_date.year, "mcd12q1_year": year}
            for acquisition_date, year in sorted(land_cover_years.items())
        },
        "heldout_landsat_reference": (
            {
                "date": args.prediction_date.isoformat(),
                "product_id": str(heldout_item["id"]),
                "scene_cloud_percent": heldout_item["properties"].get("eo:cloud_cover"),
            }
            if heldout_item is not None
            else None
        ),
        "coarse_tiles": {
            pair_date.isoformat(): sorted(daily_inputs[pair_date][0]) for pair_date in pair_dates
        } | {
            "prediction_date": sorted(prediction_daily),
        },
        "automatic_matching_window_landsat_pixels": (
            {"x_offset": window[0], "y_offset": window[1], "width": window[2], "height": window[3]}
            if window is not None
            else None
        ),
        "inputs": {
            "landsat": {
                pair_date.isoformat(): {key: str(value.absolute()) for key, value in files.items()}
                for pair_date, files in zip(pair_dates, downloaded_landsat)
            },
            "coarse_pair_dates": {
                pair_date.isoformat(): {
                    "daily_reflectance": {key: str(value.absolute()) for key, value in daily_inputs[pair_date][0].items()},
                    "brdf": {key: str(value.absolute()) for key, value in daily_inputs[pair_date][1].items()},
                    "land_cover": {key: str(value.absolute()) for key, value in daily_inputs[pair_date][2].items()},
                }
                for pair_date in pair_dates
            },
            "coarse_prediction_date": {
                "daily_reflectance": {key: str(value.absolute()) for key, value in prediction_daily.items()},
                "brdf": {key: str(value.absolute()) for key, value in prediction_product_files.items()},
                "land_cover": {key: str(value.absolute()) for key, value in prediction_land_cover.items()},
            },
            "heldout_landsat": (
                {key: str(value.absolute()) for key, value in heldout_landsat_files.items()}
                if heldout_landsat_files is not None
                else None
            ),
        },
        "outputs": {
            "final_output_geotiff": str(final_output.absolute()),
            "heldout_landsat_validation": validation_metrics,
            "intermediate_outputs_removed": not retaining_intermediate_outputs,
            "intermediate_outputs_preserved_for_heldout_validation": (
                args.validate_held_out_landsat
                and not args.keep_intermediate_outputs
                and not args.discard_intermediate_outputs_after_validation
            ),
            "intermediate_outputs_discarded_after_validation": (
                args.validate_held_out_landsat
                and args.discard_intermediate_outputs_after_validation
            ),
            "removed_intermediate_file_count": removed_intermediate_files,
        },
    }
    manifest_path = (
        test_results_root / "run_manifest.json"
        if test_results_root is not None
        else (
            data_root
            / "MANIFESTS"
            / args.coarse_sensor
            / coregistration_method_directory
            / coregistration_band_directory
            / path_row_id
            / f"{args.prediction_date:%Y%m%d}_{pair_suffix}_{args.band}.json"
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if test_results_root is not None:
        manifest["outputs"]["test_summary"] = _write_test_summary(
            test_results_root, manifest, coregistration_report
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
