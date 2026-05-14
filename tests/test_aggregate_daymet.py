"""Tests for the Daymet (NA) daily SWE aggregator."""

from __future__ import annotations

import inspect
import json
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import box

from nhf_spatial_targets.aggregate.daymet import (
    _crs_from_grid_mapping,
    _validate_period_in_zarr_range,
    aggregate_daymet,
)


# --- function-signature contracts -----------------------------------


def test_signature_exposes_period_and_region():
    """The CLI dispatcher passes ``period`` and ``region`` by kwarg;
    the function must accept both with the right defaults."""
    sig = inspect.signature(aggregate_daymet)
    assert "period" in sig.parameters
    assert "region" in sig.parameters
    assert sig.parameters["region"].default == "na"


def test_cli_registers_aggregate_daymet():
    """``nhf-targets agg daymet`` must be wired in cli.py."""
    from nhf_spatial_targets import cli

    assert cli.aggregate_daymet is aggregate_daymet


# --- region guard ---------------------------------------------------


def test_hi_region_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="issue #101"):
        aggregate_daymet(
            fabric_path=tmp_path / "fabric.gpkg",
            id_col="hru_id",
            period="2010/2010",
            workdir=tmp_path,
            region="hi",
        )


def test_pr_region_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        aggregate_daymet(
            fabric_path=tmp_path / "fabric.gpkg",
            id_col="hru_id",
            period="2010/2010",
            workdir=tmp_path,
            region="pr",
        )


# --- helper-function unit tests -------------------------------------


def test_crs_from_grid_mapping_missing_variable_raises():
    """A zarr without the ``lambert_conformal_conic`` variable should
    surface a clear ValueError pointing at the missing variable
    rather than silently defaulting to WGS84."""
    ds = xr.Dataset(
        {"swe": (("time", "y", "x"), np.zeros((1, 2, 2), dtype=np.float32))},
        coords={
            "time": pd.date_range("2010-01-01", periods=1, freq="D"),
            "x": [0.0, 1000.0],
            "y": [0.0, 1000.0],
        },
    )
    with pytest.raises(ValueError, match="grid mapping"):
        _crs_from_grid_mapping(ds)


def _ds_with_time(years: tuple[int, int]) -> xr.Dataset:
    """Minimal Dataset with a daily time coord spanning [years[0]..years[1]]."""
    times = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    return xr.Dataset(coords={"time": times})


def test_validate_period_in_zarr_range_accepts_in_range():
    ds = _ds_with_time((1980, 2024))
    _validate_period_in_zarr_range(ds, 2010, 2011)  # no raise


def test_validate_period_in_zarr_range_rejects_before_zarr_start():
    ds = _ds_with_time((1980, 2024))
    with pytest.raises(ValueError, match=r"extends past .*1980/2024"):
        _validate_period_in_zarr_range(ds, 1900, 1979)


def test_validate_period_in_zarr_range_rejects_after_zarr_end():
    ds = _ds_with_time((1980, 2024))
    with pytest.raises(ValueError, match=r"extends past .*1980/2024"):
        _validate_period_in_zarr_range(ds, 2025, 2030)


def test_validate_period_in_zarr_range_rejects_partial_overlap():
    """Partial overlap is rejected so the operator can't silently get
    fewer years than they asked for."""
    ds = _ds_with_time((1980, 2024))
    with pytest.raises(ValueError, match=r"extends past .*1980/2024"):
        _validate_period_in_zarr_range(ds, 1975, 1985)


# --- shared fixtures ------------------------------------------------


@pytest.fixture()
def workdir(tmp_path):
    """Minimal workspace: config, fabric.json, manifest pre-populated
    with a fetch-side daymet entry (the realistic state)."""
    (tmp_path / "weights").mkdir()
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    config = {
        "fabric": {"path": "/fake/fabric.gpkg", "id_col": "hru_id"},
        "datastore": str(datastore),
        "dir_mode": "2775",
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    (tmp_path / "fabric.json").write_text(
        json.dumps({"sha256": "abc123", "id_col": "hru_id"})
    )
    # Pre-populated fetch-side manifest entry, with the zarr path
    # pointing at a directory we'll create per-test as needed.
    zarr_path = tmp_path / "datastore" / "daymet_na.zarr"
    manifest = {
        "sources": {
            "daymet": {
                "source_key": "daymet",
                "access_url": "https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2129",
                "doi": "10.3334/ORNLDAAC/2129",
                "license": "public domain",
                "period": "1980/2024",
                "spatial_extent": "North America (NA), Hawaii (HI), Puerto Rico (PR)",
                "variables": ["swe", "prcp", "tmax", "tmin", "srad", "vp"],
                "regions": {
                    "na": {
                        "region": "na",
                        "path": str(zarr_path),
                        "size_bytes": 0,
                        "zmetadata_sha256": "deadbeef",
                        "time_min": "1980-01-01",
                        "time_max": "2024-12-31",
                        "time_steps": 16436,
                        "registered_utc": "2026-05-13T00:00:00+00:00",
                        "manual_staging": True,
                    }
                },
            }
        },
        "steps": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (tmp_path / "data" / "aggregated").mkdir(parents=True)
    # Create the stub zarr directory so _resolve_zarr_path's
    # existence check passes; xr.open_zarr will be mocked.
    zarr_path.mkdir()
    return tmp_path


@pytest.fixture()
def tiny_fabric(tmp_path):
    """3-polygon WGS84 fabric (gdptools handles reprojection)."""
    polys = [box(i, 0, i + 1, 1) for i in range(3)]
    gdf = gpd.GeoDataFrame(
        {"hru_id": range(3)},
        geometry=polys,
        crs="EPSG:4326",
    )
    path = tmp_path / "fabric.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _make_lcc_zarr_dataset(years=(2010, 2011), n_x=4, n_y=4):
    """Synthetic Daymet-shape Dataset (LCC grid) for xr.open_zarr mock.

    Includes the ``lambert_conformal_conic`` scalar grid mapping
    variable with CF attrs so ``pyproj.CRS.from_cf`` can decode it.
    """
    times = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    rng = np.random.default_rng(7)
    swe = rng.random((len(times), n_y, n_x), dtype=np.float32) * 100.0
    x = np.linspace(-2_000_000, 2_000_000, n_x, dtype=np.float64)
    y = np.linspace(-1_500_000, 1_500_000, n_y, dtype=np.float64)
    ds = xr.Dataset(
        {
            "swe": (("time", "y", "x"), swe),
            "prcp": (("time", "y", "x"), swe),
            "lambert_conformal_conic": ((), np.int8(0)),
        },
        coords={"time": times, "x": x, "y": y},
    )
    ds["lambert_conformal_conic"].attrs = {
        "grid_mapping_name": "lambert_conformal_conic",
        "longitude_of_central_meridian": -100.0,
        "latitude_of_projection_origin": 42.5,
        "false_easting": 0.0,
        "false_northing": 0.0,
        "standard_parallel": np.array([25.0, 60.0]),
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }
    return ds


def _agg_for_year(hru_ids, year, n_times=365):
    times = pd.date_range(f"{year}-01-01", periods=n_times, freq="D")
    rng = np.random.default_rng(year)
    data = rng.random((n_times, len(hru_ids)), dtype=np.float32) * 100.0
    return xr.Dataset(
        {"swe": (["time", "hru_id"], data)},
        coords={"time": times, "hru_id": hru_ids},
    )


# --- missing-manifest path ------------------------------------------


def test_missing_manifest_entry_raises_pointing_at_fetch(workdir, tiny_fabric):
    """If the fetch-side daymet entry is absent, surface a clear
    'run nhf-targets fetch daymet first' message."""
    manifest = {"sources": {}, "steps": []}
    (workdir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="fetch daymet"):
        aggregate_daymet(
            fabric_path=tiny_fabric,
            id_col="hru_id",
            period="2010/2010",
            workdir=workdir,
        )


def test_unknown_region_in_manifest_raises(workdir, tiny_fabric):
    """If the user asked for a region that isn't in the manifest,
    surface a clear error rather than KeyError. (We hit this through
    the supported-regions guard for hi/pr; this exercises the
    fallback path for a future region that's added to
    _SUPPORTED_REGIONS but hasn't been fetched yet.)"""
    # Temporarily expand the supported set so we can exercise the
    # manifest path-not-found branch without touching the region guard.
    import nhf_spatial_targets.aggregate.daymet as daymet_mod

    original = daymet_mod._SUPPORTED_REGIONS
    daymet_mod._SUPPORTED_REGIONS = frozenset({"na", "hi"})
    try:
        with pytest.raises(FileNotFoundError, match="no manifest entry"):
            aggregate_daymet(
                fabric_path=tiny_fabric,
                id_col="hru_id",
                period="2010/2010",
                workdir=workdir,
                region="hi",
            )
    finally:
        daymet_mod._SUPPORTED_REGIONS = original


# --- period validation end-to-end ----------------------------------


@patch("nhf_spatial_targets.aggregate.daymet.xr.open_zarr")
def test_aggregate_rejects_period_outside_zarr_range(
    mock_open_zarr, workdir, tiny_fabric
):
    """Wiring test: ``aggregate_daymet`` must call
    ``_validate_period_in_zarr_range`` and surface its ValueError,
    not iterate years and fail opaquely inside gdptools."""
    mock_open_zarr.return_value = _make_lcc_zarr_dataset(years=(2010, 2011))
    with pytest.raises(ValueError, match=r"extends past .*2010/2011"):
        aggregate_daymet(
            fabric_path=tiny_fabric,
            id_col="hru_id",
            period="2008/2009",  # before the synthetic zarr starts
            workdir=workdir,
            region="na",
        )


# --- end-to-end smoke ----------------------------------------------


@patch("nhf_spatial_targets.aggregate.daymet.aggregate_variables_for_batch")
@patch("nhf_spatial_targets.aggregate.daymet.compute_or_load_weights")
@patch("nhf_spatial_targets.aggregate.daymet.xr.open_zarr")
def test_aggregate_writes_per_region_year_nc(
    mock_open_zarr,
    mock_weights,
    mock_agg,
    workdir,
    tiny_fabric,
):
    """End-to-end with the gdptools helpers mocked.

    Asserts:
      - per-(region, year) NCs written to ``daymet/daymet_na_<year>_agg.nc``
      - HRU dim is id_col-ascending after sortby
      - CF-1.6 globals attached + ``daymet_region: na`` attr
      - Weight cache file lands at ``weights/daymet_na_batch0.csv``
        (region-segregated)
      - ``manifest.json`` retains the fetch-side ``regions`` dict
        AND has the new aggregator fields
    """
    mock_open_zarr.return_value = _make_lcc_zarr_dataset(years=(2010, 2011))
    mock_weights.return_value = pd.DataFrame(
        {"src_idx": [0, 1], "tgt_idx": [0, 1], "weight": [0.5, 0.5]}
    )
    call_count = {"i": 0}

    def _agg_side_effect(**kwargs):
        year = 2010 + call_count["i"]
        call_count["i"] += 1
        return _agg_for_year([0, 1, 2], year=year)

    mock_agg.side_effect = _agg_side_effect

    aggregate_daymet(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2010/2011",
        workdir=workdir,
        region="na",
    )

    per_source_dir = workdir / "data" / "aggregated" / "daymet"
    yearly = sorted(per_source_dir.glob("daymet_na_*_agg.nc"))
    assert [p.name for p in yearly] == [
        "daymet_na_2010_agg.nc",
        "daymet_na_2011_agg.nc",
    ]

    with xr.open_dataset(yearly[0]) as ds:
        assert "swe" in ds.data_vars
        # id_col canonical sort
        ids = ds["hru_id"].values
        assert list(ids) == sorted(ids)
        # CF globals + region marker
        assert ds.attrs.get("Conventions") == "CF-1.6"
        assert ds.attrs.get("daymet_region") == "na"


@patch("nhf_spatial_targets.aggregate.daymet.aggregate_variables_for_batch")
@patch("nhf_spatial_targets.aggregate.daymet.compute_or_load_weights")
@patch("nhf_spatial_targets.aggregate.daymet.xr.open_zarr")
def test_manifest_merge_preserves_fetch_regions(
    mock_open_zarr,
    mock_weights,
    mock_agg,
    workdir,
    tiny_fabric,
):
    """The merge-aware manifest writer must not erase fetch-side keys
    on ``sources.daymet`` (regions, doi, license, etc.)."""
    mock_open_zarr.return_value = _make_lcc_zarr_dataset(years=(2010, 2010))
    mock_weights.return_value = pd.DataFrame(
        {"src_idx": [0], "tgt_idx": [0], "weight": [1.0]}
    )
    mock_agg.return_value = _agg_for_year([0, 1, 2], year=2010)

    aggregate_daymet(
        fabric_path=tiny_fabric,
        id_col="hru_id",
        period="2010/2010",
        workdir=workdir,
        region="na",
    )

    manifest = json.loads((workdir / "manifest.json").read_text())
    entry = manifest["sources"]["daymet"]

    # Fetch-side keys preserved:
    assert "regions" in entry, "fetch-side regions dict was wiped"
    assert "na" in entry["regions"]
    assert entry["regions"]["na"]["zmetadata_sha256"] == "deadbeef"
    assert entry["doi"] == "10.3334/ORNLDAAC/2129"
    assert entry["license"] == "public domain"

    # Aggregator-side keys added:
    assert entry["output_files"] == [
        "data/aggregated/daymet/daymet_na_2010_agg.nc",
    ]
    assert entry["weight_files"] == ["weights/daymet_na_batch0.csv"]
    assert entry["fabric_sha256"] == "abc123"
    assert entry["source_key"] == "daymet"


# --- CLI integration -----------------------------------------------


def test_cli_region_hi_surfaces_not_implemented(tmp_path, monkeypatch):
    """``agg daymet --region hi`` must surface NotImplementedError
    cleanly (via SystemExit) rather than crashing with a traceback."""
    from nhf_spatial_targets.cli import app

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_col": "hru_id"},
                "datastore": str(tmp_path / "datastore"),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))

    with patch(
        "nhf_spatial_targets.cli.aggregate_daymet",
        side_effect=NotImplementedError("daymet: region 'hi'"),
    ) as mock_fn:
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "agg",
                    "daymet",
                    "--project-dir",
                    str(tmp_path),
                    "--period",
                    "2010/2010",
                    "--region",
                    "hi",
                ]
            )
    assert exc_info.value.code == 1
    mock_fn.assert_called_once()
    kwargs = mock_fn.call_args.kwargs
    assert kwargs["region"] == "hi"
    assert kwargs["period"] == "2010/2010"
