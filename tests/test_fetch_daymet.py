"""Tests for Daymet V4 R1 zarr-verify registration.

`fetch_daymet` does not download — it fingerprints the three regional
zarrs (NA/HI/PR) staged on a shared filesystem and records per-region
entries in manifest.json. These tests cover the period gate, the root
resolution precedence (`--source-path` > `config.yml: daymet_root`),
per-region registration and idempotency, the validation paths (missing
swe variable, malformed zarr), and the operator-friendly fallback
behaviour when only some regions are staged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from nhf_spatial_targets.fetch import daymet as _daymet_module
from nhf_spatial_targets.fetch.daymet import fetch_daymet


@pytest.fixture(autouse=True)
def _short_stability_window(monkeypatch):
    """Skip the stability window in tests."""
    monkeypatch.setattr(_daymet_module, "_STABILITY_SECONDS", 0.0)


def _make_project(tmp_path: Path, *, daymet_root: Path | None = None) -> Path:
    """Materialize a minimal valid project directory.

    If `daymet_root` is given, it is recorded in config.yml so we can
    exercise the config-fallback path; otherwise tests must pass
    `source_path=` explicitly.
    """
    datastore = tmp_path / "datastore"
    datastore.mkdir()
    cfg = {
        "fabric": {
            "path": str(tmp_path / "fabric.gpkg"),
            "id_col": "nhm_id",
        },
        "datastore": str(datastore),
    }
    if daymet_root is not None:
        cfg["daymet_root"] = str(daymet_root)
    (tmp_path / "config.yml").write_text(yaml.dump(cfg))
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    return tmp_path


def _write_dummy_zarr(
    zroot: Path,
    *,
    variables: tuple[str, ...] = ("swe", "prcp", "tmax", "tmin", "srad", "vp"),
    n_time: int = 5,
) -> None:
    """Materialize a tiny CF-ish Daymet-style zarr with consolidated metadata."""
    times = pd.date_range("2020-01-01", periods=n_time, freq="D")
    xs = np.arange(3, dtype=np.float64) * 1000.0
    ys = np.arange(3, dtype=np.float64) * 1000.0
    data_vars = {}
    for v in variables:
        data_vars[v] = (
            ("time", "y", "x"),
            np.zeros((n_time, 3, 3), dtype=np.float32),
            {"units": "kg m-2"},
        )
    # CRS placeholder variable (Daymet ships a scalar holding the
    # Lambert Conformal Conic spec).
    data_vars["lambert_conformal_conic"] = (
        (),
        np.int32(0),
        {"grid_mapping_name": "lambert_conformal_conic"},
    )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": ("time", times), "x": ("x", xs), "y": ("y", ys)},
    )
    # consolidated=False to mirror the on-disk Daymet zarrs (zarr v2
    # without a `.zmetadata` summary file).
    ds.to_zarr(zroot, mode="w", consolidated=False, zarr_format=2)
    ds.close()


def _stage_regions(
    root: Path, regions: tuple[str, ...] = ("na", "hi", "pr"), **kwargs
) -> dict[str, Path]:
    """Write a dummy zarr for each requested region; return the paths."""
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for r in regions:
        zroot = root / f"daymet_{r}.zarr"
        _write_dummy_zarr(zroot, **kwargs)
        paths[r] = zroot
    return paths


# ---------------------------------------------------------------------------
# Root resolution and period gate
# ---------------------------------------------------------------------------


def test_missing_zarr_root_raises_with_instructions(tmp_path):
    workdir = _make_project(tmp_path)
    with pytest.raises(FileNotFoundError, match="not configured"):
        fetch_daymet(workdir=workdir, period="2020/2020")


def test_source_path_overrides_config_yml(tmp_path):
    """An explicit `source_path=` wins over `daymet_root` in config.yml."""
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    workdir = _make_project(tmp_path, daymet_root=bogus)
    real_root = tmp_path / "real"
    _stage_regions(real_root, regions=("na",))
    result = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=real_root,
        region="na",
    )
    assert "na" in result["regions"]
    # Manifest path agrees with the real root, not the bogus one.
    assert str(real_root) in result["regions"]["na"]["path"]


def test_period_outside_data_range_rejected(tmp_path):
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    _stage_regions(root, regions=("na",))
    with pytest.raises(ValueError, match="outside the Daymet V4 R1"):
        fetch_daymet(
            workdir=workdir,
            period="1970/1970",
            source_path=root,
            region="na",
        )


def test_unknown_region_raises(tmp_path):
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="not recognised"):
        fetch_daymet(
            workdir=workdir,
            period="2020/2020",
            source_path=tmp_path,
            region="europe",
        )


# ---------------------------------------------------------------------------
# Per-region registration
# ---------------------------------------------------------------------------


def test_region_all_registers_three_regions(tmp_path):
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    _stage_regions(root, regions=("na", "hi", "pr"))
    result = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="all",
    )
    assert set(result["regions"].keys()) == {"na", "hi", "pr"}
    assert result["missing_regions"] == []
    # Manifest persists per-region records.
    manifest = json.loads((workdir / "manifest.json").read_text())
    regions = manifest["sources"]["daymet"]["regions"]
    assert set(regions.keys()) == {"na", "hi", "pr"}
    for r, rec in regions.items():
        assert rec["region"] == r
        assert rec["size_bytes"] > 0
        assert len(rec["zmetadata_sha256"]) == 64


def test_region_filter_single_region(tmp_path):
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    _stage_regions(root, regions=("na", "hi", "pr"))
    result = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="na",
    )
    assert set(result["regions"].keys()) == {"na"}


def test_partial_staging_skips_missing_regions(tmp_path):
    """region='all' with only NA staged returns NA + missing_regions=[hi, pr]."""
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    _stage_regions(root, regions=("na",))
    result = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="all",
    )
    assert set(result["regions"].keys()) == {"na"}
    assert set(result["missing_regions"]) == {"hi", "pr"}


def test_no_regions_present_raises(tmp_path):
    workdir = _make_project(tmp_path)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="None of the requested"):
        fetch_daymet(
            workdir=workdir,
            period="2020/2020",
            source_path=empty_root,
            region="all",
        )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_when_manifest_matches(tmp_path):
    """Second call with unchanged zarr is a no-op (download_timestamp None)."""
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    _stage_regions(root, regions=("na",))

    first = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="na",
    )
    assert first["download_timestamp"] is not None

    second = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="na",
    )
    # The cached record is reused — no new fingerprinting happened.
    assert second["download_timestamp"] is None
    assert (
        second["regions"]["na"]["zmetadata_sha256"]
        == first["regions"]["na"]["zmetadata_sha256"]
    )


def test_rerun_after_replacement_refingerprints(tmp_path):
    """Replacing the zarr changes the sha256 → fingerprint is rewritten."""
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    paths = _stage_regions(root, regions=("na",))
    first = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="na",
    )
    sha_before = first["regions"]["na"]["zmetadata_sha256"]

    # Rewrite the zarr with different content (different time length).
    import shutil

    shutil.rmtree(paths["na"])
    _write_dummy_zarr(paths["na"], n_time=10)

    second = fetch_daymet(
        workdir=workdir,
        period="2020/2020",
        source_path=root,
        region="na",
    )
    assert second["download_timestamp"] is not None
    assert second["regions"]["na"]["zmetadata_sha256"] != sha_before


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_zarr_missing_swe_variable(tmp_path):
    """A zarr without the `swe` variable is rejected with a clear message."""
    workdir = _make_project(tmp_path)
    root = tmp_path / "zarrs"
    root.mkdir()
    # Build a zarr that only has prcp/tmin/tmax (no swe, srad, vp).
    _write_dummy_zarr(
        root / "daymet_na.zarr",
        variables=("prcp", "tmin", "tmax"),
    )
    with pytest.raises(RuntimeError, match="missing required variables"):
        fetch_daymet(
            workdir=workdir,
            period="2020/2020",
            source_path=root,
            region="na",
        )


def test_raises_on_corrupt_manifest(tmp_path):
    """Bad JSON in manifest.json fails fast on the read."""
    workdir = _make_project(tmp_path)
    (workdir / "manifest.json").write_text("{not json")
    root = tmp_path / "zarrs"
    _stage_regions(root, regions=("na",))
    with pytest.raises(ValueError, match="manifest.json .* is corrupt"):
        fetch_daymet(
            workdir=workdir,
            period="2020/2020",
            source_path=root,
            region="na",
        )
