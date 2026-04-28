"""Tests for MWBM ClimGrid-forced fetch module."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhf_spatial_targets.fetch.mwbm_climgrid import fetch_mwbm_climgrid


def _make_project(tmp_path: Path) -> Path:
    """Materialize a minimal valid project directory."""
    import json
    import yaml

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {
                    "path": str(tmp_path / "fabric.gpkg"),
                    "id_col": "nhm_id",
                },
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    return tmp_path


def test_period_outside_data_range_rejected(tmp_path):
    """Periods outside 1900/2020 raise ValueError before any download."""
    workdir = _make_project(tmp_path)
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="1850/1900")
    with pytest.raises(ValueError, match="outside the MWBM-ClimGrid data range"):
        fetch_mwbm_climgrid(workdir=workdir, period="2000/2025")
