"""Tests for the Margulis WUS-SR daily SWE aggregation adapter."""

from __future__ import annotations

from nhf_spatial_targets import catalog as cat
from nhf_spatial_targets.aggregate.margulis_wus_sr import (
    ADAPTER,
    aggregate_margulis_wus_sr,
)


def test_adapter_declares_swe_only():
    assert ADAPTER.source_key == "margulis_wus_sr"
    assert ADAPTER.output_name == "margulis_wus_sr_agg.nc"
    assert ADAPTER.variables == ("SWE",)


def test_adapter_uses_daily_files_glob():
    """Consolidated NCs land under <datastore>/margulis_wus_sr/daily/."""
    assert ADAPTER.files_glob == "daily/margulis_wus_sr_daily_*.nc"


def test_adapter_wgs84_and_plain_mean():
    """Consolidator reprojects native EASE-Grid to regular WGS84 lat/lon;
    no per-pixel mask in a pre_aggregate_hook means default ``mean`` is
    correct (same as SNODAS)."""
    assert ADAPTER.source_crs == "EPSG:4326"
    assert ADAPTER.stat_method == "mean"


def test_adapter_has_no_hooks():
    """Per-pixel NaN over WUS-domain edges decodes via mask_and_scale;
    no pre/post hook needed."""
    assert ADAPTER.pre_aggregate_hook is None
    assert ADAPTER.post_aggregate_hook is None


def test_adapter_variable_matches_catalog():
    """The adapter's declared SWE var must appear in catalog/sources.yml
    so the cf attrs applied by the driver line up with the consolidated NCs.
    """
    entry = cat.source("margulis_wus_sr")
    catalog_vars = {v["name"] for v in entry["variables"]}
    assert set(ADAPTER.variables).issubset(catalog_vars)


def test_aggregate_margulis_is_callable_with_period_kwarg():
    """CLI dispatcher forwards --period via kwargs; the function must
    accept it for the _run_tier_agg(period=...) path."""
    import inspect

    sig = inspect.signature(aggregate_margulis_wus_sr)
    assert "period" in sig.parameters
    assert sig.parameters["period"].default is None


def test_cli_registers_agg_margulis():
    """``nhf-targets agg margulis-wus-sr`` must be wired in cli.py."""
    from nhf_spatial_targets import cli

    assert cli.aggregate_margulis_wus_sr is aggregate_margulis_wus_sr


def test_aggregate_margulis_missing_raw_raises(tmp_path):
    """#309 removed the fabric_scope token gate: margulis behaves like every
    other source — aggregating without fetched raw data raises
    FileNotFoundError pointing at the fetch command, instead of silently
    skipping based on a catalog token. (Zero spatial overlap with a fabric
    is now handled geometrically by the driver's coverage guard.)"""
    import json

    import pytest
    import yaml

    datastore = tmp_path / "datastore"
    datastore.mkdir()
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fake.gpkg"), "id_col": "hru_id"},
                "datastore": str(datastore),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {}, "steps": []}))

    with pytest.raises(FileNotFoundError, match="fetch margulis_wus_sr"):
        aggregate_margulis_wus_sr(
            fabric_path=tmp_path / "fake.gpkg",
            id_col="hru_id",
            workdir=tmp_path,
        )
