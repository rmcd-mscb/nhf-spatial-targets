"""Smoke tests for the `agg` CLI subcommands."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "subcommand,target_fn",
    [
        ("era5-land", "nhf_spatial_targets.cli.aggregate_era5_land"),
        ("gldas", "nhf_spatial_targets.cli.aggregate_gldas"),
        ("merra2", "nhf_spatial_targets.cli.aggregate_merra2"),
        ("ncep-ncar", "nhf_spatial_targets.cli.aggregate_ncep_ncar"),
        ("nldas-mosaic", "nhf_spatial_targets.cli.aggregate_nldas_mosaic"),
        ("nldas-noah", "nhf_spatial_targets.cli.aggregate_nldas_noah"),
        ("watergap22d", "nhf_spatial_targets.cli.aggregate_watergap22d"),
        ("mod16a2", "nhf_spatial_targets.cli.aggregate_mod16a2"),
        ("mod10c1", "nhf_spatial_targets.cli.aggregate_mod10c1"),
    ],
)
def test_agg_subcommand_dispatches(subcommand, target_fn, tmp_path, monkeypatch):
    import json
    import yaml
    from nhf_spatial_targets.cli import app

    # Minimal project
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_col": "nhm_id"},
                "datastore": str(tmp_path / "datastore"),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))

    with patch(target_fn) as mock_agg:
        with pytest.raises(SystemExit):
            app(["agg", subcommand, "--project-dir", str(tmp_path)])
    mock_agg.assert_called_once()
    kwargs = mock_agg.call_args.kwargs
    assert kwargs["fabric_path"] == str(tmp_path / "fabric.gpkg")
    assert kwargs["id_col"] == "nhm_id"
    assert kwargs["workdir"] == tmp_path
    assert kwargs["batch_size"] == 500


def test_agg_all_runs_every_source(tmp_path):
    from contextlib import ExitStack

    import json
    import yaml
    from nhf_spatial_targets.cli import app

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_col": "nhm_id"},
                "datastore": str(tmp_path / "datastore"),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))

    target_fns = [
        "nhf_spatial_targets.cli.aggregate_era5_land",
        "nhf_spatial_targets.cli.aggregate_gldas",
        "nhf_spatial_targets.cli.aggregate_merra2",
        "nhf_spatial_targets.cli.aggregate_ncep_ncar",
        "nhf_spatial_targets.cli.aggregate_nldas_mosaic",
        "nhf_spatial_targets.cli.aggregate_nldas_noah",
        "nhf_spatial_targets.cli.aggregate_watergap22d",
        "nhf_spatial_targets.cli.aggregate_mod16a2",
        "nhf_spatial_targets.cli.aggregate_mod10c1",
    ]
    with ExitStack() as stack:
        mocks = [stack.enter_context(patch(fn)) for fn in target_fns]
        with pytest.raises(SystemExit):
            app(["agg", "all", "--project-dir", str(tmp_path)])
    for m in mocks:
        m.assert_called_once()


def test_agg_tier_succeeds_when_aggregator_returns_none(tmp_path):
    """Regression: aggregators return None (not an xr.Dataset).

    MagicMock's default return_value silently satisfies `.sizes.get(...)`,
    so without this test a success-path `AttributeError` on a real None
    return slips through. Explicitly pin `return_value=None` here.
    """
    import json

    import yaml

    from nhf_spatial_targets.cli import app

    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_col": "nhm_id"},
                "datastore": str(tmp_path / "datastore"),
            }
        )
    )
    (tmp_path / "fabric.json").write_text(json.dumps({"sha256": "f00"}))

    with patch(
        "nhf_spatial_targets.cli.aggregate_era5_land",
        return_value=None,
    ) as mock_agg:
        with pytest.raises(SystemExit) as excinfo:
            app(["agg", "era5-land", "--project-dir", str(tmp_path)])
    mock_agg.assert_called_once()
    # Cyclopts exits cleanly (code 0 or None) on success — not with a raised
    # AttributeError wrapped in a non-zero exit from the broad except block.
    assert excinfo.value.code in (0, None), (
        f"Expected clean success exit; got code={excinfo.value.code!r}. "
        "A non-zero code here means _run_tier_agg crashed on the None "
        "return value from a successful aggregation."
    )
