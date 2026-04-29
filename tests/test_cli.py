"""Tests for the cyclopts CLI layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nhf_spatial_targets.cli import app


def _run(*tokens: str) -> None:
    """Invoke the cyclopts app, suppressing the SystemExit(0) on success."""
    try:
        app(list(tokens), exit_on_error=False)
    except SystemExit as exc:
        if exc.code != 0:
            raise


def _run_meta(*tokens: str) -> None:
    """Invoke via the meta launcher, suppressing SystemExit(0)."""
    try:
        app.meta(list(tokens), exit_on_error=False)
    except SystemExit as exc:
        if exc.code != 0:
            raise


# ---- run command -----------------------------------------------------------


def test_run_nonexistent_project_dir(tmp_path):
    """Exit code 2 when --project-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run("run", "--project-dir", str(tmp_path / "missing"))


def test_run_missing_fabric_json(tmp_path):
    """Exit code 2 when fabric.json is missing (validate not run yet)."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "config.yml").write_text("targets: {}")
    with pytest.raises(SystemExit, match="2"):
        _run("run", "--project-dir", str(workdir))


def test_run_dispatches_enabled_targets(tmp_path):
    """Dispatches to builder for each enabled target."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n  aet:\n    enabled: false\n"
    )
    (workdir / "fabric.json").write_text("{}")

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir))

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "runoff"


def test_run_single_target(tmp_path):
    """--target selects a single target by name."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n  aet:\n    enabled: true\n"
    )
    (workdir / "fabric.json").write_text("{}")

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir), "--target", "aet")

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "aet"


def test_run_unknown_target(tmp_path):
    """Exit code 1 for an unknown target name."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n"
    )
    (workdir / "fabric.json").write_text("{}")

    with pytest.raises(SystemExit, match="1"):
        _run("run", "--project-dir", str(workdir), "--target", "bogus")


# ---- init command ----------------------------------------------------------


def test_init_calls_init_project(tmp_path):
    """init command calls init_project with the provided project dir."""
    workdir = tmp_path / "new-ws"
    with patch(
        "nhf_spatial_targets.init_run.init_project", return_value=workdir
    ) as mock_init:
        _run("init", "--project-dir", str(workdir))

    mock_init.assert_called_once_with(workdir)


def test_init_existing_project_exits(tmp_path):
    """Exit code 1 when project already exists."""
    workdir = tmp_path / "existing-ws"
    with patch(
        "nhf_spatial_targets.init_run.init_project",
        side_effect=FileExistsError("already exists"),
    ):
        with pytest.raises(SystemExit, match="1"):
            _run("init", "--project-dir", str(workdir))


# ---- validate command ------------------------------------------------------


def test_validate_nonexistent_project_dir(tmp_path):
    """Exit code 2 when --project-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run("validate", "--project-dir", str(tmp_path / "missing"))


def test_validate_calls_validate_workspace(tmp_path):
    """validate command calls validate_workspace with the provided project dir."""
    workdir = tmp_path / "ws"
    workdir.mkdir()
    with patch("nhf_spatial_targets.validate.validate_workspace") as mock_validate:
        _run("validate", "--project-dir", str(workdir))

    mock_validate.assert_called_once_with(workdir)


def test_validate_failure_exits(tmp_path):
    """Exit code 1 when validation fails."""
    workdir = tmp_path / "ws"
    workdir.mkdir()
    with patch(
        "nhf_spatial_targets.validate.validate_workspace",
        side_effect=ValueError("bad config"),
    ):
        with pytest.raises(SystemExit, match="1"):
            _run("validate", "--project-dir", str(workdir))


# ---- fetch merra2 command -------------------------------------------------


def test_fetch_merra2_nonexistent_project_dir(tmp_path):
    """Exit code 2 when --project-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run(
            "fetch",
            "merra2",
            "--project-dir",
            str(tmp_path / "missing"),
            "--period",
            "2010/2010",
        )


def test_fetch_merra2_calls_fetch(tmp_path):
    """CLI wires --project-dir and --period to fetch_merra2()."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    mock_result = {
        "source_key": "merra2",
        "files": [{"path": "data/raw/merra2/f.nc4", "size_bytes": 100}],
        "access_url": "https://example.com",
        "variables": ["SFMC"],
        "period": "2010/2010",
        "bbox": {},
        "download_timestamp": "2026-01-01T00:00:00+00:00",
    }

    with patch(
        "nhf_spatial_targets.fetch.merra2.fetch_merra2",
        return_value=mock_result,
    ) as mock_fetch:
        _run(
            "fetch",
            "merra2",
            "--project-dir",
            str(workdir),
            "--period",
            "2010/2010",
        )

    mock_fetch.assert_called_once_with(workdir=workdir, period="2010/2010")


# ---- fetch nldas-mosaic command --------------------------------------------


def test_fetch_nldas_mosaic_nonexistent_project_dir():
    with pytest.raises(SystemExit):
        _run(
            "fetch",
            "nldas-mosaic",
            "--project-dir",
            "/no/such/dir",
            "--period",
            "2010/2010",
        )


@patch("nhf_spatial_targets.fetch.nldas.fetch_nldas_mosaic")
def test_fetch_nldas_mosaic_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_nldas_mosaic()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "nldas-mosaic",
        "--project-dir",
        str(workdir),
        "--period",
        "2010/2010",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2010/2010")


# ---- fetch nldas-noah command ----------------------------------------------


def test_fetch_nldas_noah_nonexistent_project_dir():
    with pytest.raises(SystemExit):
        _run(
            "fetch",
            "nldas-noah",
            "--project-dir",
            "/no/such/dir",
            "--period",
            "2010/2010",
        )


@patch("nhf_spatial_targets.fetch.nldas.fetch_nldas_noah")
def test_fetch_nldas_noah_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_nldas_noah()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "nldas-noah",
        "--project-dir",
        str(workdir),
        "--period",
        "2010/2010",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2010/2010")


# ---- fetch ncep-ncar command -----------------------------------------------


def test_fetch_ncep_ncar_nonexistent_project_dir():
    with pytest.raises(SystemExit):
        _run(
            "fetch",
            "ncep-ncar",
            "--project-dir",
            "/no/such/dir",
            "--period",
            "2010/2010",
        )


@patch("nhf_spatial_targets.fetch.ncep_ncar.fetch_ncep_ncar")
def test_fetch_ncep_ncar_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_ncep_ncar()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "ncep-ncar",
        "--project-dir",
        str(workdir),
        "--period",
        "2010/2010",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2010/2010")


# ---- fetch mod16a2 command -------------------------------------------------


def test_fetch_mod16a2_nonexistent_project_dir():
    """mod16a2 fetch fails with nonexistent --project-dir."""
    with pytest.raises(SystemExit):
        _run(
            "fetch",
            "mod16a2",
            "--project-dir",
            "/no/such/dir",
            "--period",
            "2005/2005",
        )


@patch("nhf_spatial_targets.fetch.modis.fetch_mod16a2")
def test_fetch_mod16a2_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_mod16a2()."""
    mock_fetch.return_value = {"files": [], "consolidated_ncs": {}}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "mod16a2",
        "--project-dir",
        str(workdir),
        "--period",
        "2005/2005",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005")


# ---- fetch mod10c1 command -------------------------------------------------


def test_fetch_mod10c1_nonexistent_project_dir():
    """mod10c1 fetch fails with nonexistent --project-dir."""
    with pytest.raises(SystemExit):
        _run(
            "fetch",
            "mod10c1",
            "--project-dir",
            "/no/such/dir",
            "--period",
            "2005/2005",
        )


@patch("nhf_spatial_targets.fetch.modis.fetch_mod10c1")
def test_fetch_mod10c1_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_mod10c1()."""
    mock_fetch.return_value = {"files": [], "consolidated_ncs": {}}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "mod10c1",
        "--project-dir",
        str(workdir),
        "--period",
        "2005/2005",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005")


# ---- fetch watergap22d command ---------------------------------------------


@patch("nhf_spatial_targets.fetch.pangaea.fetch_watergap22d")
def test_fetch_watergap22d_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_watergap22d()."""
    mock_fetch.return_value = {"files": []}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "watergap22d",
        "--project-dir",
        str(workdir),
        "--period",
        "2000/2009",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2000/2009")


# ---- fetch reitz2017 command -----------------------------------------------


@patch("nhf_spatial_targets.fetch.reitz2017.fetch_reitz2017")
def test_fetch_reitz2017_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_reitz2017()."""
    mock_fetch.return_value = {"files": []}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "reitz2017",
        "--project-dir",
        str(workdir),
        "--period",
        "2000/2009",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2000/2009")


# ---- fetch mwbm-climgrid command -----------------------------------------------


@patch("nhf_spatial_targets.fetch.mwbm_climgrid.fetch_mwbm_climgrid")
def test_fetch_mwbm_climgrid_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_mwbm_climgrid()."""
    mock_fetch.return_value = {"files": []}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "mwbm-climgrid",
        "--project-dir",
        str(workdir),
        "--period",
        "1980/2015",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="1980/2015")


# ---- agg ssebop command ----------------------------------------------------


def test_agg_ssebop_nonexistent_project_dir(tmp_path):
    """Exit code 2 when --project-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run(
            "agg",
            "ssebop",
            "--project-dir",
            str(tmp_path / "missing"),
            "--period",
            "2010/2010",
        )


@patch("nhf_spatial_targets.aggregate.ssebop.aggregate_ssebop")
def test_agg_ssebop_calls_aggregate(mock_agg, tmp_path):
    """CLI wires --project-dir and --period to aggregate_ssebop()."""
    import xarray as xr

    mock_agg.return_value = xr.Dataset({"et": (["time", "nhm_id"], [[1.0]])})
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "config.yml").write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n  id_col: nhm_id\ndatastore: /fake/ds\n"
    )
    (workdir / "fabric.json").write_text("{}")
    _run(
        "agg",
        "ssebop",
        "--project-dir",
        str(workdir),
        "--period",
        "2010/2010",
    )
    mock_agg.assert_called_once()


# ---- catalog commands ------------------------------------------------------


def test_catalog_sources():
    """catalog sources runs without error."""
    _run("catalog", "sources")


def test_catalog_variables():
    """catalog variables runs without error."""
    _run("catalog", "variables")


# ---- meta launcher / verbose -----------------------------------------------


def test_verbose_flag():
    """--verbose flag is accepted and calls setup_logging(verbose=True)."""
    with patch("nhf_spatial_targets.cli.setup_logging") as mock_setup:
        _run_meta("--verbose", "catalog", "sources")

    mock_setup.assert_called_once_with(True)


def test_default_no_verbose():
    """Without --verbose, setup_logging is called with False."""
    with patch("nhf_spatial_targets.cli.setup_logging") as mock_setup:
        _run_meta("catalog", "sources")

    mock_setup.assert_called_once_with(False)
