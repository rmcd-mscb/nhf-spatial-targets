"""Tests for the cyclopts CLI layer."""

from __future__ import annotations

from pathlib import Path
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


def _make_minimal_project(tmp_path: Path, config_extra: str = "") -> Path:
    """Build a minimal valid project workdir for CLI run-command tests."""
    import json as json_mod

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    datastore = tmp_path / "store"
    datastore.mkdir()
    fabric_path = tmp_path / "fabric.gpkg"
    # Write a config that satisfies workspace.load() (datastore + fabric.path required)
    (workdir / "config.yml").write_text(
        f"datastore: {datastore}\n"
        f"fabric:\n  path: {fabric_path}\n"
        "output:\n  dir: /fake/out\n" + config_extra
    )
    (workdir / "fabric.json").write_text(json_mod.dumps({"id_col": "nhm_id"}))
    return workdir


def test_run_dispatches_enabled_targets(tmp_path):
    """Dispatches to builder for each enabled target."""
    # Disable all defaults except runoff so defaults-merge doesn't add extras.
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n"
        "  runoff:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n"
        "  aet:\n    enabled: false\n"
        "  recharge:\n    enabled: false\n"
        "  soil_moisture:\n    enabled: false\n"
        "  snow_covered_area:\n    enabled: false\n",
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir))

    mock_dispatch.assert_called_once()
    args = mock_dispatch.call_args[0]
    assert args[0] == "runoff"
    # Third positional is now a Project, not a dict
    from nhf_spatial_targets.workspace import Project

    assert isinstance(args[2], Project)


def test_run_single_target(tmp_path):
    """--target selects a single target by name."""
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n  runoff:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n"
        "  aet:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n",
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir), "--target", "aet")

    mock_dispatch.assert_called_once()
    args = mock_dispatch.call_args[0]
    assert args[0] == "aet"
    # Third positional is now a Project, not a dict
    from nhf_spatial_targets.workspace import Project

    assert isinstance(args[2], Project)


def test_run_unknown_target(tmp_path):
    """Exit code 1 for an unknown target name."""
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n  runoff:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n",
    )

    with pytest.raises(SystemExit, match="1"):
        _run("run", "--project-dir", str(workdir), "--target", "bogus")


def test_run_skips_not_implemented_targets(tmp_path, capsys):
    """NotImplementedError from a stub target is logged + the run continues."""
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n"
        "  runoff:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n"
        "  aet:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n",
    )

    def _fake_dispatch(name, *a, **kw):
        if name == "aet":
            raise NotImplementedError("aet is a stub")

    with patch("nhf_spatial_targets.cli._dispatch", side_effect=_fake_dispatch) as md:
        _run("run", "--project-dir", str(workdir))
    # Both targets should have been attempted; aet skipped, runoff dispatched.
    called_names = [c.args[0] for c in md.call_args_list]
    assert "aet" in called_names
    assert "runoff" in called_names
    err = capsys.readouterr().err
    assert "WARNING" in err and "aet" in err and "skipping" in err


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
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005", force=False)


@patch("nhf_spatial_targets.fetch.modis.fetch_mod16a2")
def test_fetch_mod16a2_force_flag(mock_fetch, tmp_path):
    """--force is forwarded to fetch_mod16a2()."""
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
        "--force",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005", force=True)


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
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005", force=False)


@patch("nhf_spatial_targets.fetch.modis.fetch_mod10c1")
def test_fetch_mod10c1_force_flag(mock_fetch, tmp_path):
    """--force is forwarded to fetch_mod10c1()."""
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
        "--force",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2005/2005", force=True)


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


# ---- SWE fetch commands (issue #99) ----------------------------------------


@patch("nhf_spatial_targets.fetch.daymet.fetch_daymet")
def test_fetch_daymet_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir, --period, --source-path, --region to fetch_daymet()."""
    mock_fetch.return_value = {"regions": {}}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    zroot = tmp_path / "zarrs"
    zroot.mkdir()
    _run(
        "fetch",
        "daymet",
        "--project-dir",
        str(workdir),
        "--period",
        "2020/2020",
        "--source-path",
        str(zroot),
        "--region",
        "na",
    )
    mock_fetch.assert_called_once_with(
        workdir=workdir,
        period="2020/2020",
        source_path=zroot,
        region="na",
    )


def test_fetch_daymet_nonexistent_project_dir(tmp_path):
    """Exit code 2 when --project-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run(
            "fetch",
            "daymet",
            "--project-dir",
            str(tmp_path / "missing"),
            "--period",
            "2020/2020",
        )


@patch("nhf_spatial_targets.fetch.snodas.fetch_snodas")
def test_fetch_snodas_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir, --period, --worker-index, --n-workers to fetch_snodas()."""
    mock_fetch.return_value = {"years": []}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "snodas",
        "--project-dir",
        str(workdir),
        "--period",
        "2020/2020",
        "--worker-index",
        "0",
        "--n-workers",
        "1",
    )
    mock_fetch.assert_called_once_with(
        workdir=workdir,
        period="2020/2020",
        worker_index=0,
        n_workers=1,
    )


def test_fetch_snodas_nonexistent_project_dir(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        _run(
            "fetch",
            "snodas",
            "--project-dir",
            str(tmp_path / "missing"),
            "--period",
            "2020/2020",
        )


@patch("nhf_spatial_targets.fetch.margulis_wus_sr.fetch_margulis_wus_sr")
def test_fetch_margulis_wus_sr_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --project-dir and --period to fetch_margulis_wus_sr()."""
    mock_fetch.return_value = {"years": []}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "margulis-wus-sr",
        "--project-dir",
        str(workdir),
        "--period",
        "2000/2000",
    )
    mock_fetch.assert_called_once_with(workdir=workdir, period="2000/2000")


def test_fetch_margulis_wus_sr_nonexistent_project_dir(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        _run(
            "fetch",
            "margulis-wus-sr",
            "--project-dir",
            str(tmp_path / "missing"),
            "--period",
            "2000/2000",
        )


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


# ---- _dispatch runoff smoke test -------------------------------------------


def test_run_runoff_smoke(tmp_path):
    """Invoking _dispatch for runoff calls run.build via the Project."""
    from tests.test_targets_run import _make_runoff_project

    from nhf_spatial_targets.cli import _dispatch
    from nhf_spatial_targets.workspace import load

    workdir = _make_runoff_project(tmp_path)
    project = load(workdir)
    _dispatch("runoff", {}, project)
    assert (workdir / "targets" / "runoff_targets.nc").exists()
