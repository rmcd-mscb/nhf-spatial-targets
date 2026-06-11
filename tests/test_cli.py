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
        "  snow_covered_area:\n    enabled: false\n"
        "  snow_water_equivalent:\n    enabled: false\n",
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir))

    mock_dispatch.assert_called_once()
    args = mock_dispatch.call_args[0]
    assert args[0] == "runoff"
    # Second positional is the Project (signature: _dispatch(name, project))
    from nhf_spatial_targets.workspace import Project

    assert isinstance(args[1], Project)


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
    # Second positional is the Project (signature: _dispatch(name, project))
    from nhf_spatial_targets.workspace import Project

    assert isinstance(args[1], Project)


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


def test_run_sca_target_dispatches(tmp_path):
    """--target snow_covered_area routes to the sca builder via _dispatch.

    Mirrors test_run_single_target's pattern: mock _dispatch so a
    mis-keyed registration in the builders dict would surface as an
    AssertionError on the first-positional check. The real builder is
    not invoked here (it needs a full fabric fixture); end-to-end SCA
    behavior is covered in tests/test_targets_sca.py.
    """
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n"
        "  snow_covered_area:\n"
        "    enabled: true\n"
        "    period: 2000-01-01/2010-12-31\n",
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir), "--target", "snow_covered_area")

    mock_dispatch.assert_called_once()
    args = mock_dispatch.call_args[0]
    assert args[0] == "snow_covered_area"
    from nhf_spatial_targets.workspace import Project

    assert isinstance(args[1], Project)


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


# ---- fetch ua-swe command --------------------------------------------------


@patch("nhf_spatial_targets.fetch.ua_swe.fetch_ua_swe")
def test_fetch_ua_swe_calls_fetch(mock_fetch, tmp_path):
    """CLI wires every flag to fetch_ua_swe()."""
    mock_fetch.return_value = {"n_consolidated": 2, "n_failed": 0, "n_skipped": 0}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    _run(
        "fetch",
        "ua-swe",
        "--project-dir",
        str(workdir),
        "--period",
        "2010/2011",
        "--worker-index",
        "0",
        "--n-workers",
        "1",
    )
    mock_fetch.assert_called_once_with(
        workdir=workdir,
        period="2010/2011",
        worker_index=0,
        n_workers=1,
    )


def test_fetch_ua_swe_nonexistent_project_dir(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        _run(
            "fetch",
            "ua-swe",
            "--project-dir",
            str(tmp_path / "missing"),
            "--period",
            "2010/2011",
        )


@patch("nhf_spatial_targets.fetch.ua_swe.fetch_ua_swe")
def test_fetch_ua_swe_partial_failure_exits_3(mock_fetch, tmp_path):
    """Consolidation failures (n_failed > 0) exit EXIT_PARTIAL (3) (issue #299)."""
    mock_fetch.return_value = {"n_consolidated": 30, "n_failed": 8, "n_skipped": 2}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(
            "fetch",
            "ua-swe",
            "--project-dir",
            str(workdir),
            "--period",
            "2010/2011",
        )
    assert exc.value.code == 3


@patch("nhf_spatial_targets.fetch.ua_swe.fetch_ua_swe")
def test_fetch_ua_swe_skip_only_exits_0(mock_fetch, tmp_path):
    """Skip-only runs (no failures/errors) stay exit 0 — expected gaps don't fail."""
    mock_fetch.return_value = {"n_consolidated": 38, "n_failed": 0, "n_skipped": 2}
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    # _run re-raises only non-zero SystemExit; a clean return = exit 0.
    _run(
        "fetch",
        "ua-swe",
        "--project-dir",
        str(workdir),
        "--period",
        "2010/2011",
    )


@patch("nhf_spatial_targets.fetch.snodas.fetch_snodas")
def test_fetch_snodas_download_errors_exit_3(mock_fetch, tmp_path):
    """SNODAS download errors (n_errors > 0) exit EXIT_PARTIAL (3) (issue #299)."""
    mock_fetch.return_value = {
        "years": [],
        "n_consolidated": 18,
        "n_failed": 1,
        "n_skipped": 0,
        "n_errors": 3,
    }
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(
            "fetch",
            "snodas",
            "--project-dir",
            str(workdir),
            "--period",
            "2020/2020",
        )
    assert exc.value.code == 3


# ---- _emit_fetch_banner helper ---------------------------------------------


def _banner(summary):
    """Run _emit_fetch_banner against a wide StringIO console; return (text, rc)."""
    import io

    from rich.console import Console

    from nhf_spatial_targets.cli.fetch import _emit_fetch_banner

    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    rc = _emit_fetch_banner(console, "SRC", summary)
    return console.file.getvalue(), rc


def test_emit_banner_full_success_is_green_returns_false():
    text, rc = _banner({"n_consolidated": 40, "n_failed": 0, "n_skipped": 0})
    assert rc is False
    assert "downloaded to datastore" in text


def test_emit_banner_no_rollup_treated_as_success():
    """Single-shot fetchers omit the n_* rollup — banner stays green."""
    text, rc = _banner({"files": []})
    assert rc is False
    assert "downloaded to datastore" in text


def test_emit_banner_non_dict_summary_is_success():
    _text, rc = _banner(None)
    assert rc is False


def test_emit_banner_failures_yellow_returns_true():
    text, rc = _banner({"n_consolidated": 30, "n_failed": 8, "n_skipped": 2})
    assert rc is True
    assert "8 failed" in text
    assert "2 skipped" in text
    assert "downloaded to datastore" not in text


def test_emit_banner_skip_only_yellow_returns_false():
    """Skip-only is incomplete (yellow) but not actionable (exit 0)."""
    text, rc = _banner({"n_consolidated": 38, "n_failed": 0, "n_skipped": 2})
    assert rc is False
    assert "2 skipped" in text


def test_emit_banner_download_errors_returns_true():
    _text, rc = _banner({"n_consolidated": 18, "n_errors": 3})
    assert rc is True


# ---- fetch all command -----------------------------------------------------

# Every fetcher `fetch all` imports and dispatches, keyed by the patch target
# (the module each is imported from inside fetch_all_cmd).
_FETCH_ALL_TARGETS = (
    "nhf_spatial_targets.fetch.era5_land.fetch_era5_land",
    "nhf_spatial_targets.fetch.gldas.fetch_gldas",
    "nhf_spatial_targets.fetch.merra2.fetch_merra2",
    "nhf_spatial_targets.fetch.nldas.fetch_nldas_mosaic",
    "nhf_spatial_targets.fetch.nldas.fetch_nldas_noah",
    "nhf_spatial_targets.fetch.ncep_ncar.fetch_ncep_ncar",
    "nhf_spatial_targets.fetch.modis.fetch_mod16a2",
    "nhf_spatial_targets.fetch.modis.fetch_mod10c1",
    "nhf_spatial_targets.fetch.pangaea.fetch_watergap22d",
    "nhf_spatial_targets.fetch.reitz2017.fetch_reitz2017",
    "nhf_spatial_targets.fetch.mwbm_climgrid.fetch_mwbm_climgrid",
    "nhf_spatial_targets.fetch.daymet.fetch_daymet",
    "nhf_spatial_targets.fetch.snodas.fetch_snodas",
    "nhf_spatial_targets.fetch.margulis_wus_sr.fetch_margulis_wus_sr",
    "nhf_spatial_targets.fetch.ua_swe.fetch_ua_swe",
)


def _patch_all_fetchers(stack, overrides=None):
    """Patch every `fetch all` fetcher to return a clean ({}) summary.

    overrides maps a target string to the summary that source should return
    instead. Returns nothing — callers assert via exit code / captured output.
    """
    overrides = overrides or {}
    for target in _FETCH_ALL_TARGETS:
        mock = stack.enter_context(patch(target))
        mock.return_value = overrides.get(target, {})


def test_fetch_all_clean_run_exits_0(tmp_path):
    """Every source returns a clean summary → no partial, exit 0."""
    from contextlib import ExitStack

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    with ExitStack() as stack:
        _patch_all_fetchers(stack)
        # _run re-raises only non-zero SystemExit; clean return == exit 0.
        _run("fetch", "all", "--project-dir", str(workdir), "--period", "2010/2011")


def test_fetch_all_partial_source_exits_3(tmp_path, capsys):
    """One source reporting genuine failures → exit 3 and named in the banner."""
    from contextlib import ExitStack

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    overrides = {
        "nhf_spatial_targets.fetch.snodas.fetch_snodas": {
            "n_consolidated": 5,
            "n_failed": 2,
            "n_skipped": 0,
            "n_errors": 0,
        }
    }
    with ExitStack() as stack:
        _patch_all_fetchers(stack, overrides)
        with pytest.raises(SystemExit) as exc:
            _run(
                "fetch",
                "all",
                "--project-dir",
                str(workdir),
                "--period",
                "2010/2011",
            )
    assert exc.value.code == 3
    assert "snodas" in capsys.readouterr().out


def test_fetch_all_skip_only_source_exits_0(tmp_path):
    """A source with only skips (expected archive gaps) does not fail `fetch all`."""
    from contextlib import ExitStack

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    overrides = {
        "nhf_spatial_targets.fetch.snodas.fetch_snodas": {
            "n_consolidated": 18,
            "n_failed": 0,
            "n_skipped": 3,
            "n_errors": 0,
        }
    }
    with ExitStack() as stack:
        _patch_all_fetchers(stack, overrides)
        _run("fetch", "all", "--project-dir", str(workdir), "--period", "2010/2011")


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
    _dispatch("runoff", project)
    assert (workdir / "targets" / "runoff_targets.nc").exists()


# ---- CLI grammar (issue #319) ----------------------------------------------


def test_run_target_nickname_maps_to_long_key(tmp_path):
    """--target accepts the pixi-task nicknames (rch/som/sca/swe)."""
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n  recharge:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n",
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--project-dir", str(workdir), "--target", "rch")

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "recharge"


def test_run_unknown_nickname_still_errors(tmp_path, capsys):
    """A token that is neither a config key nor a nickname exits 1, and the
    error echoes the token the user typed (not a nickname expansion)."""
    workdir = _make_minimal_project(
        tmp_path,
        "targets:\n  runoff:\n    enabled: true\n    period: 2000-01-01/2000-12-31\n",
    )

    with pytest.raises(SystemExit) as exc:
        _run("run", "--project-dir", str(workdir), "--target", "bogus")
    assert exc.value.code == 1
    assert "Unknown target: bogus" in capsys.readouterr().err


def test_project_dir_short_alias_on_fetch(tmp_path):
    """fetch commands accept -d as an alias for --project-dir."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    with patch(
        "nhf_spatial_targets.fetch.merra2.fetch_merra2",
        return_value={"source_key": "merra2"},
    ) as mock_fetch:
        _run("fetch", "merra2", "-d", str(workdir), "-p", "2010/2010")

    mock_fetch.assert_called_once_with(workdir=workdir, period="2010/2010")


def test_project_dir_short_alias_on_agg(tmp_path):
    """agg commands accept -d as an alias for --project-dir."""
    with patch("nhf_spatial_targets.cli.aggregate_gldas") as mock_agg:
        import json as json_mod

        workdir = _make_minimal_project(tmp_path)
        (workdir / "fabric.json").write_text(json_mod.dumps({"id_col": "nhm_id"}))
        _run("agg", "gldas", "-d", str(workdir))

    mock_agg.assert_called_once()


def test_aggregate_is_an_alias_for_agg(tmp_path):
    """The spelled-out sub-app name dispatches identically to 'agg'."""
    with patch("nhf_spatial_targets.cli.aggregate_gldas") as mock_agg:
        workdir = _make_minimal_project(tmp_path)
        _run("aggregate", "gldas", "-d", str(workdir))

    mock_agg.assert_called_once()


@pytest.mark.parametrize(
    "verb", ["upgrade-config", "upgrade-manifest", "rebuild-manifest", "rechunk"]
)
def test_old_flat_maintenance_verbs_are_removed(verb, tmp_path):
    """The pre-#319 root spellings are a hard break: unknown commands."""
    from cyclopts.exceptions import UnknownCommandError

    with pytest.raises(UnknownCommandError):
        app([verb, "--project-dir", str(tmp_path)], exit_on_error=False)


def test_maintenance_subapp_hosts_all_four_verbs():
    from nhf_spatial_targets.cli.maintenance import maintenance_app

    for verb in ("check-config", "check-manifest", "rebuild-manifest", "rechunk"):
        assert verb in maintenance_app


@pytest.mark.parametrize("verb", ["check-manifest", "rebuild-manifest", "rechunk"])
def test_maintenance_verbs_parse_through_the_app(verb, tmp_path):
    """Each verb's cyclopts signature binds -d; a missing project exits 2.

    check-config gets full token-level coverage in test_upgrade_config.py;
    these smokes pin the parse path of the other three (a broken Parameter
    annotation would otherwise only surface at runtime).
    """
    missing = tmp_path / "no-such-project"
    with pytest.raises(SystemExit) as exc:
        app(["maintenance", verb, "-d", str(missing)], exit_on_error=False)
    assert exc.value.code == 2


def test_release_publish_old_scope_spelling_rejected(tmp_path):
    """--scope source (singular, pre-#319) is no longer a valid choice."""
    from cyclopts.exceptions import CoercionError

    with pytest.raises(CoercionError):
        app(
            ["release", "publish", "-d", str(tmp_path), "--scope", "source"],
            exit_on_error=False,
        )


def test_release_publish_old_source_key_flag_rejected(tmp_path):
    """--source-key (pre-#319) is no longer a recognized option."""
    from cyclopts.exceptions import UnknownOptionError

    with pytest.raises(UnknownOptionError):
        app(
            [
                "release",
                "publish",
                "-d",
                str(tmp_path),
                "--scope",
                "sources",
                "--source-key",
                "era5_land",
            ],
            exit_on_error=False,
        )
