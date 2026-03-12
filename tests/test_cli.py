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


def test_run_missing_both_options():
    """Exit code 2 when neither --run-dir nor --config is provided."""
    with pytest.raises(SystemExit, match="2"):
        _run("run")


def test_run_both_options(tmp_path):
    """Exit code 2 when both --run-dir and --config are provided."""
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    config = tmp_path / "pipeline.yml"
    config.write_text("targets: {}")
    with pytest.raises(SystemExit, match="2"):
        _run("run", "--run-dir", str(run_dir), "--config", str(config))


def test_run_nonexistent_run_dir(tmp_path):
    """Exit code 2 when --run-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run("run", "--run-dir", str(tmp_path / "missing"))


def test_run_dispatches_enabled_targets(tmp_path):
    """Dispatches to builder for each enabled target."""
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    config = run_dir / "config.yml"
    config.write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n  aet:\n    enabled: false\n"
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--run-dir", str(run_dir))

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "runoff"


def test_run_single_target(tmp_path):
    """--target selects a single target by name."""
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    config = run_dir / "config.yml"
    config.write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n  aet:\n    enabled: true\n"
    )

    with patch("nhf_spatial_targets.cli._dispatch") as mock_dispatch:
        _run("run", "--run-dir", str(run_dir), "--target", "aet")

    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[0][0] == "aet"


def test_run_unknown_target(tmp_path):
    """Exit code 1 for an unknown target name."""
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    config = run_dir / "config.yml"
    config.write_text(
        "fabric:\n  path: /fake/fabric.gpkg\n"
        "output:\n  dir: /fake/out\n"
        "targets:\n  runoff:\n    enabled: true\n"
    )

    with pytest.raises(SystemExit, match="1"):
        _run("run", "--run-dir", str(run_dir), "--target", "bogus")


# ---- init command ----------------------------------------------------------


def test_init_missing_fabric(tmp_path):
    """Exit code 1 when --fabric file does not exist."""
    with pytest.raises(SystemExit, match="1"):
        _run("init", "--fabric", str(tmp_path / "missing.gpkg"))


def test_init_fabric_is_directory(tmp_path):
    """Exit code 1 when --fabric points to a directory."""
    with pytest.raises(SystemExit, match="1"):
        _run("init", "--fabric", str(tmp_path))


def test_init_missing_config(tmp_path):
    """Exit code 1 when resolved config does not exist."""
    fabric = tmp_path / "fabric.gpkg"
    fabric.write_bytes(b"fake")
    with pytest.raises(SystemExit, match="1"):
        _run(
            "init",
            "--fabric",
            str(fabric),
            "--config",
            str(tmp_path / "no-such-config.yml"),
        )


# ---- fetch merra2 command -------------------------------------------------


def test_fetch_merra2_nonexistent_run_dir(tmp_path):
    """Exit code 2 when --run-dir does not exist."""
    with pytest.raises(SystemExit, match="2"):
        _run(
            "fetch",
            "merra2",
            "--run-dir",
            str(tmp_path / "missing"),
            "--period",
            "2010/2010",
        )


def test_fetch_merra2_calls_fetch(tmp_path):
    """CLI wires --run-dir and --period to fetch_merra2()."""
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()

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
            "--run-dir",
            str(run_dir),
            "--period",
            "2010/2010",
        )

    mock_fetch.assert_called_once_with(run_dir=run_dir, period="2010/2010")


# ---- fetch nldas-mosaic command --------------------------------------------


def test_fetch_nldas_mosaic_nonexistent_run_dir(capsys):
    from nhf_spatial_targets.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "fetch",
                "nldas-mosaic",
                "--run-dir",
                "/no/such/dir",
                "--period",
                "2010/2010",
            ]
        )


def test_fetch_nldas_noah_nonexistent_run_dir(capsys):
    from nhf_spatial_targets.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "fetch",
                "nldas-noah",
                "--run-dir",
                "/no/such/dir",
                "--period",
                "2010/2010",
            ]
        )


def test_fetch_ncep_ncar_nonexistent_run_dir(capsys):
    from nhf_spatial_targets.cli import main

    with pytest.raises(SystemExit):
        main(
            ["fetch", "ncep-ncar", "--run-dir", "/no/such/dir", "--period", "2010/2010"]
        )


@patch("nhf_spatial_targets.fetch.nldas.fetch_nldas_mosaic")
def test_fetch_nldas_mosaic_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --run-dir and --period to fetch_nldas_mosaic()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    _run(
        "fetch",
        "nldas-mosaic",
        "--run-dir",
        str(run_dir),
        "--period",
        "2010/2010",
    )
    mock_fetch.assert_called_once()


@patch("nhf_spatial_targets.fetch.nldas.fetch_nldas_noah")
def test_fetch_nldas_noah_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --run-dir and --period to fetch_nldas_noah()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    _run(
        "fetch",
        "nldas-noah",
        "--run-dir",
        str(run_dir),
        "--period",
        "2010/2010",
    )
    mock_fetch.assert_called_once()


@patch("nhf_spatial_targets.fetch.ncep_ncar.fetch_ncep_ncar")
def test_fetch_ncep_ncar_calls_fetch(mock_fetch, tmp_path):
    """CLI wires --run-dir and --period to fetch_ncep_ncar()."""
    mock_fetch.return_value = {"files": [], "consolidated_nc": "consolidated.nc"}
    run_dir = tmp_path / "workspace"
    run_dir.mkdir()
    _run(
        "fetch",
        "ncep-ncar",
        "--run-dir",
        str(run_dir),
        "--period",
        "2010/2010",
    )
    mock_fetch.assert_called_once()


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
