"""Unit tests for notebooks/consolidated/_helpers.py.

The helper module lives outside the package, so we load it via importlib
rather than a regular import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PATH = REPO_ROOT / "notebooks" / "consolidated" / "_helpers.py"


@pytest.fixture(scope="session")
def helpers():
    spec = importlib.util.spec_from_file_location("consolidated_helpers", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_loads_with_default_save_figures_off(helpers):
    assert helpers.SAVE_FIGURES is False
    assert helpers.FIGURES_DIR == Path("docs/figures/consolidated/")


def test_save_figure_no_op_when_disabled(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", False)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    fig = plt.figure()
    helpers.save_figure(fig, "test_disabled")
    plt.close(fig)
    assert not (tmp_path / "figures").exists()


def test_save_figure_writes_png_when_enabled(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(helpers, "PROJECT", "test-project")
    fig = plt.figure()
    helpers.save_figure(fig, "test_enabled")
    plt.close(fig)
    assert (tmp_path / "figures" / "test-project" / "test_enabled.png").exists()


def test_save_figure_writes_under_project_subdir(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(helpers, "PROJECT", "gfv2-spatial-targets")
    fig = plt.figure()
    helpers.save_figure(fig, "test_project")
    plt.close(fig)
    assert (tmp_path / "figures" / "gfv2-spatial-targets" / "test_project.png").exists()


def test_save_figure_no_subdir_when_project_none(helpers, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(helpers, "PROJECT", None)
    fig = plt.figure()
    with pytest.warns(UserWarning, match="PROJECT is unset"):
        helpers.save_figure(fig, "test_no_project")
    plt.close(fig)
    assert (tmp_path / "figures" / "test_no_project.png").exists()


def test_save_figure_relative_path_resolves_to_repo_root(helpers, monkeypatch):
    # The default FIGURES_DIR is relative; it should resolve to repo root,
    # not the test runner's CWD.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(helpers, "SAVE_FIGURES", True)
    monkeypatch.setattr(helpers, "FIGURES_DIR", Path("docs/figures/consolidated/"))
    # Set PROJECT so the test artefact lands in a clearly-test-only subdir,
    # outside the committed `<project>/` namespace.
    monkeypatch.setattr(helpers, "PROJECT", "_pytest")

    expected = (
        REPO_ROOT
        / "docs"
        / "figures"
        / "consolidated"
        / "_pytest"
        / "_pytest_smoke.png"
    )
    try:
        fig = plt.figure()
        helpers.save_figure(fig, "_pytest_smoke")
        plt.close(fig)
        assert expected.exists()
    finally:
        if expected.exists():
            expected.unlink()
        if expected.parent.exists():
            expected.parent.rmdir()
