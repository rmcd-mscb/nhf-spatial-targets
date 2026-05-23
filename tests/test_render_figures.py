"""Unit tests for scripts/render_figures.py.

The module isn't on the package path, so we load it via importlib.
Targets the small bits of pure logic — payload string construction, the
PROJECT_DIR swap, and the empty-glob guard — not the subprocess machinery.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_PATH = REPO_ROOT / "scripts" / "render_figures.py"


def _make_nb(cells_src: list[str]) -> dict:
    """Minimal nbformat-shaped Notebook dict with the given code-cell sources."""
    return {
        "cells": [
            {"cell_type": "code", "source": s, "metadata": {}, "outputs": []}
            for s in cells_src
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


@pytest.fixture(scope="session")
def render():
    spec = importlib.util.spec_from_file_location("render_figures", RENDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_startup_payload_without_project(render):
    payload = render._startup_payload("notebooks/aggregated", None)
    # No PROJECT line when project is None
    assert "_helpers.PROJECT" not in payload
    assert "_helpers.SAVE_FIGURES = True" in payload
    # Payload is valid Python — exec'd by ipykernel via PYTHONSTARTUP.
    compile(payload, "<test>", "exec")


def test_startup_payload_with_project(render):
    payload = render._startup_payload("notebooks/aggregated", "gfv2-spatial-targets")
    assert "_helpers.PROJECT = 'gfv2-spatial-targets'" in payload
    compile(payload, "<test>", "exec")


def test_startup_payload_quotes_are_escaped(render):
    # A pathological --project containing a quote must not break the payload.
    payload = render._startup_payload(
        "notebooks/aggregated", 'evil"; import os; os.system("rm -rf /")  # '
    )
    # Must still parse as Python — repr() escaping prevents the injection.
    compile(payload, "<test>", "exec")


def test_startup_payload_backslashes_are_escaped(render):
    payload = render._startup_payload("notebooks/aggregated", r"foo\bar")
    compile(payload, "<test>", "exec")


def test_startup_payload_empty_string_project_is_emitted(render):
    # Empty string is distinct from None: the user explicitly passed --project ""
    # and gets _helpers.PROJECT = "" (which the helper falls back from with a
    # warning). Don't silently treat "" the same as None at this layer.
    payload = render._startup_payload("notebooks/aggregated", "")
    assert "_helpers.PROJECT = ''" in payload


def test_render_group_raises_on_empty_glob(render, tmp_path, monkeypatch):
    # Point GROUPS at an empty dir to simulate the "no notebooks found" case.
    fake_groups = {
        "consolidated": {
            "dir": tmp_path,
            "helpers_dir": "notebooks/consolidated",
        },
    }
    monkeypatch.setattr(render, "GROUPS", fake_groups)
    with pytest.raises(FileNotFoundError, match="No notebooks matching"):
        render.render_group("consolidated", timeout=60, project=None)


# --- targets group + PROJECT_DIR swap (--project-dir) ---------------------


def test_targets_group_is_registered(render):
    """The deck's target-results figures render from the same driver."""
    assert "targets" in render.GROUPS
    assert render.GROUPS["targets"]["dir"].name == "targets"


def test_swap_project_dir_single_line(render):
    src = 'PROJECT_DIR = Path("/x/gfv2-spatial-targets")\n'
    new, n = render._swap_project_dir_in_source(src, "/y/or-spatial-targets")
    assert n == 1
    assert "/y/or-spatial-targets" in new
    assert "gfv2-spatial-targets" not in new


def test_swap_project_dir_multiline(render):
    src = 'PROJECT_DIR = Path(\n    "/caldera/.../gfv2-spatial-targets"\n)\n'
    new, n = render._swap_project_dir_in_source(src, "/caldera/.../or-spatial-targets")
    assert n == 1
    assert "or-spatial-targets" in new
    # Surrounding `Path(\n  "..."\n)` formatting is preserved.
    assert new.startswith("PROJECT_DIR = Path(\n")


def test_swap_project_dir_no_match_returns_zero(render):
    src = '# nothing to repoint here\nx = Path("/etc")\n'
    new, n = render._swap_project_dir_in_source(src, "/y")
    assert n == 0
    assert new == src


def test_make_temp_nb_repoints_first_match_and_leaves_original(render, tmp_path):
    target = "/data/or-spatial-targets"
    nb_path = tmp_path / "inspect_x.ipynb"
    original = _make_nb(
        [
            "# config cell\n"
            'PROJECT_DIR = Path("/data/gfv2-spatial-targets")\n'
            "TARGET = 'aet'\n",
            "# plotting cell, mentions gfv2-spatial-targets in a comment\nx = 1\n",
        ]
    )
    nb_path.write_text(json.dumps(original))
    temp = render._make_project_dir_temp_notebook(nb_path, target)
    try:
        out = json.loads(temp.read_text())
        cell0 = "".join(out["cells"][0]["source"])
        cell1 = "".join(out["cells"][1]["source"])
        assert target in cell0 and "gfv2-spatial-targets" not in cell0
        # Only the first match is swapped; an incidental mention elsewhere
        # (a comment, a docstring) must not be rewritten.
        assert "gfv2-spatial-targets" in cell1
        # Original notebook untouched.
        assert json.loads(nb_path.read_text()) == original
    finally:
        temp.unlink(missing_ok=True)


def test_make_temp_nb_raises_when_no_project_dir_assignment(render, tmp_path):
    nb_path = tmp_path / "weird.ipynb"
    nb_path.write_text(json.dumps(_make_nb(["x = 1\n"])))
    with pytest.raises(ValueError, match="No `PROJECT_DIR = Path"):
        render._make_project_dir_temp_notebook(nb_path, "/y")


def test_render_group_with_project_dir_uses_temp(render, tmp_path, monkeypatch):
    """--project-dir routes each notebook through a temp copy, then deletes it."""
    nb_path = tmp_path / "inspect_x.ipynb"
    nb_path.write_text(
        json.dumps(_make_nb(['PROJECT_DIR = Path("/data/gfv2-spatial-targets")\n']))
    )
    monkeypatch.setattr(
        render,
        "GROUPS",
        {"consolidated": {"dir": tmp_path, "helpers_dir": "notebooks/consolidated"}},
    )
    seen: list[Path] = []

    def fake_execute(nb, startup, timeout):
        # Confirm the executor sees a temp path, NOT the committed notebook,
        # and that the temp's PROJECT_DIR is already swapped at this point.
        assert nb != nb_path
        seen.append(nb)
        assert "/data/or-spatial-targets" in nb.read_text()

    monkeypatch.setattr(render, "_execute", fake_execute)
    render.render_group(
        "consolidated",
        timeout=1,
        project="or-spatial-targets",
        project_dir="/data/or-spatial-targets",
    )
    assert len(seen) == 1
    # Temp is cleaned up after execution — leaving it would clutter the
    # notebook dir and risk an accidental `git add`.
    assert not seen[0].exists()
    # Committed notebook is untouched on disk.
    assert "/data/gfv2-spatial-targets" in nb_path.read_text()
