"""Unit tests for scripts/render_figures.py.

The module isn't on the package path, so we load it via importlib.
Targets the small bits of pure logic — payload string construction and
the empty-glob guard — not the subprocess machinery.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_PATH = REPO_ROOT / "scripts" / "render_figures.py"


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
