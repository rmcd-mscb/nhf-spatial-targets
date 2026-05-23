"""Regression: every committed inspect_*.ipynb code cell parses as Python,
and the per-project rep-points wrap doesn't reference a name before it's bound.

* SyntaxError guard: #192's notebook-rewrite script doubled an import-block
  trailing comma in all 11 inspect_*.ipynb so the first cell raised
  ``SyntaxError`` during render. ``test_inspect_notebook_code_cells_compile``
  ``compile()``s every code cell — any future broken-syntax notebook edit
  fails CI before burning HPC time.
* NameError guard (#198): the same regex also wrapped
  ``REPRESENTATIVE_POINTS = { ... }`` with ``load_representative_points(project_dir, TARGET) or { ... }``.
  In the 5 inspect_target_*.ipynb the original cell layout put
  ``load_project_paths(PROJECT_DIR)`` *below* the rep-points block, so the
  wrap read ``project_dir`` before it was defined → ``NameError`` at render
  time. ``test_inspect_notebook_rep_points_arg_is_defined`` catches the
  pattern statically without executing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSPECT_NOTEBOOKS = sorted(
    list((REPO_ROOT / "notebooks" / "consolidated").glob("inspect_*.ipynb"))
    + list((REPO_ROOT / "notebooks" / "aggregated").glob("inspect_*.ipynb"))
    + list((REPO_ROOT / "notebooks" / "targets").glob("inspect_*.ipynb"))
)


@pytest.mark.parametrize("nb_path", INSPECT_NOTEBOOKS, ids=lambda p: p.name)
def test_inspect_notebook_code_cells_compile(nb_path):
    """Every code cell in *nb_path* parses (no SyntaxError / IndentationError)."""
    nb = json.loads(nb_path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        text = "".join(src) if isinstance(src, list) else src
        if not text.strip():
            continue
        try:
            compile(text, f"{nb_path.name}:cell{i}", "exec")
        except SyntaxError as exc:  # pragma: no cover - the assertion is the message
            pytest.fail(
                f"{nb_path.name} cell {i} fails to compile: "
                f"{exc.msg} at line {exc.lineno}\n--- cell ---\n{text}"
            )


# Match: load_representative_points(<name>, ...)  — capture the first arg name.
_LRP_CALL = re.compile(r"load_representative_points\(\s*(\w+)\b")
# Match: an assignment to project_dir at line start, either tuple-unpack
# (``project_dir, datastore_dir, ... =``) or bare (``project_dir =``).
_ASSIGN_PROJECT_DIR = re.compile(r"^\s*project_dir\s*[,=]", re.MULTILINE)


@pytest.mark.parametrize("nb_path", INSPECT_NOTEBOOKS, ids=lambda p: p.name)
def test_inspect_notebook_rep_points_arg_is_defined(nb_path):
    """``load_representative_points(<arg>, …)`` must not read an undefined name.

    Allowed:
      - ``load_representative_points(PROJECT_DIR, TARGET)`` — ``PROJECT_DIR``
        is the user-edited ``Path`` literal at the top of every setup cell.
      - ``load_representative_points(project_dir, TARGET)`` *iff* the same
        cell defines ``project_dir`` earlier (typically via
        ``project_dir, datastore_dir, fabric_cfg = load_project_paths(PROJECT_DIR)``).

    Disallowed: any other first argument, or lowercase ``project_dir``
    without a preceding binding in the same cell.
    """
    nb = json.loads(nb_path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        text = "".join(src) if isinstance(src, list) else src
        m = _LRP_CALL.search(text)
        if not m:
            continue
        first_arg = m.group(1)
        if first_arg == "PROJECT_DIR":
            continue
        if first_arg != "project_dir":
            pytest.fail(
                f"{nb_path.name} cell {i}: "
                f"load_representative_points({first_arg}, …) — pass "
                f"PROJECT_DIR (the Path literal) or project_dir (the "
                f"load_project_paths return) instead."
            )
        # lowercase project_dir — require an earlier assignment in the cell.
        before = text[: m.start()]
        if not _ASSIGN_PROJECT_DIR.search(before):
            pytest.fail(
                f"{nb_path.name} cell {i}: "
                f"load_representative_points(project_dir, …) references "
                f"project_dir before it's bound. Either pass PROJECT_DIR "
                f"or move `project_dir, … = load_project_paths(PROJECT_DIR)` "
                f"above the rep-points block (#198)."
            )
