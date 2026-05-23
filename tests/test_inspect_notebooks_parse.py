"""Regression: every committed inspect_*.ipynb code cell parses as Python.

A bug in #192's notebook-rewrite script doubled an import-block trailing
comma in all 11 inspect_aggregated_* + inspect_target_* notebooks, so the
first cell raised ``SyntaxError`` during render. The fix would have been
caught immediately by this test — it ``compile()``s every code cell of
every committed inspect_*.ipynb. Cheap (no kernel, no I/O beyond reading
the notebook), and it generalizes: any future broken-syntax notebook
edit fails CI before burning HPC time.
"""

from __future__ import annotations

import json
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
