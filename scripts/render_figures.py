"""Execute the inspection notebooks with SAVE_FIGURES=True.

Walks each notebook under notebooks/consolidated/ and notebooks/aggregated/,
executes it via ``jupyter nbconvert --execute --inplace``, and lets the
notebooks' embedded ``save_figure`` calls populate
``docs/figures/{consolidated,aggregated}/``.

The startup hook is the same trick used by inspect_aggregated.slurm /
inspect_consolidated.slurm: a temp file is written to disk and exported via
``PYTHONSTARTUP`` so the executing kernel sets ``_helpers.SAVE_FIGURES = True``
before any plotting cell runs.

Usage::

    pixi run -e dev render-figures              # all 10 notebooks
    pixi run -e dev render-figures-consolidated # 5 consolidated only
    pixi run -e dev render-figures-aggregated   # 5 aggregated only

For HPC-scale memory, prefer ``sbatch inspect_consolidated.slurm`` /
``sbatch inspect_aggregated.slurm`` — those run the same nbconvert command
under SLURM with 128–192 GB of RAM per task.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GROUPS = {
    "consolidated": {
        "dir": REPO_ROOT / "notebooks" / "consolidated",
        "helpers_dir": "notebooks/consolidated",
    },
    "aggregated": {
        "dir": REPO_ROOT / "notebooks" / "aggregated",
        "helpers_dir": "notebooks/aggregated",
    },
}


def _startup_payload(helpers_dir: str, project: str | None) -> str:
    """Build the PYTHONSTARTUP snippet that toggles save_figure for headless runs.

    Uses ``repr()`` to quote the string values so that paths or project tags
    containing ``"``, ``\\``, or other Python-source-significant characters
    can't break the resulting payload (or worse, smuggle code into it).
    """
    project_line = f"_helpers.PROJECT = {project!r}\n" if project is not None else ""
    return (
        "import sys\n"
        f"sys.path.insert(0, {helpers_dir!r})\n"
        "import _helpers\n"
        "_helpers.SAVE_FIGURES = True\n"
        f"{project_line}"
    )


def _execute(nb_path: Path, startup: Path, timeout: int) -> None:
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=python3",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(nb_path),
    ]
    env = os.environ.copy()
    env["PYTHONSTARTUP"] = str(startup)
    print(f"=== executing {nb_path.relative_to(REPO_ROOT)} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def render_group(group: str, timeout: int, project: str | None) -> None:
    cfg = GROUPS[group]
    notebooks = sorted(cfg["dir"].glob("inspect_*.ipynb"))
    if not notebooks:
        # Bubble up rather than warn-and-continue: an empty glob almost always
        # means a workflow mistake (wrong cwd, partial checkout, mistyped
        # group). Treating it as success would mask missing-figure bugs in
        # downstream "publish deck" steps.
        raise FileNotFoundError(
            f"No notebooks matching inspect_*.ipynb in {cfg['dir']}"
        )
    # Capture startup_path before writing so a write failure (ENOSPC,
    # encoding error, etc.) still leaves startup_path bound for the
    # finally block — without this, a write error would surface as a
    # confusing NameError masking the real cause.
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    startup_path = Path(fh.name)
    try:
        with fh:
            fh.write(_startup_payload(cfg["helpers_dir"], project))
        for nb in notebooks:
            _execute(nb, startup_path, timeout)
    finally:
        startup_path.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        choices=("consolidated", "aggregated", "all"),
        default="all",
        help="Which notebook group to render (default: all)",
    )
    p.add_argument(
        "--project",
        default=None,
        help=(
            "Project tag for figure subdir (e.g. 'gfv2-spatial-targets'). "
            "Notebooks set this themselves from PROJECT_DIR.name when run "
            "interactively; pass --project here to override / namespace "
            "headless renders. Pass when re-rendering committed deck "
            "figures so paths match; omit for ad-hoc local renders."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell execution timeout, seconds (default: 3600)",
    )
    args = p.parse_args()
    groups = ["consolidated", "aggregated"] if args.group == "all" else [args.group]
    for g in groups:
        render_group(g, args.timeout, args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
