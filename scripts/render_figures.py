"""Execute the inspection notebooks with SAVE_FIGURES=True.

Walks each notebook under notebooks/consolidated/, notebooks/aggregated/, and
notebooks/targets/, executes it via ``jupyter nbconvert --execute --inplace``,
and lets the notebooks' embedded ``save_figure`` calls populate
``docs/figures/{consolidated,aggregated,targets}/<project>/``.

The startup hook is the same trick used by slurm/shared/inspect_aggregated.slurm
/ slurm/shared/inspect_consolidated.slurm: a temp file is written to disk and
exported via ``PYTHONSTARTUP`` so the executing kernel sets
``_helpers.SAVE_FIGURES = True`` before any plotting cell runs.

Pass ``--project-dir`` to render figures for a project other than the gfv2
default baked into the committed notebooks. Each notebook's
``PROJECT_DIR = Path(...)`` is repointed in a temp copy before execution; the
committed notebook is untouched, so no per-project .ipynb need be committed.

Usage::

    pixi run -e dev render-figures              # all groups, gfv2 default
    pixi run -e dev render-figures-consolidated # consolidated only
    pixi run -e dev render-figures-aggregated   # aggregated only
    pixi run -e dev render-figures-targets      # targets only
    # Render an alternate fabric's figures into the same deck layout:
    pixi run -e dev render-figures -- \\
        --project-dir /caldera/.../or-spatial-targets

For HPC-scale memory, prefer ``sbatch slurm/shared/inspect_consolidated.slurm``
/ ``sbatch slurm/shared/inspect_aggregated.slurm`` — those run the same
nbconvert command under SLURM with 128–192 GB of RAM per task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    "targets": {
        "dir": REPO_ROOT / "notebooks" / "targets",
        "helpers_dir": "notebooks/targets",
    },
}

#: Matches the canonical `PROJECT_DIR = Path("...")` assignment that every
#: inspect_* notebook puts in its config cell, including the multi-line form
#: nbformat tends to write. The DOTALL `\s*` spans newlines between the call
#: and its string literal.
_PROJECT_DIR_RE = re.compile(
    r'(PROJECT_DIR\s*=\s*Path\(\s*["\'])[^"\']+(["\']\s*,?\s*\))',
    re.DOTALL,
)


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
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _swap_project_dir_in_source(source: str, project_dir: str) -> tuple[str, int]:
    """Return ``(new_source, n_replacements)`` with PROJECT_DIR repointed."""
    return _PROJECT_DIR_RE.subn(rf"\g<1>{project_dir}\g<2>", source, count=1)


def _make_project_dir_temp_notebook(nb_path: Path, project_dir: str) -> Path:
    """Write a temp copy of *nb_path* with its ``PROJECT_DIR`` repointed.

    The committed inspect_* notebooks pin ``PROJECT_DIR`` to gfv2. To render
    figures for a different project (e.g. Oregon) without committing a parallel
    set of notebooks, this swaps the assignment in the first config cell that
    contains one, writes the result to a temp ``.ipynb``, and returns its path.
    The original notebook is untouched; the caller is responsible for unlinking
    the temp file.

    Raises if no matching assignment is found — silently failing here would
    execute the notebook against the wrong project with no signal in the
    figure output beyond a path mismatch nobody checks.
    """
    nb = json.loads(nb_path.read_text())
    swapped = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        text = "".join(src) if isinstance(src, list) else src
        new_text, n = _swap_project_dir_in_source(text, project_dir)
        if n:
            # nbformat stores `source` as a list of lines (with keepends);
            # match that convention so the round-trip is clean.
            cell["source"] = new_text.splitlines(keepends=True)
            swapped = True
            break
    if not swapped:
        raise ValueError(
            f"No `PROJECT_DIR = Path(...)` assignment found in {nb_path} — "
            f"--project-dir cannot repoint a notebook that doesn't follow the "
            f"shared inspect_* config-cell convention."
        )
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", prefix=f"{nb_path.stem}.", delete=False
    )
    try:
        with fh:
            json.dump(nb, fh)
    except BaseException:
        Path(fh.name).unlink(missing_ok=True)
        raise
    return Path(fh.name)


def render_group(
    group: str, timeout: int, project: str | None, project_dir: str | None = None
) -> None:
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
            # Print the *logical* notebook name (the committed path under the
            # repo). On the --project-dir path the actual file handed to
            # nbconvert is a /tmp temp copy, which isn't under REPO_ROOT;
            # showing its temp filename in the log would be noise (and used to
            # crash on `.relative_to(REPO_ROOT)`, issue #182 follow-up). Fall
            # back to the bare path for notebooks outside the repo (tests,
            # scratch fixtures).
            try:
                display = nb.relative_to(REPO_ROOT)
            except ValueError:
                display = nb
            print(f"=== executing {display} ===", flush=True)
            if project_dir is None:
                _execute(nb, startup_path, timeout)
                continue
            # Repoint PROJECT_DIR via a temp notebook so the committed,
            # gfv2-pinned inspect_*.ipynb is untouched. The temp's executed
            # outputs are discarded — only the side-effect `save_figure` PNGs
            # under docs/figures/<group>/<PROJECT>/ matter.
            temp_nb = _make_project_dir_temp_notebook(nb, project_dir)
            try:
                _execute(temp_nb, startup_path, timeout)
            finally:
                temp_nb.unlink(missing_ok=True)
    finally:
        startup_path.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        choices=("consolidated", "aggregated", "targets", "all"),
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
            "headless renders. When --project-dir is set and --project is "
            "omitted, this defaults to the basename of --project-dir."
        ),
    )
    p.add_argument(
        "--project-dir",
        default=None,
        help=(
            "Absolute path to a project directory other than the gfv2 default "
            "baked into the committed inspect_* notebooks. When set, each "
            "notebook's `PROJECT_DIR = Path(...)` is repointed in a temp copy "
            "before execution — the committed notebook is untouched, so no "
            "per-project .ipynb need be committed. Use to render an alternate "
            "fabric's figures (e.g. or-spatial-targets) into the same deck."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell execution timeout, seconds (default: 3600)",
    )
    args = p.parse_args()
    # Default --project to the project dir's basename so figures land in the
    # right subdir without having to pass both flags.
    if args.project is None and args.project_dir is not None:
        args.project = Path(args.project_dir).name
    groups = (
        ["consolidated", "aggregated", "targets"]
        if args.group == "all"
        else [args.group]
    )
    for g in groups:
        render_group(g, args.timeout, args.project, project_dir=args.project_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
