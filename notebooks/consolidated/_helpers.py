"""Shared helpers for the inspect_consolidated_*.ipynb notebooks.

Sibling of the notebooks (not packaged into nhf_spatial_targets). Today
the only shared concern is a save-figure helper that populates
``docs/figures/consolidated/`` for downstream slide / documentation
work; the notebooks themselves still own their dataset-opening logic
because each consolidated source has its own quirks (zarr vs NC,
8-day vs monthly, etc.).

Notebooks import via:

    from _helpers import save_figure
    import _helpers
    _helpers.SAVE_FIGURES = True            # opt in to writing PNGs
    _helpers.PROJECT = PROJECT.name         # namespace by project dir
"""

from __future__ import annotations

from pathlib import Path

SAVE_FIGURES: bool = False
FIGURES_DIR: Path = Path("docs/figures/consolidated/")
PROJECT: str | None = None


def save_figure(fig, name: str) -> None:
    """Write ``fig`` to ``FIGURES_DIR[/PROJECT]/<name>.png`` iff ``SAVE_FIGURES``.

    No-op when ``SAVE_FIGURES`` is ``False`` (the default). Notebooks
    enable saving by setting ``_helpers.SAVE_FIGURES = True`` near the
    top before any plotting cell runs.

    When ``PROJECT`` is set (notebooks should set
    ``_helpers.PROJECT = PROJECT.name`` so figures from different
    fabrics stay separate), figures land under
    ``FIGURES_DIR / PROJECT / <name>.png``. With ``PROJECT = None``
    figures land directly in ``FIGURES_DIR`` — fine for ad-hoc local
    work, but commits should always set ``PROJECT`` so the deck's
    figure paths resolve unambiguously.

    Relative paths in ``FIGURES_DIR`` are resolved against the repo
    root (this module's grandparent's parent). Absolute paths (user
    overrides, pytest tmp_path) are honored as-is.
    """
    if not SAVE_FIGURES:
        return
    target_dir = FIGURES_DIR
    if not target_dir.is_absolute():
        # _helpers.py lives at <repo>/notebooks/consolidated/_helpers.py;
        # repo root is two parents up from that.
        target_dir = Path(__file__).resolve().parent.parent.parent / target_dir
    if PROJECT:
        target_dir = target_dir / PROJECT
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{name}.png", dpi=150, bbox_inches="tight")
