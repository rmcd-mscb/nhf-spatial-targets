"""Shared helpers for the inspect_fabric_*.ipynb notebooks.

A sibling of the fabric-inspection notebooks (not packaged into
nhf_spatial_targets), mirroring notebooks/{consolidated,aggregated,targets}/
_helpers.py. Holds the minimum the fabric notebooks need: project-path
discovery, fabric I/O reprojected to CONUS Albers, and the save-figure
helper that populates docs/figures/fabric/<project>/ for slide / docs work
(the <project> subdir is set via _helpers.PROJECT to namespace by fabric).

Notebooks import via (Jupyter puts the notebook's dir on sys.path)::

    from _helpers import load_project_paths, load_fabric, save_figure
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import yaml

SAVE_FIGURES: bool = False
FIGURES_DIR: Path = Path("docs/figures/fabric/")
PROJECT: str | None = None

ALBERS_CRS = "EPSG:5070"  # matches the aggregator's WEIGHT_GEN_CRS

DEFAULT_CALDERA_PROJECT = Path(
    "/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets"
)


def load_project_paths(
    project_dir: Path | None = None,
) -> tuple[Path, Path, dict]:
    """Read ``<project>/config.yml`` and return ``(project_dir, datastore_dir, fabric_cfg)``.

    ``fabric_cfg`` is the ``fabric`` sub-block from ``config.yml`` (keys
    typically include ``path``, ``id_col``, ``crs``, ``buffer_deg``).
    Defaults ``project_dir`` to the caldera ``gfv2-spatial-targets`` project
    when called with ``None``. Mirrors the aggregated/_helpers.py loader so
    the ``--project-dir`` repoint in scripts/render_figures.py works the
    same way for the fabric group.
    """
    project_dir = (
        Path(project_dir) if project_dir is not None else DEFAULT_CALDERA_PROJECT
    )
    cfg_path = project_dir / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.yml not found at {cfg_path}. "
            f"Edit PROJECT_DIR at the top of the notebook to point at "
            f"a real project directory."
        )
    cfg = yaml.safe_load(cfg_path.read_text())
    datastore_dir = Path(cfg["datastore"])
    fabric_cfg = dict(cfg["fabric"])
    return project_dir, datastore_dir, fabric_cfg


def load_fabric(fabric_cfg: dict) -> gpd.GeoDataFrame:
    """Read the HRU fabric file and reproject to EPSG:5070 (CONUS Albers).

    Unlike the aggregated helper (which keeps EPSG:4326 for lon/lat
    choropleths), the batch-partition map plots projected centroids in
    metres, so this reprojects to Albers up front. The reprojection also
    makes the figure's ``x [m]`` / ``y [m]`` axes correct regardless of the
    fabric's native CRS, so the same notebook renders sensibly for any
    project's fabric.

    Dispatches on the file suffix to mirror :mod:`validate`: parquet/
    geoparquet use the geopandas-native ``read_parquet`` (works even when
    the GDAL build lacks the ``ogr_Parquet`` plugin); anything else
    (.gpkg, .shp, …) goes through ``read_file``/pyogrio.
    """
    path = Path(fabric_cfg["path"])
    if path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    return gdf.to_crs(ALBERS_CRS)


def save_figure(fig, name: str) -> None:
    """Write ``fig`` to ``FIGURES_DIR[/PROJECT]/<name>.png`` iff ``SAVE_FIGURES``.

    No-op when ``SAVE_FIGURES`` is ``False`` (the default). Notebooks enable
    saving by setting ``_helpers.SAVE_FIGURES = True`` near the top before any
    plotting cell runs; scripts/render_figures.py sets it via PYTHONSTARTUP for
    headless renders.

    When ``PROJECT`` is set (notebooks should set
    ``_helpers.PROJECT = PROJECT_DIR.name`` so figures from different fabrics
    stay separate), figures land under ``FIGURES_DIR / PROJECT / <name>.png``.
    With ``PROJECT = None`` they land directly in ``FIGURES_DIR`` — fine for
    ad-hoc local work, but commits should always set ``PROJECT``.

    Relative ``FIGURES_DIR`` paths resolve against the repo root, three parents
    up from this file (``fabric/_helpers.py`` -> ``notebooks/`` -> ``<repo>``).
    Absolute paths (render_figures overrides, pytest tmp_path) are honored
    as-is.
    """
    if not SAVE_FIGURES:
        return
    if not PROJECT:
        warnings.warn(
            "save_figure: SAVE_FIGURES is True but PROJECT is unset. "
            "Figures will land directly in FIGURES_DIR with no project subdir, "
            "risking collision with other fabrics' figures. "
            "Set _helpers.PROJECT = PROJECT_DIR.name to namespace by project.",
            stacklevel=2,
        )
    target_dir = FIGURES_DIR
    if not target_dir.is_absolute():
        target_dir = Path(__file__).resolve().parent.parent.parent / target_dir
    if PROJECT:
        target_dir = target_dir / PROJECT
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{name}.png", dpi=150, bbox_inches="tight")
