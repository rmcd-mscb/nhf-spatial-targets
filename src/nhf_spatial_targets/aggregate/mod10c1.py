"""MOD10C1 v061 daily snow-covered area aggregator (CI-masked)."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._driver import (
    aggregate_variables_for_batch,
    compute_or_load_weights,
    load_and_batch_fabric,
    update_manifest,
)
from nhf_spatial_targets.catalog import source as catalog_source
from nhf_spatial_targets.workspace import load as load_project

logger = logging.getLogger(__name__)

_SOURCE_KEY = "mod10c1_v061"
_CI_THRESHOLD = 0.70  # TM 6-B10: keep cells where CI > 0.70
_OUTPUT_NAME = "mod10c1_agg.nc"
_LOW_COVERAGE_WARN_THRESHOLD = 0.10


def _log_low_valid_coverage(combined: xr.Dataset) -> None:
    """Emit a WARNING log line if too many (HRU, time) cells have zero valid area.

    NaN cells are treated as "no data available" and excluded from both
    numerator and denominator — the threshold compares zero-area cells
    against cells with any finite valid-area fraction.
    """
    vaf = combined["valid_area_fraction"]
    n_total = int(vaf.notnull().sum())
    if n_total == 0:
        return
    n_zero = int(((vaf == 0) & vaf.notnull()).sum())
    zero_frac = n_zero / n_total
    if zero_frac > _LOW_COVERAGE_WARN_THRESHOLD:
        logger.warning(
            "mod10c1: %.1f%% of (HRU, time) cells had zero valid-area "
            "after CI>%.2f filter (n=%d of %d finite). Downstream sca "
            "values are NaN for these cells.",
            zero_frac * 100,
            _CI_THRESHOLD,
            n_zero,
            n_total,
        )


def build_masked_source(ds: xr.Dataset) -> xr.Dataset:
    """Derive ``sca``, ``ci``, ``valid_mask`` from raw MOD10C1 variables.

    - ``sca``        = Day_CMG_Snow_Cover / 100, NaN where CI <= 0.70.
    - ``ci``         = Snow_Spatial_QA / 100 (passed through, unmasked).
    - ``valid_mask`` = 1.0 where CI > 0.70, 0.0 otherwise (float so
                       area-weighted mean gives valid-area fraction per HRU).
    """
    ci = ds["Snow_Spatial_QA"] / 100.0
    pass_mask = ci > _CI_THRESHOLD
    sca_raw = ds["Day_CMG_Snow_Cover"] / 100.0
    sca = sca_raw.where(pass_mask)
    valid_mask = pass_mask.astype("float64")

    out = xr.Dataset(
        {"sca": sca, "ci": ci, "valid_mask": valid_mask},
        coords=ds.coords,
    )
    out["sca"].attrs = {"long_name": "fractional snow-covered area", "units": "1"}
    out["ci"].attrs = {
        "long_name": "confidence interval (Snow_Spatial_QA/100)",
        "units": "1",
    }
    out["valid_mask"].attrs = {
        "long_name": "per-cell CI-pass indicator",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }
    return out


def _open(project) -> xr.Dataset:
    raw_dir = project.raw_dir(_SOURCE_KEY)
    ncs = sorted(Path(raw_dir).glob("*_consolidated.nc"))
    if not ncs:
        raise FileNotFoundError(
            f"No consolidated MOD10C1 NC found in {raw_dir}. "
            f"Run 'nhf-targets fetch mod10c1' first."
        )
    datasets = [xr.open_dataset(p) for p in ncs]
    try:
        if len(datasets) == 1:
            loaded = datasets[0].load()
        else:
            combined = xr.concat(datasets, dim="time")
            combined = combined.sortby("time")
            loaded = combined.load()
    finally:
        for d in datasets:
            d.close()
    return loaded


def aggregate_mod10c1(
    fabric_path: Path,
    id_col: str,
    workdir: Path,
    batch_size: int = 500,
) -> xr.Dataset:
    """Aggregate MOD10C1 v061 daily SCA to HRU polygons with CI masking.

    Applies the CI > 0.70 filter at source grid cells (cells below threshold
    become NaN in ``sca``). Writes three variables to
    ``data/aggregated/mod10c1_agg.nc``:

    - ``sca``: CI-masked fractional snow cover (NaN where CI <= 0.70).
    - ``ci``:  raw confidence interval as a fraction (no masking).
    - ``valid_area_fraction``: per-HRU fraction of grid-cell area that
      passed the CI filter on each day.

    Logs a warning if more than 10% of (HRU, time) cells end up with zero
    valid area.
    """
    workdir = Path(workdir)
    project = load_project(workdir)
    meta = catalog_source(_SOURCE_KEY)

    raw = _open(project)
    source_ds = build_masked_source(raw)
    variables = ["sca", "ci", "valid_mask"]

    batched = load_and_batch_fabric(fabric_path, batch_size=batch_size)
    n_batches = int(batched["batch_id"].nunique())
    logger.info("mod10c1: fabric split into %d spatial batches", n_batches)

    datasets: list[xr.Dataset] = []
    for bid in sorted(batched["batch_id"].unique()):
        batch_gdf = batched[batched["batch_id"] == bid].drop(columns=["batch_id"])
        weights = compute_or_load_weights(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            source_var="sca",
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col=id_col,
            source_key=_SOURCE_KEY,
            batch_id=int(bid),
            workdir=workdir,
        )
        ds = aggregate_variables_for_batch(
            batch_gdf=batch_gdf,
            source_ds=source_ds,
            variables=variables,
            source_crs="EPSG:4326",
            x_coord="lon",
            y_coord="lat",
            time_coord="time",
            id_col=id_col,
            weights=weights,
        )
        datasets.append(ds)

    combined = xr.concat(datasets, dim=id_col)
    combined = combined.rename({"valid_mask": "valid_area_fraction"})
    combined["valid_area_fraction"].attrs = {
        "long_name": "fraction of HRU area that passed CI filter",
        "units": "1",
        "ci_threshold": _CI_THRESHOLD,
    }

    _log_low_valid_coverage(combined)

    output_dir = project.aggregated_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _OUTPUT_NAME
    combined.to_netcdf(output_path)
    logger.info("mod10c1: output written to %s", output_path)
    # Load a detached in-memory copy so callers can use the return value safely
    # after the on-disk handle is closed.
    loaded = combined.load()
    combined.close()

    t0 = str(loaded["time"].values[0])[:10]
    t1 = str(loaded["time"].values[-1])[:10]
    update_manifest(
        project=project,
        source_key=_SOURCE_KEY,
        access=meta.get("access", {}),
        period=f"{t0}/{t1}",
        output_file=str(Path("data") / "aggregated" / _OUTPUT_NAME),
        weight_files=[
            str(Path("weights") / f"{_SOURCE_KEY}_batch{i}.csv")
            for i in range(n_batches)
        ],
    )
    return loaded
