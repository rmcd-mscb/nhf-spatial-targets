"""GLDAS-2.1 NOAH monthly runoff adapter (Qs_acc, Qsb_acc, runoff_total)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source
from nhf_spatial_targets.workspace import Project

_SOURCE_KEY = "gldas_noah_v21_monthly"


def _open(project: Project) -> xr.Dataset:
    """Open the consolidated GLDAS NC and derive ``runoff_total``."""
    raw_dir = project.raw_dir(_SOURCE_KEY)
    ncs = sorted(Path(raw_dir).glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(
            f"No GLDAS NC found in {raw_dir}. Run 'nhf-targets fetch gldas' first."
        )
    ds = xr.open_dataset(ncs[0])
    total = ds["Qs_acc"] + ds["Qsb_acc"]
    total.attrs = {
        "long_name": "total runoff (Qs_acc + Qsb_acc, derived)",
        "units": "kg m-2",
        "cell_methods": "time: sum",
        "derived_from": "Qs_acc + Qsb_acc",
    }
    ds["runoff_total"] = total
    return ds


ADAPTER = SourceAdapter(
    source_key=_SOURCE_KEY,
    output_name="gldas_agg.nc",
    variables=["Qs_acc", "Qsb_acc", "runoff_total"],
    open_hook=_open,
)


def aggregate_gldas(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> xr.Dataset:
    """Aggregate GLDAS-2.1 NOAH monthly runoff variables to HRU polygons."""
    return aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
