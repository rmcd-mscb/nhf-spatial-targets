"""GLDAS-2.1 NOAH monthly runoff adapter (Qs_acc, Qsb_acc, runoff_total)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from nhf_spatial_targets.aggregate._adapter import SourceAdapter
from nhf_spatial_targets.aggregate._driver import aggregate_source

_SOURCE_KEY = "gldas_noah_v21_monthly"


def _derive_runoff_total(ds: xr.Dataset) -> xr.Dataset:
    """Add ``runoff_total = Qs_acc + Qsb_acc`` to the dataset."""
    total = ds["Qs_acc"] + ds["Qsb_acc"]
    total.attrs = {
        "long_name": "total runoff (Qs_acc + Qsb_acc, derived)",
        "units": "kg m-2",
        "cell_methods": "time: sum",
        "derived_from": "Qs_acc + Qsb_acc",
    }
    return ds.assign(runoff_total=total)


ADAPTER = SourceAdapter(
    source_key=_SOURCE_KEY,
    output_name="gldas_agg.nc",
    variables=("Qs_acc", "Qsb_acc", "runoff_total"),
    files_glob="gldas_noah_v21_monthly*.nc",
    pre_aggregate_hook=_derive_runoff_total,
)


def aggregate_gldas(
    fabric_path: Path, id_col: str, workdir: Path, batch_size: int = 500
) -> None:
    aggregate_source(
        ADAPTER,
        fabric_path,
        id_col,
        workdir,
        batch_size,
    )
