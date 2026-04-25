# Inspect-Consolidated Notebooks per Target

## Purpose

Provide a per-target Jupyter notebook that visualizes one example timestamp
from every source dataset feeding that target. The intent is human
verification of the consolidated source files we write to the datastore —
catching unit, grid, time, or extent surprises before they propagate into the
aggregation and target-build stages.

## Scope

- One notebook per calibration target (5 total): runoff, aet, recharge,
  soil_moisture, snow_covered_area.
- Each notebook follows the existing `inspect_consolidated.ipynb`
  (soil_moisture) template: load → display repr → plot N-panel grid → close.
- The existing notebook is renamed to `inspect_consolidated_soil_moisture.ipynb`
  and serves as the canonical pattern.

Out of scope: HRU-fabric aggregation, range-method computation, automated
tests for notebooks, CI integration.

## Files

| Notebook | Status | Target | Time-step plotted |
|---|---|---|---|
| `notebooks/inspect_consolidated_soil_moisture.ipynb` | rename from `inspect_consolidated.ipynb` | soil_moisture | `1980-01-15` (unchanged) |
| `notebooks/inspect_consolidated_runoff.ipynb` | new | runoff | `2000-01-15` |
| `notebooks/inspect_consolidated_aet.ipynb` | new | aet | `2000-01-15` |
| `notebooks/inspect_consolidated_recharge.ipynb` | new | recharge | year `2000` (annual) |
| `notebooks/inspect_consolidated_snow_covered_area.ipynb` | new | snow_covered_area | `2000-03-01` |

The `2000` choice for the four new notebooks is the earliest year present in
all sources of each target and avoids the open end of the still-running
ERA5-Land fetch. `TARGET_TIME` (or `TARGET_YEAR` for recharge) is a top-level
constant in each notebook so it can be bumped trivially.

## Common Notebook Structure

Each notebook has the same cell layout, mirroring the existing
soil-moisture template:

1. **Markdown header** — target name, date being plotted, link to
   `catalog/variables.yml` for source rationale.
2. **Setup cell** — imports (`pathlib.Path`, `matplotlib.pyplot`, `xarray`),
   `DATASTORE` and `PROJECT` paths, `TARGET_TIME` (or `TARGET_YEAR`).
3. **`datasets` dict** — one entry per source with the keys needed to open
   and plot it. For local files: `path`, `var`, `units`. For remote zarr
   (ssebop): `zarr_url`, `storage_options`, `var`, `units`. For per-year
   files (ERA5, MOD16A2, MOD10C1): `path` is the specific year's NC.
4. **Load cell** — iterate `datasets`, open each (skip with a printed `SKIP`
   line if the local file is missing), populate an `opened` dict, and print
   a one-line summary (`vars`, `time[0]..time[-1]`, `dict(sizes)`).
5. **Repr cell** — `display(ds)` for each opened dataset, separated by a
   header line, so xarray's HTML repr is visible.
6. **Plot cell** — N-panel grid (N = number of sources), one subplot per
   source. Each subplot does
   `da.sel(time=TARGET_TIME, method="nearest").plot(ax=ax, cmap=..., robust=True)`,
   with the actual selected time and units in the title.
7. **Close cell** — call `ds.close()` for every entry in `opened`.

## Per-Target Details

### runoff (2 sources, 1×2 plot)

```python
TARGET_TIME = "2000-01-15"
datasets = {
    "ERA5-Land (ro)": {
        "path": DATASTORE / "era5_land" / "era5_land_monthly_2000.nc",
        "var": "ro",
        "units": "m",
    },
    "GLDAS-2.1 NOAH (runoff_total)": {
        "path": DATASTORE / "gldas_noah_v21_monthly" / "gldas_noah_v21_monthly.nc",
        "var": "runoff_total",
        "units": "kg m-2 (≡ mm/month)",
    },
}
```

### aet (2 sources, 1×2 plot)

MOD16A2 is a per-year consolidated NC. SSEBop is a remote zarr at the USGS
GDP STAC OSN endpoint — opened via `xr.open_zarr` with anonymous S3 storage
options. We open it with the same `xr.open_dataset`-style entry, but the
loader chooses `xr.open_zarr` when `zarr_url` is present.

```python
TARGET_TIME = "2000-01-15"
datasets = {
    "MOD16A2 v061 (ET_500m)": {
        "path": DATASTORE / "mod16a2_v061" / "mod16a2_v061_2000_consolidated.nc",
        "var": "ET_500m",
        "units": "kg m-2 (per 8-day composite)",
    },
    "SSEBop (actual_et)": {
        "zarr_url": "s3://mdmf/gdp/ssebopeta_monthly.zarr/",
        "storage_options": {
            "anon": True,
            "endpoint_url": "https://usgs.osn.mghpcc.org/",
        },
        "var": "actual_et",
        "units": "mm/month",
    },
}
```

### recharge (3 sources, 1×3 plot)

The recharge target is annual. Reitz is already annual; watergap and ERA5
ssro are monthly. Per the user's call ("if there is an issue with the
monthly data it will be seen in the annual time step"), the notebook sums
monthly→annual on the fly before plotting. A markdown cell above the plot
states this.

```python
TARGET_YEAR = 2000
datasets = {
    "Reitz 2017 (total_recharge)": {
        "path": DATASTORE / "reitz2017" / "reitz2017_consolidated.nc",
        "var": "total_recharge",
        "time_step": "annual",
        "units": "inches/year",
    },
    "WaterGAP 2.2d (qrdif)": {
        "path": DATASTORE / "watergap22d" / "watergap22d_qrdif_cf.nc",
        "var": "qrdif",
        "time_step": "monthly",
        "units": "kg m-2 s-1 (summed to annual)",
    },
    "ERA5-Land (ssro)": {
        "path": DATASTORE / "era5_land" / "era5_land_monthly_2000.nc",
        "var": "ssro",
        "time_step": "monthly",
        "units": "m (summed to annual)",
    },
}

# In the plot cell:
if info["time_step"] == "monthly":
    da = ds[var].sel(
        time=slice(f"{TARGET_YEAR}-01", f"{TARGET_YEAR}-12")
    ).sum("time")
else:
    da = ds[var].sel(time=str(TARGET_YEAR), method="nearest")
```

### soil_moisture (4 sources, 2×2 plot) — existing

Renamed from `inspect_consolidated.ipynb` to
`inspect_consolidated_soil_moisture.ipynb`. No content changes. (Optional
follow-up: bring the markdown header in line with the new notebooks; not
required for this work.)

### snow_covered_area (1 source, 1×2 plot)

Single source with two variables of interest (the SCA value and its CI).
Two side-by-side subplots from the same dataset.

```python
TARGET_TIME = "2000-03-01"
datasets = {
    "MOD10C1 v061 (Day_CMG_Snow_Cover)": {
        "path": DATASTORE / "mod10c1_v061" / "mod10c1_v061_2000_consolidated.nc",
        "var": "Day_CMG_Snow_Cover",
        "units": "percent (0-100)",
    },
    "MOD10C1 v061 (Snow_Spatial_QA)": {
        "path": DATASTORE / "mod10c1_v061" / "mod10c1_v061_2000_consolidated.nc",
        "var": "Snow_Spatial_QA",
        "units": "percent (0-100)",
    },
}
```

## Loader Helper

Each notebook embeds a small inline open function so the load cell stays a
single loop. No new module under `src/` — these are inspection notebooks,
not pipeline code.

```python
def _open(info):
    if "zarr_url" in info:
        return xr.open_zarr(
            info["zarr_url"],
            storage_options=info.get("storage_options", {}),
            consolidated=True,
        )
    return xr.open_dataset(info["path"])
```

## File Hygiene

- Each opened dataset is closed in the final cell (`ds.close()` per entry
  in `opened`). This matches the existing template and the project rule
  about explicit close of disk-backed xarray objects.
- Missing local files print a `SKIP <label>: <path> not found` line and
  the load loop continues — partial inspection is still useful when a
  source hasn't been fetched yet.
- The SSEBop remote open requires network; if it raises, the notebook
  still produces output for MOD16A2. Wrap the open call in a `try/except`
  inside the load loop and report the failure in the same `SKIP` form.

## Risks and Open Questions

- ERA5-Land fetch is in progress; if `era5_land_monthly_2000.nc` doesn't
  exist when a user runs runoff or recharge, they'll see the `SKIP` line.
  Acceptable — the notebook still verifies the other sources.
- Watergap NC variable name is `qrdif` per `catalog/sources.yml`; if the
  consolidated CF file has been renamed (e.g., to `groundwater_recharge`),
  the notebook will need a one-line update. The plan task should verify
  the actual variable name in the file before finalizing the recharge
  notebook.

## Acceptance

- All five notebooks live under `notebooks/` and run end-to-end on a
  workstation with the datastore mounted at the path set in the setup cell
  (with the caveat that ERA5 2000 may not yet be present).
- The four new notebooks visually match the existing soil-moisture
  notebook in structure and tone.
- `inspect_consolidated.ipynb` no longer exists; `git mv` preserves
  history.
