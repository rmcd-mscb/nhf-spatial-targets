# Inspect-Consolidated Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four new per-target notebooks (runoff, aet, recharge, snow_covered_area) and rename the existing soil-moisture notebook so each calibration target has a notebook that visualizes one example timestamp from every consolidated source feeding it.

**Architecture:** Each notebook follows the existing `inspect_consolidated.ipynb` template — load → display repr → plot N-panel grid → close. Local NCs are opened with `xr.open_dataset`; SSEBop's remote zarr is opened with `xr.open_zarr` against the USGS GDP STAC OSN endpoint. The recharge notebook sums monthly sources to annual on the fly, since the recharge target time-step is annual.

**Tech Stack:** Jupyter `.ipynb` (nbformat 4.5), xarray, matplotlib, fsspec/s3fs (for the SSEBop zarr open).

**Spec:** `docs/superpowers/specs/2026-04-25-inspect-consolidated-notebooks-design.md`

---

## File Structure

| Path | Action | Purpose |
|---|---|---|
| `notebooks/inspect_consolidated.ipynb` | rename → `inspect_consolidated_soil_moisture.ipynb` | Existing soil-moisture template; preserved verbatim. |
| `notebooks/inspect_consolidated_runoff.ipynb` | create | ERA5-Land `ro` + GLDAS `runoff_total` at 2000-01-15. |
| `notebooks/inspect_consolidated_aet.ipynb` | create | MOD16A2 v061 `ET_500m` + SSEBop `actual_et` (remote zarr) at 2000-01-15. |
| `notebooks/inspect_consolidated_recharge.ipynb` | create | Reitz 2017 (annual) + WaterGAP 2.2d + ERA5 ssro (monthly→annual) for year 2000. |
| `notebooks/inspect_consolidated_snow_covered_area.ipynb` | create | MOD10C1 v061 `Day_CMG_Snow_Cover` + `Snow_Spatial_QA` at 2000-03-01. |

All notebooks use the `geoenv` kernel (matching the existing notebook's `metadata.kernelspec`).

---

## Task 1: Rename existing notebook to `inspect_consolidated_soil_moisture.ipynb`

**Files:**
- Rename: `notebooks/inspect_consolidated.ipynb` → `notebooks/inspect_consolidated_soil_moisture.ipynb`

- [ ] **Step 1: Rename via `git mv` to preserve history**

```bash
git mv notebooks/inspect_consolidated.ipynb notebooks/inspect_consolidated_soil_moisture.ipynb
```

- [ ] **Step 2: Verify the rename**

```bash
git status
ls notebooks/inspect_consolidated*.ipynb
```

Expected: `R  notebooks/inspect_consolidated.ipynb -> notebooks/inspect_consolidated_soil_moisture.ipynb` and only the new filename exists.

- [ ] **Step 3: Commit**

```bash
pixi run git commit -m "$(cat <<'EOF'
chore: rename inspect_consolidated.ipynb to inspect_consolidated_soil_moisture.ipynb

Frees the bare name for new per-target inspect notebooks (runoff, aet,
recharge, snow_covered_area).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Verify WaterGAP 2.2d consolidated variable name

The spec flags this as an open question: `catalog/sources.yml` says the file variable is `qrdif`, but the consolidated CF file may have renamed it to `groundwater_recharge`. Confirm before writing the recharge notebook.

**Files:**
- Read-only: `src/nhf_spatial_targets/fetch/pangaea.py`, `catalog/sources.yml` (already checked).

- [ ] **Step 1: Inspect the consolidate path in the watergap fetch module**

Run:

```bash
grep -n "qrdif\|groundwater_recharge\|rename\|.rename(" src/nhf_spatial_targets/fetch/pangaea.py
```

Look at the lines where the dataset is written. The variable that ends up in `watergap22d_qrdif_cf.nc` is the one to use in the recharge notebook.

- [ ] **Step 2: Record the answer**

If the file variable is `qrdif`, the recharge notebook uses `var: "qrdif"` (matching the plan below). If it's `groundwater_recharge`, the recharge notebook task below must be updated to use that name. Note the result in your scratch space; no commit yet.

---

## Task 3: Create `inspect_consolidated_runoff.ipynb`

**Files:**
- Create: `notebooks/inspect_consolidated_runoff.ipynb`

- [ ] **Step 1: Write the notebook**

Write the file with the following exact content. Two sources (ERA5-Land `ro`, GLDAS `runoff_total`), 1×2 plot, `TARGET_TIME = "2000-01-15"`.

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Inspect Consolidated Runoff Datasets\n",
    "\n",
    "Load and visualize the two runoff source datasets for January 2000.\n",
    "\n",
    "Sources (see `catalog/variables.yml` → `runoff`):\n",
    "- ERA5-Land total runoff (`ro`, m/month)\n",
    "- GLDAS-2.1 NOAH total runoff (`runoff_total = Qs_acc + Qsb_acc`, kg m-2 ≡ mm/month)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import xarray as xr\n",
    "\n",
    "DATASTORE = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore\")\n",
    "PROJECT = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets\")\n",
    "\n",
    "TARGET_TIME = \"2000-01-15\"\n",
    "TARGET_YEAR = 2000"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Load both datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "datasets = {\n",
    "    \"ERA5-Land (ro)\": {\n",
    "        \"path\": DATASTORE / \"era5_land\" / f\"era5_land_monthly_{TARGET_YEAR}.nc\",\n",
    "        \"var\": \"ro\",\n",
    "        \"units\": \"m/month\",\n",
    "    },\n",
    "    \"GLDAS-2.1 NOAH (runoff_total)\": {\n",
    "        \"path\": DATASTORE / \"gldas_noah_v21_monthly\" / \"gldas_noah_v21_monthly.nc\",\n",
    "        \"var\": \"runoff_total\",\n",
    "        \"units\": \"kg m-2 (mm/month)\",\n",
    "    },\n",
    "}\n",
    "\n",
    "opened = {}\n",
    "for label, info in datasets.items():\n",
    "    nc_path = info[\"path\"]\n",
    "    if not nc_path.exists():\n",
    "        print(f\"SKIP {label}: {nc_path} not found (run fetch first)\")\n",
    "        continue\n",
    "    ds = xr.open_dataset(nc_path)\n",
    "    opened[label] = (ds, info)\n",
    "    print(f\"{label}: {list(ds.data_vars)} | time: {ds.time.values[0]} .. {ds.time.values[-1]} | shape: {dict(ds.sizes)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## Dataset representations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    print(f\"{'=' * 60}\\n{label}\\n{'=' * 60}\")\n",
    "    display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6",
   "metadata": {},
   "source": [
    "## Plot January 2000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "n = len(opened)\n",
    "if n == 0:\n",
    "    print(\"No datasets available yet. Run the fetch commands first.\")\n",
    "else:\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    axes = axes.flatten() if n > 1 else [axes]\n",
    "\n",
    "    for idx, (label, (ds, info)) in enumerate(opened.items()):\n",
    "        ax = axes[idx]\n",
    "        var = info[\"var\"]\n",
    "        da = ds[var].sel(time=TARGET_TIME, method=\"nearest\")\n",
    "        actual_time = str(da.time.values)[:10]\n",
    "\n",
    "        da.plot(ax=ax, cmap=\"YlGnBu\", robust=True)\n",
    "        ax.set_title(f\"{label}\\n{actual_time} | {info['units']}\", fontsize=11)\n",
    "        ax.set_xlabel(\"Longitude\")\n",
    "        ax.set_ylabel(\"Latitude\")\n",
    "\n",
    "    for idx in range(len(opened), len(axes)):\n",
    "        axes[idx].set_visible(False)\n",
    "\n",
    "    fig.suptitle(f\"Runoff — nearest to {TARGET_TIME}\", fontsize=14, y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "## Clean up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    ds.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "geoenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Sanity-check JSON parses**

Run:

```bash
python -c "import json; json.load(open('notebooks/inspect_consolidated_runoff.ipynb'))"
```

Expected: no output (clean JSON).

- [ ] **Step 3: Commit**

```bash
git add notebooks/inspect_consolidated_runoff.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add inspect_consolidated_runoff.ipynb

Visualizes ERA5-Land ro and GLDAS-2.1 NOAH runoff_total at 2000-01-15
to verify the consolidated NCs feeding the runoff calibration target.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create `inspect_consolidated_aet.ipynb`

**Files:**
- Create: `notebooks/inspect_consolidated_aet.ipynb`

- [ ] **Step 1: Write the notebook**

Two sources: MOD16A2 v061 (per-year local NC) and SSEBop (remote zarr at USGS GDP STAC OSN endpoint). The load loop branches on whether `zarr_url` is present.

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Inspect Consolidated AET Datasets\n",
    "\n",
    "Load and visualize the two AET source datasets for January 2000.\n",
    "\n",
    "Sources (see `catalog/variables.yml` → `aet`):\n",
    "- MOD16A2 v061 (`ET_500m`, kg m-2 per 8-day composite) — local consolidated per-year NC\n",
    "- SSEBop (`actual_et`, mm/month) — remote zarr at the USGS GDP STAC OSN endpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import xarray as xr\n",
    "\n",
    "DATASTORE = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore\")\n",
    "PROJECT = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets\")\n",
    "\n",
    "TARGET_TIME = \"2000-01-15\"\n",
    "TARGET_YEAR = 2000"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Load both datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "datasets = {\n",
    "    \"MOD16A2 v061 (ET_500m)\": {\n",
    "        \"path\": DATASTORE / \"mod16a2_v061\" / f\"mod16a2_v061_{TARGET_YEAR}_consolidated.nc\",\n",
    "        \"var\": \"ET_500m\",\n",
    "        \"units\": \"kg m-2 / 8-day\",\n",
    "    },\n",
    "    \"SSEBop (actual_et)\": {\n",
    "        \"zarr_url\": \"s3://mdmf/gdp/ssebopeta_monthly.zarr/\",\n",
    "        \"storage_options\": {\n",
    "            \"anon\": True,\n",
    "            \"endpoint_url\": \"https://usgs.osn.mghpcc.org/\",\n",
    "        },\n",
    "        \"var\": \"actual_et\",\n",
    "        \"units\": \"mm/month\",\n",
    "    },\n",
    "}\n",
    "\n",
    "\n",
    "def _open(info):\n",
    "    if \"zarr_url\" in info:\n",
    "        return xr.open_zarr(\n",
    "            info[\"zarr_url\"],\n",
    "            storage_options=info.get(\"storage_options\", {}),\n",
    "            consolidated=True,\n",
    "        )\n",
    "    return xr.open_dataset(info[\"path\"])\n",
    "\n",
    "\n",
    "opened = {}\n",
    "for label, info in datasets.items():\n",
    "    try:\n",
    "        if \"path\" in info and not info[\"path\"].exists():\n",
    "            print(f\"SKIP {label}: {info['path']} not found (run fetch first)\")\n",
    "            continue\n",
    "        ds = _open(info)\n",
    "    except Exception as exc:\n",
    "        print(f\"SKIP {label}: open failed: {exc}\")\n",
    "        continue\n",
    "    opened[label] = (ds, info)\n",
    "    print(f\"{label}: {list(ds.data_vars)} | time: {ds.time.values[0]} .. {ds.time.values[-1]} | shape: {dict(ds.sizes)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## Dataset representations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    print(f\"{'=' * 60}\\n{label}\\n{'=' * 60}\")\n",
    "    display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6",
   "metadata": {},
   "source": [
    "## Plot January 2000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "n = len(opened)\n",
    "if n == 0:\n",
    "    print(\"No datasets available yet. Run the fetch commands first.\")\n",
    "else:\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    axes = axes.flatten() if n > 1 else [axes]\n",
    "\n",
    "    for idx, (label, (ds, info)) in enumerate(opened.items()):\n",
    "        ax = axes[idx]\n",
    "        var = info[\"var\"]\n",
    "        da = ds[var].sel(time=TARGET_TIME, method=\"nearest\")\n",
    "        actual_time = str(da.time.values)[:10]\n",
    "\n",
    "        da.plot(ax=ax, cmap=\"YlOrRd\", robust=True)\n",
    "        ax.set_title(f\"{label}\\n{actual_time} | {info['units']}\", fontsize=11)\n",
    "        ax.set_xlabel(\"Longitude\")\n",
    "        ax.set_ylabel(\"Latitude\")\n",
    "\n",
    "    for idx in range(len(opened), len(axes)):\n",
    "        axes[idx].set_visible(False)\n",
    "\n",
    "    fig.suptitle(f\"AET — nearest to {TARGET_TIME}\", fontsize=14, y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "## Clean up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    ds.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "geoenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Sanity-check JSON parses**

```bash
python -c "import json; json.load(open('notebooks/inspect_consolidated_aet.ipynb'))"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add notebooks/inspect_consolidated_aet.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add inspect_consolidated_aet.ipynb

Visualizes MOD16A2 v061 ET_500m and SSEBop actual_et at 2000-01-15.
SSEBop is opened from its remote zarr at the USGS GDP STAC OSN endpoint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Create `inspect_consolidated_recharge.ipynb`

**Files:**
- Create: `notebooks/inspect_consolidated_recharge.ipynb`

This notebook plots three sources at the recharge target's annual time-step. Reitz is already annual; WaterGAP and ERA5 ssro are monthly and get summed to annual on the fly.

If Task 2 found that the WaterGAP CF file uses a variable name other than `qrdif`, replace `"var": "qrdif"` in the JSON below before writing the file.

- [ ] **Step 1: Write the notebook**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Inspect Consolidated Recharge Datasets\n",
    "\n",
    "Load and visualize the three recharge source datasets at the annual time-step for year 2000.\n",
    "\n",
    "Sources (see `catalog/variables.yml` → `recharge`):\n",
    "- Reitz 2017 (`total_recharge`, inches/year) — already annual\n",
    "- WaterGAP 2.2d (`qrdif`, kg m-2 s-1) — monthly, summed to annual on the fly\n",
    "- ERA5-Land (`ssro`, m/month) — monthly, summed to annual on the fly\n",
    "\n",
    "Note: monthly sources are summed to annual here so issues in any one month surface in the annual cell. The actual recharge target builder does the same aggregation downstream."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import xarray as xr\n",
    "\n",
    "DATASTORE = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore\")\n",
    "PROJECT = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets\")\n",
    "\n",
    "TARGET_YEAR = 2000"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Load all three datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "datasets = {\n",
    "    \"Reitz 2017 (total_recharge)\": {\n",
    "        \"path\": DATASTORE / \"reitz2017\" / \"reitz2017_consolidated.nc\",\n",
    "        \"var\": \"total_recharge\",\n",
    "        \"time_step\": \"annual\",\n",
    "        \"units\": \"inches/year\",\n",
    "    },\n",
    "    \"WaterGAP 2.2d (qrdif)\": {\n",
    "        \"path\": DATASTORE / \"watergap22d\" / \"watergap22d_qrdif_cf.nc\",\n",
    "        \"var\": \"qrdif\",\n",
    "        \"time_step\": \"monthly\",\n",
    "        \"units\": \"kg m-2 s-1 (summed to annual)\",\n",
    "    },\n",
    "    \"ERA5-Land (ssro)\": {\n",
    "        \"path\": DATASTORE / \"era5_land\" / f\"era5_land_monthly_{TARGET_YEAR}.nc\",\n",
    "        \"var\": \"ssro\",\n",
    "        \"time_step\": \"monthly\",\n",
    "        \"units\": \"m (summed to annual)\",\n",
    "    },\n",
    "}\n",
    "\n",
    "opened = {}\n",
    "for label, info in datasets.items():\n",
    "    nc_path = info[\"path\"]\n",
    "    if not nc_path.exists():\n",
    "        print(f\"SKIP {label}: {nc_path} not found (run fetch first)\")\n",
    "        continue\n",
    "    ds = xr.open_dataset(nc_path)\n",
    "    opened[label] = (ds, info)\n",
    "    print(f\"{label}: {list(ds.data_vars)} | time: {ds.time.values[0]} .. {ds.time.values[-1]} | shape: {dict(ds.sizes)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## Dataset representations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    print(f\"{'=' * 60}\\n{label}\\n{'=' * 60}\")\n",
    "    display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6",
   "metadata": {},
   "source": [
    "## Plot annual values for target year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "n = len(opened)\n",
    "if n == 0:\n",
    "    print(\"No datasets available yet. Run the fetch commands first.\")\n",
    "else:\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n",
    "    axes = axes.flatten() if n > 1 else [axes]\n",
    "\n",
    "    for idx, (label, (ds, info)) in enumerate(opened.items()):\n",
    "        ax = axes[idx]\n",
    "        var = info[\"var\"]\n",
    "        if info[\"time_step\"] == \"monthly\":\n",
    "            da = ds[var].sel(\n",
    "                time=slice(f\"{TARGET_YEAR}-01\", f\"{TARGET_YEAR}-12\")\n",
    "            ).sum(\"time\")\n",
    "            label_time = f\"{TARGET_YEAR} (sum of 12 months)\"\n",
    "        else:\n",
    "            da = ds[var].sel(time=str(TARGET_YEAR), method=\"nearest\")\n",
    "            label_time = str(da.time.values)[:10] if da.time.size else str(TARGET_YEAR)\n",
    "\n",
    "        da.plot(ax=ax, cmap=\"BuGn\", robust=True)\n",
    "        ax.set_title(f\"{label}\\n{label_time} | {info['units']}\", fontsize=11)\n",
    "        ax.set_xlabel(\"Longitude\")\n",
    "        ax.set_ylabel(\"Latitude\")\n",
    "\n",
    "    for idx in range(len(opened), len(axes)):\n",
    "        axes[idx].set_visible(False)\n",
    "\n",
    "    fig.suptitle(f\"Recharge — annual values for {TARGET_YEAR}\", fontsize=14, y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "## Clean up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    ds.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "geoenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Sanity-check JSON parses**

```bash
python -c "import json; json.load(open('notebooks/inspect_consolidated_recharge.ipynb'))"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add notebooks/inspect_consolidated_recharge.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add inspect_consolidated_recharge.ipynb

Visualizes Reitz 2017, WaterGAP 2.2d, and ERA5-Land ssro at the recharge
target's annual time-step (year 2000). Monthly sources are summed to
annual on the fly so per-month issues surface in the annual cell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Create `inspect_consolidated_snow_covered_area.ipynb`

**Files:**
- Create: `notebooks/inspect_consolidated_snow_covered_area.ipynb`

One source (MOD10C1 v061), two variables of interest plotted side-by-side: `Day_CMG_Snow_Cover` and `Snow_Spatial_QA`. The same per-year consolidated NC is opened once and both variables are pulled from `opened[label][0]`. We register the same dataset under two labels so the existing template loop works without special-casing.

- [ ] **Step 1: Write the notebook**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Inspect Consolidated Snow-Covered Area Dataset\n",
    "\n",
    "Load and visualize the MOD10C1 v061 daily SCA fields for early March 2000.\n",
    "\n",
    "Sources (see `catalog/variables.yml` → `snow_covered_area`):\n",
    "- MOD10C1 v061 `Day_CMG_Snow_Cover` (percent, 0-100)\n",
    "- MOD10C1 v061 `Snow_Spatial_QA` (percent, 0-100; CI threshold 70 applied downstream)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import xarray as xr\n",
    "\n",
    "DATASTORE = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore\")\n",
    "PROJECT = Path(\"/caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2-spatial-targets\")\n",
    "\n",
    "TARGET_TIME = \"2000-03-01\"\n",
    "TARGET_YEAR = 2000"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Load the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "mod10c1_path = DATASTORE / \"mod10c1_v061\" / f\"mod10c1_v061_{TARGET_YEAR}_consolidated.nc\"\n",
    "\n",
    "datasets = {\n",
    "    \"MOD10C1 v061 (Day_CMG_Snow_Cover)\": {\n",
    "        \"path\": mod10c1_path,\n",
    "        \"var\": \"Day_CMG_Snow_Cover\",\n",
    "        \"units\": \"percent (0-100)\",\n",
    "        \"cmap\": \"Blues\",\n",
    "    },\n",
    "    \"MOD10C1 v061 (Snow_Spatial_QA)\": {\n",
    "        \"path\": mod10c1_path,\n",
    "        \"var\": \"Snow_Spatial_QA\",\n",
    "        \"units\": \"percent (0-100)\",\n",
    "        \"cmap\": \"viridis\",\n",
    "    },\n",
    "}\n",
    "\n",
    "opened = {}\n",
    "for label, info in datasets.items():\n",
    "    nc_path = info[\"path\"]\n",
    "    if not nc_path.exists():\n",
    "        print(f\"SKIP {label}: {nc_path} not found (run fetch first)\")\n",
    "        continue\n",
    "    ds = xr.open_dataset(nc_path)\n",
    "    opened[label] = (ds, info)\n",
    "    print(f\"{label}: {list(ds.data_vars)} | time: {ds.time.values[0]} .. {ds.time.values[-1]} | shape: {dict(ds.sizes)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## Dataset representations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    print(f\"{'=' * 60}\\n{label}\\n{'=' * 60}\")\n",
    "    display(ds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6",
   "metadata": {},
   "source": [
    "## Plot March 1, 2000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7",
   "metadata": {},
   "outputs": [],
   "source": [
    "n = len(opened)\n",
    "if n == 0:\n",
    "    print(\"No datasets available yet. Run the fetch commands first.\")\n",
    "else:\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    axes = axes.flatten() if n > 1 else [axes]\n",
    "\n",
    "    for idx, (label, (ds, info)) in enumerate(opened.items()):\n",
    "        ax = axes[idx]\n",
    "        var = info[\"var\"]\n",
    "        da = ds[var].sel(time=TARGET_TIME, method=\"nearest\")\n",
    "        actual_time = str(da.time.values)[:10]\n",
    "\n",
    "        da.plot(ax=ax, cmap=info.get(\"cmap\", \"viridis\"), robust=True)\n",
    "        ax.set_title(f\"{label}\\n{actual_time} | {info['units']}\", fontsize=11)\n",
    "        ax.set_xlabel(\"Longitude\")\n",
    "        ax.set_ylabel(\"Latitude\")\n",
    "\n",
    "    for idx in range(len(opened), len(axes)):\n",
    "        axes[idx].set_visible(False)\n",
    "\n",
    "    fig.suptitle(f\"Snow-Covered Area — nearest to {TARGET_TIME}\", fontsize=14, y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8",
   "metadata": {},
   "source": [
    "## Clean up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9",
   "metadata": {},
   "outputs": [],
   "source": [
    "for label, (ds, _) in opened.items():\n",
    "    ds.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "geoenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Sanity-check JSON parses**

```bash
python -c "import json; json.load(open('notebooks/inspect_consolidated_snow_covered_area.ipynb'))"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add notebooks/inspect_consolidated_snow_covered_area.ipynb
pixi run git commit -m "$(cat <<'EOF'
docs(notebooks): add inspect_consolidated_snow_covered_area.ipynb

Visualizes MOD10C1 v061 Day_CMG_Snow_Cover and Snow_Spatial_QA at
2000-03-01 to verify the per-year consolidated NCs feeding the
snow-covered-area calibration target.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification

- [ ] **Step 1: List all inspect notebooks**

```bash
ls -1 notebooks/inspect_consolidated*.ipynb
```

Expected:
```
notebooks/inspect_consolidated_aet.ipynb
notebooks/inspect_consolidated_recharge.ipynb
notebooks/inspect_consolidated_runoff.ipynb
notebooks/inspect_consolidated_snow_covered_area.ipynb
notebooks/inspect_consolidated_soil_moisture.ipynb
```

- [ ] **Step 2: Confirm git log**

```bash
git log --oneline main..HEAD
```

Expected: 5 commits (rename + 4 new notebooks). Spec commit `bd27964` is already on main.

- [ ] **Step 3: Confirm no `inspect_consolidated.ipynb` left**

```bash
find notebooks -name "inspect_consolidated.ipynb"
```

Expected: empty output.

---

## Notes for the executor

- **Notebooks are not executed by CI.** A successful JSON parse + matching cell structure is the bar; running cells against the live datastore is left for a human reviewer with the data mounted.
- **Don't strip outputs** — the notebooks ship with empty `outputs: []` and `execution_count: null` already; the project's `nbstripout` pre-commit hook will leave them alone.
- **No new dependencies** — `xarray`, `matplotlib`, `s3fs`/`zarr` are already in the pixi environment for SSEBop access (see `src/nhf_spatial_targets/aggregate/ssebop.py`).
- **If WaterGAP variable name differs from `qrdif`** (Task 2), update the `var` field in Task 5's JSON before writing the file. No other downstream change is needed.
