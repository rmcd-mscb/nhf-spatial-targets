# MWBM ClimGrid (manual download)

The USGS Monthly Water Balance Model output forced by NOAA ClimGrid
(Wieczorek et al., 2024; doi:10.5066/P9QCLGKM) is distributed by
ScienceBase as a single ~7.5 GB CF-conformant NetCDF,
`ClimGrid_WBM.nc`. The publisher gates this file behind a CAPTCHA
("I'm not a robot") prompt, so `sciencebasepy` and other automated
clients cannot download it. The pipeline therefore expects the file
to be **manually placed** in the project's datastore before
`nhf-targets fetch mwbm-climgrid` is invoked. The fetch command then
fingerprints (sha256 + size), validates CF metadata, and records
provenance in `manifest.json`.

## Procedure

1. Open the ScienceBase item page in a browser:
   <https://www.sciencebase.gov/catalog/item/64c948dbd34e70357a34c11e>
2. Click `ClimGrid_WBM.nc` in the **Attached Files** section. Complete
   the CAPTCHA when prompted.
3. Wait for the ~7.5 GB download to finish. The file is int16-packed
   with `scale_factor` / `add_offset` (xarray decodes automatically on
   open).
4. Move the file into the project's datastore at exactly:

   ```
   <datastore>/mwbm_climgrid/ClimGrid_WBM.nc
   ```

   where `<datastore>` is the path declared in `config.yml`. Create
   the `mwbm_climgrid/` subdirectory if it does not already exist.

5. Run the registration step. It is idempotent and safe to re-run.

   ```bash
   pixi run nhf-targets fetch mwbm-climgrid \
       --project-dir /data/nhf-runs/my-run \
       --period 1900/2020
   ```

   The first invocation hashes the file (~30–60 s for 7.5 GB on a
   local SSD), validates the four required variables (`runoff`, `aet`,
   `soilstorage`, `swe`) and their `cell_methods`, and writes a
   `mwbm_climgrid` entry to `manifest.json`. Subsequent invocations
   re-verify the fingerprint and short-circuit if it matches.

6. Aggregate to the HRU fabric as usual:

   ```bash
   pixi run nhf-targets agg mwbm-climgrid --project-dir /data/nhf-runs/my-run
   ```

## Sharing across projects

The datastore is fabric-independent (see CLAUDE.md), so the same
`ClimGrid_WBM.nc` can be reused across multiple projects pointing at
the same datastore — copy or symlink it once, then run the fetch
step in each project to register provenance in that project's
`manifest.json`.

## Behaviour when the file is missing

`nhf-targets fetch mwbm-climgrid` raises a `FileNotFoundError`
pointing back at this document. `nhf-targets fetch all` instead skips
mwbm-climgrid with a yellow warning so the rest of the pipeline can
proceed; place the file and re-run that single source separately when
ready.

## Period gate

`--period` is validated against the publisher-usable window
**1900–2020**. The 1895–1899 spinup years are rejected. The file
itself spans the full range; `--period` does not select a subset of
bytes — it gates the call and records the operator-chosen window in
the manifest entry for downstream provenance.

## Catalog reference

See `catalog/sources.yml` (`mwbm_climgrid:`) for the authoritative
declaration of variables, `cell_methods`, units, DOI, and license.
The validator in `src/nhf_spatial_targets/fetch/mwbm_climgrid.py`
checks the on-disk file against that declaration; a divergence (the
publisher reorganises the dataset, or this catalog drifts) raises a
`RuntimeError` with the specific mismatch and instructions to delete
the file before re-running.
