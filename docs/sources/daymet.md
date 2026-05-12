# Daymet V4 R1 (operator-staged zarr)

The ORNL DAAC Daymet V4 R1 daily 1 km gridded surface weather product
(Thornton et al., 2022; doi:10.3334/ORNLDAAC/2129) is distributed as
three regional zarr stores covering North America (`na`), Hawaii
(`hi`), and Puerto Rico (`pr`). Each regional zarr exposes
`prcp, srad, swe, tmax, tmin, vp` plus the `lambert_conformal_conic`
CRS spec. Only `swe` is consumed by the SWE calibration target;
the other variables are documented for completeness.

The publisher distributes the zarrs in hundreds-of-GB per region.
We do **not** download them — the operator pre-stages the zarrs on a
shared filesystem and `nhf-targets fetch daymet` fingerprints each
region (sha256 over `.zgroup` / `.zarray` / `.zattrs` / `.zmetadata`
files plus the directory's total byte count) and writes a per-region
manifest entry. The fingerprint design and its residual gaps are
documented in [`src/nhf_spatial_targets/fetch/daymet.py`](../../src/nhf_spatial_targets/fetch/daymet.py).

## Procedure

1. Obtain the three regional zarrs from the publisher
   <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2129> and stage
   them on a shared filesystem location, e.g.:

   ```
   /<shared>/daymet/source/zarr/
     daymet_na.zarr/
     daymet_hi.zarr/
     daymet_pr.zarr/
   ```

   Operators on the Hovenweep HPC cluster can point at the existing
   `/caldera/hovenweep/projects/usgs/water/impd/nhgf/data_creation/daymet/source/zarr`
   tree rather than re-staging.

2. Configure the project to find the zarrs. Either:

   - Add `daymet_root: /<shared>/daymet/source/zarr` to `config.yml`, **or**
   - Pass `--source-path /<shared>/daymet/source/zarr` on the CLI.

   The `--source-path` flag wins if both are present.

3. Run the fetch:

   ```bash
   nhf-targets fetch daymet --project-dir <project> --period 1980/2024
   ```

   Optional flags:

   - `--region {na|hi|pr|all}` (default `all`). Use `--region na` for
     CONUS-only fabrics.

   The command verifies each present zarr opens cleanly, asserts the
   required variables are exposed, fingerprints the structural
   metadata, and records per-region entries under
   `manifest.json` → `sources.daymet.regions.{na,hi,pr}`. Subsequent
   runs against unchanged zarrs are a no-op (matching fingerprint =
   skip re-validation).

4. Parallel runs across regions are safe: the manifest update takes an
   exclusive `flock` on a `manifest.json.lock` sibling, so
   `fetch daymet --region na` and `fetch daymet --region hi` can run
   concurrently without losing each other's records.

## Troubleshooting

- `FileNotFoundError: Daymet zarr root is not configured` — either
  `--source-path` was not passed or `daymet_root:` is missing from
  `config.yml`. Fix one and re-run.

- `RuntimeError: <zroot> is missing required variables [...]` — the
  staged zarr is incomplete. Re-stage from the publisher.

- `RuntimeError: <zroot> is being modified concurrently` — another
  process is writing to the zarr while fetch tries to fingerprint it.
  Wait for the writer to finish, then re-run.

- Fingerprint residual risk: a chunk file overwritten in place with
  identical compressed length but different bytes is invisible to the
  structural-metadata sha256 + directory byte count. Re-stage from
  the publisher if chunk-level tampering is suspected.
