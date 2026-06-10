# USGS data release process

Summary of the USGS data release lifecycle as it applies to publishing
`nhf-spatial-targets` outputs on ScienceBase. Source: the USGS data-release
process pages (linked in [README.md](README.md)), captured 2026-05-26.

## TL;DR

1. **Prepare data and metadata** in open formats (NetCDF, CSV) + FGDC CSDGM 2.0 XML.
2. **Create an IPDS record** (Information Product Data System) — USGS internal approval workflow. **IPDS approval is required before the DOI is minted.**
3. **Create a ScienceBase landing-page item** as the data-release container.
4. **Finalize metadata** — title self-explanatory outside USGS, DOI as full URL in the citation, distribution contact + liability statement attached.
5. **Decide organization** — single item, multiple files on one item, or parent-child (the pattern this repo uses).
6. **Upload files** — data + FGDC XML to ScienceBase.
7. **Format citation** — author names, publication date, DOI URL.
8. **Submit for final review** — ScienceBase team verifies the [checklist](https://www.usgs.gov/sciencebase-instructions-and-documentation/sciencebase-data-release-checklist) (up to 2 business days).

After publication the parent item is **locked** in the sense that the DOI is
permanent, but child items can still be added or replaced. The DOI landing
page's metadata, however, is generated from a snapshot taken at IPDS time and
**does not auto-refresh** when children are added later.

## Roles and approval gates

| Gate | Who | What it verifies |
|---|---|---|
| **IPDS Review** | USGS bureau approving official + science center | Fundamental Science Practices (FSP): data accuracy, methodology, peer review |
| **ScienceBase Review** | ScienceBase staff | Checklist conformance: FGDC validates, files are open formats, citation is well-formed, DOI URL present, distribution liability statement present |
| **DOI Mint** | ScienceBase staff (post-IPDS) | Manual; registers DOI with DataCite as `10.5066/<id>`; not exposed via API |

`sciencebasepy` automates only the ScienceBase upload steps. IPDS submissions
and DOI mint requests are manual.

## ScienceBase item anatomy

ScienceBase is a "certified USGS Trusted Digital Repository" supporting
hierarchical items. For this pipeline:

- **Parent item** — landing page. Carries the DOI, top-level FGDC metadata,
  citation, and (optionally) a README + manifest describing the whole
  collection. No data files at the parent level in our design.
- **Child items** — sub-items linked via `parentId`. Each has its own FGDC
  metadata. Each child may carry data files. Children inherit citation +
  contact info from the parent by convention but maintain their own metadata
  records.

Required parent-item fields at upload time:

| Field | Constraint |
|---|---|
| Title | Meaningful outside USGS context (appears in [USGS Science Data Catalog](https://data.usgs.gov/datacatalog/)) |
| Citation block | Authors + publication date + title + DOI URL in the `<onlink>` element of the citation section |
| Distribution info | Contact organization "U.S. Geological Survey - ScienceBase", contact address/phone/email, liability statement, digital format name, online resource link (the DOI URL), fees (typically "None") |
| Abstract | Describes purpose, scope, and relevance |

## FGDC CSDGM required elements

[FGDC CSDGM 2.0](https://www.fgdc.gov/standards/projects/FGDC-standards-projects/metadata/base-metadata)
is the mandated federal standard for USGS data releases. ISO 19115 is an
acceptable alternative on paper but ScienceBase staff strongly prefer CSDGM.
At minimum:

| FGDC element | Source in this pipeline |
|---|---|
| Title | Generated from `catalog/release_defaults.yml` umbrella template + `release_registry.yml.umbrella.version` |
| Abstract | Per-item template in `release_defaults.yml`, Jinja-rendered with item-specific context |
| Purpose | Per-item template |
| Bounding coordinates | Umbrella: union of children. Source child: `catalog/sources.yml[<key>].access.bbox_nwse`. Fabric child: `<project>/fabric.json.bbox` |
| Time period | Umbrella: union of children. Source child: `catalog/sources.yml[<key>].period`. Fabric child: union of enabled-target periods |
| Lineage / Process steps | `<project>/manifest.json.steps[]`, rendered by `release/mcf.py:_step_description` — including each step's resolved `params` (e.g. the ua_swe aggregate step's `depth_threshold_mm`, see below) |
| Entity / Attribute | Per-NC variable: from `catalog/sources.yml[<key>].variables[].{name,long_name,cf_units,cell_methods}`; per-target: from `catalog/variables.yml[<target>]` |
| Keywords (theme) | `release_defaults.yml.keywords.{umbrella,source,fabric}` + per-target CF standard_names |
| Point of contact | `release_defaults.yml.contacts.point_of_contact` (overridable per project) |
| Distribution liability | `release_defaults.yml.distribution.liability_statement` (USGS standard text) |
| Use constraints | `catalog/sources.yml[<key>].license` + `release_defaults.yml.distribution.use_constraints` + per-source notes |
| Citation Format | `<project>/config.yml.release.authors` + DOI URL from `release_registry.yml.umbrella.doi` |

Validate FGDC XML with the [USGS Metadata Parser](https://geology.usgs.gov/tools/metadata/tools/doc/mp.html)
before upload. Our `release/validate_xml.py` shells out to `mp` if it's on
PATH and reports warn-only if not.

**Aggregate-step parameters travel into the metadata.** When an aggregate
step is defined by a tunable parameter, that parameter is captured in the
manifest step's `params` and surfaced in the FGDC/ISO process steps. The
first instance is the **SCA depth threshold**: `ua_swe`'s
`snow_covered_fraction` is a per-pixel `snow_depth > depth_threshold_mm`
binary evaluated *before* aggregation (a nonlinearity — see
[transformation-pipeline.md](../architecture/transformation-pipeline.md#worked-example-ua_swe-snow_covered_fraction-the-canonical-gotcha)),
so the threshold is baked into the aggregated NC and stamped on the variable.
The deterministic `rebuild-manifest` projection lifts that stamp into the
aggregate step's `params` (read from the NC attr, never config), and
`release/mcf.py` renders it into the process-step description. Because the
threshold is baked at aggregation time, editing
`targets.snow_covered_area.depth_threshold_mm` without re-aggregating `ua_swe`
is a release-blocking error: the publish preflight
`_preflight_ua_swe_threshold_current` refuses to publish a `snow_covered_fraction`
whose stamped threshold no longer matches config (the agg-layer analogue of
the `config.effective.yml` staleness gate).

## File and data requirements

- **Open formats**: NetCDF, CSV, ASCII. Our pipeline writes CF-1.6 NetCDF and
  GeoPackage; both are acceptable.
- **NetCDF and OGC services**: ScienceBase does *not* auto-generate OGC web
  services from NetCDF uploads (only from GeoTIFF / shapefile / raster). NCs
  are uploaded as files only. Document the variable schema in FGDC + README.
- **Checksums**: Not strictly mandated by the SB checklist but strongly
  recommended for preservation. Our pipeline ships per-item `checksums.csv`
  (path, sha256, size, mtime) + GNU `SHA256SUMS` (for `sha256sum -c`).
- **CRS metadata**: EPSG code in FGDC + CF `grid_mapping` variable in each
  NetCDF. Already enforced by [`io_nc.atomic_to_netcdf`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/io_nc.py).
- **Variable docs**: CF standard names where they exist, plus `long_name`,
  `units`, `cell_methods`. Already enforced by
  [`fetch/consolidate.py::apply_cf_metadata`](https://github.com/rmcd-mscb/nhf-spatial-targets/blob/main/src/nhf_spatial_targets/fetch/consolidate.py).
- **README**: Not mandated but expected. Auto-generated per child in our
  pipeline (`release/readme.py` + Jinja templates).
- **File size**: Per-file practical limit is ~10 GB on ScienceBase; per-item
  practical limit ~100 GB. We're comfortably under both. The largest single
  file we'll publish is `mwbm_climgrid` (~7.5 GB).

## DOI workflow

1. ScienceBase parent item is created (no DOI yet).
2. Operator emails the [USGS Science Quality and Integrity](https://www.usgs.gov/about/organization/science-support/office-science-quality-and-integrity)
   contact (or the bureau approving official) with the IPDS record and SB item
   link.
3. After IPDS approval, the operator (or ScienceBase staff) requests DOI mint
   via the SB web UI or by emailing `sciencebase@usgs.gov`.
4. DOI is registered with DataCite as `10.5066/<id>` and shows up in the item
   JSON within minutes.
5. Operator pulls the new DOI into `catalog/release_registry.yml.umbrella.doi`
   and re-runs `nhf-targets release publish --scope umbrella --confirm` to
   re-emit FGDC + ISO XMLs with the DOI populated and update the SB item body.

There is **no** `mint_doi()` API on `sciencebasepy`. The whole DOI step is
out-of-band.

## Incremental child additions (the multi-fabric story)

The release model in this repo is one umbrella + many children, with new
fabric children added over time. ScienceBase supports this:

- New child items can be created under a DOI-minted parent at any time. The
  parent's DOI doesn't change.
- The new child gets its own FGDC metadata but no separate DOI (children
  inherit the umbrella DOI for citation).
- The umbrella's FGDC abstract is intentionally written to describe the
  series as a whole, not enumerate children, so it stays valid as children
  are added.
- The **DOI landing page** metadata is snapshotted at IPDS time and won't
  auto-refresh. If you add a major new fabric and want the landing page to
  reflect it, you must email ScienceBase staff to update the landing-page
  metadata. There is no automated path.

For minor additions (one new fabric child, no change to umbrella abstract)
the DOI landing page can be left stale — it still resolves correctly to the
parent item, which displays the up-to-date child list.

For substantial additions (new source class, major version bump) prefer
minting a new DOI via a new umbrella version. The `release_registry.yml`
schema supports `umbrella.version` for exactly this.

## Version policy (for this pipeline)

| Change kind | Action |
|---|---|
| New fabric (e.g. add `oregon`) | Append child under existing umbrella; no DOI bump |
| Add a new source (e.g. add a new reanalysis) | Append a consolidated-source child + republish fabric children that use it; no DOI bump |
| Refresh source data (e.g. SNODAS adds 2026) | Re-upload affected child files; no DOI bump |
| Methodology change to a target | New umbrella version + new DOI; the old release stays online |
| Catalog correction that changes target values | New umbrella version + new DOI |

The `release_defaults.yml.umbrella.versioning_policy` field records the
canonical policy text for the FGDC abstract.

## Licensing and redistribution

USGS data releases are public-domain by default, but **derived products may
inherit restrictions from upstream sources**. The catalog flags this per
source.

> **Forward reference.** The `release:` block on `catalog/sources.yml`
> entries does not exist on `main` yet — it is added by PR-A in the #241
> phasing. The excerpt below shows the post-PR-A shape.

```yaml
# catalog/sources.yml (excerpt, post-PR-A)
era5_land:
  license: "Copernicus license (free, attribution)"
  release: { publishable: true, notes: "Acknowledge Copernicus in FGDC useconst" }

watergap22d:
  license: "CC BY-NC 4.0"
  release:
    publishable: false
    notes: "CC BY-NC 4.0 is incompatible with USGS-federal redistribution policy; pending OSQI review or WaterGAP 2.2e substitution"

daymet:
  release:
    publishable: true
    distribution_kind: "metadata_only"
    notes: "Source is 4.6 TB of zarr stores; publish a metadata-only child pointing to ORNL DAAC"
```

The `release.publishable` flag is the authoritative gate. The
`release.distribution_kind` flag distinguishes "data + metadata" children
(default) from "metadata-only" children (Daymet). Sources with
`status: superseded` are auto-skipped regardless of `release.publishable`.

Open license questions to resolve with OSQI before any publish:

- **WaterGAP 2.2d** — CC BY-NC redistribution by a federal agency is legally
  ambiguous. Default position: exclude until OSQI clears or 2.2e supersedes.
- **Copernicus ERA5-Land** — free with attribution. Add explicit
  acknowledgment text to FGDC `useconst` and README.
- **NASA Earthdata sources** (MODIS, MERRA-2, NLDAS, GLDAS) — public-domain
  but Earthdata acknowledgment text recommended.
- **NSIDC sources** (SNODAS, Margulis WUS-SR) — public-domain. No constraint.
- **USGS sources** (Reitz 2017, MWBM ClimGrid) — already public releases.
  Cite original DOI; we publish the HRU-aggregated derivative, not the raw
  data.
