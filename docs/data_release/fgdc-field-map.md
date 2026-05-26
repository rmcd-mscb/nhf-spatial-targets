# FGDC field map

How each FGDC CSDGM 2.0 element gets populated from this pipeline's data
sources, organized by item type (umbrella / consolidated-source child /
fabric child). The implementation reference for `release/mcf.py` and the
Jinja2 FGDC templates under `release/templates/fgdc/`.

> **Forward references in this document.** Several sources referenced below
> are landed by upcoming PRs in the #241 phasing and do not exist on `main`
> yet:
>
> - The `release:` block on each entry in `catalog/sources.yml`
>   (`release.publishable`, `release.distribution_kind`, `release.notes`) —
>   added by PR-A.
> - `catalog/release_defaults.yml` and `catalog/release_registry.yml` —
>   added by PR-A.
> - The `release:` block in `<project>/config.yml`
>   (`release.authors`, `release.fabric_label`, `release.abstract_notes`) —
>   added by PR-A.
> - The `<project>/manifest.json.steps[]` content — instrumented by PR-B.
> - The `release/` Python module — built across PR-C through PR-F.
>
> Field paths below describe the post-PR-A/PR-B shape. Cross-check against
> the catalog when implementing each downstream PR.

## Intermediate representation: MCF dict

Rather than build FGDC XML directly from catalog + manifest, we build a
[`pygeometa` Metadata Control File (MCF)](https://geopython.github.io/pygeometa/reference/mcf/)
shaped dict as the intermediate representation, then render it twice:

- **FGDC XML** via Jinja2 templates (pygeometa has no FGDC output schema —
  see `sciencebasepy-notes.md`).
- **ISO 19139 XML** via `pygeometa.schemas.iso19139.ISO19139OutputSchema`,
  shipped as a supplemental file alongside the FGDC.

Why MCF as the IR? Three reasons:

1. **Two output formats from one source of truth.** No risk of FGDC and ISO
   drifting against each other.
2. **Forward-compatible with pygeometa.** If pygeometa ever adds an FGDC
   schema, we replace the Jinja templates without touching the upstream
   builders.
3. **MCF is well-documented.** Engineers can read it without spelunking
   USGS-internal conventions.

## Common MCF fields (all item types)

| MCF path | Source |
|---|---|
| `mcf.version` | Hard-coded `"1.0"` |
| `metadata.identifier` | `release_registry.<scope>.<key>.sb_id` (post-publish) or a generated UUID4 (pre-publish) |
| `metadata.language` | `release_defaults.metadata.language` (default `"en"`) |
| `metadata.charset` | `"utf8"` |
| `metadata.datestamp` | UTC `now()` at build time |
| `metadata.hierarchylevel` | `"dataset"` for source + fabric children; `"series"` for umbrella |
| `contact.distributor` | `release_defaults.contacts.distributor` (USGS ScienceBase block) |
| `contact.pointOfContact` | `release_defaults.contacts.point_of_contact` (overridable per project) |
| `identification.charset` | `"utf8"` |
| `identification.language` | `"en"` |
| `identification.fees` | `release_defaults.distribution.fees` (default `"None"`) |
| `identification.accessconstraints` | `release_defaults.distribution.access_constraints` |

## Umbrella item

| MCF path | Source | FGDC element |
|---|---|---|
| `identification.title` | `release_defaults.umbrella.title_template` formatted with `release_registry.umbrella.version` | `<title>` |
| `identification.abstract` | `release_defaults.umbrella.abstract_template`, Jinja-rendered over list of published children pulled from `release_registry` | `<abstract>` |
| `identification.purpose` | `release_defaults.umbrella.purpose` | `<purpose>` |
| `identification.dates.publication` | `release_registry.umbrella.published_utc` | `<pubdate>` |
| `identification.dates.revision` | `max(child.uploaded_utc)` across all children in registry | `<revdate>` |
| `identification.keywords.theme` | `release_defaults.keywords.umbrella.theme` (e.g. "Hydrologic Cycle", "Calibration", "National Hydrologic Model") | `<themekey>` |
| `identification.keywords.place` | Union of all children's spatial extents (CONUS + sub-regions) | `<placekey>` |
| `identification.extents.spatial.bbox` | Union of all published children's bboxes | `<bounding>` |
| `identification.extents.temporal.begin/end` | Union of all children's `<period>` ranges | `<timeperd>` |
| `identification.useconstraints` | `release_defaults.distribution.use_constraints` + USGS standard + per-source notes from each child | `<useconst>` |
| `identification.doi` | `release_registry.umbrella.doi` (full URL form: `https://doi.org/10.5066/...`) | `<onlink>` in `<citeinfo>` |
| `identification.citation.authors` | `<project>/config.yml.release.authors` (each `{given, family, orcid, affiliation}` becomes a CSDGM `<cntinfo>` block) | `<origin>` |
| `dataquality.lineage.statement` | `release_defaults.umbrella.lineage_statement` | `<lineage>` |

## Consolidated-source child item

(One per source from `catalog/sources.yml`. Sources with `status: superseded`
or `release.publishable: false` are skipped.)

| MCF path | Source | FGDC element |
|---|---|---|
| `identification.title` | `catalog/sources.yml[<key>].name` + " (consolidated for NHF)" | `<title>` |
| `identification.abstract` | `catalog/sources.yml[<key>].description` + `release_defaults.source_abstract_suffix` (Jinja-rendered with source-specific context) | `<abstract>` |
| `identification.purpose` | `release_defaults.source.purpose_template` | `<purpose>` |
| `identification.keywords.theme` | `release_defaults.keywords.source.theme` + per-variable CF `standard_name` for each var in `catalog/sources.yml[<key>].variables` | `<themekey>` |
| `identification.keywords.place` | `catalog/sources.yml[<key>].spatial_extent` (parsed/normalized) | `<placekey>` |
| `identification.extents.spatial.bbox` | `catalog/sources.yml[<key>].access.bbox_nwse` if present, else union of bboxes from staged consolidated NCs | `<bounding>` |
| `identification.extents.temporal.begin/end` | `catalog/sources.yml[<key>].period` (parsed `"YYYY/YYYY"` → begin + end) | `<timeperd>` |
| `identification.citation.authors` | `catalog/sources.yml[<key>].citations[]` (parsed into structured form where possible; raw string otherwise) | `<origin>` |
| `identification.citation.doi` | `catalog/sources.yml[<key>].doi` or `.access.doi` | `<onlink>` |
| `identification.useconstraints` | `catalog/sources.yml[<key>].license` + `release_defaults.distribution.use_constraints` + `catalog/sources.yml[<key>].release.notes` | `<useconst>` |
| `dataquality.lineage.statement` | `catalog/sources.yml[<key>].notes` + `catalog/sources.yml[<key>].access.notes` | `<lineage>` |
| `dataquality.lineage.processstep[]` | `<project>/manifest.json.steps[]` filtered to `source_key == <key>` | `<procstep>` (one per record) |
| `distribution.online[*]` | One entry per consolidated NC: `name=<filename>`, `description="Consolidated NetCDF, <period>"`, `protocol="WWW:DOWNLOAD-1.0-http--download"` | `<digtinfo>` + `<netaddr>` |
| Per-variable `identification.entityandattribute` | Each entry in `catalog/sources.yml[<key>].variables`: `name`, `long_name`, `cf_units`, `cell_methods` | `<eainfo>` |

Special case for `release.distribution_kind: metadata_only` (Daymet):
`distribution.online[*]` carries a single entry pointing to the upstream
source (ORNL DAAC for Daymet) instead of a list of staged NCs.

## Fabric child item

(One per fabric. The "fabric" is identified by the operator's
`<project>/config.yml.fabric.path` + an optional `release.fabric_label`
override.)

| MCF path | Source | FGDC element |
|---|---|---|
| `identification.title` | `release_defaults.fabric.title_template` formatted with fabric label (from `release.fabric_label` or `Path(fabric.path).stem`) | `<title>` |
| `identification.abstract` | `release_defaults.fabric.abstract_template`, Jinja-rendered over: fabric_label, `<project>/fabric.json.hru_count`, list of targets from `<project>/config.yml.targets`, source list from `<project>/manifest.json.sources`, and `<project>/config.yml.release.abstract_notes` | `<abstract>` |
| `identification.purpose` | `release_defaults.fabric.purpose_template` | `<purpose>` |
| `identification.keywords.theme` | `release_defaults.keywords.fabric.theme` + one keyword per enabled target name + `"National Hydrologic Model"` | `<themekey>` |
| `identification.keywords.place` | Derived from `<project>/fabric.json.bbox` + fabric_label | `<placekey>` |
| `identification.extents.spatial.bbox` | `<project>/fabric.json.bbox` | `<bounding>` |
| `identification.extents.temporal.begin/end` | Union of enabled-target periods in `<project>/config.yml.targets[*].period` | `<timeperd>` |
| `identification.citation.authors` | `<project>/config.yml.release.authors` | `<origin>` |
| `identification.citation.doi` | `release_registry.umbrella.doi` (children share the umbrella DOI per SB convention) | `<onlink>` |
| `identification.useconstraints` | Union of participating source `license` strings + `<project>/config.yml.release.use_constraints_addition` (optional) | `<useconst>` |
| `dataquality.lineage.statement` | `release_defaults.fabric.lineage_statement_template`, Jinja-rendered with source + target list | `<lineage>` |
| `dataquality.lineage.processstep[]` | All entries in `<project>/manifest.json.steps[]` | `<procstep>` (one per record) |
| `distribution.online[*]` | One entry per staged file: `fabric.gpkg`, every aggregated NC, every target NC (and `_nn_filled.nc`), `manifest.json`, `README.md`, `checksums.csv`, `SHA256SUMS` | `<digtinfo>` |
| Per-target `identification.entityandattribute` | Each enabled target from `catalog/variables.yml`: `description`, `units`, `range_method`, `range_notes`, plus per-source variable mapping | `<eainfo>` |

## Lineage record → CSDGM `<procstep>`

The pipeline accumulates lineage records in `<project>/manifest.json.steps[]`
(PR-B instruments the call sites). Each record has the shape:

```json
{
  "kind": "fetch|consolidate|aggregate|target|nn_fill|validate",
  "source_key": "era5_land",
  "timestamp_utc": "2026-05-26T12:34:56+00:00",
  "software_version": "0.7.0",
  "tool": "nhf-targets",
  "command": "agg era5-land",
  "inputs":  [{"path": "...", "size_bytes": ..., "mtime_utc": "..."}],
  "outputs": [{"path": "...", "sha256": "...", "size_bytes": ..., "mtime_utc": "..."}],
  "params":  {"batch_size": 10000, ...}
}
```

In FGDC, each step renders to:

```xml
<procstep>
  <procdesc>{kind} via {tool} {software_version}, command: {command}; params: {params (formatted)}</procdesc>
  <srcused>{srcabbr-for-inputs[i]}</srcused>
  <!-- one <srcused> per input; element is repeatable -->
  <procdate>{YYYYMMDD from timestamp_utc}</procdate>
  <proctime>{HHMMSS from timestamp_utc}</proctime>
  <srcprod>{srcabbr-for-outputs[i]}</srcprod>
  <!-- one <srcprod> per output; element is repeatable -->
</procstep>
```

**CSDGM 2.0 element order is fixed** as `procdesc → srcused → procdate → proctime → srcprod`; `mp` rejects out-of-order steps. Both `<srcused>` and `<srcprod>` carry a **Source Citation Abbreviation** (a short token resolving to a separate `<srcinfo>` block elsewhere in the document), not a raw file path. They are repeatable, so a step with N inputs renders N `<srcused>` elements, not one comma-joined element. `release/mcf.py` is responsible for emitting the corresponding `<srcinfo>` entries — design that mapping when implementing PR-D.

Filtering rule:

- For a **consolidated-source child** with key `<key>`, include only steps
  where `source_key == <key>` (this captures the fetch + consolidate steps
  for that source).
- For a **fabric child**, include all steps in `manifest.json.steps[]`
  (fetch + consolidate + aggregate + target + nn_fill + validate). The
  union describes how this fabric's outputs were produced.
- For the **umbrella**, no per-step lineage in the FGDC — the umbrella's
  `<lineage>` statement is a high-level prose summary; per-step detail
  lives on the children.

## CRS metadata

All NetCDFs in the release carry the CF-1.6 `grid_mapping` variable already
(see [`docs/architecture/nc-encoding-policy.md`](../architecture/nc-encoding-policy.md)).
FGDC additionally requires CRS info in the spatial reference block:

```xml
<spref>
  <horizsys>
    <geograph>
      <latres>0.001</latres>
      <longres>0.001</longres>
      <geogunit>Decimal degrees</geogunit>
    </geograph>
    <geodetic>
      <horizdn>WGS_1984</horizdn>
      <ellips>WGS_1984</ellips>
      <semiaxis>6378137.0</semiaxis>
      <denflat>298.257223563</denflat>
    </geodetic>
  </horizsys>
</spref>
```

This is constant across all our items (everything is on WGS84 HRU
centroids) and lives in `release_defaults.yml.spatial_reference` as a block
that's spliced verbatim into the FGDC template.

## Validation

After FGDC XML is rendered, `release/validate_xml.py` shells out to the
[USGS Metadata Parser (`mp`)](https://geology.usgs.gov/tools/metadata/tools/doc/mp.html)
if it's on PATH. Three-state result:

| State | Behavior |
|---|---|
| `mp` not found | Warn-only; FGDC unvalidated; build proceeds |
| `mp` returns warnings | Surfaced in `release dry-run` Rich table; build proceeds |
| `mp` returns errors | Fatal by default; `--skip-mp` flag overrides |

Optional fallback: `xmlschema` Python library validates against the CSDGM
2.0 schema (`fgdc-std-001-1998.dtd`) without requiring `mp`. We may add this
later if operator feedback says `mp` is painful to install.

## Reading order for implementation

When you implement PR-D (MCF + FGDC + ISO), read in this order:

1. The "Common MCF fields" table — what's shared.
2. The three per-item tables — what's different per scope.
3. The lineage section — where the `<procstep>` blocks come from.
4. The validation section — what to wire up at the tail of the build.

Then look at `release/mcf.py` (build the dict) and the Jinja templates
under `release/templates/fgdc/` (render to XML). The MCF dict shape is
identical for FGDC and ISO output, so `release/iso.py` consumes the same
dict and calls `pygeometa.schemas.iso19139`.
