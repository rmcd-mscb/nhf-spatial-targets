# `docs/data_release/` — ScienceBase data release reference

Reference notes for the planned `nhf-targets release` feature (publishing this
pipeline's outputs as a USGS data release on
[ScienceBase](https://www.sciencebase.gov/)). These are background reading
captured at design time — they are **not** the spec for the feature itself.
The spec lives in the issue and PRs as the work lands.

## When to read these files

- **Designing or reviewing PRs in the `release/` track** — see the field map
  and sciencebasepy notes for what the code must do.
- **Onboarding to USGS data-release process** — start with `usgs-process.md`
  for the lifecycle.
- **Debugging an FGDC validation failure** — see `fgdc-field-map.md` for where
  each FGDC element comes from in the pipeline's data.

## Files

- [usgs-process.md](usgs-process.md) — USGS data release lifecycle (IPDS →
  ScienceBase → DOI), FGDC CSDGM 2.0 requirements, file/data requirements,
  parent/child item structure, and how new fabric children get added
  incrementally to an existing release.
- [sciencebasepy-notes.md](sciencebasepy-notes.md) — capability matrix for the
  [`sciencebasepy`](https://github.com/DOI-USGS/sciencebasepy) Python client.
  What it makes easy, what it cannot do (no DOI minting; no FGDC XML
  generation), and the sharp edges we'll hit.
- [fgdc-field-map.md](fgdc-field-map.md) — concrete mapping from
  `catalog/sources.yml` + `catalog/variables.yml` + `manifest.json` +
  `<project>/config.yml` to FGDC CSDGM 2.0 elements, via a pygeometa-shaped
  MCF intermediate representation. The implementation-side reference for
  `release/mcf.py` and the Jinja2 FGDC templates.

## External pointers

- USGS data release process: <https://www.usgs.gov/sciencebase-instructions-and-documentation/data-release-process>
- ScienceBase data release checklist: <https://www.usgs.gov/sciencebase-instructions-and-documentation/sciencebase-data-release-checklist>
- ScienceBase metadata instructions: <https://www.usgs.gov/sciencebase-instructions-and-documentation/metadata-instructions>
- `sciencebasepy` source: <https://github.com/DOI-USGS/sciencebasepy>
- `pygeometa` (MCF + ISO 19139 emitter): <https://geopython.github.io/pygeometa/>
- USGS Metadata Parser (FGDC validator): <https://geology.usgs.gov/tools/metadata/tools/doc/mp.html>
- MetadataWizard (USGS-blessed GUI; not used here): <https://doi-usgs.github.io/fort-pymdwizard/>

## Status of these notes

Captured 2026-05-26 during the design conversation that produced the
implementation plan. Treat them as the snapshot of external-world facts at
that date; if you find something out of date, fix it in place — they exist
to save the next person from re-doing the research.
