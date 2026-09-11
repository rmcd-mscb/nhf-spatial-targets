# Publishing a model fabric — guideline for fabric developers

**Goal:** anyone consuming a fabric should be able to name exactly which one
they used, download that exact one on a fresh machine, and be told loudly if it
ever changes. Today that isn't possible — fabrics reach consumers as file
copies, and a copy carries no version.

The ask is small: **publish a tagged release with two files.**

---

## The recipe

For each fabric release, in the repo that generates it:

1. **Build one GeoPackage** containing every layer (`nhru`, `nsegment`,
   `npoigages`, `npoi`, `domain`, …).

   Name it `<fabric>_v<version>.gpkg` — e.g. `or_v9.gpkg`.
   No spaces, no dates, no "final", no "copy".

2. **Generate the sidecar JSON** (script below, ~2 seconds):

   ```bash
   python make_fabric_sidecar.py or_v9.gpkg \
       --fabric or --version 9 \
       --title "Oregon NHM model layers" \
       --repo https://code.usgs.gov/ORG/REPO --tag or-v9
   ```

   This writes `or_v9.json` and embeds the GeoPackage's SHA-256.

3. **Fill in anything the script couldn't infer.** It prints, e.g.:

   ```
   NOTE: set primary_key by hand for: domain, main
   ```

4. **Tag and release**:

   ```bash
   git tag or-v9 && git push origin or-v9
   ```

   Create a release on that tag and attach `or_v9.gpkg` + `or_v9.json`.

That's it. Consumers then pin `fabric: {source: or, version: 9}` and fetch it
by a URL derived from the tag.

---

## The rules

**1. A published version is immutable.**
`or-v9` is `or-v9` forever. Never move the tag, never replace the asset.
Changed content gets a new version number. Please enable *protected tags* so
this doesn't rest on memory.

**2. Bump the version whenever anything in the GeoPackage changes** —
geometry, ids, attributes, layers. Version numbers are cheap; ambiguity is not.

**3. Never reuse an id to mean something different.**
This is the one that bites hardest. If HRU 8793 means one polygon in v9 and a
different polygon in v11, every downstream product silently misaligns. Renumber
only when you must, and when you do, say so in the release notes — it is a
breaking change, not a refresh.

**4. Declare each layer's primary key** in the sidecar, and mark ids that
belong to someone else's numbering as cross-references. The script does most of
this automatically.

---

## What the sidecar looks like

```json
{
  "fabric": "or",
  "version": 9,
  "title": "Oregon NHM model layers",
  "crs": "EPSG:5070",
  "gpkg": {
    "filename": "or_v9.gpkg",
    "bytes": 96034816,
    "sha256": "e298657ec5ca30a067a3bcecfe46171ce9d4f661ac3958aa93276125809a0316"
  },
  "produced_by": {
    "repo": "https://code.usgs.gov/ORG/REPO",
    "tag": "or-v9",
    "commit": "0db4bd3efddeb27ea5cf7cd83beab9ecca8d78bb",
    "generated_utc": "2026-09-11T18:17:11+00:00"
  },
  "layers": {
    "nhru": {
      "geometry": "MultiPolygon",
      "features": 16814,
      "primary_key": "hru_id",
      "fields": ["vpu_agg_id", "nhm_id", "nhm_hru_seg", "areasqkm",
                 "vpu", "hru_segment", "hru_id", "model_hru_idx"],
      "cross_reference_ids": {
        "nhm_id": "identifier in an external numbering - not this fabric's key",
        "nhm_hru_seg": "identifier in an external numbering - not this fabric's key"
      }
    }
  }
}
```

Three fields do work nothing else can:

- **`sha256`** — lets a consumer prove they have the right bytes, and catches a
  version that was quietly republished.
- **`primary_key`** — says which column is *the* key for that layer. Without it,
  consumers guess, and they guess differently from each other.
- **`cross_reference_ids`** — demotes `nhm_id` to "useful for joining to the
  national fabric, not our key." One line that prevents a whole class of error.

Everything else is inventory — useful, but regenerable from the file.

---

## Why we're asking

We recently keyed an entire Oregon target pipeline on `nhm_id`. A later fabric
renumbered that column to a dense local sequence while keeping the same column
name, the same schema, and byte-for-byte identical geometry.

The two numberings share 1,158 values, and only **280** of those refer to the
same polygon. So joining old data to the new fabric would have mislabelled
about **878 HRUs** and dropped the rest — with **no error, no shape change, and
no dtype change**. We caught it only because a build crashed for an unrelated
reason.

Recovery cost us a day: a crosswalk, a migration tool, a full relabel of ~800
files, and a rebuild of every target. A version number and a line saying
`"primary_key": "hru_id"` would have prevented all of it.

We're not asking for content hashing, semantic-versioning semantics, or a
metadata standard. Just: **tag it, attach two files, and never reuse a
version number or an id.**

---

## Two questions for you

1. **`main` and `npoi` look like duplicates** — identical schema, identical
   7,701 features, both Point. Is one a leftover? Dropping it would shrink the
   artifact and remove a "which do I use?" question.

2. **Is `hru_id` stable across releases, or positional?** It held steady in our
   case, but it's a dense `1..N` sequence, so it would shift if HRUs were ever
   added or removed. If it's positional, say so in the sidecar — consumers need
   to know whether they can cache things keyed on it.

---

## The script

`scripts/make_fabric_sidecar.py` in this repository — standalone, needs only `pyogrio` (ships with geopandas). Copy it into your
fabric repo and run it as the last step of the build, so the sidecar can never
drift from the artifact it describes.

It refuses to guess: layers whose primary key it does not recognise are
emitted with `"primary_key": null` and named in a note on stderr, so an
unknown key is a prompt rather than a silent wrong answer.
