# Oregon fabric HRU id crosswalk (`nhm_id` → `hru_id`)

Companion to
[`or_fabric_crosswalk_nhm_id_to_hru_id.csv`](or_fabric_crosswalk_nhm_id_to_hru_id.csv)
and its [provenance sidecar](or_fabric_crosswalk_nhm_id_to_hru_id.provenance.json).
Generated for issue #353.

Applies to **`model_layers_9`** — the Oregon fabric this project uses, and the
one the `gfv2-params` workflows use.

## What changed

Every Oregon artifact built before #353 was keyed on `nhm_id`, the **national**
NHM identifier. The project now keys on `hru_id`, the Oregon model's own index:

| | `nhm_id` (before) | `hru_id` (now) |
|---|---|---|
| range | 1 – 41,195 (sparse) | **1 – 16,814 (dense)** |
| unique values | 16,814 | 16,814 |
| owned by | the national fabric | **the Oregon model** |
| matches `gfv2-params` | no | **yes** |

Both columns live on the same row of the same fabric file, so the mapping
cannot disagree with the geometry.

## Who needs this file

Anyone holding data joined to the **old** `nhm_id` — earlier target NCs,
downstream analyses, colleagues' spreadsheets. Map through this crosswalk to
land on the `hru_id`-keyed targets.

The two numberings are **not** interchangeable: they disagree on 16,534 of
16,814 rows. Only the first 280 happen to coincide, which is exactly enough to
make a bad join look plausible on a spot check.

## What the file contains

16,814 rows, sorted by `hru_id`:

| column | meaning |
|---|---|
| `nhm_id` | the national id every pre-#353 Oregon artifact was keyed on |
| `hru_id` | the Oregon model's index — the current key |
| `model_hru_idx` | identical to `hru_id` |
| `vpu`, `areasqkm` | context for spot-checks |

Every row was verified before the file was written: bijection in both
directions, round-trips, covers every id present in all 12 Oregon target NCs,
and leaves every file strictly ascending after mapping. The provenance sidecar
records the fabric sha256 and each check.

## Why it is committed here

The mapping exists only because the fabric carries `nhm_id` and `hru_id` side
by side. It is cheap to keep and awkward to reconstruct after the fact, and it
is the only record connecting published pre-#353 Oregon data to the current
targets.

## What #353 did with it

The migration did **not** consume this CSV.
`nhf-targets maintenance relabel-id-col` derives its `{old: new}` map from the
fabric directly, so it cannot go stale against the geometry the way a separate
file can. This crosswalk is the **external** record for downstream consumers.

## The lesson

Key a project on an identifier the project controls. `nhm_id` belongs to the
national fabric producer and can be renumbered upstream without a schema change
or a column rename; `hru_id` belongs to the Oregon model.

Note the follow-on caveat: `hru_id` is a **dense positional index**, stable
while the HRU *set* is stable. Adding or removing a single HRU would shift every
id after it, just as quietly. A `{fabric}_id` convention should therefore
specify ids that are **persistent and immutable** — assigned once, never
renumbered, never reused — not merely locally scoped. See
[`lessons-learned.md`](lessons-learned.md).

The contract we ask fabric developers to adopt so this cannot recur is
[`fabric-publishing-guideline.md`](fabric-publishing-guideline.md).
