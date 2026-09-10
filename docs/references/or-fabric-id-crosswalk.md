# Oregon fabric HRU id crosswalk (v9 → v11)

Companion to [`or_fabric_crosswalk_v9_to_v11.csv`](or_fabric_crosswalk_v9_to_v11.csv)
and its [provenance sidecar](or_fabric_crosswalk_v9_to_v11.provenance.json).
Generated for issue #353.

## Why this file exists at all

The Oregon fabric shipped in two releases whose `nhru` layers are
**field-for-field identical in schema and geometry** but disagree on what
`nhm_id` means:

| | `model_layers_9` | `model_layers 11` |
|---|---|---|
| `nhm_id` | 1 – 41,195 (sparse **national**) | 1 – 16,814 (dense **local**) |
| `hru_id` | 1 – 16,814 | 1 – 16,814 (identical to v9) |
| `model_hru_idx` | identical to `hru_id` | identical to `hru_id` |
| geometry | — | **16,814/16,814 identical to v9** |

Verified directly:

```
v9 hru_id        == v11 hru_id        : True
v9 model_hru_idx == v11 model_hru_idx : True
v9 nhm_id        == v11 nhm_id        : False   <-- the only column that moved
```

`nhm_id` is controlled by the national fabric producer and was renumbered
without a schema change, a column rename, or any other visible signal.
`hru_id` is intrinsic to the Oregon model and did not move.

## The hazard this file protects against

Only **1,158** of 16,814 `nhm_id` values exist in both numberings, and only
**280** of those refer to the *same polygon*. Joining v9-keyed data to the
v11 fabric by `nhm_id`:

- drops ~15,656 rows that have no match, and
- **silently attaches ~878 HRUs to the wrong polygon.**

There is no error, no shape change and no dtype change — the column name is
the same and the geometry is the same. Nothing warns you.

## What the file contains

16,814 rows, sorted by `nhm_id_v11`:

| column | meaning |
|---|---|
| `nhm_id_v9` | the sparse national id every pre-#353 Oregon artifact was keyed on |
| `nhm_id_v11` | v11's dense local id — **equal to `hru_id`** |
| `hru_id` | the Oregon model's own index, stable across both releases |
| `model_hru_idx` | identical to `hru_id` |
| `vpu`, `areasqkm` | context for spot-checks |

Every row was verified before the file was written: bijection in both
directions, round-trips `v9 → v11 → v9`, covers every id present in all 12
Oregon target NCs, and leaves every file strictly ascending after mapping.
The provenance sidecar records both fabrics' sha256 and each check.

## It cannot be regenerated

This mapping exists **only** because v9's parquet carries `nhm_id` and
`hru_id` side by side on the same row. v11 overwrote `nhm_id` with the local
numbering, destroying that correspondence. Once the v9 fabric is gone the
crosswalk cannot be reconstructed from anything else — which is why it is
committed here rather than left in the project directory.

## What #353 actually did with it

The migration did **not** consume this CSV. `maintenance relabel-id-col`
derives its `{old: new}` map from the fabric itself, where both columns sit
on the same row, so it cannot go stale against the geometry the way a
separate file can.

This crosswalk is the **external** record: the artifact anyone who joined to
the old national ids needs in order to follow the change.

## The lesson

Key a project on an identifier the project controls. `nhm_id` belongs to the
national fabric producer; `hru_id` belongs to the Oregon model. Had the
project used `hru_id` from the start, v9 and v11 would have been equivalent
and none of this would have been necessary.

Note the follow-on caveat: `hru_id` is a **dense positional index**, stable
here only because the HRU *set* did not change. Adding or removing a single
HRU would shift every id after it, just as silently. A `{fabric}_id`
convention should therefore specify ids that are **persistent and
immutable** — assigned once, never renumbered, never reused — not merely
locally scoped. See [`lessons-learned.md`](lessons-learned.md).
