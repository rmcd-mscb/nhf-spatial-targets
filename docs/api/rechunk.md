# Rechunk

`nhf_spatial_targets.rechunk` backfills already-written aggregated and target
NetCDFs to the canonical chunking + compression layout (the #165 policy in
[`docs/architecture/nc-encoding-policy.md`](../architecture/nc-encoding-policy.md)).
It is the actuator behind `nhf-targets maintenance rechunk`.

Every conversion is **idempotent** (already-canonical files are skipped),
**atomic** (tmp file + rename), and **value-preserving** (each variable is
verified before the rename), so a run that dies mid-source resumes cleanly on
resubmit. `daymet`/`ssebop` aggregated outputs are intentionally left
unchunked and are not rechunked.

::: nhf_spatial_targets.rechunk
    options:
      show_source: true
      heading_level: 2
