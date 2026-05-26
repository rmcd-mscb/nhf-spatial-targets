"""Shared cyclopts Parameter aliases for the agg sub-app commands.

Each ``agg`` command annotates its matching parameter as
``Annotated[<type>, _AGG_*_PARAM] = <default>``. Cyclopts reads name +
help from the Parameter; the default lives on the function signature so
the type checker still sees a concrete default (a single source of
truth for both behavior + docs).
"""

from __future__ import annotations

from cyclopts import Parameter

_AGG_BATCH_SIZE_PARAM = Parameter(
    name="--batch-size",
    help=(
        "Target HRUs per spatial batch. Overrides ``fabric.batch_size`` "
        "from ``config.yml`` (default 500). Changing this value "
        "invalidates cached weight CSVs via the SHA-256 batch-HRU "
        "fingerprint."
    ),
)
_AGG_WORKER_INDEX_PARAM = Parameter(
    name=["--worker-index", "-w"],
    help=(
        "SLURM-array round-robin worker index in [0, --n-workers). "
        "Default 0 (single worker). Each worker processes a disjoint "
        "round-robin slice of the source's years; see "
        "docs/sources/ on the SLURM array recipe (issue #156)."
    ),
)
_AGG_N_WORKERS_PARAM = Parameter(
    name=["--n-workers", "-n"],
    help=(
        "Total SLURM-array workers for round-robin year sharding. "
        "Default 1 (serial). Must match the SLURM ``--array`` upper "
        "bound + 1. Each worker writes its assigned per-year NCs and "
        "merges its slice into manifest.json under a flock."
    ),
)
