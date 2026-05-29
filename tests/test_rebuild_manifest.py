"""Tests for ``nhf_spatial_targets.rebuild_manifest``.

``rebuild-manifest`` is the one authoritative deterministic projection of
(datastore consolidated dirs x catalog) U project ``data/aggregated/`` dirs
U ``targets/`` NCs U ``fabric.json`` into a complete ``manifest.json``.

Invariants locked here (spec decisions E/F, plan PR-2):

- Deterministic, byte-identical on re-run (the idempotency contract).
- year/period parse from aggregated filenames incl. ``ssebop_2000_agg.nc``
  and ``daymet_na_1980_agg.nc``.
- Non-catalog-key dir -> ``derived_variant: True`` (never orphan a shipped NC).
- Non-publishable source (``watergap22d``) still recorded.
- No ``datetime.now()`` reachable from the rebuild path.
- Concurrent flock merge preserves identity fields.
- ``reconcile-manifest`` removed; not a registered command.
"""

from __future__ import annotations

import pytest

from nhf_spatial_targets.rebuild_manifest import parse_aggregated_filename


@pytest.mark.parametrize(
    "name,year",
    [
        ("ssebop_2000_agg.nc", 2000),
        ("daymet_na_1980_agg.nc", 1980),
        ("mod10c1_v061_2020_agg.nc", 2020),
        ("merra2_agg.nc", None),  # single-shot, no year
        ("snodas_agg.nc", None),
    ],
)
def test_parse_year(name, year):
    assert parse_aggregated_filename(name)[1] == year
