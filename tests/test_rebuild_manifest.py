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


# ---------------------------------------------------------------------------
# Task 2.3: build_source_entry
# ---------------------------------------------------------------------------


def test_source_entry_catalog_key(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    (d / "merra2_2000.nc").write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=False)
    assert entry["source_key"] == "merra2"
    assert entry["provenance"] == "reconstructed"
    assert entry["derived_variant"] is False
    assert len(entry["files"]) == 1
    assert "sha256" not in entry["files"][0]  # opt-in only


def test_source_entry_noncatalog_is_derived_variant(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "era5_land_sd"
    d.mkdir()
    (d / "era5_land_sd_agg.nc").write_bytes(b"x")
    entry = build_source_entry("era5_land_sd", d, compute_sha256=False)
    assert entry["derived_variant"] is True


def test_source_entry_period_from_years(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "ssebop"
    d.mkdir()
    (d / "ssebop_2000_agg.nc").write_bytes(b"x")
    (d / "ssebop_2003_agg.nc").write_bytes(b"x")
    entry = build_source_entry("ssebop", d, compute_sha256=False)
    assert entry["period"] == "2000/2003"


def test_source_entry_sha256_optin(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    (d / "merra2_2000.nc").write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=True)
    assert "sha256" in entry["files"][0]


def test_source_entry_files_sorted(tmp_path):
    from nhf_spatial_targets.rebuild_manifest import build_source_entry

    d = tmp_path / "merra2"
    d.mkdir()
    for name in ("merra2_2002.nc", "merra2_2000.nc", "merra2_2001.nc"):
        (d / name).write_bytes(b"x")
    entry = build_source_entry("merra2", d, compute_sha256=False)
    paths = [f["path"] for f in entry["files"]]
    assert paths == sorted(paths)
