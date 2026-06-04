"""Tests for nhf_spatial_targets.release.build (offline build orchestrator).

These assert *structure / presence / consistency*, not golden XML bytes:
build.py only wires the existing pieces together, and the byte-stability of
the rendered FGDC / ISO / README is already covered by test_release_fgdc /
test_release_iso / test_release_readme. Here we check that each ``build_*``
produces a complete, internally-consistent stage directory.

The project fixture mirrors tests/test_release_payload.py's hand-built
``Project`` (so build.py is exercised against the same staging rules) but adds
the config / fabric.json / manifest.json content the MCF builders need.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from lxml import etree

from nhf_spatial_targets.release import build, payload
from nhf_spatial_targets.release.checksums import (
    CHECKSUMS_CSV,
    SHA256SUMS,
    verify_csv,
)
from nhf_spatial_targets.workspace import Project

# Clock-injected so a re-run renders byte-identical metadata (determinism check).
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build_project(tmp_path, *, fabric_label: str = "gfv_test") -> Project:
    """Synthesize a project + datastore exercising the build orchestrator.

    Aggregated subdirs: era5_land (publishable, kept), watergap22d
    (non-publishable, excluded), daymet (publishable metadata_only -- its
    aggregated NCs still ship in the fabric child). Datastore carries era5_land
    consolidated NCs (a data source child) and a daymet NC that must NOT stage
    (metadata_only).
    """
    workdir = tmp_path / "proj"
    datastore = tmp_path / "store"
    fabric_file = _touch(tmp_path / "fab" / "gfv_test.gpkg", b"GPKG-bytes")

    agg = workdir / "data" / "aggregated"
    _touch(agg / "era5_land" / "era5_land_1980_agg.nc", b"era5-1980")
    _touch(agg / "watergap22d" / "watergap22d_1980_agg.nc", b"wg-held")
    _touch(agg / "daymet" / "daymet_na_1980_agg.nc", b"daymet-1980")
    _touch(workdir / "targets" / "runoff_targets.nc", b"runoff")
    _touch(workdir / "targets" / "aet_targets.nc", b"aet")

    _touch(datastore / "era5_land" / "daily" / "era5_land_daily_1980.nc", b"d-1980")
    _touch(datastore / "era5_land" / "monthly" / "era5_land_monthly_1980.nc", b"m-1980")
    # metadata_only: present on disk but must never stage into the source child.
    _touch(datastore / "daymet" / "daymet_consolidated.nc", b"should-not-stage")

    (workdir / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {"era5_land": {}, "watergap22d": {}},
                "steps": [
                    {
                        "kind": "aggregate",
                        "source_key": "era5_land",
                        "timestamp_utc": "2026-01-02T03:04:05+00:00",
                        "software_version": "0.7.0",
                        "tool": "nhf-targets",
                        "command": "agg era5-land",
                        "inputs": [
                            {
                                "path": "/x/era5_land_daily_1980.nc",
                                "size_bytes": 6,
                                "mtime_utc": "2026-01-01T00:00:00+00:00",
                            }
                        ],
                        "outputs": [
                            {
                                "path": "agg/era5_land/era5_land_1980_agg.nc",
                                "sha256": "ab12",
                                "size_bytes": 9,
                                "mtime_utc": "2026-01-02T00:00:00+00:00",
                            }
                        ],
                        "params": {"method": "area_weighted"},
                    },
                    {
                        "kind": "validate",
                        "source_key": None,
                        "timestamp_utc": "2026-01-02T02:00:00+00:00",
                        "software_version": "0.7.0",
                        "tool": "nhf-targets",
                        "command": "validate",
                        "inputs": [],
                        "outputs": [],
                        "params": {},
                    },
                ],
            }
        )
    )

    config = {
        "fabric": {"path": str(fabric_file), "id_col": "nhm_id"},
        "release": {
            "authors": [
                {
                    "given": "Jane",
                    "family": "Doe",
                    "affiliation": "U.S. Geological Survey",
                }
            ],
            "fabric_label": fabric_label,
            "doi": None,
            "abstract_notes": "Synthetic project for build tests.",
            "ipds_number": "IP-123456",
        },
        "targets": {
            "runoff": {"enabled": True, "period": "1980/2020"},
            "aet": {"enabled": True, "period": "2000/2010"},
        },
    }
    fabric = {
        "path": str(fabric_file),
        "bbox": {"minx": -125.0, "miny": 24.0, "maxx": -66.0, "maxy": 53.0},
        "hru_count": 4242,
    }
    return Project(
        workdir=workdir,
        datastore=datastore,
        config=config,
        fabric=fabric,
        dir_mode=None,
    )


def _parse(path: Path) -> etree._Element:
    """Parse an XML file, asserting it is well-formed (raises otherwise)."""
    return etree.fromstring(path.read_bytes())


def _checksum_paths(stage_dir: Path) -> set[str]:
    """The ``path`` column of every row in the stage's checksums.csv."""
    import csv

    with (stage_dir / CHECKSUMS_CSV).open(newline="") as f:
        return {row["path"] for row in csv.DictReader(f)}


def _download_names(mcf: dict) -> set[str]:
    """Names of every function=download entry in an MCF distribution block."""
    return {
        name
        for name, entry in mcf["distribution"].items()
        if entry.get("function") == "download"
    }


# ---------------------------------------------------------------------------
# fabric child
# ---------------------------------------------------------------------------


def test_build_fabric_child_produces_full_stage(tmp_path):
    project = _build_project(tmp_path)
    result = build.build_fabric_child(project, now=NOW)

    assert result.kind == "fabric"
    assert result.key == "gfv_test"
    stage = result.stage_dir
    assert stage == project.release_build_dir / "fabric" / "gfv_test"

    # fabric.gpkg symlinked; data files present; metadata + integrity written.
    gpkg = stage / "fabric.gpkg"
    assert gpkg.is_symlink() and gpkg.read_bytes() == b"GPKG-bytes"
    assert (stage / "aggregated" / "era5_land" / "era5_land_1980_agg.nc").is_symlink()
    assert (stage / "targets" / "runoff_targets.nc").is_symlink()
    for name in (
        "manifest.json",
        "README.md",
        "fgdc.xml",
        "iso.xml",
        CHECKSUMS_CSV,
        SHA256SUMS,
    ):
        assert (stage / name).is_file(), f"missing {name}"

    # Non-publishable aggregated source must not have leaked in.
    assert not (stage / "aggregated" / "watergap22d").exists()


def test_build_fabric_child_xml_well_formed_and_consistent(tmp_path):
    project = _build_project(tmp_path)
    result = build.build_fabric_child(project, now=NOW)
    stage = result.stage_dir

    # Both records parse (well-formed) ...
    _parse(stage / "fgdc.xml")
    _parse(stage / "iso.xml")
    # ... and carry the same title the MCF declared.
    title = result.mcf["identification"]["title"]
    assert title and title in (stage / "fgdc.xml").read_text()
    assert title in (stage / "iso.xml").read_text()


def test_build_fabric_child_checksums_list_every_file_but_not_self(tmp_path):
    project = _build_project(tmp_path)
    result = build.build_fabric_child(project, now=NOW)
    stage = result.stage_dir

    listed = _checksum_paths(stage)
    # Data + metadata records + manifest are all fingerprinted ...
    for name in (
        "fabric.gpkg",
        "manifest.json",
        "README.md",
        "fgdc.xml",
        "iso.xml",
        "aggregated/era5_land/era5_land_1980_agg.nc",
        "targets/runoff_targets.nc",
    ):
        assert name in listed, f"checksums.csv omits {name}"
    # ... but the integrity files never list themselves.
    assert CHECKSUMS_CSV not in listed
    assert SHA256SUMS not in listed

    # result.entries is the same record set, and SHA256SUMS verifies.
    assert {e.path for e in result.entries} == listed
    verify_csv(stage)  # no raise


def test_build_fabric_child_distribution_includes_extras_not_metadata(tmp_path):
    """distribution.online lists every staged downloadable artifact: data +
    manifest + README + checksums.csv + SHA256SUMS, but NOT the fgdc/iso
    metadata records (the design decision flagged in the PR)."""
    project = _build_project(tmp_path)
    result = build.build_fabric_child(project, now=NOW)

    downloads = _download_names(result.mcf)
    for name in (
        "fabric.gpkg",
        "manifest.json",
        "README.md",
        CHECKSUMS_CSV,
        SHA256SUMS,
        "aggregated/era5_land/era5_land_1980_agg.nc",
        "targets/runoff_targets.nc",
    ):
        assert name in downloads, f"distribution.online omits {name}"
    assert "fgdc.xml" not in downloads
    assert "iso.xml" not in downloads


def test_build_fabric_child_copy_mode_real_files(tmp_path):
    project = _build_project(tmp_path)
    result = build.build_fabric_child(project, copy=True, now=NOW)
    gpkg = result.stage_dir / "fabric.gpkg"
    assert not gpkg.is_symlink()
    assert gpkg.read_bytes() == b"GPKG-bytes"
    era = result.stage_dir / "aggregated" / "era5_land" / "era5_land_1980_agg.nc"
    assert not era.is_symlink()
    assert era.read_bytes() == b"era5-1980"
    verify_csv(result.stage_dir)


def test_build_is_deterministic_for_fixed_now(tmp_path):
    """A second build with the same now rewrites byte-identical metadata, and
    the re-build's distribution list still excludes the metadata records left
    on disk by the first build (the load-bearing skip-set in
    _staged_data_relpaths)."""
    project = _build_project(tmp_path)
    first = build.build_fabric_child(project, now=NOW)
    fgdc_bytes = (first.stage_dir / "fgdc.xml").read_bytes()
    iso_bytes = (first.stage_dir / "iso.xml").read_bytes()
    second = build.build_fabric_child(project, now=NOW)
    assert (second.stage_dir / "fgdc.xml").read_bytes() == fgdc_bytes
    assert (second.stage_dir / "iso.xml").read_bytes() == iso_bytes
    downloads = _download_names(second.mcf)
    assert "fgdc.xml" not in downloads
    assert "iso.xml" not in downloads


# ---------------------------------------------------------------------------
# source children
# ---------------------------------------------------------------------------


def test_build_source_child_data(tmp_path):
    project = _build_project(tmp_path)
    result = build.build_source_child(project, "era5_land", now=NOW)
    assert result.kind == "source" and result.key == "era5_land"
    stage = result.stage_dir
    assert stage == project.release_build_dir / "sources" / "era5_land"

    # Consolidated NCs staged (subpaths preserved) + metadata + integrity.
    assert (stage / "daily" / "era5_land_daily_1980.nc").is_symlink()
    assert (stage / "monthly" / "era5_land_monthly_1980.nc").is_symlink()
    for name in ("README.md", "fgdc.xml", "iso.xml", CHECKSUMS_CSV, SHA256SUMS):
        assert (stage / name).is_file()
    _parse(stage / "fgdc.xml")
    _parse(stage / "iso.xml")
    verify_csv(stage)

    # The consolidated NCs are the source child's distribution payload ...
    downloads = _download_names(result.mcf)
    assert "daily/era5_land_daily_1980.nc" in downloads
    assert "monthly/era5_land_monthly_1980.nc" in downloads
    # ... and the README / metadata records / integrity files are staged and
    # checksummed but deliberately NOT enumerated there (the inverse of the
    # fabric-child decision: the source-distribution builder labels every entry
    # a NetCDF, so listing them would mislabel them).
    for name in ("README.md", "fgdc.xml", "iso.xml", CHECKSUMS_CSV, SHA256SUMS):
        assert name not in downloads


def test_build_source_child_metadata_only_renders_without_data(tmp_path):
    """daymet (metadata_only) stages no data NCs but still gets a full set of
    README / FGDC / ISO. FGDC needs a bbox; daymet now carries one in the
    catalog (the bbox added in this PR)."""
    project = _build_project(tmp_path)
    result = build.build_source_child(project, "daymet", now=NOW)
    stage = result.stage_dir

    assert list(stage.rglob("*.nc")) == []  # no consolidated NC payload
    for name in ("README.md", "fgdc.xml", "iso.xml", CHECKSUMS_CSV, SHA256SUMS):
        assert (stage / name).is_file()
    _parse(stage / "fgdc.xml")  # would raise if daymet had no bbox
    _parse(stage / "iso.xml")
    verify_csv(stage)

    # No download entries -- only the upstream landing pointer.
    assert _download_names(result.mcf) == set()


# ---------------------------------------------------------------------------
# umbrella
# ---------------------------------------------------------------------------


def test_build_umbrella_unions_children(tmp_path):
    project = _build_project(tmp_path)
    result = build.build_umbrella(project, now=NOW)
    assert result.kind == "umbrella" and result.key is None
    stage = result.stage_dir
    assert stage == project.release_build_dir / "umbrella"

    for name in ("README.md", "fgdc.xml", "iso.xml", CHECKSUMS_CSV, SHA256SUMS):
        assert (stage / name).is_file()
    _parse(stage / "fgdc.xml")  # union bbox is non-None (fabric + sources)
    _parse(stage / "iso.xml")
    verify_csv(stage)

    # Umbrella is metadata-only: no data payload staged.
    assert list(stage.rglob("*.nc")) == []
    assert not (stage / "fabric.gpkg").exists()

    # The union bbox spans the fabric bbox + the (wider) source bboxes.
    bbox = result.mcf["identification"]["extents"]["spatial"][0]["bbox"]
    assert bbox is not None
    assert bbox[0] <= -125.0 and bbox[2] >= -66.0  # west/east at least the fabric

    # The temporal extent unions the children's periods (fabric targets span
    # 1980/2020 + 2000/2010, source periods differ); the union must start no
    # later than the earliest child begin.
    begin = result.mcf["identification"]["extents"]["temporal"][0]["begin"]
    assert begin is not None and begin <= "1980-01-01"


def test_build_umbrella_with_explicit_children(tmp_path):
    """Passing pre-built child MCFs unions exactly those (no local rebuild)."""
    project = _build_project(tmp_path)
    fab = build.build_fabric_child(project, now=NOW)
    result = build.build_umbrella(project, children=[fab.mcf], now=NOW)
    # Abstract reports the child count it was handed.
    assert "1 child" in result.mcf["identification"]["abstract"]


# ---------------------------------------------------------------------------
# build_all dispatcher
# ---------------------------------------------------------------------------


def test_build_all_default_scope_is_fabric(tmp_path):
    project = _build_project(tmp_path)
    results = build.build_all(project, now=NOW)
    assert [r.kind for r in results] == ["fabric"]


def test_build_all_scope_sources(tmp_path):
    project = _build_project(tmp_path)
    results = build.build_all(project, scope="sources", now=NOW)
    assert all(r.kind == "source" for r in results)
    keys = {r.key for r in results}
    assert keys == {"era5_land", "daymet"}  # publishable, present; watergap22d excluded


def test_build_all_scope_all(tmp_path):
    project = _build_project(tmp_path)
    results = build.build_all(project, scope="all", now=NOW)
    kinds = [r.kind for r in results]
    assert kinds[0] == "fabric"
    assert kinds[-1] == "umbrella"
    assert sorted(r.kind for r in results) == ["fabric", "source", "source", "umbrella"]

    # The umbrella unioned the children built in this invocation (fabric + 2
    # sources), so its abstract reports three child items.
    umbrella = results[-1]
    assert "3 child" in umbrella.mcf["identification"]["abstract"]
    # Every built item verifies on disk.
    for r in results:
        verify_csv(r.stage_dir)


def test_build_all_unknown_scope_raises(tmp_path):
    project = _build_project(tmp_path)
    with pytest.raises(ValueError, match="Unknown scope"):
        build.build_all(project, scope="bogus", now=NOW)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# progress logging (#295): phase-level INFO + meaningful -v/DEBUG
# ---------------------------------------------------------------------------

_BUILD_LOGGER = "nhf_spatial_targets.release.build"
_PAYLOAD_LOGGER = "nhf_spatial_targets.release.payload"


def test_fabric_build_emits_phase_info_lines(tmp_path, caplog):
    """At INFO the build narrates its phases instead of running silent (#295)."""
    project = _build_project(tmp_path)
    with caplog.at_level("INFO", logger=_BUILD_LOGGER):
        build.build_fabric_child(project, now=NOW)
    messages = [r.getMessage() for r in caplog.records]
    # entry line names the item, render line names the slow metadata step,
    # completion line reports the file count + destination.
    assert any("staging fabric child 'gfv_test'" in m for m in messages)
    assert any("rendering metadata (fgdc/iso/readme) for fabric" in m for m in messages)
    assert any(m.startswith("staged fabric 'gfv_test':") for m in messages)


def test_build_is_quiet_at_warning_level(tmp_path, caplog):
    """The phase lines are INFO, so a default WARNING console stays uncluttered."""
    project = _build_project(tmp_path)
    with caplog.at_level("WARNING", logger=_BUILD_LOGGER):
        build.build_fabric_child(project, now=NOW)
    assert [r for r in caplog.records if r.levelname == "INFO"] == []


def test_verbose_debug_surfaces_per_file_staging(tmp_path, caplog):
    """`-v`/DEBUG surfaces per-file staging in payload -- previously a no-op (#295).

    The build path had zero debug calls, so ``--verbose`` produced nothing extra
    for ``release build``. The payload write chokepoints now emit one debug line
    per staged file.
    """
    project = _build_project(tmp_path)
    with caplog.at_level("DEBUG", logger=_PAYLOAD_LOGGER):
        build.build_fabric_child(project, now=NOW)
    debug_msgs = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    # fabric.gpkg is symlinked; manifest.json is copied -- both chokepoints fire.
    assert any("-> " in m and "fabric.gpkg" in m for m in debug_msgs)
    assert any(m.startswith("copied") and "manifest.json" in m for m in debug_msgs)


def test_build_all_narrates_each_child(tmp_path, caplog):
    """scope=all emits a staging line per child kind (source + umbrella too)."""
    project = _build_project(tmp_path)
    with caplog.at_level("INFO", logger=_BUILD_LOGGER):
        build.build_all(project, scope="all", now=NOW)
    messages = [r.getMessage() for r in caplog.records]
    assert any("staging fabric child" in m for m in messages)
    assert any("staging source child 'era5_land'" in m for m in messages)
    assert any(m == "staging umbrella" for m in messages)


def test_build_all_scope_all_matches_payload_sources_used(tmp_path):
    """The dispatcher is catalog/manifest-driven: the source children it builds
    are exactly payload.sources_used (no hard-coded list)."""
    project = _build_project(tmp_path)
    results = build.build_all(project, scope="all", now=NOW)
    source_keys = sorted(r.key for r in results if r.kind == "source")
    assert source_keys == sorted(payload.sources_used(project))


# ---------------------------------------------------------------------------
# manifest / dangling-symlink failure modes + BuildResult invariant
# ---------------------------------------------------------------------------


def test_corrupt_manifest_is_a_loud_failure(tmp_path):
    """A corrupt manifest.json fails loudly rather than silently building a
    release with empty lineage + source list."""
    project = _build_project(tmp_path)
    project.manifest_path.write_text("{not valid json")
    with pytest.raises(ValueError, match="manifest.json.*corrupt"):
        build.build_fabric_child(project, now=NOW)


def test_missing_manifest_is_a_hard_error(tmp_path):
    """An absent manifest.json is fatal at staging (PR-3, #279).

    Staging must never emit a fabric child with no provenance snapshot, so
    ``stage_fabric_child`` (reached via ``build_fabric_child``) raises rather
    than silently skipping the copy. The operator runs validate /
    rebuild-manifest first."""
    from nhf_spatial_targets.release._models import ReleaseError

    project = _build_project(tmp_path)
    project.manifest_path.unlink()
    with pytest.raises(ReleaseError, match="manifest.json"):
        build.build_fabric_child(project, now=NOW)


def test_staged_data_relpaths_rejects_dangling_symlink(tmp_path):
    """A dangling staged symlink is rejected up front -- before any file is
    enumerated into distribution.online or rendered into the metadata records
    (mirrors checksums._iter_staged_files)."""
    stage = tmp_path / "stage"
    stage.mkdir()
    target = _touch(tmp_path / "source.nc", b"data")
    os.symlink(target, stage / "data.nc")
    target.unlink()  # the staged link now dangles
    with pytest.raises(FileNotFoundError, match="missing target"):
        build._staged_data_relpaths(stage)


def test_build_result_enforces_kind_key_invariant():
    """umbrella must be keyless; source/fabric must carry a key."""
    with pytest.raises(ValueError, match="key invariant violated"):
        build.BuildResult(
            kind="umbrella", key="oops", stage_dir=Path("/x"), mcf={}, entries=()
        )
    with pytest.raises(ValueError, match="key invariant violated"):
        build.BuildResult(
            kind="source", key=None, stage_dir=Path("/x"), mcf={}, entries=()
        )
