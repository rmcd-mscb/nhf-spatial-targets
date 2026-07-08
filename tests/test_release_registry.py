"""Tests for ``nhf_spatial_targets.release.registry``.

The registry is the publish-intent source of truth, so the invariants we
lock here are the ones that, if broken, silently orphan live ScienceBase
items or lose another operator's publish record:

- the ``umbrella: null`` sentinel round-trips as ``None`` (loaders must
  branch on it, never index a missing dict),
- the comment header and top-level key order survive every write (ruamel
  round-trip, not ``yaml.safe_dump``),
- ``put_*`` is read-merge-write: writing one entry never clobbers siblings
  or the other sections (the #97 manifest-overwrite class of bug),
- concurrent writers under flock lose no entries,
- a corrupt file is a loud failure, never a silent re-scaffold.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from nhf_spatial_targets.release import registry

# A scaffold carrying a distinctive comment header so the preservation test
# asserts on something unmistakable. Mirrors the committed file's shape.
_SCAFFOLD = """\
# SENTINEL-HEADER-LINE: this comment must survive every write.
# Second header line.

umbrella: null
consolidated_sources: {}
fabrics: {}
"""


@pytest.fixture
def reg_path(tmp_path: Path) -> Path:
    """A fresh registry file seeded from the scaffold, inside a tmp dir."""
    path = tmp_path / "release_registry.yml"
    path.write_text(_SCAFFOLD)
    return path


def _load_raw(path: Path):
    """Parse *path* with a bare ruamel handler (no normalization)."""
    return YAML().load(path.read_text())


# ---------------------------------------------------------------------------
# umbrella: null sentinel
# ---------------------------------------------------------------------------


def test_get_umbrella_returns_none_on_scaffold(reg_path: Path) -> None:
    """The ``umbrella: null`` sentinel reads back as ``None``."""
    assert registry.get_umbrella(reg_path) is None


def test_committed_registry_scaffold_is_unpublished() -> None:
    """The committed catalog/release_registry.yml is still the empty scaffold.

    Guards against accidentally committing real publish state (as happened
    once during development) -- the committed file must always start clean.
    """
    assert registry.get_umbrella(registry.DEFAULT_REGISTRY_PATH) is None
    assert registry.get_source("era5_land") is None
    assert registry.get_fabric("gfv2") is None


def test_load_registry_normalizes_missing_containers(tmp_path: Path) -> None:
    """A file missing the container sections still yields empty dicts."""
    path = tmp_path / "release_registry.yml"
    path.write_text("umbrella: null\n")
    reg = registry.load_registry(path)
    assert reg["consolidated_sources"] == {}
    assert reg["fabrics"] == {}
    assert reg.get("umbrella") is None


def test_load_registry_seeds_when_file_absent(tmp_path: Path) -> None:
    """An absent file is seeded in-memory (not crash) with the null sentinel."""
    path = tmp_path / "does_not_exist.yml"
    reg = registry.load_registry(path)
    assert reg.get("umbrella") is None
    assert reg["consolidated_sources"] == {}
    assert reg["fabrics"] == {}


# ---------------------------------------------------------------------------
# put_umbrella: round-trip + doi sentinel
# ---------------------------------------------------------------------------


def test_put_umbrella_round_trips(reg_path: Path) -> None:
    """A written umbrella reads back field-for-field."""
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="2026-05-28T00:00:00+00:00",
        title="NHF Calibration Targets",
        path=reg_path,
    )
    umbrella = registry.get_umbrella(reg_path)
    assert umbrella == {
        "sb_id": "umb1",
        "doi": None,
        "version": "1.0",
        "published_utc": "2026-05-28T00:00:00+00:00",
        "title": "NHF Calibration Targets",
    }


def test_put_umbrella_first_publish_leaves_doi_null(reg_path: Path) -> None:
    """Omitting ``doi`` on first publish records ``doi: null`` (no DOI yet)."""
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="2026-05-28T00:00:00+00:00",
        title="Title",
        path=reg_path,
    )
    assert registry.get_umbrella(reg_path)["doi"] is None


def test_put_umbrella_doi_omit_keeps_existing(reg_path: Path) -> None:
    """A later put that omits ``doi`` keeps a previously-set DOI.

    Mirrors the ``--refresh-doi`` flow's inverse: re-publishing the umbrella
    body must not blank a DOI that staff already minted.
    """
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="2026-05-28T00:00:00+00:00",
        title="Title",
        doi="10.5066/P9XXXXXX",
        path=reg_path,
    )
    # Re-publish the body without re-supplying the DOI.
    registry.put_umbrella(
        sb_id="umb1",
        version="1.1",
        published_utc="2026-06-01T00:00:00+00:00",
        title="Title v1.1",
        path=reg_path,
    )
    umbrella = registry.get_umbrella(reg_path)
    assert umbrella["doi"] == "10.5066/P9XXXXXX"
    assert umbrella["version"] == "1.1"


def test_put_umbrella_can_set_doi_explicitly(reg_path: Path) -> None:
    """Passing ``doi`` sets it (the refresh-doi path)."""
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="2026-05-28T00:00:00+00:00",
        title="Title",
        path=reg_path,
    )
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="2026-05-28T00:00:00+00:00",
        title="Title",
        doi="10.5066/P9NEWDOI",
        path=reg_path,
    )
    assert registry.get_umbrella(reg_path)["doi"] == "10.5066/P9NEWDOI"


# ---------------------------------------------------------------------------
# put_source / put_fabric: round-trip + merge-not-overwrite
# ---------------------------------------------------------------------------


def test_put_source_round_trips(reg_path: Path) -> None:
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="2026-05-28T01:00:00+00:00",
        file_count=3,
        total_bytes=12345,
        manifest_sha256="deadbeef",
        path=reg_path,
    )
    assert registry.get_source("era5_land", reg_path) == {
        "sb_id": "src1",
        "uploaded_utc": "2026-05-28T01:00:00+00:00",
        "file_count": 3,
        "total_bytes": 12345,
        "manifest_sha256": "deadbeef",
    }


def test_put_source_output_is_group_readable(reg_path: Path) -> None:
    """The registry .yml lands group-readable (0o666 & ~umask), not 0600.

    _atomic_dump writes via mkstemp (mode 0600); guards that the registry write
    path calls apply_umask_mode so the shared release registry is group-readable.
    """
    import os
    import stat

    old = os.umask(0o002)
    try:
        registry.put_source(
            "era5_land",
            sb_id="s",
            uploaded_utc="2026-05-28T01:00:00+00:00",
            file_count=1,
            total_bytes=1,
            manifest_sha256="ab",
            path=reg_path,
        )
        mode = stat.S_IMODE(reg_path.stat().st_mode)
        assert mode == 0o664, oct(mode)
    finally:
        os.umask(old)


def test_put_source_preserves_sibling_sources(reg_path: Path) -> None:
    """Writing one source must not drop a previously-written sibling (#97)."""
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="t1",
        file_count=1,
        total_bytes=10,
        manifest_sha256="aaa",
        path=reg_path,
    )
    registry.put_source(
        "merra2",
        sb_id="src2",
        uploaded_utc="t2",
        file_count=2,
        total_bytes=20,
        manifest_sha256="bbb",
        path=reg_path,
    )
    assert registry.get_source("era5_land", reg_path)["sb_id"] == "src1"
    assert registry.get_source("merra2", reg_path)["sb_id"] == "src2"


def test_put_source_updates_in_place(reg_path: Path) -> None:
    """Re-publishing a source updates that entry without duplicating it."""
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="t1",
        file_count=1,
        total_bytes=10,
        manifest_sha256="aaa",
        path=reg_path,
    )
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="t2",
        file_count=5,
        total_bytes=99,
        manifest_sha256="zzz",
        path=reg_path,
    )
    entry = registry.get_source("era5_land", reg_path)
    assert entry["file_count"] == 5
    assert entry["manifest_sha256"] == "zzz"
    # exactly one entry under consolidated_sources
    assert list(registry.load_registry(reg_path)["consolidated_sources"]) == [
        "era5_land"
    ]


def test_put_fabric_round_trips_and_preserves_siblings(reg_path: Path) -> None:
    registry.put_fabric(
        "gfv2",
        sb_id="fab1",
        uploaded_utc="t1",
        file_count=40,
        total_bytes=1000,
        manifest_sha256="f1",
        path=reg_path,
    )
    registry.put_fabric(
        "oregon",
        sb_id="fab2",
        uploaded_utc="t2",
        file_count=12,
        total_bytes=500,
        manifest_sha256="f2",
        path=reg_path,
    )
    assert registry.get_fabric("gfv2", reg_path)["sb_id"] == "fab1"
    assert registry.get_fabric("oregon", reg_path)["sb_id"] == "fab2"


def test_writes_across_sections_do_not_interfere(reg_path: Path) -> None:
    """Umbrella, sources, and fabrics each survive writes to the others."""
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="t0",
        title="Title",
        path=reg_path,
    )
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="t1",
        file_count=1,
        total_bytes=10,
        manifest_sha256="aaa",
        path=reg_path,
    )
    registry.put_fabric(
        "gfv2",
        sb_id="fab1",
        uploaded_utc="t2",
        file_count=40,
        total_bytes=1000,
        manifest_sha256="f1",
        path=reg_path,
    )
    assert registry.get_umbrella(reg_path)["sb_id"] == "umb1"
    assert registry.get_source("era5_land", reg_path)["sb_id"] == "src1"
    assert registry.get_fabric("gfv2", reg_path)["sb_id"] == "fab1"


# ---------------------------------------------------------------------------
# Comment header + key order preservation
# ---------------------------------------------------------------------------


def test_comment_header_survives_writes(reg_path: Path) -> None:
    """The scaffold's comment header is preserved through put_* writes.

    This is the whole reason we use ruamel round-trip mode instead of
    yaml.safe_dump.
    """
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="t0",
        title="Title",
        path=reg_path,
    )
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="t1",
        file_count=1,
        total_bytes=10,
        manifest_sha256="aaa",
        path=reg_path,
    )
    text = reg_path.read_text()
    assert "# SENTINEL-HEADER-LINE: this comment must survive every write." in text
    assert "# Second header line." in text


def test_top_level_key_order_preserved(reg_path: Path) -> None:
    """Top-level keys stay umbrella -> consolidated_sources -> fabrics."""
    registry.put_fabric(
        "gfv2",
        sb_id="fab1",
        uploaded_utc="t1",
        file_count=1,
        total_bytes=1,
        manifest_sha256="f1",
        path=reg_path,
    )
    raw = _load_raw(reg_path)
    assert list(raw) == ["umbrella", "consolidated_sources", "fabrics"]


def test_umbrella_field_order_canonical(reg_path: Path) -> None:
    """Serialized umbrella fields follow the canonical schema order."""
    registry.put_umbrella(
        sb_id="umb1",
        version="1.0",
        published_utc="t0",
        title="Title",
        path=reg_path,
    )
    raw = _load_raw(reg_path)
    assert list(raw["umbrella"]) == list(registry.UMBRELLA_FIELDS)


# ---------------------------------------------------------------------------
# Corrupt-file loud failure
# ---------------------------------------------------------------------------


def test_corrupt_registry_is_loud_failure(tmp_path: Path) -> None:
    """An unparseable registry raises ValueError, never a silent reset."""
    path = tmp_path / "release_registry.yml"
    path.write_text("umbrella: [unterminated\nconsolidated_sources: {}\n")
    with pytest.raises(ValueError, match="corrupt"):
        registry.load_registry(path)


def test_non_mapping_top_level_is_loud_failure(tmp_path: Path) -> None:
    """A top-level list (not mapping) raises rather than being mis-handled."""
    path = tmp_path / "release_registry.yml"
    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        registry.load_registry(path)


def test_present_but_empty_file_is_loud_failure(tmp_path: Path) -> None:
    """A present-but-empty file is corruption, not a fresh start.

    A registry truncated to zero bytes (crashed writer) must never be silently
    re-scaffolded over live publish state -- only a genuinely *absent* file
    seeds. This is the asymmetry three reviewers flagged.
    """
    path = tmp_path / "release_registry.yml"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        registry.load_registry(path)


def test_atomic_dump_failure_leaves_original_intact(
    tmp_path: Path, monkeypatch
) -> None:
    """A serialization failure mid-write leaves no .tmp litter and an
    unchanged original -- the atomic-write safety net for the #97-class
    concern this module exists to prevent.
    """
    path = tmp_path / "release_registry.yml"
    path.write_text(_SCAFFOLD)
    before = path.read_text()

    # Force the rename to fail after the tempfile is written.
    def _boom(self, target):  # noqa: ANN001
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        registry.put_source(
            "era5_land",
            sb_id="src1",
            uploaded_utc="t1",
            file_count=1,
            total_bytes=10,
            manifest_sha256="aaa",
            path=path,
        )

    # Original content untouched and no leftover *.yml.tmp files.
    assert path.read_text() == before
    assert not list(tmp_path.glob("*.yml.tmp"))


def test_put_does_not_reset_on_reload(reg_path: Path) -> None:
    """A second writer reloads fresh state, never an in-memory stale copy."""
    registry.put_source(
        "era5_land",
        sb_id="src1",
        uploaded_utc="t1",
        file_count=1,
        total_bytes=10,
        manifest_sha256="aaa",
        path=reg_path,
    )
    # Simulate an out-of-band edit landing between writes.
    text = reg_path.read_text()
    reg_path.write_text(text.replace("src1", "src1-edited"))
    registry.put_source(
        "merra2",
        sb_id="src2",
        uploaded_utc="t2",
        file_count=2,
        total_bytes=20,
        manifest_sha256="bbb",
        path=reg_path,
    )
    # The out-of-band edit survived (we merged, didn't overwrite from a
    # stale in-memory snapshot).
    assert registry.get_source("era5_land", reg_path)["sb_id"] == "src1-edited"
    assert registry.get_source("merra2", reg_path)["sb_id"] == "src2"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_writers_lose_no_entries(reg_path: Path) -> None:
    """4-thread concurrent put_source produces exactly 4 entries, none lost.

    Mirrors the lineage concurrency test: the flock serializes the
    read-merge-write so a naive load/append/dump can't drop a sibling. A
    ``Barrier`` releases all workers into ``put_source`` simultaneously so the
    read-modify-write windows genuinely overlap (without it, the GIL tends to
    let each worker finish before the next starts, and the test would pass
    even if the lock did nothing).
    """
    n_threads = 4
    barrier = threading.Barrier(n_threads)

    def _worker(worker_id: int) -> None:
        barrier.wait()
        registry.put_source(
            f"src_{worker_id}",
            sb_id=f"sb_{worker_id}",
            uploaded_utc="t",
            file_count=worker_id,
            total_bytes=worker_id * 10,
            manifest_sha256=f"h{worker_id}",
            path=reg_path,
        )

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reg = registry.load_registry(reg_path)
    assert set(reg["consolidated_sources"]) == {f"src_{i}" for i in range(n_threads)}
    # File still parses cleanly (no torn write) and the header survived.
    assert "# SENTINEL-HEADER-LINE" in reg_path.read_text()
