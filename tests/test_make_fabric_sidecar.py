"""Unit tests for scripts/make_fabric_sidecar.py (issue #355).

The sidecar is what lets a consumer name exactly which fabric they used.
The properties worth pinning are the ones a fabric developer would not
notice going wrong: that the checksum matches the artifact, that an
externally-owned identifier is flagged rather than presented as a key,
that an unrecognised layer produces a prompt instead of a guess, and that
a declared key which does not actually identify rows is refused.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_fabric_sidecar.py"


@pytest.fixture(scope="module")
def sidecar_mod():
    spec = importlib.util.spec_from_file_location("make_fabric_sidecar", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fabric_gpkg(tmp_path: Path) -> Path:
    """A two-layer GeoPackage shaped like a real model fabric."""
    path = tmp_path / "xx_v3.gpkg"
    gpd.GeoDataFrame(
        {
            "hru_id": [1, 2, 3],
            "nhm_id": [4010, 4011, 4012],
            "hru_segment": [1, 1, 2],
            "geometry": [box(i, 0, i + 1, 1) for i in range(3)],
        },
        crs="EPSG:5070",
    ).to_file(path, layer="nhru", driver="GPKG")
    gpd.GeoDataFrame(
        {
            "segment_id": [1, 2],
            "nhm_seg_id": [900, 901],
            "to_nhm_seg": [901, 0],
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        },
        crs="EPSG:5070",
    ).to_file(path, layer="nsegment", driver="GPKG")
    return path


def _one_layer(tmp_path: Path, layer: str, data: dict, name="yy_v1") -> Path:
    path = tmp_path / f"{name}.gpkg"
    n = len(next(iter(data.values())))
    gpd.GeoDataFrame(
        {**data, "geometry": [box(i, 0, i + 1, 1) for i in range(n)]},
        crs="EPSG:5070",
    ).to_file(path, layer=layer, driver="GPKG")
    return path


def _run(
    gpkg: Path, *extra: str, fabric="xx", version="3", repo_dir=REPO_ROOT
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(gpkg),
            "--fabric",
            fabric,
            "--version",
            version,
            "--repo-dir",
            str(repo_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _sidecar(gpkg: Path) -> dict:
    return json.loads(gpkg.with_suffix(".json").read_text())


def test_sidecar_checksum_matches_the_geopackage(fabric_gpkg):
    """A wrong checksum is worse than none -- it certifies the wrong bytes."""
    assert _run(fabric_gpkg).returncode == 0
    doc = _sidecar(fabric_gpkg)
    expected = hashlib.sha256(fabric_gpkg.read_bytes()).hexdigest()
    assert doc["gpkg"]["sha256"] == expected
    assert doc["gpkg"]["bytes"] == fabric_gpkg.stat().st_size
    assert doc["gpkg"]["filename"] == fabric_gpkg.name


def test_sha256_spans_multiple_read_chunks(sidecar_mod, tmp_path):
    """Real fabrics are hundreds of MB; the chunk loop must cover every byte."""
    path = tmp_path / "big.bin"
    data = os.urandom((5 << 20) // 2)  # 2.5 MiB -> three 1 MiB reads
    path.write_bytes(data)
    assert sidecar_mod.sha256(path) == hashlib.sha256(data).hexdigest()


def test_sidecar_records_fabric_identity(fabric_gpkg):
    proc = _run(fabric_gpkg, "--tag", "xx-v3", "--repo", "https://example.org/r")
    assert proc.returncode == 0, proc.stderr
    doc = _sidecar(fabric_gpkg)
    assert doc["fabric"] == "xx"
    assert doc["version"] == 3
    assert doc["crs"] == "EPSG:5070"
    assert doc["produced_by"]["tag"] == "xx-v3"
    assert doc["produced_by"]["repo"] == "https://example.org/r"
    assert len(doc["produced_by"]["commit"]) == 40


def test_sidecar_declares_the_primary_key_per_layer(fabric_gpkg):
    proc = _run(fabric_gpkg)
    assert proc.returncode == 0, proc.stderr
    layers = _sidecar(fabric_gpkg)["layers"]
    assert layers["nhru"]["primary_key"] == "hru_id"
    assert layers["nsegment"]["primary_key"] == "segment_id"
    assert layers["nhru"]["features"] == 3
    assert layers["nsegment"]["features"] == 2
    assert layers["nhru"]["crs"] == "EPSG:5070"
    # Every layer's key is known, so there is nothing to prompt about.
    assert "NOTE" not in proc.stderr


def test_sidecar_flags_externally_owned_ids_as_cross_references(fabric_gpkg):
    """nhm_* belongs to the national fabric and can be renumbered upstream.

    Presenting it as just another field is how issue #353 happened.
    """
    assert _run(fabric_gpkg).returncode == 0
    layers = _sidecar(fabric_gpkg)["layers"]
    assert "nhm_id" in layers["nhru"]["cross_reference_ids"]
    assert "nhm_seg_id" in layers["nsegment"]["cross_reference_ids"]
    assert "to_nhm_seg" in layers["nsegment"]["cross_reference_ids"]
    # ...and they are never mistaken for the layer's own key.
    assert layers["nhru"]["primary_key"] != "nhm_id"


def test_extra_cross_references_can_be_declared(fabric_gpkg):
    assert _run(fabric_gpkg, "--xref", "hru_segment").returncode == 0
    assert (
        "hru_segment" in _sidecar(fabric_gpkg)["layers"]["nhru"]["cross_reference_ids"]
    )


def test_sidecar_prompts_rather_than_guessing_an_unknown_key(tmp_path):
    """An unrecognised layer must yield null + a note on stderr, not a key."""
    path = _one_layer(tmp_path, "mystery", {"whatever": [1]})
    proc = _run(path, fabric="yy", version="1")
    assert proc.returncode == 0
    assert _sidecar(path)["layers"]["mystery"]["primary_key"] is None
    assert "NOTE: no primary_key for: mystery" in proc.stderr


def test_key_flag_declares_an_unknown_layers_key(tmp_path):
    """--key survives a rebuild, unlike a hand edit to the JSON."""
    path = _one_layer(tmp_path, "domain", {"domain_id": [7]})
    proc = _run(path, "--key", "domain=domain_id", fabric="yy", version="1")
    assert proc.returncode == 0, proc.stderr
    assert _sidecar(path)["layers"]["domain"]["primary_key"] == "domain_id"
    assert "NOTE" not in proc.stderr


def test_key_flag_rejects_a_layer_not_in_the_geopackage(fabric_gpkg):
    proc = _run(fabric_gpkg, "--key", "nope=x")
    assert proc.returncode != 0
    assert "nope" in proc.stderr
    assert not fabric_gpkg.with_suffix(".json").exists()


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"nhm_id": [1, 2]}, "is not a column"),  # the #353 shape: no own key
        ({"hru_id": [1, 1, 2]}, "duplicate"),
        ({"hru_id": [1.0, None, 2.0]}, "nulls"),
    ],
)
def test_a_declared_key_that_does_not_identify_rows_is_refused(tmp_path, data, reason):
    """A sidecar asserting a bad key is worse than none -- refuse to write it."""
    path = _one_layer(tmp_path, "nhru", data)
    proc = _run(path, fabric="yy", version="1")
    assert proc.returncode == 1
    assert reason in proc.stderr
    assert not path.with_suffix(".json").exists()


def test_non_geopackage_input_is_refused(tmp_path):
    """A GeoJSON named .json would otherwise be overwritten by its own sidecar."""
    path = tmp_path / "xx_v3.json"
    gpd.GeoDataFrame(
        {"hru_id": [1], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:5070"
    ).to_file(path, driver="GeoJSON")
    before = path.read_bytes()
    proc = _run(path)
    assert proc.returncode != 0
    assert path.read_bytes() == before


def test_missing_git_commit_warns_instead_of_passing_silently(fabric_gpkg, tmp_path):
    proc = _run(fabric_gpkg, repo_dir=tmp_path)
    assert proc.returncode == 0
    assert _sidecar(fabric_gpkg)["produced_by"]["commit"] is None
    assert "no git commit recorded" in proc.stderr


def test_filename_not_matching_fabric_and_version_warns(fabric_gpkg):
    proc = _run(fabric_gpkg, version="4")
    assert proc.returncode == 0
    assert "xx_v4.gpkg" in proc.stderr


def test_sidecar_is_deterministic_apart_from_the_timestamp(fabric_gpkg):
    out = fabric_gpkg.with_suffix(".json")
    assert _run(fabric_gpkg).returncode == 0
    first = json.loads(out.read_text())
    out.unlink()
    assert _run(fabric_gpkg).returncode == 0
    second = json.loads(out.read_text())
    for doc in (first, second):
        doc["produced_by"].pop("generated_utc")
    assert first == second
