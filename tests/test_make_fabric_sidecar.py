"""Unit tests for scripts/make_fabric_sidecar.py (issue #355).

The sidecar is what lets a consumer name exactly which fabric they used.
The properties worth pinning are the ones a fabric developer would not
notice going wrong: that the checksum matches the artifact, that an
externally-owned identifier is flagged rather than presented as a key,
and that an unrecognised layer produces a prompt instead of a guess.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        },
        crs="EPSG:5070",
    ).to_file(path, layer="nsegment", driver="GPKG")
    return path


def _run(gpkg: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(gpkg),
            "--fabric",
            "xx",
            "--version",
            "3",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_sidecar_checksum_matches_the_geopackage(fabric_gpkg):
    """A wrong checksum is worse than none -- it certifies the wrong bytes."""
    assert _run(fabric_gpkg).returncode == 0
    doc = json.loads(fabric_gpkg.with_suffix(".json").read_text())
    expected = hashlib.sha256(fabric_gpkg.read_bytes()).hexdigest()
    assert doc["gpkg"]["sha256"] == expected
    assert doc["gpkg"]["bytes"] == fabric_gpkg.stat().st_size
    assert doc["gpkg"]["filename"] == fabric_gpkg.name


def test_sidecar_records_fabric_identity(fabric_gpkg):
    _run(fabric_gpkg, "--tag", "xx-v3", "--repo", "https://example.org/r")
    doc = json.loads(fabric_gpkg.with_suffix(".json").read_text())
    assert doc["fabric"] == "xx"
    assert doc["version"] == 3
    assert doc["crs"] == "EPSG:5070"
    assert doc["produced_by"]["tag"] == "xx-v3"
    assert doc["produced_by"]["repo"] == "https://example.org/r"


def test_sidecar_declares_the_primary_key_per_layer(fabric_gpkg):
    _run(fabric_gpkg)
    layers = json.loads(fabric_gpkg.with_suffix(".json").read_text())["layers"]
    assert layers["nhru"]["primary_key"] == "hru_id"
    assert layers["nsegment"]["primary_key"] == "segment_id"
    assert layers["nhru"]["features"] == 3
    assert layers["nsegment"]["features"] == 2


def test_sidecar_flags_externally_owned_ids_as_cross_references(fabric_gpkg):
    """nhm_* belongs to the national fabric and can be renumbered upstream.

    Presenting it as just another field is how issue #353 happened.
    """
    _run(fabric_gpkg)
    layers = json.loads(fabric_gpkg.with_suffix(".json").read_text())["layers"]
    assert "nhm_id" in layers["nhru"]["cross_reference_ids"]
    assert "nhm_seg_id" in layers["nsegment"]["cross_reference_ids"]
    # ...and they are never mistaken for the layer's own key.
    assert layers["nhru"]["primary_key"] != "nhm_id"


def test_sidecar_prompts_rather_than_guessing_an_unknown_key(tmp_path):
    """An unrecognised layer must yield null + a note, not an invented key."""
    path = tmp_path / "yy_v1.gpkg"
    gpd.GeoDataFrame(
        {"whatever": [1], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:5070"
    ).to_file(path, layer="mystery", driver="GPKG")

    proc = _run(path)
    assert proc.returncode == 0
    doc = json.loads(path.with_suffix(".json").read_text())
    assert doc["layers"]["mystery"]["primary_key"] is None
    assert "mystery" in proc.stdout


def test_sidecar_is_deterministic_apart_from_the_timestamp(fabric_gpkg):
    _run(fabric_gpkg)
    first = json.loads(fabric_gpkg.with_suffix(".json").read_text())
    _run(fabric_gpkg)
    second = json.loads(fabric_gpkg.with_suffix(".json").read_text())
    for doc in (first, second):
        doc["produced_by"].pop("generated_utc")
    assert first == second
