"""Golden-file tests for nhf_spatial_targets.release.mcf.

Each builder is run from canned fixtures with an injected timestamp, then
the resulting MCF dict is deep-equal-compared against a committed golden
JSON. Regenerate the goldens after an intentional change with::

    REGEN_GOLDENS=1 pixi run -e dev test -k test_release_mcf

The per-source goldens (era5_land, snodas) are representative examples, not
an exhaustive registry: adding a new catalog source must not require
touching them. Spot-check tests assert the load-bearing invariants
(bbox conversion, publishability filtering, metadata-only handling) so the
intent is visible without diffing the large golden dicts, and an
``ISO19139`` render smoke-test protects the shape the ISO emitter consumes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pygeometa.schemas.iso19139 import ISO19139OutputSchema

from nhf_spatial_targets.defaults import apply_defaults
from nhf_spatial_targets.release import mcf
from nhf_spatial_targets.release.defaults import load_release_defaults

FX = Path(__file__).parent / "fixtures" / "release"

# Injected so metadata.datestamp / dates are byte-stable in the golden.
NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)

# Canned staged-file lists (the build orchestrator supplies these from the
# payload stager).
SRC_FILES = {
    "era5_land": [
        "monthly/era5_land_monthly_2003.nc",
        "daily/era5_land_daily_2003.nc",
    ],
    "snodas": ["daily/snodas_2003.nc"],
}
FABRIC_FILES = [
    "fabric.gpkg",
    "aggregated/era5_land/era5_land_2003.nc",
    "aggregated/snodas/snodas_2003.nc",
    "targets/runoff_targets.nc",
    "targets/swe_targets.nc",
    "manifest.json",
]


@pytest.fixture
def defaults() -> dict:
    return load_release_defaults(FX / "fixture_release_defaults.yml")


@pytest.fixture
def config() -> dict:
    return apply_defaults(
        yaml.safe_load((FX / "fixture_project_config.yml").read_text())
    )


@pytest.fixture
def fabric() -> dict:
    return json.loads((FX / "fixture_fabric.json").read_text())


@pytest.fixture
def manifest() -> dict:
    return json.loads((FX / "fixture_manifest.json").read_text())


def _norm(obj: object) -> object:
    """JSON round-trip so tuples/types match the golden's parsed shape."""
    return json.loads(json.dumps(obj))


def _check_golden(built: dict, name: str) -> None:
    """Deep-equal *built* against golden *name*; regenerate when asked."""
    path = FX / name
    norm = _norm(built)
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(json.dumps(norm, indent=2, ensure_ascii=False) + "\n")
    assert norm == json.loads(path.read_text())


def _build_source(key: str, defaults: dict, manifest: dict) -> dict:
    return mcf.build_source_mcf(
        key,
        defaults=defaults,
        manifest=manifest,
        distribution_files=SRC_FILES.get(key),
        now=NOW,
    )


def _build_fabric(config: dict, fabric: dict, defaults: dict, manifest: dict) -> dict:
    return mcf.build_fabric_mcf(
        config=config,
        fabric=fabric,
        defaults=defaults,
        manifest=manifest,
        distribution_files=FABRIC_FILES,
        now=NOW,
    )


def _build_umbrella(defaults: dict, children: list[dict], authors: list[dict]) -> dict:
    return mcf.build_umbrella_mcf(
        defaults=defaults,
        children=children,
        authors=authors,
        version="1.0",
        now=NOW,
    )


# ---------------------------------------------------------------------------
# Golden-file tests
# ---------------------------------------------------------------------------


def test_source_era5_mcf_golden(defaults, manifest):
    built = _build_source("era5_land", defaults, manifest)
    _check_golden(built, "expected_source_era5_mcf.json")


def test_source_snodas_mcf_golden(defaults, manifest):
    built = _build_source("snodas", defaults, manifest)
    _check_golden(built, "expected_source_snodas_mcf.json")


def test_fabric_mcf_golden(config, fabric, defaults, manifest):
    built = _build_fabric(config, fabric, defaults, manifest)
    _check_golden(built, "expected_fabric_gfv2_mcf.json")


def test_umbrella_mcf_golden(config, fabric, defaults, manifest):
    children = [
        _build_source("era5_land", defaults, manifest),
        _build_source("snodas", defaults, manifest),
        _build_fabric(config, fabric, defaults, manifest),
    ]
    built = _build_umbrella(defaults, children, config["release"]["authors"])
    _check_golden(built, "expected_umbrella_mcf.json")


# ---------------------------------------------------------------------------
# Invariant spot-checks (intent visible without diffing the big goldens)
# ---------------------------------------------------------------------------


def test_bbox_nwse_converted_to_wsen(defaults, manifest):
    """Catalog bbox_nwse [N, W, S, E] becomes MCF [W, S, E, N]."""
    built = _build_source("era5_land", defaults, manifest)
    bbox = built["identification"]["extents"]["spatial"][0]["bbox"]
    assert bbox == [-125.0, 24.7, -66.0, 53.0]


def test_source_without_bbox_is_none(defaults, manifest):
    """A source with no access.bbox_nwse gets an honest None MCF bbox.

    merra2 is global and subsets at aggregation time, so the catalog carries
    no bbox for it. (SNODAS used to be the example here, but it was given a
    catalog CONUS bbox so its FGDC <bounding> is valid; the code-side None
    path is also covered by test_fabric_bbox_absent_is_none.)
    """
    built = _build_source("merra2", defaults, manifest)
    assert built["identification"]["extents"]["spatial"][0]["bbox"] is None


def test_metadata_only_source_has_no_download_entries(defaults, manifest):
    """daymet (metadata_only) carries only the upstream landing entry."""
    built = mcf.build_source_mcf(
        "daymet", defaults=defaults, manifest=manifest, now=NOW
    )
    functions = {e["function"] for e in built["distribution"].values()}
    assert functions == {"information"}


def test_fabric_excludes_nonpublishable_and_superseded(
    config, fabric, defaults, manifest
):
    """Participating sources skip watergap22d (publishable:false) + superseded."""
    built = _build_fabric(config, fabric, defaults, manifest)
    names = [e["name"] for e in built["identification"]["entityandattribute"]]
    # eainfo lists targets, not sources; sources surface via useconstraints +
    # the abstract. Assert the abstract names only publishable participants.
    abstract = built["identification"]["abstract"]
    assert "watergap22d" not in abstract
    assert "merra_land" not in abstract
    assert "era5_land" in abstract and "snodas" in abstract
    # eainfo carries the enabled targets.
    assert names == ["runoff", "aet", "snow_water_equivalent"]


def test_injected_timestamp_used(defaults, manifest):
    built = _build_source("era5_land", defaults, manifest)
    assert built["metadata"]["datestamp"] == "2026-05-28"
    assert built["metadata"]["dates"]["creation"] == "2026-05-28"


def test_source_lineage_filtered_to_source_key(defaults, manifest):
    """Source child process steps are only those tagged with its key."""
    built = _build_source("era5_land", defaults, manifest)
    steps = built["dataquality"]["lineage"]["processstep"]
    # fixture has 2 era5_land steps (consolidate + aggregate) and 3 others.
    assert len(steps) == 2
    assert all(s["procdate"].startswith("2026") for s in steps)


def test_fabric_lineage_includes_all_steps(config, fabric, defaults, manifest):
    built = _build_fabric(config, fabric, defaults, manifest)
    assert len(built["dataquality"]["lineage"]["processstep"]) == 5


def test_umbrella_unions_child_extents(config, fabric, defaults, manifest):
    children = [
        _build_source("era5_land", defaults, manifest),
        _build_fabric(config, fabric, defaults, manifest),
    ]
    built = _build_umbrella(defaults, children, config["release"]["authors"])
    bbox = built["identification"]["extents"]["spatial"][0]["bbox"]
    # union of era5 [-125,24.7,-66,53] and fabric [-124.7,24.5,-66.95,49.4]
    assert bbox == [-125.0, 24.5, -66.0, 53.0]
    assert built["identification"]["extents"]["temporal"][0]["end"] is None


@pytest.mark.parametrize("kind", ["era5_land", "snodas", "daymet"])
def test_source_mcf_renders_to_iso(kind, defaults, manifest):
    """The MCF shape is consumable by the ISO emitter (smoke)."""
    built = mcf.build_source_mcf(kind, defaults=defaults, manifest=manifest, now=NOW)
    xml = ISO19139OutputSchema().write(built)
    assert xml.lstrip().startswith("<?xml")


def test_fabric_and_umbrella_render_to_iso(config, fabric, defaults, manifest):
    fab = _build_fabric(config, fabric, defaults, manifest)
    umb = _build_umbrella(defaults, [fab], config["release"]["authors"])
    assert ISO19139OutputSchema().write(fab).lstrip().startswith("<?xml")
    assert ISO19139OutputSchema().write(umb).lstrip().startswith("<?xml")


# ---------------------------------------------------------------------------
# Attribution-snippet selection (license-substring, first-match-wins)
# ---------------------------------------------------------------------------


def test_daymet_attribution_is_ornl_not_earthdata(defaults, manifest):
    """daymet license names both NASA and ORNL; ORNL must win (it's ORNL DAAC)."""
    built = mcf.build_source_mcf(
        "daymet", defaults=defaults, manifest=manifest, now=NOW
    )
    useconst = built["identification"]["useconstraints"]
    assert "Oak Ridge National Laboratory" in useconst
    assert "NASA Earthdata system" not in useconst


def test_nasa_source_attribution_is_earthdata(defaults, manifest):
    """A NASA-licensed source with no ORNL/NSIDC token gets the Earthdata snippet."""
    built = mcf.build_source_mcf(
        "gldas_noah_v21_monthly", defaults=defaults, manifest=manifest, now=NOW
    )
    assert "NASA Earthdata system" in built["identification"]["useconstraints"]


# ---------------------------------------------------------------------------
# Edge cases the goldens don't exercise
# ---------------------------------------------------------------------------


def test_empty_children_umbrella(defaults):
    built = mcf.build_umbrella_mcf(defaults=defaults, children=[], authors=[], now=NOW)
    assert built["identification"]["extents"]["spatial"][0]["bbox"] is None
    assert built["identification"]["extents"]["temporal"][0] == {
        "begin": None,
        "end": None,
    }
    # n_children == 0 renders without the child-title clause.
    assert "0 child items" in built["identification"]["abstract"]
    assert ISO19139OutputSchema().write(built).lstrip().startswith("<?xml")


def test_fabric_label_falls_back_to_path_stem(config, fabric, defaults, manifest):
    """With no release.fabric_label, the label is the fabric path stem."""
    config = {**config, "release": {**config["release"], "fabric_label": None}}
    built = _build_fabric(config, fabric, defaults, manifest)
    assert "gfv2_nhru_merged" in built["identification"]["title"]


def test_fabric_bbox_malformed_raises(config, defaults, manifest):
    """A corrupted fabric.json bbox is loud, not a silent None."""
    bad_fabric = {"hru_count": 1, "bbox": {"minx": -1.0, "miny": 0.0}}  # missing keys
    with pytest.raises(ValueError, match="bbox is malformed"):
        _build_fabric(config, bad_fabric, defaults, manifest)


def test_fabric_bbox_absent_is_none(config, defaults, manifest):
    """A genuinely-absent bbox is None (not an error)."""
    built = _build_fabric(config, {"hru_count": 1}, defaults, manifest)
    assert built["identification"]["extents"]["spatial"][0]["bbox"] is None


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("1979/present", ("1979-01-01", None)),
        ("2000/2010", ("2000-01-01", "2010-12-31")),
        ("1980/2016-02-29", ("1980-01-01", "2016-02-29")),
        (None, (None, None)),
        ("2003", (None, None)),
    ],
)
def test_parse_period(period, expected):
    assert mcf._parse_period(period) == expected


@pytest.mark.parametrize(
    ("doi", "expected"),
    [
        (None, None),
        ("10.5066/PXXXX", "https://doi.org/10.5066/PXXXX"),
        ("doi:10.5066/PXXXX", "https://doi.org/10.5066/PXXXX"),
        ("https://doi.org/10.5066/PXXXX", "https://doi.org/10.5066/PXXXX"),
    ],
)
def test_doi_url(doi, expected):
    assert mcf._doi_url(doi) == expected
