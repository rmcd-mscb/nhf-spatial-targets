"""Tests for nhf-targets upgrade-config (#193 follow-up).

Detection is the contract: a feature is in sync if its `detect` regex matches
the project's config.yml text in any form — live value, the commented stub
operator pasted, or the commented stub from a fresh init template.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nhf_spatial_targets.upgrade_config import (
    OPTIONAL_CONFIG_FEATURES,
    OptionalConfigFeature,
    check_drift,
)


def _write_minimal_config(project_dir: Path, body: str = "") -> None:
    """Write a syntactically-valid project config with optional extra body."""
    project_dir.mkdir(parents=True, exist_ok=True)
    text = f"fabric:\n  path: /x.gpkg\n  id_col: nhm_id\ndatastore: /x\n{body}"
    (project_dir / "config.yml").write_text(text)


# --- check_drift detection ---------------------------------------------------


def test_check_drift_reports_all_features_when_none_present(tmp_path):
    _write_minimal_config(tmp_path)
    missing = check_drift(tmp_path)
    names = {f.name for f in missing}
    # All shipped features should be reported on a bare config.
    assert {
        "fabric.token",
        "representative_points",
        "release",
        "snow_covered_area.depth_threshold_mm",
        "snow_covered_area.forced_zero_combined",
        "snow_covered_area.min_sources_for_bound",
    } <= names


def test_check_drift_clean_when_live_value_present(tmp_path):
    """A real `representative_points:` block counts as in-sync."""
    _write_minimal_config(
        tmp_path,
        body=('representative_points:\n  aet:\n    "Region A": [-121.7, 45.4]\n'),
    )
    missing = check_drift(tmp_path)
    assert "representative_points" not in {f.name for f in missing}


def test_check_drift_clean_when_commented_stub_present(tmp_path):
    """Operator pasted the stub but hasn't enabled — still in-sync (the
    addition is discoverable in their file, which is the whole point)."""
    _write_minimal_config(
        tmp_path,
        body="# representative_points:\n#   aet:\n#     X: [-120, 45]\n",
    )
    missing = check_drift(tmp_path)
    assert "representative_points" not in {f.name for f in missing}


def test_check_drift_detects_nested_fabric_token_commented(tmp_path):
    """fabric.token is a nested key; both `# token: or` and `token: or`
    (under the fabric: block) count as in-sync."""
    _write_minimal_config(tmp_path, body="")
    # Append the stub form the operator would paste under fabric:
    cfg = tmp_path / "config.yml"
    cfg.write_text(cfg.read_text().rstrip() + "\n  # token: or\n")
    missing = check_drift(tmp_path)
    assert "fabric.token" not in {f.name for f in missing}


def test_check_drift_raises_when_config_missing(tmp_path):
    """Invalid project dir (no config.yml) raises rather than silently passing."""
    with pytest.raises(FileNotFoundError):
        check_drift(tmp_path / "does-not-exist")


def test_check_drift_release_detects_commented_stub(tmp_path):
    """The release: stub (commented or live) counts as in-sync."""
    _write_minimal_config(tmp_path, body="# release:\n#   ipds_number: IP-XXXXXX\n")
    assert "release" not in {f.name for f in check_drift(tmp_path)}


def test_check_drift_release_detects_live_block(tmp_path):
    """An uncommented `release:` block also counts as in-sync."""
    _write_minimal_config(
        tmp_path,
        body=(
            "release:\n"
            "  authors:\n"
            "    - given: Jane\n"
            "      family: Doe\n"
            "  ipds_number: IP-XXXXXX\n"
        ),
    )
    assert "release" not in {f.name for f in check_drift(tmp_path)}


# --- registry shape ---------------------------------------------------------


def test_each_registry_feature_has_actionable_metadata():
    """Every registered feature carries enough to render a useful report:
    a non-trivial detect regex, a non-empty block, and provenance."""
    for feat in OPTIONAL_CONFIG_FEATURES:
        assert isinstance(feat, OptionalConfigFeature)
        assert feat.name
        assert feat.detect and ":" in feat.detect  # regex covers a yaml key
        assert feat.block.strip()
        assert feat.added  # provenance shown in the report
        assert feat.why


# --- CLI: exit codes ---------------------------------------------------------
#
# Cyclopts apps are callable as ``app([...])`` and exit via SystemExit, so we
# invoke in-process (matching tests/test_cli_agg.py's pattern) rather than
# spawning a subprocess.


def test_cli_exits_one_on_drift(tmp_path, capsys):
    from nhf_spatial_targets.cli import app

    _write_minimal_config(tmp_path)
    with pytest.raises(SystemExit) as exc:
        app(["upgrade-config", "--project-dir", str(tmp_path)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    # The missing feature names appear in stdout so the operator can act.
    assert "representative_points" in out


def test_cli_exits_zero_when_in_sync(tmp_path, capsys):
    from nhf_spatial_targets.cli import app

    # Operator pasted all commented stubs from the latest template.
    (tmp_path / "config.yml").write_text(
        "fabric:\n"
        "  path: /x.gpkg\n"
        "  id_col: nhm_id\n"
        "  # token: or\n"
        "datastore: /x\n"
        "# representative_points:\n"
        "# release:\n"
    )
    # Cyclopts wraps even successful returns in SystemExit(0).
    with pytest.raises(SystemExit) as exc:
        app(["upgrade-config", "--project-dir", str(tmp_path)])
    assert exc.value.code in (None, 0)
    out = capsys.readouterr().out
    assert "in sync" in out.lower()


def test_cli_exits_two_when_project_missing(tmp_path, capsys):
    from nhf_spatial_targets.cli import app

    with pytest.raises(SystemExit) as exc:
        app(["upgrade-config", "--project-dir", str(tmp_path / "no-such")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


# --- whole-target + new-source hints (issue #279, PR-5) ----------------------
#
# upgrade-config is extended beyond the enumerated optional params to surface
# (a) whole targets present in the defaults schema but absent from the
# operator's config.yml, and (b) catalog sources eligible for a target (per
# variables.yml) but not pinned in the operator's targets.<t>.sources[]. Both
# are report-only hints: they print but never mutate config.yml and never
# change the exit-code contract (exit driven solely by optional-feature drift).


def _config_with_targets(project_dir: Path) -> Path:
    """Write a config with five of six targets; aet pins a partial sources[].

    Omits ``snow_water_equivalent`` entirely (a missing-target case) and pins
    ``aet.sources`` to a single key (an available-sources case: ssebop +
    mwbm_climgrid are eligible but not listed).
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "fabric:\n"
        "  path: /x.gpkg\n"
        "  id_col: nhm_id\n"
        "datastore: /x\n"
        "targets:\n"
        "  runoff:\n"
        '    period: "2000/2010"\n'
        "  aet:\n"
        "    sources:\n"
        "      - mod16a2_v061\n"
        '    period: "2000/2010"\n'
        "  recharge:\n"
        '    period: "2000/2010"\n'
        "  soil_moisture:\n"
        '    period: "2000/2010"\n'
        "  snow_covered_area:\n"
        '    period: "2000/2010"\n'
    )
    p = project_dir / "config.yml"
    p.write_text(text)
    return p


def test_check_missing_targets_reports_absent_target(tmp_path):
    from nhf_spatial_targets.upgrade_config import check_missing_targets

    _config_with_targets(tmp_path)
    missing = check_missing_targets(tmp_path)
    # snow_water_equivalent is in the defaults schema but absent from config.
    assert "snow_water_equivalent" in missing
    # Targets that ARE present are not reported.
    assert "aet" not in missing
    assert "runoff" not in missing


def test_check_missing_targets_reports_all_when_no_targets_block(tmp_path):
    from nhf_spatial_targets.upgrade_config import check_missing_targets
    from nhf_spatial_targets.defaults import DEFAULTS

    _write_minimal_config(tmp_path)  # no targets: mapping
    missing = check_missing_targets(tmp_path)
    assert set(missing) == set(DEFAULTS["targets"])


def test_check_available_sources_lists_omitted_eligible(tmp_path):
    from nhf_spatial_targets.upgrade_config import check_available_sources

    _config_with_targets(tmp_path)
    available = check_available_sources(tmp_path)
    # aet eligible per variables.yml = [mod16a2_v061, ssebop, mwbm_climgrid];
    # config pins only mod16a2_v061, so BOTH remaining eligible sources are
    # reported, in the eligible-list (defaults-schema) order. Exact-equality
    # locks the documented order-preserving contract and catches a partial
    # omission (e.g. dropping mwbm_climgrid).
    assert available["aet"] == ["ssebop", "mwbm_climgrid"]
    # The pinned key is not reported as "available" (already in use).
    assert "mod16a2_v061" not in available["aet"]


def test_check_available_sources_skips_unpinned_targets(tmp_path):
    """Targets without an explicit sources[] track defaults and so cannot
    fossilize — they are not reported (avoids noise)."""
    from nhf_spatial_targets.upgrade_config import check_available_sources

    _config_with_targets(tmp_path)
    available = check_available_sources(tmp_path)
    # runoff has a period but no explicit sources[] -> on defaults -> skipped.
    assert "runoff" not in available


def test_target_to_variable_mapping_is_consistent():
    """The explicit config-target -> variables.yml-key map must cover exactly
    the defaults targets and point at real variable definitions (guards the
    reconciliation against future renames)."""
    from nhf_spatial_targets.catalog import variables
    from nhf_spatial_targets.defaults import DEFAULTS
    from nhf_spatial_targets.upgrade_config import _TARGET_TO_VARIABLE

    assert set(_TARGET_TO_VARIABLE) == set(DEFAULTS["targets"])
    all_vars = variables()
    for var_key in _TARGET_TO_VARIABLE.values():
        assert var_key in all_vars


def test_cli_never_mutates_config(tmp_path):
    """upgrade-config (with the new hints) is report-only: config.yml is
    byte-identical before and after the command runs."""
    from nhf_spatial_targets.cli import app

    cfg = _config_with_targets(tmp_path)
    before = cfg.read_bytes()
    with pytest.raises(SystemExit):
        app(["upgrade-config", "--project-dir", str(tmp_path)])
    assert cfg.read_bytes() == before


def test_cli_exits_one_when_config_missing_in_existing_dir(tmp_path, capsys):
    """An existing project dir with no config.yml routes through the new check
    calls and must exit 1 (FileNotFoundError), not 2 (which is dir-not-found)."""
    from nhf_spatial_targets.cli import app

    tmp_path.mkdir(parents=True, exist_ok=True)  # dir exists; config.yml absent
    with pytest.raises(SystemExit) as exc:
        app(["upgrade-config", "--project-dir", str(tmp_path)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "upgrade-config failed" in err


def test_cli_exits_one_on_malformed_config(tmp_path, capsys):
    """A syntactically malformed config.yml fails with an actionable message,
    not a raw YAMLError traceback (the new checks parse YAML; check_drift did
    not, so this is a guarded regression)."""
    from nhf_spatial_targets.cli import app

    tmp_path.mkdir(parents=True, exist_ok=True)
    # Unclosed bracket -> yaml.ParserError (a yaml.YAMLError subclass).
    (tmp_path / "config.yml").write_text("targets: [unclosed\n")
    with pytest.raises(SystemExit) as exc:
        app(["upgrade-config", "--project-dir", str(tmp_path)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "parse config.yml" in err
