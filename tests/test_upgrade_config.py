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
    # Both shipped features (PR #193) should be reported on a bare config.
    assert {"fabric.token", "representative_points"} <= names


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

    # Operator pasted both commented stubs from the latest template.
    (tmp_path / "config.yml").write_text(
        "fabric:\n"
        "  path: /x.gpkg\n"
        "  id_col: nhm_id\n"
        "  # token: or\n"
        "datastore: /x\n"
        "# representative_points:\n"
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
