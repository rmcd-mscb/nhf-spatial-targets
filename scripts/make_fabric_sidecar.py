#!/usr/bin/env python3
"""Emit the sidecar JSON that accompanies a published fabric GeoPackage.

Run as the last step of the fabric build, from the fabric repo's working
tree, so the sidecar can never drift from the artifact it describes:

    python make_fabric_sidecar.py or_v9.gpkg --fabric or --version 9 \
        --title "Oregon NHM model layers" \
        --repo https://code.usgs.gov/ORG/REPO --tag or-v9 \
        --key domain=domain_id

Writes <stem>.json next to the .gpkg. Requires pyogrio (ships with
geopandas).

Refuses to guess: a declared primary key must exist in its layer and be
unique and non-null, or the script exits non-zero without writing a sidecar.
A layer whose key it does not know is written with ``"primary_key": null``
and named in a NOTE on stderr; declare it with ``--key LAYER=COLUMN`` rather
than hand-editing the JSON, which the next build would overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyogrio

# Layers whose primary key is known; anything else is reported without one
# so the author is prompted to fill it in rather than leaving it implicit.
# npoi is deliberately absent: in the Oregon v9 fabric its vpu_poi_id has 305
# duplicates, and the only unique column (nhm_seg_id) is externally owned.
DEFAULT_KEYS = {
    "nhru": "hru_id",
    "nsegment": "segment_id",
    "npoigages": "poi_gage_id",
}
# Columns that identify a feature in SOMEONE ELSE'S numbering. Flagged so a
# consumer never mistakes one for this fabric's key. Other external ids are
# added with --xref.
CROSS_REF_PREFIXES = ("nhm_", "to_nhm_")
XREF_NOTE = "identifier in an external numbering - not this fabric's key"


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_dir: Path) -> str | None:
    """HEAD of ``repo_dir``, or None (with a warning) if it cannot be read."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        warn(
            f"WARNING: no git commit recorded for --repo-dir {repo_dir}: "
            f"{detail.strip()}"
        )
        return None
    if dirty:
        warn(
            f"WARNING: {repo_dir} has uncommitted changes; commit {head[:12]} "
            "may not describe the bytes that were built"
        )
    return head


def check_key(gpkg: Path, layer: str, key: str, fields: list[str]) -> str | None:
    """Return an error message if ``key`` does not identify rows of ``layer``."""
    if key not in fields:
        return f"layer {layer!r}: primary key {key!r} is not a column ({fields})"
    col = pyogrio.read_dataframe(gpkg, layer=layer, columns=[key], read_geometry=False)[
        key
    ]
    if col.isna().any():
        return f"layer {layer!r}: primary key {key!r} has {col.isna().sum()} nulls"
    if not col.is_unique:
        n = int(col.duplicated().sum())
        return f"layer {layer!r}: primary key {key!r} has {n} duplicate values"
    return None


def parse_keys(pairs: list[str]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for pair in pairs:
        layer, sep, col = pair.partition("=")
        if not sep or not layer or not col:
            sys.exit(f"error: --key expects LAYER=COLUMN, got {pair!r}")
        keys[layer] = col
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("gpkg", type=Path)
    ap.add_argument("--fabric", required=True, help="short fabric name, e.g. 'or'")
    ap.add_argument("--version", required=True, type=int, help="integer version")
    ap.add_argument("--title", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="working tree of the repo that built the fabric (default: cwd)",
    )
    ap.add_argument(
        "--key",
        action="append",
        default=[],
        metavar="LAYER=COLUMN",
        help="declare or override a layer's primary key (repeatable)",
    )
    ap.add_argument(
        "--xref",
        action="append",
        default=[],
        metavar="COLUMN",
        help="flag another externally-owned id column (repeatable)",
    )
    args = ap.parse_args()

    gpkg: Path = args.gpkg
    if gpkg.suffix.lower() != ".gpkg":
        sys.exit(f"error: expected a .gpkg file, got {gpkg.name!r}")
    if not gpkg.is_file():
        sys.exit(f"error: {gpkg} does not exist")
    expected_stem = f"{args.fabric}_v{args.version}"
    if gpkg.stem != expected_stem:
        warn(
            f"WARNING: file is named {gpkg.name!r}; the guideline expects "
            f"{expected_stem}.gpkg for --fabric {args.fabric} --version "
            f"{args.version}"
        )

    user_keys = parse_keys(args.key)
    keys = {**DEFAULT_KEYS, **user_keys}
    xref_cols = set(args.xref)

    listing = pyogrio.list_layers(gpkg)
    unknown = sorted(set(user_keys) - {name for name, _ in listing})
    if unknown:
        sys.exit(f"error: --key names layers not in the GeoPackage: {unknown}")

    layers: dict[str, dict] = {}
    crs_seen: set[str] = set()
    errors: list[str] = []
    for name, geom in listing:
        info = pyogrio.read_info(gpkg, layer=name, force_feature_count=True)
        fields = list(info["fields"])
        crs = str(info["crs"]) if info.get("crs") else None
        if crs:
            crs_seen.add(crs)
        key = keys.get(name)
        if key is not None:
            err = check_key(gpkg, name, key, fields)
            if err:
                errors.append(err)
        entry: dict = {
            "geometry": str(geom) if geom else None,
            "crs": crs,
            "features": int(info["features"]),
            "primary_key": key,
            "fields": fields,
        }
        xrefs = {
            f: XREF_NOTE
            for f in fields
            if f.startswith(CROSS_REF_PREFIXES) or f in xref_cols
        }
        if xrefs:
            entry["cross_reference_ids"] = xrefs
        layers[name] = entry

    if errors:
        for err in errors:
            warn(f"ERROR: {err}")
        warn("no sidecar written")
        return 1
    if len(crs_seen) > 1:
        warn(f"WARNING: layers use different CRSs: {sorted(crs_seen)}")

    doc = {
        "fabric": args.fabric,
        "version": args.version,
        "title": args.title or f"{args.fabric} model layers",
        "crs": sorted(crs_seen)[0] if len(crs_seen) == 1 else sorted(crs_seen),
        "gpkg": {
            "filename": gpkg.name,
            "bytes": gpkg.stat().st_size,
            "sha256": sha256(gpkg),
        },
        "produced_by": {
            "repo": args.repo,
            "tag": args.tag,
            "commit": git_commit(args.repo_dir),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "layers": layers,
    }
    out = gpkg.with_suffix(".json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out}")
    missing = [k for k, v in layers.items() if v["primary_key"] is None]
    if missing:
        warn(
            f"NOTE: no primary_key for: {', '.join(missing)} -- "
            "declare with --key LAYER=COLUMN"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
