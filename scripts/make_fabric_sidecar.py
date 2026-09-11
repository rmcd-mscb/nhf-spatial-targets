#!/usr/bin/env python3
"""Emit the sidecar JSON that accompanies a published fabric GeoPackage.

Run as the last step of the fabric build, so the sidecar can never drift
from the artifact it describes:

    python make_fabric_sidecar.py or_v9.gpkg --fabric or --version 9 \
        --repo https://code.usgs.gov/ORG/REPO --tag or-v9

Writes <stem>.json next to the .gpkg. Requires pyogrio (ships with
geopandas).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyogrio

# Layers whose primary key is known; anything else is reported without one
# so the author is prompted to fill it in rather than leaving it implicit.
DEFAULT_KEYS = {
    "nhru": "hru_id",
    "nsegment": "segment_id",
    "npoigages": "poi_gage_id",
    "npoi": "vpu_poi_id",
}
# Columns that identify a feature in SOMEONE ELSE'S numbering. Flagged so a
# consumer never mistakes one for this fabric's key.
CROSS_REF_PREFIXES = ("nhm_",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpkg", type=Path)
    ap.add_argument("--fabric", required=True, help="short fabric name, e.g. 'or'")
    ap.add_argument("--version", required=True, type=int)
    ap.add_argument("--title", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--repo-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    layers: dict[str, dict] = {}
    crs_seen: set[str] = set()
    for name, geom in pyogrio.list_layers(args.gpkg):
        info = pyogrio.read_info(args.gpkg, layer=name)
        fields = list(info["fields"])
        crs = info.get("crs")
        if crs:
            crs_seen.add(str(crs))
        entry: dict = {
            "geometry": str(geom),
            "features": int(info["features"]),
            "primary_key": DEFAULT_KEYS.get(name),
            "fields": fields,
        }
        xrefs = {
            f: "identifier in an external numbering - not this fabric's key"
            for f in fields
            if f.startswith(CROSS_REF_PREFIXES)
        }
        if xrefs:
            entry["cross_reference_ids"] = xrefs
        layers[name] = entry

    doc = {
        "fabric": args.fabric,
        "version": args.version,
        "title": args.title or f"{args.fabric} model layers",
        "crs": sorted(crs_seen)[0] if len(crs_seen) == 1 else sorted(crs_seen),
        "gpkg": {
            "filename": args.gpkg.name,
            "bytes": args.gpkg.stat().st_size,
            "sha256": sha256(args.gpkg),
        },
        "produced_by": {
            "repo": args.repo,
            "tag": args.tag,
            "commit": git_commit(args.repo_dir),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "layers": layers,
    }
    out = args.gpkg.with_suffix(".json")
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    missing = [k for k, v in layers.items() if v["primary_key"] is None]
    if missing:
        print(f"NOTE: set primary_key by hand for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
