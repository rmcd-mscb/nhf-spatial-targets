"""Idempotent publish orchestration for the ScienceBase data release.

This module composes the offline build (:mod:`~nhf_spatial_targets.release.build`)
with the registry (:mod:`~nhf_spatial_targets.release.registry`, publish *intent*)
and the ScienceBase client (:mod:`~nhf_spatial_targets.release.sb_client`, what
actually *exists*) into the create-vs-update reconciliation that publishes one
umbrella / source-child / fabric-child item and records the result.

**Dependency injection.** Every ``publish_*`` takes an already-constructed
:class:`~nhf_spatial_targets.release.sb_client.SbClient`. Resolving credentials
and building a real session is handled by the (not-yet-implemented) credential
wiring; here the client is assumed ready, so
the whole module is offline-testable against a mock client (the mock patterns
in ``tests/test_release_sb_client_offline.py``).

Per-item sequence (mirrors the plan's idempotency matrix)::

    1. pre-flight (cheap, fail-fast)  manifest present+parses, release_defaults
                                      populated, config authors non-empty, and
                                      -- for a child -- the umbrella sb_id is known
    2. build (idempotent)             build.build_*()  -> BuildResult
    3. verify staged checksums        checksums.verify_csv() (ChecksumMismatch)
    4. reconcile intent vs reality    registry sb_id? -> get_item (404 => stale,
                                      fatal unless create_new); else find_child by
                                      title -> one adopts / zero creates / many fail
    5. create or update               create_item+upload all, or patch changed
                                      body fields + per-file skip/replace + orphans
    6. record                         registry.put_* (flock read-merge-write)

**The MD5 decision (per the plan).** ScienceBase records uploaded-file
checksums as **MD5**, not SHA-256 (sciencebasepy's
``SbSession._replace_file`` stamps ``checksum = {'type': 'MD5', ...}``), and
:meth:`SbClient.upload_file`'s ``skip_if_sha256_matches`` does plain
value-equality against the remote file's recorded ``checksum.value``. The
skip-if-unchanged fast path therefore computes the **local file's MD5** and
passes *that* (option (a) in the plan): a genuine content match -- local MD5
== remote MD5 -- skips the byte transfer, and any mismatch re-uploads. The
comparison is honest in both directions: an MD5 hex string (32 chars) can
never equal a SHA-256 (64 chars), so if a remote file ever records a SHA-256
the local MD5 simply won't match and the file re-uploads -- correct, just not
skipped. We pass MD5 through the (SHA-256-named) parameter rather than
extending the ``SbClient`` upload primitive: the "what checksum does ScienceBase
actually use" knowledge belongs in this orchestration layer.

**Flat ScienceBase namespace.** A staged payload is a nested tree
(``aggregated/<source>/<nc>``, ``targets/<nc>``), but a ScienceBase item holds
a flat ``files`` list keyed by basename. :func:`_payload_files` raises if two
staged files collapse to the same basename, rather than letting one silently
overwrite the other on upload.

**``manifest_sha256`` in the registry** is the SHA-256 of the staged
``SHA256SUMS`` file -- a single fingerprint of the *entire* payload that exists
for every child (a source child stages no ``manifest.json``). Its content is
the sorted ``<sha256>  <path>`` list, so the fingerprint changes iff any file's
content or the file set changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from nhf_spatial_targets import catalog
from nhf_spatial_targets.release import registry
from nhf_spatial_targets.release._models import ReleaseError
from nhf_spatial_targets.release.build import (
    BuildResult,
    build_fabric_child,
    build_source_child,
    build_umbrella,
)
from nhf_spatial_targets.release.checksums import ChecksumMismatch, verify_csv
from nhf_spatial_targets.release.defaults import (
    load_release_defaults,
    require_populated_release_defaults,
)
from nhf_spatial_targets.release.lineage import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    read_manifest,
    sha256_file,
)
from nhf_spatial_targets.release.payload import resolve_fabric_label, sources_used
from nhf_spatial_targets.release.sb_client import SbClient, SbClientError
from nhf_spatial_targets.workspace import Project

logger = logging.getLogger(__name__)

Scope = Literal["umbrella", "source", "fabric"]
PublishMode = Literal["create", "update"]

# Item-body fields this layer manages on a ScienceBase item. The rich FGDC /
# ISO metadata travels as uploaded *files* (fgdc.xml / iso.xml); the item JSON
# itself carries only the landing-page title + descriptions. `title` MUST equal
# the MCF title so a re-publish can re-adopt the item via find_child.
_BODY_FIELDS: tuple[str, ...] = ("title", "summary", "body")

_MD5_CHUNK_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PreflightError(ReleaseError):
    """A fatal pre-flight gate failed; nothing was published."""


class StaleRegistryError(ReleaseError):
    """The registry names a ScienceBase id that the server reports as gone.

    The recorded ``sb_id`` resolves to not-found on ScienceBase -- the item was
    deleted, or the id is stale. Publishing escapes this with ``create_new`` to
    mint a fresh item (and overwrite the registry id); otherwise it is fatal so
    a typo or a deleted item can't silently fork a second copy.
    """


# ---------------------------------------------------------------------------
# Typed result records (orchestration boundary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishResult:
    """Outcome of publishing one item.

    ``key`` is the source key / fabric label, ``None`` for the umbrella.
    ``mode`` is ``"create"`` or ``"update"``; ``adopted`` is True when an
    existing same-title item was adopted (a registry entry without an sb_id
    matched a child by title). ``uploaded`` / ``skipped`` are the basenames of
    files sent vs skipped (remote checksum already matched). ``orphans`` are
    remote files no longer in the staged payload; ``orphans_deleted`` is the
    subset actually removed (only with ``delete_orphans``). ``body_patched``
    lists the item-body fields changed on update (empty on create).
    ``registry_entry`` is a shallow snapshot of the merged record written back
    to the registry (a plain dict because its shape varies by scope -- umbrella
    carries version/doi, children carry file_count/total_bytes); treat it as
    read-only even though this dataclass is frozen.
    """

    scope: Scope
    key: str | None
    sb_id: str
    mode: PublishMode
    adopted: bool
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]
    orphans: tuple[str, ...]
    orphans_deleted: tuple[str, ...]
    body_patched: tuple[str, ...]
    registry_entry: dict

    def __post_init__(self) -> None:
        # The umbrella is keyless; source/fabric children always carry a key.
        # Same invariant BuildResult guards -- enforced here so a future second
        # producer (the CLI / status command) can't construct a mismatch.
        if (self.scope == "umbrella") != (self.key is None):
            raise ValueError(
                f"{self.scope!r} key invariant violated: umbrella must have "
                f"key=None and source/fabric a non-None key; got key={self.key!r}."
            )


LocalStatus = Literal["built", "stale", "missing"]
RemoteStatus = Literal["present", "drift", "missing"]


@dataclass(frozen=True)
class ScopeDiff:
    """Read-only intent-vs-reality diff for one item (status / dry-run).

    ``local`` reflects the staged payload on disk: ``built`` (checksums
    verify), ``stale`` (staged but checksums drifted / incomplete), or
    ``missing`` (not built). ``remote`` reflects ScienceBase: ``present``
    (item exists and its files match the staged payload), ``drift`` (item
    exists but files differ), or ``missing`` (not in the registry, or the
    registry id is stale). ``detail`` carries a human-readable note.
    """

    scope: Scope
    key: str | None
    sb_id: str | None
    local: LocalStatus
    remote: RemoteStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        if (self.scope == "umbrella") != (self.key is None):
            raise ValueError(
                f"{self.scope!r} key invariant violated: umbrella must have "
                f"key=None and source/fabric a non-None key; got key={self.key!r}."
            )


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _now_iso(now: datetime | None) -> str:
    return (now if now is not None else datetime.now(timezone.utc)).isoformat()


def _file_md5(path: Path) -> str:
    """Streamed MD5 hex digest of *path* (ScienceBase's native checksum)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_MD5_CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


# sciencebasepy wraps a missing item in a bare ``Exception`` whose text is
# either ``"Resource not found"`` or the catch-all ``"Other HTTP error: 404:
# ..."`` (mirrors ``sb_client._OTHER_HTTP_ERROR_RE``). Matching only these
# recognized shapes -- never a bare "not found" substring or a "404" appearing
# anywhere in a 5xx body -- keeps a non-404 error from being mistaken for a
# missing item, which under ``create_new`` would wrongly mint a duplicate.
_RESOURCE_NOT_FOUND_MARKER = "resource not found"
_HTTP_ERROR_CODE_RE = re.compile(r"Other HTTP error:\s*(\d{3})")


def _is_not_found(exc: BaseException) -> bool:
    """Whether *exc* is ScienceBase reporting that an id does not exist.

    Recognizes a ``requests``-level ``response.status_code == 404``, the
    literal ``"Resource not found"`` marker, and the
    ``"Other HTTP error: 404"`` catch-all form -- and deliberately nothing
    looser, so a 5xx whose body merely contains the words "not found" or "404"
    is not misread as a missing item.
    """
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    message = str(exc)
    if _RESOURCE_NOT_FOUND_MARKER in message.lower():
        return True
    match = _HTTP_ERROR_CODE_RE.search(message)
    return bool(match and match.group(1) == "404")


@dataclass(frozen=True)
class _PayloadFile:
    """One file to upload: absolute path, ScienceBase basename, size."""

    abs_path: Path
    name: str
    rel: str
    size: int


def _payload_files(result: BuildResult) -> list[_PayloadFile]:
    """The full upload set for *result*, sorted, with a flat-name collision guard.

    ``result.entries`` already fingerprints every staged file except the two
    integrity manifests (``checksums.csv`` / ``SHA256SUMS``), which are added
    here so the published item carries them too. Two staged files that share a
    basename would collide in ScienceBase's flat namespace, so that is a loud
    failure rather than a silent overwrite.
    """
    stage = result.stage_dir
    rels = [e.path for e in result.entries]
    for extra in ("checksums.csv", "SHA256SUMS"):
        if (stage / extra).exists():
            rels.append(extra)

    files: list[_PayloadFile] = []
    by_name: dict[str, str] = {}
    for rel in sorted(set(rels)):
        path = stage / rel
        name = path.name
        if name in by_name:
            raise ReleaseError(
                f"two staged files collapse to the ScienceBase file name "
                f"{name!r}: {by_name[name]!r} and {rel!r}. ScienceBase stores "
                f"files in a flat namespace; rename one before publishing."
            )
        by_name[name] = rel
        files.append(
            _PayloadFile(abs_path=path, name=name, rel=rel, size=path.stat().st_size)
        )
    return files


def _remote_files(item: dict | None) -> dict[str, int | None]:
    """Map ScienceBase file ``name`` -> ``size`` across ``files`` + facets."""
    out: dict[str, int | None] = {}
    if not item:
        return out
    for entry in item.get("files") or []:
        out[entry["name"]] = entry.get("size")
    for facet in item.get("facets") or []:
        for entry in facet.get("files") or []:
            out[entry["name"]] = entry.get("size")
    return out


def _item_body(mcf: dict) -> dict:
    """Derive the managed ScienceBase item body from an MCF dict.

    ``title`` must equal the MCF title (find_child adopts by exact title);
    ``body`` is the abstract; ``summary`` is the concise purpose (abstract as a
    fallback). The full FGDC/ISO record rides along as uploaded files, not item
    JSON fields.
    """
    ident = mcf.get("identification", {})
    abstract = ident.get("abstract", "")
    return {
        "title": ident.get("title", ""),
        "body": abstract,
        "summary": ident.get("purpose") or abstract,
    }


# ---------------------------------------------------------------------------
# Pre-flight gates
# ---------------------------------------------------------------------------


def _drift_fingerprint(manifest: dict) -> str:
    """JSON-normalized (sorted-key) fingerprint of a manifest's derived catalog.

    Publish compares only the *derived* catalog (``sources`` + ``steps``)
    between the on-disk manifest and the deterministic projection -- identity
    fields (``created_utc``, ``fabric`` authorship) are read-merged by the
    rebuild and are not drift. ``sort_keys`` makes the comparison insensitive
    to dict-key ordering; the projection already emits steps in a deterministic
    order, so a byte-equal fingerprint means no drift.
    """
    return json.dumps(
        {
            "sources": manifest.get("sources") or {},
            "steps": manifest.get("steps") or [],
        },
        sort_keys=True,
    )


def _preflight_provenance_complete(
    project: Project, *, allow_incomplete_sources: bool = False
) -> None:
    """Verify (don't mutate) that the on-disk manifest is complete + current.

    Runs ``rebuild_manifest`` in ``dry_run`` to obtain the deterministic
    projection of the on-disk artifacts, reads the on-disk manifest, and
    refuses to publish unless they agree. Publish stays a pure read-local /
    write-remote operation: this gate never writes the manifest -- regenerating
    it is the operator's explicit ``nhf-targets maintenance rebuild-manifest`` action, which
    is what keeps publish idempotent.

    Always-fatal (a manifest this broken cannot be published at all):

    - ``manifest_schema_version`` not equal to
      :data:`CURRENT_MANIFEST_SCHEMA_VERSION` (behind, or an unexpected future
      version -- either way the shape can't be trusted),
    - no ``fabric`` block,
    - empty ``steps[]`` (no lineage).

    Source-completeness + drift (the deliberate
    ``allow_incomplete_sources`` override downgrades these to a logged warning):

    - ``sources[]`` missing an aggregated dir or a consolidated datastore
      catalog source present on disk (the #277 ∪ #278 union),
    - a published target NC with no matching ``target`` step,
    - on-disk ``sources``/``steps`` differing from the projection (drift).
    """
    from nhf_spatial_targets.rebuild_manifest import rebuild_manifest

    # A corrupt/truncated published target NC makes the projection raise
    # (issue #283). Surface it as a fatal PreflightError -- it is a genuine
    # read failure, NOT a source/drift problem, so it must never be downgraded
    # by allow_incomplete_sources below.
    try:
        projection = rebuild_manifest(project, dry_run=True)
    except ValueError as exc:
        raise PreflightError(f"publish pre-flight: {exc}") from exc
    on_disk = read_manifest(project.manifest_path)

    rebuild_hint = (
        f"run 'nhf-targets maintenance rebuild-manifest -d {project.workdir}' to regenerate "
        f"manifest.json as a complete projection of the on-disk artifacts, then "
        f"re-publish"
    )

    # --- always-fatal structural checks (not bypassable) ---
    schema = on_disk.get("manifest_schema_version", 0)
    if schema != CURRENT_MANIFEST_SCHEMA_VERSION:
        raise PreflightError(
            f"manifest.json at {project.manifest_path} is schema version "
            f"{schema}; current is {CURRENT_MANIFEST_SCHEMA_VERSION}. {rebuild_hint}."
        )
    if on_disk.get("fabric") is None:
        raise PreflightError(
            f"manifest.json at {project.manifest_path} has no fabric block; "
            f"run 'nhf-targets validate -d {project.workdir}' then {rebuild_hint}."
        )
    if not on_disk.get("steps"):
        raise PreflightError(
            f"manifest.json at {project.manifest_path} records no steps[] "
            f"(no lineage); {rebuild_hint}."
        )

    # --- source-completeness + drift (override-downgradable) ---
    problems: list[str] = []

    on_disk_sources = set(on_disk.get("sources") or {})
    catalog_keys = set(catalog.sources())
    agg_root = project.aggregated_dir()
    agg_dirs = (
        {p.name for p in agg_root.iterdir() if p.is_dir()}
        if agg_root.is_dir()
        else set()
    )
    datastore_used = (
        {
            p.name
            for p in project.datastore.iterdir()
            if p.is_dir() and p.name in catalog_keys
        }
        if project.datastore.is_dir()
        else set()
    )
    missing_sources = sorted((agg_dirs | datastore_used) - on_disk_sources)
    if missing_sources:
        problems.append(
            f"sources[] is missing {missing_sources} (present on disk under "
            f"data/aggregated/ or the datastore, absent from the manifest)"
        )

    targets_dir = project.targets_dir()
    target_ncs = (
        sorted(p.name for p in targets_dir.glob("*.nc") if p.is_file())
        if targets_dir.is_dir()
        else []
    )
    target_step_outputs = {
        Path(out.get("path", "")).name
        for step in (on_disk.get("steps") or [])
        if step.get("kind") == "target"
        for out in (step.get("outputs") or [])
    }
    missing_targets = sorted(set(target_ncs) - target_step_outputs)
    if missing_targets:
        problems.append(
            f"published target NC(s) {missing_targets} have no matching 'target' step"
        )

    if _drift_fingerprint(on_disk) != _drift_fingerprint(projection):
        problems.append(
            "sources[]/steps[] differ from the deterministic rebuild-manifest "
            "projection (drift)"
        )

    if problems:
        message = (
            f"publish pre-flight: manifest.json at {project.manifest_path} is "
            f"incomplete or stale:\n  - "
            + "\n  - ".join(problems)
            + f"\nTo fix, {rebuild_hint}."
        )
        if allow_incomplete_sources:
            logger.warning(
                "%s\nProceeding anyway because --allow-incomplete-sources was set.",
                message,
            )
            return
        raise PreflightError(message)


def _preflight_effective_config_current(project: Project) -> None:
    """Verify (don't mutate) that ``config.effective.yml`` is current.

    ``config.effective.yml`` is a DERIVED artifact: a version/hash-stamped
    projection of (``config.yml`` x ``defaults.py`` x ``fabric.json``) written
    by ``validate``. This gate reads its ``_effective_config_meta`` stamp and
    refuses to publish unless the recorded ``source_config_sha256`` still
    matches the sha256 of the current ``config.yml`` bytes -- i.e. ``config.yml``
    has not been edited since the effective config was generated.

    Verify-don't-mutate: like :func:`_preflight_provenance_complete`, this gate
    NEVER regenerates ``config.effective.yml``. Regenerating it is the
    operator's explicit ``nhf-targets validate`` action, which is what keeps
    publish idempotent.

    Unconditionally fatal: unlike :func:`_preflight_provenance_complete`, this
    gate has NO ``allow_incomplete_sources`` override. A stale effective config
    means the published provenance would describe a configuration that no
    longer matches the operator's intent (the spec's Pillar 5 fossil) -- there
    is no defensible reason to ship it.

    Stale (any of) -> :class:`PreflightError`:

    - ``config.effective.yml`` absent or unparseable,
    - no ``_effective_config_meta`` block (predates the PR-4 stamp),
    - ``effective_config_schema_version`` missing/malformed, or not equal to
      :data:`~nhf_spatial_targets.validate.EFFECTIVE_CONFIG_SCHEMA_VERSION`
      (behind OR an unexpected future version -- parity with the manifest gate),
    - recorded ``source_config_sha256`` != sha256(current ``config.yml``).
    """
    from nhf_spatial_targets.validate import EFFECTIVE_CONFIG_SCHEMA_VERSION

    eff_path = project.workdir / "config.effective.yml"
    config_path = project.workdir / "config.yml"
    hint = (
        f"config.effective.yml is stale; re-run "
        f"'nhf-targets validate -d {project.workdir}' to regenerate it"
    )

    if not eff_path.exists():
        raise PreflightError(f"{hint} (not found at {eff_path}).")

    try:
        effective = yaml.safe_load(eff_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise PreflightError(
            f"{hint} (config.effective.yml at {eff_path} is unparseable: {exc})."
        ) from exc

    meta = effective.get("_effective_config_meta")
    if not isinstance(meta, dict):
        raise PreflightError(
            f"{hint} (no _effective_config_meta block; it predates the "
            f"version/hash stamp)."
        )

    schema = meta.get("effective_config_schema_version")
    # A malformed stamp (absent / null / non-int / bool) is itself a
    # "regenerate me" signal: turn it into a clean PreflightError rather than
    # letting the comparison leak a raw TypeError (None/str/list) or a
    # float/bool slip through silently. (bool is an int subclass, so exclude it
    # explicitly.)
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise PreflightError(
            f"{hint} (effective_config_schema_version {schema!r} is missing or "
            f"not a valid integer version)."
        )
    # Mirror the manifest gate (_preflight_provenance_complete): ANY mismatch is
    # fatal, not just "behind". A future/unexpected version is an effective
    # config this code cannot validate -- refuse rather than ship provenance we
    # did not fully check.
    if schema != EFFECTIVE_CONFIG_SCHEMA_VERSION:
        raise PreflightError(
            f"{hint} (effective_config_schema_version {schema} != the current "
            f"{EFFECTIVE_CONFIG_SCHEMA_VERSION}; behind or unexpected -- the "
            f"stamped shape cannot be trusted)."
        )

    recorded = meta.get("source_config_sha256")
    current = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if recorded != current:
        raise PreflightError(
            f"{hint} (config.yml has changed since it was generated: recorded "
            f"sha256 {recorded} != current {current})."
        )


# The resolved-param attrs persisted on a target NC (PR-7 Task 7.1) and read
# back into the manifest target step (Task 7.2), mapped to the effective-config
# key they must agree with. ``source_keys`` (NC, comma-joined string) <->
# ``sources`` (config, list) is bridged specially below; the rest are compared
# as-is (the NC ``period`` / ``range_method`` / ``normalize_period`` /
# ``ci_threshold`` attrs are stamped verbatim from the same config values, so a
# string/scalar equality is exact for an in-sync project).
_TRIANGLE_DIRECT_PARAMS: tuple[str, ...] = (
    "period",
    "range_method",
    "normalize_period",
    "ci_threshold",
)


def _config_product_problems(
    tgt_name: str, nc_name: str, tgt_cfg: dict, nc_params: dict
) -> list[str]:
    """Return the config<->product disagreements for one target NC.

    Compares the resolved params in ``nc_params`` (read back from the NC's
    global attrs) against the effective-config ``tgt_cfg``. ``normalize_period``
    is compared against the builder's ``normalize_period or period`` fallback so
    a target that leaves it unset (stamped with the period value) is not falsely
    flagged. Other ``_TRIANGLE_DIRECT_PARAMS`` are exact-equality (the NC attrs
    are stamped verbatim from the same config values). ``sources`` (config list)
    bridges to ``source_keys`` (NC comma-joined string).
    """
    problems: list[str] = []
    for key in _TRIANGLE_DIRECT_PARAMS:
        if key not in nc_params:
            continue
        if key == "normalize_period":
            # Mirror rch/som: an unset normalize_period resolves to period.
            expected = tgt_cfg.get("normalize_period") or tgt_cfg.get("period")
        else:
            if key not in tgt_cfg:
                continue
            expected = tgt_cfg[key]
        if expected is None:
            continue
        if nc_params[key] != expected:
            problems.append(
                f"target '{tgt_name}' ({nc_name}): config.effective.yml "
                f"{key}={expected!r} but the published NC says {nc_params[key]!r}"
            )

    # sources (config list) <-> source_keys (NC comma-joined string).
    cfg_sources = tgt_cfg.get("sources")
    nc_source_keys = nc_params.get("source_keys")
    if isinstance(cfg_sources, list) and nc_source_keys is not None:
        nc_sources = [s for s in str(nc_source_keys).split(",") if s]
        if nc_sources != cfg_sources:
            problems.append(
                f"target '{tgt_name}' ({nc_name}): config.effective.yml "
                f"sources={cfg_sources!r} but the published NC source_keys say "
                f"{nc_sources!r}"
            )
    return problems


def _preflight_config_product_consistency(project: Project) -> None:
    """Verify config.effective.yml agrees with the published target products.

    The config<->product<->manifest triangle (spec Pillar 6, PR-7). Each target
    NC records the resolved params it was built with as global attrs (Task 7.1);
    the deterministic projection reads them back into the ``target`` step (Task
    7.2). ``config.effective.yml`` records the resolved *intent*. When the two
    disagree -- the canonical fossil, effective config says ``period:
    2000-2010`` while the product/manifest say ``1979-2024`` -- publishing would
    ship provenance that contradicts the data.

    Verify-don't-mutate, and **unconditionally fatal**: like
    :func:`_preflight_effective_config_current`, there is no
    ``allow_incomplete_sources`` override. A genuine config/product
    inconsistency is a correctness error, not an incompleteness, so there is no
    defensible reason to ship it. The fix is to re-run the build and/or
    ``validate`` so config, product, and manifest realign.

    Targets are matched to their NC(s) by the ``output_file`` **stem prefix**,
    not the exact basename: SOM writes ``soil_moisture_targets_monthly.nc`` /
    ``_annual.nc`` (never the bare ``soil_moisture_targets.nc``), and every
    target also writes a ``<stem>_nn_filled.nc`` companion -- all carry the same
    resolved-param attrs, so every product NC for a target is checked. A target
    NC that carries no resolved-param attrs yet (a placeholder, or a pre-PR-7
    build) contributes no comparison -- this gate tightens as products are
    rebuilt under the instrumented writer, it never fabricates a mismatch.

    The ``normalize_period`` comparison mirrors the builder's resolution: when a
    target leaves ``normalize_period`` unset, rch/som stamp the NC with the
    *period* value (``normalize_period or period``), so the gate compares the
    NC against that same fallback rather than against a bare ``None`` (which
    would falsely block an in-sync SOM build whose default normalize_period is
    ``None``).
    """
    from nhf_spatial_targets.defaults import DEFAULTS
    from nhf_spatial_targets.rebuild_manifest import _target_nc_params

    eff_path = project.workdir / "config.effective.yml"
    try:
        effective = yaml.safe_load(eff_path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        # The staleness gate (_preflight_effective_config_current) runs first and
        # already turns an absent/unparseable effective config into a clean
        # PreflightError; guard here too so call-order changes can't crash.
        raise PreflightError(
            f"config.effective.yml at {eff_path} is unreadable for the "
            f"config/product consistency check: {exc}"
        ) from exc

    default_targets = DEFAULTS.get("targets") or {}
    eff_targets = effective.get("targets") or {}
    problems: list[str] = []
    for tgt_name, tgt_cfg in eff_targets.items():
        if not isinstance(tgt_cfg, dict):
            continue
        # The NC filename comes from the merged effective config when present,
        # else the code-level DEFAULTS mapping -- the output_file -> target
        # binding is a stable fact, not provenance, so resolving it from
        # DEFAULTS is safe even when a thinned effective config omits it.
        output_file = tgt_cfg.get("output_file") or (
            default_targets.get(tgt_name, {}).get("output_file")
        )
        if not output_file:
            continue
        # Match every product NC for this target by stem prefix: the bare
        # <stem>.nc, SOM's <stem>_monthly/_annual.nc variants, and every
        # <stem>_nn_filled.nc companion. A target configured but not built (no
        # matching NC) is a completeness concern owned by
        # _preflight_provenance_complete, not this consistency gate.
        stem = Path(output_file).stem
        for nc in sorted(project.targets_dir().glob(f"{stem}*.nc")):
            if not nc.is_file():
                continue
            nc_params = _target_nc_params(nc)
            if not nc_params:
                # A readable NC with no resolved-param attrs is a pre-PR-7 build
                # (or a placeholder): the gate cannot verify it, so it does NOT
                # fabricate a mismatch -- but unverifiable provenance on a
                # publish is worth a loud, named WARNING so the operator knows to
                # rebuild that product before a first release (issue #279 / I1).
                logger.warning(
                    "publish pre-flight: target '%s' product %s carries no "
                    "resolved-param attrs (a pre-PR-7 build); its config/product "
                    "consistency cannot be verified. Rebuild this target, then "
                    "rebuild-manifest, so its provenance is checkable before "
                    "release.",
                    tgt_name,
                    nc.name,
                )
                continue
            problems.extend(
                _config_product_problems(tgt_name, nc.name, tgt_cfg, nc_params)
            )

    if problems:
        joined = "\n  - ".join(problems)
        raise PreflightError(
            "publish pre-flight: config.effective.yml disagrees with the "
            "published target product(s) -- the config/product/manifest "
            "triangle is inconsistent. Re-run the affected target build (and "
            f"'nhf-targets validate -d {project.workdir}') so config, product, "
            f"and manifest realign:\n  - {joined}"
        )


def _preflight_ua_swe_threshold_current(project: Project) -> None:
    """Verify the ua_swe agg NC's baked depth threshold matches config intent.

    The **aggregate-layer analogue** of :func:`_preflight_effective_config_current`.
    ua_swe's ``snow_covered_fraction`` is a per-pixel binary
    ``snow_depth > depth_threshold_mm`` evaluated PRE-aggregation -- a
    nonlinearity that does not commute with the area-weighted mean -- so the
    threshold is *baked into the agg NC*, not a downstream knob. If an operator
    edits ``targets.snow_covered_area.depth_threshold_mm`` without re-aggregating
    ua_swe, the published fraction (and the manifest aggregate-step / FGDC / ISO
    provenance lifted from its stamp) would describe a threshold the data was NOT
    built with.

    Verify-don't-mutate, and **unconditionally fatal** (no
    ``allow_incomplete_sources`` override), parity with the config-staleness gate:
    shipping a fraction whose stamped threshold contradicts config is a
    correctness error, not an incompleteness. The fix is to re-aggregate ua_swe
    (and re-run ``validate`` / ``rebuild-manifest``) so the bake, config, and
    provenance realign.

    No-op when the project never aggregated ua_swe (not every fabric uses it):
    an absent ``data/aggregated/ua_swe/`` dir or no NCs there -> return.

    This is the **first** instance of an aggregate-affecting config param; it is
    specced narrowly for ua_swe, but the pattern -- a pre-aggregation-baked
    tunable whose stamp must match config at publish -- generalizes to any future
    such source.

    Stale / un-verifiable (any of) -> :class:`PreflightError`:

    - an agg NC carries no ``snow_covered_fraction`` ``depth_threshold_mm`` stamp
      (aggregated before depth-threshold stamping existed) -- provenance that
      cannot be verified,
    - a stamped threshold != the config-resolved
      ``targets.snow_covered_area.depth_threshold_mm`` (read from the
      staleness-gated ``config.effective.yml``; default
      :data:`~nhf_spatial_targets.aggregate.ua_swe.DEFAULT_DEPTH_THRESHOLD_MM`).

    A *corrupt* agg NC (NetCDF magic but unopenable, or a malformed stamp) raises
    from :func:`read_scf_threshold_stamp` and is caught earlier, at the
    ``_preflight_provenance_complete`` projection boundary -- so this gate only
    ever sees a healthy or genuinely-unstamped file.
    """
    import math

    from nhf_spatial_targets.aggregate.ua_swe import DEFAULT_DEPTH_THRESHOLD_MM
    from nhf_spatial_targets.rebuild_manifest import (
        _UA_SWE_KEY,
        read_scf_threshold_stamp,
    )

    ua_swe_dir = project.aggregated_dir() / _UA_SWE_KEY
    if not ua_swe_dir.is_dir():
        return
    ncs = sorted(p for p in ua_swe_dir.rglob("*.nc") if p.is_file())
    if not ncs:
        return

    # Read the resolved threshold from the (already staleness-gated)
    # config.effective.yml, the canonical resolved intent -- consistent with the
    # triangle gate. The default mirrors aggregate/ua_swe._resolve_depth_threshold_mm
    # so an unset key compares equal to a stamped default-built fraction.
    eff_path = project.workdir / "config.effective.yml"
    try:
        effective = yaml.safe_load(eff_path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        # _preflight_effective_config_current runs first and already turns an
        # absent/unparseable effective config into a clean PreflightError; guard
        # here too so a call-order change can't crash this gate.
        raise PreflightError(
            f"config.effective.yml at {eff_path} is unreadable for the ua_swe "
            f"depth-threshold check: {exc}"
        ) from exc
    sca_cfg = (effective.get("targets") or {}).get("snow_covered_area") or {}
    configured = float(sca_cfg.get("depth_threshold_mm", DEFAULT_DEPTH_THRESHOLD_MM))

    fix = (
        f"re-aggregate ua_swe (and re-run 'nhf-targets validate -d "
        f"{project.workdir}' / rebuild-manifest) so the baked fraction, config, "
        f"and published provenance realign"
    )
    problems: list[str] = []
    for nc in ncs:
        stamped = read_scf_threshold_stamp(nc)
        if stamped is None:
            problems.append(
                f"{nc.name} carries no snow_covered_fraction "
                f"depth_threshold_mm stamp (aggregated before depth-threshold "
                f"stamping was added); its threshold provenance cannot be verified"
            )
        elif not math.isclose(stamped, configured, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(
                f"{nc.name} was built with depth_threshold_mm={stamped} but "
                f"config resolves to {configured}"
            )

    if problems:
        joined = "\n  - ".join(problems)
        raise PreflightError(
            "publish pre-flight: the ua_swe snow_covered_fraction was aggregated "
            "with a depth threshold that no longer matches config "
            f"(targets.snow_covered_area.depth_threshold_mm) -- {fix}:\n  - {joined}"
        )


def _preflight_common(
    project: Project, *, allow_incomplete_sources: bool = False
) -> None:
    """Fatal, fail-fast checks shared by every publish scope.

    The early checks are constant-time, but the provenance gate (last) walks
    the datastore / ``data/aggregated/`` / ``targets/`` trees to project the
    manifest, so this is not constant-time overall -- it is still fail-fast
    (runs before any build / upload).

    Credential resolvability is intentionally *not* checked here: the publish
    orchestration assumes a ready, injected client (the credential/session
    wiring is not yet implemented).
    """
    manifest_path = project.manifest_path
    if not manifest_path.exists():
        raise PreflightError(
            f"manifest.json not found at {manifest_path}; run validate / aggregate "
            f"before publishing (the release lineage and source list come from it)."
        )
    try:
        read_manifest(manifest_path)
    except ValueError as exc:
        raise PreflightError(
            f"manifest.json at {manifest_path} is unreadable: {exc}"
        ) from exc

    try:
        require_populated_release_defaults(load_release_defaults())
    except (ValueError, FileNotFoundError) as exc:
        raise PreflightError(str(exc)) from exc

    authors = (project.config.get("release") or {}).get("authors") or []
    if not authors:
        raise PreflightError(
            "config.yml release.authors is empty; add at least one author "
            "(given/family) before publishing."
        )

    # The config.effective.yml staleness gate (verify-don't-mutate,
    # unconditionally fatal -- no override). Cheap (read one small YAML + hash
    # one small file), so it runs before the tree-walking provenance gate. A
    # stale effective config would leak a wrong-window configuration into the
    # published provenance, so there is no defensible reason to override it.
    _preflight_effective_config_current(project)

    # The provenance completeness gate (verify-don't-mutate): every publish
    # scope is gated, so a child can never ship an under-reported or stale
    # manifest. The override is the deliberate, logged escape hatch.
    _preflight_provenance_complete(
        project, allow_incomplete_sources=allow_incomplete_sources
    )

    # The config<->product<->manifest triangle (verify-don't-mutate,
    # unconditionally fatal -- no override). Runs after the provenance gate so
    # a corrupt/incomplete manifest is reported first; a genuine config/product
    # param disagreement is a correctness inconsistency that must never ship,
    # so allow_incomplete_sources does not reach it.
    _preflight_config_product_consistency(project)

    # The aggregate-layer staleness gate (verify-don't-mutate, unconditionally
    # fatal -- no override): ua_swe's snow_covered_fraction bakes a pre-aggregation
    # depth threshold into the agg NC, so an operator who edits the config
    # threshold without re-aggregating would ship a fraction whose stamped
    # provenance contradicts config. No-op for projects that never aggregated
    # ua_swe. Runs after the config-staleness gate so config.effective.yml is
    # known current.
    _preflight_ua_swe_threshold_current(project)


def _require_umbrella_sb_id(registry_path: Path | None) -> str:
    """Return the umbrella sb_id or raise -- a child can't be parented without it.

    The plan's ``--create-umbrella`` affordance (publish the umbrella and a
    child in one invocation) is a CLI-level composition of
    :func:`publish_umbrella` then a child publish; the library keeps the gate
    single-purpose so a child never silently lands under a missing parent.
    """
    umbrella = registry.get_umbrella(registry_path)
    if not umbrella or not umbrella.get("sb_id"):
        raise PreflightError(
            "the umbrella item has no recorded ScienceBase id in the registry; "
            "publish the umbrella first (publish_umbrella) before a child."
        )
    return umbrella["sb_id"]


# ---------------------------------------------------------------------------
# Create-vs-update reconciliation
# ---------------------------------------------------------------------------


def _stale_registry_error(
    scope: Scope, key: str | None, sb_id: str
) -> StaleRegistryError:
    target = scope if key is None else f"{scope}/{key}"
    return StaleRegistryError(
        f"the registry records sb_id={sb_id!r} for {target}, but ScienceBase "
        f"reports it not-found: the item was deleted or the id is stale. "
        f"Re-publish with create_new=True to mint a new item (the registry id "
        f"is then overwritten), or restore the registry from git."
    )


def _resolve_target(
    client: SbClient,
    reg_entry: dict | None,
    parent_id: str,
    title: str,
    *,
    create_new: bool,
    scope: Scope,
    key: str | None,
) -> tuple[PublishMode, str | None, dict | None, bool]:
    """Decide create vs update for one item; return (mode, sb_id, item, adopted).

    Implements the plan's matrix: a registry sb_id is fetched (not-found is
    fatal unless ``create_new``); without one, an exact-title child is sought
    under *parent_id* -- one match adopts (update + new sb_id), zero creates,
    and many is fatal (raised by :meth:`SbClient.find_child`).
    """
    if reg_entry and reg_entry.get("sb_id"):
        sb_id = reg_entry["sb_id"]
        try:
            item = client.get_item(sb_id)
        except Exception as exc:
            if _is_not_found(exc):
                if create_new:
                    logger.warning(
                        "publish: registry %s sb_id=%s not found on ScienceBase; "
                        "create_new set -- minting a fresh item.",
                        scope if key is None else f"{scope}/{key}",
                        sb_id,
                    )
                    return ("create", None, None, False)
                raise _stale_registry_error(scope, key, sb_id) from exc
            raise
        if not item:
            if create_new:
                logger.warning(
                    "publish: registry %s sb_id=%s returned an empty item; "
                    "create_new set -- minting a fresh item.",
                    scope if key is None else f"{scope}/{key}",
                    sb_id,
                )
                return ("create", None, None, False)
            raise _stale_registry_error(scope, key, sb_id)
        return ("update", sb_id, item, False)

    # No recorded sb_id: adopt an existing same-title child, else create.
    try:
        match = client.find_child(parent_id, title)
    except SbClientError:
        raise  # ambiguous title -- already a contextual ReleaseError
    except Exception as exc:
        raise SbClientError(
            f"failed to enumerate children of {parent_id!r} while searching for a "
            f"child titled {title!r} to adopt "
            f"(for {scope if key is None else f'{scope}/{key}'}): {exc}"
        ) from exc
    if match is not None:
        logger.info(
            "publish: adopting ScienceBase item %s titled %r for %s",
            match.get("id"),
            title,
            scope if key is None else f"{scope}/{key}",
        )
        return ("update", match["id"], match, True)
    return ("create", None, None, False)


def _upload_payload(
    client: SbClient, sb_id: str, payload: list[_PayloadFile]
) -> tuple[list[str], list[str]]:
    """Upload each payload file, skipping ones whose remote MD5 already matches.

    On create the item has no files, so nothing skips and every file uploads --
    same code path as update. Each transfer is logged at INFO before the next
    so that a mid-loop failure (a multi-GB child can be many files) leaves a
    record of how far the upload got -- the propagated exception aborts before
    the aggregate summary in :func:`_reconcile_and_publish` would run.
    """
    uploaded: list[str] = []
    skipped: list[str] = []
    for pf in payload:
        result = client.upload_file(
            sb_id, pf.abs_path, skip_if_sha256_matches=_file_md5(pf.abs_path)
        )
        if result.skipped:
            skipped.append(pf.name)
            logger.info("publish: skipped %s on %s (checksum match)", pf.name, sb_id)
        else:
            uploaded.append(pf.name)
            logger.info("publish: uploaded %s to %s", pf.name, sb_id)
    return uploaded, skipped


def _patch_body(
    client: SbClient, sb_id: str, remote_item: dict, body: dict
) -> tuple[str, ...]:
    """Patch only the item-body fields that differ from the remote item."""
    patch = {k: v for k, v in body.items() if remote_item.get(k) != v}
    if patch:
        client.update_item(sb_id, patch)
    return tuple(sorted(patch))


def _registry_get(
    scope: Scope, key: str | None, registry_path: Path | None
) -> dict | None:
    if scope == "umbrella":
        return registry.get_umbrella(registry_path)
    if scope == "source":
        return registry.get_source(key, registry_path)  # type: ignore[arg-type]
    return registry.get_fabric(key, registry_path)  # type: ignore[arg-type]


def _registry_put(
    scope: Scope,
    key: str | None,
    *,
    sb_id: str,
    payload: list[_PayloadFile],
    build_result: BuildResult,
    title: str,
    mcf: dict,
    now: datetime | None,
    registry_path: Path | None,
) -> dict:
    timestamp = _now_iso(now)
    if scope == "umbrella":
        version = mcf.get("identification", {}).get("edition", "1.0")
        doi = mcf.get("identification", {}).get("doi")
        doi_kwargs = {"doi": doi} if doi else {}
        return registry.put_umbrella(
            sb_id=sb_id,
            version=version,
            published_utc=timestamp,
            title=title,
            path=registry_path,
            **doi_kwargs,
        )
    put = registry.put_source if scope == "source" else registry.put_fabric
    return put(
        key,  # type: ignore[arg-type]
        sb_id=sb_id,
        uploaded_utc=timestamp,
        file_count=len(payload),
        total_bytes=sum(pf.size for pf in payload),
        manifest_sha256=sha256_file(build_result.stage_dir / "SHA256SUMS"),
        path=registry_path,
    )


def _reconcile_and_publish(
    *,
    client: SbClient,
    build_result: BuildResult,
    parent_id: str,
    scope: Scope,
    key: str | None,
    create_new: bool,
    delete_orphans: bool,
    now: datetime | None,
    registry_path: Path | None,
) -> PublishResult:
    """Shared create-vs-update body for every publish scope."""
    mcf = build_result.mcf
    title = mcf["identification"]["title"]
    reg_entry = _registry_get(scope, key, registry_path)
    mode, sb_id, remote_item, adopted = _resolve_target(
        client, reg_entry, parent_id, title, create_new=create_new, scope=scope, key=key
    )
    body = _item_body(mcf)
    payload = _payload_files(build_result)

    orphans: tuple[str, ...] = ()
    orphans_deleted: tuple[str, ...] = ()
    body_patched: tuple[str, ...] = ()

    if mode == "create":
        created = client.create_item(parent_id, body)
        sb_id = created["id"]
        uploaded, skipped = _upload_payload(client, sb_id, payload)
    else:
        assert sb_id is not None and remote_item is not None  # set by _resolve_target
        body_patched = _patch_body(client, sb_id, remote_item, body)
        uploaded, skipped = _upload_payload(client, sb_id, payload)
        staged_names = {pf.name for pf in payload}
        orphans = tuple(sorted(set(_remote_files(remote_item)) - staged_names))
        if orphans:
            if delete_orphans:
                # Deletes are destructive and the registry is only re-stamped
                # after the whole loop, so log each one as it lands -- a
                # mid-loop failure then leaves a trail of what was removed
                # rather than silent, un-recoverable divergence.
                deleted: list[str] = []
                for name in orphans:
                    client.delete_file(sb_id, name)
                    deleted.append(name)
                    logger.info("publish: deleted orphan %s from %s", name, sb_id)
                orphans_deleted = tuple(deleted)
            else:
                logger.warning(
                    "publish: %s has %d orphan file(s) on ScienceBase not in the "
                    "staged payload: %s. Re-run with delete_orphans=True to remove.",
                    scope if key is None else f"{scope}/{key}",
                    len(orphans),
                    ", ".join(orphans),
                )

    entry = _registry_put(
        scope,
        key,
        sb_id=sb_id,
        payload=payload,
        build_result=build_result,
        title=title,
        mcf=mcf,
        now=now,
        registry_path=registry_path,
    )
    logger.info(
        "publish: %s %s sb_id=%s (uploaded=%d skipped=%d orphans=%d)",
        mode,
        scope if key is None else f"{scope}/{key}",
        sb_id,
        len(uploaded),
        len(skipped),
        len(orphans),
    )
    return PublishResult(
        scope=scope,
        key=key,
        sb_id=sb_id,
        mode=mode,
        adopted=adopted,
        uploaded=tuple(uploaded),
        skipped=tuple(skipped),
        orphans=orphans,
        orphans_deleted=orphans_deleted,
        body_patched=body_patched,
        registry_entry=dict(entry),
    )


# ---------------------------------------------------------------------------
# Public publish entry points
# ---------------------------------------------------------------------------


def publish_umbrella(
    project: Project,
    client: SbClient,
    *,
    parent_id: str,
    create_new: bool = False,
    delete_orphans: bool = False,
    allow_incomplete_sources: bool = False,
    now: datetime | None = None,
    registry_path: Path | None = None,
) -> PublishResult:
    """Publish (create or update) the metadata-only umbrella under *parent_id*.

    *parent_id* is the ScienceBase community / folder that hosts the release.
    The version + DOI recorded in the registry follow the built MCF (edition /
    identification.doi), so an operator-set ``config.release.doi`` flows into
    the registry on the next publish.
    """
    _preflight_common(project, allow_incomplete_sources=allow_incomplete_sources)
    build_result = build_umbrella(project, now=now)
    verify_csv(build_result.stage_dir)
    return _reconcile_and_publish(
        client=client,
        build_result=build_result,
        parent_id=parent_id,
        scope="umbrella",
        key=None,
        create_new=create_new,
        delete_orphans=delete_orphans,
        now=now,
        registry_path=registry_path,
    )


def publish_source_child(
    project: Project,
    client: SbClient,
    source_key: str,
    *,
    create_new: bool = False,
    delete_orphans: bool = False,
    allow_incomplete_sources: bool = False,
    now: datetime | None = None,
    registry_path: Path | None = None,
) -> PublishResult:
    """Publish (create or update) the consolidated-source child for *source_key*.

    Parented under the umbrella item recorded in the registry (a missing
    umbrella sb_id is a fatal pre-flight error).
    """
    _preflight_common(project, allow_incomplete_sources=allow_incomplete_sources)
    parent_id = _require_umbrella_sb_id(registry_path)
    build_result = build_source_child(project, source_key, now=now)
    verify_csv(build_result.stage_dir)
    return _reconcile_and_publish(
        client=client,
        build_result=build_result,
        parent_id=parent_id,
        scope="source",
        key=source_key,
        create_new=create_new,
        delete_orphans=delete_orphans,
        now=now,
        registry_path=registry_path,
    )


def publish_fabric_child(
    project: Project,
    client: SbClient,
    *,
    create_new: bool = False,
    delete_orphans: bool = False,
    allow_incomplete_sources: bool = False,
    now: datetime | None = None,
    registry_path: Path | None = None,
) -> PublishResult:
    """Publish (create or update) the fabric child for *project*.

    Parented under the umbrella item recorded in the registry (a missing
    umbrella sb_id is a fatal pre-flight error). The registry key is the
    resolved fabric label.
    """
    _preflight_common(project, allow_incomplete_sources=allow_incomplete_sources)
    parent_id = _require_umbrella_sb_id(registry_path)
    build_result = build_fabric_child(project, now=now)
    verify_csv(build_result.stage_dir)
    return _reconcile_and_publish(
        client=client,
        build_result=build_result,
        parent_id=parent_id,
        scope="fabric",
        key=build_result.key,
        create_new=create_new,
        delete_orphans=delete_orphans,
        now=now,
        registry_path=registry_path,
    )


# ---------------------------------------------------------------------------
# Read-only diff (intent vs reality) -- used by dry-run and status
# ---------------------------------------------------------------------------

DiffScope = Literal["umbrella", "sources", "fabric", "all"]


def _stage_dir(project: Project, scope: Scope, key: str | None) -> Path:
    root = project.release_build_dir
    if scope == "umbrella":
        return root / "umbrella"
    if scope == "source":
        return root / "sources" / key  # type: ignore[operator]
    return root / "fabric" / key  # type: ignore[operator]


def _local_status(stage_dir: Path) -> LocalStatus:
    """Classify the staged payload on disk without rebuilding it."""
    if not (stage_dir / "checksums.csv").exists():
        return "missing"
    try:
        verify_csv(stage_dir)
    except (ChecksumMismatch, FileNotFoundError, OSError):
        # Genuine drift / unlisted / missing-file, or a dangling symlink: the
        # stage exists but is not a clean, complete payload. A narrow catch --
        # a programming error inside verify_csv must surface loudly, not be
        # reported to the operator as benign "stale".
        return "stale"
    return "built"


def _staged_names_sizes(stage_dir: Path) -> dict[str, int]:
    """Basename -> size for every regular file under *stage_dir* (read-only).

    Follows symlinks (the size that would upload); dangling links are skipped
    rather than raised, since this is the read-only diff path. Basename
    collisions resolve last-wins -- acceptable for an overview (the authoritative
    guard is in :func:`_payload_files` on the publish path).
    """
    out: dict[str, int] = {}
    for path in stage_dir.rglob("*"):
        if path.is_symlink() and not path.exists():
            continue
        if not path.is_file():
            continue
        out[path.name] = path.stat().st_size
    return out


def _remote_status(
    client: SbClient,
    reg_entry: dict | None,
    stage_dir: Path,
    local: LocalStatus,
) -> tuple[RemoteStatus, str | None, str | None]:
    """Classify the ScienceBase side; return (status, sb_id, detail)."""
    if not reg_entry or not reg_entry.get("sb_id"):
        return ("missing", None, "no ScienceBase id recorded -- would create")
    sb_id = reg_entry["sb_id"]
    try:
        item = client.get_item(sb_id)
    except Exception as exc:
        if _is_not_found(exc):
            return ("missing", sb_id, "registry sb_id is stale -- item not found")
        raise
    if not item:
        return ("missing", sb_id, "registry sb_id is stale -- item not found")
    if local == "missing":
        return ("present", sb_id, "item exists; local payload not built for comparison")

    staged = _staged_names_sizes(stage_dir)
    remote = _remote_files(item)
    missing_remote = sorted(set(staged) - set(remote))
    extra_remote = sorted(set(remote) - set(staged))
    size_drift = sorted(
        name
        for name in set(staged) & set(remote)
        if remote[name] is not None and remote[name] != staged[name]
    )
    if missing_remote or extra_remote or size_drift:
        parts = []
        if missing_remote:
            parts.append(f"not on ScienceBase: {missing_remote}")
        if extra_remote:
            parts.append(f"orphaned on ScienceBase: {extra_remote}")
        if size_drift:
            parts.append(f"size differs: {size_drift}")
        return ("drift", sb_id, "; ".join(parts))
    return ("present", sb_id, None)


def _diff_one(
    project: Project,
    client: SbClient,
    scope: Scope,
    key: str | None,
    registry_path: Path | None,
) -> ScopeDiff:
    stage_dir = _stage_dir(project, scope, key)
    local = _local_status(stage_dir)
    reg_entry = _registry_get(scope, key, registry_path)
    remote, sb_id, detail = _remote_status(client, reg_entry, stage_dir, local)
    return ScopeDiff(
        scope=scope, key=key, sb_id=sb_id, local=local, remote=remote, detail=detail
    )


def diff_local_vs_remote(
    project: Project,
    client: SbClient,
    *,
    scope: DiffScope = "all",
    registry_path: Path | None = None,
) -> list[ScopeDiff]:
    """Read-only intent-vs-reality diff for the requested scope(s).

    Does **not** build or write anything: it inspects the existing stage dirs
    (built / stale / missing), the registry (recorded sb_ids), and ScienceBase
    (present / drift / missing) via the injected client. Source children come
    from :func:`payload.sources_used` and the fabric label from
    :func:`payload.resolve_fabric_label`, so the scope set is catalog- /
    project-driven, never hard-coded.
    """
    diffs: list[ScopeDiff] = []
    if scope in ("fabric", "all"):
        label = resolve_fabric_label(project)
        diffs.append(_diff_one(project, client, "fabric", label, registry_path))
    if scope in ("sources", "all"):
        for source_key in sources_used(project):
            diffs.append(
                _diff_one(project, client, "source", source_key, registry_path)
            )
    if scope in ("umbrella", "all"):
        diffs.append(_diff_one(project, client, "umbrella", None, registry_path))
    return diffs
