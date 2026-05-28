"""Read-only dry-run report for a ScienceBase release.

A dry run answers "what *would* publishing do, and is the payload valid?"
without touching ScienceBase. It:

1. runs the offline build (idempotent) for the requested scope,
2. validates each rendered ``fgdc.xml`` with the USGS Metadata Parser
   (:func:`~nhf_spatial_targets.release.validate_xml.validate_xml_file`),
   which warns -- rather than fails -- when ``mp`` is absent, and
3. queries ScienceBase **read-only** (via the injected client, reusing
   :func:`~nhf_spatial_targets.release.publish.diff_local_vs_remote`) for a
   parity check: would each item be created, or does it already exist / drift?

It writes nothing to ScienceBase. It returns a structured :class:`DryRunReport`
(so the CLI and tests can assert on it without scraping the rendered table) and,
unless suppressed, prints a Rich summary table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nhf_spatial_targets.release import publish
from nhf_spatial_targets.release.build import BuildResult, BuildScope, build_all
from nhf_spatial_targets.release.publish import Scope, ScopeDiff
from nhf_spatial_targets.release.sb_client import SbClient
from nhf_spatial_targets.release.validate_xml import MpResult, validate_xml_file
from nhf_spatial_targets.workspace import Project


@dataclass(frozen=True)
class DryRunItem:
    """One item's dry-run findings: payload size, mp result, and SB parity."""

    scope: Scope
    key: str | None
    sb_id: str | None
    file_count: int
    total_bytes: int
    mp: MpResult
    diff: ScopeDiff

    @property
    def label(self) -> str:
        return self.scope if self.key is None else f"{self.scope}/{self.key}"

    @property
    def mp_status(self) -> str:
        """``"ok"`` / ``"warn"`` / ``"err"`` from the mp result."""
        if self.mp.errors:
            return "err"
        if self.mp.warnings:
            return "warn"
        return "ok"

    @property
    def remote_label(self) -> str:
        """ScienceBase parity word: ``exists`` / ``drift`` / ``missing``."""
        return "exists" if self.diff.remote == "present" else self.diff.remote


@dataclass(frozen=True)
class DryRunReport:
    """The structured result of a dry run; ``items`` in build order."""

    items: tuple[DryRunItem, ...]

    @property
    def ok(self) -> bool:
        """True when every item's FGDC XML validated without mp errors."""
        return all(not item.mp.errors for item in self.items)

    @property
    def would_create(self) -> tuple[DryRunItem, ...]:
        return tuple(i for i in self.items if i.diff.remote == "missing")


def _payload_size(build_result: BuildResult) -> tuple[int, int]:
    """Return (file_count, total_bytes) of everything publish would upload.

    ``BuildResult.entries`` covers every staged file except the two integrity
    manifests; add those so the count matches the real upload set.
    """
    count = len(build_result.entries)
    total = sum(e.size for e in build_result.entries)
    for extra in ("checksums.csv", "SHA256SUMS"):
        path = build_result.stage_dir / extra
        if path.exists():
            count += 1
            total += path.stat().st_size
    return count, total


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # unreachable; satisfies the type checker


def _render_table(report: DryRunReport, console: Console) -> None:
    table = Table(title="ScienceBase release dry run")
    table.add_column("scope")
    table.add_column("ScienceBase id")
    table.add_column("files", justify="right")
    table.add_column("size", justify="right")
    table.add_column("mp")
    table.add_column("sb")
    for item in report.items:
        table.add_row(
            item.label,
            item.sb_id or "would create",
            str(item.file_count),
            _human_bytes(item.total_bytes),
            item.mp_status,
            item.remote_label,
        )
    console.print(table)


def dry_run(
    project: Project,
    client: SbClient,
    *,
    scope: BuildScope = "fabric",
    now: datetime | None = None,
    registry_path: Path | None = None,
    console: Console | None = None,
    render: bool = True,
) -> DryRunReport:
    """Build, validate, and read-only-diff the requested scope; report findings.

    Parameters
    ----------
    project, client
        The project to report on and an already-constructed (mock or live)
        ScienceBase client. Only read-only client methods are called.
    scope
        ``"fabric"`` (default), ``"sources"``, or ``"all"`` -- the same scope
        vocabulary as :func:`~nhf_spatial_targets.release.build.build_all`.
    now
        Injectable build clock (keeps rendered metadata deterministic in tests).
    registry_path
        Registry file path (tests pass a tmp path); defaults to the committed
        registry.
    console, render
        ``console`` overrides the output target (e.g. a captured console);
        ``render=False`` suppresses printing entirely (the report is still
        returned). Neither affects what is queried.

    Returns
    -------
    DryRunReport
        One :class:`DryRunItem` per built item, in build order. **No ScienceBase
        writes occur.**
    """
    build_results = build_all(project, scope=scope, now=now)
    diffs = {
        (d.scope, d.key): d
        for d in publish.diff_local_vs_remote(
            project, client, scope=scope, registry_path=registry_path
        )
    }

    items: list[DryRunItem] = []
    for result in build_results:
        diff = diffs.get((result.kind, result.key))
        if diff is None:
            # Defensive: build and diff enumerate the same scope set, so a build
            # result without a matching diff means the two drifted -- surface it
            # rather than silently dropping the item from the report.
            raise RuntimeError(
                f"dry-run: built {result.kind}/{result.key} has no matching "
                f"ScienceBase diff; build and diff scope enumeration disagree."
            )
        mp = validate_xml_file(result.stage_dir / "fgdc.xml")
        file_count, total_bytes = _payload_size(result)
        items.append(
            DryRunItem(
                scope=result.kind,
                key=result.key,
                sb_id=diff.sb_id,
                file_count=file_count,
                total_bytes=total_bytes,
                mp=mp,
                diff=diff,
            )
        )

    report = DryRunReport(items=tuple(items))
    if render:
        _render_table(report, console or Console())
    return report
