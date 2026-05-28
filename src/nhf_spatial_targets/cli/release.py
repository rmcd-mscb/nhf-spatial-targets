"""``nhf-targets release`` sub-app: build and publish the ScienceBase release.

This module is the **CLI surface** over the offline build orchestrator
(:mod:`nhf_spatial_targets.release.build`) and the idempotent publish / dry-run
orchestration (:mod:`nhf_spatial_targets.release.publish` /
:mod:`~nhf_spatial_targets.release.dry_run`). It adds no release logic of its
own -- every subcommand resolves a :class:`~nhf_spatial_targets.workspace.Project`
(and, where ScienceBase is touched, an
:class:`~nhf_spatial_targets.release.sb_client.SbClient`) and dispatches.

Two seams are deliberately small so PR-G can fill them in without touching the
subcommand bodies:

- :func:`_load_project` wraps :func:`nhf_spatial_targets.workspace.load` (a test
  seam; the dispatch tests patch it to return a synthetic project).
- :func:`_resolve_client` builds the live ``SbClient``. **For PR-F** it resolves
  a Keycloak token from an explicit ``--token`` or the ``SB_TOKEN`` environment
  variable via :meth:`SbClient.from_token`. The ``.credentials.yml``
  ``sciencebase:`` block + ``materialize_sciencebase_token`` wiring is PR-G; a
  clear "no credentials found" error fires until then.

The umbrella's ScienceBase parent (community / folder id) is supplied with the
``--parent-id`` flag rather than a config key: it is a deployment-level value
shared across every project publishing to the same umbrella, it is needed only
for the rare ``publish --scope umbrella`` / ``--create-umbrella`` path, and a
per-project config key would both duplicate it and pre-commit a schema before
PR-G clarifies where ScienceBase deployment settings live.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App, Parameter

from nhf_spatial_targets.release.build import build_all
from nhf_spatial_targets.release.checksums import compute_checksums
from nhf_spatial_targets.release.config import scaffold_release_yml
from nhf_spatial_targets.release.dry_run import dry_run
from nhf_spatial_targets.release.publish import (
    PublishResult,
    diff_local_vs_remote,
    publish_fabric_child,
    publish_source_child,
    publish_umbrella,
)

if TYPE_CHECKING:
    from nhf_spatial_targets.release.sb_client import SbClient
    from nhf_spatial_targets.workspace import Project

_logger = logging.getLogger(__name__)

release_app = App(
    name="release",
    help="Build, validate, and publish the ScienceBase data release.",
)

# --project-dir is shared by every subcommand except auth-test (which needs only
# credentials). The default lives on each signature so the type checker sees it.
_PROJECT_DIR_PARAM = Parameter(
    name=["--project-dir", "-d"],
    help="Project created by 'nhf-targets init'.",
)
_TOKEN_PARAM = Parameter(
    name=["--token"],
    help=(
        "ScienceBase Keycloak token JSON. Falls back to the SB_TOKEN "
        "environment variable. (.credentials.yml wiring is PR-G.)"
    ),
)
_ENV_PARAM = Parameter(
    name=["--env"],
    help="ScienceBase environment (e.g. 'beta'); default is production.",
)

# A publish --scope maps to the dry_run build scope used for its no-confirm
# preview. dry_run has no single-source or umbrella-only scope, so a 'source'
# preview shows every source child and an 'umbrella' preview shows everything
# (the umbrella's metadata unions its children, so 'all' is the faithful
# preview). --confirm then publishes only the requested scope.
_PUBLISH_PREVIEW_SCOPE: dict[str, Literal["fabric", "sources", "all"]] = {
    "fabric": "fabric",
    "source": "sources",
    "umbrella": "all",
}

_README_NOTES_PLACEHOLDER = """\
# Release abstract notes

Text added here is appended to the fabric child's abstract by the release
metadata builders. Leave empty to use the catalog defaults. Fill `authors`
and `ipds_number` in `release.yml` (this directory) before publishing.
"""


# ---------------------------------------------------------------------------
# Errors + seams (the two PR-G fills in)
# ---------------------------------------------------------------------------


class ReleaseCliError(Exception):
    """A CLI-level precondition failed (e.g. no credentials, missing flag)."""


def _load_project(workdir: Path) -> Project:
    """Load a project from disk.

    A thin wrapper over :func:`nhf_spatial_targets.workspace.load` so the
    dispatch tests can patch one name to inject a synthetic project.
    """
    from nhf_spatial_targets.workspace import load as load_project

    return load_project(workdir)


def _resolve_client(*, token: str | None = None, env: str | None = None) -> SbClient:
    """Construct an :class:`SbClient` for live ScienceBase operations.

    **PR-F seam.** Resolves a Keycloak token from *token* (the ``--token`` flag)
    or, failing that, the ``SB_TOKEN`` environment variable, and builds the
    client via :meth:`SbClient.from_token`. PR-G extends this to read the
    ``sciencebase:`` block from ``.credentials.yml`` (via
    ``materialize_sciencebase_token``); until then a token must be supplied.

    Raises
    ------
    ReleaseCliError
        No token was provided and ``SB_TOKEN`` is unset.
    """
    import os

    from nhf_spatial_targets.release.sb_client import SbClient

    resolved = token or os.environ.get("SB_TOKEN")
    if not resolved:
        raise ReleaseCliError(
            "no ScienceBase credentials found: pass --token or set the SB_TOKEN "
            "environment variable. (Reading the `sciencebase:` block from "
            ".credentials.yml lands in PR-G.)"
        )
    return SbClient.from_token(resolved, env=env)


# ---------------------------------------------------------------------------
# Small exit helpers (mirror the run.py / agg.py conventions)
# ---------------------------------------------------------------------------


def _fail(message: str, code: int) -> None:
    """Print *message* to stderr and exit with *code* (never returns)."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _load_project_or_exit(workdir: Path) -> Project:
    """Resolve the project, exiting 2 (missing dir) / 1 (bad config) on failure."""
    if not workdir.exists():
        _fail(f"Project not found: {workdir}", 2)
    try:
        return _load_project(workdir)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc), 1)
        raise  # unreachable; satisfies the type checker that this returns Project


def _resolve_client_or_exit(*, token: str | None, env: str | None) -> SbClient:
    """Construct the client, exiting 1 with a clear message when it can't."""
    try:
        return _resolve_client(token=token, env=env)
    except ReleaseCliError as exc:
        _fail(str(exc), 1)
        raise  # unreachable; satisfies the type checker


# ---------------------------------------------------------------------------
# Subcommands -- local / offline
# ---------------------------------------------------------------------------


@release_app.command(name="init")
def init(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
):
    """Scaffold the per-project release directory.

    Writes ``<project>/release/release.yml`` (operator fills authors +
    ipds_number), the empty ``release/build/`` staging dir, and a
    ``release/README.notes.md`` placeholder. Local only; never clobbers an
    existing ``release.yml``.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    if not workdir.exists():
        _fail(f"Project not found: {workdir}", 2)

    release_dir = workdir / "release"
    yml_path = scaffold_release_yml(release_dir)
    (release_dir / "build").mkdir(parents=True, exist_ok=True)
    notes_path = release_dir / "README.notes.md"
    if not notes_path.exists():
        notes_path.write_text(_README_NOTES_PLACEHOLDER)

    msg = Text()
    msg.append("Release scaffold created:\n", style="bold green")
    msg.append(f"  {release_dir}\n\n")
    msg.append("Next steps:\n", style="bold")
    msg.append(f"  1. Edit  {yml_path}  (authors, ipds_number)\n")
    msg.append(f"  2. Run   nhf-targets release build --project-dir {workdir}\n")
    Console().print(Panel(msg, title="nhf-targets release init", border_style="green"))


@release_app.command(name="build")
def build(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    scope: Annotated[
        Literal["fabric", "sources", "all"],
        Parameter(
            name=["--scope"],
            help="fabric child (default), all source children, or all + umbrella.",
        ),
    ] = "fabric",
    copy: Annotated[
        bool,
        Parameter(
            name=["--copy"],
            help="Copy staged files instead of symlinking (no-symlink filesystems).",
        ),
    ] = False,
):
    """Build a complete release payload on disk under ``release/build/``.

    Stages files (symlink by default), renders FGDC / ISO / README, and writes
    checksums for each item the *scope* selects. Entirely offline.
    """
    from rich.console import Console

    project = _load_project_or_exit(workdir)
    try:
        results = build_all(project, scope=scope, copy=copy)
    except (FileNotFoundError, ValueError, NotADirectoryError, OSError) as exc:
        _fail(str(exc), 1)

    console = Console()
    console.print(f"[bold green]Built {len(results)} release item(s):[/bold green]")
    for result in results:
        label = result.kind if result.key is None else f"{result.kind}/{result.key}"
        console.print(f"  {label}: {len(result.entries)} file(s) -> {result.stage_dir}")


@release_app.command(name="manifest")
def manifest(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    scope: Annotated[
        Literal["fabric", "sources", "all"],
        Parameter(
            name=["--scope"],
            help="Which already-staged items to re-checksum (default: fabric).",
        ),
    ] = "fabric",
):
    """Recompute ``checksums.csv`` + ``SHA256SUMS`` for staged items.

    Re-runs the checksum pass over the staged directories already on disk (no
    XML re-render, no client). Use after editing a file that is already staged.
    """
    from rich.console import Console

    if not workdir.exists():
        _fail(f"Project not found: {workdir}", 2)

    stage_dirs = _staged_dirs(workdir / "release" / "build", scope)
    if not stage_dirs:
        _fail(
            f"no staged items found under {workdir / 'release' / 'build'} for scope "
            f"'{scope}'; run 'nhf-targets release build' first.",
            1,
        )

    console = Console()
    try:
        for stage_dir in stage_dirs:
            entries = compute_checksums(stage_dir)
            console.print(
                f"  {stage_dir.relative_to(workdir)}: re-checksummed "
                f"{len(entries)} file(s)"
            )
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        _fail(str(exc), 1)
    console.print("[bold green]Checksums recomputed.[/bold green]")


def _staged_dirs(build_root: Path, scope: str) -> list[Path]:
    """Existing staged item directories under *build_root* for *scope*.

    Discovered by globbing what is on disk (not by re-deriving the catalog's
    source / fabric set), so ``manifest`` re-checksums exactly what ``build``
    left behind.
    """
    dirs: list[Path] = []
    if scope in ("fabric", "all"):
        fabric_root = build_root / "fabric"
        if fabric_root.is_dir():
            dirs += sorted(p for p in fabric_root.iterdir() if p.is_dir())
    if scope in ("sources", "all"):
        sources_root = build_root / "sources"
        if sources_root.is_dir():
            dirs += sorted(p for p in sources_root.iterdir() if p.is_dir())
    if scope == "all":
        umbrella = build_root / "umbrella"
        if umbrella.is_dir():
            dirs.append(umbrella)
    return dirs


# ---------------------------------------------------------------------------
# Subcommands -- ScienceBase (need an SbClient)
# ---------------------------------------------------------------------------


@release_app.command(name="dry-run")
def dry_run_cmd(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    scope: Annotated[
        Literal["fabric", "sources", "all"],
        Parameter(name=["--scope"], help="fabric (default), sources, or all."),
    ] = "fabric",
    token: Annotated[str | None, _TOKEN_PARAM] = None,
    env: Annotated[str | None, _ENV_PARAM] = None,
):
    """Preview a release: build, mp-validate, and read-only-diff ScienceBase.

    Writes nothing to ScienceBase. Prints the dry-run summary table.
    """
    project = _load_project_or_exit(workdir)
    client = _resolve_client_or_exit(token=token, env=env)
    dry_run(project, client, scope=scope)


@release_app.command(name="status")
def status(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    scope: Annotated[
        Literal["umbrella", "sources", "fabric", "all"],
        Parameter(name=["--scope"], help="Which scope(s) to diff (default: all)."),
    ] = "all",
    token: Annotated[str | None, _TOKEN_PARAM] = None,
    env: Annotated[str | None, _ENV_PARAM] = None,
):
    """Read-only intent-vs-reality diff of the registry against ScienceBase."""
    from rich.console import Console
    from rich.table import Table

    project = _load_project_or_exit(workdir)
    client = _resolve_client_or_exit(token=token, env=env)
    diffs = diff_local_vs_remote(project, client, scope=scope)

    table = Table(title="ScienceBase release status")
    table.add_column("scope")
    table.add_column("ScienceBase id")
    table.add_column("local")
    table.add_column("remote")
    table.add_column("detail")
    for diff in diffs:
        label = diff.scope if diff.key is None else f"{diff.scope}/{diff.key}"
        table.add_row(
            label, diff.sb_id or "-", diff.local, diff.remote, diff.detail or ""
        )
    Console().print(table)


@release_app.command(name="publish")
def publish(
    workdir: Annotated[Path, _PROJECT_DIR_PARAM],
    scope: Annotated[
        Literal["umbrella", "source", "fabric"],
        Parameter(name=["--scope"], help="What to publish (default: fabric)."),
    ] = "fabric",
    source_key: Annotated[
        str | None,
        Parameter(
            name=["--source-key"],
            help="Source key to publish; required with --scope source --confirm.",
        ),
    ] = None,
    confirm: Annotated[
        bool,
        Parameter(
            name=["--confirm"],
            help="Actually publish. Without it, this is a dry-run preview.",
        ),
    ] = False,
    create_umbrella: Annotated[
        bool,
        Parameter(
            name=["--create-umbrella"],
            help="Publish the umbrella first, then the child (needs --parent-id).",
        ),
    ] = False,
    create_new: Annotated[
        bool,
        Parameter(
            name=["--create-new"],
            help="If the registry sb_id is gone on ScienceBase, mint a fresh item.",
        ),
    ] = False,
    delete_orphans: Annotated[
        bool,
        Parameter(
            name=["--delete-orphans"],
            help="Delete remote files no longer in the staged payload.",
        ),
    ] = False,
    parent_id: Annotated[
        str | None,
        Parameter(
            name=["--parent-id"],
            help="ScienceBase community/folder id hosting the umbrella.",
        ),
    ] = None,
    token: Annotated[str | None, _TOKEN_PARAM] = None,
    env: Annotated[str | None, _ENV_PARAM] = None,
):
    """Publish one scope to ScienceBase (idempotent create-vs-update).

    Without ``--confirm`` this previews via a dry run and writes nothing. With
    ``--confirm`` it dispatches to the matching ``publish_*`` orchestrator,
    writes a ``release/last_publish_<scope>_<key>.log``, and prints the
    registry-commit reminder.
    """
    project = _load_project_or_exit(workdir)
    client = _resolve_client_or_exit(token=token, env=env)

    if not confirm:
        # Preview only -- the dry run builds + mp-validates locally and queries
        # ScienceBase read-only. See _PUBLISH_PREVIEW_SCOPE for the mapping.
        dry_run(project, client, scope=_PUBLISH_PREVIEW_SCOPE[scope])
        target = (
            scope if scope != "source" else f"source/{source_key or '<source-key>'}"
        )
        from rich.console import Console

        Console().print(
            f"[yellow]Dry run only.[/yellow] Re-run with --confirm to publish {target}."
        )
        return

    # --- confirmed publish ---
    if scope == "source" and not source_key:
        _fail("--scope source --confirm requires --source-key.", 2)
    if (scope == "umbrella" or create_umbrella) and not parent_id:
        _fail(
            "publishing the umbrella requires --parent-id (the ScienceBase "
            "community/folder id).",
            2,
        )

    try:
        results: list[PublishResult] = []
        if create_umbrella and scope != "umbrella":
            results.append(
                publish_umbrella(
                    project,
                    client,
                    parent_id=parent_id,  # guarded above
                    create_new=create_new,
                    delete_orphans=delete_orphans,
                )
            )
        if scope == "umbrella":
            results.append(
                publish_umbrella(
                    project,
                    client,
                    parent_id=parent_id,  # guarded above
                    create_new=create_new,
                    delete_orphans=delete_orphans,
                )
            )
        elif scope == "source":
            results.append(
                publish_source_child(
                    project,
                    client,
                    source_key,  # guarded above
                    create_new=create_new,
                    delete_orphans=delete_orphans,
                )
            )
        else:  # fabric
            results.append(
                publish_fabric_child(
                    project,
                    client,
                    create_new=create_new,
                    delete_orphans=delete_orphans,
                )
            )
    except Exception as exc:  # noqa: BLE001 -- surface any publish failure cleanly
        _logger.exception("release publish failed")
        _fail(f"publish failed: {exc}", 1)

    log_path = _write_publish_log(project, scope, results)
    _print_publish_summary(results, log_path)


# ---------------------------------------------------------------------------
# Subcommands -- credentials liveness
# ---------------------------------------------------------------------------


@release_app.command(name="auth-test")
def auth_test(
    token: Annotated[str | None, _TOKEN_PARAM] = None,
    env: Annotated[str | None, _ENV_PARAM] = None,
):
    """Construct a client, ping ScienceBase, and print the session identity."""
    from rich.console import Console

    client = _resolve_client_or_exit(token=token, env=env)
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 -- a failed ping is the answer here
        _fail(f"ScienceBase ping failed: {exc}", 1)
    who = client.whoami()
    Console().print(
        f"[bold green]ScienceBase OK[/bold green] "
        f"user={who.get('username') or '<unknown>'} "
        f"logged_in={who.get('logged_in')}"
    )


# ---------------------------------------------------------------------------
# Publish logging / summary
# ---------------------------------------------------------------------------


def _summarize_result(result: PublishResult) -> str:
    """One-line human summary of a published item."""
    label = result.scope if result.key is None else f"{result.scope}/{result.key}"
    return (
        f"[{label}] mode={result.mode} sb_id={result.sb_id} "
        f"adopted={result.adopted} uploaded={len(result.uploaded)} "
        f"skipped={len(result.skipped)} orphans={len(result.orphans)} "
        f"orphans_deleted={len(result.orphans_deleted)} "
        f"body_patched={','.join(result.body_patched) or '-'}"
    )


def _write_publish_log(
    project: Project, scope: str, results: list[PublishResult]
) -> Path:
    """Write a publish log named for the command's scope; return its path.

    The log is named after the primary (last) result -- the child for a
    ``--create-umbrella`` + child run, the single item otherwise -- and records
    every item this invocation published.
    """
    primary = results[-1]
    key_token = primary.key if primary.key is not None else primary.scope
    safe_key = key_token.replace("/", "_")
    project.release_dir.mkdir(parents=True, exist_ok=True)
    log_path = project.release_dir / f"last_publish_{scope}_{safe_key}.log"
    lines = [
        "ScienceBase release publish log",
        f"written_utc: {datetime.now(timezone.utc).isoformat()}",
        f"project: {project.workdir}",
        "",
    ]
    lines += [_summarize_result(r) for r in results]
    lines.append("")
    lines.append("Reminder: commit the registry so other operators see this publish:")
    lines.append("  git add catalog/release_registry.yml && git commit")
    log_path.write_text("\n".join(lines) + "\n")
    return log_path


def _print_publish_summary(results: list[PublishResult], log_path: Path) -> None:
    from rich.console import Console

    console = Console()
    console.print("[bold green]Published:[/bold green]")
    for result in results:
        console.print(f"  {_summarize_result(result)}")
    console.print(f"\nLog: {log_path}")
    console.print(
        "\n[bold]Commit the registry so other operators see this publish:[/bold]\n"
        "  git add catalog/release_registry.yml && git commit"
    )
