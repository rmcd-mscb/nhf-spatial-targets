# CLI Migration (Click to Cyclopts) and Logging Setup

**Date:** 2026-03-11
**Status:** Approved

## Goal

Replace Click with Cyclopts for the CLI framework, leveraging type-hint-driven commands and Pydantic integration. Add structured logging using stdlib `logging` with Rich console handler.

## Design

### CLI Migration

#### App structure

Single `cli.py` file. The root `App` has `run` and `init` registered directly as commands, plus two sub-apps (`fetch`, `catalog`) for grouped subcommands:

```python
from importlib.metadata import version as _pkg_version

app = App(name="nhf-targets", help="...", version=_pkg_version("nhf-spatial-targets"))
fetch_app = App(name="fetch", help="Download source datasets into a run workspace.")
catalog_app = App(name="catalog", help="Inspect the data source catalog.")
app.command(fetch_app)
app.command(catalog_app)
```

`run` and `init` are registered directly on the root `app`. `fetch merra2`, `catalog sources`, and `catalog variables` are registered on their respective sub-apps.

#### Global `--verbose` / `-v` flag

Implemented via cyclopts meta app pattern. The meta default receives `--verbose` before dispatching to the actual command:

```python
@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    verbose: Annotated[bool, Parameter(name=["--verbose", "-v"])] = False,
):
    setup_logging(verbose)
    app(tokens)  # dispatch remaining tokens to the root app
```

Entry point: `main = app.meta` (assigned at module level, referenced by `pyproject.toml`).

#### Commands

Commands are plain functions decorated with `@app.command` or `@sub_app.command`. Parameters use `typing.Annotated` with `cyclopts.Parameter` for help text and aliases. Short aliases (`-r`, `-c`, `-f`, etc.) are preserved from the Click version.

**`run` command:**
```python
@app.command
def run(
    run_dir: Annotated[Path | None, Parameter(name=["--run-dir", "-r"], help="...")] = None,
    config: Annotated[Path | None, Parameter(name=["--config", "-c"], help="...")] = None,
    target: Annotated[str | None, Parameter(name=["--target", "-t"], help="...")] = None,
):
```

**`init` command:**
```python
@app.command
def init(
    fabric: Annotated[Path, Parameter(name=["--fabric", "-f"], help="...")],
    id_col: Annotated[str, Parameter(name="--id-col", help="...")] = "nhm_id",
    config: Annotated[Path | None, Parameter(name=["--config", "-c"], help="...")] = None,
    workdir: Annotated[Path | None, Parameter(name=["--workdir", "-w"], help="...")] = None,
    run_label: Annotated[str | None, Parameter(name="--id", help="...")] = None,
    buffer: Annotated[float, Parameter(name="--buffer", help="...")] = 0.1,
):
```

**`fetch merra2` command:**
```python
@fetch_app.command(name="merra2")
def fetch_merra2_cmd(
    run_dir: Annotated[Path, Parameter(name=["--run-dir", "-r"], help="...")],
    period: Annotated[str, Parameter(name=["--period", "-p"], help="...")],
):
```

**`catalog sources` and `catalog variables`:**
```python
@catalog_app.command(name="sources")
def catalog_sources():

@catalog_app.command(name="variables")
def catalog_variables():
```

#### Error handling

- Cyclopts handles argument parsing errors (missing required args, bad types) and exits with usage message (exit code 2).
- Business logic exceptions (`ValueError`, `RuntimeError`, `FileNotFoundError`) propagate naturally with tracebacks. No Click-specific exception types needed.
- The `run` command validates that exactly one of `--run-dir` or `--config` is provided. If both or neither are given, prints `"Error: Provide --run-dir or --config, not both."` to stderr and calls `sys.exit(2)`.
- The `init` command checks that the resolved config path exists before calling `init_run()`. If not found, prints an error and calls `sys.exit(1)`.
- The `init` command catches `FileExistsError` from `init_run()`, prints the message, and calls `sys.exit(1)`.
- `_dispatch()` raises `SystemExit(1)` with a printed error for unknown targets or unregistered builders (replaces `click.ClickException`).
- The `run` command's `yaml` usage (loading `config_path` via `yaml.safe_load`) is unchanged from the Click version.

#### Rich output

Rich usage stays the same — `Console`, `Panel`, `Text` for styled output in `init` and `fetch merra2`. `catalog` commands use `rich.print`. These are user-facing command results, not logging.

#### Dependency changes

- Remove: `click` from `pixi.toml` and `pyproject.toml` `[project.dependencies]`
- Add: `cyclopts` to `pixi.toml` and `pyproject.toml` `[project.dependencies]`

Note: Although CLAUDE.md discourages editing `pyproject.toml` dependencies, the `click` entry already exists there and must be swapped to `cyclopts` to keep the package installable outside pixi (e.g., `pip install -e .`).

#### Entry point

`pyproject.toml`:
```toml
[project.scripts]
nhf-targets = "nhf_spatial_targets.cli:main"
```

Where `main = app.meta` is assigned at module level in `cli.py`.

### Logging Setup

#### Module: `src/nhf_spatial_targets/_logging.py`

Single function `setup_logging(verbose: bool)`:

```python
import logging
from rich.logging import RichHandler

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )
    logging.getLogger("earthaccess").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
```

- Called once from the CLI meta launcher before command dispatch.
- `RichHandler` provides colored, readable console output.
- `show_path=verbose` shows source file paths only in debug mode.
- Third-party loggers (`earthaccess`, `urllib3`) suppressed to WARNING to avoid noise. If `s3transfer`, `botocore`, or `fsspec` prove noisy during integration testing, suppress them to WARNING as well.

#### Logger usage in modules

Each module creates its own logger:

```python
import logging
logger = logging.getLogger(__name__)
```

**INFO level** (default, operational status):
- Fetch: auth status, granule count found, download count, output directory
- Init: fabric path, computed bbox, workspace created

**DEBUG level** (`--verbose`):
- Search parameters (bbox tuple, temporal range)
- Full provenance dict
- Catalog metadata lookups

#### Logging vs Rich console output

- **Logging** (`logger.info/debug`): operational telemetry — what the system is doing
- **Rich console** (`console.print`, `rich.print`): command results — what the user asked for (provenance JSON, styled panels, catalog listings)

### Files

- Modify: `src/nhf_spatial_targets/cli.py` (rewrite with cyclopts)
- Create: `src/nhf_spatial_targets/_logging.py`
- Modify: `src/nhf_spatial_targets/fetch/merra2.py` (add logger calls)
- Modify: `pixi.toml` (swap click for cyclopts)
- Modify: `pyproject.toml` (swap click for cyclopts in dependencies)

### Testing

- Existing unit tests for `fetch/merra2.py`, `catalog.py`, `init_run.py` are unaffected (they test business logic, not CLI).
- No CLI tests exist currently; none added in this change (CLI is thin dispatch layer).
- Verify CLI works manually: `nhf-targets --help`, `nhf-targets fetch merra2 --help`, `nhf-targets --verbose catalog sources`.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CLI framework | cyclopts | Type-hint-driven, Pydantic support, less boilerplate than Click |
| Global verbose | meta app pattern with `-v` alias | cyclopts idiom for global options before command dispatch |
| Short aliases | Preserve all (`-r`, `-c`, `-f`, `-w`, `-p`, `-t`, `-v`) | Usability parity with Click version |
| Logging library | stdlib + RichHandler | Zero new deps (Rich already present), plays well with earthaccess |
| Log levels | INFO default, DEBUG with `-v` | Two levels sufficient for CLI tool |
| Noisy loggers | Suppress to WARNING | earthaccess/urllib3 flood console at INFO |
| Logging module | `_logging.py` | Underscore prefix signals internal module |
| Entry point | `main = app.meta` | Keeps `pyproject.toml` entry point unchanged |
| Version string | Dynamic via `importlib.metadata` | Avoids hardcoded version diverging from pyproject.toml |
| Error exit codes | 2 for usage errors, 1 for runtime errors | Consistent with Unix conventions and Click's prior behavior |
