# Contributing to nhf-spatial-targets

## Prerequisites

- [pixi](https://pixi.sh) — manages Python environments and dependencies

## Setup

```bash
git clone <repo-url>
cd nhf-spatial-targets
pixi install -e dev
pixi run -e dev pre-commit install
```

## Development Workflow

1. **Create an issue** on GitHub describing the work.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b <type>/<issue#>-short-description
   ```
   Types: `feature`, `fix`, `refactor`, `docs`, `test`, `chore`
3. **Develop** on the branch. Pre-commit hooks will automatically run formatting checks, linting, and unit tests on each commit.
4. **Open a pull request** referencing the issue (e.g., "Closes #12").
5. **CI must pass.** PRs are squash-merged after review.

## Running Checks Manually

```bash
pixi run -e dev fmt           # auto-format code
pixi run -e dev fmt-check     # check formatting without modifying
pixi run -e dev lint          # lint with ruff
pixi run -e dev test          # run full test suite
```

## Code Conventions

- Python >=3.11, `from __future__ import annotations` in all modules
- Type hints on all public functions
- Ruff for lint and format (line length 88)
- New modules in `fetch/`, `aggregate/`, `normalize/`, or `targets/` must have a corresponding `tests/test_<module>.py`

If you have Python experience but limited software-engineering background and want to understand the non-obvious patterns (`if TYPE_CHECKING:`, frozen dataclass adapters, atomic NetCDF writes, `flock`-guarded manifest writes, fingerprint-based cache invalidation), read [`docs/architecture/python-patterns.md`](docs/architecture/python-patterns.md). It explains why every module looks the way it does in ~7 short sections.

## Data Sources

- All source metadata lives in `catalog/sources.yml` — do not hardcode URLs or product names
- When adding a new source, add it to `catalog/sources.yml` first, then write the fetch module

## Keeping documentation current

When implementing a new target builder or source, update these documentation surfaces in the same PR:

- **`README.md` §Implementation Status** — flip the matching row to **Done** with the PR reference
- **`README.md` §Calibration Targets** — refresh the per-target row if sources, period, or method changes
- **`README.md` §Fetch & Consolidation Pipeline** — refresh the per-source row if a new source lands
- **`docs/sources/<source_key>.md`** — operator notes for new sources (see #220)

A stale README is the highest-impact trust signal for new operators. CI does not catch documentation drift; reviewers should.
