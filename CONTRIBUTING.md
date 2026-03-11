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

## Data Sources

- All source metadata lives in `catalog/sources.yml` — do not hardcode URLs or product names
- When adding a new source, add it to `catalog/sources.yml` first, then write the fetch module
