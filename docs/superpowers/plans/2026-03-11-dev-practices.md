# Development Best Practices Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pre-commit hooks, GitHub Actions CI, branch protection, Claude hooks, and contributor documentation so all developers follow consistent quality gates.

**Architecture:** Configuration-only changes — no application code modified. Pre-commit framework runs pixi tasks locally; GitHub Actions duplicates the same checks in CI. Claude-specific hooks layer on top for auto-formatting.

**Tech Stack:** pre-commit, pixi, ruff, pytest, GitHub Actions, gh CLI

**Spec:** `docs/superpowers/specs/2026-03-11-dev-practices-design.md`

---

## Chunk 1: pixi.toml updates and pre-commit config

### Task 1: Add `fmt-check` task and `pre-commit` dependency to pixi.toml

**Files:**
- Modify: `pixi.toml:31-36` (dev dependencies) and `pixi.toml:63-65` (tasks)

- [ ] **Step 1: Add `pre-commit` to dev dependencies**

In `pixi.toml`, add `pre-commit` to `[feature.dev.dependencies]`:

```toml
[feature.dev.dependencies]
pytest = ">=8.0"
pytest-cov = ">=5.0"
ruff = ">=0.4"
mypy = ">=1.10"
ipykernel = ">=6.0"
pre-commit = ">=4.0"
```

- [ ] **Step 2: Add `fmt-check` task**

In `pixi.toml`, add below the existing `fmt` task:

```toml
fmt-check = { cmd = "ruff format --check src/ tests/", description = "Check formatting without modifying" }
```

- [ ] **Step 3: Verify pixi environment installs**

Run: `pixi install -e dev`
Expected: Completes successfully, pre-commit now available.

- [ ] **Step 4: Verify `fmt-check` task works**

Run: `pixi run -e dev fmt-check`
Expected: Exits 0 (or exits 1 with formatting diff — either is fine, confirms task works).

- [ ] **Step 5: Commit**

```bash
git add pixi.toml
git commit -m "Add fmt-check task and pre-commit dev dependency"
```

---

### Task 2: Create `.pre-commit-config.yaml`

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write the pre-commit config**

Create `.pre-commit-config.yaml` with four local hooks:

```yaml
repos:
  - repo: local
    hooks:
      - id: pixi-install
        name: pixi install
        entry: pixi install -e dev
        language: system
        pass_filenames: false
        always_run: true

      - id: fmt-check
        name: ruff format check
        entry: pixi run -e dev fmt-check
        language: system
        pass_filenames: false
        types: [python]

      - id: lint
        name: ruff lint
        entry: pixi run -e dev lint
        language: system
        pass_filenames: false
        types: [python]

      - id: test
        name: pytest (unit only)
        entry: pixi run -e dev test -- -m "not integration"
        language: system
        pass_filenames: false
        always_run: true
```

- [ ] **Step 2: Install pre-commit hooks**

Run: `pixi run -e dev pre-commit install`
Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 3: Test pre-commit runs**

Run: `pixi run -e dev pre-commit run --all-files`
Expected: All four hooks run. `pixi install` passes, `fmt-check` passes (or shows diff), `lint` passes, `test` passes.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "Add pre-commit hooks for format, lint, and test"
```

---

## Chunk 2: GitHub Actions CI

### Task 3: Create CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow file**

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          environments: dev

      - name: Check formatting
        run: pixi run -e dev fmt-check

      - name: Lint
        run: pixi run -e dev lint

      - name: Test
        run: pixi run -e dev test
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI workflow for lint, format, and test"
```

---

## Chunk 3: Claude hooks and settings

### Task 4: Update `.claude/settings.json`

**Files:**
- Modify: `.claude/settings.json`

- [ ] **Step 1: Update settings.json**

Replace the full contents of `.claude/settings.json` with:

```json
{
  "permissions": {
    "allow": [
      "Bash(pixi run *)",
      "Bash(pixi install *)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "Bash(git status*)",
      "Bash(git branch*)",
      "Bash(ls *)",
      "Bash(wc *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(pixi run run*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "case \"$CLAUDE_FILE_PATH\" in *.credentials.yml|*/data/raw/*|*/pixi.lock) echo 'BLOCKED: do not edit this file'; exit 2;; esac"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "pixi run -e dev fmt 2>&1 | head -5"
          },
          {
            "type": "command",
            "command": "pixi run -e dev lint 2>&1 | head -30"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add .claude/settings.json
git commit -m "Update Claude hooks: add auto-format, file protection, expanded permissions"
```

---

## Chunk 4: Documentation updates

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add workflow and quality gate sections**

Add the following sections to `CLAUDE.md` after the existing "## Testing" section:

```markdown
## Git Workflow

All work follows issue-branch-PR flow:

1. Create a GitHub issue describing the work
2. Branch from main: `<type>/<issue#>-short-description`
   - Types: `feature`, `fix`, `refactor`, `docs`, `test`, `chore`
   - Example: `feature/12-add-mwbm-fetch`, `fix/13-catalog-validation`
3. Develop on the branch, committing as needed
4. Open PR referencing the issue (e.g., "Closes #12")
5. CI must pass; squash merge after review

## Pre-commit Quality Gate

Before suggesting a commit, always run:

```bash
pixi run fmt && pixi run lint && pixi run test
```

Pre-commit hooks enforce this automatically, but Claude should run these proactively.

## Test Coverage Rule

Every new module in `fetch/`, `aggregate/`, `normalize/`, or `targets/` must have a corresponding `tests/test_<module>.py`. Every PR should maintain or improve test coverage.
```

- [ ] **Step 2: Update the "Environment & Commands" section**

Add `fmt-check` to the development commands block and add the pre-commit install command:

```bash
# Development
pixi run test          # pytest tests/ -v
pixi run lint          # ruff check src/ tests/
pixi run fmt           # ruff format src/ tests/
pixi run fmt-check     # ruff format --check src/ tests/
```

Add after "Install environment" block:

```bash
# Set up pre-commit hooks (once after cloning)
pixi run -e dev pre-commit install
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Add git workflow, quality gate, and test coverage rules to CLAUDE.md"
```

---

### Task 6: Create CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write CONTRIBUTING.md**

```markdown
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
pixi run fmt           # auto-format code
pixi run fmt-check     # check formatting without modifying
pixi run lint          # lint with ruff
pixi run test          # run full test suite
```

## Code Conventions

- Python >=3.11, `from __future__ import annotations` in all modules
- Type hints on all public functions
- Ruff for lint and format (line length 88)
- New modules in `fetch/`, `aggregate/`, `normalize/`, or `targets/` must have a corresponding `tests/test_<module>.py`

## Data Sources

- All source metadata lives in `catalog/sources.yml` — do not hardcode URLs or product names
- When adding a new source, add it to `catalog/sources.yml` first, then write the fetch module
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "Add CONTRIBUTING.md with setup, workflow, and conventions"
```

---

## Chunk 5: Branch protection (deferred until remote exists)

### Task 7: Document branch protection commands

**Files:**
- Modify: `docs/superpowers/specs/2026-03-11-dev-practices-design.md` (add appendix)

- [ ] **Step 1: Add branch protection commands appendix**

Add to the end of the spec file:

```markdown
## Appendix: Branch Protection Setup

Run these commands after the repository is pushed to GitHub. Replace `OWNER/REPO`
with the actual GitHub owner and repository name.

```bash
# Enforce squash merge only
gh api repos/OWNER/REPO \
  -X PATCH \
  -f allow_squash_merge=true \
  -f allow_merge_commit=false \
  -f allow_rebase_merge=false

# Enable branch protection on main
gh api repos/OWNER/REPO/branches/main/protection \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["check"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-03-11-dev-practices-design.md
git commit -m "Add branch protection setup commands to spec appendix"
```
