# Runoff Bias Analysis (`nhf-runoff-bias`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate repo that accumulates each runoff-target member to screened Oregon gages, measures per-member volume bias, relates it to per-HRU covariates, and emits a per-HRU multiplicative correction factor.

**Architecture:** Five idempotent CLI stages (`gages`, `network`, `covariates`, `bias`, `fit`) that read the `nhf-spatial-targets` outputs as data and write Parquet/NetCDF artifacts stamped with their inputs' SHA-256. Analysis is gage-level regression first (PCA + OLS + random forest, leave-one-gage-out), then inversion of the same log-linear factor model through the sparse gage×HRU accumulation matrix.

**Tech Stack:** Python ≥ 3.11, pixi, Cyclopts, xarray/netCDF4, pandas/pyarrow, geopandas/pyogrio/shapely, networkx, scikit-learn, shap, scipy, statsmodels, matplotlib, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-25-runoff-bias-analysis-design.md` (in `nhf-spatial-targets`; Task 1 copies it into the new repo).

## Global Constraints

- New repo at `/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-runoff-bias`, package `nhf_runoff_bias`, CLI `nhf-runoff-bias`, src layout, `from __future__ import annotations` in every module, ruff line length 88.
- Nothing is ever written into `or-spatial-targets/` or `nhf-spatial-targets/`; all outputs go under the configured `run_dir`.
- Every id column and fabric layer name is a config value with Oregon defaults (`hru_id`, `nhm_id`, `hru_segment`, `segment_id`, `to_segment`, `poi_gage_id`; layers `nhru`, `nsegment`, `npoigages`, `domain`). No literal id/layer names inside `network.py` / `accumulate.py`.
- Every derived artifact records `fabric_path`, `fabric_sha256`, `id_col`, and a `source_sha256` map of its inputs (Parquet: pyarrow schema metadata; NetCDF: global attrs).
- Gage tiers: **A** = flow management index ≤ 1, ≥ 10 overlap years, |fabric area / published area − 1| ≤ 0.10, no domain-edge touch (primary); **B** = same with index 0 (strict).
- Monthly gage mean requires `n_valid_days >= min(28, days_in_month)`.
- Bias = `ln(mean(Q_obs) / mean(Q_acc))` over overlap years present in both records; monthly climatological ratios alongside; per member, never per envelope.
- Climate covariate window 1980–2020 (ClimGrid ends 2020); bias overlap window is whatever the two records share (1979–2022 for Oregon).
- Commit via `pixi run git commit` (pre-commit runs ruff through pixi). Never `git add -A`.
- Real-data tests are `@pytest.mark.integration` and excluded from the default `pixi run -e dev test`.

## Review Focus

1. A POI whose `segment_id` is not in `nsegment` (851 fabric POIs vs 7 646 segments; 5 fabric POIs are absent from the gage NC) — `build_network` must record `exclusion_reason="segment_not_in_fabric"`, not KeyError. Test in Task 2.
2. A gage whose upstream set contains an HRU that is NaN in a member for some month (partial-coverage sources) — accumulation must mark that gage-month, and `bias` must drop it rather than sum a short total. Test in Task 4 and Task 8.
3. A gage with all-zero observed flow in the overlap (ephemeral, eastern Oregon) — `ln(0)` must become `exclusion_reason="zero_observed_mean"`, not `-inf` in the fit. Test in Task 8.
4. Duplicate `nat_hru_id` rows in the gfv2-params slope directory (16 814 duplicates = the Oregon count) — terrain join must read the merged file and assert uniqueness. Test in Task 5.
5. Cached weights whose `.meta` fingerprint does not match the fabric — the climate builder must refuse to use them, not silently mis-index. Test in Task 6.

---

### Task 1: Repo scaffold, config, hashing, CLI skeleton

**Files:**
- Create: `pyproject.toml`, `pixi.toml`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`, `.github/workflows/ci.yml`
- Create: `src/nhf_runoff_bias/__init__.py`, `src/nhf_runoff_bias/config.py`, `src/nhf_runoff_bias/provenance.py`, `src/nhf_runoff_bias/cli.py`
- Create: `config.example.yml`, `docs/spec.md` (copy of the spec), `docs/plan.md` (copy of this plan), `docs/literature.md` (copy of `docs/references/runoff-bias-literature.md`)
- Test: `tests/test_config.py`, `tests/test_provenance.py`

**Interfaces:**
- Produces: `config.load_config(path: Path) -> Config`; `Config` dataclass with fields below; `provenance.sha256_of(path: Path) -> str`; `provenance.stamp(inputs: dict[str, Path], fabric: FabricRef) -> dict[str, str]`; `provenance.write_parquet(df, path, meta: dict[str, str])`; `provenance.read_parquet_meta(path) -> dict[str, str]`; `provenance.stage_is_current(outputs: list[Path], inputs: dict[str, Path]) -> bool`; `cli.app` (Cyclopts `App`).

- [ ] **Step 1: Create the repo and copy the design docs**

```bash
mkdir -p /caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-runoff-bias && cd $_
git init -b main
mkdir -p src/nhf_runoff_bias tests docs notebooks .github/workflows
SRC=/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets
cp $SRC/docs/superpowers/specs/2026-09-25-runoff-bias-analysis-design.md docs/spec.md
cp $SRC/docs/superpowers/plans/2026-09-25-runoff-bias-analysis.md docs/plan.md
cp $SRC/docs/references/runoff-bias-literature.md docs/literature.md
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nhf-runoff-bias"
version = "0.1.0"
description = "Bias analysis of NHM runoff calibration targets against gaged streamflow"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6.0", "xarray>=2024.1", "numpy>=1.26", "pandas>=2.0", "pyarrow>=15",
    "geopandas>=0.14", "pyogrio>=0.9", "shapely>=2.0", "netcdf4>=1.6",
    "networkx>=3.2", "scikit-learn>=1.4", "scipy>=1.12", "statsmodels>=0.14",
    "shap>=0.45", "matplotlib>=3.8", "cyclopts>=3.0", "rich>=13.0",
]

[project.scripts]
nhf-runoff-bias = "nhf_runoff_bias.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/nhf_runoff_bias"]

[tool.ruff]
line-length = 88
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requires the real Oregon inputs on caldera"]
```

- [ ] **Step 3: Write `pixi.toml`**

```toml
[workspace]
name = "nhf-runoff-bias"
version = "0.1.0"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11,<3.14"
pyyaml = ">=6.0"
xarray = ">=2024.1"
numpy = ">=1.26"
pandas = ">=2.0"
pyarrow = ">=15"
geopandas = ">=0.14"
pyogrio = ">=0.9"
shapely = ">=2.0"
netcdf4 = ">=1.6"
networkx = ">=3.2"
scikit-learn = ">=1.4"
scipy = ">=1.12"
statsmodels = ">=0.14"
shap = ">=0.45"
matplotlib = ">=3.8"
cyclopts = ">=3.0"
rich = ">=13.0"
proj-data = "*"

[pypi-dependencies]
nhf-runoff-bias = { path = ".", editable = true }

[feature.dev.dependencies]
pytest = ">=8.0"
pytest-xdist = ">=3.5"
ruff = ">=0.6"
pre-commit = ">=3.7"
ipykernel = "*"
nbstripout = "*"

[environments]
dev = ["dev"]

[tasks]
gages = "nhf-runoff-bias gages"
network = "nhf-runoff-bias network"
covariates = "nhf-runoff-bias covariates"
bias = "nhf-runoff-bias bias"
fit = "nhf-runoff-bias fit"

[feature.dev.tasks]
test = "pytest -n auto -m 'not integration'"
test-integration = "pytest -m integration"
lint = "ruff check src/ tests/"
fmt = "ruff format src/ tests/"
fmt-check = "ruff format --check src/ tests/"
```

- [ ] **Step 4: Write `.pre-commit-config.yaml`, `.gitignore`, `README.md`, CI**

`.pre-commit-config.yaml`:
```yaml
default_install_hook_types: [pre-commit]
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.8.1
    hooks:
      - id: nbstripout
        args: [--keep-id]
  - repo: local
    hooks:
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
```

`.gitignore`:
```
.pixi/
__pycache__/
*.pyc
runs/
datastore/
config.yml
.ipynb_checkpoints/
```

`README.md`:
```markdown
# nhf-runoff-bias

Bias analysis of the NHM runoff calibration target (from `nhf-spatial-targets`)
against gaged streamflow, producing a per-HRU multiplicative correction factor.
Design: `docs/spec.md`. Plan: `docs/plan.md`. Literature: `docs/literature.md`.

    pixi install -e dev && pixi run -e dev pre-commit install
    cp config.example.yml config.yml   # edit paths
    pixi run gages -- --config config.yml
    pixi run network -- --config config.yml
    pixi run covariates -- --config config.yml
    pixi run bias -- --config config.yml
    pixi run fit -- --config config.yml
```

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
        with: { environments: dev }
      - run: pixi run -e dev fmt-check
      - run: pixi run -e dev lint
      - run: pixi run -e dev test
```

- [ ] **Step 5: Write `config.example.yml`**

```yaml
run_dir: runs/oregon
datastore: datastore            # downloaded covariate sources (SGMC)

target_nc: /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/targets/runoff_targets.nc
members: [era5_land, gldas_noah_v21_monthly, mwbm_climgrid]

fabric:
  gpkg: /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/fabric/model_layers_9.gpkg
  fabric_json: /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/fabric.json
  layers: {hru: nhru, segment: nsegment, poi: npoigages, domain: domain}
  cols:
    hru_id: hru_id
    nhm_id: nhm_id
    hru_segment: hru_segment
    hru_area_km2: areasqkm
    segment_id: segment_id
    to_segment: to_segment
    poi_gage_id: poi_gage_id
    poi_segment: segment_id
  outlet_value: 0

gages:
  daily_nc: /caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets/gage_data/sf_efc.nc
  fmi_csv: /caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-spatial-targets/gage_data/TableA2_FlowManagementIndex.csv
  min_overlap_years: 10
  area_tolerance: 0.10

covariates:
  gfv2_params_dir: /caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2_param/nhm_params
  climgrid_nc: /caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore/mwbm_climgrid/ClimGrid_WBM.nc
  climgrid_weights_glob: /caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets/weights/mwbm_climgrid_batch*.csv
  climate_window: ["1980-01-01", "2020-12-31"]
  lithology_shp: /caldera/hovenweep/projects/usgs/water/impd/nhgf/gfv2_param/source_data/data_layers/soils_litho/Lithology_exp_Konly_Project.shp
  sgmc_doi: "10.5066/F7WH2N65"
  sgmc_class_field: ROCKTYPE1

fit:
  pca_variance: 0.90
  ridge_lambda: 1.0
  bootstrap: 200
  run_inversion: false
```

- [ ] **Step 6: Write the failing config test**

`tests/test_config.py`:
```python
from pathlib import Path

import yaml

from nhf_runoff_bias.config import Config, load_config


def _minimal(tmp_path: Path) -> Path:
    cfg = {
        "run_dir": str(tmp_path / "run"),
        "datastore": str(tmp_path / "ds"),
        "target_nc": str(tmp_path / "t.nc"),
        "members": ["m1"],
        "fabric": {
            "gpkg": str(tmp_path / "f.gpkg"),
            "fabric_json": str(tmp_path / "fabric.json"),
        },
        "gages": {"daily_nc": str(tmp_path / "g.nc"), "fmi_csv": str(tmp_path / "f.csv")},
        "covariates": {},
        "fit": {},
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_load_config_applies_oregon_defaults(tmp_path):
    cfg = load_config(_minimal(tmp_path))
    assert isinstance(cfg, Config)
    assert cfg.fabric.layers["hru"] == "nhru"
    assert cfg.fabric.cols["to_segment"] == "to_segment"
    assert cfg.fabric.outlet_value == 0
    assert cfg.gages.min_overlap_years == 10
    assert cfg.gages.area_tolerance == 0.10
    assert cfg.fit.pca_variance == 0.90
    assert cfg.run_dir == tmp_path / "run"


def test_load_config_rejects_unknown_top_level_key(tmp_path):
    p = _minimal(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["typo_key"] = 1
    p.write_text(yaml.safe_dump(d))
    import pytest

    with pytest.raises(ValueError, match="typo_key"):
        load_config(p)
```

- [ ] **Step 7: Run it to verify it fails**

Run: `pixi install -e dev && pixi run -e dev pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: nhf_runoff_bias.config`

- [ ] **Step 8: Implement `config.py`**

```python
"""Project configuration: YAML with Oregon defaults for every id/layer name."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_LAYERS = {"hru": "nhru", "segment": "nsegment", "poi": "npoigages", "domain": "domain"}
_DEFAULT_COLS = {
    "hru_id": "hru_id",
    "nhm_id": "nhm_id",
    "hru_segment": "hru_segment",
    "hru_area_km2": "areasqkm",
    "segment_id": "segment_id",
    "to_segment": "to_segment",
    "poi_gage_id": "poi_gage_id",
    "poi_segment": "segment_id",
}
_TOP_KEYS = {"run_dir", "datastore", "target_nc", "members", "fabric", "gages", "covariates", "fit"}


@dataclass
class FabricConfig:
    gpkg: Path
    fabric_json: Path
    layers: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_LAYERS))
    cols: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_COLS))
    outlet_value: int = 0


@dataclass
class GagesConfig:
    daily_nc: Path
    fmi_csv: Path
    min_overlap_years: int = 10
    area_tolerance: float = 0.10


@dataclass
class CovariatesConfig:
    gfv2_params_dir: Path | None = None
    climgrid_nc: Path | None = None
    climgrid_weights_glob: str | None = None
    climate_window: tuple[str, str] = ("1980-01-01", "2020-12-31")
    lithology_shp: Path | None = None
    sgmc_doi: str = "10.5066/F7WH2N65"
    sgmc_class_field: str = "ROCKTYPE1"


@dataclass
class FitConfig:
    pca_variance: float = 0.90
    ridge_lambda: float = 1.0
    bootstrap: int = 200
    run_inversion: bool = False


@dataclass
class Config:
    run_dir: Path
    datastore: Path
    target_nc: Path
    members: list[str]
    fabric: FabricConfig
    gages: GagesConfig
    covariates: CovariatesConfig
    fit: FitConfig


def _paths(d: dict, keys: tuple[str, ...]) -> dict:
    out = dict(d)
    for k in keys:
        if out.get(k) is not None:
            out[k] = Path(out[k])
    return out


def load_config(path: Path) -> Config:
    """Load ``config.yml``; unknown top-level keys are a hard error (typo guard)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {sorted(unknown)}")
    fab = _paths(raw["fabric"], ("gpkg", "fabric_json"))
    fab["layers"] = {**_DEFAULT_LAYERS, **fab.get("layers", {})}
    fab["cols"] = {**_DEFAULT_COLS, **fab.get("cols", {})}
    cov = _paths(raw.get("covariates", {}), ("gfv2_params_dir", "climgrid_nc", "lithology_shp"))
    if "climate_window" in cov:
        cov["climate_window"] = tuple(cov["climate_window"])
    return Config(
        run_dir=Path(raw["run_dir"]),
        datastore=Path(raw["datastore"]),
        target_nc=Path(raw["target_nc"]),
        members=list(raw["members"]),
        fabric=FabricConfig(**fab),
        gages=GagesConfig(**_paths(raw["gages"], ("daily_nc", "fmi_csv"))),
        covariates=CovariatesConfig(**cov),
        fit=FitConfig(**raw.get("fit", {})),
    )
```

- [ ] **Step 9: Write the failing provenance test**

`tests/test_provenance.py`:
```python
import pandas as pd

from nhf_runoff_bias.provenance import (
    read_parquet_meta,
    sha256_of,
    stage_is_current,
    write_parquet,
)


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert sha256_of(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_parquet_roundtrip_carries_meta(tmp_path):
    df = pd.DataFrame({"hru_id": [1, 2], "x": [0.5, 1.5]})
    out = tmp_path / "x.parquet"
    write_parquet(df, out, {"fabric_sha256": "deadbeef", "id_col": "hru_id"})
    meta = read_parquet_meta(out)
    assert meta["fabric_sha256"] == "deadbeef"
    assert meta["id_col"] == "hru_id"
    assert pd.read_parquet(out).equals(df)


def test_stage_is_current_only_when_inputs_unchanged(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("v1")
    out = tmp_path / "out.parquet"
    write_parquet(pd.DataFrame({"a": [1]}), out, {"source_sha256:in": sha256_of(inp)})
    assert stage_is_current([out], {"in": inp})
    inp.write_text("v2")
    assert not stage_is_current([out], {"in": inp})
    assert not stage_is_current([tmp_path / "missing.parquet"], {"in": inp})
```

- [ ] **Step 10: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: nhf_runoff_bias.provenance`

- [ ] **Step 11: Implement `provenance.py`**

```python
"""SHA-256 stamping of artifacts and stage-currency checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

META_PREFIX = "nhf_runoff_bias:"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def fabric_ref(fabric_json: Path) -> dict[str, str]:
    """Read fabric path / sha256 / id_col from nhf-spatial-targets' fabric.json."""
    d = json.loads(Path(fabric_json).read_text())
    return {"fabric_path": d["path"], "fabric_sha256": d["sha256"], "id_col": d["id_col"]}


def stamp(inputs: dict[str, Path], fabric: dict[str, str] | None = None) -> dict[str, str]:
    meta = {f"source_sha256:{k}": sha256_of(p) for k, p in inputs.items()}
    if fabric:
        meta.update(fabric)
    return meta


def write_parquet(df: pd.DataFrame, path: Path, meta: dict[str, str]) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    new = {**existing, **{(META_PREFIX + k).encode(): str(v).encode() for k, v in meta.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table.replace_schema_metadata(new), tmp)
    tmp.replace(path)


def read_parquet_meta(path: Path) -> dict[str, str]:
    md = pq.read_schema(path).metadata or {}
    return {
        k.decode()[len(META_PREFIX):]: v.decode()
        for k, v in md.items()
        if k.decode().startswith(META_PREFIX)
    }


def stage_is_current(outputs: list[Path], inputs: dict[str, Path]) -> bool:
    """True when every output exists and its recorded input hashes match disk."""
    want = {f"source_sha256:{k}": sha256_of(p) for k, p in inputs.items() if p.exists()}
    if len(want) != len(inputs):
        return False
    for out in outputs:
        if not out.exists():
            return False
        have = read_parquet_meta(out) if out.suffix == ".parquet" else _nc_meta(out)
        if any(have.get(k) != v for k, v in want.items()):
            return False
    return True


def _nc_meta(path: Path) -> dict[str, str]:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return {k[len(META_PREFIX):]: str(ds.getncattr(k)) for k in ds.ncattrs() if k.startswith(META_PREFIX)}


def nc_attrs(meta: dict[str, str]) -> dict[str, str]:
    """Global attrs for an xarray Dataset so ``stage_is_current`` can read them back."""
    return {META_PREFIX + k: str(v) for k, v in meta.items()}
```

- [ ] **Step 12: Write `cli.py` skeleton and `__init__.py`**

`src/nhf_runoff_bias/__init__.py`:
```python
"""nhf-runoff-bias: runoff target bias analysis against gaged streamflow."""

from __future__ import annotations

__version__ = "0.1.0"
```

`src/nhf_runoff_bias/cli.py`:
```python
"""Cyclopts CLI: one command per stage, each takes --config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from nhf_runoff_bias import __version__
from nhf_runoff_bias.config import Config, load_config

app = App(name="nhf-runoff-bias", version=__version__)
ConfigArg = Annotated[Path, Parameter(name=["--config", "-c"], help="Path to config.yml")]
log = logging.getLogger("nhf_runoff_bias")


def _setup(config: Path) -> Config:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@app.command
def gages(config: ConfigArg = Path("config.yml")) -> None:
    """Daily gage NC + FMI CSV -> gages_monthly.parquet, gage_meta.parquet."""
    raise NotImplementedError


@app.command
def network(config: ConfigArg = Path("config.yml")) -> None:
    """Fabric -> gage_hrus.parquet, gage_network.parquet."""
    raise NotImplementedError


@app.command
def covariates(config: ConfigArg = Path("config.yml")) -> None:
    """Terrain, climate, geology -> covariates_hru.parquet, covariates_basin.parquet."""
    raise NotImplementedError


@app.command
def bias(config: ConfigArg = Path("config.yml")) -> None:
    """Target + gages + network -> accumulated.nc, bias.parquet."""
    raise NotImplementedError


@app.command
def fit(config: ConfigArg = Path("config.yml")) -> None:
    """Covariates + bias -> fit.json, factor_hru.parquet."""
    raise NotImplementedError


def main() -> None:
    app()
```

- [ ] **Step 13: Run tests, lint, format**

Run: `pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test`
Expected: 5 passed.

- [ ] **Step 14: Commit**

```bash
git add pyproject.toml pixi.toml pixi.lock .pre-commit-config.yaml .gitignore README.md .github/workflows/ci.yml config.example.yml docs/ src/ tests/
pixi run -e dev pre-commit install
pixi run git commit -m "chore: scaffold nhf-runoff-bias (config, provenance, CLI skeleton)"
```

---

### Task 2: Synthetic fabric fixture and `network.py`

**Files:**
- Create: `tests/conftest.py`, `src/nhf_runoff_bias/network.py`
- Test: `tests/test_network.py`

**Interfaces:**
- Consumes: `Config`, `FabricConfig` (Task 1).
- Produces: `network.build_network(fab: FabricConfig) -> Network`; `Network` dataclass with `hru_table: pd.DataFrame` (`hru_id`, `nhm_id`, `hru_segment`, `area_km2`, `touches_domain_edge`), `graph: nx.DiGraph` (segment→downstream), `upstream_hrus(segment_id: int) -> list[int]`, `gage_hrus() -> pd.DataFrame` (`poi_gage_id`, `hru_id`), `gage_network() -> pd.DataFrame` (`poi_gage_id`, `segment_id`, `n_hru`, `fabric_area_km2`, `touches_domain_edge`, `exclusion_reason`).
- Fixture: `synthetic_fabric(tmp_path) -> Path` to a gpkg with layers `nhru`, `nsegment`, `npoigages`, `domain`; `synthetic_fabric_json(tmp_path) -> Path`.

Synthetic topology (all coordinates EPSG:5070, 1 km squares):
```
seg 1 (HRUs 1,2) ─┐
                  ├─> seg 3 (HRUs 4,5) ─> outlet (to_segment=0)
seg 2 (HRU 3)   ─┘
gage G1 on seg 1 -> upstream HRUs {1,2}, area 2 km²
gage G2 on seg 3 -> upstream HRUs {1,2,3,4,5}, area 5 km²
gage G9 on seg 99 (not in nsegment) -> excluded
HRU 5 touches the domain edge.
```

- [ ] **Step 1: Write `tests/conftest.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

from nhf_runoff_bias.config import Config, CovariatesConfig, FabricConfig, FitConfig, GagesConfig

CRS = "EPSG:5070"
HRU_IDS = [1, 2, 3, 4, 5]
HRU_SEG = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
MEMBERS = ["m_a", "m_b"]
# constant member value per HRU, cfs
MEMBER_VALUES = {"m_a": {1: 1.0, 2: 1.0, 3: 2.0, 4: 1.0, 5: 3.0}, "m_b": {1: 2.0, 2: 2.0, 3: 4.0, 4: 2.0, 5: 6.0}}
TRUE_FACTOR = {"G1": 2.0, "G2": 1.5}  # obs = factor * accumulated m_a


@pytest.fixture
def synthetic_fabric(tmp_path: Path) -> Path:
    """5 HRUs (1 km squares in a row), 3 segments, 2 real gages + 1 orphan."""
    polys = [box(i * 1000, 0, (i + 1) * 1000, 1000) for i in range(5)]
    hru = gpd.GeoDataFrame(
        {
            "hru_id": HRU_IDS,
            "nhm_id": [100 + i for i in HRU_IDS],
            "hru_segment": [HRU_SEG[i] for i in HRU_IDS],
            "areasqkm": [1.0] * 5,
        },
        geometry=polys,
        crs=CRS,
    )
    seg = gpd.GeoDataFrame(
        {"segment_id": [1, 2, 3], "to_segment": [3, 3, 0]},
        geometry=[Point(500, 500).buffer(1), Point(2500, 500).buffer(1), Point(4500, 500).buffer(1)],
        crs=CRS,
    )
    poi = gpd.GeoDataFrame(
        {"poi_gage_id": ["G1", "G2", "G9"], "segment_id": [1, 3, 99]},
        geometry=[Point(1500, 500), Point(4900, 500), Point(0, 0)],
        crs=CRS,
    )
    # domain covers HRUs 1-4 fully; HRU 5's east edge lies on the boundary
    domain = gpd.GeoDataFrame({"ohm_aoi": ["x"]}, geometry=[box(-10, -10, 5000, 1010)], crs=CRS)
    p = tmp_path / "fabric.gpkg"
    hru.to_file(p, layer="nhru", driver="GPKG")
    seg.to_file(p, layer="nsegment", driver="GPKG")
    poi.to_file(p, layer="npoigages", driver="GPKG")
    domain.to_file(p, layer="domain", driver="GPKG")
    return p


@pytest.fixture
def synthetic_fabric_json(tmp_path: Path, synthetic_fabric: Path) -> Path:
    p = tmp_path / "fabric.json"
    p.write_text(json.dumps({"path": str(synthetic_fabric), "sha256": "f" * 64, "id_col": "hru_id"}))
    return p


@pytest.fixture
def synthetic_target_nc(tmp_path: Path) -> Path:
    """24 monthly steps (2001-2002), 2 members, constant per-HRU cfs."""
    time = pd.date_range("2001-01-01", periods=24, freq="MS")
    ds = xr.Dataset(coords={"time": time, "hru_id": np.array(HRU_IDS, dtype="int32")})
    for m in MEMBERS:
        vals = np.array([[MEMBER_VALUES[m][h] for h in HRU_IDS]] * 24, dtype="float32")
        ds[m] = (("time", "hru_id"), vals)
    ds["m_a"][0, 2] = np.nan  # HRU 3 missing in m_a for 2001-01 (review focus 2)
    ds.attrs["fabric_sha256"] = "f" * 64
    ds.attrs["member_keys"] = ",".join(MEMBERS)
    p = tmp_path / "runoff_targets.nc"
    ds.to_netcdf(p)
    return p


@pytest.fixture
def synthetic_gage_nc(tmp_path: Path) -> Path:
    """Daily 2001-2002 discharge = TRUE_FACTOR * accumulated m_a; G9 empty."""
    time = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    acc = {"G1": 2.0, "G2": 8.0}  # sum of m_a over upstream HRUs
    q = np.full((3, len(time)), np.nan)
    q[0, :] = TRUE_FACTOR["G1"] * acc["G1"]
    q[1, :] = TRUE_FACTOR["G2"] * acc["G2"]
    q[1, :5] = np.nan  # Jan 2002... make Jan 2001 incomplete for G2 (5 missing days -> < 28 valid)
    ds = xr.Dataset(
        {
            "discharge": (("poi_gage_id", "time"), q, {"units": "ft3 s-1"}),
            "poi_name": ("poi_gage_id", ["gage one", "gage two", "orphan"]),
            "poi_agency": ("poi_gage_id", ["USGS", "USGS", "USGS"]),
            "latitude": ("poi_gage_id", [44.0, 44.1, 44.2]),
            "longitude": ("poi_gage_id", [-122.0, -122.1, -122.2]),
            "drainage_area": ("poi_gage_id", [np.nan] * 3),
        },
        coords={"poi_gage_id": ["G1", "G2", "G9"], "time": time},
    )
    p = tmp_path / "sf.nc"
    ds.to_netcdf(p)
    return p


@pytest.fixture
def synthetic_fmi_csv(tmp_path: Path) -> Path:
    p = tmp_path / "fmi.csv"
    # area_mi2: G1 true 2 km2 = 0.772 mi2; G2 5 km2 = 1.931 mi2
    pd.DataFrame(
        {
            "gageid": ["G1", "G2"],
            "name": ["gage one", "gage two"],
            "area_mi2": [0.772, 1.931],
            "storage_index": [0, 1],
            "use_index": [0, 1],
            "flow_management_index": [0, 1],
        }
    ).to_csv(p, index=False)
    return p


@pytest.fixture
def synthetic_config(tmp_path, synthetic_fabric, synthetic_fabric_json, synthetic_target_nc, synthetic_gage_nc, synthetic_fmi_csv) -> Config:
    return Config(
        run_dir=tmp_path / "run",
        datastore=tmp_path / "ds",
        target_nc=synthetic_target_nc,
        members=MEMBERS,
        fabric=FabricConfig(gpkg=synthetic_fabric, fabric_json=synthetic_fabric_json),
        gages=GagesConfig(daily_nc=synthetic_gage_nc, fmi_csv=synthetic_fmi_csv, min_overlap_years=1),
        covariates=CovariatesConfig(),
        fit=FitConfig(bootstrap=20),
    )
```

- [ ] **Step 2: Write the failing network tests**

`tests/test_network.py`:
```python
import pytest

from nhf_runoff_bias.network import build_network


def test_upstream_sets_and_nesting(synthetic_config):
    net = build_network(synthetic_config.fabric)
    assert sorted(net.upstream_hrus(1)) == [1, 2]
    assert sorted(net.upstream_hrus(2)) == [3]
    assert sorted(net.upstream_hrus(3)) == [1, 2, 3, 4, 5]


def test_gage_tables(synthetic_config):
    net = build_network(synthetic_config.fabric)
    gh = net.gage_hrus()
    assert set(gh.columns) == {"poi_gage_id", "hru_id"}
    assert sorted(gh.loc[gh.poi_gage_id == "G1", "hru_id"]) == [1, 2]
    gn = net.gage_network().set_index("poi_gage_id")
    assert gn.loc["G1", "fabric_area_km2"] == pytest.approx(2.0)
    assert gn.loc["G2", "n_hru"] == 5
    assert gn.loc["G1", "exclusion_reason"] == ""
    assert gn.loc["G9", "exclusion_reason"] == "segment_not_in_fabric"
    assert "G9" not in set(gh.poi_gage_id)


def test_domain_edge_flag(synthetic_config):
    net = build_network(synthetic_config.fabric)
    gn = net.gage_network().set_index("poi_gage_id")
    assert not gn.loc["G1", "touches_domain_edge"]
    assert gn.loc["G2", "touches_domain_edge"]  # HRU 5 lies on the domain boundary


def test_cycle_is_fatal(synthetic_config, tmp_path):
    import geopandas as gpd

    p = synthetic_config.fabric.gpkg
    seg = gpd.read_file(p, layer="nsegment")
    seg.loc[seg.segment_id == 3, "to_segment"] = 1  # 1 -> 3 -> 1
    seg.to_file(p, layer="nsegment", driver="GPKG")
    with pytest.raises(ValueError, match="cycle"):
        build_network(synthetic_config.fabric)


def test_unknown_downstream_is_fatal(synthetic_config):
    import geopandas as gpd

    p = synthetic_config.fabric.gpkg
    seg = gpd.read_file(p, layer="nsegment")
    seg.loc[seg.segment_id == 2, "to_segment"] = 42
    seg.to_file(p, layer="nsegment", driver="GPKG")
    with pytest.raises(ValueError, match="to_segment"):
        build_network(synthetic_config.fabric)
```

- [ ] **Step 3: Run to verify failure**

Run: `pixi run -e dev pytest tests/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError: nhf_runoff_bias.network`

- [ ] **Step 4: Implement `network.py`**

```python
"""Segment graph + HRU attachment: upstream HRU set per POI gage."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import geopandas as gpd
import networkx as nx
import pandas as pd

from nhf_runoff_bias.config import FabricConfig


@dataclass
class Network:
    hru_table: pd.DataFrame
    poi_table: pd.DataFrame
    graph: nx.DiGraph
    _seg_to_hrus: dict[int, list[int]] = field(default_factory=dict)
    _cols: dict[str, str] = field(default_factory=dict)

    def upstream_hrus(self, segment_id: int) -> list[int]:
        return list(self._upstream(int(segment_id)))

    @lru_cache(maxsize=None)  # noqa: B019 - Network is immutable after build
    def _upstream(self, segment_id: int) -> tuple[int, ...]:
        segs = nx.ancestors(self.graph, segment_id) | {segment_id}
        return tuple(sorted(h for s in segs for h in self._seg_to_hrus.get(s, ())))

    def gage_hrus(self) -> pd.DataFrame:
        rows = [
            (g, h)
            for g, s, ok in self.poi_table[["poi_gage_id", "segment_id", "in_fabric"]].itertuples(index=False)
            if ok
            for h in self.upstream_hrus(s)
        ]
        return pd.DataFrame(rows, columns=["poi_gage_id", "hru_id"])

    def gage_network(self) -> pd.DataFrame:
        area = self.hru_table.set_index("hru_id")["area_km2"]
        edge = self.hru_table.set_index("hru_id")["touches_domain_edge"]
        out = []
        for g, s, ok in self.poi_table[["poi_gage_id", "segment_id", "in_fabric"]].itertuples(index=False):
            if not ok:
                out.append((g, s, 0, float("nan"), False, "segment_not_in_fabric"))
                continue
            hrus = self.upstream_hrus(s)
            out.append((g, s, len(hrus), float(area.loc[hrus].sum()), bool(edge.loc[hrus].any()), ""))
        return pd.DataFrame(
            out,
            columns=["poi_gage_id", "segment_id", "n_hru", "fabric_area_km2", "touches_domain_edge", "exclusion_reason"],
        )


def build_network(fab: FabricConfig) -> Network:
    c, L = fab.cols, fab.layers
    hru = gpd.read_file(fab.gpkg, layer=L["hru"])
    seg = gpd.read_file(fab.gpkg, layer=L["segment"])
    poi = gpd.read_file(fab.gpkg, layer=L["poi"])
    domain = gpd.read_file(fab.gpkg, layer=L["domain"])

    seg_ids = set(seg[c["segment_id"]].astype(int))
    bad_to = set(seg[c["to_segment"]].astype(int)) - seg_ids - {fab.outlet_value}
    if bad_to:
        raise ValueError(f"to_segment values not in segment table: {sorted(bad_to)[:10]}")
    bad_hru = set(hru[c["hru_segment"]].astype(int)) - seg_ids
    if bad_hru:
        raise ValueError(f"hru_segment values not in segment table: {sorted(bad_hru)[:10]}")

    g = nx.DiGraph()
    g.add_nodes_from(seg_ids)
    g.add_edges_from(
        (int(a), int(b))
        for a, b in seg[[c["segment_id"], c["to_segment"]]].itertuples(index=False)
        if int(b) != fab.outlet_value
    )
    if not nx.is_directed_acyclic_graph(g):
        cyc = nx.find_cycle(g)
        raise ValueError(f"segment network contains a cycle: {cyc}")

    boundary = domain.geometry.unary_union.boundary
    hru_table = pd.DataFrame(
        {
            "hru_id": hru[c["hru_id"]].astype(int),
            "nhm_id": hru[c["nhm_id"]].astype(int),
            "hru_segment": hru[c["hru_segment"]].astype(int),
            "area_km2": hru[c["hru_area_km2"]].astype(float),
            "touches_domain_edge": hru.geometry.intersects(boundary).to_numpy(),
        }
    )
    seg_to_hrus = hru_table.groupby("hru_segment")["hru_id"].apply(list).to_dict()

    poi_table = pd.DataFrame(
        {
            "poi_gage_id": poi[c["poi_gage_id"]].astype(str),
            "segment_id": poi[c["poi_segment"]].astype(int),
        }
    )
    poi_table["in_fabric"] = poi_table["segment_id"].isin(seg_ids)
    return Network(hru_table=hru_table, poi_table=poi_table, graph=g, _seg_to_hrus=seg_to_hrus, _cols=c)
```

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev pytest tests/test_network.py -v`
Expected: 5 passed. (If `lru_cache` on a dataclass method trips ruff B019, keep the `noqa`.)

- [ ] **Step 6: Wire the CLI `network` command**

Replace the `network` body in `cli.py`:
```python
@app.command
def network(config: ConfigArg = Path("config.yml")) -> None:
    """Fabric -> gage_hrus.parquet, gage_network.parquet."""
    from nhf_runoff_bias.network import build_network
    from nhf_runoff_bias.provenance import fabric_ref, stage_is_current, stamp, write_parquet

    cfg = _setup(config)
    outs = [cfg.run_dir / "gage_hrus.parquet", cfg.run_dir / "gage_network.parquet"]
    inputs = {"fabric_gpkg": cfg.fabric.gpkg}
    if stage_is_current(outs, inputs):
        log.info("network: up to date")
        return
    net = build_network(cfg.fabric)
    meta = stamp(inputs, fabric_ref(cfg.fabric.fabric_json))
    write_parquet(net.gage_hrus(), outs[0], meta)
    write_parquet(net.gage_network(), outs[1], meta)
    log.info("network: %d gages, %d in fabric", len(net.poi_table), int(net.poi_table.in_fabric.sum()))
```

- [ ] **Step 7: Add a CLI test and commit**

Append to `tests/test_network.py`:
```python
def test_cli_network_writes_stamped_outputs(synthetic_config, tmp_path):
    import yaml
    from cyclopts.testing import invoke  # cyclopts >= 3.11; else call cli.network(config=...) directly

    from nhf_runoff_bias import cli
    from nhf_runoff_bias.provenance import read_parquet_meta

    cli.network(config=_write_cfg(synthetic_config, tmp_path))
    meta = read_parquet_meta(synthetic_config.run_dir / "gage_hrus.parquet")
    assert meta["fabric_sha256"] == "f" * 64
    assert "source_sha256:fabric_gpkg" in meta


def _write_cfg(cfg, tmp_path):
    import yaml
    from dataclasses import asdict

    d = asdict(cfg)
    def _s(o):
        if isinstance(o, dict):
            return {k: _s(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_s(v) for v in o]
        return str(o) if hasattr(o, "__fspath__") else o
    p = tmp_path / "config.yml"
    p.write_text(yaml.safe_dump(_s(d)))
    return p
```
(Delete the `cyclopts.testing` import line if that module is absent; the test calls the function directly.)

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add src/nhf_runoff_bias/network.py src/nhf_runoff_bias/cli.py tests/conftest.py tests/test_network.py
pixi run git commit -m "feat: segment network + upstream HRU sets per gage"
```

---

### Task 3: `gages.py` — monthly product, exclusions, drainage area, tiers input

**Files:**
- Create: `src/nhf_runoff_bias/gages.py`
- Modify: `src/nhf_runoff_bias/cli.py` (gages command)
- Test: `tests/test_gages.py`

**Interfaces:**
- Consumes: `GagesConfig`.
- Produces: `gages.load_gages(cfg: GagesConfig) -> tuple[pd.DataFrame, pd.DataFrame]` returning `gages_monthly` (`poi_gage_id`, `time` [month start], `q_cfs`, `n_valid_days`, `complete: bool`) and `gage_meta` (`poi_gage_id`, `name`, `agency`, `lat`, `lon`, `published_area_km2`, `storage_index`, `use_index`, `flow_management_index`, `exclusion_reason`). Excluded gages keep a `gage_meta` row and have no `gages_monthly` rows.
- Constants: `MI2_TO_KM2 = 2.589988110336`.

Exclusion rules (in order; first match wins): `placeholder_id` (id lowercased is `gages`), `derived_series` (id contains `-`), `negative_discharge` (any finite value < 0), `no_data` (zero valid days). Washington Ecology IDs like `32A080` are kept.

- [ ] **Step 1: Write the failing tests**

`tests/test_gages.py`:
```python
import numpy as np
import pandas as pd
import xarray as xr

from nhf_runoff_bias.gages import MI2_TO_KM2, load_gages, monthly_means


def test_monthly_completeness_rule():
    t = pd.date_range("2001-01-01", "2001-03-31", freq="D")
    q = np.ones(len(t))
    q[:5] = np.nan  # Jan has 26 valid -> incomplete
    q[31 + 3] = np.nan  # Feb has 27 valid -> incomplete (needs all 28)
    m = monthly_means(pd.Series(q, index=t))
    assert list(m.index) == list(pd.date_range("2001-01-01", periods=3, freq="MS"))
    assert m.loc["2001-01-01", "n_valid_days"] == 26 and not m.loc["2001-01-01", "complete"]
    assert m.loc["2001-02-01", "n_valid_days"] == 27 and not m.loc["2001-02-01", "complete"]
    assert m.loc["2001-03-01", "complete"] and m.loc["2001-03-01", "q_cfs"] == 1.0


def test_load_gages_applies_exclusions_and_area(synthetic_config, tmp_path):
    # add the pathological ids to the synthetic gage file
    ds = xr.open_dataset(synthetic_config.gages.daily_nc).load()
    extra_ids = ["13233300-VALO", "gages", "32A080"]
    n = len(ds.time)
    q = np.vstack([np.full(n, -5.0), np.full(n, np.nan), np.full(n, 3.0)])
    ex = xr.Dataset(
        {
            "discharge": (("poi_gage_id", "time"), q),
            "poi_name": ("poi_gage_id", ["derived", "placeholder", "ecology"]),
            "poi_agency": ("poi_gage_id", ["OWRD", "", "ECY"]),
            "latitude": ("poi_gage_id", [44.0] * 3),
            "longitude": ("poi_gage_id", [-122.0] * 3),
            "drainage_area": ("poi_gage_id", [np.nan] * 3),
        },
        coords={"poi_gage_id": extra_ids, "time": ds.time},
    )
    merged = xr.concat([ds, ex], dim="poi_gage_id")
    p = tmp_path / "sf2.nc"
    merged.to_netcdf(p)
    synthetic_config.gages.daily_nc = p

    monthly, meta = load_gages(synthetic_config.gages)
    meta = meta.set_index("poi_gage_id")
    assert meta.loc["13233300-VALO", "exclusion_reason"] == "derived_series"
    assert meta.loc["gages", "exclusion_reason"] == "placeholder_id"
    assert meta.loc["G9", "exclusion_reason"] == "no_data"
    assert meta.loc["32A080", "exclusion_reason"] == ""
    assert meta.loc["G1", "published_area_km2"] == 0.772 * MI2_TO_KM2
    assert np.isnan(meta.loc["32A080", "published_area_km2"])
    assert meta.loc["G2", "flow_management_index"] == 1
    assert set(monthly.poi_gage_id) == {"G1", "G2", "32A080"}
    g2 = monthly[monthly.poi_gage_id == "G2"].set_index("time")
    assert not g2.loc["2001-01-01", "complete"]
    assert g2.loc["2001-02-01", "q_cfs"] == 12.0


def test_negative_discharge_is_excluded(synthetic_config, tmp_path):
    ds = xr.open_dataset(synthetic_config.gages.daily_nc).load()
    ds["discharge"][0, 10] = -1.0
    p = tmp_path / "sf3.nc"
    ds.to_netcdf(p)
    synthetic_config.gages.daily_nc = p
    _, meta = load_gages(synthetic_config.gages)
    assert meta.set_index("poi_gage_id").loc["G1", "exclusion_reason"] == "negative_discharge"
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/test_gages.py -v`
Expected: FAIL with `ModuleNotFoundError: nhf_runoff_bias.gages`

- [ ] **Step 3: Implement `gages.py`**

```python
"""Gage observations: daily NC + flow-management CSV -> monthly product + metadata."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from nhf_runoff_bias.config import GagesConfig

log = logging.getLogger(__name__)
MI2_TO_KM2 = 2.589988110336


def monthly_means(daily: pd.Series) -> pd.DataFrame:
    """Monthly mean of a daily series; complete iff n_valid >= min(28, days_in_month)."""
    grp = daily.groupby(pd.Grouper(freq="MS"))
    out = pd.DataFrame({"q_cfs": grp.mean(), "n_valid_days": grp.count()})
    out["complete"] = out["n_valid_days"] >= np.minimum(28, out.index.days_in_month)
    out.loc[~out["complete"], "q_cfs"] = np.nan
    return out


def _exclusion_reason(gid: str, q: np.ndarray) -> str:
    if gid.lower() == "gages":
        return "placeholder_id"
    if "-" in gid:
        return "derived_series"
    finite = q[np.isfinite(q)]
    if finite.size == 0:
        return "no_data"
    if (finite < 0).any():
        return "negative_discharge"
    return ""


def load_gages(cfg: GagesConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(cfg.daily_nc)
    ids = ds["poi_gage_id"].values.astype(str)
    q_all = ds["discharge"].load().values
    time = pd.DatetimeIndex(ds["time"].values)

    fmi = pd.read_csv(cfg.fmi_csv, dtype={"gageid": str}).set_index("gageid")
    fmi = fmi[~fmi.index.duplicated()]

    meta_rows, monthly_parts = [], []
    for i, gid in enumerate(ids):
        q = q_all[i].astype(float)
        reason = _exclusion_reason(gid, q)
        row = {
            "poi_gage_id": gid,
            "name": str(ds["poi_name"].values[i]),
            "agency": str(ds["poi_agency"].values[i]),
            "lat": float(ds["latitude"].values[i]),
            "lon": float(ds["longitude"].values[i]),
            "published_area_km2": float(fmi["area_mi2"].get(gid, np.nan)) * MI2_TO_KM2,
            "storage_index": fmi["storage_index"].get(gid, np.nan),
            "use_index": fmi["use_index"].get(gid, np.nan),
            "flow_management_index": fmi["flow_management_index"].get(gid, np.nan),
            "exclusion_reason": reason,
        }
        meta_rows.append(row)
        if reason:
            continue
        m = monthly_means(pd.Series(q, index=time))
        m.insert(0, "poi_gage_id", gid)
        monthly_parts.append(m.rename_axis("time").reset_index())

    meta = pd.DataFrame(meta_rows)
    for c in ("storage_index", "use_index", "flow_management_index"):
        meta[c] = pd.to_numeric(meta[c], errors="coerce").astype("Int64")
    monthly = pd.concat(monthly_parts, ignore_index=True)
    log.info("gages: %d ids, %d usable, %d monthly rows", len(ids), meta.exclusion_reason.eq("").sum(), len(monthly))
    return monthly, meta
```

- [ ] **Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_gages.py -v`
Expected: 3 passed.

- [ ] **Step 5: Wire the CLI `gages` command**

```python
@app.command
def gages(config: ConfigArg = Path("config.yml")) -> None:
    """Daily gage NC + FMI CSV -> gages_monthly.parquet, gage_meta.parquet."""
    from nhf_runoff_bias.gages import load_gages
    from nhf_runoff_bias.provenance import stage_is_current, stamp, write_parquet

    cfg = _setup(config)
    outs = [cfg.run_dir / "gages_monthly.parquet", cfg.run_dir / "gage_meta.parquet"]
    inputs = {"gage_daily_nc": cfg.gages.daily_nc, "fmi_csv": cfg.gages.fmi_csv}
    if stage_is_current(outs, inputs):
        log.info("gages: up to date")
        return
    monthly, meta = load_gages(cfg.gages)
    m = stamp(inputs)
    write_parquet(monthly, outs[0], m)
    write_parquet(meta, outs[1], m)
```

- [ ] **Step 6: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add src/nhf_runoff_bias/gages.py src/nhf_runoff_bias/cli.py tests/test_gages.py
pixi run git commit -m "feat: gage monthly product with exclusions and published area"
```

---

### Task 4: `accumulate.py` — member sums per gage, NaN-aware

**Files:**
- Create: `src/nhf_runoff_bias/accumulate.py`
- Test: `tests/test_accumulate.py`

**Interfaces:**
- Consumes: `gage_hrus` DataFrame (Task 2), target NC path, member list.
- Produces: `accumulate.accumulate_members(target_nc: Path, gage_hrus: pd.DataFrame, members: list[str], id_col: str = "hru_id") -> xr.Dataset` with dims `(poi_gage_id, member, time)`, variables `q_acc_cfs` (float64) and `n_nan_hru` (int32); `accumulate.accumulation_matrix(gage_hrus, hru_ids: np.ndarray) -> scipy.sparse.csr_matrix` (gages × HRUs, 1 where upstream), also returning the gage order (used again by Task 10).

- [ ] **Step 1: Write the failing tests**

`tests/test_accumulate.py`:
```python
import numpy as np

from nhf_runoff_bias.accumulate import accumulate_members, accumulation_matrix
from nhf_runoff_bias.network import build_network


def test_accumulation_matrix_shape_and_rows(synthetic_config):
    net = build_network(synthetic_config.fabric)
    A, gages = accumulation_matrix(net.gage_hrus(), np.array([1, 2, 3, 4, 5]))
    assert A.shape == (2, 5) and list(gages) == ["G1", "G2"]
    assert A[0].toarray().tolist() == [[1, 1, 0, 0, 0]]
    assert A[1].toarray().tolist() == [[1, 1, 1, 1, 1]]


def test_accumulate_sums_and_marks_nan(synthetic_config):
    net = build_network(synthetic_config.fabric)
    acc = accumulate_members(synthetic_config.target_nc, net.gage_hrus(), synthetic_config.members)
    q = acc["q_acc_cfs"]
    assert q.sel(poi_gage_id="G1", member="m_a").isel(time=1).item() == 2.0
    assert q.sel(poi_gage_id="G2", member="m_b").isel(time=1).item() == 16.0
    # 2001-01 has HRU 3 NaN in m_a: G2 is marked, G1 is not
    assert acc["n_nan_hru"].sel(poi_gage_id="G2", member="m_a").isel(time=0).item() == 1
    assert acc["n_nan_hru"].sel(poi_gage_id="G1", member="m_a").isel(time=0).item() == 0
    assert np.isnan(q.sel(poi_gage_id="G2", member="m_a").isel(time=0).item())


def test_fabric_hash_mismatch_is_fatal(synthetic_config):
    import pytest

    net = build_network(synthetic_config.fabric)
    with pytest.raises(ValueError, match="fabric_sha256"):
        accumulate_members(synthetic_config.target_nc, net.gage_hrus(), synthetic_config.members, expected_fabric_sha256="0" * 64)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/test_accumulate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `accumulate.py`**

```python
"""Sum target members over each gage's upstream HRU set (cfs is a flow: plain sum)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

log = logging.getLogger(__name__)


def accumulation_matrix(gage_hrus: pd.DataFrame, hru_ids: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sparse (n_gage × n_hru) 0/1 matrix; rows in sorted gage order."""
    gages = np.array(sorted(gage_hrus["poi_gage_id"].unique()))
    g_idx = pd.Index(gages).get_indexer(gage_hrus["poi_gage_id"])
    h_idx = pd.Index(hru_ids).get_indexer(gage_hrus["hru_id"])
    if (h_idx < 0).any():
        missing = gage_hrus["hru_id"][h_idx < 0].unique()[:10]
        raise ValueError(f"gage_hrus references hru_id not in target: {missing}")
    A = sp.csr_matrix((np.ones(len(g_idx)), (g_idx, h_idx)), shape=(len(gages), len(hru_ids)))
    return A, gages


def accumulate_members(
    target_nc: Path,
    gage_hrus: pd.DataFrame,
    members: list[str],
    id_col: str = "hru_id",
    expected_fabric_sha256: str | None = None,
) -> xr.Dataset:
    ds = xr.open_dataset(target_nc)
    if expected_fabric_sha256 and ds.attrs.get("fabric_sha256") != expected_fabric_sha256:
        raise ValueError(
            f"target fabric_sha256 {ds.attrs.get('fabric_sha256')!r} != configured fabric {expected_fabric_sha256!r}"
        )
    hru_ids = ds[id_col].values
    A, gages = accumulation_matrix(gage_hrus, hru_ids)
    time = ds["time"].values
    q = np.full((len(gages), len(members), len(time)), np.nan)
    n_nan = np.zeros((len(gages), len(members), len(time)), dtype="int32")
    for mi, m in enumerate(members):
        vals = ds[m].transpose("time", id_col).values.astype("float64")  # (time, hru)
        isnan = np.isnan(vals)
        n_nan[:, mi, :] = np.asarray(A @ isnan.astype("int32").T).T  # (gage, time)
        summed = np.asarray(A @ np.nan_to_num(vals).T).T  # (gage, time)
        summed[n_nan[:, mi, :] > 0] = np.nan
        q[:, mi, :] = summed
        log.info("accumulate: member %s, %d gages", m, len(gages))
    return xr.Dataset(
        {
            "q_acc_cfs": (("poi_gage_id", "member", "time"), q, {"units": "ft3 s-1"}),
            "n_nan_hru": (("poi_gage_id", "member", "time"), n_nan),
        },
        coords={"poi_gage_id": gages, "member": members, "time": time},
    )
```
- [ ] **Step 4: Run tests, commit**

Run: `pixi run -e dev pytest tests/test_accumulate.py -v` → 3 passed.
```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_runoff_bias/accumulate.py tests/test_accumulate.py
pixi run git commit -m "feat: NaN-aware accumulation of members to gages"
```

---

### Task 5: Terrain and soils covariates from gfv2-params

**Files:**
- Create: `src/nhf_runoff_bias/covariates/__init__.py`, `src/nhf_runoff_bias/covariates/terrain.py`
- Test: `tests/test_cov_terrain.py`

**Interfaces:**
- Produces: `terrain.build_terrain(params_dir: Path, hru_table: pd.DataFrame) -> pd.DataFrame` on `hru_id` with `elev_m`, `slope_deg`, `northness`, `eastness`, `soil_type`, `soil_moist_max`. Reads elevation from `params_dir/elevation/*.csv` (concatenated, `nat_hru_id`,`mean`), slope from `params_dir/nhm_params_merged/nhm_slope_params.csv`, aspect from `.../nhm_aspect_params.csv` (`mean` degrees, arithmetic — used only via cos/sin), soils from `.../nhm_soils_params.csv` (`soils`), `soil_moist_max` from `.../nhm_soil_moist_max_params.csv`. Join key: `hru_table.nhm_id == nat_hru_id`.
- Shared helper in `covariates/__init__.py`: `basin_means(cov_hru: pd.DataFrame, gage_hrus: pd.DataFrame, area: pd.Series) -> pd.DataFrame` (area-weighted mean per gage of every numeric column; categorical `soil_type`/`litho_class` become the area-majority class).

- [ ] **Step 1: Write the failing tests**

`tests/test_cov_terrain.py`:
```python
import numpy as np
import pandas as pd
import pytest

from nhf_runoff_bias.covariates import basin_means
from nhf_runoff_bias.covariates.terrain import build_terrain


def _params_dir(tmp_path, dup_slope=False):
    d = tmp_path / "nhm_params"
    (d / "elevation").mkdir(parents=True)
    (d / "nhm_params_merged").mkdir()
    ids = [101, 102, 103, 104, 105]
    pd.DataFrame({"nat_hru_id": ids[:3], "mean": [100.0, 200, 300]}).to_csv(d / "elevation/base_16.csv", index=False)
    pd.DataFrame({"nat_hru_id": ids[3:], "mean": [400.0, 500]}).to_csv(d / "elevation/base_17.csv", index=False)
    slope = pd.DataFrame({"nat_hru_id": ids, "mean": [1.0, 2, 3, 4, 5]})
    if dup_slope:
        slope = pd.concat([slope, slope.iloc[:1]])
    slope.to_csv(d / "nhm_params_merged/nhm_slope_params.csv", index=False)
    pd.DataFrame({"nat_hru_id": ids, "mean": [0.0, 90, 180, 270, 45]}).to_csv(d / "nhm_params_merged/nhm_aspect_params.csv", index=False)
    pd.DataFrame({"nat_hru_id": ids, "soils": [1, 1, 2, 3, 3]}).to_csv(d / "nhm_params_merged/nhm_soils_params.csv", index=False)
    pd.DataFrame({"nat_hru_id": ids, "soil_moist_max": [2.0, 3, 4, 5, 6]}).to_csv(d / "nhm_params_merged/nhm_soil_moist_max_params.csv", index=False)
    return d


def test_build_terrain_joins_on_nhm_id(tmp_path):
    hru = pd.DataFrame({"hru_id": [1, 2, 3, 4, 5], "nhm_id": [101, 102, 103, 104, 105]})
    t = build_terrain(_params_dir(tmp_path), hru).set_index("hru_id")
    assert t.loc[4, "elev_m"] == 400.0
    assert t.loc[2, "northness"] == pytest.approx(0.0, abs=1e-12)  # aspect 90 -> cos=0
    assert t.loc[2, "eastness"] == pytest.approx(1.0)
    assert t.loc[5, "soil_type"] == 3
    assert list(t.columns) == ["elev_m", "slope_deg", "northness", "eastness", "soil_type", "soil_moist_max"]


def test_duplicate_nat_hru_id_is_fatal(tmp_path):
    hru = pd.DataFrame({"hru_id": [1], "nhm_id": [101]})
    with pytest.raises(ValueError, match="duplicate"):
        build_terrain(_params_dir(tmp_path, dup_slope=True), hru)


def test_missing_nhm_id_is_fatal(tmp_path):
    hru = pd.DataFrame({"hru_id": [1, 9], "nhm_id": [101, 999]})
    with pytest.raises(ValueError, match="999"):
        build_terrain(_params_dir(tmp_path), hru)


def test_basin_means_area_weighted_and_majority():
    cov = pd.DataFrame({"hru_id": [1, 2, 3], "elev_m": [100.0, 300.0, 500.0], "soil_type": [1, 2, 2]})
    gh = pd.DataFrame({"poi_gage_id": ["G"] * 3, "hru_id": [1, 2, 3]})
    area = pd.Series([3.0, 1.0, 1.0], index=[1, 2, 3])
    b = basin_means(cov, gh, area).set_index("poi_gage_id")
    assert b.loc["G", "elev_m"] == pytest.approx((300 + 300 + 500) / 5)
    assert b.loc["G", "soil_type"] == 1  # area 3 vs 2
```

- [ ] **Step 2: Run to verify failure** — `pixi run -e dev pytest tests/test_cov_terrain.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/nhf_runoff_bias/covariates/__init__.py`:
```python
"""Per-HRU covariate builders and the basin-mean reducer."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ("soil_type", "litho_class")


def basin_means(cov_hru: pd.DataFrame, gage_hrus: pd.DataFrame, area_km2: pd.Series) -> pd.DataFrame:
    """Area-weighted mean per gage for numeric columns; area-majority for categoricals."""
    df = gage_hrus.merge(cov_hru, on="hru_id", how="left")
    df["_w"] = area_km2.reindex(df["hru_id"]).to_numpy()
    if df["_w"].isna().any():
        raise ValueError("gage_hrus contains hru_id with no area")
    out = {}
    for col in cov_hru.columns:
        if col == "hru_id":
            continue
        if col in CATEGORICAL:
            out[col] = df.groupby(["poi_gage_id", col])["_w"].sum().reset_index().sort_values("_w").groupby("poi_gage_id")[col].last()
        else:
            w = df["_w"].where(df[col].notna(), 0.0)
            out[col] = (df[col].fillna(0.0) * w).groupby(df["poi_gage_id"]).sum() / w.groupby(df["poi_gage_id"]).sum()
    return pd.DataFrame(out).rename_axis("poi_gage_id").reset_index()
```

`src/nhf_runoff_bias/covariates/terrain.py`:
```python
"""Terrain + soils per HRU from gfv2-params outputs, joined on nhm_id == nat_hru_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_unique(paths: list[Path], col: str, out: str) -> pd.Series:
    df = pd.concat([pd.read_csv(p, usecols=["nat_hru_id", col]) for p in paths], ignore_index=True)
    if df["nat_hru_id"].duplicated().any():
        n = int(df["nat_hru_id"].duplicated().sum())
        raise ValueError(f"duplicate nat_hru_id rows ({n}) in {[p.name for p in paths]}")
    return df.set_index("nat_hru_id")[col].rename(out)


def build_terrain(params_dir: Path, hru_table: pd.DataFrame) -> pd.DataFrame:
    merged = params_dir / "nhm_params_merged"
    elev = _read_unique(sorted((params_dir / "elevation").glob("*.csv")), "mean", "elev_m")
    slope = _read_unique([merged / "nhm_slope_params.csv"], "mean", "slope_deg")
    aspect = _read_unique([merged / "nhm_aspect_params.csv"], "mean", "aspect_deg")
    soils = _read_unique([merged / "nhm_soils_params.csv"], "soils", "soil_type")
    smax = _read_unique([merged / "nhm_soil_moist_max_params.csv"], "soil_moist_max", "soil_moist_max")
    tbl = pd.concat([elev, slope, aspect, soils, smax], axis=1)
    missing = set(hru_table["nhm_id"]) - set(tbl.index)
    if missing:
        raise ValueError(f"nhm_id not in gfv2-params tables: {sorted(missing)[:10]}")
    out = tbl.reindex(hru_table["nhm_id"].to_numpy())
    out.index = hru_table["hru_id"].to_numpy()
    rad = np.radians(out.pop("aspect_deg"))
    out["northness"] = np.cos(rad)
    out["eastness"] = np.sin(rad)
    out["soil_type"] = out["soil_type"].astype(int)
    cols = ["elev_m", "slope_deg", "northness", "eastness", "soil_type", "soil_moist_max"]
    return out[cols].rename_axis("hru_id").reset_index()
```

- [ ] **Step 4: Run tests, commit**

`pixi run -e dev pytest tests/test_cov_terrain.py -v` → 4 passed.
```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_runoff_bias/covariates/ tests/test_cov_terrain.py
pixi run git commit -m "feat: terrain/soils covariates from gfv2-params + basin-mean reducer"
```

---

### Task 6: Climate covariates from ClimGrid via cached gdptools weights

**Files:**
- Create: `src/nhf_runoff_bias/covariates/climate.py`
- Test: `tests/test_cov_climate.py`, `tests/test_integration_climate.py`

**Interfaces:**
- Produces: `climate.load_weights(glob_pattern: str, expected_fingerprint: str | None) -> pd.DataFrame` (`hru_id`, `i`, `j`, `wght`); `climate.apply_weights(field: np.ndarray, weights: pd.DataFrame, hru_ids: np.ndarray) -> np.ndarray` (weighted mean over finite cells, `field` shaped `(lat, lon)` with `i` indexing lat and `j` lon — verified by the integration test); `climate.build_climate(climgrid_nc: Path, weights_glob: str, hru_table: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame` on `hru_id` with `prcp_mm_yr`, `pet_mm_yr`, `aridity` (= pet/prcp), `snow_frac` (= Σsnow/Σprcp), `tmean_c`.
- The `.meta` sidecar next to each weights CSV holds a 64-hex fingerprint. `load_weights` requires it to exist and, when `expected_fingerprint` is given, to match; otherwise raises `ValueError("weights fingerprint")`.

- [ ] **Step 1: Write the failing unit tests**

`tests/test_cov_climate.py`:
```python
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nhf_runoff_bias.covariates.climate import apply_weights, build_climate, load_weights


def _weights(tmp_path, fp="a" * 64):
    w = pd.DataFrame({"hru_id": [1, 1, 2], "i": [0, 0, 1], "j": [0, 1, 1], "wght": [0.5, 0.5, 1.0]})
    p = tmp_path / "src_batch0.csv"
    w.to_csv(p, index=False)
    (tmp_path / "src_batch0.csv.meta").write_text(fp)
    return str(tmp_path / "src_batch*.csv")


def test_apply_weights_skips_nan_cells(tmp_path):
    w = load_weights(_weights(tmp_path), None)
    field = np.array([[10.0, np.nan], [0.0, 30.0]])
    out = apply_weights(field, w, np.array([1, 2]))
    assert out.tolist() == [10.0, 30.0]  # HRU1: only finite cell; HRU2: cell (1,1)


def test_fingerprint_mismatch_is_fatal(tmp_path):
    with pytest.raises(ValueError, match="fingerprint"):
        load_weights(_weights(tmp_path), "b" * 64)


def test_build_climate_derives_indices(tmp_path):
    g = _weights(tmp_path)
    time = pd.date_range("1980-01-01", "1981-12-01", freq="MS")
    shape = (len(time), 2, 2)
    ds = xr.Dataset(
        {
            "prcp": (("time", "lat", "lon"), np.full(shape, 100.0)),
            "pet": (("time", "lat", "lon"), np.full(shape, 50.0)),
            "snow": (("time", "lat", "lon"), np.full(shape, 25.0)),
            "tmean": (("time", "lat", "lon"), np.full(shape, 8.0)),
        },
        coords={"time": time, "lat": [45.0, 44.0], "lon": [-122.0, -121.0]},
    )
    p = tmp_path / "climgrid.nc"
    ds.to_netcdf(p)
    hru = pd.DataFrame({"hru_id": [1, 2]})
    c = build_climate(p, g, hru, ("1980-01-01", "1981-12-31")).set_index("hru_id")
    assert c.loc[1, "prcp_mm_yr"] == pytest.approx(1200.0)
    assert c.loc[1, "aridity"] == pytest.approx(0.5)
    assert c.loc[2, "snow_frac"] == pytest.approx(0.25)
    assert c.loc[2, "tmean_c"] == pytest.approx(8.0)
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `climate.py`**

```python
"""Climate covariates: monthly ClimGrid (prcp, pet, snow, tmean) area-weighted to HRUs.

Reuses the gdptools weight CSVs that nhf-spatial-targets cached for the same
grid (`<project>/weights/mwbm_climgrid_batch*.csv`, columns hru_id,i,j,wght).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)
VARS = ("prcp", "pet", "snow", "tmean")


def load_weights(glob_pattern: str, expected_fingerprint: str | None) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"no weights match {glob_pattern}")
    parts = []
    for p in paths:
        meta = Path(p + ".meta")
        if not meta.exists():
            raise ValueError(f"weights fingerprint sidecar missing: {meta}")
        fp = meta.read_text().strip()
        if expected_fingerprint is not None and fp != expected_fingerprint:
            raise ValueError(f"weights fingerprint mismatch for {p}: {fp[:8]} != {expected_fingerprint[:8]}")
        parts.append(pd.read_csv(p, usecols=["hru_id", "i", "j", "wght"]))
    return pd.concat(parts, ignore_index=True)


def apply_weights(field: np.ndarray, weights: pd.DataFrame, hru_ids: np.ndarray) -> np.ndarray:
    """Weighted mean of a (lat, lon) field over finite cells, per hru_id."""
    v = field[weights["i"].to_numpy(), weights["j"].to_numpy()]
    ok = np.isfinite(v)
    w = weights["wght"].to_numpy() * ok
    num = pd.Series(np.where(ok, v, 0.0) * w).groupby(weights["hru_id"].to_numpy()).sum()
    den = pd.Series(w).groupby(weights["hru_id"].to_numpy()).sum()
    out = (num / den.replace(0.0, np.nan)).reindex(hru_ids)
    return out.to_numpy()


def build_climate(climgrid_nc: Path, weights_glob: str, hru_table: pd.DataFrame, window: tuple[str, str], fingerprint: str | None = None) -> pd.DataFrame:
    w = load_weights(weights_glob, fingerprint)
    hru_ids = hru_table["hru_id"].to_numpy()
    ds = xr.open_dataset(climgrid_nc).sel(time=slice(*window))
    n_years = len(np.unique(ds["time"].dt.year.values))
    sums = {}
    for var in VARS:
        # time-sum (or mean for tmean) on the grid first, then one weight pass
        grid = ds[var].mean("time") if var == "tmean" else ds[var].sum("time", min_count=1)
        sums[var] = apply_weights(grid.values, w, hru_ids)
        log.info("climate: %s reduced over %d years", var, n_years)
    out = pd.DataFrame({"hru_id": hru_ids})
    out["prcp_mm_yr"] = sums["prcp"] / n_years
    out["pet_mm_yr"] = sums["pet"] / n_years
    out["aridity"] = sums["pet"] / sums["prcp"]
    out["snow_frac"] = sums["snow"] / sums["prcp"]
    out["tmean_c"] = sums["tmean"]
    return out
```

- [ ] **Step 4: Run unit tests** — 3 passed.

- [ ] **Step 5: Write the integration test that pins the `i`/`j` convention**

`tests/test_integration_climate.py`:
```python
"""Recompute the pipeline's own aggregated MWBM runoff from the cached weights.

If i/j were swapped or the weights were stale this would not match, so this
is the test that licenses reusing nhf-spatial-targets' weight cache.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nhf_runoff_bias.covariates.climate import apply_weights, load_weights

OR = Path("/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets")
DS = Path("/caldera/hovenweep/projects/usgs/water/impd/nhgf/nhf-datastore/mwbm_climgrid/ClimGrid_WBM.nc")


@pytest.mark.integration
@pytest.mark.skipif(not DS.exists(), reason="Oregon inputs not on this host")
def test_weights_reproduce_pipeline_runoff():
    w = load_weights(str(OR / "weights/mwbm_climgrid_batch*.csv"), None)
    agg = xr.open_dataset(OR / "data/aggregated/mwbm_climgrid/mwbm_climgrid_2000_agg.nc")
    src = xr.open_dataset(DS).sel(time="2000-06")
    field = src["runoff"].isel(time=0).values
    mine = apply_weights(field, w, agg["hru_id"].values)
    theirs = agg["runoff"].sel(time="2000-06").isel(time=0).values
    ok = np.isfinite(theirs)
    np.testing.assert_allclose(mine[ok], theirs[ok], rtol=1e-4, atol=1e-3)
```
Run: `pixi run -e dev test-integration` → 1 passed. **If it fails with values that look transposed, swap `i`/`j` in `apply_weights` and re-run; the unit test's `field[[0,0,1],[0,1,1]]` indexing must be updated to match.**

- [ ] **Step 6: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add src/nhf_runoff_bias/covariates/climate.py tests/test_cov_climate.py tests/test_integration_climate.py
pixi run git commit -m "feat: climate covariates from ClimGrid via cached gdptools weights"
```

---

### Task 7: Geology covariates — log-permeability overlay + SGMC lithology class

**Files:**
- Create: `src/nhf_runoff_bias/covariates/geology.py`
- Test: `tests/test_cov_geology.py`

**Interfaces:**
- Produces: `geology.log_permeability(hru_gdf: gpd.GeoDataFrame, litho: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame` (`hru_id`, `log10_k_perm`, `litho_cover_frac`), area-weighted mean of `log10(k_perm)` over the intersection; `geology.majority_class(hru_gdf, poly: gpd.GeoDataFrame, class_field: str, id_col: str) -> pd.DataFrame` (`hru_id`, `litho_class`); `geology.build_geology(fab: FabricConfig, cov: CovariatesConfig, datastore: Path) -> pd.DataFrame`.
- SGMC download: `geology.fetch_sgmc(datastore: Path, doi: str) -> Path` resolves the DOI to its ScienceBase item via `https://doi.org/<doi>` redirect, downloads the geodatabase zip into `datastore/sgmc/`, unzips, and returns the `.gdb` path. Verify the DOI resolves to "State Geologic Map Compilation (SGMC) geodatabase of the conterminous United States" (Horton, San Juan, Stoeser 2017) before relying on it; the layer with polygon geology is the one whose name starts with `SGMC_Geology`, and `sgmc_class_field` defaults to `ROCKTYPE1`. If `pyogrio.read_info` shows no such field, fail with the list of available fields.

- [ ] **Step 1: Write the failing tests**

`tests/test_cov_geology.py`:
```python
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from nhf_runoff_bias.covariates.geology import log_permeability, majority_class

CRS = "EPSG:5070"


def _hrus():
    return gpd.GeoDataFrame({"hru_id": [1, 2]}, geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=CRS)


def test_log_permeability_area_weighted():
    litho = gpd.GeoDataFrame(
        {"k_perm": [1e-12, 1e-14, 1e-10]},
        geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10), box(10, 0, 30, 10)],
        crs=CRS,
    )
    out = log_permeability(_hrus(), litho, "hru_id").set_index("hru_id")
    assert out.loc[1, "log10_k_perm"] == pytest.approx(-13.0)  # half -12, half -14
    assert out.loc[2, "log10_k_perm"] == pytest.approx(-10.0)
    assert out.loc[1, "litho_cover_frac"] == pytest.approx(1.0)


def test_log_permeability_partial_cover_and_nodata():
    litho = gpd.GeoDataFrame({"k_perm": [1e-12, 0.0]}, geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10)], crs=CRS)
    out = log_permeability(_hrus(), litho, "hru_id").set_index("hru_id")
    assert out.loc[1, "log10_k_perm"] == pytest.approx(-12.0)  # k=0 treated as nodata
    assert out.loc[1, "litho_cover_frac"] == pytest.approx(0.5)
    assert np.isnan(out.loc[2, "log10_k_perm"]) and out.loc[2, "litho_cover_frac"] == 0.0


def test_majority_class():
    poly = gpd.GeoDataFrame({"ROCKTYPE1": ["basalt", "sandstone", "basalt"]}, geometry=[box(0, 0, 6, 10), box(6, 0, 10, 10), box(10, 0, 20, 10)], crs=CRS)
    out = majority_class(_hrus(), poly, "ROCKTYPE1", "hru_id").set_index("hru_id")
    assert out.loc[1, "litho_class"] == "basalt" and out.loc[2, "litho_class"] == "basalt"
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `geology.py`**

```python
"""Geology covariates: log10 permeability (Gleeson 2011 polygons) and SGMC lithology class."""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from nhf_runoff_bias.config import CovariatesConfig, FabricConfig

log = logging.getLogger(__name__)


def _intersections(hru_gdf: gpd.GeoDataFrame, poly: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    poly = poly.to_crs(hru_gdf.crs)
    inter = gpd.overlay(hru_gdf[[id_col, "geometry"]], poly, how="intersection", keep_geom_type=False)
    inter["_area"] = inter.geometry.area
    return inter


def log_permeability(hru_gdf: gpd.GeoDataFrame, litho: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame:
    litho = litho[litho["k_perm"] > 0]  # 0 / negative = nodata
    inter = _intersections(hru_gdf, litho, id_col)
    inter["_lk"] = np.log10(inter["k_perm"]) * inter["_area"]
    g = inter.groupby(id_col)
    hru_area = hru_gdf.set_index(id_col).geometry.area
    out = pd.DataFrame({"hru_id": hru_gdf[id_col].to_numpy()}).set_index("hru_id")
    out["log10_k_perm"] = (g["_lk"].sum() / g["_area"].sum()).reindex(out.index)
    out["litho_cover_frac"] = (g["_area"].sum() / hru_area).reindex(out.index).fillna(0.0)
    return out.reset_index()


def majority_class(hru_gdf: gpd.GeoDataFrame, poly: gpd.GeoDataFrame, class_field: str, id_col: str) -> pd.DataFrame:
    inter = _intersections(hru_gdf, poly[[class_field, "geometry"]], id_col)
    top = inter.groupby([id_col, class_field])["_area"].sum().reset_index().sort_values("_area").groupby(id_col)[class_field].last()
    return pd.DataFrame({"hru_id": hru_gdf[id_col].to_numpy(), "litho_class": top.reindex(hru_gdf[id_col]).to_numpy()})


def fetch_sgmc(datastore: Path, doi: str) -> Path:
    dest = datastore / "sgmc"
    gdbs = list(dest.glob("*.gdb")) if dest.exists() else []
    if gdbs:
        return gdbs[0]
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://doi.org/{doi}") as r:  # noqa: S310 - fixed DOI
        landing = r.geturl()
    item_id = landing.rstrip("/").split("/")[-1]
    listing = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json&fields=files"
    import json

    files = json.load(urllib.request.urlopen(listing))["files"]  # noqa: S310
    zips = [f for f in files if f["name"].lower().endswith(".zip") and "gdb" in f["name"].lower()]
    if not zips:
        raise RuntimeError(f"no geodatabase zip on ScienceBase item {item_id}: {[f['name'] for f in files]}")
    zpath = dest / zips[0]["name"]
    log.info("downloading %s", zips[0]["url"])
    with urllib.request.urlopen(zips[0]["url"]) as r, open(zpath, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    gdbs = list(dest.rglob("*.gdb"))
    if not gdbs:
        raise RuntimeError(f"no .gdb after extracting {zpath}")
    return gdbs[0]


def build_geology(fab: FabricConfig, cov: CovariatesConfig, datastore: Path) -> pd.DataFrame:
    import pyogrio

    idc = fab.cols["hru_id"]
    hru = gpd.read_file(fab.gpkg, layer=fab.layers["hru"])[[idc, "geometry"]]
    litho = gpd.read_file(cov.lithology_shp, columns=["k_perm"], bbox=tuple(hru.to_crs(pyogrio.read_info(cov.lithology_shp)["crs"]).total_bounds))
    out = log_permeability(hru, litho, idc)
    gdb = fetch_sgmc(datastore, cov.sgmc_doi)
    layer = next(n for n, _ in pyogrio.list_layers(gdb) if n.startswith("SGMC_Geology"))
    fields = pyogrio.read_info(gdb, layer=layer)["fields"]
    if cov.sgmc_class_field not in fields:
        raise ValueError(f"{cov.sgmc_class_field} not in SGMC layer {layer}; fields: {list(fields)}")
    sg = gpd.read_file(gdb, layer=layer, columns=[cov.sgmc_class_field], bbox=tuple(hru.to_crs(pyogrio.read_info(gdb, layer=layer)["crs"]).total_bounds))
    return out.merge(majority_class(hru, sg, cov.sgmc_class_field, idc), on="hru_id")
```

- [ ] **Step 4: Run unit tests** — 3 passed. Then a one-off real check (not a test): `pixi run python -c "from nhf_runoff_bias.covariates.geology import fetch_sgmc; from pathlib import Path; print(fetch_sgmc(Path('datastore'), '10.5066/F7WH2N65'))"` and confirm the printed `.gdb` and that `pyogrio.list_layers` shows an `SGMC_Geology*` layer with `ROCKTYPE1`. Record the resolved ScienceBase item id in `config.example.yml` as a comment.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint
git add src/nhf_runoff_bias/covariates/geology.py tests/test_cov_geology.py config.example.yml
pixi run git commit -m "feat: geology covariates (log-permeability overlay, SGMC lithology class)"
```

---

### Task 8: `bias.py` — volume bias, seasonal ratios, baseflow index, tiers

**Files:**
- Create: `src/nhf_runoff_bias/bias.py`
- Modify: `src/nhf_runoff_bias/cli.py` (`covariates` and `bias` commands)
- Test: `tests/test_bias.py`

**Interfaces:**
- Consumes: `accumulated` Dataset (Task 4), `gages_monthly`, `gage_meta` (Task 3), `gage_network` (Task 2).
- Produces: `bias.compute_bias(acc: xr.Dataset, gages_monthly: pd.DataFrame, min_years: int) -> pd.DataFrame` with one row per (`poi_gage_id`, `member`): `n_overlap_years`, `q_obs_mean`, `q_acc_mean`, `log_ratio`, `ratio_m01`..`ratio_m12`, `exclusion_reason` (`""`, `"insufficient_overlap"`, `"zero_observed_mean"`). Overlap year = calendar year with 12 complete gage months and 12 finite accumulated months for that member.
- `bias.baseflow_index(daily: pd.Series, alpha: float = 0.925, passes: int = 3) -> float` (Lyne–Hollick, forward/backward/forward).
- `bias.assign_tiers(gage_meta, gage_network, bias_df, min_years, area_tol) -> pd.DataFrame` adds `tier` (`"A"`, `"B"`, `"none"`) and `tier_reason` per gage. B ⊂ A.

- [ ] **Step 1: Write the failing tests**

`tests/test_bias.py`:
```python
import numpy as np
import pandas as pd
import pytest

from nhf_runoff_bias.accumulate import accumulate_members
from nhf_runoff_bias.bias import assign_tiers, baseflow_index, compute_bias
from nhf_runoff_bias.gages import load_gages
from nhf_runoff_bias.network import build_network


def _pieces(cfg):
    net = build_network(cfg.fabric)
    acc = accumulate_members(cfg.target_nc, net.gage_hrus(), cfg.members)
    monthly, meta = load_gages(cfg.gages)
    return net, acc, monthly, meta


def test_log_ratio_recovers_true_factor(synthetic_config):
    _, acc, monthly, _ = _pieces(synthetic_config)
    b = compute_bias(acc, monthly, min_years=1).set_index(["poi_gage_id", "member"])
    # G1, m_a: obs = 2.0 * acc every month; both years complete -> ln 2
    assert b.loc[("G1", "m_a"), "log_ratio"] == pytest.approx(np.log(2.0))
    assert b.loc[("G1", "m_a"), "n_overlap_years"] == 2
    assert b.loc[("G1", "m_a"), "ratio_m07"] == pytest.approx(2.0)
    # G2: 2001-01 is incomplete at the gage AND NaN in m_a -> 2001 dropped, only 2002
    assert b.loc[("G2", "m_a"), "n_overlap_years"] == 1
    assert b.loc[("G2", "m_a"), "log_ratio"] == pytest.approx(np.log(1.5))
    # G2, m_b: acc is 2x m_a, obs fixed -> ln(1.5/2)
    assert b.loc[("G2", "m_b"), "log_ratio"] == pytest.approx(np.log(0.75))


def test_insufficient_overlap_and_zero_obs(synthetic_config):
    _, acc, monthly, _ = _pieces(synthetic_config)
    b = compute_bias(acc, monthly, min_years=2).set_index(["poi_gage_id", "member"])
    assert b.loc[("G2", "m_a"), "exclusion_reason"] == "insufficient_overlap"
    assert np.isnan(b.loc[("G2", "m_a"), "log_ratio"])
    monthly.loc[monthly.poi_gage_id == "G1", "q_cfs"] = 0.0
    b0 = compute_bias(acc, monthly, min_years=1).set_index(["poi_gage_id", "member"])
    assert b0.loc[("G1", "m_a"), "exclusion_reason"] == "zero_observed_mean"
    assert np.isfinite(b0["log_ratio"].fillna(0)).all()  # never -inf


def test_baseflow_index_bounds():
    t = pd.date_range("2001-01-01", periods=400, freq="D")
    const = pd.Series(10.0, index=t)
    assert baseflow_index(const) == pytest.approx(1.0, abs=1e-6)
    spiky = pd.Series(np.where(np.arange(400) % 50 == 0, 1000.0, 1.0), index=t)
    assert 0.0 < baseflow_index(spiky) < 0.5


def test_assign_tiers(synthetic_config):
    net, acc, monthly, meta = _pieces(synthetic_config)
    b = compute_bias(acc, monthly, min_years=1)
    t = assign_tiers(meta, net.gage_network(), b, min_years=1, area_tol=0.10).set_index("poi_gage_id")
    assert t.loc["G1", "tier"] == "B"  # FMI 0, area ok, no edge
    assert t.loc["G2", "tier"] == "none" and "domain_edge" in t.loc["G2", "tier_reason"]
    assert t.loc["G9", "tier"] == "none"
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `bias.py`**

```python
"""Per-gage, per-member volume bias; seasonal ratio shape; baseflow index; tiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

MONTH_COLS = [f"ratio_m{m:02d}" for m in range(1, 13)]


def compute_bias(acc: xr.Dataset, gages_monthly: pd.DataFrame, min_years: int) -> pd.DataFrame:
    obs = gages_monthly.pivot(index="time", columns="poi_gage_id", values="q_cfs")  # NaN when incomplete
    rows = []
    for g in acc["poi_gage_id"].values:
        for m in acc["member"].values:
            a = acc["q_acc_cfs"].sel(poi_gage_id=g, member=m).to_series()
            if g not in obs.columns:
                rows.append(_row(g, m, 0, np.nan, np.nan, "no_observations"))
                continue
            df = pd.DataFrame({"obs": obs[g], "acc": a}).dropna()
            per_year = df.groupby(df.index.year).size()
            years = per_year[per_year == 12].index
            df = df[df.index.year.isin(years)]
            if len(years) < min_years:
                rows.append(_row(g, m, len(years), np.nan, np.nan, "insufficient_overlap"))
                continue
            qo, qa = df["obs"].mean(), df["acc"].mean()
            if qo <= 0:
                rows.append(_row(g, m, len(years), qo, qa, "zero_observed_mean"))
                continue
            clim = df.groupby(df.index.month).mean()
            r = _row(g, m, len(years), qo, qa, "")
            r["log_ratio"] = float(np.log(qo / qa))
            for k, col in enumerate(MONTH_COLS, start=1):
                r[col] = float(clim.loc[k, "obs"] / clim.loc[k, "acc"]) if clim.loc[k, "acc"] > 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _row(g, m, n, qo, qa, reason) -> dict:
    return {"poi_gage_id": str(g), "member": str(m), "n_overlap_years": int(n), "q_obs_mean": qo, "q_acc_mean": qa, "log_ratio": np.nan, **{c: np.nan for c in MONTH_COLS}, "exclusion_reason": reason}


def baseflow_index(daily: pd.Series, alpha: float = 0.925, passes: int = 3) -> float:
    """Lyne-Hollick one-parameter filter, alternating forward/backward passes."""
    q = daily.dropna().to_numpy(dtype=float)
    if q.size < 30 or q.sum() <= 0:
        return float("nan")
    b = q.copy()
    for p in range(passes):
        src = b if p % 2 == 0 else b[::-1]
        out = np.empty_like(src)
        out[0] = src[0]
        for i in range(1, len(src)):
            # Lyne-Hollick: qf_i = a*qf_{i-1} + (1+a)/2*(q_i - q_{i-1}); base = q - qf, clipped to [0, q]
            qf_prev = src[i - 1] - out[i - 1]
            qf = alpha * qf_prev + (1 + alpha) / 2 * (src[i] - src[i - 1])
            out[i] = min(max(src[i] - max(qf, 0.0), 0.0), src[i])
        b = out if p % 2 == 0 else out[::-1]
    return float(b.sum() / q.sum())


def assign_tiers(gage_meta: pd.DataFrame, gage_network: pd.DataFrame, bias_df: pd.DataFrame, min_years: int, area_tol: float) -> pd.DataFrame:
    meta = gage_meta.set_index("poi_gage_id")
    net = gage_network.set_index("poi_gage_id")
    years = bias_df.groupby("poi_gage_id")["n_overlap_years"].max()
    out = []
    for g in meta.index:
        reasons = []
        if meta.loc[g, "exclusion_reason"]:
            reasons.append(meta.loc[g, "exclusion_reason"])
        if g not in net.index or net.loc[g, "exclusion_reason"]:
            reasons.append("segment_not_in_fabric")
        else:
            if bool(net.loc[g, "touches_domain_edge"]):
                reasons.append("domain_edge")
            pub = meta.loc[g, "published_area_km2"]
            if not np.isfinite(pub):
                reasons.append("no_published_area")
            elif abs(net.loc[g, "fabric_area_km2"] / pub - 1.0) > area_tol:
                reasons.append("area_mismatch")
        if years.get(g, 0) < min_years:
            reasons.append("insufficient_overlap")
        fmi = meta.loc[g, "flow_management_index"]
        if pd.isna(fmi):
            reasons.append("no_fmi")
        tier = "none"
        if not reasons:
            tier = "B" if fmi == 0 else ("A" if fmi <= 1 else "none")
            if tier == "none":
                reasons.append("fmi_gt_1")
        out.append({"poi_gage_id": g, "tier": tier, "tier_reason": ";".join(reasons)})
    return pd.DataFrame(out)
```
- [ ] **Step 4: Run tests** — 4 passed.

- [ ] **Step 5: Wire the CLI `covariates` and `bias` commands**

```python
@app.command
def covariates(config: ConfigArg = Path("config.yml")) -> None:
    """Terrain, climate, geology -> covariates_hru.parquet, covariates_basin.parquet."""
    import pandas as pd

    from nhf_runoff_bias.covariates import basin_means
    from nhf_runoff_bias.covariates.climate import build_climate
    from nhf_runoff_bias.covariates.geology import build_geology
    from nhf_runoff_bias.covariates.terrain import build_terrain
    from nhf_runoff_bias.network import build_network
    from nhf_runoff_bias.provenance import fabric_ref, stamp, write_parquet

    cfg = _setup(config)
    net = build_network(cfg.fabric)
    hru = net.hru_table
    parts = [build_terrain(cfg.covariates.gfv2_params_dir, hru)]
    parts.append(build_climate(cfg.covariates.climgrid_nc, cfg.covariates.climgrid_weights_glob, hru, cfg.covariates.climate_window))
    parts.append(build_geology(cfg.fabric, cfg.covariates, cfg.datastore))
    cov = parts[0]
    for p in parts[1:]:
        cov = cov.merge(p, on="hru_id", how="left")
    n_nan = cov.drop(columns="hru_id").isna().sum()
    log.info("covariates: NaN counts per column: %s", n_nan[n_nan > 0].to_dict())
    inputs = {"fabric_gpkg": cfg.fabric.gpkg, "climgrid_nc": cfg.covariates.climgrid_nc, "lithology_shp": cfg.covariates.lithology_shp}
    meta = stamp(inputs, fabric_ref(cfg.fabric.fabric_json))
    write_parquet(cov, cfg.run_dir / "covariates_hru.parquet", meta)
    gh = pd.read_parquet(cfg.run_dir / "gage_hrus.parquet")
    write_parquet(basin_means(cov, gh, hru.set_index("hru_id")["area_km2"]), cfg.run_dir / "covariates_basin.parquet", meta)


@app.command
def bias(config: ConfigArg = Path("config.yml")) -> None:
    """Target + gages + network -> accumulated.nc, bias.parquet, gage_tiers.parquet."""
    import pandas as pd
    import xarray as xr

    from nhf_runoff_bias.accumulate import accumulate_members
    from nhf_runoff_bias.bias import assign_tiers, baseflow_index, compute_bias
    from nhf_runoff_bias.provenance import fabric_ref, nc_attrs, stage_is_current, stamp, write_parquet

    cfg = _setup(config)
    rd = cfg.run_dir
    outs = [rd / "accumulated.nc", rd / "bias.parquet", rd / "gage_tiers.parquet"]
    inputs = {"target_nc": cfg.target_nc, "gages_monthly": rd / "gages_monthly.parquet", "gage_hrus": rd / "gage_hrus.parquet"}
    if stage_is_current(outs, inputs):
        log.info("bias: up to date")
        return
    fab = fabric_ref(cfg.fabric.fabric_json)
    gh = pd.read_parquet(rd / "gage_hrus.parquet")
    acc = accumulate_members(cfg.target_nc, gh, cfg.members, id_col=fab["id_col"], expected_fabric_sha256=fab["fabric_sha256"])
    meta = stamp(inputs, fab)
    acc.attrs.update(nc_attrs(meta))
    acc.to_netcdf(outs[0])
    monthly = pd.read_parquet(rd / "gages_monthly.parquet")
    gmeta = pd.read_parquet(rd / "gage_meta.parquet")
    b = compute_bias(acc, monthly, cfg.gages.min_overlap_years)
    daily = xr.open_dataset(cfg.gages.daily_nc)["discharge"]
    bfi = {g: baseflow_index(daily.sel(poi_gage_id=g).to_series()) for g in b["poi_gage_id"].unique() if g in daily["poi_gage_id"].values}
    b["bfi"] = b["poi_gage_id"].map(bfi)
    write_parquet(b, outs[1], meta)
    tiers = assign_tiers(gmeta, pd.read_parquet(rd / "gage_network.parquet"), b, cfg.gages.min_overlap_years, cfg.gages.area_tolerance)
    write_parquet(tiers, outs[2], meta)
    log.info("bias: tiers %s", tiers["tier"].value_counts().to_dict())
```

- [ ] **Step 6: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add src/nhf_runoff_bias/bias.py src/nhf_runoff_bias/cli.py tests/test_bias.py
pixi run git commit -m "feat: volume bias, seasonal ratios, BFI, gage tiers + covariates/bias CLI"
```

---

### Task 9: Approach A — PCA + OLS + random forest with LOO CV, HRU prediction

**Files:**
- Create: `src/nhf_runoff_bias/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `models.PREDICTORS = ["elev_m", "slope_deg", "northness", "eastness", "soil_moist_max", "log10_k_perm", "prcp_mm_yr", "pet_mm_yr", "aridity", "snow_frac", "tmean_c"]`; `models.fit_gage_regression(bias_df, cov_basin, cov_hru, tiers, member: str, tier: str, pca_variance: float, seed: int = 0) -> FitResult`; `FitResult` dataclass: `member`, `tier`, `n_gages`, `pca_components: int`, `pca_loadings: pd.DataFrame`, `ols_coef: dict`, `ols_loo_rmse`, `ols_loo_r2`, `rf_loo_rmse`, `rf_loo_r2`, `shap_mean_abs: dict[str, float]`, `factor_hru: pd.DataFrame` (`hru_id`, `factor_ols`, `factor_rf`, `factor_se`), `to_json() -> dict`.
- Predictors present in `cov_basin` but all-NaN are dropped with a logged warning; rows with any NaN predictor are dropped and counted in `n_dropped_nan`.

- [ ] **Step 1: Write the failing test (synthetic bias with a known linear structure)**

`tests/test_models.py`:
```python
import numpy as np
import pandas as pd
import pytest

from nhf_runoff_bias.models import PREDICTORS, fit_gage_regression


def _synthetic(n_gages=60, n_hru=200, seed=1):
    rng = np.random.default_rng(seed)
    hru = pd.DataFrame({"hru_id": np.arange(1, n_hru + 1)})
    for p in PREDICTORS:
        hru[p] = rng.normal(size=n_hru)
    truth = lambda df: 0.6 * df["elev_m"] - 0.4 * df["aridity"] + 0.3 * df["snow_frac"]  # noqa: E731
    basin = pd.DataFrame({"poi_gage_id": [f"g{i}" for i in range(n_gages)]})
    for p in PREDICTORS:
        basin[p] = rng.normal(size=n_gages)
    bias = pd.DataFrame({"poi_gage_id": basin.poi_gage_id, "member": "m", "log_ratio": truth(basin) + rng.normal(scale=0.05, size=n_gages), "exclusion_reason": ""})
    tiers = pd.DataFrame({"poi_gage_id": basin.poi_gage_id, "tier": "A"})
    return bias, basin, hru, tiers, truth


def test_ols_recovers_linear_structure():
    bias, basin, hru, tiers, truth = _synthetic()
    fr = fit_gage_regression(bias, basin, hru, tiers, member="m", tier="A", pca_variance=0.99)
    assert fr.n_gages == 60
    assert fr.ols_loo_r2 > 0.85
    pred = fr.factor_hru.set_index("hru_id")["factor_ols"]
    np.testing.assert_allclose(np.log(pred.to_numpy()), truth(hru).to_numpy(), atol=0.25)
    assert set(fr.shap_mean_abs) == set(PREDICTORS)
    top = sorted(fr.shap_mean_abs, key=fr.shap_mean_abs.get)[-3:]
    assert {"elev_m", "aridity", "snow_frac"} == set(top)


def test_tier_b_subset_and_json_roundtrip():
    bias, basin, hru, tiers, _ = _synthetic()
    tiers.loc[tiers.index[:20], "tier"] = "B"
    fr = fit_gage_regression(bias, basin, hru, tiers, member="m", tier="B", pca_variance=0.9)
    assert fr.n_gages == 20
    d = fr.to_json()
    assert d["member"] == "m" and d["tier"] == "B" and "ols_coef" in d and "pca_loadings" in d


def test_all_nan_predictor_is_dropped_not_fatal():
    bias, basin, hru, tiers, _ = _synthetic()
    basin["litho_cover_frac"] = np.nan
    basin["tmean_c"] = np.nan
    fr = fit_gage_regression(bias, basin, hru, tiers, member="m", tier="A", pca_variance=0.9)
    assert "tmean_c" not in fr.predictors_used
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `models.py` (approach A)**

```python
"""Approach A: gage-level regression of log bias on basin covariates, predicted per HRU."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
PREDICTORS = ["elev_m", "slope_deg", "northness", "eastness", "soil_moist_max", "log10_k_perm", "prcp_mm_yr", "pet_mm_yr", "aridity", "snow_frac", "tmean_c"]


@dataclass
class FitResult:
    member: str
    tier: str
    n_gages: int
    n_dropped_nan: int
    predictors_used: list[str]
    pca_components: int
    pca_loadings: pd.DataFrame
    ols_coef: dict[str, float]
    ols_loo_rmse: float
    ols_loo_r2: float
    rf_loo_rmse: float
    rf_loo_r2: float
    shap_mean_abs: dict[str, float]
    factor_hru: pd.DataFrame = field(repr=False)

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("factor_hru", "pca_loadings")}
        d["pca_loadings"] = self.pca_loadings.round(4).to_dict()
        return d


def _r2(y, yhat) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_gage_regression(bias_df, cov_basin, cov_hru, tiers, member: str, tier: str, pca_variance: float, seed: int = 0) -> FitResult:
    keep = tiers.loc[tiers["tier"].isin({"A", "B"} if tier == "A" else {"B"}), "poi_gage_id"]
    b = bias_df[(bias_df["member"] == member) & (bias_df["exclusion_reason"] == "") & bias_df["poi_gage_id"].isin(keep)]
    df = b[["poi_gage_id", "log_ratio"]].merge(cov_basin, on="poi_gage_id", how="inner")
    preds = [p for p in PREDICTORS if p in df.columns and df[p].notna().any()]
    dropped = sorted(set(PREDICTORS) - set(preds))
    if dropped:
        log.warning("fit %s/%s: dropping all-NaN predictors %s", member, tier, dropped)
    n0 = len(df)
    df = df.dropna(subset=preds + ["log_ratio"])
    X, y = df[preds].to_numpy(float), df["log_ratio"].to_numpy(float)
    if len(y) < 10:
        raise ValueError(f"fit {member}/{tier}: only {len(y)} gages after screening")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=pca_variance, svd_solver="full").fit(Xs)
    Z = pca.transform(Xs)
    ols = LinearRegression().fit(Z, y)
    loo = LeaveOneOut()
    y_ols = cross_val_predict(LinearRegression(), Z, y, cv=loo)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=seed)
    y_rf = cross_val_predict(rf, Xs, y, cv=loo)
    rf.fit(Xs, y)

    import shap

    sv = shap.TreeExplainer(rf).shap_values(Xs)
    shap_mean_abs = dict(zip(preds, np.abs(sv).mean(axis=0).tolist()))

    # coefficients back in standardized-predictor space: beta = loadings^T @ ols.coef_
    beta = pca.components_.T @ ols.coef_
    ols_coef = {"intercept": float(ols.intercept_), **dict(zip(preds, beta.tolist()))}

    Xh = cov_hru[["hru_id"] + preds].dropna()
    Xhs = scaler.transform(Xh[preds].to_numpy(float))
    pred_ols = ols.predict(pca.transform(Xhs))
    pred_rf = rf.predict(Xhs)
    resid_sd = float(np.std(y - y_ols, ddof=1))
    factor = pd.DataFrame({"hru_id": Xh["hru_id"].to_numpy(), "factor_ols": np.exp(pred_ols), "factor_rf": np.exp(pred_rf), "factor_se": np.exp(pred_ols) * resid_sd})
    factor = factor.set_index("hru_id").reindex(cov_hru["hru_id"]).reset_index()

    return FitResult(
        member=member, tier=tier, n_gages=int(len(y)), n_dropped_nan=int(n0 - len(df)), predictors_used=preds,
        pca_components=int(pca.n_components_), pca_loadings=pd.DataFrame(pca.components_, columns=preds, index=[f"PC{i+1}" for i in range(pca.n_components_)]),
        ols_coef=ols_coef, ols_loo_rmse=float(np.sqrt(np.mean((y - y_ols) ** 2))), ols_loo_r2=_r2(y, y_ols),
        rf_loo_rmse=float(np.sqrt(np.mean((y - y_rf) ** 2))), rf_loo_r2=_r2(y, y_rf), shap_mean_abs=shap_mean_abs, factor_hru=factor,
    )
```

- [ ] **Step 4: Run tests** — 3 passed (RF LOO on 60 gages × 500 trees takes ~20 s; acceptable).

- [ ] **Step 5: Wire the CLI `fit` command (approach A only for now)**

```python
@app.command
def fit(config: ConfigArg = Path("config.yml")) -> None:
    """Covariates + bias -> fit.json, factor_hru.parquet."""
    import json

    import pandas as pd

    from nhf_runoff_bias.models import fit_gage_regression
    from nhf_runoff_bias.provenance import fabric_ref, stamp, write_parquet

    cfg = _setup(config)
    rd = cfg.run_dir
    bias_df = pd.read_parquet(rd / "bias.parquet")
    basin = pd.read_parquet(rd / "covariates_basin.parquet")
    hru = pd.read_parquet(rd / "covariates_hru.parquet")
    tiers = pd.read_parquet(rd / "gage_tiers.parquet")
    results, factors = [], []
    for member in cfg.members:
        for tier in ("A", "B"):
            fr = fit_gage_regression(bias_df, basin, hru, tiers, member, tier, cfg.fit.pca_variance)
            results.append(fr.to_json())
            f = fr.factor_hru.copy()
            f.insert(0, "tier", tier)
            f.insert(0, "member", member)
            factors.append(f)
            log.info("fit %s/%s: n=%d OLS LOO R2=%.2f RF LOO R2=%.2f", member, tier, fr.n_gages, fr.ols_loo_r2, fr.rf_loo_r2)
    inputs = {"bias": rd / "bias.parquet", "covariates_basin": rd / "covariates_basin.parquet", "covariates_hru": rd / "covariates_hru.parquet"}
    meta = stamp(inputs, fabric_ref(cfg.fabric.fabric_json))
    (rd / "fit.json").write_text(json.dumps({"provenance": meta, "fits": results}, indent=2, default=str))
    write_parquet(pd.concat(factors, ignore_index=True), rd / "factor_hru.parquet", meta)
```

- [ ] **Step 6: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add src/nhf_runoff_bias/models.py src/nhf_runoff_bias/cli.py tests/test_models.py
pixi run git commit -m "feat: approach A gage regression (PCA+OLS+RF, LOO CV, SHAP) and fit CLI"
```

---

### Task 10: Approach C — inversion through the accumulation matrix

**Files:**
- Create: `src/nhf_runoff_bias/inversion.py`
- Modify: `src/nhf_runoff_bias/cli.py` (`fit` gains the `run_inversion` branch)
- Test: `tests/test_inversion.py`

**Interfaces:**
- Produces: `inversion.fit_inversion(A: sp.csr_matrix, gage_ids: np.ndarray, q_obs_mean: pd.Series, member_hru_mean: np.ndarray, X_hru: np.ndarray, predictors: list[str], ridge_lambda: float, bootstrap: int, seed: int = 0) -> InversionResult` minimising `Σ_g (ln q_obs_g − ln Σ_h A[g,h]·exp(β0 + β·x_h)·m_h)² + λ‖β‖²` with `scipy.optimize.least_squares`; `InversionResult`: `beta: dict`, `rmse_log`, `factor_hru: pd.DataFrame` (`hru_id`, `factor_inv`, `factor_inv_se` from bootstrap over gages), `n_gages`.
- `member_hru_mean` is the member's long-term mean over the same overlap window per HRU (use the target's full-period mean; NaN HRUs get factor 1 and are masked from the sums).

- [ ] **Step 1: Write the failing test**

`tests/test_inversion.py`:
```python
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nhf_runoff_bias.inversion import fit_inversion


def test_inversion_recovers_hru_scale_coefficients():
    rng = np.random.default_rng(3)
    n_hru, n_gage = 300, 80
    X = rng.normal(size=(n_hru, 2))
    beta_true = np.array([0.5, -0.3])
    b0_true = 0.2
    m = rng.uniform(1, 5, size=n_hru)
    # each gage drains a random contiguous block of 5-30 HRUs
    rows, cols = [], []
    for g in range(n_gage):
        s = rng.integers(0, n_hru - 30)
        k = rng.integers(5, 30)
        rows += [g] * k
        cols += list(range(s, s + k))
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_gage, n_hru))
    q_obs = A @ (np.exp(b0_true + X @ beta_true) * m) * np.exp(rng.normal(scale=0.02, size=n_gage))
    gids = np.array([f"g{i}" for i in range(n_gage)])
    res = fit_inversion(A, gids, pd.Series(q_obs, index=gids), m, X, ["x1", "x2"], ridge_lambda=0.01, bootstrap=30)
    assert abs(res.beta["x1"] - 0.5) < 0.08 and abs(res.beta["x2"] + 0.3) < 0.08
    assert abs(res.beta["intercept"] - 0.2) < 0.1
    f = res.factor_hru.set_index("hru_id")["factor_inv"].to_numpy()
    np.testing.assert_allclose(np.log(f), b0_true + X @ beta_true, atol=0.2)
    assert (res.factor_hru["factor_inv_se"] > 0).all()
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `inversion.py`**

```python
"""Approach C: fit exp(beta . x_h) factors through the gage x HRU accumulation matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import least_squares


@dataclass
class InversionResult:
    beta: dict[str, float]
    rmse_log: float
    n_gages: int
    factor_hru: pd.DataFrame


def _residuals(theta, A, log_q_obs, m, X, lam):
    f = np.exp(theta[0] + X @ theta[1:])
    pred = A @ (f * m)
    return np.concatenate([np.log(pred) - log_q_obs, np.sqrt(lam) * theta[1:]])


def _solve(A, log_q_obs, m, X, lam):
    theta0 = np.zeros(X.shape[1] + 1)
    theta0[0] = float(np.mean(log_q_obs - np.log(A @ m)))
    return least_squares(_residuals, theta0, args=(A, log_q_obs, m, X, lam), method="trf").x


def fit_inversion(A: sp.csr_matrix, gage_ids: np.ndarray, q_obs_mean: pd.Series, member_hru_mean: np.ndarray, X_hru: np.ndarray, predictors: list[str], ridge_lambda: float, bootstrap: int, seed: int = 0) -> InversionResult:
    q = q_obs_mean.reindex(gage_ids).to_numpy(float)
    ok_g = np.isfinite(q) & (q > 0)
    m = np.where(np.isfinite(member_hru_mean), member_hru_mean, 0.0)
    ok_h = np.isfinite(X_hru).all(axis=1) & np.isfinite(member_hru_mean)
    X = np.where(ok_h[:, None], X_hru, 0.0)
    mu, sd = X[ok_h].mean(axis=0), X[ok_h].std(axis=0, ddof=1)
    Xs = np.where(ok_h[:, None], (X - mu) / sd, 0.0)
    A_ok = A[ok_g]
    log_q = np.log(q[ok_g])
    theta = _solve(A_ok, log_q, m, Xs, ridge_lambda)
    pred = A_ok @ (np.exp(theta[0] + Xs @ theta[1:]) * m)
    rmse = float(np.sqrt(np.mean((np.log(pred) - log_q) ** 2)))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(bootstrap):
        idx = rng.integers(0, A_ok.shape[0], size=A_ok.shape[0])
        boots.append(_solve(A_ok[idx], log_q[idx], m, Xs, ridge_lambda))
    boots = np.array(boots)
    logf = theta[0] + Xs @ theta[1:]
    logf_b = boots[:, :1] + Xs @ boots[:, 1:].T  # (hru, boot)
    factor = np.where(ok_h, np.exp(logf), 1.0)
    se = np.where(ok_h, np.exp(logf) * logf_b.std(axis=1, ddof=1), np.nan)
    return InversionResult(
        beta={"intercept": float(theta[0]), **dict(zip(predictors, theta[1:].tolist()))},
        rmse_log=rmse,
        n_gages=int(ok_g.sum()),
        factor_hru=pd.DataFrame({"hru_id": np.arange(len(m)), "factor_inv": factor, "factor_inv_se": se}),
    )
```
Note `factor_hru.hru_id` here is the positional index; the CLI replaces it with the target's `hru_id` coordinate.

- [ ] **Step 4: Run test** — 1 passed.

- [ ] **Step 5: Add the `run_inversion` branch to the `fit` CLI**

Inside `fit`, after the approach-A loop and before writing `fit.json`:
```python
    if cfg.fit.run_inversion:
        import xarray as xr

        from nhf_runoff_bias.accumulate import accumulation_matrix
        from nhf_runoff_bias.inversion import fit_inversion
        from nhf_runoff_bias.models import PREDICTORS

        fab = fabric_ref(cfg.fabric.fabric_json)
        target = xr.open_dataset(cfg.target_nc)
        hru_ids = target[fab["id_col"]].values
        gh = pd.read_parquet(rd / "gage_hrus.parquet")
        A, gids = accumulation_matrix(gh, hru_ids)
        preds = [p for p in PREDICTORS if p in hru.columns and hru[p].notna().any()]
        X = hru.set_index("hru_id").reindex(hru_ids)[preds].to_numpy(float)
        inv_results = []
        for member in cfg.members:
            m_mean = target[member].mean("time").values.astype(float)
            for tier in ("A", "B"):
                keep = set(tiers.loc[tiers["tier"].isin({"A", "B"} if tier == "A" else {"B"}), "poi_gage_id"])
                bm = bias_df[(bias_df["member"] == member) & (bias_df["exclusion_reason"] == "") & bias_df["poi_gage_id"].isin(keep)]
                q_obs = bm.set_index("poi_gage_id")["q_obs_mean"]
                res = fit_inversion(A, gids, q_obs, m_mean, X, preds, cfg.fit.ridge_lambda, cfg.fit.bootstrap)
                f = res.factor_hru.assign(hru_id=hru_ids, member=member, tier=tier)
                factors.append(f[["member", "tier", "hru_id", "factor_inv", "factor_inv_se"]])
                inv_results.append({"member": member, "tier": tier, "n_gages": res.n_gages, "rmse_log": res.rmse_log, "beta": res.beta})
                log.info("inversion %s/%s: n=%d rmse_log=%.3f", member, tier, res.n_gages, res.rmse_log)
        results.append({"inversion": inv_results})
```
and change the factor concat to an outer merge so A and C columns share rows:
```python
    fa = pd.concat([f for f in factors if "factor_ols" in f.columns], ignore_index=True)
    fc = [f for f in factors if "factor_inv" in f.columns]
    out = fa.merge(pd.concat(fc, ignore_index=True), on=["member", "tier", "hru_id"], how="outer") if fc else fa
    write_parquet(out, rd / "factor_hru.parquet", meta)
```

- [ ] **Step 6: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add src/nhf_runoff_bias/inversion.py src/nhf_runoff_bias/cli.py tests/test_inversion.py
pixi run git commit -m "feat: approach C inversion through the accumulation matrix (flag-gated)"
```

---

### Task 11: End-to-end synthetic pipeline test and Oregon integration run

**Files:**
- Test: `tests/test_pipeline_e2e.py`, `tests/test_integration_oregon.py`
- Modify: `README.md` (results section placeholder replaced by the run recipe + expected artifact list)

- [ ] **Step 1: Synthetic end-to-end test through the CLI functions**

`tests/test_pipeline_e2e.py`:
```python
import numpy as np
import pandas as pd
import pytest

from nhf_runoff_bias import cli as c
from tests.test_network import _write_cfg


def test_all_stages_run_on_synthetic(synthetic_config, tmp_path, monkeypatch):
    # covariates need real external inputs; stub the three builders with per-HRU constants
    def fake_terrain(_dir, hru):
        return pd.DataFrame({"hru_id": hru.hru_id, "elev_m": hru.hru_id * 100.0, "slope_deg": 1.0, "northness": 0.0, "eastness": 1.0, "soil_type": 1, "soil_moist_max": 3.0})

    def fake_climate(*_a, **_k):
        return pd.DataFrame({"hru_id": [1, 2, 3, 4, 5], "prcp_mm_yr": 1000.0, "pet_mm_yr": 500.0, "aridity": 0.5, "snow_frac": 0.2, "tmean_c": 8.0})

    def fake_geology(*_a, **_k):
        return pd.DataFrame({"hru_id": [1, 2, 3, 4, 5], "log10_k_perm": -12.0, "litho_cover_frac": 1.0, "litho_class": "basalt"})

    monkeypatch.setattr("nhf_runoff_bias.covariates.terrain.build_terrain", fake_terrain)
    monkeypatch.setattr("nhf_runoff_bias.covariates.climate.build_climate", fake_climate)
    monkeypatch.setattr("nhf_runoff_bias.covariates.geology.build_geology", fake_geology)

    cfgp = _write_cfg(synthetic_config, tmp_path)
    c.gages(config=cfgp)
    c.network(config=cfgp)
    c.covariates(config=cfgp)
    c.bias(config=cfgp)
    rd = synthetic_config.run_dir
    for name in ("gages_monthly", "gage_meta", "gage_hrus", "gage_network", "covariates_hru", "covariates_basin", "bias", "gage_tiers"):
        assert (rd / f"{name}.parquet").exists(), name
    assert (rd / "accumulated.nc").exists()
    b = pd.read_parquet(rd / "bias.parquet").set_index(["poi_gage_id", "member"])
    assert b.loc[("G1", "m_a"), "log_ratio"] == pytest.approx(np.log(2.0))
    # fit needs >= 10 gages; assert it fails loudly rather than silently on 2
    with pytest.raises(ValueError, match="only 1 gages|only 2 gages"):
        c.fit(config=cfgp)
```

- [ ] **Step 2: Oregon integration test**

`tests/test_integration_oregon.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

OR = Path("/caldera/hovenweep/projects/usgs/water/impd/nhgf/or-spatial-targets")


@pytest.mark.integration
@pytest.mark.skipif(not OR.exists(), reason="Oregon inputs not on this host")
def test_oregon_network_and_gages():
    from nhf_runoff_bias.config import load_config
    from nhf_runoff_bias.gages import load_gages
    from nhf_runoff_bias.network import build_network

    cfg = load_config(Path("config.example.yml"))
    net = build_network(cfg.fabric)
    gn = net.gage_network()
    assert len(gn) == 851
    assert (gn.exclusion_reason == "").sum() >= 840
    monthly, meta = load_gages(cfg.gages)
    assert (meta.exclusion_reason == "").sum() >= 700
    assert meta.exclusion_reason.eq("derived_series").sum() == 24
    assert meta.exclusion_reason.eq("placeholder_id").sum() == 2
    usable = meta[meta.exclusion_reason == ""].poi_gage_id
    in_fabric = gn[gn.exclusion_reason == ""].poi_gage_id
    assert len(set(usable) & set(in_fabric)) >= 640
```

- [ ] **Step 3: Run the real pipeline on Oregon**

```bash
cp config.example.yml config.yml
pixi run gages -- --config config.yml
pixi run network -- --config config.yml
pixi run covariates -- --config config.yml      # overlay of 16 814 HRUs vs 578 MB lithology: expect 10-30 min
pixi run bias -- --config config.yml
pixi run fit -- --config config.yml
pixi run -e dev test-integration
```
Record in the commit message: tier counts from the `bias` log line (expected order of magnitude: A ≈ 180, B ≈ 50) and the OLS / RF LOO R² per member from the `fit` log line. If tier A is far below 100, inspect `gage_tiers.parquet` `tier_reason` value counts before proceeding — `area_mismatch` dominating means the published-area join or the km² conversion is wrong, `no_published_area` dominating means the CSV join key needs zero-padding.

- [ ] **Step 4: Commit**

```bash
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add tests/test_pipeline_e2e.py tests/test_integration_oregon.py README.md
pixi run git commit -m "test: synthetic end-to-end + Oregon integration; first full Oregon run"
```

---

### Task 12: Report notebooks

**Files:**
- Create: `notebooks/_helpers.py`, `notebooks/01_gage_screening.ipynb`, `notebooks/02_bias_by_member.ipynb`, `notebooks/03_bias_vs_covariates.ipynb`, `notebooks/04_factor_maps.ipynb`
- Test: `tests/test_notebook_helpers.py`

**Interfaces:**
- `_helpers.load_run(run_dir: Path) -> dict[str, pd.DataFrame | xr.Dataset]` returns every artifact keyed by stem; `_helpers.gage_points(gage_meta) -> gpd.GeoDataFrame` (EPSG:4326); `_helpers.hru_polygons(cfg) -> gpd.GeoDataFrame`; `_helpers.member_colors(members) -> dict[str, str]` (fixed colorblind-safe order: `#0072B2`, `#E69F00`, `#009E73`, `#CC79A7`).

- [ ] **Step 1: Write the failing helper test**

`tests/test_notebook_helpers.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import pandas as pd  # noqa: E402

from _helpers import load_run, member_colors  # noqa: E402
from nhf_runoff_bias.provenance import write_parquet  # noqa: E402


def test_load_run_reads_every_parquet(tmp_path):
    write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "bias.parquet", {})
    write_parquet(pd.DataFrame({"b": [2]}), tmp_path / "gage_meta.parquet", {})
    r = load_run(tmp_path)
    assert set(r) == {"bias", "gage_meta"}


def test_member_colors_fixed_order():
    c = member_colors(["era5_land", "gldas_noah_v21_monthly", "mwbm_climgrid"])
    assert list(c.values()) == ["#0072B2", "#E69F00", "#009E73"]
```

- [ ] **Step 2: Implement `notebooks/_helpers.py`**

```python
"""Shared loaders and styling for the report notebooks."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

from nhf_runoff_bias.config import Config

_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def load_run(run_dir: Path) -> dict:
    out = {p.stem: pd.read_parquet(p) for p in sorted(Path(run_dir).glob("*.parquet"))}
    nc = Path(run_dir) / "accumulated.nc"
    if nc.exists():
        out["accumulated"] = xr.open_dataset(nc)
    return out


def gage_points(gage_meta: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(gage_meta, geometry=gpd.points_from_xy(gage_meta.lon, gage_meta.lat), crs="EPSG:4326")


def hru_polygons(cfg: Config) -> gpd.GeoDataFrame:
    g = gpd.read_file(cfg.fabric.gpkg, layer=cfg.fabric.layers["hru"])
    return g.rename(columns={cfg.fabric.cols["hru_id"]: "hru_id"})[["hru_id", "geometry"]]


def member_colors(members: list[str]) -> dict[str, str]:
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(members)}
```

- [ ] **Step 3: Build the four notebooks** (each: first cell sets `RUN = Path("runs/oregon")` and `CFG = load_config("config.yml")`; markdown cell stating what the figure answers; bullets not tables for any summary).

1. `01_gage_screening.ipynb`: map of all 851 POIs colored by `tier`; bar of `tier_reason` counts; histogram of `fabric_area_km2 / published_area_km2`.
2. `02_bias_by_member.ipynb`: per member, map of `log_ratio` at tier-A gages (diverging, centred 0, shared scale); boxplot of `log_ratio` by member; 12-month climatological ratio curves grouped by `litho_class` quartile of `log10_k_perm` (this is the Safeeq-type geology split); three representative gage hydrographs (obs vs three accumulated members).
3. `03_bias_vs_covariates.ipynb`: scatter grid `log_ratio` vs each predictor per member with LOWESS; SHAP mean-|value| bars from `fit.json`; PCA loadings heatmap; LOO predicted-vs-observed with R².
4. `04_factor_maps.ipynb`: per member, choropleth of `factor_ols` and `factor_rf` (and `factor_inv` when present) on one pooled log scale; map of `factor_se`; histogram of factors with the tier-A gage `exp(log_ratio)` overlaid.

- [ ] **Step 4: Execute all four against `runs/oregon`, strip outputs (nbstripout via pre-commit), commit**

```bash
pixi run -e dev jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
pixi run -e dev fmt && pixi run -e dev lint && pixi run -e dev test
git add notebooks/_helpers.py notebooks/0*.ipynb tests/test_notebook_helpers.py
pixi run git commit -m "docs: report notebooks (screening, bias by member, covariates, factor maps)"
```

---

## Self-review notes

- **Spec coverage:** §3 inputs/defects → Task 3 exclusions + Task 8 tiers; §4 contract → Task 1 (provenance, config-driven ids); §5 components → Tasks 2–10 one module each, `report/` → Task 12; §6 CLI + tiers → Tasks 2/3/8/9 + tier table in Task 8; §7 gates → Task 2 (cycle, unknown `to_segment`), Task 4 (fabric hash), Task 5 (join coverage), Task 8/9 (n gages recorded), Task 6 (weights fingerprint); §8 tests → every task; §9 resolved items → SGMC + Gleeson in Task 7, 1980–2020 window in config.
- **Deviation from spec, deliberate:** climate covariates come from ClimGrid (`prcp`, `pet`, `snow`, `tmean`) rather than Daymet, because the Daymet aggregation on disk holds only SWE; the window is therefore 1980–2020. Snow fraction is ClimGrid's own `snow / prcp` partition rather than a `tmean ≤ 0 °C` rule. Update spec §3/§5 when this plan is accepted.
- **Type consistency:** `gage_hrus` columns `poi_gage_id, hru_id` used identically in Tasks 2/4/5/8/10; `FabricConfig.cols` keys identical in Tasks 1/2/7/12; `factor_hru` column names `factor_ols/factor_rf/factor_se/factor_inv/factor_inv_se` consistent across Tasks 9/10/12.
- **Review Focus:** items 1–5 each have a test in the named task.
