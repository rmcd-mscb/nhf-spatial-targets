# Target driver

`nhf_spatial_targets.targets._driver` is the generic `build(adapter, project)` that consumes a [`TargetAdapter`](targets-adapter.md) and runs the shared target-build pipeline. Has two paths — **single-shot** (e.g. runoff, AET, recharge, soil moisture — one read-combine-write) and **year-chunked** (SWE; daily targets too large to hold the full period in memory).

SCA's year-chunked builder currently owns its year loop directly rather than using the generic year-chunked path, due to a monkeypatch-target constraint in `tests/test_targets_sca.py`. See [issue #230](https://github.com/rmcd-mscb/nhf-spatial-targets/issues/230) for the re-unification plan.

::: nhf_spatial_targets.targets._driver
    options:
      show_source: true
      heading_level: 2
