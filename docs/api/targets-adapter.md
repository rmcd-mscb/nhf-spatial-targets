# TargetAdapter

`nhf_spatial_targets.targets._adapter.TargetAdapter` is the declarative plugin for calibration target builders (PR #219). Each builder under `src/nhf_spatial_targets/targets/` declares a module-level `ADAPTER = TargetAdapter(...)` instance and delegates a thin `build(project)` to the generic [target driver](targets-driver.md).

The pattern mirrors [`SourceAdapter`](aggregate-adapter.md) on the aggregator side.

See [Contributing · Adding a new target](../contributing.md#adding-a-new-target) for the file-by-file checklist when adding a new target.

::: nhf_spatial_targets.targets._adapter
    options:
      show_source: true
      heading_level: 2
