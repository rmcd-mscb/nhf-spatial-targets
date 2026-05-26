# SourceAdapter

`nhf_spatial_targets.aggregate._adapter.SourceAdapter` is the declarative plugin for gridded sources aggregated via gdptools. Each source under `src/nhf_spatial_targets/aggregate/` declares a module-level `ADAPTER = SourceAdapter(...)` instance, and the shared driver (`aggregate/_driver.py`) consumes it generically.

See [Contributing · Adding a new source](../contributing.md#adding-a-new-source) for the file-by-file checklist when adding a new gridded source.

::: nhf_spatial_targets.aggregate._adapter
    options:
      show_source: true
      heading_level: 2
