# Catalog

`nhf_spatial_targets.catalog` is the **single Python interface** to `catalog/sources.yml` and `catalog/variables.yml`. Every other module reads source metadata through these functions — direct YAML reads outside `catalog.py` are not allowed (`CLAUDE.md` §Data & Catalog Conventions).

::: nhf_spatial_targets.catalog
    options:
      show_source: true
      heading_level: 2
