# NetCDF I/O

`nhf_spatial_targets.io_nc` owns the **canonical NetCDF encoding policy** (chunking + zlib + pinned-time encoding) and the **atomic write pattern** (tempfile + `os.replace`). Every pipeline-written NC routes through `build_encoding` + `atomic_to_netcdf` — direct `ds.to_netcdf(...)` calls are not allowed (`CLAUDE.md` §Data & Catalog Conventions, [`docs/architecture/nc-encoding-policy.md`](../architecture/nc-encoding-policy.md)).

::: nhf_spatial_targets.io_nc
    options:
      show_source: true
      heading_level: 2
