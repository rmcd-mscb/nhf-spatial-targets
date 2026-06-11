# Upgrade config

`nhf_spatial_targets.upgrade_config` is the operator-facing drift report behind
`nhf-targets maintenance check-config`. Existing projects don't pick up new optional
config features added to the init template after they were created; this module
lists what's missing and prints the literal commented block to paste.

**Report-only** — it never mutates an operator's `config.yml`. Each tracked
addition is an `OptionalConfigFeature` entry in `OPTIONAL_CONFIG_FEATURES`;
adding a new optional config key means appending an entry here (`CLAUDE.md`
§Config schema additions).

::: nhf_spatial_targets.upgrade_config
    options:
      show_source: true
      heading_level: 2
