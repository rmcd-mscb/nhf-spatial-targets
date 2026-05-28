"""Release tooling for the ScienceBase data-release feature."""

from __future__ import annotations

from nhf_spatial_targets.release._models import (
    CHECKSUM_FILES,
    RESERVED_METADATA_FILES,
    DistributionKind,
    FabricChildPlan,
    FileEntry,
    ReleaseError,
    ReleasePayload,
    SourceChildPlan,
    UmbrellaPlan,
)
from nhf_spatial_targets.release.build import (
    BuildResult,
    build_all,
    build_fabric_child,
    build_source_child,
    build_umbrella,
)
from nhf_spatial_targets.release.checksums import (
    ChecksumMismatch,
    compute_checksums,
    verify_csv,
)
from nhf_spatial_targets.release.config import (
    load_release_config,
    load_release_yml,
    scaffold_release_yml,
    validate_release_config,
)
from nhf_spatial_targets.release.defaults import (
    REQUIRED_POPULATED_PATHS,
    load_release_defaults,
    missing_release_defaults,
    require_populated_release_defaults,
    validate_release_defaults,
)

# Only the result types are re-exported here, NOT the ``dry_run`` function:
# re-exporting it would bind ``release.dry_run`` to the function and shadow the
# ``release.dry_run`` submodule. Import the function as
# ``from nhf_spatial_targets.release.dry_run import dry_run``.
from nhf_spatial_targets.release.dry_run import (
    DryRunItem,
    DryRunReport,
)
from nhf_spatial_targets.release.fgdc import render_fgdc
from nhf_spatial_targets.release.iso import render_iso
from nhf_spatial_targets.release.lineage import (
    STEP_KINDS,
    InputFileEntry,
    OutputFileEntry,
    StepKind,
    StepRecord,
    append_step,
    build_step_record,
    input_file_entry,
    merge_source_and_append_step,
    output_file_entry,
    sha256_file,
)
from nhf_spatial_targets.release.mcf import (
    MCF_VERSION,
    build_fabric_mcf,
    build_source_mcf,
    build_umbrella_mcf,
)
from nhf_spatial_targets.release.payload import (
    plan_fabric_child,
    plan_source_child,
    plan_umbrella,
    resolve_fabric_label,
    sources_used,
    stage_all,
    stage_fabric_child,
    stage_source_child,
    stage_umbrella,
)
from nhf_spatial_targets.release.publish import (
    PreflightError,
    PublishResult,
    ScopeDiff,
    StaleRegistryError,
    diff_local_vs_remote,
    publish_fabric_child,
    publish_source_child,
    publish_umbrella,
)
from nhf_spatial_targets.release.readme import render_readme
from nhf_spatial_targets.release.rebuild import rebuild_lineage
from nhf_spatial_targets.release.registry import (
    CHILD_FIELDS,
    DEFAULT_REGISTRY_PATH,
    UMBRELLA_FIELDS,
    get_fabric,
    get_source,
    get_umbrella,
    load_registry,
    put_fabric,
    put_source,
    put_umbrella,
)
from nhf_spatial_targets.release.sb_client import (
    RETRYABLE_STATUS_CODES,
    SbClient,
    SbClientError,
    UploadResult,
    is_retryable,
    remote_file_checksum,
)
from nhf_spatial_targets.release.validate_xml import (
    MpResult,
    mp_available,
    validate_xml_file,
    validate_xml_string,
)

__all__ = [
    "CHILD_FIELDS",
    "DEFAULT_REGISTRY_PATH",
    "MCF_VERSION",
    "REQUIRED_POPULATED_PATHS",
    "RETRYABLE_STATUS_CODES",
    "STEP_KINDS",
    "CHECKSUM_FILES",
    "RESERVED_METADATA_FILES",
    "UMBRELLA_FIELDS",
    "BuildResult",
    "ChecksumMismatch",
    "DistributionKind",
    "DryRunItem",
    "DryRunReport",
    "FabricChildPlan",
    "FileEntry",
    "InputFileEntry",
    "MpResult",
    "OutputFileEntry",
    "PreflightError",
    "PublishResult",
    "ReleaseError",
    "ReleasePayload",
    "SbClient",
    "SbClientError",
    "ScopeDiff",
    "SourceChildPlan",
    "StaleRegistryError",
    "StepKind",
    "StepRecord",
    "UmbrellaPlan",
    "UploadResult",
    "append_step",
    "build_all",
    "build_fabric_child",
    "build_fabric_mcf",
    "build_source_child",
    "build_source_mcf",
    "build_step_record",
    "build_umbrella",
    "build_umbrella_mcf",
    "compute_checksums",
    "diff_local_vs_remote",
    "get_fabric",
    "get_source",
    "get_umbrella",
    "input_file_entry",
    "is_retryable",
    "load_registry",
    "load_release_config",
    "load_release_defaults",
    "load_release_yml",
    "merge_source_and_append_step",
    "missing_release_defaults",
    "mp_available",
    "output_file_entry",
    "plan_fabric_child",
    "plan_source_child",
    "plan_umbrella",
    "publish_fabric_child",
    "publish_source_child",
    "publish_umbrella",
    "put_fabric",
    "put_source",
    "put_umbrella",
    "rebuild_lineage",
    "remote_file_checksum",
    "render_fgdc",
    "render_iso",
    "render_readme",
    "require_populated_release_defaults",
    "resolve_fabric_label",
    "scaffold_release_yml",
    "sha256_file",
    "sources_used",
    "stage_all",
    "stage_fabric_child",
    "stage_source_child",
    "stage_umbrella",
    "validate_release_config",
    "validate_release_defaults",
    "validate_xml_file",
    "validate_xml_string",
    "verify_csv",
]
