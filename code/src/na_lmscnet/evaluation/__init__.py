"""Evaluation and reporting utilities for reproducible experiments."""

from na_lmscnet.evaluation.baseline_report import (
    BaselineReportError,
    generate_baseline_report,
)
from na_lmscnet.evaluation.core_ablation_multiseed_report import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_SEEDS,
    CoreAblationMultiseedReportError,
    formal_contribution_decision,
    generate_core_ablation_multiseed_report,
    paired_hierarchical_bootstrap,
    source_equivalence_audit,
)
from na_lmscnet.evaluation.core_ablation_report import (
    CoreAblationReportError,
    generate_core_ablation_report,
    validate_split_audit_pair,
)
from na_lmscnet.evaluation.efficiency import EfficiencyError, count_macs, count_parameters
from na_lmscnet.evaluation.experiment_freeze import (
    ExperimentFreezeError,
    audit_freeze_manifest,
    authorize_frozen_test_dataset,
    build_freeze_manifest,
    consume_test_authorization,
    sha256_file,
    update_consumption_marker,
    write_manifest_atomic,
)
from na_lmscnet.evaluation.na_lmscnet_report import NALMSCNetReportError, generate_na_lmscnet_report
from na_lmscnet.evaluation.radioml_2018_independence import (
    RadioML2018IndependenceAuditError,
    audit_radioml_2018_test_independence,
)
from na_lmscnet.evaluation.revision_namespace import (
    RevisionNamespaceError,
    initialize_revision_namespace,
)
from na_lmscnet.evaluation.snr_auxiliary_ablation_report import (
    SNRAuxiliaryAblationReportError,
    generate_snr_auxiliary_ablation_report,
    generate_snr_auxiliary_multiseed_report,
)

__all__ = [
    "BaselineReportError",
    "CoreAblationReportError",
    "CoreAblationMultiseedReportError",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPECTED_SEEDS",
    "EfficiencyError",
    "ExperimentFreezeError",
    "RadioML2018IndependenceAuditError",
    "RevisionNamespaceError",
    "audit_freeze_manifest",
    "audit_radioml_2018_test_independence",
    "authorize_frozen_test_dataset",
    "build_freeze_manifest",
    "count_macs",
    "count_parameters",
    "consume_test_authorization",
    "generate_baseline_report",
    "generate_core_ablation_report",
    "validate_split_audit_pair",
    "formal_contribution_decision",
    "generate_core_ablation_multiseed_report",
    "paired_hierarchical_bootstrap",
    "source_equivalence_audit",
    "generate_na_lmscnet_report",
    "initialize_revision_namespace",
    "NALMSCNetReportError",
    "SNRAuxiliaryAblationReportError",
    "generate_snr_auxiliary_ablation_report",
    "generate_snr_auxiliary_multiseed_report",
    "sha256_file",
    "update_consumption_marker",
    "write_manifest_atomic",
]
