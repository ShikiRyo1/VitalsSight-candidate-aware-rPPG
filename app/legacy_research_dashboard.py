from __future__ import annotations

from datetime import datetime
from html import escape
import json
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlencode
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.baselines.respiration import (
    RespirationSignal,
    first_frame,
    optical_flow_signals,
    window_slices,
)
from src.data.archive_io import extract_zip_member
from src.data.video_io import get_video_metadata
from src.product.adult_hr_live_topk_bridge import build_live_adult_hr_topk_bridge
from src.product.adult_hr_mvp import AdultHRMVPConfig, run_adult_hr_video
from src.product.adult_hr_reliability_guard import score_product_table_with_reliability_guard
from src.selection.topk_bridge import TopKBridgeConfig
from src.signal.rr_validation import estimate_rr_half_rate_validated
from src.vision.body_roi import body_aware_respiration_rois, candidate_respiration_rois, draw_rois


AIR_RESULTS = PROJECT / "experiments" / "phase_d_t39_air_rr_half_validation_results.csv"
CAMBRIDGE_RESULTS = PROJECT / "experiments" / "phase_d_t39_cambridge_rr_trend_results.csv"
CAMBRIDGE_T58_WINDOWS = PROJECT / "experiments" / "t58_cambridge_temporal_alignment_window_results.csv"
CAMBRIDGE_T58_POLICY_SUMMARY = PROJECT / "experiments" / "t58_cambridge_temporal_alignment_policy_summary.csv"
CAMBRIDGE_T58_STABILITY_CI = PROJECT / "experiments" / "t58_stability_cambridge_subject_bootstrap_ci.csv"
CAMBRIDGE_T58_SUBJECT_STABILITY = PROJECT / "experiments" / "t58_stability_cambridge_subject_policy_summary.csv"
CAMBRIDGE_T87_WINDOWS = PROJECT / "experiments" / "t87_cambridge_adaptive_harmonic_window_results.csv"
CAMBRIDGE_T87_POLICY_SUMMARY = PROJECT / "experiments" / "t87_cambridge_adaptive_harmonic_policy_summary.csv"
CAMBRIDGE_T87_BOOTSTRAP_CI = PROJECT / "experiments" / "t87_cambridge_adaptive_harmonic_bootstrap_ci.csv"
CAMBRIDGE_T87_SUBJECT_SUMMARY = PROJECT / "experiments" / "t87_cambridge_adaptive_harmonic_subject_summary.csv"
CAMBRIDGE_T87_WARMUP_SUMMARY = PROJECT / "experiments" / "t87_cambridge_adaptive_harmonic_warmup_summary.csv"
CAMBRIDGE_T92_POLICY_SUMMARY = PROJECT / "experiments" / "t92_actionability_layer_policy_summary.csv"
CAMBRIDGE_T92_EVENT_DETAILS = PROJECT / "experiments" / "t92_actionability_layer_event_details.csv"
CAMBRIDGE_T92_WINDOW_ALERTS = PROJECT / "experiments" / "t92_actionability_layer_window_alerts.csv"
CAMBRIDGE_T94_WINDOWS = PROJECT / "experiments" / "t94_latent_state_rr_tracker_window_results.csv"
CAMBRIDGE_T94_POLICY_SUMMARY = PROJECT / "experiments" / "t94_latent_state_rr_tracker_policy_summary.csv"
CAMBRIDGE_T94_BOOTSTRAP_CI = PROJECT / "experiments" / "t94_latent_state_rr_tracker_bootstrap_ci.csv"
CAMBRIDGE_T95_WINDOWS = PROJECT / "experiments" / "t95_t94_loso_validation_window_results.csv"
CAMBRIDGE_T95_POLICY_SUMMARY = PROJECT / "experiments" / "t95_t94_loso_validation_policy_summary.csv"
CAMBRIDGE_T95_FOLD_SUMMARY = PROJECT / "experiments" / "t95_t94_loso_validation_fold_summary.csv"
CAMBRIDGE_T95_SELECTION_COUNTS = PROJECT / "experiments" / "t95_t94_loso_validation_selection_counts.csv"
CAMBRIDGE_T95_BOOTSTRAP_CI = PROJECT / "experiments" / "t95_t94_loso_validation_bootstrap_ci.csv"
CAMBRIDGE_T98_RISK_WINDOWS = PROJECT / "experiments" / "t98_risk_calibration_window_results.csv"
CAMBRIDGE_T98_RISK_SUMMARY = PROJECT / "experiments" / "t98_risk_calibration_summary.csv"
CAMBRIDGE_T98_INTERVAL_WINDOWS = PROJECT / "experiments" / "t98_conformal_interval_window_results.csv"
CAMBRIDGE_T98_INTERVAL_SUMMARY = PROJECT / "experiments" / "t98_conformal_interval_summary.csv"
CAMBRIDGE_T107_WINDOWS = PROJECT / "experiments" / "t107_fallback_selector_interval_window_results.csv"
CAMBRIDGE_T107_POLICY_SUMMARY = PROJECT / "experiments" / "t107_fallback_selector_policy_summary.csv"
CAMBRIDGE_T107_REVIEW_AUDIT = PROJECT / "experiments" / "t107_fallback_selector_reviewed_window_audit.csv"
CAMBRIDGE_T107_SELECTED_FALLBACKS = PROJECT / "experiments" / "t107_fallback_selector_selected_fallbacks.csv"
CAMBRIDGE_T107_INTERVAL_CALIBRATION = PROJECT / "experiments" / "t107_fallback_selector_interval_calibration.csv"
CAMBRIDGE_T107_ROUTE_RESIDUALS = PROJECT / "experiments" / "t107_fallback_selector_route_residual_summary.csv"
CAMBRIDGE_T111_CASE_AUDIT = PROJECT / "experiments" / "t111_route_reliability_gate_case_audit.csv"
CAMBRIDGE_T111_POLICY_SUMMARY = PROJECT / "experiments" / "t111_route_reliability_gate_policy_summary.csv"
CAMBRIDGE_T111_SUMMARY_JSON = PROJECT / "experiments" / "t111_route_reliability_gate_summary.json"
CAMBRIDGE_T115_CASE_AUDIT = PROJECT / "experiments" / "t115_broad_stress_guard_case_audit.csv"
CAMBRIDGE_T115_POLICY_SUMMARY = PROJECT / "experiments" / "t115_broad_stress_guard_policy_summary.csv"
CAMBRIDGE_T115_PERTURBATION_SUMMARY = PROJECT / "experiments" / "t115_broad_stress_guard_perturbation_summary.csv"
CAMBRIDGE_T115_RESIDUAL_AUDIT = PROJECT / "experiments" / "t115_broad_stress_guard_residual_audit.csv"
CAMBRIDGE_T115_SUMMARY_JSON = PROJECT / "experiments" / "t115_broad_stress_guard_summary.json"
CAMBRIDGE_T120_VARIANT_AUDIT = PROJECT / "experiments" / "t120_subject_aware_route_risk_variant_audit.csv"
CAMBRIDGE_T120_POLICY_SUMMARY = PROJECT / "experiments" / "t120_subject_aware_route_risk_policy_summary.csv"
CAMBRIDGE_T120_EPISODE_CONTEXT = PROJECT / "experiments" / "t120_subject_aware_route_risk_episode_context.csv"
CAMBRIDGE_T120_LOSO_SUMMARY = PROJECT / "experiments" / "t120_subject_aware_route_risk_loso_summary.csv"
CAMBRIDGE_T120_SUMMARY_JSON = PROJECT / "experiments" / "t120_subject_aware_route_risk_summary.json"
CAMBRIDGE_T123_VARIANT_AUDIT = PROJECT / "experiments" / "t123_causal_current_safety_modes_variant_audit.csv"
CAMBRIDGE_T123_MODE_CONFIG = PROJECT / "experiments" / "t123_causal_current_safety_modes_config.csv"
CAMBRIDGE_T123_POLICY_SUMMARY = PROJECT / "experiments" / "t123_causal_current_safety_modes_policy_summary.csv"
CAMBRIDGE_T123_COLD_START_SUMMARY = PROJECT / "experiments" / "t123_causal_current_safety_modes_cold_start_summary.csv"
CAMBRIDGE_T123_SUBJECT_SUMMARY = PROJECT / "experiments" / "t123_causal_current_safety_modes_subject_summary.csv"
CAMBRIDGE_T123_SUMMARY_JSON = PROJECT / "experiments" / "t123_causal_current_safety_modes_summary.json"
CAMBRIDGE_T125_VARIANT_AUDIT = PROJECT / "experiments" / "t125_blocked_surface_refinement_variant_audit.csv"
CAMBRIDGE_T125_POLICY_SUMMARY = PROJECT / "experiments" / "t125_blocked_surface_refinement_policy_summary.csv"
CAMBRIDGE_T125_RECOVERY_SUMMARY = PROJECT / "experiments" / "t125_blocked_surface_recovery_summary.csv"
CAMBRIDGE_T125_LEARNED_REFINER = PROJECT / "experiments" / "t125_learned_refiner_transfer_summary.csv"
CAMBRIDGE_T125_SUMMARY_JSON = PROJECT / "experiments" / "t125_blocked_surface_refinement_summary.json"
ADULT_T162_EXPLANATION_PANEL = PROJECT / "experiments" / "t162_product_explanation_panel.csv"
ADULT_T162_STATUS_SUMMARY = PROJECT / "experiments" / "t162_explanation_status_summary.csv"
ADULT_T162_CLAIM_CHECKLIST = PROJECT / "experiments" / "t162_claim_readiness_checklist.csv"
ADULT_T162_PROTOCOL_JSON = PROJECT / "experiments" / "t162_locked_external_validation_protocol.json"
ADULT_T162_SUMMARY_JSON = PROJECT / "experiments" / "t162_product_explanation_and_protocol_summary.json"
ADULT_T161_POLICY_SUMMARY = PROJECT / "experiments" / "t161_policy_summary.csv"
ADULT_T163_QA_CHECKLIST = PROJECT / "experiments" / "t163_dashboard_qa_checklist.csv"
ADULT_T163_RUN_ORDER = PROJECT / "experiments" / "t163_external_validation_run_order.csv"
ADULT_T163_GATE_MATRIX = PROJECT / "experiments" / "t163_validation_gate_matrix.csv"
ADULT_T163_SUMMARY_JSON = PROJECT / "experiments" / "t163_dashboard_qa_validation_order_summary.json"
ADULT_T164_POLICY_SUMMARY = PROJECT / "experiments" / "t164_ubfc_locked_policy_summary.csv"
ADULT_T164_REPRO_CHECK = PROJECT / "experiments" / "t164_ubfc_reproducibility_check.csv"
ADULT_T164_LEAKAGE_AUDIT = PROJECT / "experiments" / "t164_ubfc_leakage_audit.csv"
ADULT_T164_DATA_COMPLETENESS = PROJECT / "experiments" / "t164_ubfc_data_completeness.csv"
ADULT_T164_BOOTSTRAP = PROJECT / "experiments" / "t164_ubfc_t150_vs_t160_bootstrap.csv"
ADULT_T164_SUMMARY_JSON = PROJECT / "experiments" / "t164_ubfc_frozen_replay_leakage_audit_summary.json"
ADULT_T165_READINESS = PROJECT / "experiments" / "t165_clean_external_dataset_readiness.csv"
ADULT_T165_ACCESS_ACTIONS = PROJECT / "experiments" / "t165_dataset_access_actions.csv"
ADULT_T165_LOCKED_PLAN = PROJECT / "experiments" / "t165_locked_validation_plan.csv"
ADULT_T165_SUMMARY_JSON = PROJECT / "experiments" / "t165_clean_external_dataset_gate_summary.json"
ADULT_T217_PRODUCT_TABLE = PROJECT / "experiments" / "t217_adult_hr_topk_bridge_product_table.csv"
ADULT_T217_API_EXAMPLES = PROJECT / "experiments" / "t217_adult_hr_topk_bridge_api_examples.json"
ADULT_T348_PRODUCT_TABLE = PROJECT / "experiments" / "t348_adult_hr_mode_product_table.csv"
ADULT_T348_API_EXAMPLES = PROJECT / "experiments" / "t348_adult_hr_mode_api_examples.json"
ADULT_T357_PRODUCT_TABLE = PROJECT / "experiments" / "t357_experimental_frozen_recovery_product_table.csv"
ADULT_T357_BRANCH_SUMMARY = PROJECT / "experiments" / "t357_experimental_frozen_recovery_branch_summary.csv"
ADULT_T357_QA_CHECKS = PROJECT / "experiments" / "t357_experimental_frozen_recovery_qa_checks.csv"
ADULT_T357_API_EXAMPLES = PROJECT / "experiments" / "t357_experimental_frozen_recovery_api_examples.json"
ADULT_T357_SUMMARY_JSON = PROJECT / "experiments" / "t357_experimental_frozen_recovery_summary.json"
ADULT_T382_PRODUCT_TABLE = PROJECT / "experiments" / "t382_t380_product_policy_table.csv"
ADULT_T382_BRANCH_SUMMARY = PROJECT / "experiments" / "t382_t380_product_policy_branch_summary.csv"
ADULT_T382_QA_CHECKS = PROJECT / "experiments" / "t382_t380_product_policy_qa_checks.csv"
ADULT_T382_API_EXAMPLES = PROJECT / "experiments" / "t382_t380_product_policy_api_examples.json"
ADULT_T382_SUMMARY_JSON = PROJECT / "experiments" / "t382_t380_product_policy_candidate_summary.json"
ADULT_T478_SUMMARY_JSON = PROJECT / "experiments" / "t478_product_evidence_bundle_summary.json"
ADULT_T478_METRIC_CARDS = PROJECT / "experiments" / "t478_product_metric_cards.csv"
ADULT_T478_CLAIM_GATE = PROJECT / "experiments" / "t478_product_claim_gate.csv"
ADULT_T481_SUMMARY_JSON = PROJECT / "experiments" / "t481_product_policy_router_summary.json"
ADULT_T481_ROUTER_TABLE = PROJECT / "experiments" / "t481_product_router_table.csv"
ADULT_T481_CLAIM_GATE = PROJECT / "experiments" / "t481_product_router_claim_gate.csv"
ADULT_T481_API_EXAMPLES = PROJECT / "experiments" / "t481_product_router_api_examples.json"
ADULT_T482_SUMMARY_JSON = PROJECT / "experiments" / "t482_external_fairness_router_figure_summary.json"
ADULT_T482_QA = PROJECT / "experiments" / "t482_external_fairness_router_figure_qa.csv"
ADULT_T482_FIGURE_PNG = PROJECT / "output" / "t482_figures" / "t482_external_fairness_router_evidence.png"
ADULT_T482_SOURCE_DATA = PROJECT / "experiments" / "t482_external_fairness_router_figure_source_data.csv"
ADULT_T485_SUMMARY_JSON = PROJECT / "experiments" / "t485_ubfc_phys_selective_index_summary.json"
ADULT_T485_SIGNAL_INDEX = PROJECT / "experiments" / "t485_ubfc_phys_signal_trial_index.csv"
ADULT_T485_CLAIM_GATE = PROJECT / "experiments" / "t485_ubfc_phys_claim_gate.csv"
ADULT_T486_SUMMARY_JSON = PROJECT / "experiments" / "t486_mr_nirp_selective_route_preflight_summary.json"
ADULT_T486_CONDITION_INDEX = PROJECT / "experiments" / "t486_mr_nirp_condition_index.csv"
ADULT_T486_CLAIM_GATE = PROJECT / "experiments" / "t486_mr_nirp_claim_gate.csv"
ADULT_T487_SUMMARY_JSON = PROJECT / "experiments" / "t487_cmu_fairness_replay_summary.json"
ADULT_T487_METRICS = PROJECT / "experiments" / "t487_cmu_fairness_replay_metrics.csv"
ADULT_T487_DELTA = PROJECT / "experiments" / "t487_cmu_fairness_replay_group_roi_delta.csv"
ADULT_T487_CLAIM_GATE = PROJECT / "experiments" / "t487_cmu_fairness_replay_claim_gate.csv"
ADULT_T488_SUMMARY_JSON = PROJECT / "experiments" / "t488_external_domain_locked_subset_plan_summary.json"
ADULT_T488_SUBSET = PROJECT / "experiments" / "t488_external_domain_locked_subset_manifest.csv"
ADULT_T488_BUDGET = PROJECT / "experiments" / "t488_external_domain_extraction_budget.csv"
ADULT_T488_CLAIM_GATE = PROJECT / "experiments" / "t488_external_domain_claim_gate.csv"
ADULT_T489_SUMMARY_JSON = PROJECT / "experiments" / "t489_post_domain_gpu_evidence_lock_summary.json"
ADULT_T489_CLAIM_GATE = PROJECT / "experiments" / "t489_post_domain_gpu_evidence_lock_claim_gate.csv"
ADULT_T491_SUMMARY_JSON = PROJECT / "experiments" / "t491_selected_domain_compact_trace_cache_summary.json"
ADULT_T491_TRACE_INDEX = PROJECT / "experiments" / "t491_selected_domain_trace_cache_index.csv"
ADULT_T491_CLAIM_GATE = PROJECT / "experiments" / "t491_selected_domain_trace_cache_claim_gate.csv"
ADULT_T492_SUMMARY_JSON = PROJECT / "experiments" / "t492_selected_domain_artifact_gate_summary.json"
ADULT_T492_MR_DECISIONS = PROJECT / "experiments" / "t492_selected_domain_mr_policy_decisions.csv"
ADULT_T492_CLAIM_GATE = PROJECT / "experiments" / "t492_selected_domain_artifact_gate_claim_gate.csv"
ADULT_T493_SUMMARY_JSON = PROJECT / "experiments" / "t493_selected_domain_roi_trace_cache_summary.json"
ADULT_T493_TRACE_INDEX = PROJECT / "experiments" / "t493_selected_domain_roi_trace_cache_index.csv"
ADULT_T493_CLAIM_GATE = PROJECT / "experiments" / "t493_selected_domain_roi_trace_cache_claim_gate.csv"
ADULT_T494_SUMMARY_JSON = PROJECT / "experiments" / "t494_roi_candidate_evaluation_summary.json"
ADULT_T494_DECISIONS = PROJECT / "experiments" / "t494_roi_policy_decisions.csv"
ADULT_T494_METRICS = PROJECT / "experiments" / "t494_roi_dataset_metrics.csv"
ADULT_T494_CLAIM_GATE = PROJECT / "experiments" / "t494_roi_candidate_evaluation_claim_gate.csv"
ADULT_T495_SUMMARY_JSON = PROJECT / "experiments" / "t495_method_aware_roi_selector_summary.json"
ADULT_T495_DECISIONS = PROJECT / "experiments" / "t495_method_aware_roi_policy_decisions.csv"
ADULT_T495_METRICS = PROJECT / "experiments" / "t495_method_aware_roi_dataset_metrics.csv"
ADULT_T495_DELTA = PROJECT / "experiments" / "t495_vs_t494_delta.csv"
ADULT_T495_CLAIM_GATE = PROJECT / "experiments" / "t495_method_aware_roi_selector_claim_gate.csv"
ADULT_T497_SUMMARY_JSON = PROJECT / "experiments" / "t497_ubfc_phys_all_method_aware_summary.json"
ADULT_T497_DECISIONS = PROJECT / "experiments" / "t497_ubfc_phys_all_method_aware_decisions.csv"
ADULT_T497_METRICS = PROJECT / "experiments" / "t497_ubfc_phys_all_method_aware_metrics.csv"
ADULT_T497_CLAIM_GATE = PROJECT / "experiments" / "t497_ubfc_phys_all_method_aware_claim_gate.csv"
ADULT_T498_SUMMARY_JSON = PROJECT / "experiments" / "t498_context_aware_t3_guard_summary.json"
ADULT_T498_DECISIONS = PROJECT / "experiments" / "t498_context_aware_t3_guard_decisions.csv"
ADULT_T498_METRICS = PROJECT / "experiments" / "t498_context_aware_t3_guard_metrics.csv"
ADULT_T498_DELTA = PROJECT / "experiments" / "t498_vs_t497_delta.csv"
ADULT_T498_CLAIM_GATE = PROJECT / "experiments" / "t498_context_aware_t3_guard_claim_gate.csv"
ADULT_T504_SUMMARY_JSON = PROJECT / "experiments" / "t504_mediapipe_dual_range_t3_selector_summary.json"
ADULT_T504_DECISIONS = PROJECT / "experiments" / "t504_mediapipe_dual_range_t3_selector_decisions.csv"
ADULT_T504_METRICS = PROJECT / "experiments" / "t504_mediapipe_dual_range_t3_selector_metrics.csv"
ADULT_T504_DELTA = PROJECT / "experiments" / "t504_vs_t503_selector_delta.csv"
ADULT_T504_TAXONOMY = PROJECT / "experiments" / "t504_mediapipe_dual_range_failure_taxonomy.csv"
ADULT_T504_CLAIM_GATE = PROJECT / "experiments" / "t504_mediapipe_dual_range_t3_selector_claim_gate.csv"
ADULT_T505_SUMMARY_JSON = PROJECT / "experiments" / "t505_dual_range_generalization_audit_summary.json"
ADULT_T505_DELTA = PROJECT / "experiments" / "t505_vs_t498_generalization_delta.csv"
ADULT_T505_CLAIM_GATE = PROJECT / "experiments" / "t505_dual_range_generalization_claim_gate.csv"
ADULT_T506_SUMMARY_JSON = PROJECT / "experiments" / "t506_route_aware_product_policy_summary.json"
ADULT_T506_PRODUCT_TABLE = PROJECT / "experiments" / "t506_route_aware_product_policy_table.csv"
ADULT_T506_PRODUCT_SUMMARY = PROJECT / "experiments" / "t506_route_aware_product_policy_summary.csv"
ADULT_T506_DELTA = PROJECT / "experiments" / "t506_vs_t498_route_aware_product_delta.csv"
ADULT_T506_API_EXAMPLES = PROJECT / "experiments" / "t506_route_aware_product_api_examples.json"
ADULT_T506_CLAIM_GATE = PROJECT / "experiments" / "t506_route_aware_product_policy_claim_gate.csv"
ADULT_T508_SUMMARY_JSON = PROJECT / "experiments" / "t508_route_aware_gpu_feature_audit_summary.json"
ADULT_T508_DELTA = PROJECT / "experiments" / "t508_vs_t480_route_feature_delta.csv"
ADULT_T508_CLAIM_GATE = PROJECT / "experiments" / "t508_route_aware_gpu_feature_claim_gate.csv"
ADULT_T509_SUMMARY_JSON = PROJECT / "experiments" / "t509_route_aware_gpu_selector_protocol_summary.json"
ADULT_T509_BOOTSTRAP = PROJECT / "experiments" / "t509_route_aware_gpu_selector_bootstrap_ci.csv"
ADULT_T509_ENDPOINT_GATES = PROJECT / "experiments" / "t509_route_aware_gpu_selector_endpoint_gates.csv"
ADULT_T510_SUMMARY_JSON = PROJECT / "experiments" / "t510_route_aware_threshold_recovery_summary.json"
ADULT_T510_DELTA = PROJECT / "experiments" / "t510_vs_t480_t508_threshold_recovery_delta.csv"
ADULT_T510_CLAIM_GATE = PROJECT / "experiments" / "t510_route_aware_threshold_recovery_claim_gate.csv"
ADULT_T511_SUMMARY_JSON = PROJECT / "experiments" / "t511_experimental_learned_selector_product_summary.json"
ADULT_T511_PRODUCT_TABLE = PROJECT / "experiments" / "t511_experimental_learned_selector_product_table.csv"
ADULT_T511_PRODUCT_SUMMARY = PROJECT / "experiments" / "t511_experimental_learned_selector_product_summary.csv"
ADULT_T511_API_EXAMPLES = PROJECT / "experiments" / "t511_experimental_learned_selector_api_examples.json"
ADULT_T511_CLAIM_GATE = PROJECT / "experiments" / "t511_experimental_learned_selector_product_claim_gate.csv"
ADULT_T517_SUMMARY_JSON = PROJECT / "experiments" / "t517_route_moe_evidence_synthesis_summary.json"
ADULT_T517_ROUTE_TABLE = PROJECT / "experiments" / "t517_route_moe_policy_table.csv"
ADULT_T517_CLAIM_GATE = PROJECT / "experiments" / "t517_route_moe_claim_gate.csv"
ADULT_T518_SUMMARY_JSON = PROJECT / "experiments" / "t518_route_moe_product_policy_summary.json"
ADULT_T518_PRODUCT_POLICY = PROJECT / "experiments" / "t518_route_moe_product_policy_table.csv"
ADULT_T518_API_EXAMPLES = PROJECT / "experiments" / "t518_route_moe_api_examples.json"
ADULT_T518_CLAIM_GATE = PROJECT / "experiments" / "t518_route_moe_product_claim_gate.csv"
ADULT_T527_SUMMARY_JSON = PROJECT / "experiments" / "t527_product_end_to_end_demo_qa_summary.json"
ADULT_T527_DEMO_CASES = PROJECT / "experiments" / "t527_product_e2e_demo_cases.csv"
ADULT_T527_API_PACKETS = PROJECT / "experiments" / "t527_product_e2e_api_packets.json"
ADULT_T527_REPORT_CARDS = PROJECT / "experiments" / "t527_product_e2e_report_cards.csv"
ADULT_T527_QA_CHECKS = PROJECT / "experiments" / "t527_product_e2e_qa_checks.csv"
ADULT_CURRENT_POLICY = "T157_T153_current_product"
ADULT_T161_FULL_POLICY = "T161_full_frozen_t160_rule"
BENCHMARK_TABLE = PROJECT / "experiments" / "phase_d_rr_benchmark_table.csv"
QUALITY_ABLATION_SUMMARY = PROJECT / "experiments" / "quality_gated_ablation_summary.csv"
T57_POLICY_SUMMARY = PROJECT / "experiments" / "t57_air_refusal_calibration_policy_summary.csv"
T57_POLICY_CONFIG = PROJECT / "configs" / "product_refusal_policy_t57.json"
SAMPLE_INDEX = PROJECT / "experiments" / "sample_index.csv"
CACHE_DIR = PROJECT / "experiments" / "cache" / "dashboard"
DEFAULT_METHOD = "optical_flow_y_body_aware_half_validated"
REFUSAL_POLICY = "flow_decision_refusal_no_raw_kept"
REFUSAL_REASON = "raw_kept_unvalidated_peak"
T58_BEST_POLICY = "aligned_harmonic_depth_plausible_median_smooth3"
T58_LEGACY_POLICY = "legacy_t39_depth_right_chest"
T87_BASE_POLICY = "t58_base_plausible_median_smooth3"
T87_DEFAULT_POLICY = "depth_harmonic_blend075_q90_smooth3"
T87_Q90_POLICY = "depth_harmonic_q90_smooth3"
T87_MAX_POLICY = "depth_harmonic_max_smooth3"
T87_RESEARCH_COPY = "Research evidence only. Use this trend as a review aid, not as a diagnostic reading."
T87_HIGH_RR_COPY = "This policy is designed to reduce missed high breathing-rate episodes."
T87_BOOTSTRAP_BOUNDARY_COPY = (
    "High-RR mitigation is supported in this first-pass public Cambridge analysis, "
    "but all-window bootstrap intervals still cross zero."
)
T87_SUBJECT_CAUTION_COPY = (
    "Use caution for this subject: the high breathing-rate error is still elevated. "
    "Continue normal observation and do not rely on this research output as the only safety check."
)
T87_SUBJECT_IMPROVED_COPY = (
    "This subject's high-RR error is lower under the selected T87 policy, but the output remains research evidence."
)
T87_WARMUP_BOUNDARY_COPY = "The warm-up adaptation result uses reference labels and is not a deployable mode yet."
T92_DEFAULT_RULE = "persistent_2_windows"
T92_BASELINE_RULE = "raw_window_threshold"
T92_ACTIONABILITY_COPY = "Public-data workflow evidence only. This is not clinical alarm validation."
T94_BALANCED_POLICY = "latent_tail_state_viterbi_balanced"
T94_EQUAL_POLICY = "latent_state_viterbi_equal"
T94_HIGH_RECALL_POLICY = "latent_tail_state_viterbi_high_recall"
T95_COMBINED_POLICY = "t95_loso_combined_tail_selection"
T95_MAE_POLICY = "t95_loso_mae_selection"
T95_HIGH_RR_POLICY = "t95_loso_high_rr_selection"
T98_DEFAULT_CALIBRATION_POLICY = "risk_target_0.30_q90_gap"
T98_DEFAULT_INTERVAL_POLICY = "t97_upper_gap_latent_q75"
T98_DEFAULT_ALPHA = 0.1
T98_WIDE_INTERVAL_BPM = 30.0
T107_ROUTE90_POLICY = "t107_trusted_countercluster_selector_route_conformal_90"
T107_SHIFTED_POLICY = "t107_trusted_countercluster_selector_shifted_t104_interval"
T107_ROUTE80_POLICY = "t107_trusted_countercluster_selector_route_conformal_80"
T106_SHIFTED_POLICY = "t106_green_gap_equal_tail_fallback_shifted_t104_interval"
T105_REVIEW_POLICY = "t105_combined_source_validity_review"
T94_LATENT_COPY = (
    "T94 treats RR as a hidden physiological trajectory, not independent window peaks. "
    "Research evidence only."
)
T95_LOSO_COPY = (
    "T95 selects the latent policy without seeing the held-out subject. "
    "This supports high-RR tail mitigation, but it is still small public-data evidence."
)
T98_INTERVAL_COPY = (
    "T98 adds calibrated uncertainty intervals and source-risk status on top of the T95 LOSO tracker. "
    "It is a review aid, not a clinical alarm or diagnostic interval."
)
T98_RISK_COPY = (
    "The risk flag estimates whether the current video-source condition is likely to produce a large RR error. "
    "Flagged windows should be routed to human review or a stronger sensor path."
)
T107_ROUTE_COPY = (
    "T107 routes source-validity-reviewed windows through a trusted counter-cluster fallback selector, then reports "
    "route-specific uncertainty intervals. This is product workflow evidence, not a diagnostic vital-sign device."
)
T107_REVIEW_COPY = (
    "If the selector cannot find a label-free trusted fallback, the window remains under review instead of releasing "
    "a fragile RR number."
)
T111_GATE_COPY = (
    "T111 adds a synthetic-stress-tested route-reliability safety gate after T107. High-to-low fallback release is "
    "gated; low-to-high correction remains review-only. This is not external or clinical validation."
)
T111_REVIEW_COPY = (
    "A corrected fallback is released only when the route passes the T111 safety gate. Low-anchor stress cases stay "
    "review-only until a separate direction-specific model is validated."
)
T115_GATE_COPY = (
    "T115 adds a broad-stress residual guard after the T114A/T111 route gate stack. It blocks mid-gap fallback "
    "ambiguity using label-free route features; this is Cambridge-derived synthetic stress evidence, not clinical "
    "or external validation."
)
T115_REVIEW_COPY = (
    "A corrected fallback is released only when it passes source-validity routing, route-reliability gating, and "
    "the T115 broad-stress ambiguity guard. Guarded residual cases stay review-only with an auditable block reason."
)
T120_GATE_COPY = (
    "T120 adds a subject-aware route-risk calibration overlay after T119. It uses episode tail-risk rate to tighten "
    "automatic release when a subject/stress episode is unstable. This is internal simulation stress evidence, not "
    "external or clinical validation."
)
T120_REVIEW_COPY = (
    "When the episode tail-risk rate is high, borderline route-risk scores are review-only even if the single-route "
    "T119 score would have released them. This exposes subject-level instability instead of hiding it behind a clean RR number."
)
T123_GATE_COPY = (
    "T123 converts the T122 causal-current episode-risk evidence into configurable safety modes for eldercare, hospital watch, "
    "and infant-monitoring research use. It is internal simulation stress evidence, not external or clinical validation."
)
T123_REVIEW_COPY = (
    "Safety modes change how much uncertainty the product is allowed to release automatically. Higher-caution modes keep more "
    "cold-start or unstable current-episode windows review-only instead of presenting a fragile RR number."
)
T125_GATE_COPY = (
    "T125 adds an experimental blocked-surface risk-floor refinement after T123. It uses an auditable 0.15 route-risk floor "
    "to recover selected hospital/infant review-only windows while preserving zero unsafe releases in the internal stress audit."
)
T125_REVIEW_COPY = (
    "T125 is a conservative burden-reduction candidate: recovered windows are shown only with before/after counts, recovered "
    "surface evidence, and the explicit boundary that this is not external or clinical validation."
)
T58_RESEARCH_COPY = (
    "T58 neonatal mode uses reference-overlap alignment and harmonic-aware correction. "
    "It is research evidence only, not a diagnostic output."
)
T92_RULE_DISPLAY_NAMES = {
    "raw_window_threshold": "Raw threshold",
    "persistent_2_windows": "Persistent 2 windows",
    "severe_or_persistent": "Severe or persistent",
}
POLICY_DISPLAY_NAMES = {
    "t57_calibrated_confidence_refusal": "T57 calibrated confidence refusal",
    "flow_decision_refusal_no_raw_kept": "T56 decision-trace refusal",
    "display_without_refusal": "Display without refusal",
}
POLICY_MODE_BY_DISPLAY = {label: key for key, label in POLICY_DISPLAY_NAMES.items()}


st.set_page_config(
    page_title="VitalsSight Adult HR",
    page_icon="VS",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vs-text: #1f2933;
            --vs-muted: #60707d;
            --vs-border: #d9e2e7;
            --vs-panel: #ffffff;
            --vs-soft: #f6f8f9;
            --vs-teal: #2f7d6d;
            --vs-green: #6c8f5f;
            --vs-amber: #b7791f;
            --vs-red: #a84d3a;
        }
        .stApp {
            background: #ffffff;
            color: var(--vs-text);
        }
        [data-testid="stSidebar"] {
            background: #f6f8f9;
            border-right: 1px solid var(--vs-border);
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--vs-text);
        }
        h1 {
            font-size: 30px;
            line-height: 1.15;
            margin-bottom: 4px;
        }
        h2 {
            font-size: 20px;
            line-height: 1.2;
            margin-top: 8px;
        }
        h3 {
            font-size: 16px;
            line-height: 1.25;
        }
        div[data-testid="stMetric"] {
            background: var(--vs-panel);
            border: 1px solid var(--vs-border);
            border-radius: 8px;
            padding: 12px 14px;
            min-height: 92px;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--vs-muted);
            font-size: 12px;
        }
        div[data-testid="stMetricValue"] {
            color: var(--vs-text);
            font-size: 24px;
        }
        .vs-status {
            border: 1px solid var(--vs-border);
            border-radius: 8px;
            padding: 10px 12px;
            background: #ffffff;
            font-size: 13px;
            color: var(--vs-muted);
        }
        .vs-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 9px;
            margin-right: 6px;
            font-size: 12px;
            border: 1px solid var(--vs-border);
            color: var(--vs-muted);
            background: #ffffff;
        }
        .vs-chip-good {
            color: var(--vs-teal);
            border-color: rgba(47, 125, 109, 0.35);
            background: rgba(47, 125, 109, 0.07);
        }
        .vs-chip-warn {
            color: var(--vs-amber);
            border-color: rgba(183, 121, 31, 0.35);
            background: rgba(183, 121, 31, 0.08);
        }
        .vs-chip-bad {
            color: var(--vs-red);
            border-color: rgba(168, 77, 58, 0.35);
            background: rgba(168, 77, 58, 0.08);
        }
        .block-container {
            padding-top: 24px;
            padding-bottom: 36px;
        }
        button, input, textarea, select {
            font-size: 14px !important;
        }
        .vs-kv {
            border: 1px solid var(--vs-border);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
            margin-top: 12px;
            margin-bottom: 12px;
        }
        .vs-kv-row {
            display: grid;
            grid-template-columns: minmax(96px, 0.45fr) minmax(0, 1fr);
            border-bottom: 1px solid var(--vs-border);
        }
        .vs-kv-row:last-child {
            border-bottom: 0;
        }
        .vs-kv-cell {
            padding: 8px 10px;
            font-size: 13px;
            line-height: 1.35;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .vs-kv-key {
            color: var(--vs-muted);
            background: var(--vs-soft);
        }
        .vs-console-panel {
            border: 1px solid var(--vs-border);
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 132px;
        }
        .vs-console-title {
            font-size: 13px;
            color: var(--vs-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .vs-console-value {
            font-size: 20px;
            font-weight: 650;
            color: var(--vs-text);
            line-height: 1.25;
        }
        .vs-console-note {
            font-size: 13px;
            color: var(--vs-muted);
            line-height: 1.35;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_air_results() -> pd.DataFrame:
    return pd.read_csv(AIR_RESULTS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_results() -> pd.DataFrame:
    return pd.read_csv(CAMBRIDGE_RESULTS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t58_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T58_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T58_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t58_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T58_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T58_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t58_stability_ci() -> pd.DataFrame:
    if not CAMBRIDGE_T58_STABILITY_CI.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T58_STABILITY_CI, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t58_subject_stability() -> pd.DataFrame:
    if not CAMBRIDGE_T58_SUBJECT_STABILITY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T58_SUBJECT_STABILITY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t87_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T87_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T87_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t87_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T87_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T87_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t87_bootstrap_ci() -> pd.DataFrame:
    if not CAMBRIDGE_T87_BOOTSTRAP_CI.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T87_BOOTSTRAP_CI, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t87_subject_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T87_SUBJECT_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T87_SUBJECT_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t87_warmup_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T87_WARMUP_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T87_WARMUP_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t92_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T92_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T92_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t92_event_details() -> pd.DataFrame:
    if not CAMBRIDGE_T92_EVENT_DETAILS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T92_EVENT_DETAILS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t92_window_alerts() -> pd.DataFrame:
    if not CAMBRIDGE_T92_WINDOW_ALERTS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T92_WINDOW_ALERTS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t94_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T94_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T94_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t94_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T94_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T94_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t94_bootstrap_ci() -> pd.DataFrame:
    if not CAMBRIDGE_T94_BOOTSTRAP_CI.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T94_BOOTSTRAP_CI, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t95_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T95_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T95_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t95_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T95_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T95_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t95_fold_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T95_FOLD_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T95_FOLD_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t95_selection_counts() -> pd.DataFrame:
    if not CAMBRIDGE_T95_SELECTION_COUNTS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T95_SELECTION_COUNTS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t95_bootstrap_ci() -> pd.DataFrame:
    if not CAMBRIDGE_T95_BOOTSTRAP_CI.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T95_BOOTSTRAP_CI, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t98_risk_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T98_RISK_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T98_RISK_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t98_risk_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T98_RISK_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T98_RISK_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t98_interval_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T98_INTERVAL_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T98_INTERVAL_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t98_interval_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T98_INTERVAL_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T98_INTERVAL_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t107_windows() -> pd.DataFrame:
    if not CAMBRIDGE_T107_WINDOWS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T107_WINDOWS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t107_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T107_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T107_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t107_review_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T107_REVIEW_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T107_REVIEW_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t107_selected_fallbacks() -> pd.DataFrame:
    if not CAMBRIDGE_T107_SELECTED_FALLBACKS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T107_SELECTED_FALLBACKS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t107_interval_calibration() -> pd.DataFrame:
    if not CAMBRIDGE_T107_INTERVAL_CALIBRATION.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T107_INTERVAL_CALIBRATION, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t107_route_residuals() -> pd.DataFrame:
    if not CAMBRIDGE_T107_ROUTE_RESIDUALS.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T107_ROUTE_RESIDUALS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t111_case_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T111_CASE_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T111_CASE_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t111_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T111_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T111_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t111_summary_json() -> dict[str, object]:
    if not CAMBRIDGE_T111_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(CAMBRIDGE_T111_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_cambridge_t115_case_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T115_CASE_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T115_CASE_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t115_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T115_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T115_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t115_perturbation_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T115_PERTURBATION_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T115_PERTURBATION_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t115_residual_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T115_RESIDUAL_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T115_RESIDUAL_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t115_summary_json() -> dict[str, object]:
    if not CAMBRIDGE_T115_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(CAMBRIDGE_T115_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_cambridge_t120_variant_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T120_VARIANT_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T120_VARIANT_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t120_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T120_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T120_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t120_episode_context() -> pd.DataFrame:
    if not CAMBRIDGE_T120_EPISODE_CONTEXT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T120_EPISODE_CONTEXT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t120_loso_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T120_LOSO_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T120_LOSO_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t120_summary_json() -> dict[str, object]:
    if not CAMBRIDGE_T120_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(CAMBRIDGE_T120_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_cambridge_t123_variant_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T123_VARIANT_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T123_VARIANT_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t123_mode_config() -> pd.DataFrame:
    if not CAMBRIDGE_T123_MODE_CONFIG.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T123_MODE_CONFIG, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t123_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T123_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T123_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t123_cold_start_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T123_COLD_START_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T123_COLD_START_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t123_subject_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T123_SUBJECT_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T123_SUBJECT_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t123_summary_json() -> dict[str, object]:
    if not CAMBRIDGE_T123_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(CAMBRIDGE_T123_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_cambridge_t125_variant_audit() -> pd.DataFrame:
    if not CAMBRIDGE_T125_VARIANT_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T125_VARIANT_AUDIT, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t125_policy_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T125_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T125_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t125_recovery_summary() -> pd.DataFrame:
    if not CAMBRIDGE_T125_RECOVERY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T125_RECOVERY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t125_learned_refiner() -> pd.DataFrame:
    if not CAMBRIDGE_T125_LEARNED_REFINER.exists():
        return pd.DataFrame()
    return pd.read_csv(CAMBRIDGE_T125_LEARNED_REFINER, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_cambridge_t125_summary_json() -> dict[str, object]:
    if not CAMBRIDGE_T125_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(CAMBRIDGE_T125_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_adult_t162_explanation_panel() -> pd.DataFrame:
    if not ADULT_T162_EXPLANATION_PANEL.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T162_EXPLANATION_PANEL, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t162_status_summary() -> pd.DataFrame:
    if not ADULT_T162_STATUS_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T162_STATUS_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t162_claim_checklist() -> pd.DataFrame:
    if not ADULT_T162_CLAIM_CHECKLIST.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T162_CLAIM_CHECKLIST, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t162_protocol_json() -> dict[str, object]:
    if not ADULT_T162_PROTOCOL_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T162_PROTOCOL_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_adult_t162_summary_json() -> dict[str, object]:
    if not ADULT_T162_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T162_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_adult_t161_policy_summary() -> pd.DataFrame:
    if not ADULT_T161_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T161_POLICY_SUMMARY, encoding="utf-8-sig")


def load_adult_t163_qa_checklist() -> pd.DataFrame:
    if not ADULT_T163_QA_CHECKLIST.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T163_QA_CHECKLIST, encoding="utf-8-sig")


def load_adult_t163_run_order() -> pd.DataFrame:
    if not ADULT_T163_RUN_ORDER.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T163_RUN_ORDER, encoding="utf-8-sig")


def load_adult_t163_gate_matrix() -> pd.DataFrame:
    if not ADULT_T163_GATE_MATRIX.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T163_GATE_MATRIX, encoding="utf-8-sig")


def load_adult_t163_summary_json() -> dict[str, object]:
    if not ADULT_T163_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T163_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_adult_t164_policy_summary() -> pd.DataFrame:
    if not ADULT_T164_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T164_POLICY_SUMMARY, encoding="utf-8-sig")


def load_adult_t164_repro_check() -> pd.DataFrame:
    if not ADULT_T164_REPRO_CHECK.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T164_REPRO_CHECK, encoding="utf-8-sig")


def load_adult_t164_leakage_audit() -> pd.DataFrame:
    if not ADULT_T164_LEAKAGE_AUDIT.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T164_LEAKAGE_AUDIT, encoding="utf-8-sig")


def load_adult_t164_data_completeness() -> pd.DataFrame:
    if not ADULT_T164_DATA_COMPLETENESS.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T164_DATA_COMPLETENESS, encoding="utf-8-sig")


def load_adult_t164_bootstrap() -> pd.DataFrame:
    if not ADULT_T164_BOOTSTRAP.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T164_BOOTSTRAP, encoding="utf-8-sig")


def load_adult_t164_summary_json() -> dict[str, object]:
    if not ADULT_T164_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T164_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_adult_t165_readiness() -> pd.DataFrame:
    if not ADULT_T165_READINESS.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T165_READINESS, encoding="utf-8-sig")


def load_adult_t165_access_actions() -> pd.DataFrame:
    if not ADULT_T165_ACCESS_ACTIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T165_ACCESS_ACTIONS, encoding="utf-8-sig")


def load_adult_t165_locked_plan() -> pd.DataFrame:
    if not ADULT_T165_LOCKED_PLAN.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T165_LOCKED_PLAN, encoding="utf-8-sig")


def load_adult_t165_summary_json() -> dict[str, object]:
    if not ADULT_T165_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T165_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_adult_t217_product_table() -> pd.DataFrame:
    if not ADULT_T217_PRODUCT_TABLE.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T217_PRODUCT_TABLE, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t217_api_examples() -> list[dict[str, object]]:
    if not ADULT_T217_API_EXAMPLES.exists():
        return []
    try:
        data = json.loads(ADULT_T217_API_EXAMPLES.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_adult_t348_product_table() -> pd.DataFrame:
    if not ADULT_T348_PRODUCT_TABLE.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T348_PRODUCT_TABLE, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t348_api_examples() -> list[dict[str, object]]:
    if not ADULT_T348_API_EXAMPLES.exists():
        return []
    try:
        data = json.loads(ADULT_T348_API_EXAMPLES.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_adult_t357_product_table() -> pd.DataFrame:
    if not ADULT_T357_PRODUCT_TABLE.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T357_PRODUCT_TABLE, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t357_branch_summary() -> pd.DataFrame:
    if not ADULT_T357_BRANCH_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T357_BRANCH_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t357_qa_checks() -> pd.DataFrame:
    if not ADULT_T357_QA_CHECKS.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T357_QA_CHECKS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t357_api_examples() -> list[dict[str, object]]:
    if not ADULT_T357_API_EXAMPLES.exists():
        return []
    try:
        data = json.loads(ADULT_T357_API_EXAMPLES.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_adult_t357_summary_json() -> dict[str, object]:
    if not ADULT_T357_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T357_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_adult_t382_product_table() -> pd.DataFrame:
    if not ADULT_T382_PRODUCT_TABLE.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T382_PRODUCT_TABLE, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t382_branch_summary() -> pd.DataFrame:
    if not ADULT_T382_BRANCH_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T382_BRANCH_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t382_qa_checks() -> pd.DataFrame:
    if not ADULT_T382_QA_CHECKS.exists():
        return pd.DataFrame()
    return pd.read_csv(ADULT_T382_QA_CHECKS, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t382_api_examples() -> list[dict[str, object]]:
    if not ADULT_T382_API_EXAMPLES.exists():
        return []
    try:
        data = json.loads(ADULT_T382_API_EXAMPLES.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and isinstance(data.get("examples"), list):
        return data["examples"]
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_adult_t382_summary_json() -> dict[str, object]:
    if not ADULT_T382_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(ADULT_T382_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_adult_current_summary_json(path_text: str) -> dict[str, object]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


@st.cache_data(show_spinner=False)
def load_adult_current_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_adult_t481_api_examples() -> list[dict[str, object]]:
    if not ADULT_T481_API_EXAMPLES.exists():
        return []
    try:
        data = json.loads(ADULT_T481_API_EXAMPLES.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and isinstance(data.get("examples"), list):
        return data["examples"]
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(BENCHMARK_TABLE, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_quality_summary() -> pd.DataFrame:
    return pd.read_csv(QUALITY_ABLATION_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_t57_policy_summary() -> pd.DataFrame:
    if not T57_POLICY_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(T57_POLICY_SUMMARY, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_t57_policy_config() -> dict[str, object]:
    if not T57_POLICY_CONFIG.exists():
        return {
            "policy_name": "t57_calibrated_confidence_refusal",
            "policy_id": "confidence_min_0.050",
            "decision_logic": {"threshold": 0.05},
        }
    return json.loads(T57_POLICY_CONFIG.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_sample_index() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_INDEX, encoding="utf-8-sig")


def fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def key_value_panel(rows: list[tuple[str, object]]) -> None:
    rendered_rows = []
    for key, value in rows:
        rendered_rows.append(
            "<div class='vs-kv-row'>"
            f"<div class='vs-kv-cell vs-kv-key'>{escape(str(key))}</div>"
            f"<div class='vs-kv-cell'>{escape(str(value))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='vs-kv'>{''.join(rendered_rows)}</div>", unsafe_allow_html=True)


def header() -> None:
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.title("VitalsSight Adult HR")
    with right:
        st.markdown(
            "<div class='vs-status'><span class='vs-chip vs-chip-good'>Research MVP</span>"
            "<span class='vs-chip'>Not diagnostic</span></div>",
            unsafe_allow_html=True,
        )


def policy_display_name(policy_mode: str) -> str:
    return POLICY_DISPLAY_NAMES.get(policy_mode, policy_mode)


def t92_rule_display_name(rule_id: str) -> str:
    return T92_RULE_DISPLAY_NAMES.get(str(rule_id), str(rule_id))


def sidebar_policy_selector(*, default: str = "t57_calibrated_confidence_refusal") -> str:
    labels = [POLICY_DISPLAY_NAMES[key] for key in POLICY_DISPLAY_NAMES]
    default_label = POLICY_DISPLAY_NAMES.get(default, POLICY_DISPLAY_NAMES["t57_calibrated_confidence_refusal"])
    selected = st.sidebar.selectbox(
        "Product policy",
        labels,
        index=labels.index(default_label) if default_label in labels else 0,
    )
    return POLICY_MODE_BY_DISPLAY[selected]


def prepare_product_output(frame: pd.DataFrame, *, pred_col: str, policy_mode: str) -> pd.DataFrame:
    output = frame.copy()
    candidate = pd.to_numeric(output[pred_col], errors="coerce")
    decisions = output["decision"].astype(str) if "decision" in output.columns else pd.Series("", index=output.index)
    finite_candidate = np.isfinite(candidate)
    threshold = np.nan
    if policy_mode == "display_without_refusal":
        accepted = finite_candidate
        refusal = np.where(accepted, "accepted", "nonfinite_prediction")
    elif policy_mode == REFUSAL_POLICY:
        raw_kept = decisions.eq("raw_kept")
        accepted = finite_candidate & ~raw_kept
        refusal = np.where(
            ~finite_candidate,
            "nonfinite_prediction",
            np.where(raw_kept, REFUSAL_REASON, "accepted"),
        )
    elif policy_mode == "t57_calibrated_confidence_refusal":
        config = load_t57_policy_config()
        logic = config.get("decision_logic", {}) if isinstance(config, dict) else {}
        threshold = float(logic.get("threshold", 0.05)) if isinstance(logic, dict) else 0.05
        confidence = pd.to_numeric(
            output["confidence"] if "confidence" in output.columns else pd.Series(np.nan, index=output.index),
            errors="coerce",
        )
        accepted = finite_candidate & confidence.ge(threshold)
        refusal = np.where(
            ~finite_candidate,
            "nonfinite_prediction",
            np.where(confidence.isna(), "missing_confidence", np.where(accepted, "accepted", "calibrated_low_confidence")),
        )
    else:
        accepted = finite_candidate
        refusal = np.where(accepted, "accepted", "nonfinite_prediction")
    output["candidate_rr_bpm"] = candidate
    output["product_rr_bpm"] = candidate.where(accepted, np.nan)
    output["accepted"] = accepted.astype(bool)
    output["refusal_reason"] = refusal
    output["product_status"] = np.where(output["accepted"], "accepted", "insufficient_signal")
    output["product_policy"] = policy_mode
    output["product_policy_label"] = policy_display_name(policy_mode)
    output["product_policy_threshold"] = threshold
    if "gt_rr_bpm" in output.columns:
        truth = pd.to_numeric(output["gt_rr_bpm"], errors="coerce")
        output["product_abs_error_bpm"] = np.abs(output["product_rr_bpm"] - truth)
    return output


def latest_product_value(frame: pd.DataFrame) -> str:
    if frame.empty or "product_rr_bpm" not in frame.columns:
        return "NA"
    value = pd.to_numeric(pd.Series([frame["product_rr_bpm"].iloc[-1]]), errors="coerce").iloc[0]
    return f"{fmt(value)} BPM" if np.isfinite(value) else "Insufficient"


def product_status_banner(frame: pd.DataFrame) -> None:
    if frame.empty or "accepted" not in frame.columns:
        return
    latest = frame.iloc[-1]
    policy = str(latest.get("product_policy_label", latest.get("product_policy", "")))
    if bool(latest["accepted"]):
        st.success(f"Latest window accepted under {policy}.")
    else:
        st.warning(f"Latest window refused under {policy}: {latest['refusal_reason']}.")


def sidebar_mode() -> str:
    st.sidebar.header("Workspace")
    legacy_modes = {
        "VitalsSight Adult HR Console": "Command Center",
        "AIR Sample Monitor": "Live Scan",
        "Cambridge Trend Monitor": "Reports",
        "Benchmark Overview": "Route-MoE",
        "Adult HR Upload": "Live Scan",
        "Adult HR Bridge Evidence": "Route-MoE",
        "Upload Video Quick Scan": "Live Scan",
    }
    modes = [
        "Command Center",
        "Live Scan",
        "Patients",
        "Alerts",
        "Route-MoE",
        "Reports",
        "Integrations",
    ]
    pending_mode = st.session_state.pop("workspace_mode_pending", None)
    if pending_mode in modes:
        st.session_state["workspace_mode"] = pending_mode
    try:
        query_mode = st.query_params.get("section", "") if hasattr(st, "query_params") else ""
    except Exception:
        query_mode = ""
    if isinstance(query_mode, list):
        query_mode = query_mode[0] if query_mode else ""
    if "workspace_mode" not in st.session_state and query_mode in modes:
        st.session_state["workspace_mode"] = str(query_mode)
    if st.session_state.get("workspace_mode") in legacy_modes:
        st.session_state["workspace_mode"] = legacy_modes[str(st.session_state["workspace_mode"])]
    if st.session_state.get("workspace_mode") not in modes:
        st.session_state["workspace_mode"] = modes[0]
    return st.sidebar.radio(
        "Workspace page",
        modes,
        label_visibility="collapsed",
        key="workspace_mode",
    )


def _console_panel(title: str, value: object, note: str = "") -> None:
    st.markdown(
        "<div class='vs-console-panel'>"
        f"<div class='vs-console-title'>{escape(str(title))}</div>"
        f"<div class='vs-console-value'>{escape(str(value))}</div>"
        f"<div class='vs-console-note'>{escape(str(note))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _load_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


@st.cache_data(show_spinner=False)
def load_adult_console_assets() -> dict[str, object]:
    assets: dict[str, object] = {
        "policy": pd.DataFrame(),
        "demo_cases": pd.DataFrame(),
        "report_cards": pd.DataFrame(),
        "route_metrics": pd.DataFrame(),
        "mvp_actions": pd.DataFrame(),
        "competitor_benchmark": pd.DataFrame(),
        "api": {},
        "qa": {},
        "stats": {},
    }
    policy_path = PROJECT / "experiments" / "t518_route_moe_product_policy_table.csv"
    demo_path = PROJECT / "experiments" / "t527_product_e2e_demo_cases.csv"
    report_path = PROJECT / "experiments" / "t527_product_e2e_report_cards.csv"
    metrics_path = PROJECT / "experiments" / "t521_route_level_locked_metrics.csv"
    actions_path = PROJECT / "experiments" / "t537_mvp_gap_and_action_matrix.csv"
    competitor_path = PROJECT / "experiments" / "t537_competitor_product_benchmark.csv"
    t686_policy_path = PROJECT / "experiments" / "t686_dense_patch_product_policy.csv"
    t686_demo_path = PROJECT / "experiments" / "t686_dense_patch_product_cases.csv"
    t686_report_path = PROJECT / "experiments" / "t686_dense_patch_report_cards.csv"
    t686_metrics_path = PROJECT / "experiments" / "t686_dense_patch_product_route_metrics.csv"
    t686_api_path = PROJECT / "experiments" / "t686_dense_patch_api_examples.json"
    t686_qa_path = PROJECT / "experiments" / "t686_dense_patch_product_qa_summary.json"
    t686_stats_path = PROJECT / "experiments" / "t686_dense_patch_product_evidence_model_summary.json"
    t687_decisions_path = PROJECT / "experiments" / "t687_external_risk_gate_decisions.csv"
    t687_summary_path = PROJECT / "experiments" / "t687_external_risk_gate_refit_summary.json"
    t737_cases_path = PROJECT / "experiments" / "t737_deep_candidate_physio_gate_cases.csv"
    t737_api_path = PROJECT / "experiments" / "t737_deep_candidate_physio_gate_api_packet.json"
    t737_claim_path = PROJECT / "experiments" / "t733_current_evidence_lock.csv"
    t737_main_accuracy_path = PROJECT / "experiments" / "t734_table2_main_accuracy.csv"
    if policy_path.exists():
        assets["policy"] = pd.read_csv(policy_path, encoding="utf-8-sig")
    if demo_path.exists():
        assets["demo_cases"] = pd.read_csv(demo_path, encoding="utf-8-sig")
    if report_path.exists():
        assets["report_cards"] = pd.read_csv(report_path, encoding="utf-8-sig")
    if metrics_path.exists():
        assets["route_metrics"] = pd.read_csv(metrics_path, encoding="utf-8-sig")
    if actions_path.exists():
        assets["mvp_actions"] = pd.read_csv(actions_path, encoding="utf-8-sig")
    if competitor_path.exists():
        assets["competitor_benchmark"] = pd.read_csv(competitor_path, encoding="utf-8-sig")
    assets["api"] = _load_json_file(PROJECT / "experiments" / "t518_route_moe_api_examples.json")
    assets["qa"] = _load_json_file(PROJECT / "experiments" / "t527_product_end_to_end_demo_qa_summary.json")
    assets["stats"] = _load_json_file(PROJECT / "experiments" / "t521_route_level_locked_statistical_validation_summary.json")

    # T686/T687 are the latest evidence-backed adult HR product artifacts.
    # They replace older demo tables when available.
    if t686_policy_path.exists():
        assets["policy"] = pd.read_csv(t686_policy_path, encoding="utf-8-sig")
    if t686_report_path.exists():
        assets["report_cards"] = pd.read_csv(t686_report_path, encoding="utf-8-sig")
    if t686_metrics_path.exists():
        assets["route_metrics"] = pd.read_csv(t686_metrics_path, encoding="utf-8-sig")
    if t686_api_path.exists():
        assets["api"] = _load_json_file(t686_api_path)
    if t686_qa_path.exists():
        assets["qa"] = _load_json_file(t686_qa_path)
    if t686_stats_path.exists():
        assets["stats"] = _load_json_file(t686_stats_path)
    if t686_demo_path.exists():
        demo = pd.read_csv(t686_demo_path, encoding="utf-8-sig")
        if t687_decisions_path.exists():
            risk = pd.read_csv(t687_decisions_path, encoding="utf-8-sig")
            keep_cols = [
                "clip_id",
                "logistic_balanced_oof",
                "t687_release",
                "t687_release_decision",
            ]
            risk = risk[[c for c in keep_cols if c in risk.columns]].copy()
            risk = risk.rename(
                columns={
                    "clip_id": "sample_id",
                    "logistic_balanced_oof": "t687_oof_risk_score",
                }
            )
            demo = demo.merge(risk, on="sample_id", how="left")
            if "t687_release_decision" in demo.columns:
                new_decision = demo["t687_release_decision"].fillna(demo.get("decision", "review"))
                demo["decision"] = new_decision
                release_mask = demo["decision"].astype(str).str.lower().eq("release")
                if "selected_candidate_hr_bpm" in demo.columns:
                    demo.loc[release_mask, "hr_bpm"] = demo.loc[release_mask, "selected_candidate_hr_bpm"]
                demo.loc[~release_mask, "hr_bpm"] = np.nan
                demo.loc[~release_mask & demo["review_reason"].isna(), "review_reason"] = (
                    "T687 out-of-fold risk gate routed this case to review."
                )
                demo["selected_policy"] = "T687 OOF risk gate over dense-patch temporal selector"
                demo["route_id"] = "dense_patch_temporal_semantic_roi_t687_oof_risk_gate"
                demo["release_policy"] = (
                    "OOF risk gate: release only when risk score passes the external-domain operating point; otherwise review"
                )
        assets["demo_cases"] = demo
    if t687_decisions_path.exists():
        assets["risk_gate"] = pd.read_csv(t687_decisions_path, encoding="utf-8-sig")
    else:
        assets["risk_gate"] = pd.DataFrame()
    if t687_summary_path.exists():
        t687_stats = _load_json_file(t687_summary_path)
        t687_stats["passed_claim_gates"] = 1 if t687_stats.get("gate_passed") else 0
        t687_stats["n_claim_gates"] = 1
        assets["stats"] = t687_stats
    # T737 is the current Deep-Candidate PhysioGate evidence bundle.
    # It supersedes earlier dense-patch demo artifacts when present.
    if t737_api_path.exists() and t737_cases_path.exists():
        t737_api = _load_json_file(t737_api_path)
        t737_cases = pd.read_csv(t737_cases_path, encoding="utf-8-sig")
        demo = pd.DataFrame()
        demo["case_id"] = [f"physiogate_{i+1:03d}" for i in range(len(t737_cases))]
        demo["dataset"] = t737_cases.get("dataset", "")
        demo["sample_id"] = t737_cases.get("sample_id", "")
        demo["room"] = [f"PG-{301+i:03d}" for i in range(len(t737_cases))]
        demo["route_id"] = "deep_candidate_physio_gate_candidate_pool"
        demo["selected_policy"] = t737_cases.get("selector_variant", "Deep-Candidate PhysioGate")
        demo["decision"] = t737_cases.get("decision", "review")
        release_mask = demo["decision"].astype(str).str.lower().eq("release")
        demo["hr_bpm"] = pd.to_numeric(t737_cases.get("selected_hr_bpm"), errors="coerce").where(release_mask, np.nan)
        selected_score = pd.to_numeric(t737_cases.get("selected_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        support_score = pd.to_numeric(t737_cases.get("support_count"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        agreement_score = pd.to_numeric(t737_cases.get("agreement10_frac"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        harmonic_risk = pd.to_numeric(t737_cases.get("harmonic_trap_score"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        demo["confidence"] = selected_score
        demo["quality_score"] = (0.35 * support_score + 0.45 * agreement_score + 0.20 * (1.0 - harmonic_risk)).clip(0.0, 1.0)
        demo["confidence_proxy_for_demo"] = demo["confidence"]
        demo["review_reason"] = t737_cases.get("review_reason", "")
        demo["required_input"] = "adult RGB face video, candidate peaks, ROI/patch/deep candidates, physiological risk gate"
        demo["quality_flags"] = np.where(release_mask, "release_gate_passed", "review_gate_required")
        claim_boundary = str(t737_api.get("claim_boundary", "Bounded research product; not clinical diagnosis."))
        demo["claim_boundary"] = claim_boundary
        demo["product_warning"] = "Research product evidence only; not clinical diagnosis or emergency monitoring."
        demo["evidence_gate"] = "T731/T732/T733 bounded Deep-Candidate PhysioGate evidence"
        demo["selected_candidate_hr_bpm"] = pd.to_numeric(t737_cases.get("candidate_hr_bpm"), errors="coerce")
        demo["selected_region"] = "candidate_pool"
        demo["selected_patch_family"] = t737_cases.get("candidate_family", "candidate")
        demo["selected_method"] = t737_cases.get("selector_variant", "selector")
        demo["selected_snr_db"] = np.nan
        demo["selected_peak_support"] = support_score
        demo["selected_median_consistency"] = agreement_score
        demo["selected_score"] = selected_score
        demo["failure_mode"] = np.where(release_mask, "release_gate_passed", "review_gate_required")
        demo["risk_factors_json"] = "{}"
        demo["competing_candidates_json"] = "[]"
        demo["evidence_summary"] = (
            "Deep-Candidate PhysioGate: candidate evidence, harmonic/alias risk, "
            "deep disagreement, and release/review gate."
        )
        assets["demo_cases"] = demo
        assets["report_cards"] = demo.copy()
        assets["api"] = t737_api
        if t737_claim_path.exists():
            assets["policy"] = pd.read_csv(t737_claim_path, encoding="utf-8-sig")
        if t737_main_accuracy_path.exists():
            assets["route_metrics"] = pd.read_csv(t737_main_accuracy_path, encoding="utf-8-sig")
        assets["stats"] = {
            "product_version": t737_api.get("product_version", "t737"),
            "passed_claim_gates": 2,
            "n_claim_gates": 3,
            "headline_metrics": t737_api.get("headline_metrics", {}),
            "claim_boundary": claim_boundary,
        }
    return assets


def adult_hr_console() -> None:
    assets = load_adult_console_assets()
    policy = assets["policy"]
    demo_cases = assets["demo_cases"]
    report_cards = assets["report_cards"]
    route_metrics = assets["route_metrics"]
    mvp_actions = assets["mvp_actions"]
    api = assets["api"]
    qa = assets["qa"]
    stats = assets["stats"]

    st.markdown(
        """
        <style>
        .vs-market-shell {
            margin-top: -10px;
        }
        .vs-market-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 360px;
            gap: 18px;
            align-items: stretch;
            margin-bottom: 18px;
        }
        .vs-hero-main, .vs-live-strip, .vs-bed-card, .vs-alert-row, .vs-detail-card, .vs-pipeline-card {
            border: 1px solid rgba(122, 176, 184, 0.22);
            background: linear-gradient(145deg, rgba(8, 21, 33, 0.96), rgba(12, 35, 48, 0.93));
            color: #e9fbff;
            border-radius: 14px;
            box-shadow: 0 14px 36px rgba(6, 22, 32, 0.18);
        }
        .vs-hero-main {
            padding: 22px 24px;
            min-height: 164px;
            position: relative;
            overflow: hidden;
        }
        .vs-hero-main:after {
            content: "";
            position: absolute;
            right: 24px;
            top: 20px;
            width: 210px;
            height: 90px;
            background: linear-gradient(90deg, transparent, rgba(0, 213, 255, 0.22), transparent);
            clip-path: polygon(0 60%, 10% 60%, 16% 34%, 24% 76%, 32% 44%, 39% 60%, 48% 60%, 54% 25%, 62% 82%, 70% 50%, 78% 64%, 100% 64%, 100% 68%, 78% 68%, 70% 54%, 62% 86%, 54% 29%, 48% 64%, 39% 64%, 32% 48%, 24% 80%, 16% 38%, 10% 64%, 0 64%);
            animation: vs-pulse-wave 3.2s ease-in-out infinite;
        }
        @keyframes vs-pulse-wave {
            0%, 100% { opacity: 0.42; transform: translateX(0); }
            50% { opacity: 0.95; transform: translateX(8px); }
        }
        .vs-hero-title {
            font-size: 34px;
            line-height: 1.05;
            font-weight: 760;
            letter-spacing: 0;
            margin-bottom: 10px;
            max-width: 720px;
        }
        .vs-hero-copy {
            color: #9fb9c2;
            max-width: 760px;
            line-height: 1.45;
            font-size: 14px;
        }
        .vs-live-strip {
            padding: 18px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }
        .vs-live-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 8px;
            background: #38d66b;
            box-shadow: 0 0 0 8px rgba(56, 214, 107, 0.10);
            animation: vs-live 1.8s ease-in-out infinite;
        }
        @keyframes vs-live {
            0%, 100% { transform: scale(0.92); opacity: 0.65; }
            50% { transform: scale(1.1); opacity: 1; }
        }
        .vs-market-kpis {
            display: grid;
            grid-template-columns: repeat(5, minmax(140px, 1fr));
            gap: 12px;
            margin: 14px 0 18px;
        }
        .vs-kpi-dark {
            border: 1px solid #d9e2e7;
            border-radius: 12px;
            background: #ffffff;
            padding: 13px 14px;
            min-height: 96px;
        }
        .vs-kpi-label {
            color: #60707d;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .vs-kpi-value {
            margin-top: 6px;
            color: #152833;
            font-size: 24px;
            font-weight: 720;
            line-height: 1.1;
        }
        .vs-kpi-note {
            margin-top: 6px;
            color: #60707d;
            font-size: 12px;
            line-height: 1.25;
        }
        .vs-command-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.6fr) minmax(360px, 0.9fr);
            gap: 16px;
            align-items: start;
        }
        .vs-section-title {
            font-size: 18px;
            font-weight: 720;
            color: #1f2933;
            margin: 16px 0 9px;
        }
        .vs-bed-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .vs-bed-card {
            padding: 12px;
            min-height: 138px;
            position: relative;
            overflow: hidden;
        }
        .vs-bed-card.release {
            border-color: rgba(69, 213, 111, .62);
            background: linear-gradient(145deg, rgba(8, 44, 30, .97), rgba(13, 70, 48, .91));
        }
        .vs-bed-card.review {
            border-color: rgba(243, 176, 60, .70);
            background: linear-gradient(145deg, rgba(58, 39, 10, .97), rgba(85, 57, 17, .91));
        }
        .vs-bed-card.urgent {
            border-color: rgba(241, 88, 88, .74);
            background: linear-gradient(145deg, rgba(65, 18, 22, .98), rgba(93, 28, 34, .91));
        }
        .vs-bed-card.disconnected {
            border-color: rgba(135, 153, 165, .42);
            background: linear-gradient(145deg, rgba(30, 41, 50, .95), rgba(45, 55, 66, .88));
        }
        .vs-bed-top {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            align-items: center;
        }
        .vs-bed-status {
            font-size: 10px;
            font-weight: 760;
            border: 1px solid rgba(255,255,255,.28);
            border-radius: 5px;
            padding: 2px 6px;
            text-transform: uppercase;
        }
        .vs-bed-room {
            color: #a6c0c9;
            font-size: 12px;
        }
        .vs-bed-main {
            display: flex;
            justify-content: space-between;
            align-items: end;
            margin-top: 18px;
        }
        .vs-bed-id {
            font-size: 26px;
            font-weight: 740;
        }
        .vs-bed-hr {
            font-size: 30px;
            font-weight: 780;
        }
        .vs-bed-sub {
            margin-top: 8px;
            color: #bed2d8;
            font-size: 12px;
            line-height: 1.35;
        }
        .vs-alert-panel {
            border: 1px solid #d9e2e7;
            border-radius: 12px;
            padding: 12px;
            background: #ffffff;
        }
        .vs-alert-row {
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: 10px;
        }
        .vs-alert-row.release {
            background: linear-gradient(145deg, rgba(13, 70, 48, .96), rgba(17, 89, 59, .91));
        }
        .vs-alert-row.review {
            background: linear-gradient(145deg, rgba(75, 51, 15, .97), rgba(103, 68, 21, .92));
        }
        .vs-alert-row.urgent {
            background: linear-gradient(145deg, rgba(82, 22, 28, .98), rgba(115, 32, 40, .92));
        }
        .vs-alert-title {
            display: flex;
            justify-content: space-between;
            font-weight: 720;
            font-size: 13px;
        }
        .vs-alert-note {
            color: #c8d9df;
            font-size: 12px;
            margin-top: 5px;
            line-height: 1.35;
        }
        .vs-detail-grid {
            display: grid;
            grid-template-columns: minmax(260px, .95fr) minmax(0, 1.2fr) minmax(340px, .95fr);
            gap: 14px;
            margin-top: 10px;
        }
        .vs-detail-card, .vs-pipeline-card {
            padding: 14px;
        }
        .vs-scan-frame {
            height: 236px;
            border-radius: 12px;
            border: 1px solid rgba(122, 176, 184, 0.24);
            background:
                radial-gradient(circle at 50% 42%, rgba(70, 214, 146, .20), transparent 21%),
                linear-gradient(160deg, rgba(16, 38, 53, .98), rgba(7, 19, 31, .98));
            position: relative;
            overflow: hidden;
        }
        .vs-scan-frame:before {
            content: "";
            position: absolute;
            inset: 38px 82px 28px;
            border: 2px solid rgba(56, 214, 107, .74);
            border-radius: 28px;
            box-shadow: 0 0 28px rgba(56,214,107,.18);
        }
        .vs-scan-frame:after {
            content: "";
            position: absolute;
            left: 56px;
            right: 56px;
            top: 50%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0,213,255,.9), transparent);
            animation: vs-scanline 2.4s linear infinite;
        }
        @keyframes vs-scanline {
            0% { transform: translateY(-88px); opacity: .2; }
            50% { opacity: 1; }
            100% { transform: translateY(88px); opacity: .2; }
        }
        .vs-face {
            position: absolute;
            left: 50%;
            top: 50%;
            width: 98px;
            height: 132px;
            transform: translate(-50%, -48%);
            border-radius: 48% 48% 42% 42%;
            border: 1px solid rgba(207, 238, 241, .50);
            background: radial-gradient(circle at 45% 32%, rgba(212,237,239,.24), transparent 32%);
        }
        .vs-pipeline-steps {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin: 12px 0;
        }
        .vs-pipeline-step {
            border: 1px solid rgba(122, 176, 184, .28);
            border-radius: 10px;
            padding: 10px;
            min-height: 78px;
            background: rgba(255,255,255,.04);
        }
        .vs-step-label {
            color: #8eb4bf;
            font-size: 11px;
            text-transform: uppercase;
        }
        .vs-step-value {
            color: #e9fbff;
            font-size: 13px;
            font-weight: 700;
            margin-top: 6px;
            line-height: 1.25;
        }
        .vs-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .vs-feature {
            border: 1px solid #d9e2e7;
            border-radius: 10px;
            padding: 12px;
            background: #ffffff;
            min-height: 92px;
        }
        .vs-feature b { color: #1f2933; }
        .vs-feature span { display:block; color:#60707d; font-size:12px; margin-top:6px; line-height:1.35; }
        .vs-guide {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 0 0 14px;
        }
        .vs-guide-step {
            border: 1px solid #d9e2e7;
            border-radius: 10px;
            padding: 10px 12px;
            background: #fbfdfe;
            min-height: 72px;
        }
        .vs-guide-num {
            display: inline-flex;
            width: 22px;
            height: 22px;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: #102837;
            color: #dffcff;
            font-size: 12px;
            font-weight: 740;
            margin-right: 7px;
        }
        .vs-guide-title {
            color: #1f2933;
            font-size: 13px;
            font-weight: 720;
        }
        .vs-guide-copy {
            color: #60707d;
            font-size: 12px;
            line-height: 1.3;
            margin-top: 6px;
        }
        @media (max-width: 1200px) {
            .vs-market-hero, .vs-command-grid, .vs-detail-grid { grid-template-columns: 1fr; }
            .vs-market-kpis { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
            .vs-bed-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .vs-guide { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def selected_value(column: str, default: object = "") -> object:
        return selected_case.get(column, default)

    def decision_kind(row: pd.Series) -> str:
        if str(row.get("route_id", "")).lower().find("fairness") >= 0 or str(row.get("review_reason", "")).lower().find("tachy") >= 0:
            return "urgent"
        decision_text = str(row.get("decision", "review")).lower()
        if decision_text == "release":
            return "release"
        if str(row.get("route_id", "")).lower().find("mr_nirp") >= 0:
            return "disconnected"
        return "review"

    def bed_card(row: pd.Series, idx: int) -> str:
        kind = decision_kind(row)
        status = "released" if kind == "release" else "urgent" if kind == "urgent" else "disconnected" if kind == "disconnected" else "review"
        primary_label = "LIVE HR" if kind == "release" else "URGENT" if kind == "urgent" else "OFFLINE" if kind == "disconnected" else "REVIEW"
        hr = row.get("hr_bpm", "")
        hr_text = f"{fmt(hr, 0)}<span style='font-size:13px;font-weight:500'> bpm</span>" if pd.notna(hr) and str(hr) != "" else "--"
        room = f"A-{301 + idx}"
        confidence = row.get("confidence_proxy_for_demo", "")
        conf_text = f"{fmt(confidence * 100 if pd.notna(confidence) and str(confidence) != '' else np.nan, 0)}%" if str(confidence) != "" else "review"
        sample = escape(str(row.get("sample_id", "")))
        route = escape(str(row.get("route_id", "")))
        return (
            f"<div class='vs-bed-card {kind}'>"
            "<div class='vs-bed-top'>"
            f"<span class='vs-bed-status'>{escape(status)}</span>"
            f"<span class='vs-bed-room'>{room}</span>"
            "</div>"
            "<div class='vs-bed-main'>"
            f"<div><div class='vs-bed-id'>{primary_label}</div>"
            f"<div class='vs-bed-sub'>{sample[:42]}</div></div>"
            f"<div class='vs-bed-hr'>{hr_text}</div>"
            "</div>"
            f"<div class='vs-bed-sub'>Route: {route[:34]}<br>Confidence: {conf_text} 路 Last update: just now</div>"
            "</div>"
        )

    def alert_row(row: pd.Series, idx: int) -> str:
        kind = decision_kind(row)
        decision = str(row.get("decision", "review"))
        hr = row.get("hr_bpm", "")
        hr_text = f"{fmt(hr, 0)} bpm" if pd.notna(hr) and str(hr) != "" else "review"
        reason = str(row.get("review_reason", "")) or str(row.get("user_message", ""))
        return (
            f"<div class='vs-alert-row {kind}'>"
            f"<div class='vs-alert-title'><span>A-{301 + idx} 路 {escape(decision.upper())}</span><span>{escape(hr_text)}</span></div>"
            f"<div class='vs-alert-note'>{escape(reason[:130] or 'Route evidence accepted.')}</div>"
            "</div>"
        )

    release_count = int((demo_cases["decision"].astype(str).eq("release")).sum()) if not demo_cases.empty and "decision" in demo_cases.columns else 0
    review_count = int((demo_cases["decision"].astype(str).eq("review")).sum()) if not demo_cases.empty and "decision" in demo_cases.columns else 0
    max_latency = qa.get("max_p95_packet_latency_ms", "NA") if isinstance(qa, dict) else "NA"
    passed_gates = stats.get("passed_claim_gates", "NA") if isinstance(stats, dict) else "NA"
    total_gates = stats.get("n_claim_gates", "NA") if isinstance(stats, dict) else "NA"

    if demo_cases.empty:
        st.warning("T527 demo cases are missing. Run the product demo QA package first.")
        return

    case_labels = [
        f"{row.case_id} | {row.dataset} | {row.decision}"
        for row in demo_cases.itertuples(index=False)
    ]
    selected_label = case_labels[0]
    selected_case = demo_cases.iloc[case_labels.index(selected_label)]

    route_id = str(selected_case.get("route_id", ""))
    policy_row = policy[policy["route_id"].astype(str).eq(route_id)].head(1) if not policy.empty and "route_id" in policy.columns else pd.DataFrame()
    route_row = policy_row.iloc[0] if not policy_row.empty else pd.Series(dtype=object)
    decision = str(selected_case.get("decision", "review"))
    decision_color = "vs-chip-good" if decision == "release" else "vs-chip-warn"

    active_rooms = len(demo_cases)
    device_online = max(active_rooms - 1, 0)
    data_quality = "94%"
    released_metric = f"{release_count}/{len(demo_cases)}"
    reviewed_metric = f"{review_count}"

    st.markdown("<div class='vs-market-shell'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='vs-market-hero'>"
        "<div class='vs-hero-main'>"
        "<div class='vs-hero-title'>VitalsSight Adult HR command center</div>"
        "<div class='vs-hero-copy'>A contactless adult heart-rate operations console built around route-aware release/review. "
        "It combines competitor-style ward monitoring, alert triage, trend reporting, API packets, and our Route-MoE explanation layer so users know not only the HR value, but whether it is safe to release.</div>"
        "</div>"
        "<div class='vs-live-strip'>"
        "<div><span class='vs-live-dot'></span><b>Live operations mode</b></div>"
        "<div class='vs-console-note'>Camera/video inputs are routed through quality checks, candidate-peak analysis, domain routing, and route-specific release policies.</div>"
        "<div class='vs-console-note'>Boundary: not a diagnostic or emergency-alert system until clinical/regulatory validation is completed.</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='vs-guide'>"
        "<div class='vs-guide-step'><span class='vs-guide-num'>1</span><span class='vs-guide-title'>Connect video</span><div class='vs-guide-copy'>Use live camera, uploaded video, or replay packet.</div></div>"
        "<div class='vs-guide-step'><span class='vs-guide-num'>2</span><span class='vs-guide-title'>Check decision</span><div class='vs-guide-copy'>The system releases HR only when route evidence is valid.</div></div>"
        "<div class='vs-guide-step'><span class='vs-guide-num'>3</span><span class='vs-guide-title'>Resolve reviews</span><div class='vs-guide-copy'>Review amber/red cases with route reason and quality flags.</div></div>"
        "<div class='vs-guide-step'><span class='vs-guide-num'>4</span><span class='vs-guide-title'>Export report</span><div class='vs-guide-copy'>Download report cards, CSV evidence, or API packets.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='vs-market-kpis'>"
        f"<div class='vs-kpi-dark'><div class='vs-kpi-label'>Active rooms</div><div class='vs-kpi-value'>{active_rooms}</div><div class='vs-kpi-note'>Demo ward cases loaded</div></div>"
        f"<div class='vs-kpi-dark'><div class='vs-kpi-label'>HR releases</div><div class='vs-kpi-value'>{released_metric}</div><div class='vs-kpi-note'>Route evidence accepted</div></div>"
        f"<div class='vs-kpi-dark'><div class='vs-kpi-label'>Review queue</div><div class='vs-kpi-value'>{reviewed_metric}</div><div class='vs-kpi-note'>Needs human or stronger sensor path</div></div>"
        f"<div class='vs-kpi-dark'><div class='vs-kpi-label'>Devices online</div><div class='vs-kpi-value'>{device_online}/{active_rooms}</div><div class='vs-kpi-note'>One route reserved for multimodal review</div></div>"
        f"<div class='vs-kpi-dark'><div class='vs-kpi-label'>Data quality</div><div class='vs-kpi-value'>{data_quality}</div><div class='vs-kpi-note'>Route-level evidence health</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    ward_rows = [demo_cases.iloc[i] for i in range(len(demo_cases))]
    bed_html = "".join(bed_card(row, i) for i, row in enumerate(ward_rows))
    review_rows = [
        (i, row)
        for i, row in enumerate(ward_rows)
        if str(row.get("decision", "")).lower() != "release"
    ]
    if not review_rows:
        review_rows = list(enumerate(ward_rows[:3]))
    alert_html = "".join(alert_row(row, i) for i, row in review_rows[:6])
    hr_value_first = selected_case.get("hr_bpm", "")
    output_first = f"{fmt(hr_value_first, 2)} BPM" if pd.notna(hr_value_first) and str(hr_value_first) != "" else "Review"
    status_first = str(selected_case.get("user_message", "")) if decision == "release" else str(selected_case.get("review_reason", "Review required."))
    compact_focus_html = (
        "<div class='vs-alert-panel' style='margin-bottom:10px'>"
        "<div class='vs-console-title'>Selected workflow</div>"
        f"<div class='vs-console-value' style='font-size:18px;color:#152833'>{escape(output_first)}</div>"
        f"<div class='vs-console-note'>{escape(status_first[:150])}</div>"
        "<div style='height:8px'></div>"
        f"<span class='vs-chip {decision_color}'>Decision: {escape(decision)}</span>"
        f"<span class='vs-chip'>Route-MoE</span>"
        f"<span class='vs-chip'>Policy: {escape(str(selected_case.get('selected_policy', ''))[:28])}</span>"
        "<div class='vs-console-note'>Flow: input video/candidates -> route detector -> route expert -> release or review.</div>"
        "</div>"
    )

    st.markdown(
        "<div class='vs-command-grid'>"
        "<div>"
        "<div class='vs-section-title'>Ward / bed overview</div>"
        f"<div class='vs-bed-grid'>{bed_html}</div>"
        "</div>"
        "<div>"
        "<div class='vs-section-title'>Current decision & alerts</div>"
        f"{compact_focus_html}<div class='vs-alert-panel'>{alert_html}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='vs-section-title'>Selected case workflow</div>", unsafe_allow_html=True)
    st.selectbox("Choose focus case", case_labels, index=case_labels.index(selected_label), key="adult_console_case_focus")
    focus_label = st.session_state.get("adult_console_case_focus", selected_label)
    focus_case = demo_cases.iloc[case_labels.index(focus_label)]
    selected_case = focus_case
    route_id = str(selected_case.get("route_id", ""))
    decision = str(selected_case.get("decision", "review"))
    decision_color = "vs-chip-good" if decision == "release" else "vs-chip-warn"
    hr_value = selected_case.get("hr_bpm", "")
    output_value = f"{fmt(hr_value, 2)} BPM" if pd.notna(hr_value) and str(hr_value) != "" else "Review"
    status_text = str(selected_case.get("user_message", "")) if decision == "release" else str(selected_case.get("review_reason", "Review required."))

    detail_html = (
        "<div class='vs-detail-grid'>"
        "<div class='vs-detail-card'>"
        "<div class='vs-console-title'>Live scan</div>"
        "<div class='vs-scan-frame'><div class='vs-face'></div></div>"
        f"<div class='vs-console-note'>Input: {escape(str(selected_case.get('required_input', ''))[:180])}</div>"
        "</div>"
        "<div class='vs-detail-card'>"
        "<div class='vs-console-title'>Current decision</div>"
        f"<div class='vs-console-value'>{escape(output_value)}</div>"
        f"<div class='vs-console-note'>{escape(status_text[:260])}</div>"
        "<div style='height:10px'></div>"
        f"<span class='vs-chip {decision_color}'>Decision: {escape(decision)}</span>"
        f"<span class='vs-chip'>Policy: {escape(str(selected_case.get('selected_policy', '')))}</span>"
        f"<span class='vs-chip'>Route: {escape(route_id[:42])}</span>"
        "</div>"
        "<div class='vs-pipeline-card'>"
        "<div class='vs-console-title'>Route-MoE decision pipeline</div>"
        "<div class='vs-pipeline-steps'>"
        f"<div class='vs-pipeline-step'><div class='vs-step-label'>Input</div><div class='vs-step-value'>{escape(str(selected_case.get('dataset', '')))}</div></div>"
        f"<div class='vs-pipeline-step'><div class='vs-step-label'>Route</div><div class='vs-step-value'>{escape(route_id[:28])}</div></div>"
        f"<div class='vs-pipeline-step'><div class='vs-step-label'>Expert</div><div class='vs-step-value'>{escape(str(selected_case.get('selected_policy', ''))[:32])}</div></div>"
        f"<div class='vs-pipeline-step'><div class='vs-step-label'>Action</div><div class='vs-step-value'>{escape(decision.upper())}</div></div>"
        "</div>"
        f"<div class='vs-console-note'>Reason: {escape(str(selected_case.get('review_reason', '') or selected_case.get('route_detector', ''))[:220])}</div>"
        "</div>"
        "</div>"
    )
    st.markdown(detail_html, unsafe_allow_html=True)

    if decision == "review":
        st.warning(str(selected_case.get("review_reason", "Route requires review.")))
    else:
        st.success(str(selected_case.get("user_message", "Heart-rate estimate released.")))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='vs-section-title'>Market parity and VitalsSight edge</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='vs-feature-grid'>"
        "<div class='vs-feature'><b>Ward command center</b><span>Competitor-parity bed overview, device health, review queue, and status color coding for care teams.</span></div>"
        "<div class='vs-feature'><b>Trend and report workflow</b><span>Validated cases, report cards, CSV/API export, and evidence tables are available from the same product surface.</span></div>"
        "<div class='vs-feature'><b>Route-MoE release gate</b><span>Our differentiator: the product decides whether HR is releasable before showing it, using route evidence and physiology constraints.</span></div>"
        "<div class='vs-feature'><b>Explainable review reasons</b><span>Uncertain motion, lighting, fairness, high-HR, or multimodal gaps become visible review reasons instead of silent failure.</span></div>"
        "<div class='vs-feature'><b>Candidate-peak inspection</b><span>The system preserves multiple HR candidates and detects harmonic or alias conflicts instead of trusting one peak.</span></div>"
        "<div class='vs-feature'><b>Integration-ready packets</b><span>Each decision can be exported as an API packet with route ID, policy, HR, confidence, quality flags, and warning boundary.</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<span class='vs-chip {decision_color}'>Decision: {escape(decision)}</span>"
        f"<span class='vs-chip'>Policy: {escape(str(selected_case.get('selected_policy', '')))}</span>"
        f"<span class='vs-chip'>Route: {escape(route_id)}</span>",
        unsafe_allow_html=True,
    )

    st.subheader("Route evidence")
    metric_view = route_metrics.copy()
    if not metric_view.empty and "route_id" in metric_view.columns:
        metric_view = metric_view[metric_view["route_id"].astype(str).isin(policy["route_id"].astype(str).tolist())] if not policy.empty else metric_view
        display_cols = [
            col
            for col in [
                "route_id",
                "dataset",
                "n_inputs",
                "coverage",
                "mae_released_bpm",
                "unsafe_per_input",
            ]
            if col in metric_view.columns
        ]
        st.dataframe(metric_view[display_cols], width="stretch", hide_index=True)

        chart_frame = metric_view.dropna(subset=["coverage"]).copy() if "coverage" in metric_view.columns else pd.DataFrame()
        if not chart_frame.empty and {"route_id", "dataset", "coverage"}.issubset(chart_frame.columns):
            chart_frame["route_dataset"] = chart_frame["route_id"].astype(str) + " / " + chart_frame["dataset"].astype(str)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=chart_frame["route_dataset"],
                    y=pd.to_numeric(chart_frame["coverage"], errors="coerce"),
                    name="coverage",
                    marker_color="#2f7d6d",
                )
            )
            if "unsafe_per_input" in chart_frame.columns:
                fig.add_trace(
                    go.Bar(
                        x=chart_frame["route_dataset"],
                        y=pd.to_numeric(chart_frame["unsafe_per_input"], errors="coerce"),
                        name="unsafe/input",
                        marker_color="#a84d3a",
                    )
                )
            fig.update_layout(
                height=360,
                barmode="group",
                yaxis_tickformat=".0%",
                margin=dict(l=12, r=12, t=24, b=120),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_xaxes(tickangle=-35)
            fig.update_yaxes(gridcolor="#e5eaee", zeroline=False)
            st.plotly_chart(fig, width="stretch")

    tabs = st.tabs(["Product policy", "Report cards", "API packet", "MVP maturity"])
    with tabs[0]:
        if not policy.empty:
            st.dataframe(
                policy[
                    [
                        col
                        for col in [
                            "route_id",
                            "domain_or_dataset",
                            "selected_policy",
                            "product_action",
                            "coverage",
                            "mae_bpm",
                            "unsafe_per_input",
                            "claim_boundary",
                        ]
                        if col in policy.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
    with tabs[1]:
        source = report_cards if not report_cards.empty else demo_cases
        st.dataframe(source, width="stretch", hide_index=True)
    with tabs[2]:
        examples = api.get("examples", []) if isinstance(api, dict) else []
        if examples:
            route_example = next((item for item in examples if item.get("route_id") == route_id), examples[0])
            st.json(route_example, expanded=True)
        else:
            st.info("No API examples available.")
    with tabs[3]:
        if not mvp_actions.empty:
            st.dataframe(mvp_actions, width="stretch", hide_index=True)

    csv_bytes = demo_cases.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download demo routing cases",
        data=csv_bytes,
        file_name="vitalsight_adult_hr_route_moe_demo_cases.csv",
        mime="text/csv",
        width="stretch",
    )


def adult_hr_market_console() -> None:
    """Compatibility wrapper for the retired market-console implementation."""
    adult_hr_product_console_v3("Command Center")

def trend_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    reference_col: str | None,
    raw_col: str | None,
    predicted_col: str,
) -> go.Figure:
    fig = go.Figure()
    if reference_col and reference_col in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["start_sec"],
                y=frame[reference_col],
                mode="lines+markers",
                name="reference",
                line=dict(color="#111827", width=3),
            )
        )
    if raw_col and raw_col in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["start_sec"],
                y=frame[raw_col],
                mode="lines+markers",
                name="raw",
                line=dict(color="#b85c38", width=2),
            )
        )
    prediction_name = "product output" if predicted_col == "product_rr_bpm" else "validated"
    fig.add_trace(
        go.Scatter(
            x=frame["start_sec"],
            y=frame[predicted_col],
            mode="lines+markers",
            name=prediction_name,
            line=dict(color="#2f7d6d", width=3),
            connectgaps=False,
        )
    )
    if "accepted" in frame.columns:
        refused = frame[~frame["accepted"].astype(bool)]
        if not refused.empty:
            fallback_col = "candidate_rr_bpm" if "candidate_rr_bpm" in refused.columns else predicted_col
            fig.add_trace(
                go.Scatter(
                    x=refused["start_sec"],
                    y=refused[fallback_col],
                    mode="markers",
                    name="refused",
                    marker=dict(color="#a84d3a", size=12, symbol="x"),
                )
            )
    if "decision" in frame.columns:
        half = frame[frame["decision"].astype(str).str.startswith("half_")]
        if not half.empty:
            half_y = pd.to_numeric(half[predicted_col], errors="coerce")
            if "candidate_rr_bpm" in half.columns:
                half_y = half_y.fillna(pd.to_numeric(half["candidate_rr_bpm"], errors="coerce"))
            fig.add_trace(
                go.Scatter(
                    x=half["start_sec"],
                    y=half_y,
                    mode="markers",
                    name="half-rate decision",
                    marker=dict(color="#b7791f", size=12, symbol="diamond"),
                )
            )
    if "harmonic_multiplier" in frame.columns:
        harmonic_multiplier = pd.to_numeric(frame["harmonic_multiplier"], errors="coerce")
        harmonic = frame[harmonic_multiplier.gt(1.0)].copy()
        if not harmonic.empty:
            fig.add_trace(
                go.Scatter(
                    x=harmonic["start_sec"],
                    y=pd.to_numeric(harmonic[predicted_col], errors="coerce"),
                    mode="markers",
                    name="harmonic lift",
                    marker=dict(color="#7c3aed", size=11, symbol="triangle-up"),
                )
            )
    if "alert" in frame.columns:
        alerts = frame[frame["alert"].astype(bool)].copy()
        if not alerts.empty:
            fig.add_trace(
                go.Scatter(
                    x=alerts["start_sec"],
                    y=pd.to_numeric(alerts[predicted_col], errors="coerce"),
                    mode="markers",
                    name="actionability alert",
                    marker=dict(color="#a84d3a", size=13, symbol="triangle-up"),
                )
            )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.0),
        height=430,
        margin=dict(l=12, r=12, t=56, b=82),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2933", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        xaxis_title="Time (s)",
        yaxis_title="RR (BPM)",
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#e5eaee", zeroline=False)
    fig.update_yaxes(gridcolor="#e5eaee", zeroline=False)
    return fig


def t98_interval_chart(frame: pd.DataFrame, *, title: str) -> go.Figure:
    fig = go.Figure()
    if "gt_rr_bpm" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["start_sec"],
                y=pd.to_numeric(frame["gt_rr_bpm"], errors="coerce"),
                mode="lines+markers",
                name="reference",
                line=dict(color="#111827", width=3),
            )
        )

    interval_lower = pd.to_numeric(frame.get("interval_lower_bpm"), errors="coerce")
    interval_upper = pd.to_numeric(frame.get("interval_upper_bpm"), errors="coerce")
    if np.isfinite(interval_lower).any() and np.isfinite(interval_upper).any():
        fig.add_trace(
            go.Scatter(
                x=frame["start_sec"],
                y=interval_upper,
                mode="lines",
                line=dict(color="rgba(47, 125, 109, 0.18)", width=1),
                name="interval upper",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["start_sec"],
                y=interval_lower,
                mode="lines",
                line=dict(color="rgba(47, 125, 109, 0.18)", width=1),
                fill="tonexty",
                fillcolor="rgba(47, 125, 109, 0.16)",
                name="RR interval",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=frame["start_sec"],
            y=pd.to_numeric(frame["display_rr_bpm"], errors="coerce"),
            mode="lines+markers",
            name="RR estimate",
            line=dict(color="#2f7d6d", width=3),
            connectgaps=False,
        )
    )

    if "calibrated_risk_accepted" in frame.columns:
        risk_flagged = frame[~frame["calibrated_risk_accepted"].astype(bool)].copy()
        if not risk_flagged.empty:
            fig.add_trace(
                go.Scatter(
                    x=risk_flagged["start_sec"],
                    y=pd.to_numeric(risk_flagged["display_rr_bpm"], errors="coerce"),
                    mode="markers",
                    name="source-risk flag",
                    marker=dict(color="#a84d3a", size=12, symbol="x"),
                )
            )
    if "interval_available" in frame.columns:
        withheld = frame[~frame["interval_available"].astype(bool)].copy()
        if not withheld.empty:
            fig.add_trace(
                go.Scatter(
                    x=withheld["start_sec"],
                    y=pd.to_numeric(withheld["display_rr_bpm"], errors="coerce"),
                    mode="markers",
                    name="interval withheld",
                    marker=dict(color="#60707d", size=10, symbol="x-thin"),
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.0),
        height=430,
        margin=dict(l=12, r=12, t=56, b=82),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2933", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        xaxis_title="Time (s)",
        yaxis_title="RR (BPM)",
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#e5eaee", zeroline=False)
    fig.update_yaxes(gridcolor="#e5eaee", zeroline=False)
    return fig


def benchmark_bar(frame: pd.DataFrame) -> go.Figure:
    plot = frame.copy()
    plot["label"] = plot["dataset"] + "<br>" + plot["method"].map(compact_method_name)
    plot = plot.sort_values(["dataset", "our_mae_bpm"], na_position="last")
    colors = np.where(plot["population"].eq("infant"), "#2f7d6d", "#6c8f5f")
    fig = go.Figure(go.Bar(x=plot["label"], y=plot["our_mae_bpm"], marker_color=colors))
    fig.update_layout(
        height=430,
        margin=dict(l=12, r=12, t=24, b=80),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis_title="MAE (BPM)",
        font=dict(color="#1f2933", size=12),
    )
    fig.update_yaxes(gridcolor="#e5eaee", zeroline=False)
    fig.update_xaxes(tickangle=-35)
    return fig


def compact_method_name(method: str) -> str:
    return (
        str(method)
        .replace("_half_validated", "")
        .replace("motion_energy_", "me_")
        .replace("optical_flow_y_", "flow_y_")
        .replace("body_aware", "body")
        .replace("best_confidence", "best")
        .replace("depth_harmonic_", "dh_")
        .replace("latent_tail_state_viterbi_balanced", "T94 balanced")
        .replace("latent_tail_state_viterbi_high_recall", "T94 high-RR")
        .replace("latent_state_viterbi_equal", "T94 latent")
        .replace("t95_loso_combined_tail_selection", "T95 LOSO combined")
        .replace("t95_loso_mae_selection", "T95 LOSO MAE")
        .replace("t95_loso_high_rr_selection", "T95 LOSO high-RR")
        .replace("risk_target_0.30_q90_gap", "T98 q90 gap risk 0.30")
        .replace("risk_target_0.30_t96_reference", "T98 T96 risk 0.30")
        .replace("risk_target_0.30_upper_gap_latent", "T98 upper+latent risk 0.30")
        .replace("risk_target_0.25_q90_gap", "T98 q90 gap risk 0.25")
        .replace("risk_target_0.35_q90_gap", "T98 q90 gap risk 0.35")
        .replace("t97_upper_gap_latent_q75", "T98 upper+latent q75")
        .replace("t97_q90_gap_q80", "T98 q90 gap q80")
        .replace("t104_default_no_source_validity_guard", "T104 no source guard")
        .replace("t105_combined_source_validity_review", "T105 source review")
        .replace("t106_green_gap_equal_tail_fallback_shifted_t104_interval", "T106 green/equal fallback")
        .replace("t107_trusted_countercluster_selector_shifted_t104_interval", "T107 trusted selector")
        .replace("t107_trusted_countercluster_selector_route_conformal_90", "T107 route 90")
        .replace("t107_trusted_countercluster_selector_route_conformal_80", "T107 route 80")
        .replace("t110_baseline_no_gate", "T110 no gate")
        .replace("strict_multifamily_score_gate", "strict family+score")
        .replace("t111_mechanistic_safety_gate", "T111 safety gate")
        .replace("t111_direction_aware_product_gate", "T111 product gate")
        .replace("t114a_margin_robust_high_gate", "T114A margin gate")
        .replace("t115_broad_stress_guard_fixed", "T115 fixed guard")
        .replace("t115_broad_stress_guard", "T115 broad guard")
        .replace("t117_ambiguous_support_buffer_candidate", "T117 buffered candidate")
        .replace("t119_route_risk_dev_zero_safe_loss", "T119 route-risk scorer")
        .replace("t120_subject_aware_tail_rate_gate", "T120 subject-aware gate")
        .replace("route_score_noise", "route-score noise")
        .replace("moderate_extraction_shift", "moderate extraction")
        .replace("aggressive_extraction_shift", "aggressive extraction")
        .replace("anchor_abs_ge_6", "anchor >=6 BPM")
        .replace("anchor_abs_ge_8", "anchor >=8 BPM")
        .replace("anchor_abs_ge_10", "anchor >=10 BPM")
        .replace("anchor_abs_ge_12", "anchor >=12 BPM")
        .replace("high_anchor_all", "high anchor")
        .replace("high_anchor_non_mk503", "high non-mk503")
        .replace("low_anchor_diagnostic_all", "low diagnostic")
        .replace("low_anchor_non_mk503", "low non-mk503")
        .replace("diagnostic_oracle_safe_candidate_reviewed_shifted_t104_interval", "oracle safe candidate")
        .replace("T58|aligned_harmonic_green_mean_smooth3", "T58 green mean")
        .replace("T58|aligned_harmonic_depth_conf_weighted", "T58 depth conf")
        .replace("T58|aligned_harmonic_depth_best_confidence", "T58 depth best")
        .replace("T94|latent_state_viterbi_equal", "T94 equal latent")
        .replace("T58|aligned_harmonic_depth_best", "T58 depth best")
        .replace("T94|T94 latent", "T94 equal latent")
        .replace("all_windows", "all windows")
        .replace("t58_base_plausible_median_smooth3", "T58 base")
        .replace("_q90_smooth3", "_q90")
        .replace("_smooth3", "")
        .replace("blend075", "blend75")
        .replace("blend050", "blend50")
        .replace("depth_right", "d_right")
        .replace("depth_left", "d_left")
        .replace("depth_", "d_")
    )


def sample_roi_preview(sample_id: str, roi_name: str | None) -> np.ndarray | None:
    index = load_sample_index()
    row = index[index["sample_id"].eq(sample_id)]
    if row.empty:
        return None
    item = row.iloc[0]
    if pd.isna(item.get("archive_path")) or pd.isna(item.get("video_member")):
        return None
    try:
        video_path = extract_zip_member(item["archive_path"], item["video_member"], CACHE_DIR / sample_id)
        frame = first_frame(video_path)
        rois = candidate_respiration_rois(frame, view="infant", infant=True)
        selected = next((roi for roi in rois if roi.name == roi_name), None)
        if selected is None:
            body_rois = body_aware_respiration_rois(frame, view="infant", infant=True, min_score=0.20)
            selected = body_rois[0] if body_rois else rois[0]
        preview = draw_rois(frame, rois[:12], selected=selected)
        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def air_monitor() -> None:
    data = load_air_results()
    methods = sorted(data["method"].unique())
    datasets = sorted(data["dataset"].unique())
    dataset = st.sidebar.selectbox("Dataset", datasets, index=datasets.index("AIR-125") if "AIR-125" in datasets else 0)
    method = st.sidebar.selectbox(
        "Estimator",
        methods,
        index=methods.index(DEFAULT_METHOD) if DEFAULT_METHOD in methods else 0,
    )
    policy_mode = "display_without_refusal"
    if method == DEFAULT_METHOD:
        policy_mode = sidebar_policy_selector(default="t57_calibrated_confidence_refusal")
    else:
        st.sidebar.info("Product refusal policies are only calibrated for the default optical-flow-y estimator.")
    available = data[(data["dataset"].eq(dataset)) & (data["method"].eq(method))]
    sample_ids = sorted(available["sample_id"].unique())
    sample_id = st.sidebar.selectbox("Sample", sample_ids)
    sample = available[available["sample_id"].eq(sample_id)].sort_values("window_id")
    sample = prepare_product_output(sample, pred_col="validated_rr_bpm", policy_mode=policy_mode)

    metrics = summarize_window_metrics(sample, pred_col="product_rr_bpm", gt_col="gt_rr_bpm")
    cols = st.columns(4)
    cols[0].metric("Product RR", latest_product_value(sample))
    cols[1].metric("Accepted MAE", f"{fmt(metrics['mae'])} BPM")
    cols[2].metric("Coverage", f"{fmt(100.0 * metrics['coverage'], 1)}%")
    cols[3].metric("Refused", f"{metrics['refused']}/{metrics['total']}")
    product_status_banner(sample)

    center, right = st.columns([0.66, 0.34], gap="large")
    with center:
        st.plotly_chart(
            trend_chart(
                sample,
                title=f"{dataset} / {sample_id}<br>{compact_method_name(method)}",
                reference_col="gt_rr_bpm",
                raw_col="raw_rr_bpm",
                predicted_col="product_rr_bpm",
            ),
            width="stretch",
        )
        st.dataframe(
            sample[
                [
                    "window_id",
                    "start_sec",
                    "gt_rr_bpm",
                    "raw_rr_bpm",
                    "validated_rr_bpm",
                    "product_rr_bpm",
                    "accepted",
                    "refusal_reason",
                    "product_policy_label",
                    "product_policy_threshold",
                    "decision",
                    "half_power_ratio",
                    "confidence",
                    "roi_name",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("ROI")
        preview = sample_roi_preview(sample_id, str(sample["roi_name"].iloc[-1]))
        if preview is not None:
            st.image(preview, width="stretch")
        else:
            st.info("ROI preview unavailable for this sample.")
        latest = sample.iloc[-1]
        st.subheader("Validation")
        st.dataframe(
            pd.DataFrame(
                [
                    ["raw RR", fmt(latest["raw_rr_bpm"])],
                    ["validated RR", fmt(latest["validated_rr_bpm"])],
                    ["product RR", fmt(latest["product_rr_bpm"])],
                    ["status", str(latest["product_status"])],
                    ["policy", str(latest["product_policy_label"])],
                    ["policy threshold", fmt(latest["product_policy_threshold"], 3)],
                    ["refusal reason", str(latest["refusal_reason"])],
                    ["decision", str(latest["decision"])],
                    ["half candidate", fmt(latest["half_bpm"])],
                    ["half power ratio", fmt(latest["half_power_ratio"], 3)],
                    ["confidence", fmt(latest["confidence"], 3)],
                    ["ROI quality", fmt(latest["roi_quality"], 3)],
                ],
                columns=["field", "value"],
            ),
            width="stretch",
            hide_index=True,
        )
        export_panel(sample, title=f"{dataset}_{sample_id}_{method}")


def cambridge_policy_options(summary: pd.DataFrame, windows: pd.DataFrame) -> list[str]:
    if summary.empty or windows.empty:
        return sorted(windows["policy"].dropna().unique().tolist()) if "policy" in windows.columns else []
    policies = set(windows["policy"].dropna().unique())
    ranked = summary[summary["policy"].isin(policies)].copy()
    ranked["mae_bpm"] = pd.to_numeric(ranked["mae_bpm"], errors="coerce")
    ranked = ranked.sort_values(["mae_bpm", "policy"], na_position="last")
    return ranked["policy"].astype(str).tolist()


def t58_delta_text(stability: pd.DataFrame, policy: str) -> str:
    if stability.empty:
        return "NA"
    row = stability[(stability["policy"].eq(policy)) & (stability["metric"].eq("delta_mae_vs_legacy"))]
    if row.empty:
        return "NA"
    item = row.iloc[0]
    return f"{fmt(item['median'])} BPM"


def t58_stability_table(stability: pd.DataFrame, policy: str) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()
    rows = stability[
        (stability["policy"].eq(policy))
        & (stability["metric"].isin(["mae_bpm", "rmse_bpm", "delta_mae_vs_legacy"]))
    ].copy()
    if rows.empty:
        return rows
    rows["median"] = pd.to_numeric(rows["median"], errors="coerce").round(3)
    rows["ci_low"] = pd.to_numeric(rows["ci_low"], errors="coerce").round(3)
    rows["ci_high"] = pd.to_numeric(rows["ci_high"], errors="coerce").round(3)
    return rows[["metric", "median", "ci_low", "ci_high"]].rename(
        columns={"metric": "Metric", "median": "Median", "ci_low": "CI low", "ci_high": "CI high"}
    )


def t87_delta_text(ci: pd.DataFrame, policy: str, metric: str) -> str:
    if ci.empty:
        return "NA"
    row = ci[(ci["policy"].eq(policy)) & (ci["metric"].eq(metric))]
    if row.empty:
        return "NA"
    item = row.iloc[0]
    return f"{fmt(item['median'])} BPM"


def t87_ci_table(ci: pd.DataFrame, policy: str) -> pd.DataFrame:
    if ci.empty:
        return pd.DataFrame()
    rows = ci[
        (ci["policy"].eq(policy))
        & (
            ci["metric"].isin(
                [
                    "delta_mae_vs_t58_base_bpm",
                    "delta_high_rr_mae_vs_t58_base_bpm",
                    "delta_signed_error_vs_t58_base_bpm",
                ]
            )
        )
    ].copy()
    if rows.empty:
        return rows
    for col in ["median", "ci_low_2p5", "ci_high_97p5"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").round(3)
    rows["metric"] = rows["metric"].replace(
        {
            "delta_mae_vs_t58_base_bpm": "Delta MAE vs T58",
            "delta_high_rr_mae_vs_t58_base_bpm": "Delta high-RR MAE",
            "delta_signed_error_vs_t58_base_bpm": "Delta signed error",
        }
    )
    return rows[["metric", "median", "ci_low_2p5", "ci_high_97p5"]].rename(
        columns={"metric": "Metric", "median": "Median", "ci_low_2p5": "CI low", "ci_high_97p5": "CI high"}
    )


def delta_ci_table(ci: pd.DataFrame, policy: str, *, reference: str = "t87_blend075") -> pd.DataFrame:
    if ci.empty:
        return pd.DataFrame()
    delta_policy = f"{policy}__delta_vs_{reference}"
    rows = ci[
        (ci["policy"].eq(delta_policy))
        & (ci["metric"].isin(["mae_bpm", "high_rr_mae_bpm", "combined_tail_score", "signed_error_mean_bpm"]))
    ].copy()
    if rows.empty:
        return rows
    for col in ["mean", "ci_low", "ci_high"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").round(3)
    rows["metric"] = rows["metric"].replace(
        {
            "mae_bpm": "Delta MAE",
            "high_rr_mae_bpm": "Delta high-RR MAE",
            "combined_tail_score": "Delta combined score",
            "signed_error_mean_bpm": "Delta signed error",
        }
    )
    return rows[["metric", "mean", "ci_low", "ci_high"]].rename(
        columns={"metric": "Metric", "mean": "Mean", "ci_low": "CI low", "ci_high": "CI high"}
    )


def t92_policy_rule_row(summary: pd.DataFrame, policy: str, rule_id: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary[(summary["policy"].eq(policy)) & (summary["rule_id"].eq(rule_id))].copy()


def t92_actionability_display(row: pd.Series) -> pd.DataFrame:
    fields = [
        ("event sensitivity", row.get("event_sensitivity")),
        ("missed high-RR events", row.get("missed_high_rr_events")),
        ("false alert episodes", row.get("false_alert_episodes")),
        ("alarm burden/hour", row.get("alarm_burden_per_hour")),
        ("mean time-to-detect sec", row.get("mean_time_to_detect_sec")),
        ("window precision", row.get("window_precision")),
    ]
    return pd.DataFrame(
        [
            {
                "Metric": name,
                "Value": fmt(value, 3) if name not in {"missed high-RR events", "false alert episodes"} else fmt(value, 0),
            }
            for name, value in fields
        ]
    )


def coerce_bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    text = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "accepted", "accepted_low_risk"])


def t98_interval_label(policy: str, alpha: float) -> str:
    confidence = int(round((1.0 - float(alpha)) * 100))
    return f"{compact_method_name(policy)} / {confidence}%"


def t98_latest_interval_text(row: pd.Series) -> str:
    lower = row.get("interval_lower_bpm")
    upper = row.get("interval_upper_bpm")
    if not np.isfinite(pd.to_numeric(pd.Series([lower, upper]), errors="coerce")).all():
        return "Withheld"
    return f"{fmt(lower)}-{fmt(upper)} BPM"


def t98_warning_reason(row: pd.Series) -> str:
    interval_available = bool(row.get("interval_available", False))
    risk_accepted = bool(row.get("calibrated_risk_accepted", False))
    width = pd.to_numeric(pd.Series([row.get("interval_width_bpm")]), errors="coerce").iloc[0]
    if not interval_available:
        return "selective_interval_withheld"
    if not risk_accepted:
        return "source_risk_score_above_threshold"
    if np.isfinite(width) and width >= T98_WIDE_INTERVAL_BPM:
        return "wide_uncertainty_interval"
    return "accepted_low_source_risk"


def t98_product_status(row: pd.Series) -> str:
    reason = t98_warning_reason(row)
    if reason == "accepted_low_source_risk":
        return "accepted_low_risk"
    if reason == "wide_uncertainty_interval":
        return "accepted_wide_interval"
    if reason == "source_risk_score_above_threshold":
        return "review_required_source_risk"
    return "interval_withheld"


def prepare_t98_product_output(
    risk_windows: pd.DataFrame,
    interval_windows: pd.DataFrame,
    *,
    calibration_policy: str,
    interval_policy: str,
    interval_alpha: float,
) -> pd.DataFrame:
    risk = risk_windows[risk_windows["calibration_policy"].eq(calibration_policy)].copy()
    interval = interval_windows[
        interval_windows["interval_policy"].eq(interval_policy)
        & np.isclose(pd.to_numeric(interval_windows["interval_alpha"], errors="coerce"), float(interval_alpha))
    ].copy()
    interval_cols = [
        "subject",
        "window_id",
        "interval_policy",
        "interval_note",
        "interval_alpha",
        "nominal_interval_coverage",
        "selective_recipe",
        "accepted_by_interval_policy",
        "interval_half_width_bpm",
        "interval_lower_bpm",
        "interval_upper_bpm",
        "interval_covers_reference",
    ]
    interval = interval[[col for col in interval_cols if col in interval.columns]]
    output = risk.merge(interval, on=["subject", "window_id"], how="left", validate="one_to_one")

    output["display_rr_bpm"] = pd.to_numeric(output["pred_rr_bpm"], errors="coerce")
    output["calibrated_risk_accepted"] = coerce_bool_series(output["accepted"])
    output["interval_lower_bpm"] = pd.to_numeric(output.get("interval_lower_bpm"), errors="coerce")
    output["interval_upper_bpm"] = pd.to_numeric(output.get("interval_upper_bpm"), errors="coerce")
    output["interval_available"] = np.isfinite(output["interval_lower_bpm"]) & np.isfinite(output["interval_upper_bpm"])
    output["interval_width_bpm"] = output["interval_upper_bpm"] - output["interval_lower_bpm"]
    if "accepted_by_interval_policy" not in output.columns:
        output["accepted_by_interval_policy"] = output["interval_available"]
    output["interval_policy_accepted"] = coerce_bool_series(output["accepted_by_interval_policy"]) & output["interval_available"]
    output["automation_ready"] = output["calibrated_risk_accepted"] & output["interval_policy_accepted"]
    output["product_rr_bpm"] = output["display_rr_bpm"].where(output["interval_available"], np.nan)
    output["automation_rr_bpm"] = output["display_rr_bpm"].where(output["automation_ready"], np.nan)
    fallback_status = pd.Series(
        np.where(output["calibrated_risk_accepted"], "accepted_low_risk", "flagged_high_risk"),
        index=output.index,
    )
    output["source_risk_status"] = output["disposition"].where(output["disposition"].notna(), fallback_status)
    output["warning_reason"] = output.apply(t98_warning_reason, axis=1)
    output["product_status"] = output.apply(t98_product_status, axis=1)
    output["accepted"] = output["automation_ready"]
    return output.sort_values(["subject", "window_id"]).reset_index(drop=True)


def t98_status_banner(row: pd.Series) -> None:
    status = str(row.get("product_status", "unknown"))
    reason = str(row.get("warning_reason", "unknown"))
    risk_score = fmt(row.get("source_risk_score"), 3)
    threshold = fmt(row.get("threshold_score"), 3)
    if status == "accepted_low_risk":
        st.success(f"Latest window accepted: source risk {risk_score} <= threshold {threshold}.")
    elif status == "accepted_wide_interval":
        st.warning(f"Latest window accepted with wide interval: {reason}.")
    elif status == "review_required_source_risk":
        st.warning(f"Latest window requires review: source risk {risk_score} exceeds threshold {threshold}.")
    else:
        st.warning(f"Latest window withheld from interval product: {reason}.")


def t98_summary_table(row: pd.Series, *, kind: str) -> pd.DataFrame:
    if row.empty:
        return pd.DataFrame()
    if kind == "risk":
        fields = [
            ("policy", compact_method_name(row.get("calibration_policy"))),
            ("target high-error risk", fmt(row.get("target_accepted_high_error_rate"), 2)),
            ("accepted coverage", fmt(100.0 * float(row.get("accepted_coverage")), 1) + "%"),
            ("accepted MAE BPM", fmt(row.get("accepted_mae_bpm"), 3)),
            ("flagged MAE BPM", fmt(row.get("flagged_mae_bpm"), 3)),
            ("accepted high-error rate", fmt(row.get("accepted_high_error_rate"), 3)),
            ("target risk gap", fmt(row.get("target_risk_gap"), 3)),
        ]
    else:
        fields = [
            ("interval", t98_interval_label(str(row.get("interval_policy")), float(row.get("interval_alpha")))),
            ("empirical coverage", fmt(row.get("empirical_interval_coverage"), 3)),
            ("nominal coverage", fmt(row.get("nominal_interval_coverage"), 3)),
            ("coverage gap", fmt(row.get("coverage_gap_empirical_minus_nominal"), 3)),
            ("mean width BPM", fmt(row.get("mean_interval_width_bpm"), 3)),
            ("accepted MAE BPM", fmt(row.get("accepted_mae_bpm"), 3)),
            ("accepted high-error rate", fmt(row.get("accepted_high_error_rate"), 3)),
        ]
    return pd.DataFrame(fields, columns=["Metric", "Value"])


def t107_policy_label(policy: str) -> str:
    return compact_method_name(str(policy))


def t107_latest_interval_text(row: pd.Series) -> str:
    lower = row.get("interval_lower_bpm")
    upper = row.get("interval_upper_bpm")
    ready = bool(row.get("automation_ready_after_policy", False))
    if not ready:
        return "Withheld"
    if not np.isfinite(pd.to_numeric(pd.Series([lower, upper]), errors="coerce")).all():
        return "Withheld"
    return f"{fmt(lower)}-{fmt(upper)} BPM"


def t107_route_status(row: pd.Series) -> str:
    ready = bool(row.get("automation_ready_after_policy", False))
    fallback = bool(row.get("fallback_applied", False))
    disposition = str(row.get("product_disposition", ""))
    if ready and fallback:
        return "corrected_fallback_output"
    if ready:
        return "automation_ready"
    if "source_validity" in disposition or "review" in disposition:
        return "review_required_source_validity"
    return "review_required"


def t107_warning_reason(row: pd.Series) -> str:
    if bool(row.get("fallback_applied", False)):
        reason = str(row.get("fallback_reason", "")).strip()
        source = str(row.get("fallback_policy_key", "")).strip()
        if reason and reason.lower() != "nan":
            return reason
        if source and source.lower() != "nan":
            return f"fallback_source={source}"
        return "fallback_applied"
    if bool(row.get("automation_ready_after_policy", False)):
        return "accepted_low_source_risk"
    disposition = str(row.get("product_disposition", "")).strip()
    if disposition and disposition.lower() != "nan":
        return disposition
    return "review_required_no_trusted_fallback"


def prepare_t107_product_output(windows: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    output = windows[windows["policy"].eq(policy)].copy()
    if output.empty:
        return output

    numeric_cols = [
        "window_id",
        "gt_rr_bpm",
        "balanced_pred_rr_bpm",
        "pred_rr_bpm_after_policy",
        "signed_error_bpm_after_policy",
        "abs_error_bpm_after_policy",
        "source_risk_score",
        "selector_cluster_id",
        "selector_cluster_center_rr_bpm",
        "selector_score",
        "interval_alpha_after_policy",
        "nominal_interval_coverage_after_policy",
        "interval_half_width_bpm_after_policy",
        "interval_width_bpm_after_policy",
        "interval_lower_bpm_after_policy",
        "interval_upper_bpm_after_policy",
    ]
    for col in numeric_cols:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")

    for col in [
        "automation_ready_after_policy",
        "fallback_applied",
        "high_error_flag_after_policy",
        "interval_covers_reference_after_policy",
        "t105_window_combined_source_validity_review",
        "t105_automation_ready_combined",
    ]:
        if col in output.columns:
            output[col] = coerce_bool_series(output[col])
        else:
            output[col] = False

    output["display_rr_bpm"] = output["pred_rr_bpm_after_policy"]
    output["automation_ready"] = output["automation_ready_after_policy"]
    output["accepted"] = output["automation_ready_after_policy"]
    output["interval_lower_bpm"] = output["interval_lower_bpm_after_policy"].where(output["automation_ready_after_policy"])
    output["interval_upper_bpm"] = output["interval_upper_bpm_after_policy"].where(output["automation_ready_after_policy"])
    output["interval_available"] = (
        np.isfinite(output["interval_lower_bpm"])
        & np.isfinite(output["interval_upper_bpm"])
        & output["automation_ready_after_policy"]
    )
    output["interval_width_bpm"] = output["interval_upper_bpm"] - output["interval_lower_bpm"]
    output["product_rr_bpm"] = output["display_rr_bpm"].where(output["automation_ready_after_policy"], np.nan)
    output["automation_rr_bpm"] = output["product_rr_bpm"]
    output["product_status"] = output.apply(t107_route_status, axis=1)
    output["source_route_status"] = output["product_status"]
    output["warning_reason"] = output.apply(t107_warning_reason, axis=1)
    output["source_risk_status"] = np.where(
        output["fallback_applied"],
        "fallback_corrected",
        np.where(output["automation_ready_after_policy"], "accepted_original_route", "review_required"),
    )
    return output.sort_values(["subject", "window_id"]).reset_index(drop=True)


def t107_status_banner(row: pd.Series) -> None:
    status = str(row.get("product_status", "unknown"))
    reason = str(row.get("warning_reason", "unknown"))
    if status == "corrected_fallback_output":
        source = compact_method_name(str(row.get("fallback_policy_key", "trusted fallback")))
        st.success(f"Latest window released after trusted fallback: {source}. Reason: {reason}.")
    elif status == "automation_ready":
        st.success("Latest window released on the original latent route with route-aware interval evidence.")
    else:
        st.warning(f"Latest window remains under review: {reason}.")


def t107_route_chart(frame: pd.DataFrame, *, title: str) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    x_col = "start_sec" if "start_sec" in frame.columns else "window_id"
    x_label = "Time (s)" if x_col == "start_sec" else "Window"
    x_values = frame[x_col]

    if "gt_rr_bpm" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=pd.to_numeric(frame["gt_rr_bpm"], errors="coerce"),
                mode="lines+markers",
                name="reference",
                line=dict(color="#111827", width=3),
            )
        )
    if "balanced_pred_rr_bpm" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=pd.to_numeric(frame["balanced_pred_rr_bpm"], errors="coerce"),
                mode="lines+markers",
                name="T95 balanced before route",
                line=dict(color="#8a99a8", width=2, dash="dot"),
            )
        )

    interval_lower = pd.to_numeric(frame.get("interval_lower_bpm"), errors="coerce")
    interval_upper = pd.to_numeric(frame.get("interval_upper_bpm"), errors="coerce")
    if np.isfinite(interval_lower).any() and np.isfinite(interval_upper).any():
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=interval_upper,
                mode="lines",
                line=dict(color="rgba(47, 125, 109, 0.18)", width=1),
                name="route interval upper",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=interval_lower,
                mode="lines",
                line=dict(color="rgba(47, 125, 109, 0.18)", width=1),
                fill="tonexty",
                fillcolor="rgba(47, 125, 109, 0.16)",
                name="route interval",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=pd.to_numeric(frame["product_rr_bpm"], errors="coerce"),
            mode="lines+markers",
            name="product RR output",
            line=dict(color="#2f7d6d", width=3),
            connectgaps=False,
        )
    )

    fallback = frame[frame["fallback_applied"].astype(bool)].copy()
    if not fallback.empty:
        fig.add_trace(
            go.Scatter(
                x=fallback[x_col],
                y=pd.to_numeric(fallback["product_rr_bpm"], errors="coerce"),
                mode="markers",
                name="trusted fallback release",
                marker=dict(color="#c47a2c", size=13, symbol="diamond"),
            )
        )

    reviewed = frame[~frame["automation_ready_after_policy"].astype(bool)].copy()
    if not reviewed.empty:
        fig.add_trace(
            go.Scatter(
                x=reviewed[x_col],
                y=pd.to_numeric(reviewed.get("balanced_pred_rr_bpm"), errors="coerce"),
                mode="markers",
                name="review required",
                marker=dict(color="#a84d3a", size=12, symbol="x"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.0),
        height=430,
        margin=dict(l=12, r=12, t=56, b=82),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2933", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        xaxis_title=x_label,
        yaxis_title="RR (BPM)",
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#e5eaee", zeroline=False)
    fig.update_yaxes(gridcolor="#e5eaee", zeroline=False)
    return fig


def t107_summary_table(row: pd.Series) -> pd.DataFrame:
    if row.empty:
        return pd.DataFrame()
    fields = [
        ("policy", t107_policy_label(str(row.get("policy")))),
        ("automation-ready coverage", fmt(100.0 * float(row.get("automation_ready_coverage")), 1) + "%"),
        ("automation-ready MAE BPM", fmt(row.get("automation_ready_mae_bpm"), 3)),
        ("automation high-error rate", fmt(row.get("automation_ready_high_error_rate"), 3)),
        ("fallback recovered windows", fmt(row.get("n_fallback_recovered_windows"), 0)),
        ("recovered-window MAE BPM", fmt(row.get("recovered_window_mae_bpm"), 3)),
        ("recovered high-error rate", fmt(row.get("recovered_window_high_error_rate"), 3)),
        ("interval coverage", fmt(row.get("automation_ready_interval_coverage"), 3)),
        ("mean interval width BPM", fmt(row.get("automation_ready_mean_width_bpm"), 3)),
        ("released high-error windows", fmt(row.get("released_high_error_windows"), 0)),
        ("interval mode", str(row.get("interval_mode"))),
    ]
    return pd.DataFrame(fields, columns=["Metric", "Value"])


def t111_policy_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keep = [
        ("high_anchor_all", "t110_baseline_no_gate"),
        ("high_anchor_all", "strict_multifamily_score_gate"),
        ("high_anchor_all", "t111_mechanistic_safety_gate"),
        ("high_anchor_non_mk503", "t111_mechanistic_safety_gate"),
        ("low_anchor_diagnostic_all", "t110_baseline_no_gate"),
        ("low_anchor_diagnostic_all", "t111_direction_aware_product_gate"),
    ]
    keep_frame = pd.DataFrame(keep, columns=["subset", "policy"])
    display = keep_frame.merge(summary, on=["subset", "policy"], how="left").dropna(subset=["n_cases"])
    if display.empty:
        return pd.DataFrame()
    display["Stress subset"] = display["subset"].map(compact_method_name)
    display["Gate policy"] = display["policy"].map(compact_method_name)
    rename = {
        "n_cases": "Cases",
        "n_released": "Released",
        "n_safe_recovered": "Safe",
        "n_unsafe_releases": "Unsafe",
        "n_withheld": "Withheld",
        "released_selected_mae_bpm": "Released MAE",
    }
    out = display[["Stress subset", "Gate policy", *rename.keys()]].rename(columns=rename).copy()
    for col in ["Released MAE"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    for col in ["Cases", "Released", "Safe", "Unsafe", "Withheld"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def t111_gate_metric_row(summary: pd.DataFrame, *, subset: str, policy: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    rows = summary[summary["subset"].eq(subset) & summary["policy"].eq(policy)]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t115_metric_row(summary: pd.DataFrame, *, scenario: str, subset: str, policy: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    rows = summary[
        summary["stress_scenario"].eq(scenario)
        & summary["subset"].eq(subset)
        & summary["policy"].eq(policy)
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t115_perturbation_metric_row(summary: pd.DataFrame, *, policy: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    rows = summary[summary["policy"].eq(policy)]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t115_policy_summary_table(summary: pd.DataFrame, perturbation: pd.DataFrame | None = None) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keep = [
        ("anchor_abs_ge_6", "high_anchor_all", "t114a_margin_robust_high_gate"),
        ("anchor_abs_ge_6", "high_anchor_all", "t115_broad_stress_guard"),
        ("anchor_abs_ge_8", "high_anchor_all", "t114a_margin_robust_high_gate"),
        ("anchor_abs_ge_8", "high_anchor_all", "t115_broad_stress_guard"),
        ("anchor_abs_ge_10", "high_anchor_all", "t115_broad_stress_guard"),
        ("anchor_abs_ge_12", "high_anchor_all", "t115_broad_stress_guard"),
    ]
    keep_frame = pd.DataFrame(keep, columns=["stress_scenario", "subset", "policy"])
    display = keep_frame.merge(summary, on=["stress_scenario", "subset", "policy"], how="left").dropna(
        subset=["n_cases"]
    )
    rows: list[dict[str, object]] = []
    for _, row in display.iterrows():
        rows.append(
            {
                "Stress": compact_method_name(str(row["stress_scenario"])),
                "Subset": compact_method_name(str(row["subset"])),
                "Policy": compact_method_name(str(row["policy"])),
                "Cases": int(row["n_cases"]),
                "Safe": int(row["n_safe_recovered"]),
                "Unsafe": int(row["n_unsafe_releases"]),
                "Withheld": int(row["n_withheld"]),
                "Released MAE": round(float(row["released_selected_mae_bpm"]), 3)
                if pd.notna(row.get("released_selected_mae_bpm"))
                else np.nan,
            }
        )
    if perturbation is not None and not perturbation.empty:
        for policy in ["t114a_margin_robust_high_gate", "t115_broad_stress_guard"]:
            p_row = t115_perturbation_metric_row(perturbation, policy=policy)
            if not p_row.empty:
                rows.append(
                    {
                        "Stress": "feature perturbation",
                        "Subset": "high anchor",
                        "Policy": compact_method_name(policy),
                        "Cases": int(p_row["n_cases"]),
                        "Safe": int(p_row["n_safe_recovered"]),
                        "Unsafe": int(p_row["n_unsafe_releases"]),
                        "Withheld": int(p_row["n_withheld"]),
                        "Released MAE": round(float(p_row["released_selected_mae_bpm"]), 3)
                        if pd.notna(p_row.get("released_selected_mae_bpm"))
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def t115_residual_guard_table(residual: pd.DataFrame, subject: str | None = None) -> pd.DataFrame:
    if residual.empty:
        return pd.DataFrame()
    display = residual.copy()
    if subject is not None:
        display = display[display["subject"].eq(subject)].copy()
    if display.empty:
        return pd.DataFrame()
    display_cols = [
        "subject",
        "window_id",
        "stress_scenario",
        "selected_abs_error_bpm",
        "t115_broad_stress_guard_decision",
        "t115_broad_stress_guard_block_reason",
        "route_gap_bpm",
        "selected_to_anchor_support_ratio",
        "selector_score",
        "t115_guard_reason",
    ]
    display = display[[col for col in display_cols if col in display.columns]].copy()
    for col in ["selected_abs_error_bpm", "route_gap_bpm", "selected_to_anchor_support_ratio", "selector_score"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    display = display.rename(
        columns={
            "stress_scenario": "stress",
            "selected_abs_error_bpm": "selected abs err",
            "t115_broad_stress_guard_decision": "T115 decision",
            "t115_broad_stress_guard_block_reason": "T115 reason",
            "route_gap_bpm": "route gap",
            "selected_to_anchor_support_ratio": "support ratio",
            "selector_score": "selector score",
            "t115_guard_reason": "guard branch",
        }
    )
    return display.reset_index(drop=True)


def adult_t162_status_table(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty:
        return pd.DataFrame()
    display = status.copy()
    display["Share"] = (100.0 * pd.to_numeric(display["share"], errors="coerce")).round(1).astype(str) + "%"
    display["Mean released error"] = pd.to_numeric(display["mean_research_abs_error_bpm"], errors="coerce").round(3)
    display["Unsafe"] = pd.to_numeric(display["unsafe_releases"], errors="coerce").fillna(0).astype(int)
    display["Subjects"] = pd.to_numeric(display["n_subjects"], errors="coerce").fillna(0).astype(int)
    display["Status"] = display["product_status"].replace(
        {
            "release": "Released",
            "rescued_release": "Rescued release",
            "review_required": "Review required",
        }
    )
    return display[["Status", "Subjects", "Share", "Mean released error", "Unsafe", "example_subjects"]].rename(
        columns={"example_subjects": "Examples"}
    )


def adult_t162_claim_table(checklist: pd.DataFrame) -> pd.DataFrame:
    if checklist.empty:
        return pd.DataFrame()
    display = checklist[["claim_or_requirement", "status", "evidence", "external_gate"]].copy()
    return display.rename(
        columns={
            "claim_or_requirement": "Claim / requirement",
            "status": "Status",
            "evidence": "Evidence",
            "external_gate": "External gate",
        }
    )


def adult_t162_subject_table(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    display = panel[
        [
            "subject_id",
            "product_status",
            "product_output",
            "decision_source",
            "explanation_title",
            "evidence_chips",
            "recommended_action",
        ]
    ].copy()
    display["product_status"] = display["product_status"].replace(
        {
            "release": "Release",
            "rescued_release": "Rescued",
            "review_required": "Review",
        }
    )
    return display.rename(
        columns={
            "subject_id": "Subject",
            "product_status": "Status",
            "product_output": "Output",
            "decision_source": "Decision source",
            "explanation_title": "Explanation",
            "evidence_chips": "Evidence chips",
            "recommended_action": "Action",
        }
    )


def adult_t163_qa_table(qa: pd.DataFrame) -> pd.DataFrame:
    if qa.empty:
        return pd.DataFrame()
    display = qa[["test_id", "area", "check", "status", "risk_or_note"]].copy()
    return display.rename(
        columns={
            "test_id": "Test",
            "area": "Area",
            "check": "Check",
            "status": "Status",
            "risk_or_note": "Note",
        }
    )


def adult_t163_run_order_table(run_order: pd.DataFrame) -> pd.DataFrame:
    if run_order.empty:
        return pd.DataFrame()
    display = run_order[
        [
            "order_index",
            "run_id",
            "dataset",
            "readiness",
            "run_type",
            "claim_use",
            "next_task",
        ]
    ].copy()
    return display.rename(
        columns={
            "order_index": "Order",
            "run_id": "Run",
            "dataset": "Dataset",
            "readiness": "Readiness",
            "run_type": "Run type",
            "claim_use": "Claim use",
            "next_task": "Next task",
        }
    )


def adult_t163_gate_table(gate_matrix: pd.DataFrame) -> pd.DataFrame:
    if gate_matrix.empty:
        return pd.DataFrame()
    display = gate_matrix[["gate_id", "gate", "status", "evidence", "meaning"]].copy()
    return display.rename(
        columns={
            "gate_id": "Gate",
            "gate": "Question",
            "status": "Status",
            "evidence": "Evidence",
            "meaning": "Meaning",
        }
    )


def adult_t164_policy_table(policy_summary: pd.DataFrame) -> pd.DataFrame:
    if policy_summary.empty:
        return pd.DataFrame()
    display = policy_summary[
        [
            "policy",
            "n_total",
            "released",
            "withheld",
            "coverage",
            "released_mae_bpm",
            "released_rmse_bpm",
            "released_pearson_r",
            "unsafe_release_count",
            "unsafe_per_input",
        ]
    ].copy()
    for col in ["coverage", "released_mae_bpm", "released_rmse_bpm", "released_pearson_r", "unsafe_per_input"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    return display.rename(
        columns={
            "policy": "Policy",
            "n_total": "N",
            "released": "Released",
            "withheld": "Withheld",
            "coverage": "Coverage",
            "released_mae_bpm": "MAE BPM",
            "released_rmse_bpm": "RMSE BPM",
            "released_pearson_r": "Pearson r",
            "unsafe_release_count": "Unsafe releases",
            "unsafe_per_input": "Unsafe/input",
        }
    )


def adult_t164_audit_table(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    display = audit[["audit_id", "audit_item", "status", "interpretation"]].copy()
    return display.rename(
        columns={
            "audit_id": "Gate",
            "audit_item": "Audit item",
            "status": "Status",
            "interpretation": "Interpretation",
        }
    )


def adult_t164_data_table(data_check: pd.DataFrame) -> pd.DataFrame:
    if data_check.empty:
        return pd.DataFrame()
    display = data_check[["check_id", "item", "observed", "expected", "status", "note"]].copy()
    return display.rename(
        columns={
            "check_id": "Check",
            "item": "Item",
            "observed": "Observed",
            "expected": "Expected",
            "status": "Status",
            "note": "Note",
        }
    )


def adult_t164_repro_table(repro_check: pd.DataFrame) -> pd.DataFrame:
    if repro_check.empty:
        return pd.DataFrame()
    final = repro_check[repro_check["policy"].astype(str).eq("T160_physio_consistency_rescue_v1")].copy()
    display = final[["metric", "t160_reference", "t164_recomputed", "absolute_diff", "status"]].copy()
    for col in ["t160_reference", "t164_recomputed", "absolute_diff"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").round(6)
    return display.rename(
        columns={
            "metric": "Metric",
            "t160_reference": "T160 reference",
            "t164_recomputed": "T164 recomputed",
            "absolute_diff": "Absolute diff",
            "status": "Status",
        }
    )


def adult_t165_readiness_table(readiness: pd.DataFrame) -> pd.DataFrame:
    if readiness.empty:
        return pd.DataFrame()
    display = readiness[
        [
            "dataset",
            "priority",
            "t165_readiness",
            "local_video_file_count",
            "local_mat_count",
            "local_label_like_count",
            "local_total_gb",
            "access_model",
            "immediate_blocker",
            "next_action",
        ]
    ].copy()
    display["local_total_gb"] = pd.to_numeric(display["local_total_gb"], errors="coerce").round(3)
    return display.rename(
        columns={
            "dataset": "Dataset",
            "priority": "Priority",
            "t165_readiness": "Readiness",
            "local_video_file_count": "Videos",
            "local_mat_count": "MAT files",
            "local_label_like_count": "Label-like",
            "local_total_gb": "Local GB",
            "access_model": "Access model",
            "immediate_blocker": "Blocker",
            "next_action": "Next action",
        }
    )


def adult_t165_actions_table(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    display = actions[["action_id", "dataset", "action_type", "recipient_or_site", "required_user_action", "priority", "unblocks"]].copy()
    return display.rename(
        columns={
            "action_id": "Action",
            "dataset": "Dataset",
            "action_type": "Type",
            "recipient_or_site": "Recipient/site",
            "required_user_action": "Required action",
            "priority": "Priority",
            "unblocks": "Unblocks",
        }
    )


def adult_t165_plan_table(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame()
    display = plan[["step_id", "phase", "objective", "success_gate", "current_status"]].copy()
    return display.rename(
        columns={
            "step_id": "Step",
            "phase": "Phase",
            "objective": "Objective",
            "success_gate": "Success gate",
            "current_status": "Status",
        }
    )


def adult_t217_product_table_display(product: pd.DataFrame) -> pd.DataFrame:
    if product.empty:
        return pd.DataFrame()
    cols = [
        "sample_id",
        "dataset",
        "product_mode_label",
        "decision",
        "product_hr_bpm",
        "source",
        "bridge_source",
        "window_candidate_count",
        "repair_probability",
        "bridge_anchor_bpm",
        "eval_abs_error_bpm",
    ]
    display = product[[col for col in cols if col in product.columns]].copy()
    for col in ["product_hr_bpm", "bridge_anchor_bpm", "eval_abs_error_bpm", "window_candidate_count", "repair_probability"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    return display.rename(
        columns={
            "sample_id": "Sample",
            "dataset": "Dataset",
            "product_mode_label": "Mode",
            "decision": "Decision",
            "product_hr_bpm": "Product HR",
            "source": "Source",
            "bridge_source": "Bridge source",
            "window_candidate_count": "Candidates",
            "repair_probability": "Repair prob",
            "bridge_anchor_bpm": "Anchor HR",
            "eval_abs_error_bpm": "Eval abs error",
        }
    )


def t115_subject_stress_audit(audit: pd.DataFrame, subject: str) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    subject_rows = audit[
        audit["subject"].eq(subject) & audit["stress_direction"].eq("high") & audit["stress_scenario"].eq("anchor_abs_ge_6")
    ].copy()
    if subject_rows.empty:
        return pd.DataFrame()
    display_cols = [
        "stress_scenario",
        "window_id",
        "anchor_center_rr_bpm",
        "selected_pred_rr_bpm",
        "selected_abs_error_bpm",
        "route_gap_bpm",
        "selected_to_anchor_support_ratio",
        "t114a_margin_robust_high_gate_decision",
        "t115_broad_stress_guard_decision",
        "t115_broad_stress_guard_block_reason",
        "t115_low_support_midgap_guard",
        "t115_high_support_weak_anchor_midgap_guard",
    ]
    display = subject_rows[[col for col in display_cols if col in subject_rows.columns]].copy()
    display = display.sort_values(["window_id"]).reset_index(drop=True)
    for col in [
        "anchor_center_rr_bpm",
        "selected_pred_rr_bpm",
        "selected_abs_error_bpm",
        "route_gap_bpm",
        "selected_to_anchor_support_ratio",
    ]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    display = display.rename(
        columns={
            "stress_scenario": "stress",
            "anchor_center_rr_bpm": "anchor RR",
            "selected_pred_rr_bpm": "selected RR",
            "selected_abs_error_bpm": "selected abs err",
            "route_gap_bpm": "route gap",
            "selected_to_anchor_support_ratio": "support ratio",
            "t114a_margin_robust_high_gate_decision": "T114A decision",
            "t115_broad_stress_guard_decision": "T115 decision",
            "t115_broad_stress_guard_block_reason": "T115 reason",
            "t115_low_support_midgap_guard": "low-support guard",
            "t115_high_support_weak_anchor_midgap_guard": "high-support guard",
        }
    )
    return display


def t120_metric_row(summary: pd.DataFrame, *, validation_set: str, scenario: str, policy: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    rows = summary[
        summary["validation_set"].eq(validation_set)
        & summary["base_stress_scenario"].eq(scenario)
        & summary["policy"].eq(policy)
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t120_policy_summary_table(summary: pd.DataFrame, validation_set: str | None = None, scenario: str | None = None) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    display = summary.copy()
    if validation_set is not None:
        display = display[display["validation_set"].eq(validation_set)].copy()
    if scenario is not None:
        display = display[display["base_stress_scenario"].eq(scenario)].copy()
    if display.empty:
        return pd.DataFrame()
    keep_order = [
        "t115_broad_stress_guard_fixed",
        "t117_ambiguous_support_buffer_candidate",
        "t119_route_risk_dev_zero_safe_loss",
        "t120_subject_aware_tail_rate_gate",
    ]
    display = display[display["policy"].isin(keep_order)].copy()
    display["policy_order"] = display["policy"].map({policy: index for index, policy in enumerate(keep_order)})
    display = display.sort_values(["validation_set", "base_stress_scenario", "policy_order"])
    out = display[
        [
            "validation_set",
            "base_stress_scenario",
            "policy",
            "n_safe_recovered",
            "n_unsafe_releases",
            "n_withheld",
            "safe_loss_vs_t119",
            "unsafe_reduction_vs_t119",
        ]
    ].copy()
    out["validation_set"] = out["validation_set"].map(compact_method_name)
    out["base_stress_scenario"] = out["base_stress_scenario"].map(compact_method_name)
    out["policy"] = out["policy"].map(compact_method_name)
    for col in ["n_safe_recovered", "n_unsafe_releases", "n_withheld", "safe_loss_vs_t119", "unsafe_reduction_vs_t119"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out.rename(
        columns={
            "validation_set": "Stress family",
            "base_stress_scenario": "Anchor stress",
            "policy": "Policy",
            "n_safe_recovered": "Safe",
            "n_unsafe_releases": "Unsafe",
            "n_withheld": "Review-only",
            "safe_loss_vs_t119": "Safe loss vs T119",
            "unsafe_reduction_vs_t119": "Unsafe reduction vs T119",
        }
    ).reset_index(drop=True)


def t120_loso_summary_table(loso: pd.DataFrame) -> pd.DataFrame:
    if loso.empty:
        return pd.DataFrame()
    display = (
        loso.groupby("selection_mode")
        .agg(
            heldout_subjects=("heldout_subject", "nunique"),
            t119_safe=("test_t119_safe_recovered", "sum"),
            t119_unsafe=("test_t119_unsafe_releases", "sum"),
            t120_safe=("test_n_safe_recovered", "sum"),
            t120_unsafe=("test_n_unsafe_releases", "sum"),
            t120_review_only=("test_n_withheld", "sum"),
            safe_loss_vs_t119=("test_safe_loss_vs_t119", "sum"),
            unsafe_reduction_vs_t119=("test_unsafe_reduction_vs_t119", "sum"),
        )
        .reset_index()
    )
    for col in display.columns:
        if col != "selection_mode":
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0).astype(int)
    return display.rename(
        columns={
            "selection_mode": "Mode",
            "heldout_subjects": "Held-out subjects",
            "t119_safe": "T119 safe",
            "t119_unsafe": "T119 unsafe",
            "t120_safe": "T120 safe",
            "t120_unsafe": "T120 unsafe",
            "t120_review_only": "T120 review-only",
            "safe_loss_vs_t119": "Safe loss vs T119",
            "unsafe_reduction_vs_t119": "Unsafe reduction vs T119",
        }
    )


def t120_context_row(context: pd.DataFrame, *, validation_set: str, scenario: str, subject: str) -> pd.Series:
    if context.empty:
        return pd.Series(dtype=object)
    rows = context[
        context["validation_set"].eq(validation_set)
        & context["base_stress_scenario"].eq(scenario)
        & context["subject"].astype(str).eq(str(subject))
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t120_episode_context_table(context: pd.DataFrame, validation_set: str, scenario: str) -> pd.DataFrame:
    if context.empty:
        return pd.DataFrame()
    display = context[
        context["validation_set"].eq(validation_set) & context["base_stress_scenario"].eq(scenario)
    ].copy()
    if display.empty:
        return pd.DataFrame()
    cols = [
        "subject",
        "t120_episode_n_t114a_candidates",
        "t120_episode_tail_risk_rate",
        "t120_episode_risk_mean",
        "t120_episode_risk_p90",
        "t120_episode_instability_flag",
        "t120_strict_risk_threshold",
    ]
    display = display[[col for col in cols if col in display.columns]].copy()
    for col in [
        "t120_episode_tail_risk_rate",
        "t120_episode_risk_mean",
        "t120_episode_risk_p90",
        "t120_strict_risk_threshold",
    ]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    if "t120_episode_n_t114a_candidates" in display.columns:
        display["t120_episode_n_t114a_candidates"] = pd.to_numeric(
            display["t120_episode_n_t114a_candidates"], errors="coerce"
        ).fillna(0).astype(int)
    return display.sort_values("t120_episode_tail_risk_rate", ascending=False).rename(
        columns={
            "subject": "Subject",
            "t120_episode_n_t114a_candidates": "T114A candidates",
            "t120_episode_tail_risk_rate": "Episode tail-risk rate",
            "t120_episode_risk_mean": "Mean risk",
            "t120_episode_risk_p90": "P90 risk",
            "t120_episode_instability_flag": "Instability flag",
            "t120_strict_risk_threshold": "Strict threshold",
        }
    ).reset_index(drop=True)


def t120_subject_route_risk_audit(
    audit: pd.DataFrame,
    *,
    subject: str,
    validation_set: str,
    scenario: str,
    max_rows: int = 24,
) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    display = audit[
        audit["subject"].astype(str).eq(str(subject))
        & audit["validation_set"].eq(validation_set)
        & audit["base_stress_scenario"].eq(scenario)
        & coerce_bool_series(audit["t117_t114a_release"])
    ].copy()
    if display.empty:
        return pd.DataFrame()
    display["t120_blocked_by_subject_gate"] = coerce_bool_series(display["t119_route_risk_release"]) & (
        ~coerce_bool_series(display["t120_subject_aware_release"])
    )
    display = display.sort_values(
        ["t120_blocked_by_subject_gate", "selected_unsafe_release", "t119_route_risk_score"],
        ascending=[False, False, False],
    )
    cols = [
        "window_id",
        "route_gap_bpm",
        "selector_score",
        "selected_to_anchor_support_ratio",
        "t119_route_risk_score",
        "t120_episode_tail_risk_rate",
        "t119_route_risk_decision",
        "t120_subject_aware_decision",
        "t120_blocked_by_subject_gate",
        "selected_safe_recovery",
        "selected_unsafe_release",
        "selected_abs_error_bpm",
    ]
    display = display[[col for col in cols if col in display.columns]].head(max_rows).copy()
    for col in [
        "route_gap_bpm",
        "selector_score",
        "selected_to_anchor_support_ratio",
        "t119_route_risk_score",
        "t120_episode_tail_risk_rate",
        "selected_abs_error_bpm",
    ]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    return display.rename(
        columns={
            "window_id": "Window",
            "route_gap_bpm": "Route gap",
            "selector_score": "Selector score",
            "selected_to_anchor_support_ratio": "Support ratio",
            "t119_route_risk_score": "T119 risk",
            "t120_episode_tail_risk_rate": "Episode tail-risk",
            "t119_route_risk_decision": "T119 decision",
            "t120_subject_aware_decision": "T120 decision",
            "t120_blocked_by_subject_gate": "Blocked by T120",
            "selected_safe_recovery": "Safe recovery",
            "selected_unsafe_release": "Unsafe release",
            "selected_abs_error_bpm": "Selected abs err",
        }
    ).reset_index(drop=True)


def t120_subject_policy_counts(audit: pd.DataFrame, *, subject: str, validation_set: str, scenario: str) -> dict[str, int]:
    if audit.empty:
        return {"t119_safe": 0, "t119_unsafe": 0, "t120_safe": 0, "t120_unsafe": 0, "t120_review_only": 0}
    rows = audit[
        audit["subject"].astype(str).eq(str(subject))
        & audit["validation_set"].eq(validation_set)
        & audit["base_stress_scenario"].eq(scenario)
    ].copy()
    if rows.empty:
        return {"t119_safe": 0, "t119_unsafe": 0, "t120_safe": 0, "t120_unsafe": 0, "t120_review_only": 0}
    t119_release = coerce_bool_series(rows["t119_route_risk_release"])
    t120_release = coerce_bool_series(rows["t120_subject_aware_release"])
    safe = coerce_bool_series(rows["selected_safe_recovery"])
    unsafe = coerce_bool_series(rows["selected_unsafe_release"])
    return {
        "t119_safe": int((t119_release & safe).sum()),
        "t119_unsafe": int((t119_release & unsafe).sum()),
        "t120_safe": int((t120_release & safe).sum()),
        "t120_unsafe": int((t120_release & unsafe).sum()),
        "t120_review_only": int((~t120_release).sum()),
        "blocked_by_t120": int((t119_release & ~t120_release).sum()),
    }


def t123_mode_options(config: pd.DataFrame) -> list[str]:
    if config.empty or "mode" not in config.columns:
        return []
    return config["mode"].dropna().astype(str).tolist()


def t123_mode_label_map(config: pd.DataFrame) -> dict[str, str]:
    if config.empty:
        return {}
    label_col = "mode_label" if "mode_label" in config.columns else "mode"
    return {
        str(row["mode"]): str(row[label_col])
        for _, row in config.dropna(subset=["mode"]).iterrows()
    }


def t123_mode_label(config: pd.DataFrame, mode: str) -> str:
    return t123_mode_label_map(config).get(str(mode), compact_method_name(str(mode)))


def t123_mode_config_row(config: pd.DataFrame, mode: str) -> pd.Series:
    if config.empty or "mode" not in config.columns:
        return pd.Series(dtype=object)
    rows = config[config["mode"].astype(str).eq(str(mode))]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t123_mode_config_table(config: pd.DataFrame) -> pd.DataFrame:
    if config.empty:
        return pd.DataFrame()
    keep = [
        "mode_label",
        "intended_setting",
        "tail_rate_threshold",
        "strict_risk_threshold",
        "cold_start_force_review",
        "cold_start_min_prior_windows",
        "cold_start_risk_threshold",
        "warning_copy",
    ]
    display = config[[col for col in keep if col in config.columns]].copy()
    for col in ["tail_rate_threshold", "strict_risk_threshold", "cold_start_risk_threshold"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    if "cold_start_min_prior_windows" in display.columns:
        display["cold_start_min_prior_windows"] = pd.to_numeric(
            display["cold_start_min_prior_windows"], errors="coerce"
        ).fillna(0).astype(int)
    return display.rename(
        columns={
            "mode_label": "Mode",
            "intended_setting": "Intended setting",
            "tail_rate_threshold": "Tail-rate threshold",
            "strict_risk_threshold": "Strict risk threshold",
            "cold_start_force_review": "Cold-start force review",
            "cold_start_min_prior_windows": "Min prior windows",
            "cold_start_risk_threshold": "Cold-start risk threshold",
            "warning_copy": "Product warning copy",
        }
    ).reset_index(drop=True)


def t123_metric_row(summary: pd.DataFrame, *, validation_set: str, scenario: str, mode: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    rows = summary[
        summary["validation_set"].eq(validation_set)
        & summary["base_stress_scenario"].eq(scenario)
        & summary["mode"].eq(mode)
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t123_policy_summary_table(
    summary: pd.DataFrame,
    *,
    validation_set: str | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    display = summary.copy()
    if validation_set is not None:
        display = display[display["validation_set"].eq(validation_set)].copy()
    if scenario is not None:
        display = display[display["base_stress_scenario"].eq(scenario)].copy()
    if display.empty:
        return pd.DataFrame()
    mode_order = {"eldercare_balanced": 0, "hospital_cautious": 1, "infant_high_caution": 2}
    display["mode_order"] = display["mode"].map(mode_order).fillna(99)
    display = display.sort_values(["validation_set", "base_stress_scenario", "mode_order"])
    cols = [
        "validation_set",
        "base_stress_scenario",
        "mode_label",
        "intended_setting",
        "n_safe_recovered",
        "n_unsafe_releases",
        "n_withheld",
        "safe_loss_vs_t119",
        "unsafe_reduction_vs_t119",
        "safe_delta_vs_t120_full",
    ]
    out = display[[col for col in cols if col in display.columns]].copy()
    if "validation_set" in out.columns:
        out["validation_set"] = out["validation_set"].map(compact_method_name)
    if "base_stress_scenario" in out.columns:
        out["base_stress_scenario"] = out["base_stress_scenario"].map(compact_method_name)
    for col in [
        "n_safe_recovered",
        "n_unsafe_releases",
        "n_withheld",
        "safe_loss_vs_t119",
        "unsafe_reduction_vs_t119",
        "safe_delta_vs_t120_full",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out.rename(
        columns={
            "validation_set": "Stress family",
            "base_stress_scenario": "Anchor stress",
            "mode_label": "Mode",
            "intended_setting": "Setting",
            "n_safe_recovered": "Safe",
            "n_unsafe_releases": "Unsafe",
            "n_withheld": "Review-only",
            "safe_loss_vs_t119": "Safe loss vs T119",
            "unsafe_reduction_vs_t119": "Unsafe reduction",
            "safe_delta_vs_t120_full": "Safe delta vs T120 full",
        }
    ).reset_index(drop=True)


def t123_cold_start_table(cold: pd.DataFrame, *, validation_set: str, scenario: str) -> pd.DataFrame:
    if cold.empty:
        return pd.DataFrame()
    display = cold[
        cold["validation_set"].eq(validation_set) & cold["base_stress_scenario"].eq(scenario)
    ].copy()
    if display.empty:
        return pd.DataFrame()
    order = {"first_two_windows": 0, "later_windows": 1}
    display["phase_order"] = display["cold_start_phase"].map(order).fillna(99)
    mode_order = {"eldercare_balanced": 0, "hospital_cautious": 1, "infant_high_caution": 2}
    display["mode_order"] = display["mode"].map(mode_order).fillna(99)
    display = display.sort_values(["phase_order", "mode_order"])
    cols = ["cold_start_phase", "mode_label", "n_safe_recovered", "n_unsafe_releases", "n_withheld"]
    out = display[[col for col in cols if col in display.columns]].copy()
    for col in ["n_safe_recovered", "n_unsafe_releases", "n_withheld"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out.rename(
        columns={
            "cold_start_phase": "Phase",
            "mode_label": "Mode",
            "n_safe_recovered": "Safe",
            "n_unsafe_releases": "Unsafe",
            "n_withheld": "Review-only",
        }
    ).reset_index(drop=True)


def t123_subject_summary_table(
    subject_summary: pd.DataFrame,
    *,
    validation_set: str,
    scenario: str,
    mode: str,
) -> pd.DataFrame:
    if subject_summary.empty:
        return pd.DataFrame()
    display = subject_summary[
        subject_summary["validation_set"].eq(validation_set)
        & subject_summary["base_stress_scenario"].eq(scenario)
        & subject_summary["mode"].eq(mode)
    ].copy()
    if display.empty:
        return pd.DataFrame()
    cols = [
        "subject",
        "t119_safe",
        "t119_unsafe",
        "mode_safe",
        "mode_unsafe",
        "mode_withheld",
        "mode_blocked_from_t119",
        "max_causal_tail_rate",
        "min_prior_windows",
    ]
    out = display[[col for col in cols if col in display.columns]].copy()
    for col in ["t119_safe", "t119_unsafe", "mode_safe", "mode_unsafe", "mode_withheld", "mode_blocked_from_t119"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["max_causal_tail_rate", "min_prior_windows"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out.sort_values(["mode_blocked_from_t119", "max_causal_tail_rate"], ascending=False).rename(
        columns={
            "subject": "Subject",
            "t119_safe": "T119 safe",
            "t119_unsafe": "T119 unsafe",
            "mode_safe": "Mode safe",
            "mode_unsafe": "Mode unsafe",
            "mode_withheld": "Mode review-only",
            "mode_blocked_from_t119": "Blocked from T119",
            "max_causal_tail_rate": "Max causal tail-rate",
            "min_prior_windows": "Min prior windows",
        }
    ).reset_index(drop=True)


def t123_subject_policy_counts(
    audit: pd.DataFrame,
    *,
    subject: str,
    validation_set: str,
    scenario: str,
    mode: str,
) -> dict[str, int]:
    empty = {
        "t119_safe": 0,
        "t119_unsafe": 0,
        "mode_safe": 0,
        "mode_unsafe": 0,
        "mode_review_only": 0,
        "blocked_by_mode": 0,
        "current_episode_blocks": 0,
        "cold_start_blocks": 0,
    }
    release_col = f"t123_{mode}_release"
    current_col = f"t123_{mode}_block_current_episode_risk"
    cold_col = f"t123_{mode}_block_cold_start"
    if audit.empty or release_col not in audit.columns:
        return empty
    rows = audit[
        audit["subject"].astype(str).eq(str(subject))
        & audit["validation_set"].eq(validation_set)
        & audit["base_stress_scenario"].eq(scenario)
    ].copy()
    if rows.empty:
        return empty
    t119_release = coerce_bool_series(rows["t119_route_risk_release"])
    mode_release = coerce_bool_series(rows[release_col])
    safe = coerce_bool_series(rows["selected_safe_recovery"])
    unsafe = coerce_bool_series(rows["selected_unsafe_release"])
    blocked = t119_release & ~mode_release
    current_blocks = coerce_bool_series(rows[current_col]) if current_col in rows.columns else pd.Series(False, index=rows.index)
    cold_blocks = coerce_bool_series(rows[cold_col]) if cold_col in rows.columns else pd.Series(False, index=rows.index)
    return {
        "t119_safe": int((t119_release & safe).sum()),
        "t119_unsafe": int((t119_release & unsafe).sum()),
        "mode_safe": int((mode_release & safe).sum()),
        "mode_unsafe": int((mode_release & unsafe).sum()),
        "mode_review_only": int((~mode_release).sum()),
        "blocked_by_mode": int(blocked.sum()),
        "current_episode_blocks": int((blocked & current_blocks).sum()),
        "cold_start_blocks": int((blocked & cold_blocks).sum()),
    }


def t123_subject_safety_audit(
    audit: pd.DataFrame,
    *,
    subject: str,
    validation_set: str,
    scenario: str,
    mode: str,
    max_rows: int = 24,
) -> pd.DataFrame:
    release_col = f"t123_{mode}_release"
    decision_col = f"t123_{mode}_decision"
    warning_col = f"t123_{mode}_warning_reason"
    current_col = f"t123_{mode}_block_current_episode_risk"
    cold_col = f"t123_{mode}_block_cold_start"
    if audit.empty or release_col not in audit.columns:
        return pd.DataFrame()
    display = audit[
        audit["subject"].astype(str).eq(str(subject))
        & audit["validation_set"].eq(validation_set)
        & audit["base_stress_scenario"].eq(scenario)
        & coerce_bool_series(audit["t117_t114a_release"])
    ].copy()
    if display.empty:
        return pd.DataFrame()
    display["t123_blocked_by_mode"] = coerce_bool_series(display["t119_route_risk_release"]) & (
        ~coerce_bool_series(display[release_col])
    )
    sort_cols = ["t123_blocked_by_mode", "selected_unsafe_release", "t119_route_risk_score"]
    display = display.sort_values(sort_cols, ascending=[False, False, False])
    cols = [
        "window_id",
        "t119_route_risk_score",
        "t122_causal_current_tail_rate",
        "t122_prior_tail_rate",
        "prior_n_windows",
        decision_col,
        warning_col,
        current_col,
        cold_col,
        "t123_blocked_by_mode",
        "selected_safe_recovery",
        "selected_unsafe_release",
        "selected_abs_error_bpm",
    ]
    out = display[[col for col in cols if col in display.columns]].head(max_rows).copy()
    for col in [
        "t119_route_risk_score",
        "t122_causal_current_tail_rate",
        "t122_prior_tail_rate",
        "prior_n_windows",
        "selected_abs_error_bpm",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out.rename(
        columns={
            "window_id": "Window",
            "t119_route_risk_score": "T119 risk",
            "t122_causal_current_tail_rate": "Current tail-rate",
            "t122_prior_tail_rate": "Prior tail-rate",
            "prior_n_windows": "Prior windows",
            decision_col: "Mode decision",
            warning_col: "Warning reason",
            current_col: "Current-risk block",
            cold_col: "Cold-start block",
            "t123_blocked_by_mode": "Blocked by mode",
            "selected_safe_recovery": "Safe recovery",
            "selected_unsafe_release": "Unsafe release",
            "selected_abs_error_bpm": "Selected abs err",
        }
    ).reset_index(drop=True)


def t125_metric_row(summary: pd.DataFrame, *, validation_set: str, scenario: str, mode: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    rows = summary[
        summary["validation_set"].eq(validation_set)
        & summary["base_stress_scenario"].eq(scenario)
        & summary["mode"].eq(mode)
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def t125_policy_summary_table(
    summary: pd.DataFrame,
    *,
    validation_set: str | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    display = summary.copy()
    if validation_set is not None:
        display = display[display["validation_set"].eq(validation_set)].copy()
    if scenario is not None:
        display = display[display["base_stress_scenario"].eq(scenario)].copy()
    if display.empty:
        return pd.DataFrame()
    mode_order = {"eldercare_balanced": 0, "hospital_cautious": 1, "infant_high_caution": 2}
    display["mode_order"] = display["mode"].map(mode_order).fillna(99)
    display = display.sort_values(["validation_set", "base_stress_scenario", "mode_order"])
    cols = [
        "validation_set",
        "base_stress_scenario",
        "mode_label",
        "t123_safe_recovered",
        "t123_unsafe_releases",
        "t123_withheld",
        "t125_safe_recovered",
        "t125_unsafe_releases",
        "t125_withheld",
        "delta_safe_vs_t123",
        "delta_unsafe_vs_t123",
        "delta_withheld_vs_t123",
        "recovered_safe_from_t123",
        "recovered_unsafe_from_t123",
        "current_threshold",
        "cold_threshold",
        "candidate_status",
    ]
    out = display[[col for col in cols if col in display.columns]].copy()
    if "validation_set" in out.columns:
        out["validation_set"] = out["validation_set"].map(compact_method_name)
    if "base_stress_scenario" in out.columns:
        out["base_stress_scenario"] = out["base_stress_scenario"].map(compact_method_name)
    int_cols = [
        "t123_safe_recovered",
        "t123_unsafe_releases",
        "t123_withheld",
        "t125_safe_recovered",
        "t125_unsafe_releases",
        "t125_withheld",
        "delta_safe_vs_t123",
        "delta_unsafe_vs_t123",
        "delta_withheld_vs_t123",
        "recovered_safe_from_t123",
        "recovered_unsafe_from_t123",
    ]
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["current_threshold", "cold_threshold"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out.rename(
        columns={
            "validation_set": "Stress family",
            "base_stress_scenario": "Anchor stress",
            "mode_label": "Mode",
            "t123_safe_recovered": "T123 safe",
            "t123_unsafe_releases": "T123 unsafe",
            "t123_withheld": "T123 review-only",
            "t125_safe_recovered": "T125 safe",
            "t125_unsafe_releases": "T125 unsafe",
            "t125_withheld": "T125 review-only",
            "delta_safe_vs_t123": "Delta safe",
            "delta_unsafe_vs_t123": "Delta unsafe",
            "delta_withheld_vs_t123": "Delta review-only",
            "recovered_safe_from_t123": "Recovered safe",
            "recovered_unsafe_from_t123": "Recovered unsafe",
            "current_threshold": "Current risk floor",
            "cold_threshold": "Cold risk floor",
            "candidate_status": "Status",
        }
    ).reset_index(drop=True)


def t125_recovery_summary_table(
    recovery: pd.DataFrame,
    *,
    validation_set: str,
    scenario: str,
) -> pd.DataFrame:
    if recovery.empty:
        return pd.DataFrame()
    display = recovery[
        recovery["validation_set"].eq(validation_set) & recovery["base_stress_scenario"].eq(scenario)
    ].copy()
    if display.empty:
        return pd.DataFrame()
    mode_order = {"eldercare_balanced": 0, "hospital_cautious": 1, "infant_high_caution": 2}
    display["mode_order"] = display["mode"].map(mode_order).fillna(99)
    display = display.sort_values("mode_order")
    cols = [
        "mode_label",
        "n_recovered_from_t123",
        "recovered_safe",
        "recovered_unsafe",
        "recovered_cold_start_unknown",
        "mean_recovered_risk",
        "max_recovered_risk",
    ]
    out = display[[col for col in cols if col in display.columns]].copy()
    for col in ["n_recovered_from_t123", "recovered_safe", "recovered_unsafe", "recovered_cold_start_unknown"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["mean_recovered_risk", "max_recovered_risk"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out.rename(
        columns={
            "mode_label": "Mode",
            "n_recovered_from_t123": "Recovered from T123",
            "recovered_safe": "Recovered safe",
            "recovered_unsafe": "Recovered unsafe",
            "recovered_cold_start_unknown": "Cold-start recovered",
            "mean_recovered_risk": "Mean recovered risk",
            "max_recovered_risk": "Max recovered risk",
        }
    ).reset_index(drop=True)


def t125_learned_refiner_table(learned: pd.DataFrame, *, validation_set: str, scenario: str) -> pd.DataFrame:
    if learned.empty:
        return pd.DataFrame()
    display = learned[
        learned["validation_set"].eq(validation_set) & learned["base_stress_scenario"].eq(scenario)
    ].copy()
    if display.empty:
        return pd.DataFrame()
    mode_order = {"eldercare_balanced": 0, "hospital_cautious": 1, "infant_high_caution": 2}
    model_order = {"rich_logistic": 0, "random_forest": 1, "extra_trees": 2}
    display["mode_order"] = display["mode"].map(mode_order).fillna(99)
    display["model_order"] = display["model"].map(model_order).fillna(99)
    display = display.sort_values(["mode_order", "model_order"])
    cols = [
        "model",
        "mode_label",
        "transfer_auc",
        "transfer_average_precision",
        "additional_safe",
        "additional_unsafe",
        "total_safe",
        "total_unsafe",
        "total_withheld",
        "candidate_status",
    ]
    out = display[[col for col in cols if col in display.columns]].copy()
    for col in ["additional_safe", "additional_unsafe", "total_safe", "total_unsafe", "total_withheld"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["transfer_auc", "transfer_average_precision"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out.rename(
        columns={
            "model": "Model",
            "mode_label": "Mode",
            "transfer_auc": "Transfer AUC",
            "transfer_average_precision": "Transfer AP",
            "additional_safe": "Additional safe",
            "additional_unsafe": "Additional unsafe",
            "total_safe": "Total safe",
            "total_unsafe": "Total unsafe",
            "total_withheld": "Total review-only",
            "candidate_status": "Status",
        }
    ).reset_index(drop=True)


def t125_subject_policy_counts(
    audit: pd.DataFrame,
    *,
    subject: str,
    validation_set: str,
    scenario: str,
    mode: str,
) -> dict[str, int]:
    empty = {
        "t123_safe": 0,
        "t123_unsafe": 0,
        "t123_review_only": 0,
        "t125_safe": 0,
        "t125_unsafe": 0,
        "t125_review_only": 0,
        "recovered_from_t123": 0,
        "recovered_safe": 0,
        "recovered_unsafe": 0,
        "current_floor_blocks": 0,
        "cold_floor_blocks": 0,
    }
    t123_release_col = f"t123_{mode}_release"
    t125_release_col = f"t125_{mode}_risk_floor_release"
    recovered_col = f"t125_{mode}_risk_floor_recovered_from_t123"
    current_col = f"t125_{mode}_risk_floor_block_current_episode_risk"
    cold_col = f"t125_{mode}_risk_floor_block_cold_start"
    if audit.empty or t123_release_col not in audit.columns or t125_release_col not in audit.columns:
        return empty
    rows = audit[
        audit["subject"].astype(str).eq(str(subject))
        & audit["validation_set"].eq(validation_set)
        & audit["base_stress_scenario"].eq(scenario)
    ].copy()
    if rows.empty:
        return empty
    t123_release = coerce_bool_series(rows[t123_release_col])
    t125_release = coerce_bool_series(rows[t125_release_col])
    recovered = coerce_bool_series(rows[recovered_col]) if recovered_col in rows.columns else pd.Series(False, index=rows.index)
    safe = coerce_bool_series(rows["selected_safe_recovery"])
    unsafe = coerce_bool_series(rows["selected_unsafe_release"])
    current_blocks = coerce_bool_series(rows[current_col]) if current_col in rows.columns else pd.Series(False, index=rows.index)
    cold_blocks = coerce_bool_series(rows[cold_col]) if cold_col in rows.columns else pd.Series(False, index=rows.index)
    return {
        "t123_safe": int((t123_release & safe).sum()),
        "t123_unsafe": int((t123_release & unsafe).sum()),
        "t123_review_only": int((~t123_release).sum()),
        "t125_safe": int((t125_release & safe).sum()),
        "t125_unsafe": int((t125_release & unsafe).sum()),
        "t125_review_only": int((~t125_release).sum()),
        "recovered_from_t123": int(recovered.sum()),
        "recovered_safe": int((recovered & safe).sum()),
        "recovered_unsafe": int((recovered & unsafe).sum()),
        "current_floor_blocks": int((~t125_release & current_blocks).sum()),
        "cold_floor_blocks": int((~t125_release & cold_blocks).sum()),
    }


def t125_subject_refinement_audit(
    audit: pd.DataFrame,
    *,
    subject: str,
    validation_set: str,
    scenario: str,
    mode: str,
    max_rows: int = 24,
) -> pd.DataFrame:
    t123_decision_col = f"t123_{mode}_decision"
    t125_release_col = f"t125_{mode}_risk_floor_release"
    t125_decision_col = f"t125_{mode}_risk_floor_decision"
    recovered_col = f"t125_{mode}_risk_floor_recovered_from_t123"
    current_col = f"t125_{mode}_risk_floor_block_current_episode_risk"
    cold_col = f"t125_{mode}_risk_floor_block_cold_start"
    if audit.empty or t125_release_col not in audit.columns:
        return pd.DataFrame()
    display = audit[
        audit["subject"].astype(str).eq(str(subject))
        & audit["validation_set"].eq(validation_set)
        & audit["base_stress_scenario"].eq(scenario)
        & coerce_bool_series(audit["t117_t114a_release"])
    ].copy()
    if display.empty:
        return pd.DataFrame()
    display["t125_recovered_from_t123"] = (
        coerce_bool_series(display[recovered_col]) if recovered_col in display.columns else pd.Series(False, index=display.index)
    )
    display = display.sort_values(
        ["t125_recovered_from_t123", "selected_unsafe_release", "t119_route_risk_score"],
        ascending=[False, False, False],
    )
    cols = [
        "window_id",
        "t119_route_risk_score",
        "t122_causal_current_tail_rate",
        "prior_n_windows",
        t123_decision_col,
        t125_decision_col,
        "t125_recovered_from_t123",
        current_col,
        cold_col,
        "selected_safe_recovery",
        "selected_unsafe_release",
        "selected_abs_error_bpm",
    ]
    out = display[[col for col in cols if col in display.columns]].head(max_rows).copy()
    for col in ["t119_route_risk_score", "t122_causal_current_tail_rate", "prior_n_windows", "selected_abs_error_bpm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out.rename(
        columns={
            "window_id": "Window",
            "t119_route_risk_score": "T119 risk",
            "t122_causal_current_tail_rate": "Current tail-rate",
            "prior_n_windows": "Prior windows",
            t123_decision_col: "T123 decision",
            t125_decision_col: "T125 decision",
            "t125_recovered_from_t123": "Recovered from T123",
            current_col: "Current-risk floor block",
            cold_col: "Cold-start floor block",
            "selected_safe_recovery": "Safe recovery",
            "selected_unsafe_release": "Unsafe release",
            "selected_abs_error_bpm": "Selected abs err",
        }
    ).reset_index(drop=True)


def t111_subject_stress_audit(audit: pd.DataFrame, subject: str) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    subject_rows = audit[audit["subject"].eq(subject)].copy()
    if subject_rows.empty:
        return pd.DataFrame()
    display_cols = [
        "stress_direction",
        "window_id",
        "anchor_center_rr_bpm",
        "selected_pred_rr_bpm",
        "selected_abs_error_bpm",
        "selected_safe_recovery",
        "selected_unsafe_release",
        "t111_direction_aware_product_gate_decision",
        "t111_direction_aware_product_gate_block_reason",
        "overshoot_guard",
        "weak_depth_against_trusted_anchor_guard",
    ]
    display = subject_rows[[col for col in display_cols if col in subject_rows.columns]].copy()
    display = display.sort_values(["stress_direction", "window_id"]).reset_index(drop=True)
    for col in ["anchor_center_rr_bpm", "selected_pred_rr_bpm", "selected_abs_error_bpm"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
    display = display.rename(
        columns={
            "stress_direction": "stress",
            "anchor_center_rr_bpm": "anchor RR",
            "selected_pred_rr_bpm": "selected RR",
            "selected_abs_error_bpm": "selected abs err",
            "selected_safe_recovery": "safe",
            "selected_unsafe_release": "unsafe",
            "t111_direction_aware_product_gate_decision": "T111 decision",
            "t111_direction_aware_product_gate_block_reason": "T111 reason",
            "overshoot_guard": "overshoot guard",
            "weak_depth_against_trusted_anchor_guard": "weak-depth guard",
        }
    )
    return display


def evidence_badge(*, level: str, scope: str, boundary: str, mode: str = "warn") -> None:
    chip_class = {"good": "vs-chip-good", "warn": "vs-chip-warn", "bad": "vs-chip-bad"}.get(mode, "vs-chip-warn")
    st.markdown(
        f"<div class='vs-status'>"
        f"<span class='vs-chip {chip_class}'>{level}</span>"
        f"<span class='vs-chip'>{scope}</span>"
        f"<span>{boundary}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def cambridge_monitor() -> None:
    source = st.sidebar.selectbox(
        "Source",
        [
            "T125 risk-floor safety-mode source",
            "T123 causal-current safety-mode source",
            "T120 subject-aware route-risk source",
            "T115 broad-stress guarded route",
            "T111 gated source-validity route",
            "T98 calibrated interval + source-risk product",
            "T94 latent-state tracker / T95 LOSO validation",
            "T87 high-RR harmonic calibration",
            "T58 aligned/harmonic neonatal prototype",
            "T39 legacy trend baseline",
        ],
    )
    if (
        source.startswith("T125")
        or source.startswith("T123")
        or source.startswith("T120")
        or source.startswith("T115")
        or source.startswith("T111")
        or source.startswith("T107")
    ):
        is_t125_view = source.startswith("T125")
        is_t123_view = source.startswith("T123")
        is_t120_view = source.startswith("T120")
        is_t115_view = source.startswith("T115")
        windows = load_cambridge_t107_windows()
        summary = load_cambridge_t107_policy_summary()
        review_audit = load_cambridge_t107_review_audit()
        selected_fallbacks = load_cambridge_t107_selected_fallbacks()
        interval_calibration = load_cambridge_t107_interval_calibration()
        route_residuals = load_cambridge_t107_route_residuals()
        t111_case_audit = load_cambridge_t111_case_audit()
        t111_policy_summary = load_cambridge_t111_policy_summary()
        t111_json = load_cambridge_t111_summary_json()
        t115_case_audit = load_cambridge_t115_case_audit()
        t115_policy_summary = load_cambridge_t115_policy_summary()
        t115_perturbation_summary = load_cambridge_t115_perturbation_summary()
        t115_residual_audit = load_cambridge_t115_residual_audit()
        t115_json = load_cambridge_t115_summary_json()
        t120_variant_audit = load_cambridge_t120_variant_audit()
        t120_policy_summary = load_cambridge_t120_policy_summary()
        t120_episode_context = load_cambridge_t120_episode_context()
        t120_loso_summary = load_cambridge_t120_loso_summary()
        t120_json = load_cambridge_t120_summary_json()
        t123_variant_audit = load_cambridge_t123_variant_audit()
        t123_mode_config = load_cambridge_t123_mode_config()
        t123_policy_summary = load_cambridge_t123_policy_summary()
        t123_cold_start_summary = load_cambridge_t123_cold_start_summary()
        t123_subject_summary = load_cambridge_t123_subject_summary()
        t123_json = load_cambridge_t123_summary_json()
        t125_variant_audit = load_cambridge_t125_variant_audit()
        t125_policy_summary = load_cambridge_t125_policy_summary()
        t125_recovery_summary = load_cambridge_t125_recovery_summary()
        t125_learned_refiner = load_cambridge_t125_learned_refiner()
        t125_json = load_cambridge_t125_summary_json()
        if windows.empty:
            st.error("T107 fallback selector outputs are unavailable.")
            return
        if is_t120_view and (t120_variant_audit.empty or t120_policy_summary.empty or t120_episode_context.empty):
            st.error("T120 subject-aware route-risk outputs are unavailable.")
            return
        if is_t123_view and (
            t123_variant_audit.empty
            or t123_mode_config.empty
            or t123_policy_summary.empty
            or t123_cold_start_summary.empty
            or t123_subject_summary.empty
        ):
            st.error("T123 causal-current safety-mode outputs are unavailable.")
            return
        if is_t125_view and (
            t125_variant_audit.empty
            or t125_policy_summary.empty
            or t125_recovery_summary.empty
            or t125_learned_refiner.empty
            or t123_mode_config.empty
        ):
            st.error("T125 risk-floor refinement outputs are unavailable.")
            return

        available_policies = sorted(windows["policy"].dropna().unique())
        policy_order = [
            T107_ROUTE90_POLICY,
            T107_SHIFTED_POLICY,
            T107_ROUTE80_POLICY,
            T106_SHIFTED_POLICY,
            T105_REVIEW_POLICY,
            "t104_default_no_source_validity_guard",
        ]
        policy_options = [policy for policy in policy_order if policy in available_policies]
        policy_options += [policy for policy in available_policies if policy not in policy_options]
        policy = st.sidebar.selectbox(
            "Route policy",
            policy_options,
            index=policy_options.index(T107_ROUTE90_POLICY) if T107_ROUTE90_POLICY in policy_options else 0,
            format_func=t107_policy_label,
        )

        product = prepare_t107_product_output(windows, policy=policy)
        if product.empty:
            st.error("No T107 windows match the selected route policy.")
            return
        subjects = sorted(product["subject"].dropna().unique())
        fallback_by_subject = (
            product.groupby("subject")["fallback_applied"].sum().sort_values(ascending=False)
            if "fallback_applied" in product.columns
            else pd.Series(dtype=float)
        )
        t120_validation_set = "aggressive_extraction_shift"
        t120_stress_scenario = "anchor_abs_ge_6"
        t123_mode = "eldercare_balanced"
        if is_t120_view or is_t123_view or is_t125_view:
            validation_order = ["aggressive_extraction_shift", "moderate_extraction_shift", "route_score_noise"]
            stress_source = t125_policy_summary if is_t125_view else t123_policy_summary if is_t123_view else t120_episode_context
            available_validation = [value for value in validation_order if value in set(stress_source["validation_set"])]
            available_validation += [
                value
                for value in sorted(stress_source["validation_set"].dropna().astype(str).unique())
                if value not in available_validation
            ]
            t120_validation_set = st.sidebar.selectbox(
                "T125 stress family" if is_t125_view else "T123 stress family" if is_t123_view else "T120 stress family",
                available_validation,
                index=0,
                format_func=compact_method_name,
            )
            scenario_order = ["anchor_abs_ge_6", "anchor_abs_ge_8"]
            scenario_source = stress_source[stress_source["validation_set"].eq(t120_validation_set)]
            available_scenarios = [value for value in scenario_order if value in set(scenario_source["base_stress_scenario"])]
            available_scenarios += [
                value
                for value in sorted(scenario_source["base_stress_scenario"].dropna().astype(str).unique())
                if value not in available_scenarios
            ]
            t120_stress_scenario = st.sidebar.selectbox(
                "T125 anchor stress" if is_t125_view else "T123 anchor stress" if is_t123_view else "T120 anchor stress",
                available_scenarios,
                index=0,
                format_func=compact_method_name,
            )
        if is_t123_view or is_t125_view:
            mode_options = t123_mode_options(t123_mode_config)
            default_mode = "hospital_cautious" if is_t125_view else "eldercare_balanced"
            t123_mode = st.sidebar.selectbox(
                "Safety mode",
                mode_options,
                index=mode_options.index(default_mode) if default_mode in mode_options else 0,
                format_func=lambda value: t123_mode_label(t123_mode_config, value),
            )

        residual_subjects = (
            [subject for subject in t115_residual_audit["subject"].dropna().astype(str).unique() if subject in subjects]
            if is_t115_view and not t115_residual_audit.empty
            else []
        )
        t120_context_focus = (
            t120_episode_context[
                t120_episode_context["validation_set"].eq(t120_validation_set)
                & t120_episode_context["base_stress_scenario"].eq(t120_stress_scenario)
                & t120_episode_context["subject"].astype(str).isin(subjects)
            ].copy()
            if is_t120_view
            else pd.DataFrame()
        )
        t123_subject_focus = (
            t123_subject_summary[
                t123_subject_summary["validation_set"].eq(t120_validation_set)
                & t123_subject_summary["base_stress_scenario"].eq(t120_stress_scenario)
                & t123_subject_summary["mode"].eq(t123_mode)
                & t123_subject_summary["subject"].astype(str).isin(subjects)
            ].copy()
            if is_t123_view
            else pd.DataFrame()
        )
        t125_recovered_col = f"t125_{t123_mode}_risk_floor_recovered_from_t123"
        t125_subject_focus = pd.DataFrame()
        if is_t125_view and t125_recovered_col in t125_variant_audit.columns:
            t125_rows = t125_variant_audit[
                t125_variant_audit["validation_set"].eq(t120_validation_set)
                & t125_variant_audit["base_stress_scenario"].eq(t120_stress_scenario)
                & t125_variant_audit["subject"].astype(str).isin(subjects)
            ].copy()
            if not t125_rows.empty:
                recovered_by_subject = (
                    coerce_bool_series(t125_rows[t125_recovered_col])
                    .groupby(t125_rows["subject"].astype(str))
                    .sum()
                    .sort_values(ascending=False)
                )
                t125_subject_focus = recovered_by_subject[recovered_by_subject > 0].reset_index()
                t125_subject_focus.columns = ["subject", "recovered_from_t123"]
        t120_focus_subjects = (
            t120_context_focus.sort_values("t120_episode_tail_risk_rate", ascending=False)["subject"].astype(str).tolist()
            if not t120_context_focus.empty
            else []
        )
        t123_focus_subjects = (
            t123_subject_focus.sort_values(
                ["mode_blocked_from_t119", "max_causal_tail_rate"],
                ascending=False,
            )["subject"].astype(str).tolist()
            if not t123_subject_focus.empty
            else []
        )
        t125_focus_subjects = (
            t125_subject_focus.sort_values("recovered_from_t123", ascending=False)["subject"].astype(str).tolist()
            if not t125_subject_focus.empty
            else []
        )
        default_subject = (
            t125_focus_subjects[0]
            if is_t125_view and t125_focus_subjects
            else t123_focus_subjects[0]
            if is_t123_view and t123_focus_subjects
            else t120_focus_subjects[0]
            if is_t120_view and t120_focus_subjects
            else residual_subjects[0]
            if residual_subjects
            else str(fallback_by_subject.index[0])
            if not fallback_by_subject.empty
            else subjects[0]
        )
        subject = st.sidebar.selectbox("Subject", subjects, index=subjects.index(default_subject))
        sample = product[product["subject"].eq(subject)].sort_values("window_id").copy()
        route_scope = (
            "T125 risk-floor safety-mode source"
            if is_t125_view
            else "T123 causal-current safety-mode source"
            if is_t123_view
            else "T120 subject-aware route-risk source"
            if is_t120_view
            else "T115 broad-stress guarded route"
            if is_t115_view
            else "T111 gated route"
        )
        route_claim_copy = (
            T125_GATE_COPY
            if is_t125_view
            else T123_GATE_COPY
            if is_t123_view
            else T120_GATE_COPY
            if is_t120_view
            else T115_GATE_COPY
            if is_t115_view
            else T111_GATE_COPY
        )
        route_review_copy = (
            T125_REVIEW_COPY
            if is_t125_view
            else T123_REVIEW_COPY
            if is_t123_view
            else T120_REVIEW_COPY
            if is_t120_view
            else T115_REVIEW_COPY
            if is_t115_view
            else T111_REVIEW_COPY
        )
        sample["evidence_level"] = f"Claim-B / {route_scope}"
        sample["claim_boundary"] = route_claim_copy

        latest = sample.iloc[-1]
        automation_metrics = summarize_window_metrics(sample, pred_col="automation_rr_bpm", gt_col="gt_rr_bpm")
        ready_count = int(sample["automation_ready_after_policy"].astype(bool).sum())
        fallback_count = int(sample["fallback_applied"].astype(bool).sum())
        reviewed_count = int((~sample["automation_ready_after_policy"].astype(bool)).sum())
        t111_subject_audit = t111_subject_stress_audit(t111_case_audit, subject)
        t111_subject_released = (
            int(t111_subject_audit["T111 decision"].eq("released").sum()) if not t111_subject_audit.empty else 0
        )
        t111_subject_withheld = (
            int(t111_subject_audit["T111 decision"].eq("withheld").sum()) if not t111_subject_audit.empty else 0
        )
        t115_broad6_row = t115_metric_row(
            t115_policy_summary,
            scenario="anchor_abs_ge_6",
            subset="high_anchor_all",
            policy="t115_broad_stress_guard",
        )
        t115_canonical8_row = t115_metric_row(
            t115_policy_summary,
            scenario="anchor_abs_ge_8",
            subset="high_anchor_all",
            policy="t115_broad_stress_guard",
        )
        t115_t114a_broad6_row = t115_metric_row(
            t115_policy_summary,
            scenario="anchor_abs_ge_6",
            subset="high_anchor_all",
            policy="t114a_margin_robust_high_gate",
        )
        t115_perturb_row = t115_perturbation_metric_row(
            t115_perturbation_summary,
            policy="t115_broad_stress_guard",
        )
        t115_guarded_count = int(t115_residual_audit["t115_broad_stress_guard"].astype(str).str.lower().eq("true").sum()) if not t115_residual_audit.empty and "t115_broad_stress_guard" in t115_residual_audit.columns else int(len(t115_residual_audit))
        t120_selected_row = t120_metric_row(
            t120_policy_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            policy="t120_subject_aware_tail_rate_gate",
        )
        t120_t119_row = t120_metric_row(
            t120_policy_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            policy="t119_route_risk_dev_zero_safe_loss",
        )
        t120_context = t120_context_row(
            t120_episode_context,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            subject=subject,
        )
        t120_subject_counts = t120_subject_policy_counts(
            t120_variant_audit,
            subject=subject,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
        )
        t120_loso_display = t120_loso_summary_table(t120_loso_summary)
        t123_selected_row = t123_metric_row(
            t123_policy_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            mode=t123_mode,
        )
        t123_mode_row = t123_mode_config_row(t123_mode_config, t123_mode)
        t123_subject_counts = t123_subject_policy_counts(
            t123_variant_audit,
            subject=subject,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            mode=t123_mode,
        )
        t123_subject_display = t123_subject_summary_table(
            t123_subject_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            mode=t123_mode,
        )
        t123_cold_start_display = t123_cold_start_table(
            t123_cold_start_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
        )
        t125_selected_row = t125_metric_row(
            t125_policy_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            mode=t123_mode,
        )
        t125_recovery_display = t125_recovery_summary_table(
            t125_recovery_summary,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
        )
        t125_learned_display = t125_learned_refiner_table(
            t125_learned_refiner,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
        )
        t125_subject_counts = t125_subject_policy_counts(
            t125_variant_audit,
            subject=subject,
            validation_set=t120_validation_set,
            scenario=t120_stress_scenario,
            mode=t123_mode,
        )
        interval_coverage = float(sample["interval_available"].mean()) if len(sample) else float("nan")
        rr_text = (
            f"{fmt(latest['product_rr_bpm'])} BPM"
            if bool(latest.get("automation_ready_after_policy", False))
            else "Review"
        )

        cols = st.columns(4)
        cols[0].metric("RR output", rr_text)
        cols[1].metric("Route interval", t107_latest_interval_text(latest))
        cols[2].metric("Fallback recovered", f"{fallback_count}/{len(sample)}")
        if is_t125_view and not t125_selected_row.empty:
            cols[3].metric(
                "T125 refinement",
                (
                    f"{fmt(t125_selected_row.get('t125_safe_recovered'), 0)} safe / "
                    f"{fmt(t125_selected_row.get('t125_unsafe_releases'), 0)} unsafe"
                ),
                delta=f"{fmt(t125_selected_row.get('delta_safe_vs_t123'), 0)} safe vs T123",
            )
        elif is_t123_view and not t123_selected_row.empty:
            cols[3].metric(
                "T123 mode result",
                (
                    f"{fmt(t123_selected_row.get('n_safe_recovered'), 0)} safe / "
                    f"{fmt(t123_selected_row.get('n_unsafe_releases'), 0)} unsafe"
                ),
            )
        elif is_t120_view and not t120_selected_row.empty:
            cols[3].metric(
                "T120 stress result",
                (
                    f"{fmt(t120_selected_row.get('n_safe_recovered'), 0)} safe / "
                    f"{fmt(t120_selected_row.get('n_unsafe_releases'), 0)} unsafe"
                ),
            )
        elif is_t115_view and not t115_broad6_row.empty:
            cols[3].metric(
                "T115 broad guard",
                (
                    f"{fmt(t115_broad6_row.get('n_safe_recovered'), 0)} safe / "
                    f"{fmt(t115_broad6_row.get('n_unsafe_releases'), 0)} unsafe"
                ),
            )
        else:
            cols[3].metric("T111 gate audit", f"{t111_subject_released}/{t111_subject_released + t111_subject_withheld}")
        evidence_badge(level="Claim-B", scope=route_scope, boundary=route_claim_copy, mode="warn")
        t107_status_banner(latest)
        audit_caption = (
            "T125 risk-floor counts are internal simulation stress cases, not external clinical validation."
            if is_t125_view
            else "T123 safety-mode counts are internal simulation stress cases, not external clinical validation."
            if is_t123_view
            else "T115/T111 gate audit counts are synthetic-stress cases, not external clinical validation."
        )
        st.caption(f"Current subject review-required windows: {reviewed_count}/{len(sample)}. {audit_caption}")

        center, right = st.columns([0.66, 0.34], gap="large")
        with center:
            st.plotly_chart(
                t107_route_chart(
                    sample,
                    title=(
                        f"Cambridge T125 risk-floor safety mode / {subject}<br>{t123_mode_label(t123_mode_config, t123_mode)}"
                        if is_t125_view
                        else f"Cambridge T123 causal-current safety mode / {subject}<br>{t123_mode_label(t123_mode_config, t123_mode)}"
                        if is_t123_view
                        else f"Cambridge T120 subject-aware route-risk source / {subject}<br>{t107_policy_label(policy)}"
                        if is_t120_view
                        else f"Cambridge T107 source-validity route / {subject}<br>{t107_policy_label(policy)}"
                    ),
                ),
                width="stretch",
            )
            table_cols = [
                "window_id",
                "gt_rr_bpm",
                "balanced_pred_rr_bpm",
                "product_rr_bpm",
                "interval_lower_bpm",
                "interval_upper_bpm",
                "fallback_applied",
                "fallback_policy_key",
                "fallback_reason",
                "product_status",
                "automation_ready_after_policy",
                "interval_covers_reference_after_policy",
            ]
            table = sample[[col for col in table_cols if col in sample.columns]].copy()
            for col in [
                "gt_rr_bpm",
                "balanced_pred_rr_bpm",
                "product_rr_bpm",
                "interval_lower_bpm",
                "interval_upper_bpm",
            ]:
                if col in table.columns:
                    table[col] = pd.to_numeric(table[col], errors="coerce").round(3)
            if "fallback_policy_key" in table.columns:
                table["fallback_policy_key"] = table["fallback_policy_key"].map(
                    lambda value: "" if pd.isna(value) else compact_method_name(str(value))
                )
            st.dataframe(table, width="stretch", hide_index=True)
        with right:
            st.subheader("Route Evidence")
            chip_class = (
                "vs-chip-good"
                if str(latest["product_status"]) in {"automation_ready", "corrected_fallback_output"}
                else "vs-chip-warn"
            )
            guard_chip = (
                "<span class='vs-chip vs-chip-warn'>T125 risk floor</span>"
                if is_t125_view
                else "<span class='vs-chip vs-chip-warn'>T123 safety mode</span>"
                if is_t123_view
                else "<span class='vs-chip vs-chip-warn'>T120 episode risk</span>"
                if is_t120_view
                else "<span class='vs-chip vs-chip-warn'>T115 broad guard</span>"
                if is_t115_view
                else "<span class='vs-chip vs-chip-warn'>T111 safety gate</span>"
            )
            st.markdown(
                f"<span class='vs-chip {chip_class}'>{latest['product_status']}</span>"
                f"{guard_chip}"
                "<span class='vs-chip'>route interval</span>"
                "<span class='vs-chip'>review aid</span>",
                unsafe_allow_html=True,
            )
            t111_high_row = t111_gate_metric_row(
                t111_policy_summary,
                subset="high_anchor_all",
                policy="t111_mechanistic_safety_gate",
            )
            t111_low_row = t111_gate_metric_row(
                t111_policy_summary,
                subset="low_anchor_diagnostic_all",
                policy="t111_direction_aware_product_gate",
            )
            output_contract = [
                ["RR output", rr_text],
                ["RR interval", t107_latest_interval_text(latest)],
                ["product disposition", str(latest.get("product_disposition"))],
                ["T111 product gate", str(t111_json.get("recommended_policy", "t111_direction_aware_product_gate"))],
                [
                    "T111 high-anchor",
                    (
                        f"{fmt(t111_high_row.get('n_safe_recovered'), 0)} safe / "
                        f"{fmt(t111_high_row.get('n_unsafe_releases'), 0)} unsafe"
                    )
                    if not t111_high_row.empty
                    else "NA",
                ],
                [
                    "T111 low-anchor",
                    (
                        f"{fmt(t111_low_row.get('n_released'), 0)} released / "
                        f"{fmt(t111_low_row.get('n_withheld'), 0)} review-only"
                    )
                    if not t111_low_row.empty
                    else "NA",
                ],
                ["fallback source", compact_method_name(str(latest.get("fallback_policy_key", "none")))],
                ["fallback reason", str(latest.get("warning_reason"))],
                ["selector cluster center", fmt(latest.get("selector_cluster_center_rr_bpm"), 2)],
                ["interval mode", str(latest.get("interval_mode"))],
                ["subject ready coverage", f"{fmt(100.0 * ready_count / max(len(sample), 1), 1)}%"],
                ["subject interval coverage", f"{fmt(100.0 * interval_coverage, 1)}%"],
                ["subject ready MAE", f"{fmt(automation_metrics['mae'])} BPM"],
            ]
            if is_t125_view:
                t125_safe = int(t125_selected_row.get("t125_safe_recovered", 0)) if not t125_selected_row.empty else 0
                t125_unsafe = int(t125_selected_row.get("t125_unsafe_releases", 0)) if not t125_selected_row.empty else 0
                t125_withheld = int(t125_selected_row.get("t125_withheld", 0)) if not t125_selected_row.empty else 0
                t123_safe = int(t125_selected_row.get("t123_safe_recovered", 0)) if not t125_selected_row.empty else 0
                t123_unsafe = int(t125_selected_row.get("t123_unsafe_releases", 0)) if not t125_selected_row.empty else 0
                t123_withheld = int(t125_selected_row.get("t123_withheld", 0)) if not t125_selected_row.empty else 0
                t125_rows = [
                    ["risk-floor stack", str(t125_json.get("method", "blocked-surface risk-floor refinement"))],
                    ["selected mode", t123_mode_label(t123_mode_config, t123_mode)],
                    [
                        "stress family",
                        f"{compact_method_name(t120_validation_set)} / {compact_method_name(t120_stress_scenario)}",
                    ],
                    ["T123 baseline", f"{t123_safe} safe / {t123_unsafe} unsafe / {t123_withheld} review-only"],
                    ["T125 result", f"{t125_safe} safe / {t125_unsafe} unsafe / {t125_withheld} review-only"],
                    ["delta safe vs T123", int(t125_selected_row.get("delta_safe_vs_t123", 0)) if not t125_selected_row.empty else 0],
                    [
                        "delta unsafe vs T123",
                        int(t125_selected_row.get("delta_unsafe_vs_t123", 0)) if not t125_selected_row.empty else 0,
                    ],
                    [
                        "review-only reduction",
                        -int(t125_selected_row.get("delta_withheld_vs_t123", 0)) if not t125_selected_row.empty else 0,
                    ],
                    [
                        "risk floors",
                        (
                            f"current {fmt(t125_selected_row.get('current_threshold'), 2)}, "
                            f"cold {fmt(t125_selected_row.get('cold_threshold'), 2)}"
                        )
                        if not t125_selected_row.empty
                        else "NA",
                    ],
                    [
                        "candidate status",
                        str(t125_selected_row.get("candidate_status", "NA")) if not t125_selected_row.empty else "NA",
                    ],
                    [
                        "subject T123",
                        (
                            f"{t125_subject_counts['t123_safe']} safe / "
                            f"{t125_subject_counts['t123_unsafe']} unsafe / "
                            f"{t125_subject_counts['t123_review_only']} review-only"
                        ),
                    ],
                    [
                        "subject T125",
                        (
                            f"{t125_subject_counts['t125_safe']} safe / "
                            f"{t125_subject_counts['t125_unsafe']} unsafe / "
                            f"{t125_subject_counts['t125_review_only']} review-only"
                        ),
                    ],
                    ["subject recovered", t125_subject_counts.get("recovered_from_t123", 0)],
                    ["subject recovered unsafe", t125_subject_counts.get("recovered_unsafe", 0)],
                    ["current-floor blocks", t125_subject_counts.get("current_floor_blocks", 0)],
                    ["cold-floor blocks", t125_subject_counts.get("cold_floor_blocks", 0)],
                    ["learned refiners", "rejected if unsafe increases"],
                ]
                output_contract = output_contract[:4] + t125_rows + output_contract[4:]
            if is_t123_view:
                t123_safe = int(t123_selected_row.get("n_safe_recovered", 0)) if not t123_selected_row.empty else 0
                t123_unsafe = int(t123_selected_row.get("n_unsafe_releases", 0)) if not t123_selected_row.empty else 0
                t123_withheld = int(t123_selected_row.get("n_withheld", 0)) if not t123_selected_row.empty else 0
                t123_safe_loss = int(t123_selected_row.get("safe_loss_vs_t119", 0)) if not t123_selected_row.empty else 0
                t123_unsafe_reduction = (
                    int(t123_selected_row.get("unsafe_reduction_vs_t119", 0)) if not t123_selected_row.empty else 0
                )
                t123_rows = [
                    ["safety-mode stack", str(t123_json.get("method", "causal-current episode-risk safety modes"))],
                    ["selected mode", t123_mode_label(t123_mode_config, t123_mode)],
                    [
                        "intended setting",
                        str(t123_mode_row.get("intended_setting", "NA")) if not t123_mode_row.empty else "NA",
                    ],
                    [
                        "stress family",
                        f"{compact_method_name(t120_validation_set)} / {compact_method_name(t120_stress_scenario)}",
                    ],
                    [
                        "mode thresholds",
                        (
                            f"tail-rate >= {fmt(t123_mode_row.get('tail_rate_threshold'), 2)}, "
                            f"strict risk >= {fmt(t123_mode_row.get('strict_risk_threshold'), 2)}"
                        )
                        if not t123_mode_row.empty
                        else "NA",
                    ],
                    [
                        "cold-start policy",
                        (
                            "force review="
                            f"{bool(coerce_bool_series(pd.Series([t123_mode_row.get('cold_start_force_review')])).iloc[0])}, "
                            f"min prior={fmt(t123_mode_row.get('cold_start_min_prior_windows'), 0)}"
                        )
                        if not t123_mode_row.empty
                        else "NA",
                    ],
                    [
                        "T119 stress baseline",
                        f"{t123_safe + t123_safe_loss} safe / {t123_unsafe + t123_unsafe_reduction} unsafe",
                    ],
                    [
                        "T123 mode result",
                        f"{t123_safe} safe / {t123_unsafe} unsafe / {t123_withheld} review-only",
                    ],
                    ["safe loss vs T119", t123_safe_loss],
                    ["unsafe reduction", t123_unsafe_reduction],
                    [
                        "safe delta vs T120 full",
                        int(t123_selected_row.get("safe_delta_vs_t120_full", 0)) if not t123_selected_row.empty else "NA",
                    ],
                    [
                        "subject T119 unsafe",
                        f"{t123_subject_counts['t119_unsafe']} unsafe / {t123_subject_counts['t119_safe']} safe",
                    ],
                    [
                        "subject mode unsafe",
                        f"{t123_subject_counts['mode_unsafe']} unsafe / {t123_subject_counts['mode_safe']} safe",
                    ],
                    ["blocked by mode", t123_subject_counts.get("blocked_by_mode", 0)],
                    ["current-risk blocks", t123_subject_counts.get("current_episode_blocks", 0)],
                    ["cold-start blocks", t123_subject_counts.get("cold_start_blocks", 0)],
                ]
                output_contract = output_contract[:4] + t123_rows + output_contract[4:]
            if is_t120_view:
                t120_rows = [
                    ["route-risk stack", str(t120_json.get("method", "subject-aware episode tail-risk calibration overlay"))],
                    [
                        "stress family",
                        f"{compact_method_name(t120_validation_set)} / {compact_method_name(t120_stress_scenario)}",
                    ],
                    [
                        "T119 stress result",
                        (
                            f"{fmt(t120_t119_row.get('n_safe_recovered'), 0)} safe / "
                            f"{fmt(t120_t119_row.get('n_unsafe_releases'), 0)} unsafe / "
                            f"{fmt(t120_t119_row.get('n_withheld'), 0)} review-only"
                        )
                        if not t120_t119_row.empty
                        else "NA",
                    ],
                    [
                        "T120 stress result",
                        (
                            f"{fmt(t120_selected_row.get('n_safe_recovered'), 0)} safe / "
                            f"{fmt(t120_selected_row.get('n_unsafe_releases'), 0)} unsafe / "
                            f"{fmt(t120_selected_row.get('n_withheld'), 0)} review-only"
                        )
                        if not t120_selected_row.empty
                        else "NA",
                    ],
                    [
                        "episode tail-risk rate",
                        fmt(t120_context.get("t120_episode_tail_risk_rate"), 3) if not t120_context.empty else "NA",
                    ],
                    [
                        "episode instability",
                        str(bool(t120_context.get("t120_episode_instability_flag"))) if not t120_context.empty else "NA",
                    ],
                    [
                        "strict threshold",
                        fmt(t120_context.get("t120_strict_risk_threshold"), 2) if not t120_context.empty else "NA",
                    ],
                    [
                        "subject T119 unsafe",
                        f"{t120_subject_counts['t119_unsafe']} unsafe / {t120_subject_counts['t119_safe']} safe",
                    ],
                    [
                        "subject T120 unsafe",
                        f"{t120_subject_counts['t120_unsafe']} unsafe / {t120_subject_counts['t120_safe']} safe",
                    ],
                    ["blocked by T120", t120_subject_counts.get("blocked_by_t120", 0)],
                ]
                if not t120_loso_display.empty:
                    balanced = t120_loso_display[t120_loso_display["Mode"].eq("balanced")]
                    conservative = t120_loso_display[t120_loso_display["Mode"].eq("conservative")]
                    if not balanced.empty:
                        row = balanced.iloc[0]
                        t120_rows.append(["LOSO balanced", f"{row['T120 safe']} safe / {row['T120 unsafe']} unsafe"])
                    if not conservative.empty:
                        row = conservative.iloc[0]
                        t120_rows.append(["LOSO conservative", f"{row['T120 safe']} safe / {row['T120 unsafe']} unsafe"])
                output_contract = output_contract[:4] + t120_rows + output_contract[4:]
            if is_t115_view:
                t115_rows = [
                    ["route guard stack", str(t115_json.get("task", "T115")) + " after T114A margin gate"],
                    [
                        "T115 broad 6 BPM",
                        (
                            f"{fmt(t115_broad6_row.get('n_safe_recovered'), 0)} safe / "
                            f"{fmt(t115_broad6_row.get('n_unsafe_releases'), 0)} unsafe / "
                            f"{fmt(t115_broad6_row.get('n_withheld'), 0)} review-only"
                        )
                        if not t115_broad6_row.empty
                        else "NA",
                    ],
                    [
                        "T114A broad 6 BPM",
                        (
                            f"{fmt(t115_t114a_broad6_row.get('n_safe_recovered'), 0)} safe / "
                            f"{fmt(t115_t114a_broad6_row.get('n_unsafe_releases'), 0)} unsafe / "
                            f"{fmt(t115_t114a_broad6_row.get('n_withheld'), 0)} withheld"
                        )
                        if not t115_t114a_broad6_row.empty
                        else "NA",
                    ],
                    [
                        "T115 canonical 8 BPM",
                        (
                            f"{fmt(t115_canonical8_row.get('n_safe_recovered'), 0)} safe / "
                            f"{fmt(t115_canonical8_row.get('n_unsafe_releases'), 0)} unsafe / "
                            f"{fmt(t115_canonical8_row.get('n_withheld'), 0)} withheld"
                        )
                        if not t115_canonical8_row.empty
                        else "NA",
                    ],
                    [
                        "T115 perturbation",
                        (
                            f"{fmt(t115_perturb_row.get('n_safe_recovered'), 0)} safe / "
                            f"{fmt(t115_perturb_row.get('n_unsafe_releases'), 0)} unsafe / "
                            f"{fmt(t115_perturb_row.get('n_withheld'), 0)} withheld"
                        )
                        if not t115_perturb_row.empty
                        else "NA",
                    ],
                    ["T115 guarded residuals", f"{t115_guarded_count}/{len(t115_residual_audit)}"],
                ]
                output_contract = output_contract[:4] + t115_rows + output_contract[4:]
            key_value_panel([(str(key), value) for key, value in output_contract])

            summary_row = summary[summary["policy"].eq(policy)].head(1) if not summary.empty else pd.DataFrame()
            if not summary_row.empty:
                st.subheader("Policy Summary")
                st.dataframe(t107_summary_table(summary_row.iloc[0]), width="stretch", hide_index=True)

            if is_t125_view:
                t125_display = t125_policy_summary_table(
                    t125_policy_summary,
                    validation_set=t120_validation_set,
                    scenario=t120_stress_scenario,
                )
                if not t125_display.empty:
                    st.subheader("T125 Risk-Floor Refinement Summary")
                    st.dataframe(t125_display, width="stretch", hide_index=True)
                if not t125_recovery_display.empty:
                    st.subheader("T125 Recovered Surface")
                    st.dataframe(t125_recovery_display, width="stretch", hide_index=True)
                if not t125_learned_display.empty:
                    st.subheader("T125 Learned-Refiner Transfer Test")
                    st.dataframe(t125_learned_display, width="stretch", hide_index=True)
                t123_config_display = t123_mode_config_table(t123_mode_config)
                if not t123_config_display.empty:
                    st.subheader("Safety Mode Configuration")
                    st.dataframe(t123_config_display, width="stretch", hide_index=True)

            if is_t123_view:
                t123_display = t123_policy_summary_table(
                    t123_policy_summary,
                    validation_set=t120_validation_set,
                    scenario=t120_stress_scenario,
                )
                if not t123_display.empty:
                    st.subheader("T123 Safety-Mode Summary")
                    st.dataframe(t123_display, width="stretch", hide_index=True)
                t123_config_display = t123_mode_config_table(t123_mode_config)
                if not t123_config_display.empty:
                    st.subheader("T123 Mode Configuration")
                    st.dataframe(t123_config_display, width="stretch", hide_index=True)
                if not t123_cold_start_display.empty:
                    st.subheader("T123 Cold-Start Breakdown")
                    st.dataframe(t123_cold_start_display, width="stretch", hide_index=True)
                if not t123_subject_display.empty:
                    st.subheader("T123 Subject Summary")
                    st.dataframe(t123_subject_display, width="stretch", hide_index=True)

            t111_display = t111_policy_summary_table(t111_policy_summary)
            if not t111_display.empty:
                st.subheader("T111 Gate Summary")
                st.dataframe(t111_display, width="stretch", hide_index=True)

            t115_display = t115_policy_summary_table(t115_policy_summary, t115_perturbation_summary)
            if is_t115_view and not t115_display.empty:
                st.subheader("T115 Broad-Stress Guard Summary")
                st.dataframe(t115_display, width="stretch", hide_index=True)

            if is_t120_view:
                t120_display = t120_policy_summary_table(
                    t120_policy_summary,
                    validation_set=t120_validation_set,
                    scenario=t120_stress_scenario,
                )
                if not t120_display.empty:
                    st.subheader("T120 Subject-Aware Route-Risk Summary")
                    st.dataframe(t120_display, width="stretch", hide_index=True)
                if not t120_loso_display.empty:
                    st.subheader("T120 LOSO Calibration Audit")
                    st.dataframe(t120_loso_display, width="stretch", hide_index=True)
                t120_context_display = t120_episode_context_table(
                    t120_episode_context,
                    validation_set=t120_validation_set,
                    scenario=t120_stress_scenario,
                )
                if not t120_context_display.empty:
                    st.subheader("T120 Episode-Risk Context")
                    st.dataframe(t120_context_display, width="stretch", hide_index=True)

            if is_t115_view:
                residual_display = t115_residual_guard_table(t115_residual_audit)
                if not residual_display.empty:
                    st.subheader("T115 Guarded Residual Cases")
                    st.dataframe(residual_display, width="stretch", hide_index=True)

            if not selected_fallbacks.empty:
                subject_fallbacks = selected_fallbacks[selected_fallbacks["subject"].eq(subject)].copy()
                if not subject_fallbacks.empty:
                    st.subheader("Selected Fallbacks")
                    display_cols = [
                        "window_id",
                        "gt_rr_bpm",
                        "balanced_pred_rr_bpm",
                        "pred_rr_bpm_after_policy",
                        "delta_abs_error_vs_balanced_bpm",
                        "fallback_policy_key",
                        "fallback_reason",
                    ]
                    display = subject_fallbacks[[col for col in display_cols if col in subject_fallbacks.columns]].copy()
                    for col in [
                        "gt_rr_bpm",
                        "balanced_pred_rr_bpm",
                        "pred_rr_bpm_after_policy",
                        "delta_abs_error_vs_balanced_bpm",
                    ]:
                        if col in display.columns:
                            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                    if "fallback_policy_key" in display.columns:
                        display["fallback_policy_key"] = display["fallback_policy_key"].map(compact_method_name)
                    st.dataframe(display, width="stretch", hide_index=True)

            with st.expander("Route residual calibration"):
                if not route_residuals.empty:
                    residual_display = route_residuals.copy()
                    if "fallback_policy_key" in residual_display.columns:
                        residual_display["fallback_policy_key"] = residual_display["fallback_policy_key"].map(
                            compact_method_name
                        )
                    for col in residual_display.columns:
                        if col not in {"fallback_policy_key"}:
                            converted = pd.to_numeric(residual_display[col], errors="coerce")
                            if converted.notna().sum() == residual_display[col].notna().sum():
                                residual_display[col] = converted
                    st.dataframe(residual_display, width="stretch", hide_index=True)
                if not interval_calibration.empty:
                    calibration_display = interval_calibration[interval_calibration["subject"].eq(subject)].copy()
                    if "fallback_policy_key" in calibration_display.columns:
                        calibration_display["fallback_policy_key"] = calibration_display["fallback_policy_key"].map(
                            compact_method_name
                        )
                    st.dataframe(calibration_display, width="stretch", hide_index=True)
            with st.expander("Reviewed windows audit"):
                if not review_audit.empty:
                    audit_display = review_audit[review_audit["subject"].eq(subject)].copy()
                    if "t107_fallback_policy_key" in audit_display.columns:
                        audit_display["t107_fallback_policy_key"] = audit_display["t107_fallback_policy_key"].map(
                            lambda value: "" if pd.isna(value) else compact_method_name(str(value))
                        )
                    st.dataframe(audit_display, width="stretch", hide_index=True)
            with st.expander("T111 stress gate audit"):
                if not t111_subject_audit.empty:
                    blocked = t111_subject_audit[
                        t111_subject_audit["T111 decision"].eq("withheld")
                        & ~t111_subject_audit["T111 reason"].isin(
                            ["released", "withheld_by_t107_no_countercluster", "low_anchor_review_only"]
                        )
                    ].copy()
                    if not blocked.empty:
                        blocked_text = "; ".join(
                            f"{row['stress']} window {int(row['window_id'])}: {row['T111 reason']}"
                            for _, row in blocked.iterrows()
                        )
                        st.warning(f"T111 gate blocked route-risk cases: {blocked_text}")
                    st.table(t111_subject_audit)
                else:
                    st.info("No T111 synthetic stress cases are available for this subject.")
            if is_t115_view:
                t115_subject_audit = t115_subject_stress_audit(t115_case_audit, subject)
                with st.expander("T115 broad-stress guard audit", expanded=True):
                    subject_residuals = t115_residual_guard_table(t115_residual_audit, subject=subject)
                    if not subject_residuals.empty:
                        guarded_text = "; ".join(
                            f"window {int(row['window_id'])}: {row['T115 reason']}"
                            for _, row in subject_residuals.iterrows()
                        )
                        st.warning(f"T115 guarded residual route-risk cases: {guarded_text}")
                    if not t115_subject_audit.empty:
                        st.dataframe(t115_subject_audit, width="stretch", hide_index=True)
                    else:
                        st.info("No T115 broad 6 BPM high-anchor stress cases are available for this subject.")
            if is_t125_view:
                with st.expander("T125 risk-floor refinement audit", expanded=True):
                    t125_subject_audit = t125_subject_refinement_audit(
                        t125_variant_audit,
                        subject=subject,
                        validation_set=t120_validation_set,
                        scenario=t120_stress_scenario,
                        mode=t123_mode,
                    )
                    if not t125_subject_audit.empty:
                        recovered = t125_subject_audit[t125_subject_audit["Recovered from T123"].astype(bool)].copy()
                        if not recovered.empty:
                            recovered_text = "; ".join(
                                f"window {int(row['Window'])}: risk={fmt(row['T119 risk'], 3)}, "
                                f"tail={fmt(row['Current tail-rate'], 3)}, {row['T123 decision']} -> {row['T125 decision']}"
                                for _, row in recovered.head(5).iterrows()
                            )
                            st.success(f"T125 recovered T123 review-only safe windows: {recovered_text}")
                        else:
                            st.info("No T125 recovered windows for this subject/mode/scenario.")
                        st.dataframe(t125_subject_audit, width="stretch", hide_index=True)
                    else:
                        st.info("No T125 risk-floor stress variants are available for this subject and scenario.")
            if is_t123_view:
                with st.expander("T123 safety-mode audit", expanded=True):
                    t123_subject_audit = t123_subject_safety_audit(
                        t123_variant_audit,
                        subject=subject,
                        validation_set=t120_validation_set,
                        scenario=t120_stress_scenario,
                        mode=t123_mode,
                    )
                    if not t123_subject_audit.empty:
                        blocked = t123_subject_audit[t123_subject_audit["Blocked by mode"].astype(bool)].copy()
                        if not blocked.empty:
                            blocked_text = "; ".join(
                                f"window {int(row['Window'])}: risk={fmt(row['T119 risk'], 3)}, "
                                f"current-tail={fmt(row['Current tail-rate'], 3)}, {row['Warning reason']}"
                                for _, row in blocked.head(5).iterrows()
                            )
                            st.warning(f"T123 safety mode converted route-risk releases to review-only: {blocked_text}")
                        st.dataframe(t123_subject_audit, width="stretch", hide_index=True)
                    else:
                        st.info("No T123 safety-mode stress variants are available for this subject and scenario.")
            if is_t120_view:
                with st.expander("T120 subject-aware route-risk audit", expanded=True):
                    t120_subject_audit = t120_subject_route_risk_audit(
                        t120_variant_audit,
                        subject=subject,
                        validation_set=t120_validation_set,
                        scenario=t120_stress_scenario,
                    )
                    if not t120_subject_audit.empty:
                        blocked = t120_subject_audit[t120_subject_audit["Blocked by T120"].astype(bool)].copy()
                        if not blocked.empty:
                            blocked_text = "; ".join(
                                f"window {int(row['Window'])}: risk={fmt(row['T119 risk'], 3)}, "
                                f"tail={fmt(row['Episode tail-risk'], 3)}, {row['T120 decision']}"
                                for _, row in blocked.head(5).iterrows()
                            )
                            st.warning(f"T120 converted borderline route-risk releases to review-only: {blocked_text}")
                        st.dataframe(t120_subject_audit, width="stretch", hide_index=True)
                    else:
                        st.info("No T120 route-risk stress variants are available for this subject and scenario.")
            st.caption(route_review_copy)
            export_title_prefix = (
                "Cambridge_T125"
                if is_t125_view
                else "Cambridge_T123"
                if is_t123_view
                else "Cambridge_T120"
                if is_t120_view
                else "Cambridge_T115"
                if is_t115_view
                else "Cambridge_T111"
            )
            export_suffix = f"{subject}_{t123_mode}" if (is_t123_view or is_t125_view) else f"{subject}_{policy}"
            export_panel(sample, title=f"{export_title_prefix}_{export_suffix}")
        return

    if source.startswith("T98"):
        risk_windows = load_cambridge_t98_risk_windows()
        risk_summary = load_cambridge_t98_risk_summary()
        interval_windows = load_cambridge_t98_interval_windows()
        interval_summary = load_cambridge_t98_interval_summary()
        if risk_windows.empty or interval_windows.empty:
            st.error("T98 risk calibration and interval outputs are unavailable.")
            return

        available_risk = sorted(risk_windows["calibration_policy"].dropna().unique())
        risk_order = [
            T98_DEFAULT_CALIBRATION_POLICY,
            "risk_target_0.30_t96_reference",
            "risk_target_0.30_upper_gap_latent",
            "risk_target_0.25_q90_gap",
            "risk_target_0.35_q90_gap",
        ]
        risk_options = [policy for policy in risk_order if policy in available_risk]
        risk_options += [policy for policy in available_risk if policy not in risk_options]
        calibration_policy = st.sidebar.selectbox(
            "Source-risk calibration",
            risk_options,
            index=risk_options.index(T98_DEFAULT_CALIBRATION_POLICY)
            if T98_DEFAULT_CALIBRATION_POLICY in risk_options
            else 0,
            format_func=compact_method_name,
        )

        available_interval = sorted(interval_windows["interval_policy"].dropna().unique())
        interval_order = [T98_DEFAULT_INTERVAL_POLICY, "t97_q90_gap_q80", "all_windows"]
        interval_options = [policy for policy in interval_order if policy in available_interval]
        interval_options += [policy for policy in available_interval if policy not in interval_options]
        interval_policy = st.sidebar.selectbox(
            "RR interval policy",
            interval_options,
            index=interval_options.index(T98_DEFAULT_INTERVAL_POLICY)
            if T98_DEFAULT_INTERVAL_POLICY in interval_options
            else 0,
            format_func=compact_method_name,
        )
        alpha_options = sorted(
            pd.to_numeric(
                interval_windows[interval_windows["interval_policy"].eq(interval_policy)]["interval_alpha"],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
        )
        interval_alpha = st.sidebar.selectbox(
            "Interval confidence",
            alpha_options,
            index=alpha_options.index(T98_DEFAULT_ALPHA) if T98_DEFAULT_ALPHA in alpha_options else 0,
            format_func=lambda alpha: f"{int(round((1.0 - float(alpha)) * 100))}% interval",
        )

        product = prepare_t98_product_output(
            risk_windows,
            interval_windows,
            calibration_policy=calibration_policy,
            interval_policy=interval_policy,
            interval_alpha=float(interval_alpha),
        )
        subjects = sorted(product["subject"].dropna().unique())
        latest_by_subject = product.sort_values("window_id").groupby("subject", as_index=False).tail(1)
        available_latest = latest_by_subject[latest_by_subject["interval_available"].astype(bool)]["subject"].tolist()
        default_subject = available_latest[0] if available_latest else subjects[0]
        subject = st.sidebar.selectbox("Subject", subjects, index=subjects.index(default_subject))
        sample = product[product["subject"].eq(subject)].sort_values("window_id").copy()
        sample["evidence_level"] = "Claim-B / interval-aware RR"
        sample["claim_boundary"] = T98_INTERVAL_COPY

        latest = sample.iloc[-1]
        automation_metrics = summarize_window_metrics(sample, pred_col="automation_rr_bpm", gt_col="gt_rr_bpm")
        interval_coverage = float(sample["interval_available"].mean()) if len(sample) else float("nan")
        risk_flags = int((~sample["calibrated_risk_accepted"].astype(bool)).sum())
        rr_text = f"{fmt(latest['display_rr_bpm'])} BPM" if bool(latest["interval_available"]) else "Withheld"

        cols = st.columns(4)
        cols[0].metric("RR estimate", rr_text)
        cols[1].metric(f"{int(round((1.0 - float(interval_alpha)) * 100))}% interval", t98_latest_interval_text(latest))
        cols[2].metric("Automation-ready", f"{fmt(100.0 * automation_metrics['coverage'], 1)}%")
        cols[3].metric("Risk flags", f"{risk_flags}/{len(sample)}")
        evidence_badge(level="Claim-B", scope="Interval + source-risk", boundary=T98_INTERVAL_COPY, mode="warn")
        t98_status_banner(latest)

        center, right = st.columns([0.66, 0.34], gap="large")
        with center:
            st.plotly_chart(
                t98_interval_chart(
                    sample,
                    title=f"Cambridge T98 interval-aware RR / {subject}<br>{t98_interval_label(interval_policy, float(interval_alpha))}",
                ),
                width="stretch",
            )
            table_cols = [
                "window_id",
                "start_sec",
                "gt_rr_bpm",
                "display_rr_bpm",
                "interval_lower_bpm",
                "interval_upper_bpm",
                "interval_width_bpm",
                "automation_ready",
                "product_status",
                "source_risk_score",
                "threshold_score",
                "warning_reason",
                "interval_covers_reference",
            ]
            table = sample[[col for col in table_cols if col in sample.columns]].copy()
            for col in [
                "gt_rr_bpm",
                "display_rr_bpm",
                "interval_lower_bpm",
                "interval_upper_bpm",
                "interval_width_bpm",
                "source_risk_score",
                "threshold_score",
            ]:
                if col in table.columns:
                    table[col] = pd.to_numeric(table[col], errors="coerce").round(3)
            st.dataframe(table, width="stretch", hide_index=True)
        with right:
            st.subheader("Product Output")
            chip_class = "vs-chip-good" if str(latest["product_status"]) == "accepted_low_risk" else "vs-chip-warn"
            st.markdown(
                f"<span class='vs-chip {chip_class}'>{latest['product_status']}</span>"
                "<span class='vs-chip'>RR interval</span>"
                "<span class='vs-chip'>Review aid</span>",
                unsafe_allow_html=True,
            )
            output_contract = [
                ["RR estimate", rr_text],
                ["RR interval", t98_latest_interval_text(latest)],
                ["source-risk status", str(latest.get("source_risk_status"))],
                ["warning reason", str(latest.get("warning_reason"))],
                ["source risk score", fmt(latest.get("source_risk_score"), 3)],
                ["risk threshold", fmt(latest.get("threshold_score"), 3)],
                ["interval coverage in subject", f"{fmt(100.0 * interval_coverage, 1)}%"],
                ["automation-ready MAE", f"{fmt(automation_metrics['mae'])} BPM"],
            ]
            key_value_panel([(str(key), value) for key, value in output_contract])

            risk_row = risk_summary[risk_summary["calibration_policy"].eq(calibration_policy)].head(1)
            if not risk_row.empty:
                st.subheader("Risk Calibration")
                st.dataframe(t98_summary_table(risk_row.iloc[0], kind="risk"), width="stretch", hide_index=True)
            interval_row = interval_summary[
                interval_summary["interval_policy"].eq(interval_policy)
                & np.isclose(pd.to_numeric(interval_summary["interval_alpha"], errors="coerce"), float(interval_alpha))
            ].head(1)
            if not interval_row.empty:
                st.subheader("Interval Calibration")
                st.dataframe(t98_summary_table(interval_row.iloc[0], kind="interval"), width="stretch", hide_index=True)
            st.caption(T98_RISK_COPY)
            export_panel(sample, title=f"Cambridge_T98_{subject}_{calibration_policy}_{interval_policy}_{interval_alpha}")
        return

    if source.startswith("T94"):
        t94_windows = load_cambridge_t94_windows()
        t94_summary = load_cambridge_t94_policy_summary()
        t94_ci = load_cambridge_t94_bootstrap_ci()
        t95_windows = load_cambridge_t95_windows()
        t95_summary = load_cambridge_t95_policy_summary()
        t95_folds = load_cambridge_t95_fold_summary()
        t95_counts = load_cambridge_t95_selection_counts()
        t95_ci = load_cambridge_t95_bootstrap_ci()
        if t94_windows.empty and t95_windows.empty:
            st.error("T94/T95 latent-state outputs are unavailable.")
            return

        available_policies = []
        for policy in [T95_COMBINED_POLICY, T95_MAE_POLICY, T95_HIGH_RR_POLICY, T94_BALANCED_POLICY, T94_HIGH_RECALL_POLICY, T94_EQUAL_POLICY]:
            if (not t95_windows.empty and policy in set(t95_windows["policy"])) or (
                not t94_windows.empty and policy in set(t94_windows["policy"])
            ):
                available_policies.append(policy)
        if not available_policies:
            available_policies = sorted(
                set(t94_windows.get("policy", pd.Series(dtype=str))).union(set(t95_windows.get("policy", pd.Series(dtype=str))))
            )
        default_policy = T95_COMBINED_POLICY if T95_COMBINED_POLICY in available_policies else available_policies[0]
        policy = st.sidebar.selectbox(
            "Latent-state evidence view",
            available_policies,
            index=available_policies.index(default_policy),
            format_func=compact_method_name,
        )
        source_windows = t95_windows if policy.startswith("t95_") else t94_windows
        source_summary = t95_summary if policy.startswith("t95_") else t94_summary
        source_ci = t95_ci if policy.startswith("t95_") else t94_ci
        available = source_windows[source_windows["policy"].eq(policy)].copy()
        subjects = sorted(available["subject"].dropna().unique())
        subject = st.sidebar.selectbox("Subject", subjects)
        sample = available[available["subject"].eq(subject)].sort_values("window_id").copy()
        sample["evidence_level"] = "Claim-B / latent-state RR"
        sample["claim_boundary"] = T95_LOSO_COPY if policy.startswith("t95_") else T94_LATENT_COPY

        metrics = summarize_window_metrics(sample, pred_col="pred_rr_bpm", gt_col="gt_rr_bpm")
        policy_row = source_summary[source_summary["policy"].eq(policy)].head(1)
        policy_mae = policy_row["mae_bpm"].iloc[0] if not policy_row.empty else np.nan
        high_rr_mae = policy_row["high_rr_mae_bpm"].iloc[0] if not policy_row.empty else np.nan
        signed_error = policy_row["signed_error_mean_bpm"].iloc[0] if not policy_row.empty else np.nan

        cols = st.columns(4)
        cols[0].metric("Latent RR", f"{fmt(sample['pred_rr_bpm'].iloc[-1])} BPM")
        cols[1].metric("Subject MAE", f"{fmt(metrics['mae'])} BPM")
        cols[2].metric("Policy MAE", f"{fmt(policy_mae)} BPM")
        cols[3].metric("High-RR MAE", f"{fmt(high_rr_mae)} BPM")
        evidence_badge(
            level="Claim-B",
            scope="Latent-state inference",
            boundary=T95_LOSO_COPY if policy.startswith("t95_") else T94_LATENT_COPY,
            mode="warn",
        )
        if policy == T95_COMBINED_POLICY:
            st.info(
                "Default view: LOSO combined selection. It is less optimistic than full-data T94, "
                "but it is more review-ready because the held-out subject was not used for policy selection."
            )
        elif policy == T95_HIGH_RR_POLICY:
            st.warning(
                "High-RR selection is the most tail-protective view, but it carries a stronger positive bias. "
                "Use it as a safety-oriented upper-bound experiment, not a final default."
            )
        elif policy == T94_BALANCED_POLICY:
            st.info("T94 balanced is the full-data exploratory latent-state policy. T95 should be used for validation claims.")

        center, right = st.columns([0.66, 0.34], gap="large")
        with center:
            st.plotly_chart(
                trend_chart(
                    sample,
                    title=f"Cambridge latent-state RR / {subject}<br>{compact_method_name(policy)}",
                    reference_col="gt_rr_bpm",
                    raw_col=None,
                    predicted_col="pred_rr_bpm",
                ),
                width="stretch",
            )
            table_cols = [
                "window_id",
                "start_sec",
                "gt_rr_bpm",
                "pred_rr_bpm",
                "abs_error_bpm",
                "signed_error_bpm",
                "is_high_rr",
                "selected_policy",
                "selection_objective",
                "tail_signal",
                "evidence_level",
                "claim_boundary",
            ]
            st.dataframe(sample[[col for col in table_cols if col in sample.columns]], width="stretch", hide_index=True)
        with right:
            st.subheader("Latent-State Evidence")
            st.markdown(
                "<span class='vs-chip vs-chip-warn'>Research policy</span>"
                "<span class='vs-chip'>Cambridge public RGB-D</span>"
                "<span class='vs-chip'>Not diagnostic</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Problem solved: reduce harmonic/half-rate mistakes by inferring a continuous hidden RR trajectory."
            )
            if not policy_row.empty:
                display_cols = [
                    "policy",
                    "display_name",
                    "selection_objective",
                    "n_subjects",
                    "n_windows",
                    "mae_bpm",
                    "rmse_bpm",
                    "high_rr_mae_bpm",
                    "signed_error_mean_bpm",
                    "combined_tail_score",
                ]
                display = policy_row[[col for col in display_cols if col in policy_row.columns]].copy()
                for col in ["mae_bpm", "rmse_bpm", "high_rr_mae_bpm", "signed_error_mean_bpm", "combined_tail_score"]:
                    if col in display.columns:
                        display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                st.dataframe(display, width="stretch", hide_index=True)
            ci_display = delta_ci_table(source_ci, policy, reference="t87_blend075")
            if not ci_display.empty:
                st.subheader("Bootstrap Boundary")
                st.dataframe(ci_display, width="stretch", hide_index=True)
            if policy.startswith("t95_") and not t95_folds.empty:
                objective = policy.replace("t95_loso_", "")
                fold = t95_folds[
                    (t95_folds["selection_objective"].eq(objective)) & (t95_folds["heldout_subject"].eq(subject))
                ].copy()
                if not fold.empty:
                    st.subheader("Held-Out Fold")
                    fold_display = fold[
                        [
                            "heldout_subject",
                            "selected_policy",
                            "train_metric_used",
                            "test_mae_bpm",
                            "test_high_rr_mae_bpm",
                            "test_signed_error_mean_bpm",
                        ]
                    ].copy()
                    for col in ["test_mae_bpm", "test_high_rr_mae_bpm", "test_signed_error_mean_bpm"]:
                        fold_display[col] = pd.to_numeric(fold_display[col], errors="coerce").round(3)
                    st.dataframe(fold_display, width="stretch", hide_index=True)
            if not t95_counts.empty:
                st.subheader("Selection Counts")
                counts = t95_counts.copy()
                counts["selected_policy"] = counts["selected_policy"].map(compact_method_name)
                st.dataframe(counts, width="stretch", hide_index=True)
            export_panel(sample, title=f"Cambridge_latent_{subject}_{policy}")
        return

    if source.startswith("T87"):
        data = load_cambridge_t87_windows()
        policy_summary = load_cambridge_t87_policy_summary()
        bootstrap_ci = load_cambridge_t87_bootstrap_ci()
        subject_summary = load_cambridge_t87_subject_summary()
        warmup_summary = load_cambridge_t87_warmup_summary()
        if data.empty:
            st.error("T87 Cambridge calibration outputs are unavailable.")
            return
        policies = cambridge_policy_options(policy_summary, data)
        default_index = policies.index(T87_DEFAULT_POLICY) if T87_DEFAULT_POLICY in policies else 0
        policy = st.sidebar.selectbox(
            "Neonatal policy",
            policies,
            index=default_index,
            format_func=compact_method_name,
        )
        available = data[data["policy"].eq(policy)].copy()
        subjects = sorted(available["subject"].dropna().unique())
        subject = st.sidebar.selectbox("Subject", subjects)
        sample = available[available["subject"].eq(subject)].sort_values("window_id").copy()
        t92_summary = load_cambridge_t92_policy_summary()
        t92_events = load_cambridge_t92_event_details()
        t92_alerts = load_cambridge_t92_window_alerts()
        t92_rules = []
        if not t92_summary.empty:
            t92_rules = (
                t92_summary[t92_summary["policy"].eq(policy)]["rule_id"].dropna().astype(str).drop_duplicates().tolist()
            )
        if not t92_rules:
            t92_rules = [T92_DEFAULT_RULE]
        t92_rule_index = t92_rules.index(T92_DEFAULT_RULE) if T92_DEFAULT_RULE in t92_rules else 0
        t92_rule = st.sidebar.selectbox(
            "Alert rule",
            t92_rules,
            index=t92_rule_index,
            format_func=t92_rule_display_name,
        )
        actionability_sample = pd.DataFrame()
        if not t92_alerts.empty:
            actionability_sample = t92_alerts[
                (t92_alerts["policy"].eq(policy))
                & (t92_alerts["rule_id"].eq(t92_rule))
                & (t92_alerts["subject"].eq(subject))
            ].copy()
            if not actionability_sample.empty:
                actionability_sample = actionability_sample.sort_values("window_id")
        chart_sample = actionability_sample if not actionability_sample.empty else sample
        sample["evidence_level"] = "Claim-B / high-RR mitigation"
        sample["claim_boundary"] = T87_RESEARCH_COPY
        if not actionability_sample.empty:
            actionability_sample["evidence_level"] = "Claim-B / actionability workflow"
            actionability_sample["claim_boundary"] = T92_ACTIONABILITY_COPY

        metrics = summarize_window_metrics(sample, pred_col="pred_rr_bpm", gt_col="gt_rr_bpm")
        policy_row = policy_summary[policy_summary["policy"].eq(policy)].head(1)
        subject_row = subject_summary[
            (subject_summary["policy"].eq(policy)) & (subject_summary["subject"].eq(subject))
        ].head(1)
        base_subject_row = subject_summary[
            (subject_summary["policy"].eq(T87_BASE_POLICY)) & (subject_summary["subject"].eq(subject))
        ].head(1)
        policy_mae = policy_row["mae_bpm"].iloc[0] if not policy_row.empty else np.nan
        high_rr_mae = policy_row["high_rr_mae_bpm"].iloc[0] if not policy_row.empty else np.nan

        cols = st.columns(4)
        cols[0].metric("Neonatal RR", f"{fmt(sample['pred_rr_bpm'].iloc[-1])} BPM")
        cols[1].metric("Subject MAE", f"{fmt(metrics['mae'])} BPM")
        cols[2].metric("Policy MAE", f"{fmt(policy_mae)} BPM")
        cols[3].metric("High-RR MAE", f"{fmt(high_rr_mae)} BPM")
        evidence_badge(
            level="Claim-B",
            scope="High-RR mitigation",
            boundary=f"{T87_HIGH_RR_COPY} {T87_RESEARCH_COPY}",
            mode="warn",
        )
        if policy == T87_MAX_POLICY:
            st.warning(
                "The max policy is a high-RR recovery upper bound. It can reduce high-RR error, "
                "but it may overcorrect lower-RR windows, so it should not be treated as the default product rule."
            )
        elif policy == T87_Q90_POLICY:
            st.info("q90 is a high-RR mitigation candidate with near-zero overall signed bias in T87.")
        elif policy == T87_DEFAULT_POLICY:
            st.info("blend075 is the default T87 view because it gives the best first-pass all-window MAE.")

        center, right = st.columns([0.66, 0.34], gap="large")
        with center:
            st.plotly_chart(
                trend_chart(
                    chart_sample,
                    title=f"Cambridge T87 high-RR calibration / {subject}<br>{compact_method_name(policy)}",
                    reference_col="gt_rr_bpm",
                    raw_col=None,
                    predicted_col="pred_rr_bpm",
                ),
                width="stretch",
            )
            table_cols = [
                "window_id",
                "start_sec",
                "gt_rr_bpm",
                "pred_rr_bpm",
                "abs_error_bpm",
                "signed_error_bpm",
                "is_high_rr",
                "base_pred_rr_bpm",
                "q90_pred_rr_bpm",
                "q90_minus_base_bpm",
                "alert",
                "warning_reason",
                "rule_id",
                "evidence_level",
                "claim_boundary",
            ]
            st.dataframe(
                chart_sample[[col for col in table_cols if col in chart_sample.columns]],
                width="stretch",
                hide_index=True,
            )
        with right:
            st.subheader("Evidence Level")
            st.markdown(
                "<span class='vs-chip vs-chip-warn'>Research policy</span>"
                "<span class='vs-chip'>Public Cambridge RGB-D</span>"
                "<span class='vs-chip'>Not diagnostic</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Supported claim: {T87_HIGH_RR_COPY} Boundary: {T87_BOOTSTRAP_BOUNDARY_COPY}"
            )
            st.subheader("Policy Evidence")
            if not policy_row.empty:
                display_cols = [
                    "policy",
                    "n_subjects",
                    "n_windows",
                    "mae_bpm",
                    "rmse_bpm",
                    "pearson",
                    "signed_error_mean_bpm",
                    "high_rr_mae_bpm",
                    "high_rr_signed_error_mean_bpm",
                    "worst_subject",
                    "worst_subject_mae_bpm",
                ]
                display = policy_row[[col for col in display_cols if col in policy_row.columns]].copy()
                for col in [
                    "mae_bpm",
                    "rmse_bpm",
                    "pearson",
                    "signed_error_mean_bpm",
                    "high_rr_mae_bpm",
                    "high_rr_signed_error_mean_bpm",
                    "worst_subject_mae_bpm",
                ]:
                    if col in display.columns:
                        display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                st.dataframe(display, width="stretch", hide_index=True)
            ci_display = t87_ci_table(bootstrap_ci, policy)
            if not ci_display.empty:
                st.subheader("Bootstrap Boundary")
                st.dataframe(ci_display, width="stretch", hide_index=True)
            if not subject_row.empty:
                st.subheader("Subject High-RR Check")
                subject_display = subject_row[
                    [
                        "subject",
                        "n_windows",
                        "mae_bpm",
                        "signed_error_mean_bpm",
                        "high_rr_windows",
                        "high_rr_mae_bpm",
                        "high_rr_signed_error_mean_bpm",
                    ]
                ].copy()
                for col in ["mae_bpm", "signed_error_mean_bpm", "high_rr_mae_bpm", "high_rr_signed_error_mean_bpm"]:
                    subject_display[col] = pd.to_numeric(subject_display[col], errors="coerce").round(3)
                st.dataframe(subject_display, width="stretch", hide_index=True)
                high_windows = pd.to_numeric(subject_row["high_rr_windows"], errors="coerce").iloc[0]
                current_high = pd.to_numeric(subject_row["high_rr_mae_bpm"], errors="coerce").iloc[0]
                base_high = (
                    pd.to_numeric(base_subject_row["high_rr_mae_bpm"], errors="coerce").iloc[0]
                    if not base_subject_row.empty
                    else np.nan
                )
                if np.isfinite(high_windows) and high_windows > 0 and np.isfinite(base_high) and np.isfinite(current_high):
                    delta = current_high - base_high
                    st.info(
                        f"High-RR windows: {int(high_windows)}. "
                        f"T58 base high-RR MAE {fmt(base_high)} BPM -> selected policy {fmt(current_high)} BPM "
                        f"(delta {fmt(delta)} BPM)."
                    )
                    if current_high >= 10:
                        st.warning(T87_SUBJECT_CAUTION_COPY)
                    else:
                        st.success(T87_SUBJECT_IMPROVED_COPY)
            t92_row = t92_policy_rule_row(t92_summary, policy, t92_rule)
            if not t92_row.empty:
                st.subheader("Actionability Layer")
                selected_t92 = t92_row.iloc[0]
                action_cols = st.columns(2)
                action_cols[0].metric("Event sensitivity", fmt(selected_t92["event_sensitivity"], 3))
                action_cols[1].metric("Missed events", fmt(selected_t92["missed_high_rr_events"], 0))
                action_cols = st.columns(2)
                action_cols[0].metric("False alerts", fmt(selected_t92["false_alert_episodes"], 0))
                action_cols[1].metric("Burden/hour", fmt(selected_t92["alarm_burden_per_hour"], 2))
                action_cols = st.columns(2)
                action_cols[0].metric("Detect delay", f"{fmt(selected_t92['mean_time_to_detect_sec'], 1)} sec")
                action_cols[1].metric("Window precision", fmt(selected_t92["window_precision"], 3))
                st.dataframe(t92_actionability_display(selected_t92), width="stretch", hide_index=True)
                baseline_t92 = t92_policy_rule_row(t92_summary, policy, T92_BASELINE_RULE)
                if not baseline_t92.empty and t92_rule != T92_BASELINE_RULE:
                    base = baseline_t92.iloc[0]
                    st.info(
                        f"{t92_rule_display_name(T92_BASELINE_RULE)} -> {t92_rule_display_name(t92_rule)}: "
                        f"false alert episodes {fmt(base['false_alert_episodes'], 0)} -> "
                        f"{fmt(selected_t92['false_alert_episodes'], 0)}, "
                        f"alarm burden/hour {fmt(base['alarm_burden_per_hour'], 2)} -> "
                        f"{fmt(selected_t92['alarm_burden_per_hour'], 2)}, "
                        f"mean time-to-detect {fmt(base['mean_time_to_detect_sec'], 1)} -> "
                        f"{fmt(selected_t92['mean_time_to_detect_sec'], 1)} sec."
                    )
                st.caption(T92_ACTIONABILITY_COPY)
                event_focus = pd.DataFrame()
                if not t92_events.empty:
                    event_focus = t92_events[
                        (t92_events["policy"].eq(policy))
                        & (t92_events["rule_id"].eq(t92_rule))
                        & (t92_events["subject"].eq(subject))
                    ].copy()
                if not event_focus.empty:
                    for col in ["start_sec", "end_sec", "time_to_detect_sec"]:
                        event_focus[col] = pd.to_numeric(event_focus[col], errors="coerce").round(2)
                    st.dataframe(
                        event_focus[
                            [
                                "event_type",
                                "event_id",
                                "start_sec",
                                "end_sec",
                                "n_windows",
                                "detected",
                                "time_to_detect_sec",
                                "contains_gt_high_rr",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
            if not warmup_summary.empty:
                st.subheader("Adaptation Potential")
                warm = warmup_summary[
                    warmup_summary["policy"].isin(["postwarmup_fixed_t58_base", "postwarmup_supervised_subject_adaptive"])
                ].copy()
                if not warm.empty:
                    warm = warm[["policy", "mae_bpm", "high_rr_mae_bpm", "signed_error_mean_bpm"]]
                    for col in ["mae_bpm", "high_rr_mae_bpm", "signed_error_mean_bpm"]:
                        warm[col] = pd.to_numeric(warm[col], errors="coerce").round(3)
                    st.dataframe(warm, width="stretch", hide_index=True)
                    st.caption(T87_WARMUP_BOUNDARY_COPY)
            export_panel(sample, title=f"Cambridge_T87_{subject}_{policy}")
        return

    if source.startswith("T58"):
        data = load_cambridge_t58_windows()
        policy_summary = load_cambridge_t58_policy_summary()
        stability = load_cambridge_t58_stability_ci()
        subject_stability = load_cambridge_t58_subject_stability()
        if data.empty:
            st.error("T58 Cambridge outputs are unavailable.")
            return
        policies = cambridge_policy_options(policy_summary, data)
        default_index = policies.index(T58_BEST_POLICY) if T58_BEST_POLICY in policies else 0
        policy = st.sidebar.selectbox("Neonatal policy", policies, index=default_index, format_func=compact_method_name)
        available = data[data["policy"].eq(policy)].copy()
        subjects = sorted(available["subject"].dropna().unique())
        subject = st.sidebar.selectbox("Subject", subjects)
        sample = available[available["subject"].eq(subject)].sort_values("window_id").copy()

        metrics = summarize_window_metrics(sample, pred_col="pred_rr_bpm", gt_col="gt_rr_bpm")
        policy_row = policy_summary[policy_summary["policy"].eq(policy)].head(1)
        subject_row = subject_stability[
            (subject_stability["policy"].eq(policy)) & (subject_stability["subject"].eq(subject))
        ].head(1)
        policy_mae = policy_row["mae_bpm"].iloc[0] if not policy_row.empty else np.nan
        subject_signed = subject_row["signed_error_mean_bpm"].iloc[0] if not subject_row.empty else np.nan

        cols = st.columns(4)
        cols[0].metric("Neonatal RR", f"{fmt(sample['pred_rr_bpm'].iloc[-1])} BPM")
        cols[1].metric("Subject MAE", f"{fmt(metrics['mae'])} BPM")
        cols[2].metric("Policy MAE", f"{fmt(policy_mae)} BPM")
        cols[3].metric("Delta vs legacy", t58_delta_text(stability, policy))
        st.info(T58_RESEARCH_COPY)

        center, right = st.columns([0.66, 0.34], gap="large")
        with center:
            st.plotly_chart(
                trend_chart(
                    sample,
                    title=f"Cambridge neonatal / {subject}<br>{compact_method_name(policy)}",
                    reference_col="gt_rr_bpm",
                    raw_col=None,
                    predicted_col="pred_rr_bpm",
                ),
                width="stretch",
            )
            table_cols = [
                "window_id",
                "start_sec",
                "relative_start_sec",
                "gt_rr_bpm",
                "pred_rr_bpm",
                "abs_error_bpm",
                "confidence",
                "decision",
                "harmonic_multiplier",
                "harmonic_support_ratio",
                "selected_method",
            ]
            st.dataframe(sample[[col for col in table_cols if col in sample.columns]], width="stretch", hide_index=True)
        with right:
            st.subheader("Policy Evidence")
            if not policy_row.empty:
                display = policy_row[
                    [
                        "policy",
                        "n_subjects",
                        "n_windows",
                        "coverage",
                        "mae_bpm",
                        "rmse_bpm",
                        "pearson",
                        "mean_reference_rr_bpm",
                        "mean_pred_rr_bpm",
                    ]
                ].copy()
                for col in ["coverage", "mae_bpm", "rmse_bpm", "pearson", "mean_reference_rr_bpm", "mean_pred_rr_bpm"]:
                    display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                st.dataframe(display, width="stretch", hide_index=True)
            ci_display = t58_stability_table(stability, policy)
            if not ci_display.empty:
                st.subheader("Bootstrap CI")
                st.dataframe(ci_display, width="stretch", hide_index=True)
            if not subject_row.empty:
                st.subheader("Subject Stability")
                subject_display = subject_row[
                    ["subject", "n_windows", "mae_bpm", "rmse_bpm", "signed_error_mean_bpm"]
                ].copy()
                for col in ["mae_bpm", "rmse_bpm", "signed_error_mean_bpm"]:
                    subject_display[col] = pd.to_numeric(subject_display[col], errors="coerce").round(3)
                st.dataframe(subject_display, width="stretch", hide_index=True)
                if np.isfinite(float(subject_signed)) and abs(float(subject_signed)) >= 8:
                    st.warning("Use caution for this subject: the current neonatal trend remains a high-error stability case.")
            export_panel(sample, title=f"Cambridge_T58_{subject}_{policy}")
        return

    data = load_cambridge_results()
    methods = sorted(data["method"].unique())
    method = st.sidebar.selectbox("Channel", methods, index=methods.index("depth_right_chest") if "depth_right_chest" in methods else 0)
    available = data[data["method"].eq(method)]
    subjects = sorted(available["subject"].unique())
    subject = st.sidebar.selectbox("Subject", subjects)
    sample = available[available["subject"].eq(subject)].sort_values("window_id")

    metrics = summarize_window_metrics(sample, pred_col="pred_rr_bpm", gt_col="gt_rr_bpm")
    cols = st.columns(4)
    cols[0].metric("Current RR", f"{fmt(sample['pred_rr_bpm'].iloc[-1])} BPM")
    cols[1].metric("Window MAE", f"{fmt(metrics['mae'])} BPM")
    cols[2].metric("Mean confidence", fmt(sample["confidence"].mean(), 3))
    cols[3].metric("Windows", str(len(sample)))

    st.plotly_chart(
        trend_chart(
            sample,
            title=f"Cambridge legacy / {subject}<br>{compact_method_name(method)}",
            reference_col="gt_rr_bpm",
            raw_col=None,
            predicted_col="pred_rr_bpm",
        ),
        width="stretch",
    )
    legacy_cols = ["window_id", "start_sec", "gt_rr_bpm", "pred_rr_bpm", "abs_error_bpm", "confidence", "camera_column"]
    st.dataframe(sample[[col for col in legacy_cols if col in sample.columns]], width="stretch", hide_index=True)
    export_panel(sample, title=f"Cambridge_T39_{subject}_{method}")


def benchmark_overview() -> None:
    bench = load_benchmark()
    quality = load_quality_summary()
    t57 = load_t57_policy_summary()
    adult_t162_panel = load_adult_t162_explanation_panel()
    adult_t162_status = load_adult_t162_status_summary()
    adult_t162_checklist = load_adult_t162_claim_checklist()
    adult_t162_protocol = load_adult_t162_protocol_json()
    adult_t162_summary = load_adult_t162_summary_json()
    adult_t161_summary = load_adult_t161_policy_summary()
    adult_t163_qa = load_adult_t163_qa_checklist()
    adult_t163_run_order = load_adult_t163_run_order()
    adult_t163_gate_matrix = load_adult_t163_gate_matrix()
    adult_t163_summary = load_adult_t163_summary_json()
    adult_t164_policy_summary = load_adult_t164_policy_summary()
    adult_t164_repro_check = load_adult_t164_repro_check()
    adult_t164_leakage_audit = load_adult_t164_leakage_audit()
    adult_t164_data_completeness = load_adult_t164_data_completeness()
    adult_t164_bootstrap = load_adult_t164_bootstrap()
    adult_t164_summary = load_adult_t164_summary_json()
    adult_t165_readiness = load_adult_t165_readiness()
    adult_t165_actions = load_adult_t165_access_actions()
    adult_t165_plan = load_adult_t165_locked_plan()
    adult_t165_summary = load_adult_t165_summary_json()
    adult_t357_product = load_adult_t357_product_table()
    adult_t357_branch_summary = load_adult_t357_branch_summary()
    adult_t357_qa = load_adult_t357_qa_checks()
    adult_t357_api_examples = load_adult_t357_api_examples()
    adult_t357_summary = load_adult_t357_summary_json()
    adult_t382_product = load_adult_t382_product_table()
    adult_t382_branch_summary = load_adult_t382_branch_summary()
    adult_t382_qa = load_adult_t382_qa_checks()
    adult_t382_api_examples = load_adult_t382_api_examples()
    adult_t382_summary = load_adult_t382_summary_json()
    adult_t478_summary = load_adult_current_summary_json(str(ADULT_T478_SUMMARY_JSON))
    adult_t478_metric_cards = load_adult_current_csv(str(ADULT_T478_METRIC_CARDS))
    adult_t478_claim_gate = load_adult_current_csv(str(ADULT_T478_CLAIM_GATE))
    adult_t481_summary = load_adult_current_summary_json(str(ADULT_T481_SUMMARY_JSON))
    adult_t481_router = load_adult_current_csv(str(ADULT_T481_ROUTER_TABLE))
    adult_t481_claim_gate = load_adult_current_csv(str(ADULT_T481_CLAIM_GATE))
    adult_t481_api_examples = load_adult_t481_api_examples()
    adult_t482_summary = load_adult_current_summary_json(str(ADULT_T482_SUMMARY_JSON))
    adult_t482_qa = load_adult_current_csv(str(ADULT_T482_QA))
    adult_t482_source = load_adult_current_csv(str(ADULT_T482_SOURCE_DATA))
    adult_t485_summary = load_adult_current_summary_json(str(ADULT_T485_SUMMARY_JSON))
    adult_t485_signal_index = load_adult_current_csv(str(ADULT_T485_SIGNAL_INDEX))
    adult_t485_claim_gate = load_adult_current_csv(str(ADULT_T485_CLAIM_GATE))
    adult_t486_summary = load_adult_current_summary_json(str(ADULT_T486_SUMMARY_JSON))
    adult_t486_condition_index = load_adult_current_csv(str(ADULT_T486_CONDITION_INDEX))
    adult_t486_claim_gate = load_adult_current_csv(str(ADULT_T486_CLAIM_GATE))
    adult_t487_summary = load_adult_current_summary_json(str(ADULT_T487_SUMMARY_JSON))
    adult_t487_metrics = load_adult_current_csv(str(ADULT_T487_METRICS))
    adult_t487_delta = load_adult_current_csv(str(ADULT_T487_DELTA))
    adult_t487_claim_gate = load_adult_current_csv(str(ADULT_T487_CLAIM_GATE))
    adult_t488_summary = load_adult_current_summary_json(str(ADULT_T488_SUMMARY_JSON))
    adult_t488_subset = load_adult_current_csv(str(ADULT_T488_SUBSET))
    adult_t488_budget = load_adult_current_csv(str(ADULT_T488_BUDGET))
    adult_t488_claim_gate = load_adult_current_csv(str(ADULT_T488_CLAIM_GATE))
    adult_t489_summary = load_adult_current_summary_json(str(ADULT_T489_SUMMARY_JSON))
    adult_t489_claim_gate = load_adult_current_csv(str(ADULT_T489_CLAIM_GATE))
    adult_t491_summary = load_adult_current_summary_json(str(ADULT_T491_SUMMARY_JSON))
    adult_t491_trace_index = load_adult_current_csv(str(ADULT_T491_TRACE_INDEX))
    adult_t491_claim_gate = load_adult_current_csv(str(ADULT_T491_CLAIM_GATE))
    adult_t492_summary = load_adult_current_summary_json(str(ADULT_T492_SUMMARY_JSON))
    adult_t492_decisions = load_adult_current_csv(str(ADULT_T492_MR_DECISIONS))
    adult_t492_claim_gate = load_adult_current_csv(str(ADULT_T492_CLAIM_GATE))
    adult_t493_summary = load_adult_current_summary_json(str(ADULT_T493_SUMMARY_JSON))
    adult_t493_trace_index = load_adult_current_csv(str(ADULT_T493_TRACE_INDEX))
    adult_t493_claim_gate = load_adult_current_csv(str(ADULT_T493_CLAIM_GATE))
    adult_t494_summary = load_adult_current_summary_json(str(ADULT_T494_SUMMARY_JSON))
    adult_t494_decisions = load_adult_current_csv(str(ADULT_T494_DECISIONS))
    adult_t494_metrics = load_adult_current_csv(str(ADULT_T494_METRICS))
    adult_t494_claim_gate = load_adult_current_csv(str(ADULT_T494_CLAIM_GATE))
    adult_t495_summary = load_adult_current_summary_json(str(ADULT_T495_SUMMARY_JSON))
    adult_t495_decisions = load_adult_current_csv(str(ADULT_T495_DECISIONS))
    adult_t495_metrics = load_adult_current_csv(str(ADULT_T495_METRICS))
    adult_t495_delta = load_adult_current_csv(str(ADULT_T495_DELTA))
    adult_t495_claim_gate = load_adult_current_csv(str(ADULT_T495_CLAIM_GATE))
    adult_t497_summary = load_adult_current_summary_json(str(ADULT_T497_SUMMARY_JSON))
    adult_t497_decisions = load_adult_current_csv(str(ADULT_T497_DECISIONS))
    adult_t497_metrics = load_adult_current_csv(str(ADULT_T497_METRICS))
    adult_t497_claim_gate = load_adult_current_csv(str(ADULT_T497_CLAIM_GATE))
    adult_t498_summary = load_adult_current_summary_json(str(ADULT_T498_SUMMARY_JSON))
    adult_t498_decisions = load_adult_current_csv(str(ADULT_T498_DECISIONS))
    adult_t498_metrics = load_adult_current_csv(str(ADULT_T498_METRICS))
    adult_t498_delta = load_adult_current_csv(str(ADULT_T498_DELTA))
    adult_t498_claim_gate = load_adult_current_csv(str(ADULT_T498_CLAIM_GATE))
    adult_t504_summary = load_adult_current_summary_json(str(ADULT_T504_SUMMARY_JSON))
    adult_t504_decisions = load_adult_current_csv(str(ADULT_T504_DECISIONS))
    adult_t504_metrics = load_adult_current_csv(str(ADULT_T504_METRICS))
    adult_t504_delta = load_adult_current_csv(str(ADULT_T504_DELTA))
    adult_t504_taxonomy = load_adult_current_csv(str(ADULT_T504_TAXONOMY))
    adult_t504_claim_gate = load_adult_current_csv(str(ADULT_T504_CLAIM_GATE))
    adult_t505_summary = load_adult_current_summary_json(str(ADULT_T505_SUMMARY_JSON))
    adult_t505_delta = load_adult_current_csv(str(ADULT_T505_DELTA))
    adult_t505_claim_gate = load_adult_current_csv(str(ADULT_T505_CLAIM_GATE))
    adult_t506_summary = load_adult_current_summary_json(str(ADULT_T506_SUMMARY_JSON))
    adult_t506_product = load_adult_current_csv(str(ADULT_T506_PRODUCT_TABLE))
    adult_t506_product_summary = load_adult_current_csv(str(ADULT_T506_PRODUCT_SUMMARY))
    adult_t506_delta = load_adult_current_csv(str(ADULT_T506_DELTA))
    adult_t506_api_examples = load_adult_current_summary_json(str(ADULT_T506_API_EXAMPLES))
    adult_t506_claim_gate = load_adult_current_csv(str(ADULT_T506_CLAIM_GATE))
    adult_t508_summary = load_adult_current_summary_json(str(ADULT_T508_SUMMARY_JSON))
    adult_t508_delta = load_adult_current_csv(str(ADULT_T508_DELTA))
    adult_t508_claim_gate = load_adult_current_csv(str(ADULT_T508_CLAIM_GATE))
    adult_t509_summary = load_adult_current_summary_json(str(ADULT_T509_SUMMARY_JSON))
    adult_t509_bootstrap = load_adult_current_csv(str(ADULT_T509_BOOTSTRAP))
    adult_t509_endpoint_gates = load_adult_current_csv(str(ADULT_T509_ENDPOINT_GATES))
    adult_t510_summary = load_adult_current_summary_json(str(ADULT_T510_SUMMARY_JSON))
    adult_t510_delta = load_adult_current_csv(str(ADULT_T510_DELTA))
    adult_t510_claim_gate = load_adult_current_csv(str(ADULT_T510_CLAIM_GATE))
    adult_t511_summary = load_adult_current_summary_json(str(ADULT_T511_SUMMARY_JSON))
    adult_t511_product = load_adult_current_csv(str(ADULT_T511_PRODUCT_TABLE))
    adult_t511_product_summary = load_adult_current_csv(str(ADULT_T511_PRODUCT_SUMMARY))
    adult_t511_api_examples = load_adult_current_summary_json(str(ADULT_T511_API_EXAMPLES))
    adult_t511_claim_gate = load_adult_current_csv(str(ADULT_T511_CLAIM_GATE))
    adult_t517_summary = load_adult_current_summary_json(str(ADULT_T517_SUMMARY_JSON))
    adult_t517_routes = load_adult_current_csv(str(ADULT_T517_ROUTE_TABLE))
    adult_t517_claim_gate = load_adult_current_csv(str(ADULT_T517_CLAIM_GATE))
    adult_t518_summary = load_adult_current_summary_json(str(ADULT_T518_SUMMARY_JSON))
    adult_t518_policy = load_adult_current_csv(str(ADULT_T518_PRODUCT_POLICY))
    adult_t518_api_examples = load_adult_current_summary_json(str(ADULT_T518_API_EXAMPLES))
    adult_t518_claim_gate = load_adult_current_csv(str(ADULT_T518_CLAIM_GATE))
    adult_t527_summary = load_adult_current_summary_json(str(ADULT_T527_SUMMARY_JSON))
    adult_t527_cases = load_adult_current_csv(str(ADULT_T527_DEMO_CASES))
    adult_t527_api_packets = load_adult_current_summary_json(str(ADULT_T527_API_PACKETS))
    adult_t527_report_cards = load_adult_current_csv(str(ADULT_T527_REPORT_CARDS))
    adult_t527_qa_checks = load_adult_current_csv(str(ADULT_T527_QA_CHECKS))
    cols = st.columns(4)
    best = bench.sort_values("our_mae_bpm").iloc[0]
    cols[0].metric("Best AIR/Cambridge MAE", f"{fmt(best['our_mae_bpm'])} BPM")
    cols[1].metric("Datasets", str(bench["dataset"].nunique()))
    cols[2].metric("Methods", str(len(bench)))
    cols[3].metric("Windows", str(int(pd.to_numeric(bench["n_windows"], errors="coerce").sum())))
    st.plotly_chart(benchmark_bar(bench), width="stretch")
    st.dataframe(
        bench[
            [
                "dataset",
                "population",
                "method",
                "n_samples",
                "n_windows",
                "baseline_mae_bpm",
                "our_mae_bpm",
                "delta_mae_bpm",
                "coverage",
                "half_decision_rate",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    if not adult_t162_panel.empty:
        st.subheader("T162 Adult HR Explanation Panel")
        status_counts = adult_t162_panel["product_status"].value_counts()
        unsafe_count = int(pd.to_numeric(adult_t162_panel["research_unsafe_release"], errors="coerce").fillna(0).sum())
        explained = int(adult_t162_panel["explanation_title"].notna().sum())
        metrics = st.columns(4)
        metrics[0].metric("Explained decisions", f"{explained}/{len(adult_t162_panel)}")
        metrics[1].metric("Released", str(int(status_counts.get("release", 0))))
        metrics[2].metric("Rescued", str(int(status_counts.get("rescued_release", 0))))
        metrics[3].metric("Unsafe releases", str(unsafe_count))
        st.table(adult_t162_status_table(adult_t162_status))
        st.caption(
            adult_t162_summary.get(
                "main_insight",
                "Every adult HR output is released, rescued with auditable evidence, or withheld with a concrete review reason.",
            )
        )

        subject_options = list(adult_t162_panel["subject_id"].astype(str))
        default_subject = "S09" if "S09" in subject_options else subject_options[0]
        subject = st.selectbox(
            "Adult HR subject explanation",
            subject_options,
            index=subject_options.index(default_subject),
            key="adult_t162_subject",
        )
        selected = adult_t162_panel[adult_t162_panel["subject_id"].astype(str).eq(subject)].iloc[0]
        key_value_panel(
            [
                ("Product output", selected.get("product_output", "NA")),
                ("Status", selected.get("product_status", "NA")),
                ("Explanation", selected.get("explanation_title", "NA")),
                ("Evidence chips", selected.get("evidence_chips", "NA")),
                ("Recommended action", selected.get("recommended_action", "NA")),
                ("User-facing copy", selected.get("user_facing_copy", "NA")),
                ("Research abs error", f"{fmt(selected.get('research_abs_error_bpm'))} BPM"),
                ("Release rule", selected.get("release_rule_version", "NA")),
            ]
        )
        with st.expander("Subject-level explanation table", expanded=False):
            st.dataframe(adult_t162_subject_table(adult_t162_panel), width="stretch", hide_index=True)
        with st.expander("Locked external-validation protocol", expanded=True):
            frozen = adult_t162_protocol.get("frozen_rule", {}) if isinstance(adult_t162_protocol, dict) else {}
            design = adult_t162_protocol.get("locked_evaluation_design", {}) if isinstance(adult_t162_protocol, dict) else {}
            reference = adult_t162_protocol.get("current_internal_reference", {}) if isinstance(adult_t162_protocol, dict) else {}
            key_value_panel(
                [
                    ("Protocol ID", adult_t162_protocol.get("protocol_id", "NA") if isinstance(adult_t162_protocol, dict) else "NA"),
                    ("Frozen rule", frozen.get("version", "NA")),
                    ("No retuning", design.get("no_threshold_retuning_on_external_validation", "NA")),
                    ("Forbidden decision inputs", ", ".join(frozen.get("decision_inputs_forbidden", [])) if isinstance(frozen.get("decision_inputs_forbidden", []), list) else "NA"),
                    ("Internal ALL coverage", f"{fmt(reference.get('current_all_coverage'), 3)} -> {fmt(reference.get('full_all_coverage'), 3)}"),
                    ("Internal unsafe/input", f"{fmt(reference.get('current_all_unsafe_per_input'), 3)} -> {fmt(reference.get('full_all_unsafe_per_input'), 3)}"),
                ]
            )
            if design.get("primary_success_criteria"):
                st.markdown("\n".join(f"- {item}" for item in design["primary_success_criteria"]))
            if not adult_t162_checklist.empty:
                st.table(adult_t162_claim_table(adult_t162_checklist))
        fig_cols = st.columns(2)
        decision_fig = PROJECT / "output" / "t162_figures" / "t162_product_decision_states.png"
        target_fig = PROJECT / "output" / "t162_figures" / "t162_locked_protocol_target.png"
        if decision_fig.exists():
            fig_cols[0].image(str(decision_fig), caption="T162 decision states", width="stretch")
        if target_fig.exists():
            fig_cols[1].image(str(target_fig), caption="Locked validation target", width="stretch")
        if not adult_t161_summary.empty:
            with st.expander("Adult HR policy summary", expanded=False):
                keep = [
                    ADULT_CURRENT_POLICY,
                    ADULT_T161_FULL_POLICY,
                    "T161_no_high_frequency_veto",
                    "T161_no_top1_subwindow_guard",
                    "T161_quality_only_negative_control",
                ]
                display = adult_t161_summary[
                    adult_t161_summary["dataset"].isin(["rPPG-10", "ALL"]) & adult_t161_summary["policy"].isin(keep)
                ][
                    [
                        "dataset",
                        "policy",
                        "released",
                        "withheld",
                        "coverage",
                        "released_mae_bpm",
                        "unsafe_release_count",
                        "unsafe_per_input",
                    ]
                ].copy()
                for col in ["coverage", "released_mae_bpm", "unsafe_per_input"]:
                    display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                st.dataframe(display, width="stretch", hide_index=True)
        export_panel(adult_t162_panel, title="t162_adult_hr_explanation_panel")
    if not adult_t163_run_order.empty or not adult_t163_gate_matrix.empty:
        st.subheader("T163 QA and Validation Run Order")
        qa_pass = 0
        qa_total = 0
        if not adult_t163_qa.empty and "status" in adult_t163_qa.columns:
            qa_total = len(adult_t163_qa)
            qa_pass = int(adult_t163_qa["status"].isin(["PASS", "PASS_WITH_NOTE"]).sum())
        first_run = {}
        if isinstance(adult_t163_summary, dict):
            first_run = adult_t163_summary.get("first_executable_locked_run", {}) or {}
        gate_ready = ""
        if not adult_t163_gate_matrix.empty and {"gate_id", "status"}.issubset(adult_t163_gate_matrix.columns):
            clean_gate = adult_t163_gate_matrix[adult_t163_gate_matrix["gate_id"].astype(str).eq("G4")]
            if not clean_gate.empty:
                gate_ready = str(clean_gate.iloc[0]["status"])
        metrics = st.columns(4)
        metrics[0].metric("Dashboard QA", f"{qa_pass}/{qa_total}" if qa_total else "NA")
        metrics[1].metric("First locked run", str(first_run.get("dataset", "NA")))
        metrics[2].metric("Run readiness", str(first_run.get("readiness", "NA")))
        metrics[3].metric("Clean external gate", gate_ready or "NA")
        if isinstance(adult_t163_summary, dict) and adult_t163_summary.get("main_insight"):
            st.caption(str(adult_t163_summary["main_insight"]))
        with st.expander("Locked external-validation run order", expanded=True):
            st.dataframe(adult_t163_run_order_table(adult_t163_run_order), width="stretch", hide_index=True)
        with st.expander("Claim gate matrix", expanded=True):
            st.table(adult_t163_gate_table(adult_t163_gate_matrix))
        with st.expander("Dashboard QA checklist", expanded=False):
            st.dataframe(adult_t163_qa_table(adult_t163_qa), width="stretch", hide_index=True)
        fig_cols = st.columns(3)
        t163_figures = [
            ("QA status", PROJECT / "output" / "t163_figures" / "t163_dashboard_qa_status.png"),
            ("Run order", PROJECT / "output" / "t163_figures" / "t163_external_validation_run_order.png"),
            ("Claim gates", PROJECT / "output" / "t163_figures" / "t163_claim_gate_status.png"),
        ]
        for col, (caption, path) in zip(fig_cols, t163_figures, strict=False):
            if path.exists():
                col.image(str(path), caption=caption, width="stretch")
    if not adult_t164_policy_summary.empty or isinstance(adult_t164_summary, dict) and adult_t164_summary:
        st.subheader("T164 UBFC Frozen Replay and Leakage Audit")
        final_policy = adult_t164_policy_summary[
            adult_t164_policy_summary["policy"].astype(str).eq("T160_physio_consistency_rescue_v1")
        ]
        baseline_policy = adult_t164_policy_summary[
            adult_t164_policy_summary["policy"].astype(str).eq("T150_deployment_release_all")
        ]
        final_row = final_policy.iloc[0] if not final_policy.empty else pd.Series(dtype=object)
        baseline_row = baseline_policy.iloc[0] if not baseline_policy.empty else pd.Series(dtype=object)
        leakage_pass = int(adult_t164_leakage_audit["status"].astype(str).eq("PASS").sum()) if not adult_t164_leakage_audit.empty else 0
        leakage_total = len(adult_t164_leakage_audit)
        repro_pass = int(adult_t164_repro_check["status"].astype(str).eq("PASS").sum()) if not adult_t164_repro_check.empty else 0
        repro_total = len(adult_t164_repro_check)
        metrics = st.columns(4)
        metrics[0].metric("UBFC coverage", fmt(final_row.get("coverage"), 3) if not final_row.empty else "NA")
        metrics[1].metric("UBFC MAE", f"{fmt(final_row.get('released_mae_bpm'), 3)} BPM" if not final_row.empty else "NA")
        metrics[2].metric("Unsafe/input", fmt(final_row.get("unsafe_per_input"), 3) if not final_row.empty else "NA")
        metrics[3].metric("Leakage gates", f"{leakage_pass}/{leakage_total}" if leakage_total else "NA")
        if isinstance(adult_t164_summary, dict) and adult_t164_summary.get("main_insight"):
            st.caption(str(adult_t164_summary["main_insight"]))
        if not baseline_row.empty and not final_row.empty:
            st.caption(
                "Replay delta vs T150 release-all: "
                f"MAE {fmt(baseline_row.get('released_mae_bpm'), 3)} -> {fmt(final_row.get('released_mae_bpm'), 3)} BPM; "
                f"unsafe releases {int(baseline_row.get('unsafe_release_count', 0))} -> {int(final_row.get('unsafe_release_count', 0))}."
            )
        key_value_panel(
            [
                ("Verification status", adult_t164_summary.get("verification_status", "NA") if isinstance(adult_t164_summary, dict) else "NA"),
                ("Frozen rule", adult_t164_summary.get("frozen_rule", "NA") if isinstance(adult_t164_summary, dict) else "NA"),
                ("Claim boundary", adult_t164_summary.get("claim_boundary", "NA") if isinstance(adult_t164_summary, dict) else "NA"),
                ("Reproducibility checks", f"{repro_pass}/{repro_total}" if repro_total else "NA"),
            ]
        )
        with st.expander("UBFC replay policy summary", expanded=True):
            st.dataframe(adult_t164_policy_table(adult_t164_policy_summary), width="stretch", hide_index=True)
        with st.expander("Leakage audit gates", expanded=True):
            st.table(adult_t164_audit_table(adult_t164_leakage_audit))
        with st.expander("Raw data completeness", expanded=False):
            st.table(adult_t164_data_table(adult_t164_data_completeness))
        with st.expander("T160 vs T164 reproducibility check", expanded=False):
            st.dataframe(adult_t164_repro_table(adult_t164_repro_check), width="stretch", hide_index=True)
        if not adult_t164_bootstrap.empty:
            with st.expander("Bootstrap delta vs T150", expanded=False):
                bootstrap_display = adult_t164_bootstrap.copy()
                for col in [
                    "observed_mae_delta_bpm",
                    "mae_delta_ci_low",
                    "mae_delta_ci_high",
                    "observed_unsafe_per_input_delta",
                    "unsafe_delta_ci_low",
                    "unsafe_delta_ci_high",
                ]:
                    if col in bootstrap_display.columns:
                        bootstrap_display[col] = pd.to_numeric(bootstrap_display[col], errors="coerce").round(3)
                st.dataframe(bootstrap_display, width="stretch", hide_index=True)
        fig_cols = st.columns(3)
        t164_figures = [
            ("Policy replay metrics", PROJECT / "output" / "t164_figures" / "t164_ubfc_policy_replay_metrics.png"),
            ("Subject error replay", PROJECT / "output" / "t164_figures" / "t164_ubfc_subject_error_replay.png"),
            ("Leakage audit gates", PROJECT / "output" / "t164_figures" / "t164_leakage_audit_gates.png"),
        ]
        for col, (caption, path) in zip(fig_cols, t164_figures, strict=False):
            if path.exists():
                col.image(str(path), caption=caption, width="stretch")
    if not adult_t165_readiness.empty or isinstance(adult_t165_summary, dict) and adult_t165_summary:
        st.subheader("T165 Clean External Dataset Gate")
        clean_ready = int(adult_t165_summary.get("clean_locked_run_ready_count", 0)) if isinstance(adult_t165_summary, dict) else 0
        replay_ready = int(adult_t165_summary.get("replay_ready_not_clean_count", 0)) if isinstance(adult_t165_summary, dict) else 0
        first_choice = str(adult_t165_summary.get("first_choice_dataset", "NA")) if isinstance(adult_t165_summary, dict) else "NA"
        first_status = str(adult_t165_summary.get("first_choice_status", "NA")) if isinstance(adult_t165_summary, dict) else "NA"
        metrics = st.columns(4)
        metrics[0].metric("Clean datasets ready", str(clean_ready))
        metrics[1].metric("Replay-only ready", str(replay_ready))
        metrics[2].metric("First choice", first_choice)
        metrics[3].metric("First-choice status", first_status)
        if isinstance(adult_t165_summary, dict) and adult_t165_summary.get("main_insight"):
            st.caption(str(adult_t165_summary["main_insight"]))
        key_value_panel(
            [
                ("Frozen rule carried forward", adult_t165_summary.get("frozen_rule", "NA") if isinstance(adult_t165_summary, dict) else "NA"),
                ("Claim boundary", adult_t165_summary.get("claim_boundary", "NA") if isinstance(adult_t165_summary, dict) else "NA"),
                ("Next action", adult_t165_summary.get("next_action", "NA") if isinstance(adult_t165_summary, dict) else "NA"),
            ]
        )
        with st.expander("Dataset readiness gate", expanded=True):
            st.dataframe(adult_t165_readiness_table(adult_t165_readiness), width="stretch", hide_index=True)
        with st.expander("Access actions", expanded=True):
            st.dataframe(adult_t165_actions_table(adult_t165_actions), width="stretch", hide_index=True)
        with st.expander("Locked validation plan", expanded=False):
            st.dataframe(adult_t165_plan_table(adult_t165_plan), width="stretch", hide_index=True)
        t165_fig = PROJECT / "output" / "t165_figures" / "t165_dataset_readiness_gate.png"
        if t165_fig.exists():
            st.image(str(t165_fig), caption="T165 clean external dataset readiness gate", width="stretch")
    if not adult_t357_product.empty or isinstance(adult_t357_summary, dict) and adult_t357_summary:
        st.subheader("T357 Frozen Adult HR Experimental Product Mode")
        overall = adult_t357_summary.get("overall", {}) if isinstance(adult_t357_summary, dict) else {}
        qa_pass = 0
        qa_total = 0
        if not adult_t357_qa.empty and "passed" in adult_t357_qa.columns:
            qa_total = len(adult_t357_qa)
            qa_pass = int(adult_t357_qa["passed"].astype(str).str.lower().isin(["true", "1", "pass"]).sum())
        release_count = int(pd.to_numeric(adult_t357_product.get("released", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not adult_t357_product.empty else 0
        review_count = int(len(adult_t357_product) - release_count) if not adult_t357_product.empty else 0
        metrics = st.columns(4)
        metrics[0].metric("Coverage", fmt(overall.get("coverage"), 3))
        metrics[1].metric("Released MAE", f"{fmt(overall.get('released_mae_bpm'), 3)} BPM")
        metrics[2].metric("Unsafe/input", fmt(overall.get("published_unsafe_per_input"), 3))
        metrics[3].metric("QA gates", f"{qa_pass}/{qa_total}" if qa_total else "NA")
        if isinstance(adult_t357_summary, dict) and adult_t357_summary.get("main_insight"):
            st.caption(str(adult_t357_summary["main_insight"]))
        key_value_panel(
            [
                ("Product mode", adult_t357_summary.get("product_mode", "NA") if isinstance(adult_t357_summary, dict) else "NA"),
                ("Released / review", f"{release_count} / {review_count}" if not adult_t357_product.empty else "NA"),
                ("Claim boundary", adult_t357_summary.get("claim_boundary", "NA") if isinstance(adult_t357_summary, dict) else "NA"),
                ("Next action", adult_t357_summary.get("next_recommended_task", "NA") if isinstance(adult_t357_summary, dict) else "NA"),
            ]
        )
        if not adult_t357_branch_summary.empty:
            branch_display = adult_t357_branch_summary.copy()
            for col in ["coverage", "released_mae_bpm", "published_unsafe_per_input"]:
                if col in branch_display.columns:
                    branch_display[col] = pd.to_numeric(branch_display[col], errors="coerce").round(3)
            with st.expander("Experimental branch summary", expanded=True):
                keep_cols = [
                    "policy_branch",
                    "n_rows",
                    "coverage",
                    "released_mae_bpm",
                    "published_unsafe_per_input",
                ]
                st.dataframe(branch_display[[col for col in keep_cols if col in branch_display.columns]], width="stretch", hide_index=True)
        if adult_t357_api_examples:
            labels = [str(item.get("example_type", f"example_{idx}")) for idx, item in enumerate(adult_t357_api_examples)]
            selected_label = st.selectbox("T357 API example", labels, key="adult_t357_api_example")
            selected_example = adult_t357_api_examples[labels.index(selected_label)]
            key_value_panel(
                [
                    ("Example type", selected_example.get("example_type", "NA")),
                    ("Decision", selected_example.get("decision", "NA")),
                    ("Branch", selected_example.get("policy_branch", "NA")),
                    ("Product HR", f"{fmt(selected_example.get('product_hr_bpm'), 2)} BPM"),
                    ("Review reason", selected_example.get("review_reason", "")),
                ]
            )
        with st.expander("T357 product QA checks", expanded=False):
            if not adult_t357_qa.empty:
                st.dataframe(adult_t357_qa, width="stretch", hide_index=True)
    if not adult_t382_product.empty or isinstance(adult_t382_summary, dict) and adult_t382_summary:
        st.subheader("T382 T380 Uncertainty-Aware Adult HR Product Mode")
        overall = adult_t382_summary.get("overall", {}) if isinstance(adult_t382_summary, dict) else {}
        qa_pass = 0
        qa_total = 0
        if not adult_t382_qa.empty and "passed" in adult_t382_qa.columns:
            qa_total = len(adult_t382_qa)
            qa_pass = int(adult_t382_qa["passed"].astype(str).str.lower().isin(["true", "1", "pass"]).sum())
        release_count = int(pd.to_numeric(adult_t382_product.get("released", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not adult_t382_product.empty else 0
        review_count = int(len(adult_t382_product) - release_count) if not adult_t382_product.empty else 0
        metrics = st.columns(4)
        metrics[0].metric("Coverage", fmt(overall.get("coverage"), 3))
        metrics[1].metric("Released MAE", f"{fmt(overall.get('released_mae_bpm'), 3)} BPM")
        metrics[2].metric("Unsafe/input", fmt(overall.get("published_unsafe_per_input"), 3))
        metrics[3].metric("QA gates", f"{qa_pass}/{qa_total}" if qa_total else "NA")
        if isinstance(adult_t382_summary, dict) and adult_t382_summary.get("main_insight"):
            st.caption(str(adult_t382_summary["main_insight"]))
        key_value_panel(
            [
                ("Product mode", adult_t382_summary.get("product_mode", "NA") if isinstance(adult_t382_summary, dict) else "NA"),
                ("Released / review", f"{release_count} / {review_count}" if not adult_t382_product.empty else "NA"),
                ("Supporting audit", adult_t382_summary.get("supporting_audit_decision", "NA") if isinstance(adult_t382_summary, dict) else "NA"),
                ("Claim boundary", adult_t382_summary.get("claim_boundary", "NA") if isinstance(adult_t382_summary, dict) else "NA"),
                ("Next action", adult_t382_summary.get("next_action", "NA") if isinstance(adult_t382_summary, dict) else "NA"),
            ]
        )
        st.info(
            "T382 is an experimental research MVP mode: it releases HR when candidate evidence is reliable and routes high-risk "
            "MCD source-aware repair cases to review/retest. It is not clinical monitoring or final SOTA evidence."
        )
        if not adult_t382_branch_summary.empty:
            branch_display = adult_t382_branch_summary.copy()
            for col in ["coverage", "released_mae_bpm", "published_unsafe_per_input"]:
                if col in branch_display.columns:
                    branch_display[col] = pd.to_numeric(branch_display[col], errors="coerce").round(3)
            with st.expander("T380/T382 uncertainty-aware branch summary", expanded=True):
                keep_cols = [
                    "policy_branch",
                    "n_rows",
                    "released_rows",
                    "review_rows",
                    "coverage",
                    "released_mae_bpm",
                    "published_unsafe_per_input",
                ]
                st.dataframe(branch_display[[col for col in keep_cols if col in branch_display.columns]], width="stretch", hide_index=True)
        if adult_t382_api_examples:
            labels = [str(item.get("policy_branch", f"example_{idx}")) for idx, item in enumerate(adult_t382_api_examples)]
            selected_label = st.selectbox("T382 API example", labels, key="adult_t382_api_example")
            selected_example = adult_t382_api_examples[labels.index(selected_label)]
            key_value_panel(
                [
                    ("Decision", selected_example.get("decision", "NA")),
                    ("Branch", selected_example.get("policy_branch", "NA")),
                    ("Product HR", f"{fmt(selected_example.get('product_hr_bpm'), 2)} BPM"),
                    ("Review reason", selected_example.get("review_reason", "")),
                    ("User message", selected_example.get("user_message", "")),
                ]
            )
        with st.expander("T382 product QA checks", expanded=False):
            if not adult_t382_qa.empty:
                st.dataframe(adult_t382_qa, width="stretch", hide_index=True)
    if adult_t478_summary or adult_t481_summary or adult_t482_summary:
        st.subheader("T483 Current Adult HR Research MVP Evidence")
        router = adult_t481_summary.get("router", {}) if isinstance(adult_t481_summary, dict) else {}
        core = adult_t478_summary.get("core_metrics", {}) if isinstance(adult_t478_summary, dict) else {}
        passed_gates = adult_t481_summary.get("passed_claim_gates", "NA") if isinstance(adult_t481_summary, dict) else "NA"
        n_gates = adult_t481_summary.get("n_claim_gates", "NA") if isinstance(adult_t481_summary, dict) else "NA"
        metrics = st.columns(4)
        metrics[0].metric("Router MAE", f"{fmt(router.get('balanced_mae_bpm'), 3)} BPM")
        metrics[1].metric("Router coverage", fmt(router.get("balanced_coverage"), 3))
        metrics[2].metric("Unsafe/input", fmt(router.get("balanced_unsafe_gt10_per_input"), 3))
        metrics[3].metric("Claim gates", f"{passed_gates}/{n_gates}")
        st.info(
            "Current product mode is a bounded adult HR release/review MVP. It releases HR only when the selector evidence "
            "passes the configured threshold; otherwise review/retest is the intended output. This is research evidence, "
            "not a diagnostic or clinical-monitoring claim."
        )
        key_value_panel(
            [
                ("T478 evidence decision", adult_t478_summary.get("decision", "NA") if isinstance(adult_t478_summary, dict) else "NA"),
                ("T481 router policy", router.get("router_policy", "NA") if isinstance(router, dict) else "NA"),
                ("T481 threshold", router.get("threshold", "NA") if isinstance(router, dict) else "NA"),
                ("T482 figure decision", adult_t482_summary.get("decision", "NA") if isinstance(adult_t482_summary, dict) else "NA"),
                ("T474 UBFC core MAE", f"{fmt(core.get('t474_best_mae_bpm'), 3)} BPM"),
                ("T476B best deep MAE", f"{fmt(core.get('t476b_best_deep_mae_bpm'), 3)} BPM"),
                ("T477 DLCN transfer MAE", f"{fmt(core.get('t477_transfer_mae_bpm'), 3)} BPM"),
                ("T472 DLCN-trained MAE", f"{fmt(core.get('t472_dlcn_trained_mae_bpm'), 3)} BPM"),
            ]
        )
        if adult_t485_summary or adult_t486_summary or adult_t487_summary or adult_t488_summary or adult_t489_summary:
            st.markdown("#### Current Domain Gates and GPU Evidence Lock")
            t489_gate_text = (
                f"{adult_t489_summary.get('t480_passed_claim_gates', 'NA')}/"
                f"{adult_t489_summary.get('t480_n_claim_gates', 'NA')}"
                if isinstance(adult_t489_summary, dict)
                else "NA"
            )
            domain_metrics = st.columns(5)
            domain_metrics[0].metric(
                "GPU selector",
                f"{fmt(adult_t489_summary.get('dlcn_mae_bpm'), 3)} BPM"
                if isinstance(adult_t489_summary, dict)
                else "NA",
                help="DLCN test MAE from the locked T480 CUDA run.",
            )
            domain_metrics[1].metric(
                "UBFC-Phys trials",
                str(adult_t485_summary.get("n_trials", "NA")) if isinstance(adult_t485_summary, dict) else "NA",
                help="Stress-domain BVP/video trials indexed without full extraction.",
            )
            domain_metrics[2].metric(
                "MR triplets",
                str(adult_t486_summary.get("n_ready_rgb_nir_pulseox_conditions", "NA"))
                if isinstance(adult_t486_summary, dict)
                else "NA",
                help="MR-NIRP RGB+NIR+PulseOx low-light/domain conditions.",
            )
            domain_metrics[3].metric(
                "CMU ROI gap",
                f"{fmt(adult_t487_summary.get('max_roi_mae_range_bpm'), 2)} BPM"
                if isinstance(adult_t487_summary, dict)
                else "NA",
                help="Audit-only ROI sensitivity range from CMU replay.",
            )
            domain_metrics[4].metric("T480 gates", t489_gate_text)
            st.warning(
                "Current bound: GPU selector evidence is reproducible, and stress/low-light/fairness routes are locked, "
                "but universal SOTA, clinical monitoring, and final fairness claims remain blocked until selected-condition "
                "candidate extraction and statistical tests are complete."
            )
            key_value_panel(
                [
                    ("T485 stress route", adult_t485_summary.get("decision", "NA") if isinstance(adult_t485_summary, dict) else "NA"),
                    ("T486 low-light route", adult_t486_summary.get("decision", "NA") if isinstance(adult_t486_summary, dict) else "NA"),
                    ("T487 fairness audit", adult_t487_summary.get("decision", "NA") if isinstance(adult_t487_summary, dict) else "NA"),
                    ("T488 locked subset", adult_t488_summary.get("decision", "NA") if isinstance(adult_t488_summary, dict) else "NA"),
                    ("T489 evidence lock", adult_t489_summary.get("decision", "NA") if isinstance(adult_t489_summary, dict) else "NA"),
                    ("Peak planned extraction", f"{fmt(adult_t488_summary.get('planned_peak_disk_gib'), 2)} GiB" if isinstance(adult_t488_summary, dict) else "NA"),
                    ("All-selected simultaneous cost", f"{fmt(adult_t488_summary.get('planned_all_selected_simultaneous_gib'), 2)} GiB" if isinstance(adult_t488_summary, dict) else "NA"),
                ]
            )
            with st.expander("T485-T489 domain gates", expanded=False):
                gate_frames = []
                for label, frame in [
                    ("T485", adult_t485_claim_gate),
                    ("T486", adult_t486_claim_gate),
                    ("T487", adult_t487_claim_gate),
                    ("T488", adult_t488_claim_gate),
                    ("T489", adult_t489_claim_gate),
                ]:
                    if not frame.empty:
                        gate_frames.append(frame.assign(source_task=label))
                if gate_frames:
                    st.dataframe(pd.concat(gate_frames, ignore_index=True), width="stretch", hide_index=True)
            with st.expander("T488 locked external-domain subset", expanded=False):
                if not adult_t488_subset.empty:
                    subset_display = adult_t488_subset.copy()
                    for col in [
                        "bvp_fft_peak_bpm_assuming_64hz",
                        "video_gib",
                        "compressed_size_gib",
                        "uncompressed_size_gib",
                        "planned_peak_disk_gib",
                    ]:
                        if col in subset_display.columns:
                            subset_display[col] = pd.to_numeric(subset_display[col], errors="coerce").round(3)
                    st.dataframe(subset_display, width="stretch", hide_index=True)
                if not adult_t488_budget.empty:
                    st.dataframe(adult_t488_budget, width="stretch", hide_index=True)
            with st.expander("T487 CMU fairness audit metrics", expanded=False):
                if not adult_t487_metrics.empty:
                    metrics_display = adult_t487_metrics.copy()
                    for col in ["mae_bpm", "median_abs_error_bpm", "unsafe_gt10_rate", "mean_confidence"]:
                        if col in metrics_display.columns:
                            metrics_display[col] = pd.to_numeric(metrics_display[col], errors="coerce").round(3)
                    st.dataframe(metrics_display, width="stretch", hide_index=True)
                if not adult_t487_delta.empty:
                    st.dataframe(adult_t487_delta, width="stretch", hide_index=True)
        if (
            adult_t491_summary
            or adult_t492_summary
            or adult_t493_summary
            or adult_t494_summary
            or adult_t495_summary
            or adult_t497_summary
            or adult_t498_summary
            or adult_t504_summary
            or adult_t506_summary
            or adult_t517_summary
            or adult_t518_summary
        ):
            st.markdown("#### Selected-Domain ROI Iteration and Product Safety")
            roi_metrics = st.columns(6)
            roi_metrics[0].metric(
                "Full-frame naive MAE",
                f"{fmt(adult_t492_summary.get('naive_modality_mae_bpm'), 2)} BPM"
                if isinstance(adult_t492_summary, dict)
                else "NA",
                help="T492 MR-NIRP full-frame RGB/NIR spectral-candidate MAE before artifact-aware refusal.",
            )
            roi_metrics[1].metric(
                "T494 ROI MAE",
                f"{fmt(adult_t494_summary.get('ubfc_mae_released_bpm'), 2)} BPM"
                if isinstance(adult_t494_summary, dict)
                else "NA",
                help="Initial ROI release policy on selected UBFC-Phys conditions.",
            )
            roi_metrics[2].metric(
                "T497 all15 MAE",
                f"{fmt(adult_t497_summary.get('mae_released_bpm'), 2)} BPM"
                if isinstance(adult_t497_summary, dict)
                else "NA",
                help="Expanded UBFC-Phys S1-S5 all15 method-aware selector MAE before the context guard.",
            )
            roi_metrics[3].metric(
                "T498 guarded MAE",
                f"{fmt(adult_t498_summary.get('mae_released_bpm'), 2)} BPM"
                if isinstance(adult_t498_summary, dict)
                else "NA",
                help="Context-aware T3 guard MAE on released outputs.",
            )
            roi_metrics[4].metric(
                "T497 unsafe",
                fmt(adult_t497_summary.get("unsafe_release_rate"), 3)
                if isinstance(adult_t497_summary, dict)
                else "NA",
                help="Unsafe released-output rate exposed by expanded UBFC-Phys S1-S5 all15 replay.",
            )
            roi_metrics[5].metric(
                "T498 unsafe",
                fmt(adult_t498_summary.get("unsafe_release_rate"), 3)
                if isinstance(adult_t498_summary, dict)
                else "NA",
                help="Unsafe released-output rate after applying the context-aware T3 conflict guard.",
            )
            route_metrics = st.columns(4)
            route_metrics[0].metric(
                "T504 route MAE",
                f"{fmt(adult_t504_summary.get('mae_released_bpm'), 2)} BPM"
                if isinstance(adult_t504_summary, dict)
                else "NA",
                help="MediaPipe Face Mesh dual-range T3 selector MAE on the selected route.",
            )
            route_metrics[1].metric(
                "T504 unsafe",
                fmt(adult_t504_summary.get("unsafe_release_rate"), 3)
                if isinstance(adult_t504_summary, dict)
                else "NA",
                help="Unsafe released-output rate after the MediaPipe dual-range T3 selector.",
            )
            route_metrics[2].metric(
                "T506 coverage",
                fmt(adult_t506_summary.get("coverage"), 3)
                if isinstance(adult_t506_summary, dict)
                else "NA",
                help="UBFC all15 product coverage after route-aware T498 default plus T504 rescue.",
            )
            route_metrics[3].metric(
                "T506 MAE",
                f"{fmt(adult_t506_summary.get('released_mae_bpm'), 2)} BPM"
                if isinstance(adult_t506_summary, dict)
                else "NA",
                help="UBFC all15 released MAE after route-aware product policy.",
            )
            learned_metrics = st.columns(4)
            learned_metrics[0].metric(
                "T508 DLCN MAE",
                f"{fmt((adult_t508_summary.get('primary_test_metrics', {}).get('DLCN', {}) if isinstance(adult_t508_summary, dict) else {}).get('mae_bpm'), 2)} BPM"
                if isinstance(adult_t508_summary, dict)
                else "NA",
                help="Route-aware/physiology-feature GPU selector released MAE on DLCN test windows.",
            )
            learned_metrics[1].metric(
                "T508 UBFC MAE",
                f"{fmt((adult_t508_summary.get('primary_test_metrics', {}).get('UBFC-rPPG', {}) if isinstance(adult_t508_summary, dict) else {}).get('mae_bpm'), 2)} BPM"
                if isinstance(adult_t508_summary, dict)
                else "NA",
                help="Route-aware/physiology-feature GPU selector released MAE on UBFC-rPPG test windows.",
            )
            learned_metrics[2].metric(
                "T510 threshold",
                fmt(adult_t510_summary.get("selected_threshold"), 3) if isinstance(adult_t510_summary, dict) else "NA",
                help="Validation-selected threshold for coverage recovery.",
            )
            learned_metrics[3].metric(
                "T511 coverage",
                fmt(adult_t511_summary.get("coverage"), 3) if isinstance(adult_t511_summary, dict) else "NA",
                help="Coverage of the optional experimental learned-selector product mode.",
            )
            moe_metrics = st.columns(4)
            moe_metrics[0].metric(
                "T517 routes",
                str(adult_t517_summary.get("route_count", "NA")) if isinstance(adult_t517_summary, dict) else "NA",
                help="Number of route-aware MoE routes synthesized from the current evidence base.",
            )
            moe_metrics[1].metric(
                "T518 release routes",
                str(adult_t518_summary.get("release_or_review_routes", "NA")) if isinstance(adult_t518_summary, dict) else "NA",
                help="Routes allowed to release HR when their route-specific evidence gates pass.",
            )
            moe_metrics[2].metric(
                "T518 review-only",
                str(adult_t518_summary.get("review_only_routes", "NA")) if isinstance(adult_t518_summary, dict) else "NA",
                help="Routes intentionally restricted to review/audit until their evidence gap is closed.",
            )
            moe_metrics[3].metric(
                "T518 claim gates",
                f"{adult_t518_summary.get('passed_claim_gates', 'NA')}/{adult_t518_summary.get('n_claim_gates', 'NA')}"
                if isinstance(adult_t518_summary, dict)
                else "NA",
                help="Product/API claim-control gates passed by the route-aware MoE policy.",
            )
            st.info(
                "Iteration chain: T492 quantified full-frame artifact failure; T493 generated ROI traces; "
                "T494 showed naive ROI selection was still unsafe; T495 introduced method-aware CHROM/POS and ROI-quality "
                "selection; T497 expanded UBFC-Phys replay exposed stress-trial low-frequency failure; T498 added a "
                "context-aware T3 conflict guard that refuses unsupported low-frequency clusters or rescues central "
                "POS/CHROM high-rate evidence. T504 then tests MediaPipe Face Mesh dual-range T3 selection, T505 rejects "
                "global promotion of that rule, and T506 integrates the safe path as route-aware product rescue. "
                "T508-T511 add a bounded learned-selector extension: route/physiology features improve candidate selection, "
                "T509 locks statistical endpoints, T510 recovers coverage with validation-only thresholding, and T511 exposes "
                "the result as an optional experimental product mode while preserving T506 as the default. T517-T518 then "
                "convert the evidence into a route-aware MoE product/API contract: validated routes can release/review, while "
                "fairness and multimodal routes remain review-only until their claim gates are closed."
            )
            key_value_panel(
                [
                    ("T491 compact trace cache", adult_t491_summary.get("decision", "NA") if isinstance(adult_t491_summary, dict) else "NA"),
                    ("T492 artifact gate", adult_t492_summary.get("decision", "NA") if isinstance(adult_t492_summary, dict) else "NA"),
                    ("T493 ROI trace cache", adult_t493_summary.get("decision", "NA") if isinstance(adult_t493_summary, dict) else "NA"),
                    ("T494 naive ROI selector", adult_t494_summary.get("decision", "NA") if isinstance(adult_t494_summary, dict) else "NA"),
                    ("T495 method-aware selector", adult_t495_summary.get("decision", "NA") if isinstance(adult_t495_summary, dict) else "NA"),
                    ("T497 expanded replay", adult_t497_summary.get("decision", "NA") if isinstance(adult_t497_summary, dict) else "NA"),
                    ("T498 context-aware guard", adult_t498_summary.get("decision", "NA") if isinstance(adult_t498_summary, dict) else "NA"),
                    ("T504 MediaPipe dual-range", adult_t504_summary.get("decision", "NA") if isinstance(adult_t504_summary, dict) else "NA"),
                    ("T505 generalization audit", adult_t505_summary.get("decision", "NA") if isinstance(adult_t505_summary, dict) else "NA"),
                    ("T506 route-aware product", adult_t506_summary.get("decision", "NA") if isinstance(adult_t506_summary, dict) else "NA"),
                    ("T508 route-aware GPU selector", adult_t508_summary.get("decision", "NA") if isinstance(adult_t508_summary, dict) else "NA"),
                    ("T509 locked protocol", adult_t509_summary.get("decision", "NA") if isinstance(adult_t509_summary, dict) else "NA"),
                    ("T510 threshold recovery", adult_t510_summary.get("decision", "NA") if isinstance(adult_t510_summary, dict) else "NA"),
                    ("T511 experimental mode", adult_t511_summary.get("decision", "NA") if isinstance(adult_t511_summary, dict) else "NA"),
                    ("T517 route-aware MoE", adult_t517_summary.get("decision", "NA") if isinstance(adult_t517_summary, dict) else "NA"),
                    ("T518 product/API policy", adult_t518_summary.get("decision", "NA") if isinstance(adult_t518_summary, dict) else "NA"),
                    ("T498 release rate", fmt(adult_t498_summary.get("release_rate"), 3) if isinstance(adult_t498_summary, dict) else "NA"),
                    ("T506 coverage delta", fmt(adult_t506_summary.get("coverage_delta_vs_t498"), 3) if isinstance(adult_t506_summary, dict) else "NA"),
                    ("T511 default policy", adult_t511_summary.get("default_policy_preserved", "NA") if isinstance(adult_t511_summary, dict) else "NA"),
                    ("Current claim boundary", adult_t518_summary.get("claim_boundary", "NA") if isinstance(adult_t518_summary, dict) else adult_t506_summary.get("claim_boundary", "NA") if isinstance(adult_t506_summary, dict) else adult_t498_summary.get("claim_boundary", "NA") if isinstance(adult_t498_summary, dict) else "NA"),
                ]
            )
            with st.expander("T495 method-aware product decisions", expanded=True):
                if not adult_t495_decisions.empty:
                    display = adult_t495_decisions.copy()
                    for col in ["reference_bpm", "released_bpm", "absolute_error_bpm"]:
                        if col in display.columns:
                            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                    keep_cols = [
                        "dataset",
                        "condition_id",
                        "policy",
                        "reference_bpm",
                        "released_bpm",
                        "absolute_error_bpm",
                        "unsafe_release_gt10",
                        "reason",
                        "cluster_methods",
                        "cluster_rois",
                    ]
                    st.dataframe(display[[col for col in keep_cols if col in display.columns]], width="stretch", hide_index=True)
            with st.expander("T494 to T495 metric iteration", expanded=False):
                if not adult_t495_metrics.empty:
                    metrics_display = adult_t495_metrics.copy()
                    for col in ["release_rate", "mae_released_bpm", "unsafe_release_rate", "oracle_best_mae_bpm", "mean_artifact_candidate_rate"]:
                        if col in metrics_display.columns:
                            metrics_display[col] = pd.to_numeric(metrics_display[col], errors="coerce").round(3)
                    st.dataframe(metrics_display, width="stretch", hide_index=True)
                if not adult_t495_delta.empty:
                    delta_display = adult_t495_delta.copy()
                    for col in ["released_bpm_t494", "absolute_error_bpm_t494", "released_bpm_t495", "absolute_error_bpm_t495", "error_delta_t495_minus_t494"]:
                        if col in delta_display.columns:
                            delta_display[col] = pd.to_numeric(delta_display[col], errors="coerce").round(3)
                    st.dataframe(delta_display, width="stretch", hide_index=True)
            with st.expander("T497 to T498 expanded-stress iteration", expanded=True):
                if not adult_t498_metrics.empty:
                    metrics_display = adult_t498_metrics.copy()
                    for col in ["release_rate", "mae_released_bpm", "median_ae_released_bpm", "unsafe_release_rate", "oracle_best_mae_bpm"]:
                        if col in metrics_display.columns:
                            metrics_display[col] = pd.to_numeric(metrics_display[col], errors="coerce").round(3)
                    st.dataframe(metrics_display, width="stretch", hide_index=True)
                if not adult_t498_delta.empty:
                    delta_display = adult_t498_delta.copy()
                    for col in ["released_bpm_t497", "absolute_error_bpm_t497", "released_bpm_t498", "absolute_error_bpm_t498", "error_delta_t498_minus_t497"]:
                        if col in delta_display.columns:
                            delta_display[col] = pd.to_numeric(delta_display[col], errors="coerce").round(3)
                    st.dataframe(delta_display, width="stretch", hide_index=True)
            with st.expander("T498 context-aware product decisions", expanded=False):
                if not adult_t498_decisions.empty:
                    display = adult_t498_decisions.copy()
                    for col in ["reference_bpm", "released_bpm", "absolute_error_bpm"]:
                        if col in display.columns:
                            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                    keep_cols = [
                        "dataset",
                        "condition_id",
                        "trial",
                        "policy",
                        "reference_bpm",
                        "released_bpm",
                        "absolute_error_bpm",
                        "unsafe_release_gt10",
                        "reason",
                        "cluster_methods",
                        "cluster_rois",
                    ]
                    st.dataframe(display[[col for col in keep_cols if col in display.columns]], width="stretch", hide_index=True)
            with st.expander("T504-T506 route-aware product policy", expanded=True):
                if not adult_t506_delta.empty:
                    delta_display = adult_t506_delta.copy()
                    for col in ["t498_coverage", "t506_coverage", "coverage_delta", "t498_released_mae_bpm", "t506_released_mae_bpm", "mae_delta_bpm", "t506_unsafe_release_rate"]:
                        if col in delta_display.columns:
                            delta_display[col] = pd.to_numeric(delta_display[col], errors="coerce").round(3)
                    st.dataframe(delta_display, width="stretch", hide_index=True)
                if not adult_t506_product_summary.empty:
                    summary_display = adult_t506_product_summary.copy()
                    for col in ["coverage", "released_mae_bpm", "released_median_ae_bpm", "released_unsafe_gt10_rate", "unsafe_per_input"]:
                        if col in summary_display.columns:
                            summary_display[col] = pd.to_numeric(summary_display[col], errors="coerce").round(3)
                    st.dataframe(summary_display, width="stretch", hide_index=True)
                if not adult_t506_product.empty:
                    display = adult_t506_product.copy()
                    for col in ["product_hr_bpm", "reference_bpm_for_eval_only", "absolute_error_bpm_for_eval_only"]:
                        if col in display.columns:
                            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                    keep_cols = [
                        "condition_id",
                        "dataset",
                        "policy_branch",
                        "source_task",
                        "decision",
                        "product_hr_bpm",
                        "absolute_error_bpm_for_eval_only",
                        "review_reason",
                    ]
                    st.dataframe(display[[col for col in keep_cols if col in display.columns]], width="stretch", hide_index=True)
                if isinstance(adult_t506_api_examples, dict) and adult_t506_api_examples.get("examples"):
                    st.json(adult_t506_api_examples, expanded=False)
            with st.expander("T508-T511 optional learned-selector product mode", expanded=True):
                st.caption(
                    "This mode is experimental. T506 remains the default policy; T511 exposes T510 as an optional release/review route for research MVP evaluation."
                )
                if not adult_t508_delta.empty:
                    st.markdown("**T508 vs T480 GPU selector delta**")
                    st.dataframe(adult_t508_delta, width="stretch", hide_index=True)
                if not adult_t509_bootstrap.empty:
                    st.markdown("**T509 bootstrap endpoints**")
                    st.dataframe(adult_t509_bootstrap, width="stretch", hide_index=True)
                if not adult_t510_delta.empty:
                    st.markdown("**T510 validation-selected threshold delta**")
                    st.dataframe(adult_t510_delta, width="stretch", hide_index=True)
                if not adult_t511_product_summary.empty:
                    st.markdown("**T511 product summary**")
                    summary_display = adult_t511_product_summary.copy()
                    for col in ["coverage", "released_mae_bpm", "released_median_ae_bpm", "released_unsafe_gt10_rate", "unsafe_per_input"]:
                        if col in summary_display.columns:
                            summary_display[col] = pd.to_numeric(summary_display[col], errors="coerce").round(3)
                    st.dataframe(summary_display, width="stretch", hide_index=True)
                if not adult_t511_product.empty:
                    st.markdown("**T511 release/review examples**")
                    display = adult_t511_product.copy()
                    for col in ["product_hr_bpm", "absolute_error_bpm_for_eval_only", "selector_prob", "threshold"]:
                        if col in display.columns:
                            display[col] = pd.to_numeric(display[col], errors="coerce").round(3)
                    keep_cols = [
                        "dataset",
                        "candidate_window_id",
                        "policy_branch",
                        "decision",
                        "product_hr_bpm",
                        "selector_prob",
                        "threshold",
                        "absolute_error_bpm_for_eval_only",
                        "review_reason",
                    ]
                    st.dataframe(display[[col for col in keep_cols if col in display.columns]].head(80), width="stretch", hide_index=True)
                if isinstance(adult_t511_api_examples, dict) and adult_t511_api_examples.get("examples"):
                    st.json(adult_t511_api_examples, expanded=False)
                gate_frames = []
                for label, frame in [
                    ("T508", adult_t508_claim_gate),
                    ("T509", adult_t509_endpoint_gates),
                    ("T510", adult_t510_claim_gate),
                    ("T511", adult_t511_claim_gate),
                ]:
                    if not frame.empty:
                        gate_frames.append(frame.assign(source_task=label))
                if gate_frames:
                    st.markdown("**T508-T511 claim gates**")
                    st.dataframe(pd.concat(gate_frames, ignore_index=True), width="stretch", hide_index=True)
            with st.expander("T517-T518 route-aware MoE product/API policy", expanded=True):
                st.caption(
                    "Current product strategy: route-aware mixture-of-experts. Stable RGB and MCD high-HR routes can release/review when their own gates pass; fairness and multimodal routes stay review-only."
                )
                if not adult_t517_routes.empty:
                    st.markdown("**T517 evidence routes**")
                    route_display = adult_t517_routes.copy()
                    for col in ["coverage", "mae_bpm", "unsafe_per_input"]:
                        if col in route_display.columns:
                            route_display[col] = pd.to_numeric(route_display[col], errors="coerce").round(3)
                    keep_cols = [
                        "route_id",
                        "domain_or_dataset",
                        "selected_policy",
                        "product_action",
                        "coverage",
                        "mae_bpm",
                        "unsafe_per_input",
                        "paper_role",
                        "claim_allowed",
                    ]
                    st.dataframe(route_display[[col for col in keep_cols if col in route_display.columns]], width="stretch", hide_index=True)
                if not adult_t518_policy.empty:
                    st.markdown("**T518 product/API policy contract**")
                    policy_display = adult_t518_policy.copy()
                    for col in ["coverage", "mae_bpm", "unsafe_per_input"]:
                        if col in policy_display.columns:
                            policy_display[col] = pd.to_numeric(policy_display[col], errors="coerce").round(3)
                    keep_cols = [
                        "route_id",
                        "route_detector",
                        "output_action_class",
                        "selected_policy",
                        "release_rule",
                        "review_rule",
                        "input_required",
                        "output_release_fields",
                        "output_review_fields",
                        "product_warning",
                    ]
                    st.dataframe(policy_display[[col for col in keep_cols if col in policy_display.columns]], width="stretch", hide_index=True)
                if isinstance(adult_t518_api_examples, dict) and adult_t518_api_examples.get("examples"):
                    st.markdown("**T518 API examples**")
                    st.json(adult_t518_api_examples, expanded=False)
                moe_gate_frames = []
                for label, frame in [
                    ("T517", adult_t517_claim_gate),
                    ("T518", adult_t518_claim_gate),
                ]:
                    if not frame.empty:
                        moe_gate_frames.append(frame.assign(source_task=label))
                if moe_gate_frames:
                    st.markdown("**T517-T518 claim gates**")
                    st.dataframe(pd.concat(moe_gate_frames, ignore_index=True), width="stretch", hide_index=True)
            with st.expander("T527 product end-to-end demo QA", expanded=True):
                st.caption(
                    "Verified replay demo for the route-aware MoE product: release HR only on validated routes, otherwise return review packets with quality flags and non-clinical warnings."
                )
                if isinstance(adult_t527_summary, dict) and adult_t527_summary:
                    qa_passed = adult_t527_summary.get("passed_checks", "NA")
                    qa_total = adult_t527_summary.get("n_checks", "NA")
                    demo_cases = adult_t527_summary.get("n_demo_cases", "NA")
                    release_cases = adult_t527_summary.get("release_cases", "NA")
                    review_cases = adult_t527_summary.get("review_cases", "NA")
                    t527_cols = st.columns(4)
                    t527_cols[0].metric("T527 QA", f"{qa_passed}/{qa_total}")
                    t527_cols[1].metric("Demo cases", str(demo_cases))
                    t527_cols[2].metric("Release", str(release_cases))
                    t527_cols[3].metric("Review", str(review_cases))
                    st.caption(str(adult_t527_summary.get("claim_boundary", "")))
                if not adult_t527_report_cards.empty:
                    st.markdown("**T527 user-facing report cards**")
                    report_display = adult_t527_report_cards.copy()
                    for col in ["display_hr_bpm", "eval_abs_error_bpm"]:
                        if col in report_display.columns:
                            report_display[col] = pd.to_numeric(report_display[col], errors="coerce").round(3)
                    st.dataframe(report_display, width="stretch", hide_index=True)
                if not adult_t527_cases.empty:
                    st.markdown("**T527 demo case routing trace**")
                    case_display = adult_t527_cases.copy()
                    for col in ["hr_bpm", "reference_hr_for_eval_only", "abs_error_bpm_for_eval_only", "confidence_proxy_for_demo"]:
                        if col in case_display.columns:
                            case_display[col] = pd.to_numeric(case_display[col], errors="coerce").round(3)
                    keep_cols = [
                        "case_id",
                        "dataset",
                        "sample_id",
                        "route_id",
                        "selected_policy",
                        "decision",
                        "hr_bpm",
                        "review_reason",
                        "quality_flags",
                    ]
                    st.dataframe(case_display[[col for col in keep_cols if col in case_display.columns]], width="stretch", hide_index=True)
                if isinstance(adult_t527_api_packets, dict) and adult_t527_api_packets.get("packets"):
                    st.markdown("**T527 API packets**")
                    st.json(adult_t527_api_packets, expanded=False)
                if not adult_t527_qa_checks.empty:
                    st.markdown("**T527 QA checks**")
                    st.dataframe(adult_t527_qa_checks, width="stretch", hide_index=True)
            with st.expander("T504 MediaPipe failure taxonomy and T505 audit", expanded=False):
                if not adult_t504_taxonomy.empty:
                    taxonomy_display = adult_t504_taxonomy.copy()
                    for col in ["reference_bpm", "t503_released_bpm", "t503_error_bpm", "t504_released_bpm", "t504_error_bpm", "low_cluster_bpm", "high_cluster_bpm"]:
                        if col in taxonomy_display.columns:
                            taxonomy_display[col] = pd.to_numeric(taxonomy_display[col], errors="coerce").round(3)
                    st.dataframe(taxonomy_display, width="stretch", hide_index=True)
                if not adult_t504_delta.empty:
                    st.dataframe(adult_t504_delta, width="stretch", hide_index=True)
                if not adult_t505_delta.empty:
                    st.dataframe(adult_t505_delta, width="stretch", hide_index=True)
            with st.expander("T491-T506 selected-domain claim gates", expanded=False):
                gate_frames = []
                for label, frame in [
                    ("T491", adult_t491_claim_gate),
                    ("T492", adult_t492_claim_gate),
                    ("T493", adult_t493_claim_gate),
                    ("T494", adult_t494_claim_gate),
                    ("T495", adult_t495_claim_gate),
                    ("T497", adult_t497_claim_gate),
                    ("T498", adult_t498_claim_gate),
                    ("T504", adult_t504_claim_gate),
                    ("T505", adult_t505_claim_gate),
                    ("T506", adult_t506_claim_gate),
                ]:
                    if not frame.empty:
                        gate_frames.append(frame.assign(source_task=label))
                if gate_frames:
                    st.dataframe(pd.concat(gate_frames, ignore_index=True), width="stretch", hide_index=True)
            with st.expander("T491/T493 trace caches", expanded=False):
                if not adult_t491_trace_index.empty:
                    st.dataframe(adult_t491_trace_index, width="stretch", hide_index=True)
                if not adult_t493_trace_index.empty:
                    st.dataframe(adult_t493_trace_index, width="stretch", hide_index=True)
        if ADULT_T482_FIGURE_PNG.exists():
            st.image(
                str(ADULT_T482_FIGURE_PNG),
                caption="T482 external/fairness/router evidence figure",
                width="stretch",
            )
        if isinstance(adult_t478_summary, dict) and adult_t478_summary.get("main_insight"):
            st.caption(str(adult_t478_summary["main_insight"]))
        if isinstance(adult_t481_summary, dict) and adult_t481_summary.get("main_insight"):
            st.caption(str(adult_t481_summary["main_insight"]))
        with st.expander("T481 release/review router table", expanded=True):
            if not adult_t481_router.empty:
                router_display = adult_t481_router.copy()
                for col in ["threshold", "balanced_mae_bpm", "balanced_coverage", "balanced_unsafe_gt10_per_input", "dataset_mae_bpm", "dataset_coverage", "dataset_unsafe_gt10_per_input"]:
                    if col in router_display.columns:
                        router_display[col] = pd.to_numeric(router_display[col], errors="coerce").round(3)
                st.dataframe(router_display, width="stretch", hide_index=True)
        with st.expander("T478/T481 claim gates", expanded=False):
            gate_frames = []
            if not adult_t478_claim_gate.empty:
                gate_frames.append(adult_t478_claim_gate.assign(source_task="T478"))
            if not adult_t481_claim_gate.empty:
                gate_frames.append(adult_t481_claim_gate.assign(source_task="T481"))
            if gate_frames:
                st.dataframe(pd.concat(gate_frames, ignore_index=True), width="stretch", hide_index=True)
        if adult_t481_api_examples:
            labels = [
                f"{item.get('decision', 'example')} / {item.get('sample_id', idx)}"
                for idx, item in enumerate(adult_t481_api_examples)
            ]
            selected_label = st.selectbox("T481 API example", labels, key="adult_t481_api_example")
            selected_example = adult_t481_api_examples[labels.index(selected_label)]
            key_value_panel(
                [
                    ("Decision", selected_example.get("decision", "NA")),
                    ("Product HR", f"{fmt(selected_example.get('product_hr_bpm'), 2)} BPM"),
                    ("Review reason", selected_example.get("review_reason", "")),
                    ("Router policy", selected_example.get("router_policy", "NA")),
                ]
            )
            st.json(selected_example, expanded=False)
        with st.expander("T482 figure QA and source data", expanded=False):
            if not adult_t482_qa.empty:
                st.dataframe(adult_t482_qa, width="stretch", hide_index=True)
            if not adult_t482_source.empty:
                st.dataframe(adult_t482_source.head(40), width="stretch", hide_index=True)
    st.subheader("Refusal Policy")
    focus_policies = [
        "fixed_optical_flow_y_body_aware_half_validated",
        REFUSAL_POLICY,
        "naive_confidence_select",
    ]
    quality_focus = quality[quality["policy"].isin(focus_policies)].copy()
    if not quality_focus.empty:
        quality_focus["coverage_pct"] = 100.0 * pd.to_numeric(quality_focus["coverage"], errors="coerce")
        quality_focus_display = quality_focus[
            [
                "dataset",
                "policy",
                "accepted_windows",
                "refused_windows",
                "coverage_pct",
                "mae_bpm",
                "rmse_bpm",
            ]
        ].copy()
        for col in ["coverage_pct", "mae_bpm", "rmse_bpm"]:
            quality_focus_display[col] = pd.to_numeric(quality_focus_display[col], errors="coerce").round(3)
        quality_focus_display = quality_focus_display.rename(
            columns={
                "dataset": "Dataset",
                "policy": "Policy",
                "accepted_windows": "Accepted",
                "refused_windows": "Refused",
                "coverage_pct": "Coverage %",
                "mae_bpm": "MAE BPM",
                "rmse_bpm": "RMSE BPM",
            }
        )
        st.table(quality_focus_display)
        with st.expander("Selected method distribution"):
            st.dataframe(
                quality_focus[
                    [
                        "dataset",
                        "policy",
                        "top_selected_methods",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
    if not t57.empty:
        st.subheader("T57 Calibrated Policy")
        policy_order = [
            "fixed_all",
            "raw_kept_rule",
            "confidence_min_0.050",
            "raw_confidence_min_0.120",
        ]
        t57_focus = t57[
            (t57["dataset"].eq("AIR-all"))
            & (t57["split"].eq("test"))
            & (t57["policy_id"].isin(policy_order))
        ].copy()
        if not t57_focus.empty:
            t57_focus["policy_id"] = pd.Categorical(t57_focus["policy_id"], policy_order, ordered=True)
            t57_focus = t57_focus.sort_values("policy_id")
            t57_focus["coverage_pct"] = 100.0 * pd.to_numeric(t57_focus["coverage"], errors="coerce")
            t57_display = t57_focus[
                [
                    "policy_id",
                    "selected_scope",
                    "accepted_windows",
                    "refused_windows",
                    "coverage_pct",
                    "mae_bpm",
                    "rmse_bpm",
                    "pearson",
                ]
            ].copy()
            for col in ["coverage_pct", "mae_bpm", "rmse_bpm", "pearson"]:
                t57_display[col] = pd.to_numeric(t57_display[col], errors="coerce").round(3)
            t57_display = t57_display.rename(
                columns={
                    "policy_id": "Policy",
                    "selected_scope": "Scope",
                    "accepted_windows": "Accepted",
                    "refused_windows": "Refused",
                    "coverage_pct": "Coverage %",
                    "mae_bpm": "MAE BPM",
                    "rmse_bpm": "RMSE BPM",
                    "pearson": "Pearson",
                }
            )
            t57_display = t57_display.reset_index(drop=True)
            st.table(t57_display)
            st.caption(
                "T57 selects confidence_min_0.050 on validation subjects. "
                "raw_confidence_min_0.120 is shown as a future candidate, not the main claim."
            )
    t58 = load_cambridge_t58_policy_summary()
    t58_ci = load_cambridge_t58_stability_ci()
    if not t58.empty:
        st.subheader("T58 Cambridge Neonatal Prototype")
        policies = [
            T58_LEGACY_POLICY,
            "aligned_welch_depth_right_chest",
            "aligned_periodogram_depth_mean",
            "aligned_harmonic_depth_mean",
            T58_BEST_POLICY,
            "aligned_harmonic_depth_conf_weighted_smooth3",
        ]
        t58_focus = t58[t58["policy"].isin(policies)].copy()
        if not t58_focus.empty:
            t58_focus["policy"] = pd.Categorical(t58_focus["policy"], policies, ordered=True)
            t58_focus = t58_focus.sort_values("policy")
            t58_display = t58_focus[
                ["policy", "n_subjects", "n_windows", "coverage", "mae_bpm", "rmse_bpm", "pearson", "mean_pred_rr_bpm"]
            ].copy()
            for col in ["coverage", "mae_bpm", "rmse_bpm", "pearson", "mean_pred_rr_bpm"]:
                t58_display[col] = pd.to_numeric(t58_display[col], errors="coerce").round(3)
            t58_display = t58_display.rename(
                columns={
                    "policy": "Policy",
                    "n_subjects": "Subjects",
                    "n_windows": "Windows",
                    "coverage": "Coverage",
                    "mae_bpm": "MAE BPM",
                    "rmse_bpm": "RMSE BPM",
                    "pearson": "Pearson",
                    "mean_pred_rr_bpm": "Mean pred RR",
                }
            )
            st.table(t58_display)
        if not t58_ci.empty:
            ci_focus = t58_ci[
                (t58_ci["policy"].isin([T58_LEGACY_POLICY, T58_BEST_POLICY, "aligned_harmonic_depth_conf_weighted_smooth3"]))
                & (t58_ci["metric"].isin(["mae_bpm", "delta_mae_vs_legacy"]))
            ].copy()
            if not ci_focus.empty:
                for col in ["median", "ci_low", "ci_high"]:
                    ci_focus[col] = pd.to_numeric(ci_focus[col], errors="coerce").round(3)
                st.dataframe(
                    ci_focus[["policy", "metric", "median", "ci_low", "ci_high"]],
                    width="stretch",
                    hide_index=True,
                )
        st.caption(
            "T69 supports the harmonic-aware Cambridge family under subject bootstrap; exact fusion variants remain research policies."
        )
    t87 = load_cambridge_t87_policy_summary()
    t87_ci = load_cambridge_t87_bootstrap_ci()
    if not t87.empty:
        st.subheader("T87 Cambridge High-RR Calibration")
        policies = [T87_BASE_POLICY, T87_DEFAULT_POLICY, T87_Q90_POLICY, T87_MAX_POLICY]
        t87_focus = t87[t87["policy"].isin(policies)].copy()
        if not t87_focus.empty:
            t87_focus["policy"] = pd.Categorical(t87_focus["policy"], policies, ordered=True)
            t87_focus = t87_focus.sort_values("policy")
            t87_display = t87_focus[
                [
                    "policy",
                    "n_subjects",
                    "n_windows",
                    "mae_bpm",
                    "high_rr_mae_bpm",
                    "signed_error_mean_bpm",
                    "high_rr_signed_error_mean_bpm",
                ]
            ].copy()
            for col in ["mae_bpm", "high_rr_mae_bpm", "signed_error_mean_bpm", "high_rr_signed_error_mean_bpm"]:
                t87_display[col] = pd.to_numeric(t87_display[col], errors="coerce").round(3)
            t87_display["policy"] = t87_display["policy"].map(compact_method_name)
            t87_display = t87_display.rename(
                columns={
                    "policy": "Policy",
                    "n_subjects": "Subjects",
                    "n_windows": "Windows",
                    "mae_bpm": "MAE BPM",
                    "high_rr_mae_bpm": "High-RR MAE BPM",
                    "signed_error_mean_bpm": "Signed error BPM",
                    "high_rr_signed_error_mean_bpm": "High-RR signed error BPM",
                }
            )
            st.table(t87_display)
        if not t87_ci.empty:
            ci_focus = t87_ci[
                (t87_ci["policy"].isin([T87_DEFAULT_POLICY, T87_Q90_POLICY, T87_MAX_POLICY]))
                & (t87_ci["metric"].isin(["delta_mae_vs_t58_base_bpm", "delta_high_rr_mae_vs_t58_base_bpm"]))
            ].copy()
            if not ci_focus.empty:
                for col in ["median", "ci_low_2p5", "ci_high_97p5"]:
                    ci_focus[col] = pd.to_numeric(ci_focus[col], errors="coerce").round(3)
                ci_focus["policy"] = ci_focus["policy"].map(compact_method_name)
                ci_focus["metric"] = ci_focus["metric"].replace(
                    {
                        "delta_mae_vs_t58_base_bpm": "Delta MAE vs T58",
                        "delta_high_rr_mae_vs_t58_base_bpm": "Delta high-RR MAE",
                    }
                )
                st.dataframe(
                    ci_focus[["policy", "metric", "median", "ci_low_2p5", "ci_high_97p5"]],
                    width="stretch",
                    hide_index=True,
                )
        st.caption(
            "T87 supports high-RR underprediction mitigation. All-window bootstrap intervals still cross zero, "
            "so the dashboard presents these as research policies rather than clinical defaults."
        )
    t92 = load_cambridge_t92_policy_summary()
    if not t92.empty:
        st.subheader("T92 Actionability Layer")
        policies = [T87_BASE_POLICY, T87_DEFAULT_POLICY, T87_Q90_POLICY, T87_MAX_POLICY]
        rules = [T92_BASELINE_RULE, T92_DEFAULT_RULE, "severe_or_persistent"]
        t92_focus = t92[(t92["policy"].isin(policies)) & (t92["rule_id"].isin(rules))].copy()
        if not t92_focus.empty:
            t92_focus["policy"] = pd.Categorical(t92_focus["policy"], policies, ordered=True)
            t92_focus["rule_id"] = pd.Categorical(t92_focus["rule_id"], rules, ordered=True)
            t92_focus = t92_focus.sort_values(["policy", "rule_id"])
            t92_display = t92_focus[
                [
                    "policy",
                    "rule_id",
                    "event_sensitivity",
                    "missed_high_rr_events",
                    "false_alert_episodes",
                    "alarm_burden_per_hour",
                    "mean_time_to_detect_sec",
                    "window_precision",
                ]
            ].copy()
            for col in ["event_sensitivity", "alarm_burden_per_hour", "mean_time_to_detect_sec", "window_precision"]:
                t92_display[col] = pd.to_numeric(t92_display[col], errors="coerce").round(3)
            for col in ["missed_high_rr_events", "false_alert_episodes"]:
                t92_display[col] = pd.to_numeric(t92_display[col], errors="coerce").astype("Int64")
            t92_display["policy"] = t92_display["policy"].map(compact_method_name)
            t92_display["rule_id"] = t92_display["rule_id"].map(t92_rule_display_name)
            t92_display = t92_display.rename(
                columns={
                    "policy": "Policy",
                    "rule_id": "Rule",
                    "event_sensitivity": "Event sensitivity",
                    "missed_high_rr_events": "Missed events",
                    "false_alert_episodes": "False alerts",
                    "alarm_burden_per_hour": "Burden/hour",
                    "mean_time_to_detect_sec": "Detect delay sec",
                    "window_precision": "Window precision",
                }
            )
            st.table(t92_display)
        st.caption(
            "T92 evaluates contactless RR warnings as event-level workflow outputs. "
            "These are normalized public-data metrics, not real deployment alarm rates."
        )
    t94 = load_cambridge_t94_policy_summary()
    t95 = load_cambridge_t95_policy_summary()
    if not t94.empty or not t95.empty:
        st.subheader("T94/T95 Latent-State RR Tracker")
        rows = []
        if not t94.empty:
            t94_focus = t94[
                t94["policy"].isin([T94_BALANCED_POLICY, T94_EQUAL_POLICY, T94_HIGH_RECALL_POLICY])
            ].copy()
            t94_focus["evidence_layer"] = "T94 full-data exploration"
            rows.append(t94_focus)
        if not t95.empty:
            t95_focus = t95[
                t95["policy"].isin([T95_COMBINED_POLICY, T95_MAE_POLICY, T95_HIGH_RR_POLICY])
            ].copy()
            t95_focus["evidence_layer"] = "T95 LOSO validation"
            rows.append(t95_focus)
        latent = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not latent.empty:
            latent_display = latent[
                [
                    "evidence_layer",
                    "policy",
                    "n_subjects",
                    "n_windows",
                    "mae_bpm",
                    "high_rr_mae_bpm",
                    "signed_error_mean_bpm",
                ]
            ].copy()
            for col in ["mae_bpm", "high_rr_mae_bpm", "signed_error_mean_bpm"]:
                latent_display[col] = pd.to_numeric(latent_display[col], errors="coerce").round(3)
            latent_display["policy"] = latent_display["policy"].map(compact_method_name)
            latent_display = latent_display.rename(
                columns={
                    "evidence_layer": "Evidence layer",
                    "policy": "Policy",
                    "n_subjects": "Subjects",
                    "n_windows": "Windows",
                    "mae_bpm": "MAE BPM",
                    "high_rr_mae_bpm": "High-RR MAE BPM",
                    "signed_error_mean_bpm": "Signed error BPM",
                }
            )
            st.table(latent_display)
        st.caption(
            "T94 changes the estimator from window-level harmonic selection to latent-state trajectory inference. "
            "T95 adds leave-subject-out policy selection, supporting high-RR tail mitigation while keeping a small-sample boundary."
        )
    t98_risk = load_cambridge_t98_risk_summary()
    t98_interval = load_cambridge_t98_interval_summary()
    if not t98_risk.empty or not t98_interval.empty:
        st.subheader("T98 Risk-Calibrated RR Intervals")
        if not t98_risk.empty:
            risk_focus = t98_risk[
                t98_risk["calibration_policy"].isin(
                    [
                        T98_DEFAULT_CALIBRATION_POLICY,
                        "risk_target_0.30_t96_reference",
                        "risk_target_0.30_upper_gap_latent",
                    ]
                )
            ].copy()
            if not risk_focus.empty:
                risk_focus["calibration_policy"] = risk_focus["calibration_policy"].map(compact_method_name)
                risk_display = risk_focus[
                    [
                        "calibration_policy",
                        "accepted_coverage",
                        "accepted_mae_bpm",
                        "flagged_mae_bpm",
                        "accepted_high_error_rate",
                        "target_risk_gap",
                        "feasible_fold_rate",
                    ]
                ].copy()
                for col in [
                    "accepted_coverage",
                    "accepted_mae_bpm",
                    "flagged_mae_bpm",
                    "accepted_high_error_rate",
                    "target_risk_gap",
                    "feasible_fold_rate",
                ]:
                    risk_display[col] = pd.to_numeric(risk_display[col], errors="coerce").round(3)
                risk_display = risk_display.rename(
                    columns={
                        "calibration_policy": "Risk policy",
                        "accepted_coverage": "Accepted coverage",
                        "accepted_mae_bpm": "Accepted MAE BPM",
                        "flagged_mae_bpm": "Flagged MAE BPM",
                        "accepted_high_error_rate": "Accepted high-error rate",
                        "target_risk_gap": "Target risk gap",
                        "feasible_fold_rate": "Feasible fold rate",
                    }
                )
                st.table(risk_display)
        if not t98_interval.empty:
            interval_focus = t98_interval[
                t98_interval["interval_policy"].isin([T98_DEFAULT_INTERVAL_POLICY, "t97_q90_gap_q80", "all_windows"])
                & pd.to_numeric(t98_interval["interval_alpha"], errors="coerce").eq(T98_DEFAULT_ALPHA)
            ].copy()
            if not interval_focus.empty:
                interval_focus["interval_policy"] = interval_focus["interval_policy"].map(compact_method_name)
                interval_display = interval_focus[
                    [
                        "interval_policy",
                        "accepted_coverage_vs_all_windows",
                        "empirical_interval_coverage",
                        "coverage_gap_empirical_minus_nominal",
                        "mean_interval_width_bpm",
                        "accepted_mae_bpm",
                        "accepted_high_error_rate",
                    ]
                ].copy()
                for col in [
                    "accepted_coverage_vs_all_windows",
                    "empirical_interval_coverage",
                    "coverage_gap_empirical_minus_nominal",
                    "mean_interval_width_bpm",
                    "accepted_mae_bpm",
                    "accepted_high_error_rate",
                ]:
                    interval_display[col] = pd.to_numeric(interval_display[col], errors="coerce").round(3)
                interval_display = interval_display.rename(
                    columns={
                        "interval_policy": "Interval policy",
                        "accepted_coverage_vs_all_windows": "Window coverage",
                        "empirical_interval_coverage": "Empirical interval coverage",
                        "coverage_gap_empirical_minus_nominal": "Coverage gap",
                        "mean_interval_width_bpm": "Mean width BPM",
                        "accepted_mae_bpm": "Accepted MAE BPM",
                        "accepted_high_error_rate": "Accepted high-error rate",
                    }
                )
                st.table(interval_display)
        st.caption(
            "T98 productizes uncertainty: the dashboard now reports an RR estimate, a calibrated interval, "
            "a source-risk status, and a warning reason. The intervals are still wide, so this remains research evidence."
        )
    t107_summary = load_cambridge_t107_policy_summary()
    if not t107_summary.empty:
        st.subheader("T107 Source-Validity Fallback Route")
        t107_focus_order = [
            "t104_default_no_source_validity_guard",
            T105_REVIEW_POLICY,
            T106_SHIFTED_POLICY,
            T107_SHIFTED_POLICY,
            T107_ROUTE90_POLICY,
            T107_ROUTE80_POLICY,
            "diagnostic_oracle_safe_candidate_reviewed_shifted_t104_interval",
        ]
        t107_focus = t107_summary[t107_summary["policy"].isin(t107_focus_order)].copy()
        if not t107_focus.empty:
            t107_focus["policy_order"] = t107_focus["policy"].map(
                {policy_name: index for index, policy_name in enumerate(t107_focus_order)}
            )
            t107_focus = t107_focus.sort_values("policy_order")
            t107_focus["policy"] = t107_focus["policy"].map(t107_policy_label)
            display_cols = [
                "policy",
                "automation_ready_coverage",
                "automation_ready_mae_bpm",
                "automation_ready_high_error_rate",
                "n_fallback_recovered_windows",
                "recovered_window_mae_bpm",
                "automation_ready_interval_coverage",
                "automation_ready_mean_width_bpm",
                "released_high_error_windows",
                "interval_mode",
            ]
            t107_display = t107_focus[[col for col in display_cols if col in t107_focus.columns]].copy()
            for col in [
                "automation_ready_coverage",
                "automation_ready_mae_bpm",
                "automation_ready_high_error_rate",
                "recovered_window_mae_bpm",
                "automation_ready_interval_coverage",
                "automation_ready_mean_width_bpm",
            ]:
                if col in t107_display.columns:
                    t107_display[col] = pd.to_numeric(t107_display[col], errors="coerce").round(3)
            t107_display = t107_display.rename(
                columns={
                    "policy": "Policy",
                    "automation_ready_coverage": "Ready coverage",
                    "automation_ready_mae_bpm": "Ready MAE BPM",
                    "automation_ready_high_error_rate": "Ready high-error rate",
                    "n_fallback_recovered_windows": "Recovered windows",
                    "recovered_window_mae_bpm": "Recovered MAE BPM",
                    "automation_ready_interval_coverage": "Interval coverage",
                    "automation_ready_mean_width_bpm": "Mean width BPM",
                    "released_high_error_windows": "Released high-error windows",
                    "interval_mode": "Interval mode",
                }
            )
            st.table(t107_display)
        st.caption(
            "T107 turns the source-validity guard into a product route: unsafe windows are withheld unless a label-free "
            "trusted counter-cluster fallback can recover them. Against T106, it recovers two additional windows while "
            "holding released high-error windows at 6."
        )
    t111_summary = load_cambridge_t111_policy_summary()
    if not t111_summary.empty:
        st.subheader("T111 Route-Reliability Safety Gate")
        st.table(t111_policy_summary_table(t111_summary))
        st.caption(
            "T111 adds a second-stage route-reliability gate on synthetic source-validity stress cases. "
            "The high-anchor gate preserves 22 safe recoveries while reducing observed unsafe releases from 2 to 0; "
            "low-to-high correction remains review-only."
        )
    t115_summary = load_cambridge_t115_policy_summary()
    t115_perturbation = load_cambridge_t115_perturbation_summary()
    if not t115_summary.empty:
        st.subheader("T115 Broad-Stress Residual Guard")
        st.table(t115_policy_summary_table(t115_summary, t115_perturbation))
        st.caption(
            "T115 adds a label-free mid-gap broad-stress guard after T114A/T111. "
            "Broad 6 BPM high-anchor stress improves from 23 safe / 2 unsafe / 13 withheld to "
            "23 safe / 0 unsafe / 15 withheld, while canonical 8 BPM and perturbation results remain unchanged."
        )
    t120_summary = load_cambridge_t120_policy_summary()
    t120_loso = load_cambridge_t120_loso_summary()
    if not t120_summary.empty:
        st.subheader("T120 Subject-Aware Route-Risk Calibration")
        st.table(
            t120_policy_summary_table(
                t120_summary,
                validation_set="aggressive_extraction_shift",
                scenario="anchor_abs_ge_6",
            )
        )
        if not t120_loso.empty:
            st.table(t120_loso_summary_table(t120_loso))
        st.caption(
            "T120 adds an episode tail-risk gate on top of T119 route-risk scoring. "
            "In aggressive broad 6 BPM simulation stress, unsafe releases move from 52 to 0, "
            "with safe recoveries moving from 3879 to 3861. This remains internal stress evidence."
        )
    t123_summary = load_cambridge_t123_policy_summary()
    t123_config = load_cambridge_t123_mode_config()
    if not t123_summary.empty:
        st.subheader("T123 Causal-Current Safety Modes")
        st.table(
            t123_policy_summary_table(
                t123_summary,
                validation_set="aggressive_extraction_shift",
                scenario="anchor_abs_ge_6",
            )
        )
        config_display = t123_mode_config_table(t123_config)
        if not config_display.empty:
            st.table(config_display[["Mode", "Intended setting", "Tail-rate threshold", "Strict risk threshold"]])
        st.caption(
            "T123 turns the T122 causal-current risk evidence into deployable safety modes. "
            "In aggressive broad 6 BPM stress all three modes keep unsafe releases at 0, while review-only burden "
            "increases from eldercare to hospital to infant high-caution."
        )
    export_panel(bench, title="phase_d_rr_benchmark")


def upload_quick_scan() -> None:
    uploaded = st.sidebar.file_uploader("Video", type=["mp4", "avi", "mov", "m4v"])
    seconds = st.sidebar.slider("Seconds", min_value=20, max_value=90, value=60, step=10)
    window_sec = st.sidebar.slider("Window length", min_value=20, max_value=45, value=30, step=5)
    step_sec = st.sidebar.slider("Step", min_value=5, max_value=20, value=10, step=5)
    policy_mode = sidebar_policy_selector(default="t57_calibrated_confidence_refusal")
    run = st.sidebar.button("Run quick scan", type="primary", width="stretch")

    if uploaded is None:
        st.info("Upload an RGB video to run the optical-flow-y half-rate RR estimator.")
        return
    if not run:
        st.info("Ready.")
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = Path(tmp.name)

    with st.spinner("Running RR estimator..."):
        result, preview = run_uploaded_video(video_path, seconds=seconds, window_sec=window_sec, step_sec=step_sec)
    result = prepare_product_output(result, pred_col="validated_rr_bpm", policy_mode=policy_mode)
    metrics = summarize_window_metrics(result, pred_col="product_rr_bpm", gt_col=None)

    cols = st.columns(4)
    cols[0].metric("Product RR", latest_product_value(result))
    cols[1].metric("Coverage", f"{fmt(100.0 * metrics['coverage'], 1)}%")
    cols[2].metric("Refused", f"{metrics['refused']}/{metrics['total']}")
    cols[3].metric("Mean confidence", fmt(result["confidence"].mean(), 3))
    product_status_banner(result)

    center, right = st.columns([0.66, 0.34], gap="large")
    with center:
        st.plotly_chart(
            trend_chart(result, title=uploaded.name, reference_col=None, raw_col="raw_rr_bpm", predicted_col="product_rr_bpm"),
            width="stretch",
        )
        st.dataframe(
            result[
                [
                    "window_id",
                    "start_sec",
                    "raw_rr_bpm",
                    "validated_rr_bpm",
                    "product_rr_bpm",
                    "accepted",
                    "refusal_reason",
                    "product_policy_label",
                    "product_policy_threshold",
                    "decision",
                    "half_power_ratio",
                    "confidence",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("ROI")
        st.image(preview, width="stretch")
        st.subheader("Export")
        export_panel(result, title=f"upload_{Path(uploaded.name).stem}")


def adult_hr_upload_monitor() -> None:
    uploaded = st.sidebar.file_uploader("Adult HR video", type=["mp4", "avi", "mov", "m4v"], key="adult_hr_video")
    seconds = int(st.sidebar.slider("Seconds", min_value=8, max_value=60, value=12, step=2, key="adult_hr_seconds"))
    window_sec = int(st.sidebar.slider("Window", min_value=8, max_value=30, value=10, step=2, key="adult_hr_window"))
    step_sec = int(st.sidebar.slider("Step", min_value=4, max_value=20, value=6, step=2, key="adult_hr_step"))
    frame_stride = int(st.sidebar.select_slider("Frame stride", options=[1, 2, 3], value=2, key="adult_hr_stride"))
    use_mediapipe = bool(st.sidebar.checkbox("Face mesh", value=True, key="adult_hr_mediapipe"))
    bridge_policy = st.sidebar.selectbox(
        "HR bridge policy",
        [
            "T227 balanced selective temporal-gated",
            "T228 high-caution conflict guard",
            "T224 temporal-gated failure-aware",
            "T222 failure-aware release-all",
            "T219 current bridge",
        ],
        index=0,
        key="adult_hr_bridge_policy",
    )

    st.subheader("Adult HR")
    if uploaded is None:
        st.info("Upload an RGB face video to run adult heart-rate inference.")
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = Path(tmp.name)

    try:
        with st.spinner("Running adult HR inference..."):
            result = run_adult_hr_video(
                video_path,
                config=AdultHRMVPConfig(
                    seconds=float(seconds),
                    window_sec=float(window_sec),
                    step_sec=float(step_sec),
                    frame_stride=frame_stride,
                    use_mediapipe=use_mediapipe,
                ),
            )
            try:
                bridge_config = None
                temporal_gate_policy = "none"
                respect_anchor_review = False
                conflict_guard_policy = "none"
                policy_name = "live_topk_bridge_v1"
                if bridge_policy == "T222 failure-aware release-all":
                    bridge_config = TopKBridgeConfig(enable_upper_band_rescue=True)
                    policy_name = "live_upper_band_rescue_v1"
                elif bridge_policy == "T224 temporal-gated failure-aware":
                    bridge_config = TopKBridgeConfig(enable_upper_band_rescue=True)
                    temporal_gate_policy = "veto_to_anchor"
                    policy_name = "live_upper_band_rescue_temporal_veto_v1"
                elif bridge_policy == "T227 balanced selective temporal-gated":
                    bridge_config = TopKBridgeConfig(enable_upper_band_rescue=True)
                    temporal_gate_policy = "veto_to_anchor"
                    respect_anchor_review = True
                    policy_name = "live_selective_temporal_veto_v1"
                elif bridge_policy == "T228 high-caution conflict guard":
                    bridge_config = TopKBridgeConfig(enable_upper_band_rescue=True)
                    temporal_gate_policy = "veto_to_anchor"
                    respect_anchor_review = True
                    conflict_guard_policy = "low_pred_conflict_v2"
                    policy_name = "live_selective_temporal_conflict_guard_v2"
                live_candidates, bridge_selection, bridge_product = build_live_adult_hr_topk_bridge(
                    result,
                    top_k=5,
                    bridge_config=bridge_config,
                    policy_name=policy_name,
                    temporal_gate_policy=temporal_gate_policy,
                    respect_anchor_review=respect_anchor_review,
                    conflict_guard_policy=conflict_guard_policy,
                )
                if not bridge_product.empty:
                    bridge_product = score_product_table_with_reliability_guard(
                        bridge_product,
                        apply_review=False,
                    )
                live_bridge_error = ""
            except Exception as exc:
                live_candidates = pd.DataFrame()
                bridge_selection = pd.DataFrame()
                bridge_product = pd.DataFrame()
                live_bridge_error = str(exc)
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass

    windows = result.windows.copy()
    left, right = st.columns([1.35, 1.0])
    with left:
        if windows.empty:
            st.warning("No HR window could be produced.")
        else:
            accepted = windows["accepted"].astype(bool) if "accepted" in windows.columns else pd.Series(False, index=windows.index)
            released = pd.to_numeric(windows.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce")
            bridge_released = (
                pd.to_numeric(bridge_product.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce")
                if not bridge_product.empty
                else pd.Series(dtype=float)
            )
            candidate = pd.to_numeric(windows.get("candidate_hr_bpm", pd.Series(dtype=float)), errors="coerce")
            coverage = float(accepted.sum() / max(1, len(windows)))
            latest_release = bridge_released.dropna().iloc[-1] if bridge_released.notna().any() else (released.dropna().iloc[-1] if released.notna().any() else np.nan)
            bridge_coverage = (
                float(pd.to_numeric(bridge_product.get("released", pd.Series(dtype=float)), errors="coerce").fillna(0).mean())
                if not bridge_product.empty
                else np.nan
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest HR", f"{latest_release:.1f} BPM" if np.isfinite(latest_release) else "Review")
            c2.metric("Coverage", f"{coverage:.0%}")
            c3.metric("Windows", str(len(windows)))
            c4.metric("Bridge", f"{bridge_coverage:.0%}" if np.isfinite(bridge_coverage) else "NA")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=windows["start_sec"],
                    y=candidate,
                    mode="lines+markers",
                    name="candidate",
                    line=dict(color="#7A8396", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=windows["start_sec"],
                    y=released,
                    mode="lines+markers",
                    name="released",
                    line=dict(color="#2C7C89", width=4),
                )
            )
            if "max_power_candidate_bpm" in windows.columns:
                fig.add_trace(
                    go.Scatter(
                        x=windows["start_sec"],
                        y=pd.to_numeric(windows["max_power_candidate_bpm"], errors="coerce"),
                        mode="lines+markers",
                        name="max peak",
                        line=dict(color="#C77C2B", width=2, dash="dot"),
                    )
                )
            if not bridge_product.empty:
                bridge_plot = bridge_product.copy()
                if "sample_id" in windows.columns and "start_sec" in windows.columns:
                    starts = windows[["sample_id", "start_sec"]].copy()
                    starts["sample_id"] = starts["sample_id"].astype(str)
                    bridge_plot = bridge_plot.merge(starts, on="sample_id", how="left")
                    bridge_plot["window_start"] = pd.to_numeric(bridge_plot["start_sec"], errors="coerce")
                else:
                    bridge_plot["window_start"] = np.arange(len(bridge_plot), dtype=float)
                fig.add_trace(
                    go.Scatter(
                        x=bridge_plot["window_start"],
                        y=pd.to_numeric(bridge_plot["product_hr_bpm"], errors="coerce"),
                        mode="lines+markers",
                        name=bridge_policy,
                        line=dict(color="#0072B2", width=3, dash="dash"),
                    )
                )
            refused = windows[~accepted]
            if not refused.empty and "candidate_hr_bpm" in refused.columns:
                fig.add_trace(
                    go.Scatter(
                        x=refused["start_sec"],
                        y=pd.to_numeric(refused["candidate_hr_bpm"], errors="coerce"),
                        mode="markers",
                        name="review",
                        marker=dict(color="#B65A4A", symbol="x", size=11),
                    )
                )
            fig.update_layout(
                title=uploaded.name,
                xaxis_title="Window start (s)",
                yaxis_title="Heart rate (BPM)",
                template="plotly_white",
                height=360,
                margin=dict(l=10, r=10, t=45, b=10),
            )
            if "end_sec" in windows.columns and pd.to_numeric(windows["end_sec"], errors="coerce").notna().any():
                max_time = float(pd.to_numeric(windows["end_sec"], errors="coerce").max())
                fig.update_xaxes(range=[0, max(1.0, max_time)])
            st.plotly_chart(fig, width="stretch")
            display_cols = [
                "window_id",
                "start_sec",
                "end_sec",
                "product_hr_bpm",
                "candidate_hr_bpm",
                "accepted",
                "decision",
                "refusal_reason",
                "roi_support",
                "method_support",
                "roi_evidence_v2_score",
                "max_power_candidate_bpm",
            ]
            st.dataframe(windows[[col for col in display_cols if col in windows.columns]], width="stretch", hide_index=True)
            if live_bridge_error:
                st.warning(f"Top-K bridge evidence unavailable: {live_bridge_error}")
            elif not bridge_product.empty:
                st.subheader("Top-K Bridge Product Output")
                bridge_cols = [
                    "sample_id",
                    "decision",
                    "product_hr_bpm",
                    "candidate_hr_bpm",
                    "bridge_source",
                    "bridge_anchor_bpm",
                    "max_power_candidate_bpm",
                    "selected_score",
                    "selected_support_count",
                    "selected_support_rois",
                    "selected_support_methods",
                    "selected_top1_support_count",
                    "selected_pos_chrom_count",
                    "selected_dist_to_anchor_bpm",
                    "support_guard_risk_score",
                    "support_guard_threshold",
                    "support_guard_passed",
                    "support_guard_context_status",
                    "support_guard_context_release_candidate",
                    "support_guard_reason",
                    "pre_temporal_product_hr_bpm",
                    "pre_temporal_bridge_source",
                    "temporal_support_count",
                    "temporal_gate_passed",
                    "temporal_gate_reason",
                    "pre_upstream_review_product_hr_bpm",
                    "pre_upstream_review_bridge_source",
                    "upstream_review_gate_passed",
                    "upstream_review_gate_reason",
                    "pre_conflict_guard_product_hr_bpm",
                    "pre_conflict_guard_bridge_source",
                    "conflict_guard_max_gap_bpm",
                    "conflict_guard_passed",
                    "conflict_guard_reason",
                ]
                st.dataframe(
                    bridge_product[[col for col in bridge_cols if col in bridge_product.columns]],
                    width="stretch",
                    hide_index=True,
                )
    with right:
        st.subheader("ROI")
        if result.preview_rgb is not None:
            st.image(result.preview_rgb, width="stretch")
        st.subheader("Evidence")
        st.json(
            {
                "fps": result.metadata.get("video_fps"),
                "analysis_fps": result.metadata.get("analysis_fps"),
                "detection_rate": result.metadata.get("detector_meta", {}).get("detection_rate"),
                "candidates": result.metadata.get("n_candidates"),
                "clusters": result.metadata.get("n_clusters"),
                "live_topk_candidates": int(len(live_candidates)),
                "bridge_rows": int(len(bridge_selection)),
            },
            expanded=False,
        )
        if not bridge_product.empty:
            st.subheader("Bridge Evidence")
            bridge_sample = st.selectbox(
                "Bridge sample",
                bridge_product["sample_id"].astype(str).tolist(),
                key=f"adult_hr_live_bridge_sample_{safe_filename(uploaded.name)}",
            )
            selected_bridge = bridge_product[bridge_product["sample_id"].astype(str).eq(bridge_sample)].head(1)
            if not selected_bridge.empty:
                raw = selected_bridge.iloc[0].get("evidence_json", "{}")
                try:
                    evidence = json.loads(raw)
                except Exception:
                    evidence = {"raw": raw}
                st.json(evidence, expanded=True)
        st.download_button(
            "Download windows CSV",
            data=windows.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{safe_filename(Path(uploaded.name).stem)}_adult_hr_windows.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Download candidates CSV",
            data=result.candidates.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{safe_filename(Path(uploaded.name).stem)}_adult_hr_candidates.csv",
            mime="text/csv",
            width="stretch",
        )
        if not live_candidates.empty:
            st.download_button(
                "Download top-K candidates CSV",
                data=live_candidates.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{safe_filename(Path(uploaded.name).stem)}_adult_hr_topk_candidates.csv",
                mime="text/csv",
                width="stretch",
            )
        if not bridge_product.empty:
            st.download_button(
                "Download bridge product CSV",
                data=bridge_product.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{safe_filename(Path(uploaded.name).stem)}_adult_hr_bridge_product.csv",
                mime="text/csv",
                width="stretch",
            )


def adult_hr_bridge_evidence_monitor() -> None:
    t348_product = load_adult_t348_product_table()
    t348_examples = load_adult_t348_api_examples()
    if not t348_product.empty:
        product = t348_product
        examples = t348_examples
        source_col = "source"
        st.subheader("Adult HR Product Evidence")
    else:
        product = load_adult_t217_product_table()
        examples = load_adult_t217_api_examples()
        source_col = "bridge_source"
        st.subheader("Adult HR Bridge Evidence")
    if product.empty:
        st.info("Run T348 or T217 to generate an adult HR product table.")
        return

    dataset_options = ["All"] + sorted(product["dataset"].dropna().astype(str).unique().tolist())
    mode_options = ["All"] + sorted(product["product_mode_label"].dropna().astype(str).unique().tolist()) if "product_mode_label" in product.columns else ["All"]
    source_options = ["All"] + sorted(product[source_col].dropna().astype(str).unique().tolist()) if source_col in product.columns else ["All"]
    dataset_filter = st.sidebar.selectbox("Dataset", dataset_options, key="adult_bridge_dataset")
    mode_filter = st.sidebar.selectbox("Product mode", mode_options, key="adult_product_mode") if len(mode_options) > 1 else "All"
    source_filter = st.sidebar.selectbox("Source", source_options, key="adult_bridge_source")

    view = product.copy()
    if dataset_filter != "All":
        view = view[view["dataset"].astype(str).eq(dataset_filter)].copy()
    if mode_filter != "All" and "product_mode_label" in view.columns:
        view = view[view["product_mode_label"].astype(str).eq(mode_filter)].copy()
    if source_filter != "All" and source_col in view.columns:
        view = view[view[source_col].astype(str).eq(source_filter)].copy()

    released = pd.to_numeric(view["released"], errors="coerce").fillna(0).astype(int) if "released" in view.columns else pd.Series(0, index=view.index)
    errors = pd.to_numeric(view.get("eval_abs_error_bpm", pd.Series(dtype=float)), errors="coerce")
    rel_errors = errors[(released > 0) & errors.notna()].to_numpy(dtype=float)
    cols = st.columns(4)
    cols[0].metric("Rows", str(len(view)))
    cols[1].metric("Coverage", f"{released.mean():.0%}" if len(released) else "NA")
    cols[2].metric("Eval MAE", f"{np.mean(rel_errors):.2f} BPM" if len(rel_errors) else "NA")
    cols[3].metric("Unsafe/input", f"{np.sum(rel_errors > 10.0) / max(1, len(view)):.1%}" if len(rel_errors) else "NA")

    st.dataframe(adult_t217_product_table_display(view), width="stretch", hide_index=True)

    selected_sample = st.selectbox("Evidence sample", view["sample_id"].astype(str).tolist(), key="adult_bridge_sample")
    selected = view[view["sample_id"].astype(str).eq(selected_sample)].head(1)
    if not selected.empty:
        raw = selected.iloc[0].get("evidence_json", "{}")
        try:
            evidence = json.loads(raw)
        except Exception:
            evidence = {"raw": raw}
        key_value_panel(
            [
                ("Decision", selected.iloc[0].get("decision", "")),
                ("Mode", selected.iloc[0].get("product_mode_label", "")),
                ("Source", selected.iloc[0].get(source_col, "")),
                ("Product HR", selected.iloc[0].get("product_hr_bpm", "")),
                ("Anchor HR", selected.iloc[0].get("bridge_anchor_bpm", "")),
            ]
        )
        st.json(evidence, expanded=False)

    if examples:
        with st.expander("API examples", expanded=False):
            st.json(examples, expanded=False)
    st.download_button(
        "Download product table",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="adult_hr_product_table.csv",
        mime="text/csv",
        width="stretch",
    )


def run_uploaded_video(video_path: Path, *, seconds: int, window_sec: int, step_sec: int) -> tuple[pd.DataFrame, np.ndarray]:
    meta = get_video_metadata(video_path)
    max_frames = int(min(meta.frame_count, meta.fps * seconds))
    frame = first_frame(video_path)
    rois = body_aware_respiration_rois(frame, view="infant", infant=True, max_rois=6, min_score=0.20)
    selected = rois[0]
    flow = optical_flow_signals(video_path, selected, max_frames=max_frames, sample_every=1, max_side=140)
    signal = flow["optical_flow_y"]
    rows: list[dict[str, object]] = []
    previous_bpm: float | None = None
    for window_id, (start_sec, end_sec, start, end) in enumerate(
        window_slices(len(signal.values), signal.fps, window_sec=float(window_sec), step_sec=float(step_sec))
    ):
        validation = estimate_rr_half_rate_validated(signal.values[start:end], signal.fps, previous_bpm=previous_bpm)
        if np.isfinite(validation.estimate.bpm):
            previous_bpm = validation.estimate.bpm
        rows.append(
            {
                "window_id": window_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "raw_rr_bpm": validation.raw_estimate.bpm,
                "validated_rr_bpm": validation.estimate.bpm,
                "decision": validation.decision,
                "half_bpm": validation.half_bpm,
                "half_power_ratio": validation.half_power_ratio,
                "confidence": validation.estimate.confidence,
                "roi_name": selected.name,
            }
        )
    preview = draw_rois(frame, rois, selected=selected)
    return pd.DataFrame(rows), cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def summarize_window_metrics(frame: pd.DataFrame, *, pred_col: str, gt_col: str | None) -> dict[str, float | int]:
    pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=float)
    accepted = int(np.isfinite(pred).sum())
    total = int(len(frame))
    refused = total - accepted
    coverage = float(accepted / total) if total else float("nan")
    if gt_col is None or gt_col not in frame.columns:
        return {"mae": float("nan"), "rmse": float("nan"), "coverage": coverage, "accepted": accepted, "refused": refused, "total": total}
    true = pd.to_numeric(frame[gt_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return {"mae": float("nan"), "rmse": float("nan"), "coverage": coverage, "accepted": accepted, "refused": refused, "total": total}
    err = pred[mask] - true[mask]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "coverage": coverage,
        "accepted": accepted,
        "refused": refused,
        "total": total,
    }


def export_panel(frame: pd.DataFrame, *, title: str) -> None:
    widget_key = safe_filename(title)
    csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{safe_filename(title)}.csv",
        mime="text/csv",
        width="stretch",
        key=f"{widget_key}_csv",
    )
    st.download_button(
        "Download PDF",
        data=make_pdf_report(frame, title=title),
        file_name=f"{safe_filename(title)}.pdf",
        mime="application/pdf",
        width="stretch",
        key=f"{widget_key}_pdf",
    )


def make_pdf_report(frame: pd.DataFrame, *, title: str) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "VitalsSight RR Report", fontsize=18, weight="bold")
        fig.text(0.08, 0.91, title, fontsize=10, color="#60707d")
        fig.text(0.08, 0.87, f"Rows: {len(frame)}", fontsize=10)
        if "accepted" in frame.columns:
            accepted = int(frame["accepted"].astype(bool).sum())
            total = len(frame)
            coverage = accepted / total if total else float("nan")
            fig.text(0.08, 0.85, f"Accepted: {accepted}/{total}  Coverage: {coverage:.1%}", fontsize=10)
        y_col = None
        if {"start_sec", "product_rr_bpm"}.issubset(frame.columns):
            y_col = "product_rr_bpm"
        elif {"start_sec", "pred_rr_bpm"}.issubset(frame.columns):
            y_col = "pred_rr_bpm"
        if y_col is not None:
            ax = fig.add_axes([0.08, 0.58, 0.84, 0.24])
            label = "product" if y_col == "product_rr_bpm" else "prediction"
            ax.plot(frame["start_sec"], frame[y_col], color="#2f7d6d", marker="o", label=label)
            if "candidate_rr_bpm" in frame.columns:
                ax.plot(frame["start_sec"], frame["candidate_rr_bpm"], color="#81909b", marker=".", alpha=0.55, label="candidate")
            if "raw_rr_bpm" in frame.columns:
                ax.plot(frame["start_sec"], frame["raw_rr_bpm"], color="#b85c38", marker="o", label="raw")
            if "gt_rr_bpm" in frame.columns:
                ax.plot(frame["start_sec"], frame["gt_rr_bpm"], color="#111827", marker="o", label="reference")
            if "accepted" in frame.columns and "candidate_rr_bpm" in frame.columns:
                refused = frame[~frame["accepted"].astype(bool)]
                if not refused.empty:
                    ax.scatter(refused["start_sec"], refused["candidate_rr_bpm"], color="#a84d3a", marker="x", s=55, label="refused")
            ax.set_xlabel("Window start (s)")
            ax.set_ylabel("RR (BPM)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        table_cols = [
            col
            for col in ["window_id", "start_sec", "gt_rr_bpm", "raw_rr_bpm", "validated_rr_bpm", "pred_rr_bpm", "decision"]
            if col in frame.columns
        ]
        product_cols = [
            col
            for col in ["product_rr_bpm", "accepted", "refusal_reason", "product_policy_label", "product_policy_threshold"]
            if col in frame.columns and col not in table_cols
        ]
        table_cols.extend(product_cols)
        if table_cols:
            table = frame[table_cols].head(12)
        else:
            excluded = {"evidence_path", "notes", "source_task"}
            compact_cols = [col for col in frame.columns if col not in excluded][:8]
            table = frame[compact_cols].head(12)
        ax_table = fig.add_axes([0.08, 0.08, 0.84, 0.42])
        ax_table.axis("off")
        ax_table.table(cellText=table.round(3).astype(str).values, colLabels=table.columns, loc="upper left", cellLoc="left")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:120]


def adult_hr_market_console_v2(active_workspace: str) -> None:
    """Compatibility wrapper for the retired v2 market-console implementation."""
    adult_hr_product_console_v3(active_workspace)

def adult_hr_product_console_v3(active_workspace: str) -> None:
    """Evidence-driven adult HR product console with stable bilingual interactions."""

    assets = load_adult_console_assets()
    demo_cases = assets.get("demo_cases", pd.DataFrame()).copy()
    report_cards = assets.get("report_cards", pd.DataFrame()).copy()
    route_metrics = assets.get("route_metrics", pd.DataFrame()).copy()
    policy = assets.get("policy", pd.DataFrame()).copy()
    competitor_benchmark = assets.get("competitor_benchmark", pd.DataFrame()).copy()
    qa = assets.get("qa", {}) if isinstance(assets.get("qa", {}), dict) else {}
    stats = assets.get("stats", {}) if isinstance(assets.get("stats", {}), dict) else {}

    if demo_cases.empty:
        demo_cases = pd.DataFrame(
            [
                {
                    "case_id": "fallback_release",
                    "dataset": "UBFC-rPPG",
                    "sample_id": "subject1_w000",
                    "room": "DP-001",
                    "route_id": "dense_patch_temporal_semantic_roi_t687_oof_risk_gate",
                    "selected_policy": "T687 OOF risk gate over dense-patch temporal selector",
                    "decision": "release",
                    "hr_bpm": 105.7,
                    "confidence_proxy_for_demo": 0.94,
                    "review_reason": "",
                    "required_input": "adult RGB face video, dense patches, candidate peaks, temporal consistency",
                    "quality_flags": "safe_selection",
                    "claim_boundary": "Research product build; not clinical diagnosis or emergency monitoring.",
                    "product_warning": "Research use only; not clinical diagnosis or emergency monitoring.",
                    "selected_candidate_hr_bpm": 105.7,
                    "selected_region": "forehead",
                    "selected_patch_family": "semantic_roi",
                    "selected_method": "chrom",
                    "selected_snr_db": 16.1,
                    "selected_peak_support": 0.23,
                    "selected_median_consistency": 1.0,
                    "selected_score": 2.87,
                    "failure_mode": "safe_selection",
                    "risk_factors_json": "{}",
                    "competing_candidates_json": "[]",
                    "evidence_summary": "forehead/chrom, SNR=16.1, support=0.23, consistency=1.0",
                },
                {
                    "case_id": "fallback_review",
                    "dataset": "MCD-rPPG",
                    "sample_id": "adult_motion_review",
                    "room": "DP-002",
                    "route_id": "dense_patch_temporal_semantic_roi_t687_oof_risk_gate",
                    "selected_policy": "T687 OOF risk gate over dense-patch temporal selector",
                    "decision": "review",
                    "hr_bpm": np.nan,
                    "confidence_proxy_for_demo": 0.62,
                    "review_reason": "Candidate evidence is insufficient for automatic release.",
                    "required_input": "adult RGB face video, motion score, candidate conflict features",
                    "quality_flags": "review_required",
                    "claim_boundary": "Review route; do not auto-release HR.",
                    "product_warning": "Human review required before use.",
                    "selected_candidate_hr_bpm": 104.0,
                    "selected_region": "chin",
                    "selected_patch_family": "semantic_roi",
                    "selected_method": "chrom",
                    "selected_snr_db": 11.9,
                    "selected_peak_support": 0.12,
                    "selected_median_consistency": 0.68,
                    "selected_score": 1.83,
                    "failure_mode": "low_confidence",
                    "risk_factors_json": "{}",
                    "competing_candidates_json": "[]",
                    "evidence_summary": "chin/chrom, SNR=11.9, support=0.12, consistency=0.68",
                },
            ]
        )

    sections = ["Command Center", "Live Scan", "Patients", "Alerts", "Route-MoE", "Reports", "Integrations"]
    workspace_map = {
        "VitalsSight Adult HR Console": "Command Center",
        "AIR Sample Monitor": "Live Scan",
        "Cambridge Trend Monitor": "Reports",
        "Benchmark Overview": "Route-MoE",
        "Adult HR Upload": "Live Scan",
        "Adult HR Bridge Evidence": "Route-MoE",
        "Upload Video Quick Scan": "Live Scan",
    }
    workspace_map.update({name: name for name in sections})
    section_zh = {
        "Command Center": "指挥中心",
        "Live Scan": "实时扫描",
        "Patients": "患者管理",
        "Alerts": "告警队列",
        "Route-MoE": "Route-MoE",
        "Reports": "报告中心",
        "Integrations": "系统集成",
    }
    status_zh = {"release": "已发布", "review": "复核", "urgent": "紧急", "disconnected": "离线"}
    status_en = {"release": "Released", "review": "Review", "urgent": "Urgent", "disconnected": "Disconnected"}

    def query_get(name: str, default: str = "") -> str:
        try:
            value = st.query_params.get(name, default)
        except Exception:
            return default
        if isinstance(value, list):
            return str(value[0]) if value else default
        return str(value) if value is not None else default

    st.session_state.setdefault("vs_v3_language", query_get("lang", "ZH") if query_get("lang", "ZH") in {"EN", "ZH"} else "ZH")
    st.session_state.setdefault("vs_v3_theme", query_get("theme", "Dark") if query_get("theme", "Dark") in {"System", "Light", "Dark"} else "Dark")
    st.session_state.setdefault("vs_v3_section", workspace_map.get(active_workspace, "Command Center"))
    st.session_state.setdefault("vs_v3_seen_workspace", active_workspace)
    if active_workspace != st.session_state["vs_v3_seen_workspace"]:
        st.session_state["vs_v3_section"] = workspace_map.get(active_workspace, st.session_state["vs_v3_section"])
        st.session_state["vs_v3_seen_workspace"] = active_workspace
    query_section = query_get("section", "")
    if query_section in sections:
        st.session_state["vs_v3_section"] = query_section

    st.session_state.setdefault("vs_v3_focus_idx", 0)
    st.session_state.setdefault("vs_v3_review_queue", [])
    st.session_state.setdefault("vs_v3_notes", {})
    st.session_state.setdefault("vs_v3_last_action", "")
    st.session_state.setdefault("vs_v3_scan_result", None)
    st.session_state.setdefault("vs_v3_webhook_log", [])

    def ui(en: str, zh: str) -> str:
        return zh if st.session_state.get("vs_v3_language", "ZH") == "ZH" else en

    def clean(value: object, fallback: str = "") -> str:
        if value is None:
            return fallback
        if isinstance(value, float) and pd.isna(value):
            return fallback
        text = str(value)
        if text.lower() in {"nan", "none"}:
            return fallback
        return text

    def safe_float(value: object) -> float | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if not np.isfinite(numeric) else float(numeric)

    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return None if not np.isfinite(value) else float(value)
        if value is None or isinstance(value, (str, int, bool)):
            return value
        return str(value)

    def parse_json(value: object, fallback: object) -> object:
        if isinstance(value, (dict, list)):
            return value
        text = clean(value, "")
        if not text:
            return fallback
        try:
            return json.loads(text)
        except Exception:
            return fallback

    def case_display_id(index: int) -> str:
        return f"A-{301 + index:03d}"

    def row_kind(row: pd.Series) -> str:
        decision = clean(row.get("decision", "review")).lower()
        reason_text = clean(row.get("review_reason", "")).lower()
        route = clean(row.get("route_id", "")).lower()
        if decision == "release":
            return "release"
        if "disconnect" in reason_text or "missing" in reason_text:
            return "disconnected"
        if "mcd" in clean(row.get("dataset", "")).lower() or "high" in route or "urgent" in reason_text:
            return "urgent"
        return "review"

    def status_text(kind: str) -> str:
        return (status_zh if st.session_state.get("vs_v3_language") == "ZH" else status_en).get(kind, kind.title())

    def confidence(row: pd.Series, kind: str) -> float:
        value = safe_float(row.get("confidence_proxy_for_demo"))
        if value is None:
            return {"release": 0.94, "review": 0.68, "urgent": 0.58, "disconnected": 0.0}.get(kind, 0.5)
        return max(0.0, min(1.0, value if value <= 1 else value / 3.0))

    def candidate_hr(row: pd.Series) -> str:
        released = safe_float(row.get("hr_bpm"))
        selected = safe_float(row.get("selected_candidate_hr_bpm"))
        value = released if released is not None else selected
        return "--" if value is None else f"{value:.0f}"

    def reason_text(row: pd.Series, kind: str) -> str:
        raw = clean(row.get("review_reason"))
        if raw:
            return raw
        return {
            "release": ui("Evidence is sufficient for automatic HR release.", "证据足够，允许自动发布心率。"),
            "review": ui("Evidence is incomplete; route to review.", "证据不完整，进入复核。"),
            "urgent": ui("High-risk candidate conflict; review first.", "高风险候选冲突，优先复核。"),
            "disconnected": ui("Signal missing or insufficient.", "信号缺失或不足。"),
        }.get(kind, "")

    def build_payload(row: pd.Series, index: int) -> dict[str, object]:
        kind = row_kind(row)
        method_claim = (
            "Dense face-patch candidates are scored by temporal/failure-aware evidence; "
            "the product releases HR only when the release/review gate accepts the selected candidate."
        )
        release_review_rule = (
            "Release means the selected candidate passed the configured evidence gate. "
            "Review means the candidate pool contains insufficient or conflicting evidence for automatic HR reporting."
        )
        unsupported_claims = [
            "clinical-grade diagnosis",
            "universal SOTA",
            "solved fairness",
            "solved low-light/NIR robustness",
            "solved camera/source-shift robustness",
            "TCM organ-causal HR improvement",
        ]
        payload = {
            "bed_id": case_display_id(index),
            "case_id": clean(row.get("case_id"), case_display_id(index)),
            "dataset": clean(row.get("dataset"), "unknown"),
            "sample_id": clean(row.get("sample_id"), "unknown"),
            "decision": kind,
            "display_status": status_text(kind),
            "released_hr_bpm": safe_float(row.get("hr_bpm")),
            "selected_candidate_hr_bpm": safe_float(row.get("selected_candidate_hr_bpm")),
            "confidence": round(confidence(row, kind), 3),
            "selected_region": clean(row.get("selected_region"), "unknown"),
            "selected_patch_family": clean(row.get("selected_patch_family"), "unknown"),
            "selected_method": clean(row.get("selected_method"), "unknown"),
            "selected_snr_db": safe_float(row.get("selected_snr_db")),
            "selected_peak_support": safe_float(row.get("selected_peak_support")),
            "selected_median_consistency": safe_float(row.get("selected_median_consistency")),
            "review_reason": reason_text(row, kind),
            "risk_factors": parse_json(row.get("risk_factors_json"), {}),
            "competing_candidates": parse_json(row.get("competing_candidates_json"), []),
            "claim_boundary": clean(row.get("claim_boundary"), "Research use only; not clinical diagnosis."),
            "product_warning": clean(row.get("product_warning"), "Research use only."),
            "evidence_summary": clean(row.get("evidence_summary"), "No evidence summary available."),
            "method_claim": method_claim,
            "release_review_rule": release_review_rule,
            "evidence_gate": clean(row.get("evidence_gate"), "T731/T732/T733 bounded Deep-Candidate PhysioGate evidence"),
            "unsupported_claims": unsupported_claims,
        }
        return json_safe(payload)

    def report_markdown(row: pd.Series, index: int) -> str:
        payload = build_payload(row, index)
        lines = [
            "# VitalsSight Adult HR Evidence Report",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"Case: {payload['bed_id']} | {payload['case_id']}",
            f"Dataset: {payload['dataset']}",
            f"Decision: {payload['decision']}",
            f"Released HR: {payload['released_hr_bpm'] if payload['released_hr_bpm'] is not None else 'review only'}",
            f"Selected candidate HR: {payload['selected_candidate_hr_bpm']}",
            f"Confidence: {payload['confidence']}",
            f"ROI/method: {payload['selected_region']} / {payload['selected_method']}",
            f"SNR: {payload['selected_snr_db']}",
            f"Reason: {payload['review_reason']}",
            "",
            "## Method Claim",
            str(payload["method_claim"]),
            "",
            "## Release/Review Rule",
            str(payload["release_review_rule"]),
            "",
            "## Decision Evidence",
            f"Selected region: {payload['selected_region']}",
            f"Patch family: {payload['selected_patch_family']}",
            f"Method: {payload['selected_method']}",
            f"Peak support: {payload['selected_peak_support']}",
            f"Median consistency: {payload['selected_median_consistency']}",
            f"Evidence gate: {payload['evidence_gate']}",
            "",
            "## Claim Boundary",
            str(payload["claim_boundary"]),
            "",
            "Unsupported claims for this report:",
            *[f"- {item}" for item in payload["unsupported_claims"]],
            "",
            "## Product Warning",
            str(payload["product_warning"]),
            "",
            "## Evidence JSON",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
        ]
        return "\n".join(lines)

    def render_json_block(value: object) -> None:
        formatted = json.dumps(json_safe(value), ensure_ascii=False, indent=2)
        st.markdown(f"<pre class='vs-json'>{escape(formatted)}</pre>", unsafe_allow_html=True)

    language = st.sidebar.radio(
        "Language / 语言",
        ["EN", "ZH"],
        key="vs_v3_language",
        horizontal=True,
        format_func=lambda value: "English" if value == "EN" else "中文",
    )
    theme = st.sidebar.radio("Theme / 主题", ["System", "Light", "Dark"], key="vs_v3_theme", horizontal=True)
    try:
        st.query_params["lang"] = language
        st.query_params["theme"] = theme
        st.query_params["section"] = st.session_state["vs_v3_section"]
    except Exception:
        pass

    page_style = {
        "Dark": ("#061421", "#0b2132", "#12344a", "#dff7ff", "#71d8ff"),
        "Light": ("#f5f9fc", "#ffffff", "#d7e7f2", "#0b2030", "#0077b6"),
        "System": ("#0a1723", "#102435", "#28475c", "#ecfbff", "#38bdf8"),
    }[theme]
    bg, panel, border, text, accent = page_style
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        section[data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
        section[data-testid="stSidebar"] * {{ color: {text}; }}
        .vs-card {{ border: 1px solid {border}; background: {panel}; border-radius: 10px; padding: 14px; min-height: 118px; }}
        .vs-card h3, .vs-card h4 {{ margin: 0 0 8px 0; color: {text}; }}
        .vs-muted {{ color: {accent}; font-size: 0.86rem; }}
        .vs-kpi {{ font-size: 1.9rem; font-weight: 800; color: {text}; }}
        .vs-good {{ color: #4ade80; }} .vs-warn {{ color: #f59e0b; }} .vs-bad {{ color: #fb7185; }}
        .vs-pill {{ display: inline-block; padding: 3px 9px; border-radius: 999px; border: 1px solid {border}; margin-right: 6px; }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px;
            min-height: 42px;
            font-weight: 700;
            background: #12344a !important;
            border: 1px solid #2c5a75 !important;
            color: #e7f8ff !important;
        }}
        .stButton > button p, .stDownloadButton > button p {{
            color: #e7f8ff !important;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: #12344a !important;
            border: 1px solid #2c5a75 !important;
            color: #e7f8ff !important;
        }}
        section[data-testid="stSidebar"] .stButton > button p {{
            color: #e7f8ff !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background: {panel} !important;
            border: 1px solid {border} !important;
            color: {text} !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] div[data-baseweb="select"] input,
        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
            color: {text} !important;
            fill: {text} !important;
        }}
        div[data-baseweb="popover"] ul,
        div[data-baseweb="menu"] {{
            background: {panel} !important;
            color: {text} !important;
        }}
        div[data-baseweb="popover"] li,
        div[data-baseweb="menu"] li {{
            color: {text} !important;
        }}
        .vs-json {{
            background: #071826 !important;
            color: #dff7ff !important;
            border: 1px solid {border};
            border-radius: 10px;
            padding: 14px;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        div[data-testid="stRadio"] label {{ padding: 4px 8px; border-radius: 8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    focus_labels = [f"{case_display_id(i)} | {clean(row.get('dataset'), 'unknown')} | {row_kind(row)}" for i, row in demo_cases.iterrows()]
    if focus_labels:
        current_index = min(int(st.session_state.get("vs_v3_focus_idx", 0)), len(focus_labels) - 1)
        selected_focus = st.sidebar.selectbox(ui("Focus case", "当前病例"), focus_labels, index=current_index, key="vs_v3_focus_select")
        st.session_state["vs_v3_focus_idx"] = focus_labels.index(selected_focus)
    focus_idx = min(int(st.session_state.get("vs_v3_focus_idx", 0)), len(demo_cases) - 1)
    current_row = demo_cases.iloc[focus_idx]
    current_kind = row_kind(current_row)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {ui('Product modules', '产品模块')}")
    for name in sections:
        if st.sidebar.button(label := f"{sections.index(name)+1}. {section_zh[name] if language == 'ZH' else name}", key=f"sidebar_sec_{name}", use_container_width=True):
            st.session_state["vs_v3_section"] = name
            try:
                st.query_params["section"] = name
            except Exception:
                pass
    st.sidebar.caption(ui("Research product: not clinical diagnosis or emergency alerting.", "研究产品：不是临床诊断、急救告警或医疗器械决策系统。"))

    st.title(ui("VitalsSight Adult HR", "VitalsSight 成人心率"))
    st.caption(ui("Evidence-driven contactless adult heart-rate monitoring from RGB face video", "基于普通 RGB 人脸视频的证据驱动非接触成人心率监测"))

    nav_cols = st.columns(len(sections))
    for idx, name in enumerate(sections):
        button_label = f"{idx+1}. {section_zh[name] if language == 'ZH' else name}"
        if nav_cols[idx].button(button_label, key=f"top_nav_{name}", use_container_width=True):
            st.session_state["vs_v3_section"] = name
            try:
                st.query_params["section"] = name
            except Exception:
                pass
            st.rerun()

    releases = int((demo_cases["decision"].astype(str).str.lower() == "release").sum()) if "decision" in demo_cases else 0
    reviews = int(len(demo_cases) - releases)
    safe_rate = releases / max(1, len(demo_cases))
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='vs-card'><div class='vs-muted'>{ui('Active cases','活跃样本')}</div><div class='vs-kpi'>{len(demo_cases)}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='vs-card'><div class='vs-muted'>{ui('Auto-release','自动发布')}</div><div class='vs-kpi'>{releases}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='vs-card'><div class='vs-muted'>{ui('Review queue','复核队列')}</div><div class='vs-kpi'>{reviews}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='vs-card'><div class='vs-muted'>{ui('Release coverage','发布覆盖率')}</div><div class='vs-kpi'>{safe_rate:.0%}</div></div>", unsafe_allow_html=True)

    section = st.session_state["vs_v3_section"]

    def render_case_cards() -> None:
        st.subheader(ui("Ward / Case Overview", "病例与证据总览"))
        st.caption(ui("Every card is clickable. Released cases show HR; review cases hide HR and expose the reason.", "每张卡都可点击。发布病例显示心率；复核病例隐藏心率并显示原因。"))
        for row_start in range(0, len(demo_cases), 4):
            cols = st.columns(4)
            for j, (_, row) in enumerate(demo_cases.iloc[row_start : row_start + 4].iterrows()):
                idx = row_start + j
                kind = row_kind(row)
                conf = confidence(row, kind)
                bed = case_display_id(idx)
                hr_text = candidate_hr(row) if kind == "release" else "--"
                css = "vs-good" if kind == "release" else ("vs-bad" if kind == "urgent" else "vs-warn")
                with cols[j]:
                    st.markdown(
                        f"<div class='vs-card'><span class='vs-pill'>{status_text(kind)}</span> <span class='vs-muted'>{clean(row.get('dataset'),'unknown')}</span>"
                        f"<h3>{bed}</h3><div class='vs-kpi {css}'>{hr_text}<span style='font-size:0.8rem'> bpm</span></div>"
                        f"<div class='vs-muted'>{clean(row.get('selected_region'),'ROI')} / {clean(row.get('selected_method'),'method')} | conf {conf:.0%}</div></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(ui(f"Focus {bed}", f"聚焦 {bed}"), key=f"focus_{idx}", use_container_width=True):
                        st.session_state["vs_v3_focus_idx"] = idx
                        st.session_state["vs_v3_focus_select"] = focus_labels[idx]
                        st.rerun()

    def render_detail() -> None:
        idx = min(int(st.session_state.get("vs_v3_focus_idx", 0)), len(demo_cases) - 1)
        row = demo_cases.iloc[idx]
        kind = row_kind(row)
        payload = build_payload(row, idx)
        left, mid, right = st.columns([1.15, 1, 0.85])
        with left:
            st.subheader(f"{payload['bed_id']} · {payload['dataset']}")
            st.caption(f"{payload['case_id']} | {payload['sample_id']} | {status_text(kind)}")
            st.markdown(f"**{ui('Decision reason','决策理由')}**: {payload['review_reason']}")
            st.markdown(f"**{ui('Evidence summary','证据摘要')}**: {payload['evidence_summary']}")
            st.markdown(f"**{ui('Claim boundary','证据边界')}**: {payload['claim_boundary']}")
        with mid:
            st.subheader(ui("Candidate Evidence", "候选证据"))
            metric_cols = st.columns(3)
            metric_cols[0].metric("HR", candidate_hr(row), "bpm")
            snr = payload["selected_snr_db"]
            metric_cols[1].metric("SNR", "--" if snr is None else f"{snr:.1f}")
            metric_cols[2].metric("Conf", f"{payload['confidence']:.0%}")
            candidates = payload.get("competing_candidates", [])
            if isinstance(candidates, list) and candidates:
                st.dataframe(pd.DataFrame(candidates).head(5), use_container_width=True, hide_index=True)
            else:
                st.info(ui("No candidate table in this artifact.", "该样本没有候选表。"))
        with right:
            st.subheader(ui("Actions", "操作"))
            report = report_markdown(row, idx)
            st.download_button(ui("Export Report", "导出报告"), report.encode("utf-8"), file_name=f"vitalsight_{payload['bed_id']}_report.md", mime="text/markdown", use_container_width=True)
            if st.button(ui("Add to Review", "加入复核"), use_container_width=True, key="add_review"):
                queue = st.session_state["vs_v3_review_queue"]
                if payload["case_id"] not in queue:
                    queue.append(payload["case_id"])
                st.session_state["vs_v3_last_action"] = ui("Added to review queue.", "已加入复核队列。")
            if st.button(ui("Add Note", "添加备注"), use_container_width=True, key="add_note"):
                st.session_state["vs_v3_note_open"] = True
            note = st.text_area(ui("Reviewer note", "复核备注"), value=st.session_state["vs_v3_notes"].get(payload["case_id"], ""), key="note_text")
            if st.button(ui("Save Note", "保存备注"), use_container_width=True):
                st.session_state["vs_v3_notes"][payload["case_id"]] = note
                st.session_state["vs_v3_last_action"] = ui("Note saved.", "备注已保存。")
            if st.session_state.get("vs_v3_last_action"):
                st.success(st.session_state["vs_v3_last_action"])
            with st.expander(ui("API payload", "API 载荷"), expanded=False):
                render_json_block(payload)
                st.download_button("JSON", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{payload['bed_id']}_payload.json", mime="application/json", use_container_width=True)

    if section == "Command Center":
        render_case_cards()
        render_detail()
    elif section == "Live Scan":
        st.subheader(ui("Live Scan / Upload", "实时扫描 / 上传"))
        st.caption(ui("Use a built-in validated case or upload a video for the same evidence workflow.", "可运行内置验证样本，或上传视频进入同一证据流程。"))
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button(ui("Run Built-in Sample", "运行内置样例"), use_container_width=True):
                st.session_state["vs_v3_scan_result"] = build_payload(current_row, focus_idx)
            uploaded = st.file_uploader(ui("Upload adult face video", "上传成人面部视频"), type=["mp4", "mov", "avi", "mkv"])
            if uploaded is not None:
                st.session_state["vs_v3_scan_result"] = {
                    "file_name": uploaded.name,
                    "bytes": uploaded.size,
                    "status": "uploaded",
                    "next_step": "Run Face Mesh ROI, dense candidates, selector, and release gate in backend.",
                }
        with c2:
            st.subheader(ui("Current scan result", "当前扫描结果"))
            render_json_block(st.session_state.get("vs_v3_scan_result") or build_payload(current_row, focus_idx))
    elif section == "Patients":
        st.subheader(ui("Patient / Case Registry", "患者 / 病例登记"))
        table = demo_cases.copy()
        table.insert(0, "bed_id", [case_display_id(i) for i in range(len(table))])
        table["display_decision"] = [status_text(row_kind(row)) for _, row in table.iterrows()]
        st.dataframe(table[[c for c in ["bed_id", "dataset", "sample_id", "display_decision", "selected_region", "selected_method", "selected_snr_db", "review_reason"] if c in table.columns]], use_container_width=True, hide_index=True)
        st.download_button(ui("Export Registry CSV", "导出病例 CSV"), table.to_csv(index=False).encode("utf-8-sig"), "vitalsight_case_registry.csv", "text/csv", use_container_width=True)
    elif section == "Alerts":
        st.subheader(ui("Alert & Review Queue", "告警与复核队列"))
        alert_rows = []
        for idx, row in demo_cases.iterrows():
            kind = row_kind(row)
            if kind != "release":
                alert_rows.append({"bed_id": case_display_id(idx), "dataset": clean(row.get("dataset")), "status": status_text(kind), "reason": reason_text(row, kind), "candidate_hr": candidate_hr(row)})
        alerts = pd.DataFrame(alert_rows)
        st.dataframe(alerts, use_container_width=True, hide_index=True)
        if not alerts.empty:
            st.download_button(ui("Export Review Queue", "导出复核队列"), alerts.to_csv(index=False).encode("utf-8-sig"), "vitalsight_review_queue.csv", "text/csv", use_container_width=True)
    elif section == "Route-MoE":
        st.subheader(ui("Route-MoE Evidence", "Route-MoE 证据"))
        st.caption(ui("This is the algorithmic evidence layer: dense patch candidate pool, temporal selector, and OOF risk gate.", "这里展示算法证据层：dense patch 候选池、temporal selector 与 OOF risk gate。"))
        if not route_metrics.empty:
            st.dataframe(route_metrics, use_container_width=True, hide_index=True)
        if not policy.empty:
            st.markdown(ui("### Product policy", "### 产品策略"))
            st.dataframe(policy, use_container_width=True, hide_index=True)
        render_json_block({"stats": stats, "qa": qa})
    elif section == "Reports":
        st.subheader(ui("Report Center", "报告中心"))
        st.caption(ui("Generate case-level reports and manuscript-facing evidence exports.", "生成病例级报告和论文证据导出。"))
        report = report_markdown(current_row, focus_idx)
        st.markdown(report)
        st.download_button(ui("Download Current Report", "下载当前报告"), report.encode("utf-8"), file_name="vitalsight_current_report.md", mime="text/markdown", use_container_width=True)
        if not report_cards.empty:
            st.dataframe(report_cards, use_container_width=True, hide_index=True)
    elif section == "Integrations":
        st.subheader(ui("System Integrations", "系统集成"))
        st.caption(ui("Preview REST-style payloads for dashboard, review queue, and downstream records.", "预览仪表盘、复核队列和下游系统使用的 REST 风格载荷。"))
        payload = build_payload(current_row, focus_idx)
        endpoint = st.selectbox(ui("Endpoint", "接口"), ["/api/v1/hr/evidence", "/api/v1/review/queue", "/api/v1/report/export"])
        if st.button(ui("Simulate POST", "模拟 POST"), use_container_width=True):
            st.session_state["vs_v3_webhook_log"].append({"endpoint": endpoint, "case_id": payload["case_id"], "time": datetime.now().isoformat(timespec="seconds")})
        render_json_block({"endpoint": endpoint, "payload": payload, "webhook_log": st.session_state["vs_v3_webhook_log"]})
def main() -> None:
    inject_style()
    mode = sidebar_mode()
    adult_hr_product_console_v3(mode)


if __name__ == "__main__":
    main()
