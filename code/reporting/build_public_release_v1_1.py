#!/usr/bin/env python3
"""Build the v1.1.0 software and group-summary release from explicit file lists.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
Original inputs are never changed. Refuses an existing output directory.
"""
import argparse, collections, csv, datetime, hashlib, io, json, re, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = ROOT / 'submissions/hbm_release_v1.1.0_candidate'
HISTORICAL_FILES = ['.gitignore', '.zenodo.json', 'AGGREGATION_NOTES.json', 'CITATION.cff', 'COLUMN_DICTIONARY.json', 'FILE_DICTIONARY.csv', 'LICENSE.txt', 'LICENSES/CC-BY-4.0.txt', 'LICENSES/MIT.txt', 'README.txt', 'RELEASE_STATUS.txt', 'SHA256_manifest.json', 'code/common/srtt_paths.py', 'code/glm_volume/100_run_fir_analysis.py', 'code/glm_volume/101_summarize_fir_analysis.py', 'code/glm_volume/102_validate_extended_analyses.py', 'code/glm_volume/103_make_censored_roi_figure.py', 'code/glm_volume/90_run_float32_eventrelated.py', 'code/glm_volume/91_run_float32_group_inference.py', 'code/glm_volume/92_make_brain_figures.py', 'code/glm_volume/93_audit_roi_provenance.py', 'code/glm_volume/94_check_dicom_demographics.py', 'code/glm_volume/95_run_motion_filter_sensitivity.py', 'code/glm_volume/96_summarize_motion_filter_sensitivity.py', 'code/glm_volume/97_audit_nilearn_diagnostics.py', 'code/glm_volume/98_run_model_alternatives.py', 'code/glm_volume/99_summarize_model_alternatives.py', 'code/glm_volume/_layout.py', 'code/glm_volume/validate_manuscript.py', 'code/replication/config/rois.txt', 'code/replication/requirements-nilearn.txt', 'code/reporting/analyze_block_errors.py', 'code/reporting/augment_roi_reporting.py', 'code/reporting/build_pdfs.py', 'code/reporting/build_public_release.py', 'code/reporting/coverage_significant_table.py', 'code/reporting/fit_error_mixed_models.R', 'code/reporting/fit_fmri_rt_lme.R', 'code/reporting/make_error_rate_figure.py', 'code/reporting/make_fmri_rt_figure.py', 'code/reporting/make_reporting_tables.py', 'code/reporting/polish_reporting_figures.py', 'code/reporting/prepare_fmri_rt.R', 'code/reporting/prepare_workbench_wholebrain.py', 'code/reporting/render_workbench_wholebrain.py', 'code/reporting/reporting_spec.json', 'code/reporting/summarize_error_mixed_models.py', 'code/reporting/summarize_rt_lme.py', 'code/reporting/validate_reporting.py', 'code/reporting/verify_public_summaries.py', 'environment/error_R_session_info.txt', 'environment/python_environment.freeze.txt', 'environment/rt_R_session_info.txt', 'results/behavioral/block_errors_20261013/group_block_error_rates.csv', 'results/behavioral/block_errors_20261013/group_run_error_rates.csv', 'results/behavioral/block_errors_20261013/mixed_model/binomial_cell_estimates.csv', 'results/behavioral/block_errors_20261013/mixed_model/binomial_contrasts.csv', 'results/behavioral/block_errors_20261013/mixed_model/binomial_fixed_coefficients.csv', 'results/behavioral/block_errors_20261013/mixed_model/binomial_fixed_covariance_bootstrap.csv', 'results/behavioral/block_errors_20261013/mixed_model/cell_metadata.csv', 'results/behavioral/block_errors_20261013/mixed_model/contrast_weights.csv', 'results/behavioral/block_errors_20261013/mixed_model/gaussian_cell_estimates.csv', 'results/behavioral/block_errors_20261013/mixed_model/gaussian_contrasts.csv', 'results/behavioral/block_errors_20261013/mixed_model/gaussian_fixed_coefficients.csv', 'results/behavioral/block_errors_20261013/mixed_model/gaussian_fixed_covariance_bootstrap.csv', 'results/behavioral/block_errors_20261013/mixed_model/run_condition_wald.csv', 'results/behavioral/block_errors_20261013/paired_contrasts.csv', 'results/behavioral/fmri_rt_20261012/group_cycles.csv', 'results/behavioral/fmri_rt_20261012/learning_benefit_summary.csv', 'results/behavioral/fmri_rt_20261012/lme/cell_metadata.csv', 'results/behavioral/fmri_rt_20261012/lme/min100_cell_estimates.csv', 'results/behavioral/fmri_rt_20261012/lme/min100_contrast_weights.csv', 'results/behavioral/fmri_rt_20261012/lme/min100_contrasts.csv', 'results/behavioral/fmri_rt_20261012/lme/min100_fixed_coefficients.csv', 'results/behavioral/fmri_rt_20261012/lme/min100_fixed_covariance.csv', 'results/behavioral/fmri_rt_20261012/lme/min100_run_condition_wald.csv', 'results/behavioral/fmri_rt_20261012/lme/primary_cell_estimates.csv', 'results/behavioral/fmri_rt_20261012/lme/primary_contrast_weights.csv', 'results/behavioral/fmri_rt_20261012/lme/primary_contrasts.csv', 'results/behavioral/fmri_rt_20261012/lme/primary_fixed_coefficients.csv', 'results/behavioral/fmri_rt_20261012/lme/primary_fixed_covariance.csv', 'results/behavioral/fmri_rt_20261012/lme/primary_run_condition_wald.csv', 'results/behavioral/fmri_rt_20261012/preprocessing_summary.csv', 'results/behavioral/fmri_rt_20261012/run_summary.csv', 'results/glm_volume/reporting_20261012/all_model_roi_results.csv', 'results/glm_volume/reporting_20261012/all_roi_condition_profiles.pdf', 'results/glm_volume/reporting_20261012/analysis_spec.json', 'results/glm_volume/reporting_20261012/direct_stage_comparisons.csv', 'results/glm_volume/reporting_20261012/model_stability.csv', 'results/glm_volume/reporting_20261012/model_stage_summary.csv', 'results/glm_volume/reporting_20261012/phase_component_results.csv', 'results/glm_volume/reporting_20261012/reported_roi_results.csv', 'results/glm_volume/reporting_20261012/roi_coverage_summary.csv', 'results/glm_volume/whole_brain/components_peaks.csv', 'tables/observed_significant_rois.csv', 'tables/observed_significant_rois.json', 'tables/observed_significant_rois.tex']
LEDGER_SOURCES = ['docs/t9/demographics_n22.json', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/cell_event_counts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/counts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/descriptive_probability_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/glmm_status.json', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/position_rates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/error_models/matched_cell_estimates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/error_models/matched_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/error_models/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/response_log_audit/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/response_log_audit/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_comparison/matched_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_cell_estimates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_run_condition_wald.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_status.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_cell_estimates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_run_condition_wald.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_status.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_cell_estimates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_run_condition_wald.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_status.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_cell_estimates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_run_condition_wald.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_status.json', 'results/behavioral/reanalysis_20261018_updated_n22/source_replay/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/source_replay/preprocessing_summary.csv', 'results/behavioral/reanalysis_20261018_updated_n22/source_replay/recovered_learner_test.csv', 'results/behavioral/reanalysis_20261018_updated_n22/source_replay/validation.json', 'results/glm_volume/reanalysis_20261018/T3_censor_reconstruction/REPORT.md', 'results/glm_volume/reanalysis_20261018/T3_censor_reconstruction/run_censor_counts.csv', 'results/glm_volume/reanalysis_20261018/T3_censor_reconstruction/validation.json', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/REPORT.md', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/changed_event_segments.csv', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/event_inventory.csv', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/native_clock_checks.csv', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/anatomical_summary.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/annotated_components.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/atlas_overlap.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/original_roi_overlap.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/design_diagnostics/median_diagnostics_by_run.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/design_diagnostics/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/motion_quality/cohort_motion_summary.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/direct_phase_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/phase_component_changes.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/phase_component_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/stage_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/trend_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/within_run_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_right_accumbens_influence.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_stage_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise/permutation_summary.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise_inputs/validation.json', 'results/reanalysis_20261018_updated_n22/SUMMARY.md']
ADDITIONAL_FILES = ['results/behavioral/reanalysis_20261018_updated_n22/anticipation/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/anticipation/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/error_models/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/error_models/matched_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/error_models/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/response_log_audit/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/response_log_audit/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_comparison/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/rt_comparison/matched_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_comparison/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/analysis_spec.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_contrast_weights.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_fixed_coefficients.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/min100_fixed_covariance.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_contrast_weights.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_fixed_coefficients.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/primary_fixed_covariance.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/analysis_spec.json', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_contrast_weights.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_fixed_coefficients.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/min100_fixed_covariance.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_cell_estimates.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_contrast_weights.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_contrasts.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_fixed_coefficients.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/primary_fixed_covariance.csv', 'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/validation.json', 'results/behavioral/reanalysis_20261018_updated_n22/source_replay/REPORT.md', 'results/behavioral/reanalysis_20261018_updated_n22/source_replay/validation.json', 'results/glm_volume/reanalysis_20261018/T3_censor_reconstruction/REPORT.md', 'results/glm_volume/reanalysis_20261018/T3_censor_reconstruction/validation.json', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/REPORT.md', 'results/glm_volume/reanalysis_20261018/T3_timing_repair/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N20_reference_axial_page1.png', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N20_reference_axial_page2.png', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_censored_axial_page1.png', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_censored_axial_page2.png', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_reference_axial_page1.png', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_reference_axial_page2.png', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/design_diagnostics/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/design_diagnostics/median_diagnostics_by_run.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/design_diagnostics/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/figures/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/figures/condition_did_components.pdf', 'results/glm_volume/reanalysis_20261018_updated_n22/figures/condition_did_components.png', 'results/glm_volume/reanalysis_20261018_updated_n22/figures/sequence_trajectories.pdf', 'results/glm_volume/reanalysis_20261018_updated_n22/figures/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/motion_quality/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/motion_quality/cohort_motion_summary.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/motion_quality/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/direct_phase_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/phase_component_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/stage_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/trend_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting/within_run_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_N22_specifications.pdf', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_direct_phase_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_right_accumbens_influence.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_stage_results.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise/permutation_summary.csv', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise/validation.json', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise_inputs/REPORT.md', 'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise_inputs/validation.json', 'results/reanalysis_20261018_updated_n22/REPRODUCE.md', 'results/reanalysis_20261018_updated_n22/SUMMARY.md', 'results/reanalysis_20261018_updated_n22/validation.json']
CODE_FILES = ['code/reanalysis/README.md', 'code/reanalysis/audit_alternatives.py', 'code/reanalysis/audit_recovered_events.py', 'code/reanalysis/audit_response_logs.py', 'code/reanalysis/behavior.py', 'code/reanalysis/build_censored_pooled.py', 'code/reanalysis/common.py', 'code/reanalysis/discover_extended_sources.py', 'code/reanalysis/finalize.py', 'code/reanalysis/finalize_updated_behavior.py', 'code/reanalysis/fit_anticipation.R', 'code/reanalysis/polish_figures.py', 'code/reanalysis/postprocess_updated_tfce.py', 'code/reanalysis/prepare_n22_inputs.py', 'code/reanalysis/replay_updated_behavior.R', 'code/reanalysis/report_updated_analysis.py', 'code/reanalysis/run_refits.py', 'code/reanalysis/run_saved.py', 'code/reanalysis/run_shared_rt.py', 'code/reanalysis/run_tfce.py', 'code/reanalysis/run_tfce_censored_float32.py', 'code/reanalysis/run_tfce_float32.py', 'code/reanalysis/run_updated_permutations.py', 'code/reanalysis/summarize_sensitivity.py', 'code/reanalysis/tfce_backend.py', 'code/reanalysis/tfce_direct.py', 'code/reanalysis/tfce_sparse.cpp', 'code/reanalysis/tfce_sparse.py', 'code/reanalysis/tfce_union.cpp', 'code/reanalysis/tfce_union.py', 'code/reanalysis/update_t7_shared_rt.py', 'code/reanalysis/updated_anatomical_display.py', 'code/reanalysis/updated_behavior_results.py', 'code/reanalysis/updated_common.py', 'code/reanalysis/updated_design_figures.py', 'code/reanalysis/updated_error_glmm.R', 'code/reanalysis/updated_first_level.py', 'code/reanalysis/updated_motion.py', 'code/reanalysis/updated_results.py', 'code/reanalysis/updated_rt_lme.R', 'code/reanalysis/updated_targeted_sensitivity.py', 'code/reanalysis/updated_tfce.py', 'code/reanalysis/validate_direct_backend.py', 'code/reanalysis/validate_reanalysis.py', 'code/reanalysis/validate_sparse_backend.py', 'code/reanalysis/validate_updated_analysis.py', 'code/reanalysis/visualize_within_run.py', 'code/reporting/analyze_block_errors.py', 'code/reporting/augment_roi_reporting.py', 'code/reporting/build_pdfs.py', 'code/reporting/build_public_release.py', 'code/reporting/build_public_release_v1_1.py', 'code/reporting/coverage_significant_table.py', 'code/reporting/fit_error_mixed_models.R', 'code/reporting/fit_fmri_rt_lme.R', 'code/reporting/make_error_rate_figure.py', 'code/reporting/make_fmri_rt_figure.py', 'code/reporting/make_n22_manuscript_assets.py', 'code/reporting/make_reporting_tables.py', 'code/reporting/polish_reporting_figures.py', 'code/reporting/prepare_fmri_rt.R', 'code/reporting/prepare_workbench_wholebrain.py', 'code/reporting/render_workbench_wholebrain.py', 'code/reporting/reporting_spec.json', 'code/reporting/summarize_error_mixed_models.py', 'code/reporting/summarize_rt_lme.py', 'code/reporting/validate_n22_manuscript.py', 'code/reporting/validate_reporting.py', 'code/reporting/verify_public_summaries.py', 'code/reporting/verify_public_summaries_n22.py']
METADATA = {'README.txt','RELEASE_STATUS.txt','CITATION.cff','.zenodo.json','FILE_DICTIONARY.csv','COLUMN_DICTIONARY.json','AGGREGATION_NOTES.json','LICENSE.txt','SHA256_manifest.json'}
PRIVATE = re.compile(r'/'+r'(?:home|media)/[^\s\x60\"\'<>),;]+|'+r'sub-'+r'\d{2,3}\b|\b(?:acl'+r'-super|node'+r'37)\b|(?<![.\w-])0[0-2]\d\b')
FORBIDDEN_COLUMNS = {'subject','participant','participant_id','subject_id','sid','sub','omitted_subject','filename','file_path','path','original','repaired','raw_csv','raw_log'}
STAT_FIELDS = ['mean','sd','se','ci_low','ci_high','cohen_dz','t','df','p','estimate','probability']
CODE_DATA_EXCLUSIONS = {'code/reanalysis/audit_recovered_events.py': 'Contains observed trial timings linked to individual participants.', 'code/reanalysis/prepare_n22_inputs.py': 'Contains trigger and event timings linked to individual participants.', 'code/reanalysis/report_updated_analysis.py': 'Contains individual censor counts, learning-test outcomes and trial timings.', 'code/reanalysis/validate_updated_analysis.py': 'Contains individual censor counts used as expected values.', 'code/reanalysis/updated_behavior_results.py': 'Contains individual learning-test outcomes.', 'code/reanalysis/run_saved.py': 'Contains individual missing-response rates and inferred learning-test outcomes.', 'code/reanalysis/finalize.py': 'Contains inferred learning-test outcomes for individual participants.'}
FIGURE_REVIEW = {'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N20_reference_axial_page1.png': {'source_sha256': '8579cccf08af3292551bf2c71081834c0192ab1cc17f48a4a64e773952522566', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N20_reference_axial_page2.png': {'source_sha256': '2761479bf30d56707618930a56216df6042fc0fa118ed5f08c368e5cab4095db', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_censored_axial_page1.png': {'source_sha256': '0cda630b1c96185201c2c415f1ceaddeacaa1a8b293db8b232a1c852d5e1f060', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_censored_axial_page2.png': {'source_sha256': '72ddba406128c26ac872b58a0c238993a4956e0122f78aacbe54e6e969623ad9', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_reference_axial_page1.png': {'source_sha256': '55dfb5cb2322a546e1811006678c25af21eb427f1e91bbb642a1d1b6f968cde2', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/anatomical_display/N22_reference_axial_page2.png': {'source_sha256': 'b6226338bc26647ba043004b117fe8652b0201d8a402bfc98f6a80fac308ff4f', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/figures/condition_did_components.pdf': {'source_sha256': '44a90a8e8b962b56878ffa3ec89a839eec598ff04adb698a679319fec9c54b9a', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/figures/condition_did_components.png': {'source_sha256': '74a5c24dbbdc709e6dd4306ab3146b0fa68a43671fa3700f41f7ea55f77ccaa9', 'original_pages': 1, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/figures/sequence_trajectories.pdf': {'source_sha256': '2ae5d50b5d9436700b1f6abbf70a97258c2e255ae2e3a8c2aaf42662d71f16aa', 'original_pages': 18, 'visual_review': 'All pages inspected: group summaries only; no individual traces/labels.', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}, 'results/glm_volume/reanalysis_20261018_updated_n22/targeted_sensitivity/all_N22_specifications.pdf': {'source_sha256': 'c13147e716124dcadff35fab6f0837fc356059038cbcc5c892fc5e6382870972', 'original_pages': 48, 'visual_review': 'Group summaries only, excluding original page 8 (singleton estimates).', 'source_metadata': 'Matplotlib software/version and creation date or PNG dpi only; no private paths/user/host names.'}}
# Pseudonymous labels and input paths in code are retained; individual observations are excluded.


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(p, rows, columns):
    with p.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=columns);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dest',type=Path,default=DEFAULT_DEST)
    ap.add_argument('--refresh-owned-candidate',action='store_true',help='Rebuild the existing manifest-checked default output.')
    ap.add_argument('--release-date',required=True,help='Release date in YYYY-MM-DD form.')
    ap.add_argument('--doi',help='Verified version DOI for the post-release citation update; requires --commit.')
    ap.add_argument('--commit',help='Frozen 40-character Git commit for the post-release citation update; requires --doi.')
    args=ap.parse_args()
    try:
        release_date=datetime.date.fromisoformat(args.release_date).isoformat()
    except ValueError:ap.error('--release-date must be YYYY-MM-DD')
    if release_date!=args.release_date:ap.error('--release-date must be YYYY-MM-DD')
    if bool(args.doi)!=bool(args.commit):ap.error('--doi and --commit must be supplied together')
    if args.doi and not re.fullmatch(r'10\.\d{4,9}/[^\s<>]+',args.doi):ap.error('--doi must be a DOI beginning with 10.')
    if args.commit and not re.fullmatch(r'[0-9a-fA-F]{40}',args.commit):ap.error('--commit must be a full 40-character hexadecimal SHA')
    dest=args.dest.resolve()
    # Accept new/empty temporary destinations for independent rebuilds; protect the project.
    if dest!=DEFAULT_DEST and (dest.is_relative_to(ROOT) or not dest.is_relative_to(Path('/tmp')) or dest==Path('/tmp')):raise ValueError('Unowned output path')
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        if not args.refresh_owned_candidate or dest!=DEFAULT_DEST:raise FileExistsError('Preserve existing output; destination must not exist')
        manifest=json.loads((dest/'SHA256_manifest.json').read_text())
        expected={x['path'] for x in manifest}|{'SHA256_manifest.json'}
        actual={str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}
        assert expected==actual,'Refuse to overwrite unexpected candidate files'
        for x in manifest:
            p=dest/x['path'];assert p.resolve().is_relative_to(dest) and sha(p)==x['sha256'],'Candidate changed; preserve it'
        for rel in sorted(expected):(dest/rel).unlink()
        for p in sorted((x for x in dest.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):p.rmdir()
        dest.rmdir()
    dest.mkdir(parents=True,exist_ok=True)
    old=ROOT/'submissions/hbm_release_v1.0.0'
    notes=json.loads((old/'AGGREGATION_NOTES.json').read_text())
    inventory=[]; excluded=[]; decisions={}; suppressed={}; included=set()
    def save(rel, data):
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);included.add(rel)
    def exclude(rel, reason, historical=False):
        excluded.append(dict(path=rel,reason=reason,historical=historical));decisions[rel]=('withhold',reason)
    # Preserve historical nonmetadata files verbatim.
    for rel in HISTORICAL_FILES:
        if rel in METADATA:continue
        p=old/rel
        save(rel,p.read_bytes())
    ledger=json.loads((ROOT/'docs/t9/facts_n22.json').read_text())
    for rel in sorted(set(LEDGER_SOURCES+ADDITIONAL_FILES+CODE_FILES)):
        if rel in included:continue # Historical code must remain byte-identical.
        p=ROOT/rel
        if not p.exists():exclude(rel,'Source unavailable');continue
        if 'influence' in p.name or 'learner_test' in p.name:
            exclude(rel,'Contains individual observations or values from which individual observations can be recovered.');continue
        if p.suffix in {'.pdf','.png'}:
            # All pages reviewed; omit singleton page 8.
            assert rel in FIGURE_REVIEW and sha(p)==FIGURE_REVIEW[rel]['source_sha256'],'Figure changed since visual review'
            if p.name=='all_N22_specifications.pdf':
                target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run(['gs','-q','-dBATCH','-dNOPAUSE','-dSAFER','-dOmitInfoDate=true','-dOmitID=true','-sDEVICE=pdfwrite','-sPageList=1-7,9-48','-sOutputFile='+str(target),str(p)],check=True,capture_output=True)
                # Ghostscript's XMP UUID includes runtime/machine identity; fixed-length replacement preserves xref offsets.
                raw_pdf=target.read_bytes()
                normalized,n_ids=re.subn(rb'uuid:[0-9a-fA-F-]{36}',b'uuid:00000000-0000-0000-0000-000000000000',raw_pdf)
                assert n_ids>=1,'Expected generated XMP UUID'
                target.write_bytes(normalized);included.add(rel)
                notes.append(dict(path=rel,source_sha256=sha(p),release_sha256=sha(target),operation='page suppression and generated metadata normalization',removed_original_pages=[8],source_pages=48,release_pages=47,removed_metadata_fields=['generated XMP DocumentID runtime/machine UUID','generated Info dates and trailer ID omitted'],reason='Original page 8 displays N=1 LeftFrontalPole estimates; remove all singleton points without refitting. Normalize generated metadata for privacy and byte reproducibility.'))
                decisions[rel]=('include with suppression/redaction','Original page 8 (N=1) omitted; 47 reviewed group/empty pages remain')
            else:
                save(rel,p.read_bytes());decisions[rel]=('include verbatim','All pages/group plots visually inspected; metadata has no private paths/user/host names')
            continue
        raw=p.read_text()
        if rel.startswith('code/'):
            if rel in CODE_DATA_EXCLUSIONS:exclude(rel,CODE_DATA_EXCLUSIONS[rel]);continue
            save(rel,p.read_bytes());decisions[rel]=('include verbatim','Source code with input labels and paths; no embedded individual observations.');continue
        if p.suffix=='.csv':
            frame=pd.read_csv(p)
            bad=set(frame.columns.str.lower())&FORBIDDEN_COLUMNS
            if bad:exclude(rel,'Contains individual records or restricted input paths in columns: '+', '.join(sorted(bad)));continue
            if PRIVATE.search(raw):exclude(rel,'Identifiers/private paths in CSV values');continue
            mask=(pd.to_numeric(frame['n'],errors='coerce')<2) if 'n' in frame else pd.Series(False,index=frame.index)
            fields=[c for c in STAT_FIELDS if c in frame]
            count=int(frame.loc[mask,fields].notna().sum().sum())
            if count:
                frame.loc[mask,fields]=np.nan
                save(rel,frame.to_csv(index=False).encode());suppressed[rel]=(frame,mask,fields)
                notes.append(dict(path=rel,source_sha256=sha(p),operation='suppression',rows=int(mask.sum()),cells_suppressed=count,columns=fields,reason='N<2; retain N, planned slots, BH q; blanks mean missing.'))
                decisions[rel]=('include with suppression/redaction','Estimates from fewer than two participants are blank; sample sizes and adjusted probabilities are retained.')
            else:save(rel,p.read_bytes());decisions[rel]=('include verbatim','Group summaries.')
            inventory.append(dict(path=rel,rows=len(frame),units='Source-defined: ROI coefficient units; RT natural log milliseconds / milliseconds or log differences; proportions, counts or dimensionless statistics. See task report and column dictionary.',description='Group summary for the primary 22-participant analysis and overlapping 20-participant sensitivity analysis.'))
        elif rel=='docs/t9/demographics_n22.json':
            j=json.loads(raw);safe={k:j[k] for k in ['n22','n20','n_dicom_samples','all_samples_match_tsv','status','purpose']}
            save(rel,(json.dumps(safe,indent=2)+'\n').encode())
            notes.append(dict(path=rel,source_sha256=sha(p),operation='redaction',removed_keys=sorted(set(j)-set(safe)),reason='Keep existing cohort summaries only; remove individual demographics, identifiers, source paths and header samples.'))
            decisions[rel]=('include with suppression/redaction','Group demographics; individual records and scanner-header details are omitted.')
        elif p.suffix=='.json':
            # Remove whole object members containing private paths/identifiers, recording exact JSON pointers.
            removed=[]
            def clean(x, pointer=''):
                if isinstance(x,dict):
                    out={}
                    for k,v in x.items():
                        loc=pointer+'/'+str(k).replace('~','~0').replace('/','~1')
                        if PRIVATE.search(str(k)) or (not isinstance(v,(dict,list)) and PRIVATE.search(str(v))):removed.append(loc)
                        else:out[k]=clean(v,loc)
                    return out
                if isinstance(x,list):
                    out=[]
                    for i,v in enumerate(x):
                        loc=pointer+'/'+str(i)
                        if PRIVATE.search(json.dumps(v)):removed.append(loc)
                        else:out.append(clean(v,loc))
                    return out
                return x
            safe=clean(json.loads(raw))
            # Output hashes are historical provenance, not checksums of redacted release outputs.
            save(rel,(json.dumps(safe,indent=2)+'\n').encode() if removed else p.read_bytes())
            if removed:notes.append(dict(path=rel,source_sha256=sha(p),operation='redaction',removed_json_pointers=removed,reason='Remove participant identifiers/private paths; original task hashes remain provenance only.'))
            decisions[rel]=('include with suppression/redaction' if removed else 'include verbatim','Saved analysis specification or validation record. Source hashes describe the original files; release hashes are in SHA256_manifest.json.')
        elif p.suffix in {'.md','.txt'}:
            lines=raw.splitlines(keepends=True); redacted=[]
            for i,line in enumerate(lines):
                if PRIVATE.search(line):redacted.append(i+1);lines[i]='[Line withheld: participant identifier or private path.]\n'
            save(rel,''.join(lines).encode())
            if redacted:notes.append(dict(path=rel,source_sha256=sha(p),operation='redaction',line_numbers=redacted,reason='Entire lines with identifiers, host names or private absolute paths withheld.'))
            decisions[rel]=('include with suppression/redaction' if redacted else 'include verbatim','Saved analysis report; see README for the current analysis hierarchy.')
        else:exclude(rel,'Unreviewed file type')
    coverage=[]
    for rel in LEDGER_SOURCES:
        facts=[x for x in ledger if x['source']==rel]
        decision,reason=decisions[rel];verified=0
        if decision!='withhold':
            p=dest/rel
            if p.suffix=='.csv':
                df=pd.read_csv(p)
                indexes={}
                for fact in facts:
                    keys=fact['row_key'];col=fact['column']
                    if not isinstance(col,str) or col not in df or not set(keys)<=set(df):continue
                    names=tuple(sorted(keys))
                    if names not in indexes:
                        ix={}
                        for row in df.to_dict('records'):
                            identity=tuple(str(row[k]) for k in names)
                            ix.setdefault(identity,[]).append(row)
                        indexes[names]=ix
                    rows=indexes[names].get(tuple(str(keys[k]) for k in names),[])
                    if len(rows)==1:
                        actual=rows[0][col];expected=fact['value']
                        if pd.isna(actual):ok=expected is None
                        elif isinstance(expected,(int,float)):ok=np.isclose(float(actual),expected,rtol=1e-10,atol=1e-13)
                        else:ok=str(actual)==str(expected)
                        verified+=int(ok)
            elif p.suffix=='.json':
                obj=json.loads(p.read_text())
                for fact in facts:
                    try:
                        x=obj
                        for key in fact['row_key']['json_pointer'].strip('/').split('/'):
                            if key:x=x[int(key)] if isinstance(x,list) else x[key]
                        # Ledger column may label a scalar pointer rather than a member.
                        if isinstance(x,dict):x=x[fact['column']]
                        verified+=int(x==fact['value'])
                    except (KeyError,TypeError,ValueError,IndexError):pass
            # Prose line numbers shift after redaction; conservatively withhold ledger verification.
        coverage.append(dict(source=rel,source_sha256=sha(ROOT/rel),decision=decision,reason=reason,ledger_entries=len(facts),verifiable_entries=verified,withheld_entries=len(facts)-verified,release_path=rel if decision!='withhold' else ''))
    write_csv(dest/'RELEASE_COVERAGE.csv',coverage,list(coverage[0]))
    dump(dest/'RELEASE_EXCLUSIONS.json',excluded)
    dump(dest/'AGGREGATION_NOTES.json',notes)
    code_review=[]
    for rel in sorted(set(CODE_FILES)|{x for x in HISTORICAL_FILES if x.startswith('code/')}):
        source=old/rel if rel in HISTORICAL_FILES else ROOT/rel
        code_review.append(dict(path=rel,source_sha256=sha(source),credentials_or_tokens='none found',embedded_participant_data='present' if rel in CODE_DATA_EXCLUSIONS else 'none found',decision='withhold' if rel in CODE_DATA_EXCLUSIONS else 'include verbatim',review=CODE_DATA_EXCLUSIONS.get(rel,'Source code contains no credentials or individual observations. Subject labels and paths identify restricted inputs.')))
    write_csv(dest/'CODE_REVIEW.csv',code_review,list(code_review[0]))
    dump(dest/'FIGURE_REVIEW.json',{rel:{**record,'release_sha256':sha(dest/rel),'release_pages':47 if 'all_N22_specifications' in rel else record['original_pages']} for rel,record in FIGURE_REVIEW.items()})
    # Explicit arithmetic contracts, supported by the released task reports/validation counts.
    contracts=[]
    for rel in sorted(included):
        if '/roi_reporting/' in rel or '/targeted_sensitivity/' in rel:
            if rel.endswith('.csv'):
                frame=pd.read_csv(dest/rel)
                if not {'mean','se','t','n','df'}<=set(frame):continue
                families=[]
                if 'q' in frame:
                    group=[x for x in ['cohort','model','policy','stage','contrast'] if x in frame]
                    families.append(dict(q='q',p='p',group=group,slots=48))
                for q,extra,slots in [('q_three_pairs_144',[],144),('q_within_pair_48',['pair'],48)]:
                    if q in frame:families.append(dict(q=q,p='p',group=[x for x in ['cohort','model','policy']+extra if x in frame],slots=slots))
                contracts.append(dict(path=rel,estimate='mean',se='se',t='t',df='df',low='ci_low',high='ci_high',families=families,suppression_n='n'))
    for rel in sorted(included):
        if 'reanalysis_20261018_updated_n22' not in rel or not rel.endswith('.csv') or '/rt_' not in rel and '/error_models/' not in rel:continue
        f=pd.read_csv(dest/rel)
        if not {'t','df'}<=set(f):continue
        est='log_difference' if 'log_difference' in f else 'estimate'
        se='se' if 'se' in f else 'se_bootstrap'
        lo='log_ci_low' if 'log_ci_low' in f else 'ci_low';hi='log_ci_high' if 'log_ci_high' in f else 'ci_high'
        if not {est,se,lo,hi}<=set(f):continue
        fam=[]
        if 'q_bh_24' in f:fam=[dict(q='q_bh_24',p='p_two_sided',group=[x for x in ['cohort','policy','model'] if x in f],slots=24)]
        contracts.append(dict(path=rel,estimate=est,se=se,t='t',df='df',low=lo,high=hi,families=fam,ci_method='percentile bootstrap; not derived from SE' if '/error_models/' in rel else 't'))
    for rel in sorted(included):
        if '/rt_lme_' in rel and rel.endswith('_cell_estimates.csv') and 'reanalysis_20261018_updated_n22' in rel:
            contracts.append(dict(path=rel,estimate='log_rt',se='se',df='df',low='log_ci_low',high='log_ci_high',families=[],transforms=[dict(source='log_rt',target='geometric_rt_ms'),dict(source='log_ci_low',target='geometric_ci_low_ms'),dict(source='log_ci_high',target='geometric_ci_high_ms')]))
    dump(dest/'AGGREGATE_CHECK_SPEC.json',dict(description='Checks only supplied aggregates, no participant model fitting. ROI BH 48 per contrast and 144 direct slots; see saved REPORT/validation records. Missing p becomes 1 for planned BH slots.',contracts=contracts))
    release_url='https://github.com/audachang/ids-srtt-hbm/releases/tag/v1.1.0'
    original_cff=(old/'CITATION.cff').read_text()
    original_cff='\n'.join(line for line in original_cff.splitlines() if not line.startswith(('doi:','commit:')))+'\n'
    original_cff=re.sub(r'^message:.*$', 'message: "Please cite the specific Zenodo version DOI linked from the GitHub release when available; otherwise cite the release URL."',original_cff,flags=re.M)
    original_cff=original_cff.replace('version: 1.0.0','version: 1.1.0').replace('date-released: 2026-10-17','date-released: '+release_date).replace('releases/tag/v1.0.0','releases/tag/v1.1.0')
    if args.doi:
        original_cff=re.sub(r'^message:.*$', 'message: "Please cite this software and group-summary release using its version-specific Zenodo DOI."',original_cff,flags=re.M)
        original_cff+='doi: "'+args.doi+'"\ncommit: "'+args.commit+'"\n'
    (dest/'CITATION.cff').write_text(original_cff)
    meta=json.loads((old/'.zenodo.json').read_text());meta.update(version='1.1.0',publication_date=release_date);meta.pop('doi',None)
    meta['keywords']=['region-of-interest analysis' if x=='coverage'+'-aware ROI' else x for x in meta['keywords']]
    meta['description']=('<p>Analysis and reporting source code, software-environment records, and group-level imaging and behavioral summaries for an interleaved dual-structure serial reaction time fMRI study of motor statistical learning and gradual sequence updating.</p>'
        '<p>Version 1.1.0 adds results for 22 participants. The uncensored reference model with at least 80% observed regional support is primary. The learner-selected 20-participant subsample and motion-censored models are sensitivity analyses; the subsample appears only in the supplement. The samples overlap. The analyses were not preregistered, and the reporting model was chosen after earlier analyses; see the article Methods. Earlier 20-participant products from version 1.0.0 are retained unchanged.</p>'
        '<p>The release includes group summaries, figures, source-file coverage, dictionaries, checksums and aggregate-verification instructions. Participant-level inputs, fitted models and values recoverable by leave-one-out arithmetic are withheld. Seven scripts containing individual observations are also withheld. Estimates from fewer than two participants are blank. Subject labels and filesystem paths in code refer to restricted inputs. The article table-generation script is included for reference and requires restricted inputs. Full model re-estimation requires approved data access and separate software and atlas resources.</p>'
        '<p>Original software and configuration under code/ are licensed under MIT. Group summaries, figures, documentation, environment records and descriptive metadata are licensed under CC BY 4.0. See LICENSE.txt and LICENSES/ for the file-specific license terms. Third-party resources retain their own terms.</p>')
    dump(dest/'.zenodo.json',meta)
    archival=('Archived version: v1.1.0.\nVersion DOI: https://doi.org/'+args.doi+'\nFrozen commit: '+args.commit+'\n') if args.doi else ('Archival version: v1.1.0. Its Zenodo version DOI will be linked from the GitHub\nrelease page after automatic archival completes.\n')
    status=('Version 1.1.0 published on '+release_date+'.\nFrozen commit: '+args.commit+'\nVersion DOI: https://doi.org/'+args.doi+'\n') if args.doi else ('Version 1.1.0 prepared on '+release_date+'.\nThe Zenodo version DOI will be linked from the GitHub release page after automatic archival completes.\n')
    (dest/'RELEASE_STATUS.txt').write_text(status+'GitHub release: '+release_url+'\nVersion 1.0.0 remains available at doi:10.5281/zenodo.23438462.\nOriginal software: MIT. Group summaries and documentation: CC BY 4.0.\nParticipant-level data are not distributed.\n')
    license_text=(old/'LICENSE.txt').read_text()
    license_text+='\nVersion 1.1.0: original software under code/ is licensed under MIT. Group summaries, figures, source descriptions, verification specifications, dictionaries and documentation are licensed under CC BY 4.0. Attribution should identify the authors and version 1.1.0 release at '+release_url+'.\n'
    (dest/'LICENSE.txt').write_text(license_text)
    cols=json.loads((old/'COLUMN_DICTIONARY.json').read_text());cols.update(log_rt='Fixed-effect natural log RT in milliseconds; exponentiation gives geometric milliseconds.',log_difference='Contrast of log RT; exponentiation gives geometric RT ratio.',model='Updated reference means uncensored reference (primary N22); censored and alternative specifications are sensitivities. Historical v1.0.0 labels have earlier post hoc reporting roles.',q='BH 48 planned ROI slots per cohort/model/policy/contrast or stage; missing p treated as 1. See AGGREGATE_CHECK_SPEC.json.',cohort='N22 primary; overlapping learner-selected N20 sensitivity, supplement only.',n='Eligible participant count for ROI inference; in cell_event_counts.csv n is trial opportunities, not participant count. N<2 estimate cells blank.')
    dump(dest/'COLUMN_DICTIONARY.json',cols)
    total=sum(x['ledger_entries'] for x in coverage);verified=sum(x['verifiable_entries'] for x in coverage)
    (dest/'README.txt').write_text(f'''IDS-SRTT HBM companion — v1.1.0 ({release_date})

Scope: analysis source code and group summaries only.
Public repository: https://github.com/audachang/ids-srtt-hbm
GitHub release: {release_url}
{archival}Version 1.0.0 remains available at doi:10.5281/zenodo.23438462.
Licenses: original software under code/ is MIT; group summaries, figures,
documentation and environment records are CC BY 4.0. See LICENSE.txt and LICENSES/.

Version 1.1.0 adds imaging and behavioral summaries for 22 participants,
group figures, source descriptions and verification instructions. The earlier
20-participant products from version 1.0.0 are retained unchanged.
The 22-participant analysis with the uncensored reference model and at least
80% observed regional support is primary. The learner-selected 20-participant
subsample and motion-censored models are sensitivity analyses. The subsample
appears only in the article supplement. The samples overlap and are not
independent groups. The analyses were not preregistered. The reporting model
was chosen after earlier analyses; see the article Methods.

Quick start (NumPy, pandas and SciPy required):
  python code/reporting/verify_public_summaries_n22.py .
Checks file checksums and inventory, t statistics, pointwise t intervals,
RT back-transformations, planned BH correction families and blank suppressed
cells. Bootstrap percentile intervals are checked for ordering. These checks
verify supplied aggregate outputs, not the underlying participant models.
Software-environment records from version 1.0.0 are retained for reference.

FILE_DICTIONARY.csv describes file units and contents. COLUMN_DICTIONARY.json
explains statistical columns. RELEASE_COVERAGE.csv lists each source file
behind the article's reported values and whether it is included, included
with suppression, or withheld. Of {total:,} recorded values, {verified:,} can be
checked against released files. Some values remain unavailable because their
sources contain individual observations or their text references changed
when identifying details were removed.

Estimates from fewer than two participants are blank; blank means missing,
not zero. Sample sizes, planned comparison slots and BH-adjusted probabilities
are retained. AGGREGATION_NOTES.json describes suppression and redaction.
Original source hashes in saved validation reports describe their source files;
SHA256_manifest.json gives release hashes for all files except itself.
FIGURE_REVIEW.json gives source and release hashes for all ten added figures.
AGGREGATION_NOTES.json records the specifications PDF, whose original page 8
was omitted because its estimates came from one participant.

Restricted inputs include raw/preprocessed MRI, participant maps, masks,
confounds and censor vectors, trial logs, individual summaries, random effects,
fitted models, bootstrap identities and participant input manifests.
Leave-one-out and influence tables are withheld because individual values
can be recovered from them. Seven scripts containing individual observations
are also withheld; RELEASE_EXCLUSIONS.json lists them and explains why.
Subject labels and filesystem paths in code refer to restricted inputs that
are not distributed. Subject-list configuration and data paths must be set
for an approved copy of the inputs. Separate software dependencies and
atlas/template resources are required under their own terms.

The article's table-generation script, make_n22_manuscript_assets.py, is
included for reference. It requires restricted source records, original inputs
with matching checksums and individual influence values, so it cannot run
unchanged using this public subset. Released group estimates can be used to
format tables. Full model re-estimation requires approved restricted inputs
and any excluded helper scripts needed by the selected analysis.

Requests for participant data go to the corresponding authors and depend on
participant consent, ethics approval and institutional requirements; access
is not guaranteed. No participant-level data access is granted by this release.
''')
    # Inventory every file, including historical code; no files are silently undescribed.
    detail={x['path']:x for x in inventory}
    for p in sorted(dest.rglob('*')):
        if not p.is_file():continue
        rel=str(p.relative_to(dest))
        if rel not in detail:detail[rel]=dict(path=rel,rows='',units='Source-defined / not applicable',description='')
    for rel,row in detail.items():
        if not row['description']:row['description']='Earlier 20-participant reporting from version 1.0.0; retained unchanged.' if rel in HISTORICAL_FILES and rel not in METADATA else 'Source code, documentation, validation record or descriptive metadata; see README.'
    detail['FILE_DICTIONARY.csv']=dict(path='FILE_DICTIONARY.csv',rows='',units='Not applicable',description='Complete file inventory.')
    detail['SHA256_manifest.json']=dict(path='SHA256_manifest.json',rows='',units='bytes / SHA-256',description='All release files except the manifest itself.')
    write_csv(dest/'FILE_DICTIONARY.csv',[detail[k] for k in sorted(detail)],['path','rows','units','description'])
    # All text is scanned; report/exclusion descriptions contain no actual private matches.
    for p in dest.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.py','.r','.cpp','.json','.csv','.md','.txt','.cff','.tex'}:
            if not str(p.relative_to(dest)).startswith('code/') and PRIVATE.search(p.read_text()):raise ValueError('Unredacted private content: '+str(p.relative_to(dest)))
    manifest=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert all(x['bytes']<100_000_000 for x in manifest)
    dump(dest/'SHA256_manifest.json',manifest)
    print(json.dumps(dict(files=len(manifest)+1,bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),ledger_entries=total,verifiable_entries=verified,excluded=len(excluded)),indent=2))

if __name__=='__main__':main()
