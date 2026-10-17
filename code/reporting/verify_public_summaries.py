#!/usr/bin/env python3
"""Verify released aggregate arithmetic; does not refit restricted-input models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10, equal_nan=True)


def check_estimates(frame):
    x = frame[(frame.n >= 2) & frame.sd.notna()]
    close(x.df, x.n - 1)
    close(x.se, x.sd / np.sqrt(x.n))
    close(x.t, x['mean'] / x.se)
    close(x.cohen_dz, x['mean'] / x.sd)
    close(x.p, 2 * stats.t.sf(abs(x.t), x.df))
    half = stats.t.ppf(.975, x.df) * x.se
    close(x.ci_low, x['mean'] - half)
    close(x.ci_high, x['mean'] + half)
    assert frame.loc[frame.n < 2, 'mean'].isna().all(), 'Singleton means must be suppressed'


def check_bh(frame, keys, column):
    for _, group in frame.groupby(keys, sort=False):
        expected = stats.false_discovery_control(group.p.fillna(1).to_numpy(), method='bh')
        close(group[column], expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root / 'SHA256_manifest.json').read_text())
    for row in manifest:
        file = root / row['path']
        assert file.stat().st_size == row['bytes'], row['path']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256'], row['path']
    base = root / 'results/glm_volume/reporting_20261012'
    all_models = pd.read_csv(base / 'all_model_roi_results.csv')
    assert len(all_models) == 6912
    check_estimates(all_models)
    check_bh(all_models, ['model', 'policy', 'stage'], 'q_within_contrast')
    check_bh(all_models, ['model', 'policy'], 'q_joint_stages')
    check_bh(all_models, ['policy'], 'q_enumerated_models_stages')
    main_results = pd.read_csv(base / 'reported_roi_results.csv')
    expected = all_models[(all_models.model == 'censor_hp128') & (all_models.policy == 'observed80')]
    pd.testing.assert_frame_equal(main_results.reset_index(drop=True), expected.reset_index(drop=True))
    counts = {stage: int(((main_results.stage == stage) & (main_results.q_within_contrast <= .05)).sum())
              for stage in ('initial', 'early', 'middle', 'late')}
    assert list(counts.values()) == [29, 15, 5, 0]
    paired = pd.read_csv(base / 'direct_stage_comparisons.csv')
    assert len(paired) == 576
    check_estimates(paired)
    check_bh(paired, ['policy'], 'q_three_pairs_144')
    check_bh(paired, ['policy', 'pair'], 'q_within_pair_48')
    components = pd.read_csv(base / 'phase_component_results.csv')
    check_estimates(components)
    stability = pd.read_csv(base / 'model_stability.csv')
    assert stability.loc[stability.n_testable == 0, ['mean_min', 'mean_max', 'same_direction']].isna().all().all()
    rt = root / 'results/behavioral/fmri_rt_20261012/lme'
    for policy in ('primary', 'min100'):
        coef = pd.read_csv(rt / f'{policy}_fixed_coefficients.csv').set_index('term').Value
        weights = pd.read_csv(rt / f'{policy}_contrast_weights.csv').set_index('contrast')
        covariance = pd.read_csv(rt / f'{policy}_fixed_covariance.csv', index_col=0).loc[coef.index, coef.index]
        weights = weights.loc[:, coef.index]
        contrasts = pd.read_csv(rt / f'{policy}_contrasts.csv').set_index('contrast').loc[weights.index]
        estimate = weights.to_numpy() @ coef.to_numpy()
        se = np.sqrt(np.einsum('ij,jk,ik->i', weights, covariance, weights))
        close(contrasts.log_difference, estimate); close(contrasts.se, se)
        close(contrasts.q_bh_24, stats.false_discovery_control(2 * stats.t.sf(abs(estimate / se), 19)))
    error = root / 'results/behavioral/block_errors_20261013/mixed_model'
    for model in ('binomial', 'gaussian'):
        contrast = pd.read_csv(error / f'{model}_contrasts.csv')
        assert len(contrast) == 24
        close(contrast.q_bh_24, stats.false_discovery_control(contrast.p_two_sided))
    report = dict(status='passed', manifest_files=len(manifest), imaging_rows=len(all_models),
                  direct_phase_rows=len(paired), component_rows=len(components),
                  rt_contrasts=48, error_contrasts=48, main_significant_counts=counts,
                  limits='Aggregate arithmetic and supplied-file integrity only; no participant-level refit, access validation, or correction for model selection.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
