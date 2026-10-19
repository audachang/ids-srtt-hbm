#!/usr/bin/env python3
"""Validate A8 independently by reading scalar saved coefficients and manual BH ranks."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import nilearn
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting'
DEFAULT_OUT = ROOT / 'results/glm_volume/review_r2_benchmark_20261020'
HEADLINE = {'LeftSMA', 'RightSMA', 'RightPallidum', 'RightThalamusProper'}
FIELDS = ['n', 'mean', 'sd', 'se', 'ci_low', 'ci_high', 'cohen_dz', 't', 'df', 'p']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-8, equal_nan=True)


def estimate(values):
    clean = [float(value) for value in values if math.isfinite(value)]
    n = len(clean)
    result = {field: math.nan for field in FIELDS}
    result.update(n=n, df=n - 1, mean=statistics.mean(clean) if n else math.nan)
    if n < 2:
        return result
    sd = statistics.stdev(clean)
    se = sd / math.sqrt(n)
    result.update(sd=sd, se=se)
    if sd:
        t = result['mean'] / se
        half = float(stats.t.isf(.025, n - 1)) * se
        result.update(ci_low=result['mean'] - half, ci_high=result['mean'] + half,
            cohen_dz=result['mean'] / sd, t=t, p=float(2 * stats.t.sf(abs(t), n - 1)))
    return result


def manual_bh(values):
    values = np.asarray([p if math.isfinite(p) else 1. for p in values])
    order = np.argsort(values, kind='stable')
    ranked = np.minimum(1., values[order] * len(values) / np.arange(1, len(values) + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = ranked
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out == DEFAULT_OUT or out.is_relative_to(Path('/tmp/r3-r3-benchmark'))
    assert nilearn.__version__ == '0.13.1'
    manifest = json.loads((out / 'input_manifest.json').read_text())
    for path, item in manifest['inputs'].items():
        assert digest(ROOT / path) == item['sha256'], path
    for path, expected in manifest['sources'].items():
        assert digest(ROOT / path) == expected, path
    validation = json.loads((out / 'validation.json').read_text())
    for path, expected in validation['output_sha256'].items():
        assert digest(out / path) == expected, path
    n20 = (ROOT / 'code/replication/config/subjects.txt').read_text().split()
    cohorts = {'N22': n20 + ['sub-005', 'sub-023'], 'N20': n20}
    rois = (ROOT / 'code/replication/config/rois.txt').read_text().split()
    coefficients = {}
    with (SOURCE / 'participant_run_condition_values.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['model'] != 'reference' or row['policy'] != 'observed80':
                continue
            key = (row['subject'], row['roi'], int(row['run']), row['condition'])
            assert key not in coefficients
            coefficients[key] = float(row['value']) if row['value'] else math.nan
    assert len(coefficients) == 22 * 48 * 20
    continuing = pd.read_csv(out / 'continuing_sequence_results.csv')
    new = pd.read_csv(out / 'early_new_sequence_results.csv')
    difference = pd.read_csv(out / 'early_paired_difference_results.csv')
    participant = pd.read_csv(out / 'participant_continuing_sequence_values.csv')
    participant_early = pd.read_csv(out / 'participant_early_comparison_values.csv')
    assert len(continuing) == 192 and len(new) == len(difference) == 96
    assert len(participant) == 42 * 48 * 2 and len(participant_early) == 42 * 48
    assert not continuing.duplicated(['cohort', 'phase', 'roi']).any()
    assert not participant.duplicated(['cohort', 'phase', 'roi', 'subject']).any()
    lookups = [frame.set_index(['cohort', 'phase', 'roi']) for frame in [continuing, new, difference]]
    p_lookup = participant.set_index(['cohort', 'phase', 'roi', 'subject'])
    pe_lookup = participant_early.set_index(['cohort', 'roi', 'subject'])
    summary_checks = 0
    participant_checks = 0
    for cohort, subjects in cohorts.items():
        assert set(participant[participant.cohort == cohort].subject) == set(subjects)
        for phase, runs in [('early', (3, 4)), ('middle', (5, 6))]:
            for roi in rois:
                cv, nv, dv = [], [], []
                for subject in subjects:
                    c = lambda run, condition: coefficients[subject, roi, run, condition]
                    baseline = (c(1, 'Str1') - c(1, 'Ran')) + (c(2, 'Str1') - c(2, 'Ran'))
                    continuing_value = sum(c(run, 'Str1') - c(run, 'Ran') for run in runs) - baseline
                    cv.append(continuing_value)
                    compare(p_lookup.loc[(cohort, phase, roi, subject), 'continuing_did'], continuing_value)
                    participant_checks += 1
                    if phase == 'early':
                        new_value = sum(c(run, 'Str2') - c(run, 'Ran') for run in runs) - baseline
                        diff_value = sum(c(run, 'Str2') - c(run, 'Str1') for run in runs)
                        nv.append(new_value)
                        dv.append(diff_value)
                        row = pe_lookup.loc[(cohort, roi, subject)]
                        compare(row.new_sequence_did, new_value)
                        compare(row.paired_difference_new_minus_continuing, diff_value)
                        compare(new_value - continuing_value, diff_value)
                        participant_checks += 3
                for index, values in [(0, cv)] + ([(1, nv), (2, dv)] if phase == 'early' else []):
                    row = lookups[index].loc[(cohort, phase, roi)]
                    assert row.model == 'reference' and row.policy == 'observed80'
                    expected = estimate(values)
                    for field in FIELDS:
                        compare(row[field], expected[field])
                        summary_checks += 1
    bh_checks = 0
    for index, frame in enumerate([continuing, new, difference]):
        for _, group in frame.groupby(['cohort', 'phase']):
            assert len(group) == 48 and set(group.roi) == set(rois)
            compare(group.q_BH48, manual_bh(group.p))
            bh_checks += 48
            if index == 0:
                target = group[group.roi.isin(HEADLINE)]
                assert len(target) == 4
                compare(target.q_BH4, manual_bh(target.p))
                assert group[~group.roi.isin(HEADLINE)].q_BH4.isna().all()
                bh_checks += 4
    # Check the saved participant and group tables directly, without trusting reproduction CSVs.
    saved_participant = pd.read_csv(SOURCE / 'participant_within_run_values.csv')
    saved_participant = saved_participant.query("model == 'reference' and policy == 'observed80' and contrast == 'early_r34'")
    merged = participant_early.merge(saved_participant[['cohort', 'roi', 'subject', 'value']],
        on=['cohort', 'roi', 'subject'], validate='one_to_one')
    assert len(merged) == 42 * 48
    compare(merged.paired_difference_new_minus_continuing, 2 * merged.value)
    saved_group = pd.read_csv(SOURCE / 'within_run_results.csv')
    saved_group = saved_group.query("model == 'reference' and policy == 'observed80' and contrast == 'early_r34'")
    merged_group = difference.merge(saved_group, on=['cohort', 'roi'], validate='one_to_one', suffixes=('_new', '_saved'))
    assert len(merged_group) == 96
    for field in FIELDS:
        compare(merged_group[field + '_new'], merged_group[field + '_saved'] *
            (2 if field in {'mean', 'sd', 'se', 'ci_low', 'ci_high'} else 1))
    compare(merged_group.q_BH48, merged_group.q)
    # Ensure N20 is literally an overlapping subset, with identical participant values.
    for frame, columns in [(participant, ['continuing_did']), (participant_early,
            ['new_sequence_did', 'paired_difference_new_minus_continuing'])]:
        n22 = frame[frame.cohort == 'N22']
        n20_frame = frame[frame.cohort == 'N20']
        merged = n20_frame.merge(n22, on=['subject', 'roi', 'phase'], suffixes=('_N20', '_N22'), validate='one_to_one')
        assert len(merged) == len(n20_frame)
        for column in columns:
            compare(merged[column + '_N20'], merged[column + '_N22'])
    wide = pd.read_csv(out / 'early_comparison_all_rois.csv')
    assert len(wide) == 96
    for prefix, frame in [('continuing_', continuing.query("phase == 'early'")),
                           ('new_sequence_', new), ('paired_difference_', difference)]:
        joined = wide.merge(frame, on=['cohort', 'phase', 'roi'], validate='one_to_one')
        for field in FIELDS + ['q_BH48']:
            compare(joined[prefix + field], joined[field])
    headline = pd.read_csv(out / 'headline_early_comparison.csv')
    headline_cont = pd.read_csv(out / 'headline_continuing_sequence_results.csv')
    assert len(headline) == 8 and len(headline_cont) == 16
    assert set(headline.roi) == HEADLINE and set(headline_cont.roi) == HEADLINE
    for frame in [headline, headline_cont]:
        assert frame['continuing_n' if frame is headline else 'n'].isin([20, 22]).all()
    reproduction = pd.read_csv(out / 'paired_difference_reproduction.csv')
    assert len(reproduction) == 96 * 11
    compare(reproduction.computed, reproduction.expected_from_saved)
    for cohort in cohorts:
        assert len(continuing.query('cohort == @cohort and phase == "early"')) == 48
        assert len(continuing.query('cohort == @cohort and phase == "middle"')) == 48
    for filename, row_count in validation['row_counts'].items():
        assert len(pd.read_csv(out / filename)) == row_count, filename
    audit = dict(passed=True, status='complete', validator= str(Path(__file__).relative_to(ROOT)),
        validator_sha256=digest(Path(__file__)), no_helper_summary_or_bh_imports=True,
        scalar_input_rows=len(coefficients), participant_numeric_checks=participant_checks,
        independently_recomputed_summary_values=summary_checks, manual_BH_values=bh_checks,
        within_run_participant_reproduction_rows=len(merged_group) * 0 + 42 * 48,
        within_run_group_reproduction_rows=len(merged_group),
        saved_within_run_scaling_and_tests_passed=True, exact_N20_subset_passed=True,
        full_slot_and_headline_membership_passed=True, wide_table_consistency_passed=True,
        input_and_source_hashes_verified=True, producer_output_hashes_verified=True,
        runtime_seconds=time.perf_counter() - started)
    (out / 'independent_validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    validation.update(passed=True, status='complete', independent_validation='passed', independent_checks=audit)
    validation['output_sha256'] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'validation.json'}
    (out / 'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
