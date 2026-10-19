#!/usr/bin/env python3
"""A8: continuing-sequence benchmark from saved reference/observed80 values only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import nilearn
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/reanalysis'))
from common import summarize, bh

SOURCE = ROOT / 'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting'
DEFAULT_OUT = ROOT / 'results/glm_volume/review_r2_benchmark_20261020'
HEADLINE = ['LeftSMA', 'RightSMA', 'RightPallidum', 'RightThalamusProper']
LABELS = dict(zip(HEADLINE, ['Left BA6', 'Right BA6', 'Right pallidum', 'Right thalamus']))
PHASES = {'early': [3, 4], 'middle': [5, 6]}
STAT_FIELDS = ['n', 'mean', 'sd', 'se', 'ci_low', 'ci_high', 'cohen_dz', 't', 'df', 'p']
SCALED_FIELDS = {'mean', 'sd', 'se', 'ci_low', 'ci_high'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')


def selected(path):
    frame = pd.read_csv(path)
    return frame[(frame.model == 'reference') & (frame.policy == 'observed80')].copy()


def close(a, b):
    np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-8, equal_nan=True)


def corrected(frame):
    frame = frame.copy()
    frame['q_BH48'] = np.nan
    frame['q_BH4'] = np.nan
    for _, group in frame.groupby(['cohort', 'phase']):
        assert len(group) == 48
        frame.loc[group.index, 'q_BH48'] = bh(group.p)
        target = group[group.roi.isin(HEADLINE)]
        assert len(target) == 4
        frame.loc[target.index, 'q_BH4'] = bh(target.p)
    return frame


def interval(row, prefix=''):
    return f"{row[prefix+'mean']:+.6f} [{row[prefix+'ci_low']:+.6f}, {row[prefix+'ci_high']:+.6f}]"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out == DEFAULT_OUT or out.is_relative_to(Path('/tmp/r3-r3-benchmark'))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {out}')
    assert nilearn.__version__ == '0.13.1'
    inputs = [SOURCE / name for name in [
        'participant_run_condition_values.csv', 'participant_within_run_values.csv',
        'within_run_results.csv', 'participant_stage_values.csv', 'stage_results.csv',
        'participant_roi_coverage.csv', 'validation.json']]
    inputs += [ROOT / name for name in [
        'code/replication/config/subjects.txt', 'code/replication/config/rois.txt',
        'code/reanalysis/common.py', 'code/reanalysis/updated_common.py',
        'code/reanalysis/updated_results.py', 'docs/REVIEW_R1_ANALYSIS_PLAN_2026-10-19.md',
        'docs/REVIEW_R1_ANALYSIS_PLAN_2026-10-20_A8.md',
        'docs/review_round1/briefs/R1_COMMON.md', 'docs/review_round1/briefs/r3-benchmark.md',
        'docs/review_round1/reviewer-1.md', 'docs/review_round1/reviewer-2.md',
        'docs/review_round1/round2/reviewer-1.md', 'docs/review_round1/round2/reviewer-2.md']]
    input_hashes = {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in inputs}
    assert json.loads((SOURCE / 'validation.json').read_text())['passed']
    n20 = (ROOT / 'code/replication/config/subjects.txt').read_text().split()
    n22 = n20 + ['sub-005', 'sub-023']
    rois = (ROOT / 'code/replication/config/rois.txt').read_text().split()
    assert len(set(n20)) == 20 and len(set(n22)) == 22 and len(set(rois)) == 48
    data = selected(SOURCE / 'participant_run_condition_values.csv')
    assert len(data) == 22 * 20 * 48
    assert set(data.subject) == set(n22) and set(data.roi) == set(rois)
    assert not data.duplicated(['subject', 'roi', 'run', 'condition']).any()
    coverage = pd.read_csv(SOURCE / 'participant_roi_coverage.csv').set_index(['subject', 'roi'])
    pivot = data.pivot(index=['subject', 'roi'], columns=['run', 'condition'], values='value')
    assert len(pivot) == 22 * 48
    eligible = coverage.loc[pivot.index, 'fraction'].ge(.8 - 1e-12).to_numpy()
    assert np.array_equal(np.isfinite(pivot.to_numpy()).all(axis=1), eligible)
    assert np.array_equal(np.isnan(pivot.to_numpy()).all(axis=1), ~eligible)
    saved_within = selected(SOURCE / 'participant_within_run_values.csv')
    saved_within_results = selected(SOURCE / 'within_run_results.csv')
    saved_stages = selected(SOURCE / 'participant_stage_values.csv')
    saved_stage_results = selected(SOURCE / 'stage_results.csv')
    participant_rows, early_rows, results_rows, new_rows, diff_rows = [], [], [], [], []
    participant_checks, inference_checks, stage_checks = [], [], []
    for cohort, subjects in [('N22', n22), ('N20', n20)]:
        p = pivot.loc[pd.IndexSlice[subjects, :], :].sort_index()
        baseline_s1 = (p[1, 'Str1'] + p[2, 'Str1']) / 2
        baseline_r = (p[1, 'Ran'] + p[2, 'Ran']) / 2
        for phase, runs in PHASES.items():
            phase_s1 = (p[runs[0], 'Str1'] + p[runs[1], 'Str1']) / 2
            phase_r = (p[runs[0], 'Ran'] + p[runs[1], 'Ran']) / 2
            delta_s1, delta_r = phase_s1 - baseline_s1, phase_r - baseline_r
            continuing = 2 * (delta_s1 - delta_r)
            block = pd.DataFrame({
                'baseline_Str1_mean': baseline_s1, 'baseline_Ran_mean': baseline_r,
                'phase_Str1_mean': phase_s1, 'phase_Ran_mean': phase_r,
                'delta_Str1': delta_s1, 'delta_Ran': delta_r,
                'continuing_did': continuing}).reset_index()
            block['cohort'], block['phase'] = cohort, phase
            block['model'], block['policy'] = 'reference', 'observed80'
            block['eligible'] = np.isfinite(block.continuing_did)
            participant_rows.append(block)
            for roi in rois:
                group = block[block.roi == roi]
                results_rows.append(dict(cohort=cohort, model='reference', policy='observed80', phase=phase,
                                         roi=roi, **summarize(group.continuing_did)))
            if phase != 'early':
                continue
            phase_s2 = (p[3, 'Str2'] + p[4, 'Str2']) / 2
            new = 2 * ((phase_s2 - baseline_s1) - delta_r)
            difference = new - continuing
            close(difference, 2 * (phase_s2 - phase_s1))
            block['phase_Str2_mean'] = phase_s2.to_numpy()
            block['delta_new_Str2_vs_initial_Str1'] = (phase_s2 - baseline_s1).to_numpy()
            block['new_sequence_did'] = new.to_numpy()
            block['paired_difference_new_minus_continuing'] = difference.to_numpy()
            block['twice_within_run_Str2_minus_Str1'] = (2 * (phase_s2 - phase_s1)).to_numpy()
            early_rows.append(block)
            w = saved_within[(saved_within.cohort == cohort) & (saved_within.contrast == 'early_r34')]
            s = saved_stages[(saved_stages.cohort == cohort) & (saved_stages.stage == 'early')]
            check = block.merge(w[['subject', 'roi', 'value']], on=['subject', 'roi'], validate='one_to_one')
            assert len(check) == len(subjects) * 48
            close(check.paired_difference_new_minus_continuing, 2 * check.value)
            for row in check.itertuples():
                participant_checks.append(dict(cohort=cohort, subject=row.subject, roi=row.roi,
                    paired_difference=row.paired_difference_new_minus_continuing,
                    saved_within_run_mean=row.value, expected_twice_saved=2 * row.value,
                    absolute_error=abs(row.paired_difference_new_minus_continuing - 2 * row.value)))
            check_s = block.merge(s[['subject', 'roi', 'value']], on=['subject', 'roi'], validate='one_to_one')
            assert len(check_s) == len(subjects) * 48
            close(check_s.new_sequence_did, check_s.value)
            for roi in rois:
                group = block[block.roi == roi]
                finite = np.isfinite(group[['continuing_did', 'new_sequence_did',
                    'paired_difference_new_minus_continuing']].to_numpy())
                assert np.array_equal(finite[:, 0], finite[:, 1]) and np.array_equal(finite[:, 0], finite[:, 2])
                new_summary = summarize(group.new_sequence_did)
                diff_summary = summarize(group.paired_difference_new_minus_continuing)
                key = dict(cohort=cohort, model='reference', policy='observed80', phase='early', roi=roi)
                new_rows.append(key | new_summary)
                diff_rows.append(key | diff_summary)
                original_w = saved_within_results[(saved_within_results.cohort == cohort) &
                    (saved_within_results.contrast == 'early_r34') & (saved_within_results.roi == roi)]
                original_s = saved_stage_results[(saved_stage_results.cohort == cohort) &
                    (saved_stage_results.stage == 'early') & (saved_stage_results.roi == roi)]
                assert len(original_w) == len(original_s) == 1
                for field in STAT_FIELDS:
                    expected = float(original_w.iloc[0][field]) * (2 if field in SCALED_FIELDS else 1)
                    close(diff_summary[field], expected)
                    inference_checks.append(key | dict(metric=field, computed=diff_summary[field],
                        expected_from_saved=expected, absolute_error=abs(diff_summary[field] - expected)))
                    close(new_summary[field], float(original_s.iloc[0][field]))
                    stage_checks.append(key | dict(metric=field, computed=new_summary[field],
                        saved=float(original_s.iloc[0][field]),
                        absolute_error=abs(new_summary[field] - float(original_s.iloc[0][field]))))
    continuing_results = corrected(pd.DataFrame(results_rows))
    # Only the continuing benchmark has newly specified BH4 families. For the two early
    # companion contrasts, BH48 reproduces the existing stage and within-run families.
    new_results = corrected(pd.DataFrame(new_rows)).drop(columns='q_BH4')
    diff_results = corrected(pd.DataFrame(diff_rows)).drop(columns='q_BH4')
    for frame, saved, key, value in [(new_results, saved_stage_results, 'stage', 'early'),
                                    (diff_results, saved_within_results, 'contrast', 'early_r34')]:
        reference = saved[saved[key] == value][['cohort', 'roi', 'q']]
        check = frame.merge(reference, on=['cohort', 'roi'], validate='one_to_one')
        assert len(check) == 96
        close(check.q_BH48, check.q)
        if frame is diff_results:
            for row in check.itertuples():
                inference_checks.append(dict(cohort=row.cohort, model='reference', policy='observed80',
                    phase='early', roi=row.roi, metric='q_BH48', computed=row.q_BH48,
                    expected_from_saved=row.q, absolute_error=abs(row.q_BH48 - row.q)))
    wide_keys = ['cohort', 'model', 'policy', 'phase', 'roi']
    wide = None
    for prefix, frame in [('new_sequence_', new_results), ('continuing_', continuing_results.query("phase == 'early'")),
                          ('paired_difference_', diff_results)]:
        renamed = frame.rename(columns={column: prefix + column for column in frame if column not in wide_keys})
        wide = renamed if wide is None else wide.merge(renamed, on=wide_keys, validate='one_to_one')
    wide['roi_label'] = wide.roi.map(LABELS).fillna(wide.roi)
    headline = wide[wide.roi.isin(HEADLINE)].copy()
    continuing_headline = continuing_results[continuing_results.roi.isin(HEADLINE)].copy()
    continuing_headline['roi_label'] = continuing_headline.roi.map(LABELS)
    counts = []
    for (cohort, phase), group in continuing_results.groupby(['cohort', 'phase']):
        counts.append(dict(cohort=cohort, phase=phase, slots_BH48=len(group), slots_BH4=4,
            testable_slots=int(group.p.notna().sum()),
            discoveries_BH48=int(group.q_BH48.le(.05).sum()),
            headline_discoveries_BH4=int(group.q_BH4.le(.05).sum())))
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        'participant_continuing_sequence_values.csv': pd.concat(participant_rows, ignore_index=True),
        'participant_early_comparison_values.csv': pd.concat(early_rows, ignore_index=True),
        'continuing_sequence_results.csv': continuing_results,
        'early_new_sequence_results.csv': new_results,
        'early_paired_difference_results.csv': diff_results,
        'early_comparison_all_rois.csv': wide,
        'headline_early_comparison.csv': headline,
        'headline_continuing_sequence_results.csv': continuing_headline,
        'participant_paired_difference_reproduction.csv': pd.DataFrame(participant_checks),
        'paired_difference_reproduction.csv': pd.DataFrame(inference_checks),
        'new_sequence_reproduction.csv': pd.DataFrame(stage_checks),
        'family_summary.csv': pd.DataFrame(counts),
    }
    for filename, frame in outputs.items():
        save(frame, out / filename)
    lines = ['# A8: Continuing-sequence benchmark', '',
        'Specified 2026-10-20 before A8 results; secondary check of specificity. N=22 uncensored '
        'reference/observed80 is primary; N=20 is its supplement-only subset. No refits.', '',
        '## Definitions and inference', '',
        'For each eligible participant and ROI, initial means use runs 1–2; early means runs 3–4; '
        'middle means runs 5–6. ΔS1 = phase Str1 mean − initial Str1 mean; '
        'ΔR = phase Ran mean − initial Ran mean. Continuing DiD = 2(ΔS1 − ΔR). '
        'The factor 2 gives the historical two-run summed scale in percent-mean-scaled coefficient units. '
        'Early new-sequence DiD = 2[(early Str2 mean − initial Str1 mean) − ΔR]. '
        'The participant-paired difference new − continuing equals 2 × early mean(Str2 − Str1). '
        'All three early contrasts use exactly the same eligible participants within each ROI. '
        'The saved observed80 participant-specific eight-run support is retained, without imputation.', '',
        'Two-sided one-sample t tests use df = eligible n − 1 and pointwise 95% t intervals. '
        'Untestable slots have missing estimates/tests and enter BH as p = 1. '
        'The continuing benchmark uses BH4 across left/right BA6, right pallidum, and right thalamus '
        'separately for each phase/cohort (headline), and BH48 across all slots separately for each '
        'phase/cohort (secondary). Non-headline BH4 cells are blank. Early new and paired companion '
        'BH48 values reproduce the existing early-stage and early-within-run families respectively; '
        'no new BH4 family is assigned to those companions. No pooling over cohorts, phases, '
        'or contrast types is implied. The BA6 masks retain their historical LeftSMA/RightSMA file identifiers.', '',
        '## N=22 headline early estimates', '',
        '| ROI | New DiD [95% CI] | Continuing DiD [95% CI] | New − continuing [95% CI] | Continuing q BH4 | Continuing q BH48 |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in headline[headline.cohort == 'N22'].to_dict('records'):
        lines.append(f"| {row['roi_label']} | {interval(row, 'new_sequence_')} | {interval(row, 'continuing_')} | "
                     f"{interval(row, 'paired_difference_')} | {row['continuing_q_BH4']:.6g} | {row['continuing_q_BH48']:.6g} |")
    lines += ['', '## N=22 headline middle continuing estimates', '',
        '| ROI | n | Continuing DiD [95% CI] | q BH4 | q BH48 |', '| --- | --- | --- | --- | --- |']
    for row in continuing_headline.query("cohort == 'N22' and phase == 'middle'").to_dict('records'):
        lines.append(f"| {row['roi_label']} | {row['n']} | {interval(row)} | {row['q_BH4']:.6g} | {row['q_BH48']:.6g} |")
    lines += ['', '## Coverage, reproduction and files', '',
        'The full continuing table has 192 rows (2 cohorts × 2 phases × 48 slots); '
        'each early companion table has 96 rows. Full early comparisons and headline comparisons '
        'include new, continuing, and paired estimates with CIs. N=20 rows are supplement-only. '
        'Participant-prefixed CSVs are internal participant-level outputs.', '',
        'Participant-level early differences reproduce twice the saved early_r34 within-run values; '
        'the group mean, SD, SE and CI reproduce twice the saved values, while n, df, t, p and '
        'Cohen dz are unchanged. The 48-slot q values also reproduce the saved within-run results. '
        'The new-sequence early contrasts reproduce the saved early-stage participant values and '
        'group inference. Errors are tabulated in the reproduction CSVs; validation.json records '
        'checks and the separate validator outcome. input_manifest.json records SHA-256 inputs, '
        'source code, command and software versions.', '',
        '## Limits and handoff', '',
        'A shift in the continuing contrast is not by itself evidence for or against a difference '
        'between the sequences; the paired difference supplies that comparison. An interval '
        'including zero does not establish equivalence or absence of specificity. The changing '
        'prevalence schedule confounds time, practice and exposure, and this benchmark does not '
        'establish a causal effect of sequence replacement. These exploratory results do not '
        'change the primary model, ROI set, or cohort hierarchy. CIs are pointwise and do not '
        'account for earlier selection. No new nuisance models, whole-brain analyses, manuscript '
        'edits, builds, public release, commits, or external messages were performed.', '',
        'Deviations: none. The coordinator owns manuscript integration, public disclosure review, '
        'and updates to shared status/revision logs; this agent writes only its assigned paths.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    assert all(digest(ROOT / path) == entry['sha256'] for path, entry in input_hashes.items())
    source_paths = [Path(__file__), ROOT / 'code/review_r1/benchmark_validate.py']
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), inputs=input_hashes,
        sources={str(p.relative_to(ROOT)): digest(p) for p in source_paths}, argv=sys.argv,
        source_root=str(ROOT), software=dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, pandas=pd.__version__, nilearn=nilearn.__version__),
        thread_limits={key: os.environ[key] for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']})
    write_json(out / 'input_manifest.json', manifest)
    finite_errors = pd.DataFrame(inference_checks).absolute_error.dropna()
    validation = dict(passed=True, status='producer_checks_passed',
        independent_validation='pending', no_refits=True, input_hashes_unchanged=True,
        nilearn_version_assertion=True, both_cohorts_and_all_slots_retained=True,
        coverage_finiteness_matches_observed80=True, matched_participants_for_early_comparisons=True,
        paired_algebra_passed=True, participant_saved_within_run_reproduction=True,
        saved_within_run_inference_reproduction=True, saved_new_sequence_stage_reproduction=True,
        bh4_per_phase_and_bh48_per_phase=True, untestable_slots_use_p1=True,
        n_subjects_N22=22, n_subjects_N20=20,
        row_counts={name: len(frame) for name, frame in outputs.items()},
        max_paired_inference_absolute_error=float(finite_errors.max()),
        runtime_seconds=time.perf_counter() - started,
        output_sha256={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'validation.json'})
    write_json(out / 'validation.json', validation)
    print(json.dumps({'output': str(out), 'passed': True, 'runtime_seconds': validation['runtime_seconds']}))


if __name__ == '__main__':
    main()
