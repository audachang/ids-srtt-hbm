#!/usr/bin/env python3
"""Read-only R1 imaging preflight; evidence stays in the assigned scratch directory."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import json
import platform
import time
from pathlib import Path
import nilearn
import pandas as pd

ROOT = Path('/media/data3/chiaoen_srtt')
OUT = Path('/tmp/r1-r1-imaging')


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    started = time.monotonic()
    assert nilearn.__version__ == '0.13.1'
    OUT.mkdir(parents=True, exist_ok=True)
    config = ROOT / 'code/replication/config/subjects.txt'
    subjects = config.read_text().split() + ['sub-005', 'sub-023']
    assert len(set(subjects)) == 22
    inputs = [config, Path(__file__).resolve(), ROOT / 'docs/REVIEW_R1_ANALYSIS_PLAN_2026-10-19.md',
              ROOT / 'docs/review_round1/briefs/R1_COMMON.md', ROOT / 'docs/review_round1/briefs/r1-imaging.md',
              ROOT / 'code/reanalysis/discover_extended_sources.py']
    rows, runs = [], []
    for subject in subjects:
        for run in range(1, 9):
            path = ROOT / 'data/behavioral' / f'{subject[-3:]}_run_{run}_srtt_prob_new.csv'
            inputs.append(path)
            d = pd.read_csv(path)
            assert len(d) == 320 and d.keypress.notna().all()
            repeat = d.keypress.eq(d.keypress.shift())
            boundary = d.block.ne(d.block.shift())
            random = d.rule_group.eq('Ran')
            gaps = d.stim_onset.diff()
            runs.append(dict(subject=subject, run=run,
                             random_repeats_any_previous_trial=int((repeat & random).sum()),
                             random_repeats_within_block=int((repeat & random & ~boundary).sum()),
                             structured_repeats_any_previous_trial=int((repeat & ~random).sum()),
                             structured_repeats_within_block=int((repeat & ~random & ~boundary).sum())))
            for i in d.index[repeat & random]:
                rows.append(dict(subject=subject, run=run, zero_based_row=int(i),
                                 block=d.block[i], previous_block=d.block[i-1],
                                 target=int(d.keypress[i]), previous_target=int(d.keypress[i-1]),
                                 onset=float(d.stim_onset[i]), onset_gap=float(gaps[i]),
                                 block_boundary=bool(boundary[i])))
    evidence = pd.DataFrame(rows)
    assert len(evidence) == 48 and evidence.block_boundary.all()
    assert set(evidence.onset_gap) == {7., 9., 11.}
    assert sum(r['random_repeats_within_block'] for r in runs) == 0
    evidence.to_csv(OUT / 'participant_random_boundary_repeat_trials.csv', index=False)
    pd.DataFrame(runs).to_csv(OUT / 'participant_repeat_definition_counts.csv', index=False)
    validation = dict(status='blocked_pending_user_definition', analysis_passed=False,
                      preflight_checks_passed=True, n_subjects=22, n_runs=176,
                      n_random_repeats_any_previous_trial=48, n_random_repeats_within_block=0,
                      all_random_repeats_cross_block_boundaries=True, onset_gaps_seconds=[7, 9, 11],
                      nilearn=nilearn.__version__, python=platform.python_version(),
                      elapsed_seconds=time.monotonic()-started)
    (OUT / 'preflight_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    manifest = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in inputs}
    (OUT / 'input_manifest.json').write_text(json.dumps(dict(inputs=manifest), indent=2)+'\n')
    print(json.dumps(validation, indent=2))
    print('Scratch output hashes:')
    for name in ['participant_random_boundary_repeat_trials.csv', 'participant_repeat_definition_counts.csv',
                 'preflight_validation.json', 'input_manifest.json']:
        print(sha(OUT/name), OUT/name)
    print('Producer SHA256:', sha(Path(__file__).resolve()))


if __name__ == '__main__':
    main()
