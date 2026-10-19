#!/usr/bin/env python3
"""Check retained-cohort TSV demographics against two sampled DICOMs per person.

Requires pydicom. Reads only PatientAge and PatientSex. Exports aggregate
values and per-sample agreement/checksums, never names or birth dates.
"""
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pydicom

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR as MANUSCRIPT, PROJECT_ROOT as PROJECT, SUBJECT_LIST, ROI_LIST, BIDS_DIR, ROI_MASK_DIR
DICOM = Path('/home/fmri2404/chiaoen/SrttNewV2/Dicom')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    tsv = PROJECT / 'data/derivatives/fmriprep/participants.tsv'
    subject_list = SUBJECT_LIST
    conversion = PROJECT / 'code/replication/scripts/01_run_heudiconv.sh'
    subjects = subject_list.read_text().split()
    rows = list(csv.DictReader(tsv.open(), delimiter='\t'))
    assert len(subjects) == len(set(subjects)) == len(rows) == 20
    by_id = {row['participant_id']: row for row in rows}
    assert set(by_id) == set(subjects)
    assert 'CE{subject}' in conversion.read_text()
    samples, ages, sexes = [], [], []
    for subject in subjects:
        folder = DICOM / ('CE' + subject.removeprefix('sub-'))
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.upper() in ('.IMA', '.DCM'))
        assert len(files) >= 2, subject
        values = []
        for position, path in [('first', files[0]), ('last', files[-1])]:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=['PatientAge', 'PatientSex'])
            age, sex = str(ds.get('PatientAge', '')).strip(), str(ds.get('PatientSex', '')).strip()
            assert re.fullmatch(r'\d{3}Y', age), (subject, 'missing/non-year age')
            years = int(age[:3])
            assert years == float(by_id[subject]['age']), (subject, 'age mismatch')
            assert sex == by_id[subject]['sex'], (subject, 'sex mismatch')
            values.append((years, sex))
            samples.append({'participant_id': subject, 'sample_position_by_sorted_path': position,
                            'sha256': sha(path), 'age_matches_tsv': True, 'sex_matches_tsv': True})
        assert values[0] == values[1], (subject, 'sample mismatch')
        ages.append(values[0][0]); sexes.append(values[0][1])
    out = MANUSCRIPT / 'data/provenance_audit'
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS', 'n_subjects': len(subjects), 'n_dicom_samples': len(samples),
        'age_matches': len(subjects), 'sex_matches': len(subjects),
        'within_subject_sample_agreement': len(subjects),
        'age_mean': statistics.mean(ages), 'age_sd_sample_ddof1': statistics.stdev(ages),
        'age_range': [min(ages), max(ages)],
        'sex_counts': {s: sexes.count(s) for s in sorted(set(sexes))},
        'sampling': 'First and last DICOM by lexicographically sorted path in each CE-number directory; not chronological order or an exhaustive header audit.',
        'mapping': 'CE{subject} -> sub-{subject}, documented by scripts/01_run_heudiconv.sh',
        'tags': {'age': '0010,1010 PatientAge', 'sex': '0010,0040 PatientSex'},
        'pydicom_version': pydicom.__version__,
        'sources': [{'path': str(p), 'sha256': sha(p)} for p in (tsv, subject_list, conversion, Path(__file__))],
        'limitations': 'Verifies retained-sample consistency with scanner records, not accuracy of initial scanner entry, self-report provenance, or all-25 recruitment demographics.'
    }
    (out / 'dicom_demographic_check.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'dicom_demographic_samples.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    print(f'PASS: {len(subjects)} participants / {len(samples)} sampled DICOMs; all age and sex values match the fMRIPrep TSV.')
    print('Aggregate:', summary['sex_counts'], 'age mean', summary['age_mean'], 'SD', summary['age_sd_sample_ddof1'], 'range', summary['age_range'])


if __name__ == '__main__':
    main()
