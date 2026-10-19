#!/usr/bin/env python3
"""Assemble v1.2.2 from an immutable public archive and saved group summaries.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
No analyses or network operations. Scientific inputs and earlier releases are read only.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import numpy as np
import pandas as pd
import build_public_release_v1_2 as inherited
from verify_public_summaries_r2 import A8, summary_blocks, disclosure_plan, disclosure_check, blocked_fields

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path('/tmp/rev-rel-build-v122')
DEFAULT_DEST = ROOT/'submissions/hbm_release_v1.2.2_candidate'
BASE_COMMIT = '810829c435c9e6aac102d2d719aba7c3d6e68476'
CODE_HASHES = {'benchmark_run.py': 'b8004bb28f66ca8ae1b582fc96e80981fc0ebab0e08543b8138d4f24875164c5', 'benchmark_validate.py': 'cf745a2de6a5a87ba3432d122cc8212aee129970b539deb98949ad9cbbb1aa24'}
ALLOWLIST = [A8+'/'+name for name in [
 'validation.json', 'independent_validation.json', 'continuing_sequence_results.csv',
 'early_new_sequence_results.csv', 'early_paired_difference_results.csv',
 'early_comparison_all_rois.csv', 'headline_early_comparison.csv',
 'headline_continuing_sequence_results.csv', 'family_summary.csv']]
sha = inherited.sha
dump = inherited.dump
write_csv = inherited.write_csv


def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*')), 'Symlinks forbidden'
    return [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(root.rglob('*')) if p.is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--release-date', required=True)
    parser.add_argument('--doi')
    parser.add_argument('--commit')
    parser.add_argument('--refresh-owned-candidate', action='store_true')
    args = parser.parse_args()
    try:
        date = datetime.date.fromisoformat(args.release_date).isoformat()
    except ValueError:
        parser.error('--release-date must be YYYY-MM-DD')
    if date != args.release_date:
        parser.error('--release-date must be YYYY-MM-DD')
    if bool(args.doi) != bool(args.commit):
        parser.error('--doi and --commit must be supplied together')
    if args.doi and not re.fullmatch(r'10\.\d{4,9}/[^\s<>]+', args.doi):
        parser.error('Invalid DOI')
    if args.commit and not re.fullmatch(r'[0-9a-fA-F]{40}', args.commit):
        parser.error('Full 40-character commit required')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER'] = str(SCRATCH/'joblib')
    os.environ['MPLCONFIGDIR'] = str(SCRATCH/'matplotlib')
    dest = args.dest.resolve()
    assert dest == DEFAULT_DEST or (dest.is_relative_to(SCRATCH) and dest != SCRATCH), 'Unowned destination'
    # Candidate replacement is permitted only for a fully intact previous builder output.
    if dest.exists() and any(dest.iterdir()):
        assert args.refresh_owned_candidate and dest == DEFAULT_DEST, 'Destination already populated'
        manifest = json.loads((dest/'SHA256_manifest.json').read_text())
        assert inventory(dest) == sorted(manifest+[dict(path='SHA256_manifest.json',
            bytes=(dest/'SHA256_manifest.json').stat().st_size, sha256=sha(dest/'SHA256_manifest.json'))], key=lambda r:r['path'])
        for row in inventory(dest):
            (dest/row['path']).unlink()
        for p in sorted((p for p in dest.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
            p.rmdir()
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(['git', '-C', str(ROOT/'submissions/hbm_release_v1.0.0'),
                                       'archive', '--format=tar', BASE_COMMIT])
    with tempfile.TemporaryDirectory(prefix='archive-', dir=SCRATCH) as temporary:
        old = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(old, filter='data')
        base = {r['path']:r['sha256'] for r in inventory(old)}
        shutil.copytree(old, dest, dirs_exist_ok=True)
    (dest/'SHA256_manifest.json').unlink()
    notes = json.loads((dest/'AGGREGATION_NOTES.json').read_text())
    exclusions = json.loads((dest/'RELEASE_EXCLUSIONS.json').read_text())
    coverage = pd.read_csv(dest/'RELEASE_COVERAGE.csv').fillna('')
    decisions = {r.source:(r.decision, r.reason) for r in coverage.itertuples()}
    source_hashes = {r.source:r.source_sha256 for r in coverage.itertuples()}
    added = []
    for rel in sorted(ALLOWLIST):
        source = ROOT/rel; target = dest/rel
        assert source.is_file(), rel
        source_hashes[rel] = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        operations = []
        if source.suffix == '.csv':
            frame = pd.read_csv(source)
            assert not (set(frame.columns.str.lower()) & inherited.FORBIDDEN), rel
            assert not inherited.PRIVATE.search(source.read_text()), rel
            shutil.copyfile(source, target)
        elif source.suffix == '.json':
            removed = []
            obj = inherited.clean_json(json.loads(source.read_text()), removed)
            if removed:
                dump(target, obj)
                operations.append(dict(operation='redaction', removed_json_pointers=removed,
                                       reason='Withhold private/runtime paths and individual-file provenance.'))
            else:
                shutil.copyfile(source, target)
        elif source.suffix == '.md':
            lines = source.read_text().splitlines(keepends=True); removed = []
            for i, line in enumerate(lines):
                if inherited.PRIVATE.search(line):
                    lines[i] = '[Line withheld: participant identifier or private path.]\n'; removed.append(i+1)
            target.write_text(''.join(lines))
            if removed:
                operations.append(dict(operation='redaction', line_numbers=removed, reason='Withhold identifiers and private/runtime paths.'))
        else:
            raise ValueError(rel)
        for operation in operations:
            notes.append(dict(path=rel, source_sha256=sha(source), release_sha256=sha(target), **operation))
        decisions[rel] = ('include with suppression/redaction' if operations else 'include verbatim',
                          'Reviewed group summary; see AGGREGATION_NOTES.json for privacy changes.')
        added.append(rel)
    excluded = collections.Counter()
    for source in sorted((ROOT/A8).rglob('*')):
        if not source.is_file():
            continue
        rel = str(source.relative_to(ROOT))
        if rel in ALLOWLIST:
            continue
        if 'first_level' in source.parts or 'voxelwise_inputs' in source.parts:
            reason = 'Individual first-level/run outputs, participant maps and input provenance.'
        elif source.name.startswith('participant_') or 'controls' in source.parts:
            reason = 'Participant/run-level values, metadata or individual control results.'
        elif source.suffix in {'.nii', '.gz', '.npz', '.npy'}:
            reason = 'MRI images, participant arrays or raw permutation nulls.'
        elif source.name in {'REPORT.md', 'paired_difference_reproduction.csv', 'new_sequence_reproduction.csv'}:
            reason = 'Derived numerical report or reproduction table could restore suppressed statistics.'
        else:
            reason = 'Restricted provenance or outputs outside the reviewed public file set.'
        excluded[reason] += 1
        if not inherited.PRIVATE.search(rel):
            exclusions.append(dict(path=rel, reason=reason, historical=False))
        if reason.startswith('Derived numerical'):
            notes.append(dict(path=rel, operation='withholding derived display', source_sha256=sha(source), reason=reason))
    exclusions += [dict(path=A8+'/', reason=reason, files=count,
                       detail='Excluded files counted by category; individual identifier paths are not named.')
                   for reason, count in sorted(excluded.items())]
    # Reviewed source files are pinned and copied without redaction under U1.
    review = list(csv.DictReader((dest/'CODE_REVIEW.csv').open()))
    assert {p.name for p in (ROOT/'code/review_r1').glob('benchmark_*.py')} == set(CODE_HASHES)
    code_paths = ['code/review_r1/'+name for name in sorted(CODE_HASHES)]
    code_paths += ['code/reporting/build_public_release_v1_2_2.py', 'code/reporting/verify_public_summaries_r2.py']
    for rel in code_paths:
        source = ROOT/rel
        if source.name in CODE_HASHES:
            assert sha(source) == CODE_HASHES[source.name], 'Unreviewed source change: '+rel
        target = dest/rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        assert not any(r['path'] == rel for r in review)
        review.append(dict(path=rel, source_sha256=sha(source), credentials_or_tokens='none found',
            embedded_participant_data='none found', decision='include verbatim',
            review='Inspected for credentials and literal individual observations. Labels and local paths in source refer to restricted inputs.',
            origin_sha256='', origin_line_range='', release_sha256=sha(target)))
    write_csv(dest/'CODE_REVIEW.csv', sorted(review, key=lambda r:r['path']))
    # One global pass covers inherited and benchmark slots, comparison blocks and duplicates.
    frames, plan, pairs = disclosure_plan(dest)
    privacy = {}; records = []
    for rel, frame in frames.items():
        changes = []; before = sha(dest/rel)
        for block in summary_blocks(rel, frame):
            if (rel, block['index'], block['suffix']) not in plan:
                continue
            fields = blocked_fields(frame, block)
            if not frame.loc[block['index'], fields].notna().any():
                continue
            row = frame.loc[block['index']]
            keys = {k:str(row[k]) for k in ['cohort','model','policy','roi']+inherited.SLOT_KEYS if k in frame}
            changes.append(dict(row_key=keys, csv_line=int(block['index'])+2,
                                statistic_block=block['suffix'] or 'main', columns=fields))
            frame.loc[block['index'], fields] = np.nan
        if changes:
            frame.to_csv(dest/rel, index=False)
            notes.append(dict(path=rel, operation='complementary suppression', source_sha256=before,
                release_sha256=sha(dest/rel), blocks=changes,
                reason='Smaller nested and equivalent sample sets lose all statistics, including related components; N, slots and BH q remain.'))
            records.append(dict(path=rel, blocks=len(changes)))
            decisions[rel] = ('include with suppression/redaction', 'Complementary suppression prevents recovery from nested sample summaries; N and BH q retained.')
            if rel in base:
                privacy[rel] = 'complementary suppression across inherited and continuing-sequence summary slots'
    disclosure = disclosure_check(dest)
    dump(dest/'AGGREGATION_NOTES.json', notes)
    dump(dest/'RELEASE_EXCLUSIONS.json', exclusions)
    ledger_bytes = (ROOT/'docs/t9/facts_n22.json').read_bytes()
    ledger = json.loads(ledger_bytes); by_source = collections.defaultdict(list)
    for fact in ledger:
        by_source[fact['source']].append(fact)
    new_coverage = []
    for rel, facts in sorted(by_source.items()):
        if rel not in source_hashes:
            source_hashes[rel] = sha(ROOT/rel)
        new_coverage.append(inherited.coverage_row(rel, facts, dest, decisions, source_hashes[rel]))
    write_csv(dest/'RELEASE_COVERAGE.csv', new_coverage)
    handoff = (ROOT/'docs/review_round1/reports/rev2-supp.md').read_text()
    ready = any(rel.startswith(A8+'/') for rel in by_source) and 'A8' in handoff and 'A8 pending' not in handoff.splitlines()[0]
    status = dict(complete=ready, source_sha256=hashlib.sha256(ledger_bytes).hexdigest(), source_files=len(new_coverage),
        recorded_values=len(ledger), verifiable_values=sum(r['verifiable_entries'] for r in new_coverage),
        withheld_values=sum(r['withheld_entries'] for r in new_coverage),
        description='Coverage uses the completed article value inventory.' if ready else 'Refresh coverage after article assets are completed.')
    dump(dest/'COVERAGE_STATUS.json', status)
    dump(dest/'AGGREGATE_CHECK_SPEC_R2.json', dict(description='Saved continuing-sequence benchmark aggregates only; no fitting or new inference.',
        added_results=sorted(added), complementary_suppression=records, disclosure=disclosure,
        planned_BH_slots=dict(continuing_per_phase=48, continuing_headline_per_phase=4, early_new=48, early_paired=48), excluded_files=sum(excluded.values())))
    release_url = 'https://github.com/audachang/ids-srtt-hbm/releases/tag/v1.2.2'
    cff = (dest/'CITATION.cff').read_text()
    cff = '\n'.join(line for line in cff.splitlines() if not line.startswith(('doi:', 'commit:')))+'\n'
    cff = re.sub(r'^version:.*$', 'version: 1.2.2', cff, flags=re.M)
    cff = re.sub(r'^date-released:.*$', 'date-released: '+date, cff, flags=re.M)
    cff = cff.replace('releases/tag/v1.2.1', 'releases/tag/v1.2.2')
    cff = re.sub(r'^message:.*$', 'message: "Please cite the version DOI when available; otherwise cite the release URL."', cff, flags=re.M)
    if args.doi:
        cff += 'doi: "'+args.doi+'"\ncommit: "'+args.commit+'"\n'
    cff = re.sub(r'^title:.*$', 'title: \"Analysis code and group summaries for Neural responses during motor statistical learning and gradual sequence replacement\"', cff, flags=re.M)
    (dest/'CITATION.cff').write_text(cff)
    public_status = ('Version 1.2.2 published on '+date+'.\nVersion DOI: https://doi.org/'+args.doi+'\nFrozen commit: '+args.commit+'\n') if args.doi else (
        'Version 1.2.2 prepared on '+date+'.\nThe version DOI will be linked after archival completes.\n')
    previous = ('Version 1.2.1: doi:10.5281/zenodo.23484681.\nVersion 1.2.0: doi:10.5281/zenodo.23474855.\nVersion 1.1.0: doi:10.5281/zenodo.23471923.\nVersion 1.0.0: doi:10.5281/zenodo.23438462.\n')
    (dest/'RELEASE_STATUS.txt').write_text(public_status+'GitHub release: '+release_url+'\n'+previous+
        'Participant-level data are not distributed.\nOriginal software: MIT. Group summaries and documentation: CC BY 4.0.\n')
    old_readme = (dest/'README.txt').read_text()
    # Preserve the inherited substantive methods/privacy discussion after its introduction.
    body = old_readme[old_readme.index('The 22-participant uncensored reference analysis'):]
    body = body.replace('python -B code/reporting/verify_public_summaries_r1_1.py', 'python -B code/reporting/verify_public_summaries_r2.py')
    body = re.sub(r'RELEASE_COVERAGE.csv accounts for [\d,]+ article\nvalues from [\d,]+ sources, of which\n[\d,]+ values can be checked against public files\.',
        f"RELEASE_COVERAGE.csv accounts for {status['recorded_values']:,} article\nvalues from {status['source_files']:,} sources, of which\n{status['verifiable_values']:,} values can be checked against public files.", body)
    body = body.replace('AGGREGATE_CHECK_SPEC_R1.json and AGGREGATE_CHECK_SPEC_R1_1.json specify the added checks.',
        'AGGREGATE_CHECK_SPEC_R1.json, AGGREGATE_CHECK_SPEC_R1_1.json and AGGREGATE_CHECK_SPEC_R2.json specify the added checks.')
    (dest/'README.txt').write_text(f'''Analysis code and group summaries for Neural responses during motor statistical learning and gradual sequence replacement
IDS-SRTT HBM companion — v1.2.2 ({date})

Analysis source code and group summaries only.
Public repository: https://github.com/audachang/ids-srtt-hbm
GitHub release: {release_url}
{public_status}{previous}
What's new in 1.2.2

This version adds the continuing-sequence benchmark, early new-sequence estimates,
and participant-paired new-minus-continuing comparisons with their source code.
The continuing contrast compares the old sequence with Random from the initial
phase to early and middle replacement. All contrasts retain the historical two-run
summed scale and the same eligible participants within each parcel. The continuing
benchmark reports separate four-parcel headline and 48-parcel families for each
phase and cohort; the early new and paired comparisons retain their 48-parcel
families. The article title now uses "gradual sequence replacement".

These secondary analyses are exploratory and do not change the primary uncensored
reference analysis. Confidence intervals are pointwise. Overlapping cohorts are
not independent groups; intervals containing zero do not establish equivalence.
The changing prevalence schedule cannot identify a causal effect of replacement.
Brodmann area 6 parcels retain the storage names LeftSMA and RightSMA; they cover
the full Brodmann parcels. The slice-timing-aligned sensitivity is retained.

Complementary suppression applies jointly to the added and inherited group summaries,
including duplicates and related components. Numerical ROI reports that could restore
suppressed values are withheld. This version supersedes version 1.2.1 for citation.
Earlier published versions remain available and unaltered; this release cannot
remove disclosures already present in earlier archives.

'''+body)
    metadata = json.loads((dest/'.zenodo.json').read_text())
    metadata.update(version='1.2.2', publication_date=date); metadata.pop('doi', None)
    metadata['title'] = 'Analysis code and group summaries for Neural responses during motor statistical learning and gradual sequence replacement'
    metadata['description'] = ('<p>Analysis source code and group-level imaging and behavioral summaries for an interleaved dual-structure serial reaction time fMRI study.</p>'
        '<p>Version 1.2.2 adds the continuing-sequence benchmark and early participant-paired new-minus-continuing summaries, with the source code that produced them. The article title now uses gradual sequence replacement. Continuing contrasts have separate four-parcel headline and 48-parcel families per phase and cohort; early new and paired companions retain their 48-parcel families. The slice-timing-aligned sensitivity is retained. The 22-participant unaligned uncensored reference analysis with at least 80% observed regional support remains primary. The overlapping learner-selected 20-participant sample appears only in the article supplement. Findings are exploratory; confidence intervals are pointwise. Analysis history is disclosed in the article Methods.</p>'
        '<p>Participant-level observations, fitted models, MRI images, raw permutation arrays and recoverable individual values are withheld. Complementary suppression applies across added and inherited summaries, overlapping cohorts, coverage rules and related components. Numerical ROI reports that could restore suppressed statistics are withheld. Code labels and paths refer to restricted inputs; approved data access and separate dependencies are required to re-estimate models. The complete original run_saved.py remains withheld; its generic direct function and necessary imports are distributed.</p>'
        '<p>This version supersedes version 1.2.1 for citation. Earlier archives remain unaltered: version 1.2.1 at doi:10.5281/zenodo.23484681, version 1.2.0 at doi:10.5281/zenodo.23474855, version 1.1.0 at doi:10.5281/zenodo.23471923, and version 1.0.0 at doi:10.5281/zenodo.23438462. Original software is MIT; group summaries, figures and documentation are CC BY 4.0.</p>')
    dump(dest/'.zenodo.json', metadata)
    license_text = (dest/'LICENSE.txt').read_text()
    (dest/'LICENSE.txt').write_text(license_text+'\nVersion 1.2.2: original software under code/ is MIT; group summaries, figures, documentation and descriptive metadata are CC BY 4.0. Attribute the authors and version at '+release_url+'.\n')
    columns = json.loads((dest/'COLUMN_DICTIONARY.json').read_text())
    columns.update(q_BH48='Saved Benjamini-Hochberg q across all 48 parcel slots within the specified cohort, phase and contrast.',
        q_BH4='Saved Benjamini-Hochberg q for the four continuing-sequence headline parcels within cohort and phase; other slots blank.',
        phase='For the continuing-sequence benchmark, early uses runs 3–4, middle uses runs 5–6, and the initial baseline uses runs 1–2; other files retain their source-defined phase labels.',
        continuing_mean='Continuing-sequence two-run summed difference in differences; blank when suppressed.',
        new_sequence_mean='Early new-sequence two-run summed difference in differences; blank when suppressed.',
        paired_difference_mean='Participant-paired early new-minus-continuing difference, twice the two-run mean Str2-minus-Str1; blank when suppressed.')
    dump(dest/'COLUMN_DICTIONARY.json', columns)
    permitted = {'README.txt','RELEASE_STATUS.txt','CITATION.cff','.zenodo.json','LICENSE.txt',
                 'COLUMN_DICTIONARY.json','FILE_DICTIONARY.csv','SHA256_manifest.json','CODE_REVIEW.csv',
                 'RELEASE_COVERAGE.csv','COVERAGE_STATUS.json','RELEASE_EXCLUSIONS.json',
                 'AGGREGATION_NOTES.json','BASE_PRESERVATION.json'}
    preserved = {rel:digest for rel,digest in base.items() if rel not in permitted and rel not in privacy}
    assert all((dest/rel).is_file() and sha(dest/rel) == digest for rel,digest in preserved.items()), 'Undeclared published-base change'
    dump(dest/'BASE_PRESERVATION.json', dict(commit=BASE_COMMIT, base_files=len(base),
        permitted_metadata_changes=sorted(permitted), privacy_exceptions=privacy, preserved_files=preserved))
    detail = {r['path']:r for r in csv.DictReader((dest/'FILE_DICTIONARY.csv').open())}
    for path in sorted(dest.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(dest))
        rows = len(pd.read_csv(path)) if path.suffix == '.csv' else ''
        if rel not in detail:
            detail[rel] = dict(path=rel, rows=rows, units='Source-defined / not applicable',
                description='Continuing-sequence benchmark group summary, verbatim source code or release documentation; see README and column definitions.')
        elif path.suffix == '.csv':
            detail[rel]['rows'] = rows
    detail['SHA256_manifest.json'] = dict(path='SHA256_manifest.json', rows='', units='bytes / SHA-256', description='All release files except the manifest itself.')
    detail['FILE_DICTIONARY.csv']['rows'] = len(detail)
    write_csv(dest/'FILE_DICTIONARY.csv', [detail[k] for k in sorted(detail)], ['path','rows','units','description'])
    for path in dest.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.json','.csv','.md','.txt','.cff','.tex'} and not str(path.relative_to(dest)).startswith('code/'):
            assert not inherited.PRIVATE.search(path.read_text()), 'Private content: '+str(path.relative_to(dest))
    manifest = inventory(dest)
    assert all(r['bytes'] < 100_000_000 for r in manifest)
    dump(dest/'SHA256_manifest.json', manifest)
    print(json.dumps(dict(files=len(manifest)+1, bytes=sum(r['bytes'] for r in inventory(dest)),
        base_files_preserved=len(preserved), base_privacy_exceptions=privacy, added_result_files=len(added),
        excluded_new_source_files=sum(excluded.values()), disclosure=disclosure, coverage=status), indent=2))


if __name__ == '__main__':
    main()
