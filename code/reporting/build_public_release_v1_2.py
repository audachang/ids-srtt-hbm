#!/usr/bin/env python3
"""Build v1.2.0 from an immutable public archive and reviewed group summaries.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
No analyses or network operations. Original inputs are never changed.
"""
import argparse
import ast
import collections
import csv
import datetime
from decimal import Decimal
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
from verify_public_summaries_r1 import disclosure_plan, disclosure_check, summary_blocks, SLOT_KEYS

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path('/tmp/rev-rel-build')
DEFAULT_DEST = ROOT / 'submissions/hbm_release_v1.2.0_candidate'
BASE_COMMIT = '603cdce29fb1df87e7eaa1d9b0b998616b2cc9bf'
IMAGING = 'results/glm_volume/review_r1_20261019'
JOINT = 'results/glm_volume/review_r1_20261019_joint_family'
BEHAVIOR = 'results/behavioral/review_r1_20261019'
SOURCE_ROOTS = [IMAGING, JOINT, BEHAVIOR]
PRIVATE = re.compile(r'/(?:home|media|tmp|dev/shm)/[^\s`\"\'<>),;]+|sub-\d{2,3}\b|\b(?:acl-super|node37)\b|(?<![.\w-])0[0-2]\d\b')
FORBIDDEN = {'subject','participant','participant_id','subject_id','sid','sub','omitted_subject','filename','file_path','path','original','repaired','raw_csv','raw_log','source_file'}
FIELDS = ['mean','sd','se','ci_low','ci_high','cohen_dz','t','df','p','estimate','probability']
CODE_HASHES = {
 'imaging_common.py':'2a80d045e0303f0627490fd7dc2c599be361ccf9593875085e07cc63e0dd49ef',
 'imaging_finalize.py':'69a53570f9936b98850872755d277faa6085f66ac27c133b5a9a31e2638d2943',
 'imaging_fits.py':'6826fa8143c5c4acec12f566fdf7c8d73f963f39dda668ae8edc58f7ff572e55',
 'imaging_preflight.py':'0c1c8062b36ea4d0c09461c29cc948f9246622db47b07eee54f0659cbe21ee1a',
 'imaging_summaries.py':'b82718395e451536b60a19506c3c7a44e06e50a2b731dcb7f95b533642e812b8',
 'imaging_validate.py':'200afde093d7953038796f28fcd811202faf90e947bc66d277c3910e227b90ae',
 'imaging_voxelwise.py':'a699857d8f4f03e6a3ac6e3e6375de05bbbe978c35e13e0c2e22d6b5d27760e3',
 'stats_cleaning_replay.R':'22b92ee31dfcbd674d14d714992631c800e1ba3c10b37a457d4b9be61dcf5c06',
 'stats_common.py':'4f35908022ffd60107da87d5d25bdeb40220ed3886e46c4a1c5e69ad10734bbd',
 'stats_design.py':'ba8d422bbc7ebdb9f2aec9d8e2c0b5d75ce34e55a56b8824661aa570665bc188',
 'stats_run.py':'05cbf6f378e90a04b4d69af724c5269e08a4b2e56386324b2e88f0107efbaea7',
 'stats_validate.py':'c1767e7a2856c21ac133b224b64708ca4f50af35491bee9af2fb30843c72087b',
}
# Explicit allowlist; directory discovery below is used only to document exclusions.
ALLOWLIST = []
for folder, names in [
 (IMAGING, ['REPORT.md','validation.json','completion_checks.json']),
 (IMAGING+'/roi', ['REPORT.md','validation.json','stage_results.csv','direct_phase_results.csv',
  'component_balance_results.csv','phase_component_results.csv','headline_early_component_balance.csv',
  'headline_tidy_all_specifications.csv','motion12_tissue_vs_reference.csv','motion6_acompcor8_vs_reference.csv',
  'motion12_acompcor8_vs_reference.csv','censored_spike_vs_reference.csv','repeat_regressor_vs_reference.csv',
  'censored_spike_vs_row_deletion.csv']),
 (IMAGING+'/event_motion_diagnostics', ['REPORT.md','validation.json','phase_motion_summary.csv',
  'repeat_counts_by_condition.csv','repeat_counts_by_run_condition.csv']),
 (IMAGING+'/voxelwise', ['REPORT.md','validation.json','permutation_summary.csv','spike_vs_row_deletion_counts.csv']),
 (IMAGING+'/validation', ['REPORT.md','validation.json']),
 (JOINT, ['REPORT.md','validation.json','joint_stage_results.csv','joint_only_discoveries.csv',
  'stage_discovery_survival.csv','family_discovery_counts.csv','duplicate_source_checks.csv']),
 (BEHAVIOR, ['REPORT.md','validation.json','independent_validation.json']),
 (BEHAVIOR+'/B1_participant_rt', ['REPORT.md','validation.json','cleaning_replay_validation.json','R_session_info.txt',
  'primary_contrast_comparison.csv','min100_contrast_comparison.csv','all_contrast_comparisons.csv',
  'primary_BH_family_slots.csv','min100_BH_family_slots.csv']),
 (BEHAVIOR+'/B2_repetition_behavior', ['REPORT.md','validation.json','primary_contrast_comparison.csv',
  'min100_contrast_comparison.csv','key_contrast_comparisons.csv','primary_BH_family_slots.csv','min100_BH_family_slots.csv',
  'D1_repeat_counts_by_condition.csv','D1_repeat_counts_by_condition_run.csv','trial_retention.csv',
  'descriptive_transition_category_advantage.csv']),
 (BEHAVIOR+'/B3_design_statistics', ['REPORT.md','validation.json','D1_repeat_rates_by_condition_run.csv',
  'cyclic_pair_triplet_statistics.csv','fixation_duration_frequencies.csv','fixation_duration_summary.csv',
  'order_variability_by_run.csv','outer_block_order_frequencies.csv','source_match_checks_by_run.csv',
  'target_frequencies_by_condition_run.csv','transition_frequencies_by_condition_run.csv',
  'transition_probability_matrices.csv','representative_run_timeline_totals.csv',
  'representative_run_timeline.pdf','representative_run_timeline.png']),
]: ALLOWLIST += [folder+'/'+name for name in names]
for cohort in ['N20','N22']:
 for contrast in ['initial','early','middle','late','pooled_r36_Str2_minus_Str1']:
  ALLOWLIST += [f'{IMAGING}/voxelwise/{cohort}/censored_spike/{contrast}/{name}' for name in ['REPORT.md','validation.json','components.csv']]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path, rows, fields=None):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def clean_json(value, removed, pointer=''):
    if isinstance(value,dict):
        out={}
        for key, item in value.items():
            loc=pointer+'/'+str(key).replace('~','~0').replace('/','~1')
            # File hashes remain historical provenance; participant file names do not.
            if PRIVATE.search(str(key)) or str(key).startswith('participant_') or (
                    not isinstance(item,(dict,list)) and PRIVATE.search(str(item))):
                removed.append(PRIVATE.sub('[withheld]',loc))
            else: out[key]=clean_json(item,removed,loc)
        return out
    if isinstance(value,list):
        out=[]
        for i,item in enumerate(value):
            loc=pointer+'/'+str(i)
            if PRIVATE.search(json.dumps(item)):
                removed.append(PRIVATE.sub('[withheld]',loc))
            else: out.append(clean_json(item,removed,loc))
        return out
    return value


def suppression(frame):
    """Suppress each statistic block using its own participant count; preserve q and N."""
    changed=[]
    counts=[c for c in frame if c=='n' or c in ['n_variant','n_reference','n_spike','n_row_deletion']]
    for count in counts:
        suffix=count[1:]
        fields=[f+suffix for f in FIELDS if f+suffix in frame]
        fields += [c for c in frame if c.startswith('mean_shift')]
        if count=='n':
            fields += [c for c in frame if c.startswith(('structured_','random_','condition_mean_')) and
                       c.endswith(('_mean','_ci_low','_ci_high','_p'))]
        mask=pd.to_numeric(frame[count],errors='coerce')<2
        cells=int(frame.loc[mask,fields].notna().sum().sum())
        if mask.any():
            frame.loc[mask,fields]=np.nan
            changed.append(dict(count_column=count,rows=int(mask.sum()),cells_suppressed=cells,columns=fields))
    return changed


def coverage_row(rel, facts, dest, decisions, source_hash):
    decision, reason=decisions.get(rel,('withhold','Source contains restricted observations or is outside the reviewed public file set.'))
    verified=0
    if decision!='withhold':
        path=dest/rel
        if path.suffix=='.csv':
            frame=pd.read_csv(path);indexes={}
            def identity(value):
                if pd.isna(value) or value=='':return ''
                value=str(value)
                if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',value):
                    return str(Decimal(value).normalize())
                return value
            for fact in facts:
                keys=fact['row_key'];col=fact['column']
                if not isinstance(col,str) or col not in frame or not set(keys)<=set(frame):continue
                names=tuple(sorted(keys))
                if names not in indexes:
                    ix={}
                    for row in frame.to_dict('records'):
                        ix.setdefault(tuple(identity(row[k]) for k in names),[]).append(row)
                    indexes[names]=ix
                rows=indexes[names].get(tuple(identity(keys[k]) for k in names),[])
                if len(rows)!=1:continue
                actual=rows[0][col];expected=fact['value']
                if pd.isna(actual):ok=expected is None
                elif isinstance(expected,(int,float)):ok=np.isclose(float(actual),expected,rtol=1e-10,atol=1e-13)
                else:ok=str(actual)==str(expected)
                verified+=int(ok)
        elif path.suffix=='.json':
            obj=json.loads(path.read_text())
            for fact in facts:
                try:
                    item=obj
                    for key in fact['row_key']['json_pointer'].strip('/').split('/'):
                        if key:item=item[int(key)] if isinstance(item,list) else item[key.replace('~1','/').replace('~0','~')]
                    if isinstance(item,dict):item=item[fact['column']]
                    verified+=int(item==fact['value'])
                except (KeyError,TypeError,ValueError,IndexError):pass
    # A ledger can reference restricted participant-level provenance. Account for
    # that source without publishing its identifier in the coverage inventory.
    public_source=rel
    if PRIVATE.search(rel):
        assert decision=='withhold','Private source must remain withheld'
        public_source='withheld-source/'+hashlib.sha256(rel.encode()).hexdigest()+'/'+Path(rel).name
    return dict(source=public_source,source_sha256=source_hash,decision=decision,reason=reason,
                ledger_entries=len(facts),verifiable_entries=verified,withheld_entries=len(facts)-verified,
                release_path=rel if decision!='withhold' else '')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',type=Path,default=DEFAULT_DEST)
    parser.add_argument('--release-date',required=True)
    parser.add_argument('--doi')
    parser.add_argument('--commit')
    parser.add_argument('--refresh-owned-candidate',action='store_true')
    parser.add_argument('--incomplete-candidate-sha256',help='Recover a failed owned build only when its full current file inventory matches this digest.')
    args=parser.parse_args()
    try: date=datetime.date.fromisoformat(args.release_date).isoformat()
    except ValueError:parser.error('--release-date must be YYYY-MM-DD')
    if date!=args.release_date:parser.error('--release-date must be YYYY-MM-DD')
    if bool(args.doi)!=bool(args.commit):parser.error('--doi and --commit must be supplied together')
    if args.doi and not re.fullmatch(r'10\.\d{4,9}/[^\s<>]+',args.doi):parser.error('Invalid DOI')
    if args.commit and not re.fullmatch(r'[0-9a-fA-F]{40}',args.commit):parser.error('Full 40-character commit required')
    SCRATCH.mkdir(parents=True,exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER']=str(SCRATCH/'joblib')
    os.environ['MPLCONFIGDIR']=str(SCRATCH/'matplotlib')
    dest=args.dest.resolve()
    if dest!=DEFAULT_DEST and (not dest.is_relative_to(SCRATCH) or dest==SCRATCH):
        raise ValueError('Destination must be the owned candidate or a child of the assigned scratch directory')
    if dest.exists() and any(dest.iterdir()):
        if not args.refresh_owned_candidate or dest!=DEFAULT_DEST:raise FileExistsError(dest)
        if not (dest/'SHA256_manifest.json').is_file():
            assert args.incomplete_candidate_sha256,'Incomplete candidate must be explicitly fingerprinted before recovery'
            manifest=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p))
                      for p in sorted(dest.rglob('*')) if p.is_file()]
            fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert fingerprint==args.incomplete_candidate_sha256,'Incomplete candidate changed; preserve it'
            expected={r['path'] for r in manifest}
        else:
            assert not args.incomplete_candidate_sha256,'Recovery flag is only for incomplete builds'
            manifest=json.loads((dest/'SHA256_manifest.json').read_text())
            expected={r['path'] for r in manifest}|{'SHA256_manifest.json'}
        assert {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}==expected
        assert not any(p.is_symlink() for p in dest.rglob('*'))
        for row in manifest:
            p=dest/row['path'];assert p.resolve().is_relative_to(dest) and sha(p)==row['sha256']
        for rel in sorted(expected):(dest/rel).unlink()
        for p in sorted((p for p in dest.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
    dest.mkdir(parents=True,exist_ok=True)
    archive=subprocess.run(['git','-C',str(ROOT/'submissions/hbm_release_v1.0.0'),
                            'archive','--format=tar',BASE_COMMIT],check=True,capture_output=True).stdout
    with tempfile.TemporaryDirectory(prefix='archive-',dir=SCRATCH) as temporary:
        old=Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(old,filter='data')
        base_files={str(p.relative_to(old)):sha(p) for p in old.rglob('*') if p.is_file()}
        shutil.copytree(old,dest,dirs_exist_ok=True)
    (dest/'SHA256_manifest.json').unlink()
    notes=json.loads((dest/'AGGREGATION_NOTES.json').read_text())
    exclusions=json.loads((dest/'RELEASE_EXCLUSIONS.json').read_text())
    legacy_coverage=pd.read_csv(dest/'RELEASE_COVERAGE.csv')
    shutil.copyfile(dest/'RELEASE_COVERAGE.csv',dest/'RELEASE_COVERAGE_V1_1.csv')
    decisions={r.source:(r.decision,r.reason) for r in legacy_coverage.itertuples()}
    source_hashes={r.source:r.source_sha256 for r in legacy_coverage.itertuples()}
    added=[];figure_review={};suppressed=[]
    for rel in sorted(ALLOWLIST):
        source=ROOT/rel;target=dest/rel
        assert source.is_file(),rel
        target.parent.mkdir(parents=True,exist_ok=True)
        source_hashes[rel]=sha(source)
        operation=[]
        if source.suffix=='.csv':
            frame=pd.read_csv(source)
            assert not (set(frame.columns.str.lower())&FORBIDDEN),rel
            assert not PRIVATE.search(source.read_text()),rel
            if source.name=='representative_run_timeline.csv':
                removed=['logged_start_s','logged_end_s']
                frame=frame.drop(columns=removed)
                operation.append(dict(operation='column removal',columns=removed,
                                      reason='Withhold individual logged timing precision; retain the nonidentifying design illustration.'))
            # Frequency cells based on a single participant do not disclose their duration.
            if source.name=='fixation_duration_frequencies.csv':
                mask=frame.n_participants.lt(2)
                if mask.any():
                    frame.loc[mask,'nominal_duration_s']=np.nan
                    operation.append(dict(operation='suppression',rows=int(mask.sum()),columns=['nominal_duration_s'],
                                          reason='Singleton design-frequency cells; counts retained.'))
            changes=suppression(frame)
            if changes:
                suppressed.append(dict(path=rel,blocks=changes))
                operation.append(dict(operation='suppression',blocks=changes,reason='N<2; keep sample sizes, planned slots and BH q values.'))
            target.write_text(frame.to_csv(index=False)) if operation else shutil.copyfile(source,target)
        elif source.suffix=='.json':
            removed=[];obj=clean_json(json.loads(source.read_text()),removed)
            if removed:
                dump(target,obj);operation.append(dict(operation='redaction',removed_json_pointers=removed,
                                                      reason='Withhold identifiers, private paths and individual-file provenance.'))
            else:shutil.copyfile(source,target)
        elif source.suffix.lower() in {'.md','.txt'}:
            lines=source.read_text().splitlines(keepends=True);removed=[]
            for i,line in enumerate(lines):
                if PRIVATE.search(line):lines[i]='[Line withheld: participant identifier or private path.]\n';removed.append(i+1)
            target.write_text(''.join(lines))
            if removed:operation.append(dict(operation='redaction',line_numbers=removed,reason='Withhold identifiers and private paths.'))
        elif source.suffix in {'.png','.pdf'}:
            from PIL import Image
            expected={'representative_run_timeline.pdf':'e282ed7a9fa595c219814b9d92f549d601b0cf09bfb9ed77b71febc1aa2c2264',
                      'representative_run_timeline.png':'056b23614ffc878fdbe2452190988f42fc81697664679ead47f7b4b40e757e7e'}
            assert sha(source)==expected[source.name],'Figure changed since review'
            if source.suffix=='.pdf':
                metadata=subprocess.run(['pdfinfo',str(source)],capture_output=True,text=True,check=True).stdout
            else:
                with Image.open(source) as im:metadata=json.dumps(im.info)
            assert not PRIVATE.search(metadata),rel
            shutil.copyfile(source,target)
            figure_review[rel]=dict(source_sha256=sha(source),release_sha256=sha(target),pages=1,
                                   review='Nonidentifying design illustration only; no response traces, participant labels or private metadata.')
        else:raise ValueError(rel)
        for entry in operation:notes.append(dict(path=rel,source_sha256=sha(source),release_sha256=sha(target),**entry))
        decisions[rel]=('include with suppression/redaction' if operation else 'include verbatim',
                        'Reviewed group summary; privacy changes are described in AGGREGATION_NOTES.json.' if operation else 'Reviewed group summary or nonidentifying design illustration.')
        added.append(rel)
    # Exhaustive per-file exclusions, without disclosing paths containing participant IDs.
    excluded_groups=collections.Counter()
    explicit_exclusions=[]
    for root in SOURCE_ROOTS:
        for source in sorted((ROOT/root).rglob('*')):
            if not source.is_file():continue
            rel=str(source.relative_to(ROOT))
            if rel in ALLOWLIST:continue
            if source.name=='representative_run_timeline.csv':reason='Single-participant realized run schedule withheld; the figure is an example stimulus-design schedule only, without responses.'
            elif 'first_level' in source.parts:reason='First-level individual outputs and input provenance.'
            elif source.name.startswith('participant_'):reason='Individual observations, trial records or bootstrap identities.'
            elif source.name in ['fixation_pattern_frequencies.csv','physical_subblock_order_frequencies.csv','role_subblock_order_frequencies.csv','whole_run_role_order_frequencies.csv']:
                reason='Complete realized orders or singleton design patterns can reveal individual run designs.'
            elif source.suffix in {'.nii','.gz','.npz','.npy'}:reason='MRI images, participant arrays or raw permutation nulls.'
            else:reason='Restricted input provenance, individual validation details or unreviewed outputs.'
            excluded_groups[reason]+=1
            if not PRIVATE.search(rel):explicit_exclusions.append(dict(path=rel,reason=reason,historical=False))
    exclusions.extend(explicit_exclusions)
    exclusions.extend(dict(path=root+'/',reason=reason,files=count,detail='Files with individual identifiers in their names are counted rather than named.')
                      for reason,count in sorted(excluded_groups.items()) for root in ['review sensitivity outputs'])
    notes.append(dict(path=BEHAVIOR+'/B3_design_statistics/representative_run_timeline.csv',operation='withholding',
                      source_sha256=sha(ROOT/BEHAVIOR/'B3_design_statistics/representative_run_timeline.csv'),
                      reason='Single-participant realized run schedule; retain only the authorized example stimulus-design figure, without responses.'))
    # Reviewed new source is copied verbatim, pinned to the reviewed bytes.
    review=list(csv.DictReader((dest/'CODE_REVIEW.csv').open()))
    extra_fields=['origin_sha256','origin_line_range','release_sha256']
    for record in review:
        for field in extra_fields:record[field]=''
    assert {p.name for p in (ROOT/'code/review_r1').iterdir() if p.suffix.lower() in {'.py','.r'}}==set(CODE_HASHES)
    for name,expected in sorted(CODE_HASHES.items()):
        rel='code/review_r1/'+name;source=ROOT/rel
        assert sha(source)==expected,'Unreviewed source change: '+rel
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        review.append(dict(path=rel,source_sha256=expected,credentials_or_tokens='none found',embedded_participant_data='none found',
                           decision='include verbatim',review='Source inspected for secrets and literal individual measurements. Input labels, paths and component-selection assertions refer to restricted inputs.',
                           origin_sha256='',origin_line_range='',release_sha256=expected))
    original=ROOT/'code/reanalysis/run_saved.py';raw=original.read_text()
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='direct')
    function=''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    helper=('# The original script is withheld because it embeds individual observations.\n'
            '# This file provides only the generic direct phase-comparison function so\n'
            '# that released scripts importing it can run with approved restricted inputs.\n'
            'import numpy as np\nimport pandas as pd\nfrom common import ROIS, summarize, bh\n\n'+function)
    rel='code/reanalysis/run_saved.py';(dest/rel).write_text(helper)
    record=next(r for r in review if r['path']==rel)
    record.update(source_sha256=sha(dest/rel),credentials_or_tokens='none found',embedded_participant_data='none found',
                  decision='include verbatim',review='Reviewed public helper only. Generic direct function copied verbatim; full source containing individual observations remains withheld.',
                  origin_sha256=sha(original),origin_line_range=f'{node.lineno}-{node.end_lineno}',release_sha256=sha(dest/rel))
    notes.append(dict(path=rel,operation='generic function extraction',source_sha256=sha(original),
                      source_line_range=[node.lineno,node.end_lineno],release_sha256=sha(dest/rel),reason='Only direct and its necessary imports are distributed; the complete original script remains withheld.'))
    for record in exclusions:
        if record.get('path')==rel:record.update(reason='Full original source remains withheld because it embeds individual observations; the released file is a generic-function-only derivative.',scope='full original source only')
    imported=[]
    for path in sorted((dest/'code').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module=='run_saved':
                imported.append(dict(path=str(path.relative_to(dest)),line=node.lineno,names=[alias.name for alias in node.names]))
            elif isinstance(node,ast.Import) and any(alias.name=='run_saved' for alias in node.names):
                raise ValueError('Unreviewed module import: '+str(path))
    assert all(r['names']==['direct'] for r in imported),imported
    for name in ['build_public_release_v1_2.py','verify_public_summaries_r1.py']:
        rel='code/reporting/'+name;shutil.copyfile(ROOT/rel,dest/rel)
        review.append(dict(path=rel,source_sha256=sha(ROOT/rel),credentials_or_tokens='none found',embedded_participant_data='none found',
                           decision='include verbatim',review='Release assembly or aggregate verification only; no credentials or embedded individual observations.',
                           origin_sha256='',origin_line_range='',release_sha256=sha(ROOT/rel)))
    write_csv(dest/'CODE_REVIEW.csv',sorted(review,key=lambda r:r['path']))
    dump(dest/'FIGURE_REVIEW_R1.json',figure_review)
    dump(dest/'AGGREGATION_NOTES.json',notes)
    dump(dest/'RELEASE_EXCLUSIONS.json',exclusions)
    # Complementary suppression applies to inherited and new files together.
    # Coarse ROI/sample blocks also suppress linear component reconstructions.
    frames,plan,nested_relations=disclosure_plan(dest)
    privacy_exceptions={};suppression_records=[]
    for rel,frame in frames.items():
        records=[];before=sha(dest/rel)
        for block in summary_blocks(rel,frame):
            if (rel,block['index'],block['suffix']) not in plan:continue
            fields=list(block['fields'])+[c for c in frame if c.startswith('mean_shift')]
            if block['suffix']=='':
                fields += [c for c in frame if c.startswith(('structured_','random_','condition_mean_')) and '_q' not in c]
                fields += [c for c in ['component_wording','direction'] if c in frame]
            fields=sorted(set(fields))
            if not frame.loc[block['index'],fields].notna().any():continue
            row=frame.loc[block['index']]
            keys={k:str(row[k]) for k in ['cohort','model','policy','roi']+SLOT_KEYS if k in frame}
            records.append(dict(row_key=keys,csv_line=int(block['index'])+2,
                                statistic_block=block['suffix'] or 'main',columns=fields))
            frame.loc[block['index'],fields]=np.nan
        if records:
            frame.to_csv(dest/rel,index=False)
            notes.append(dict(path=rel,operation='complementary suppression',
                              source_sha256=before,release_sha256=sha(dest/rel),blocks=records,
                              reason='Nested samples can differ by one observation. Smaller and equivalent sets lose all statistics; related ROI components and contrasts are suppressed together. N, planned slots and BH q remain.'))
            suppression_records.append(dict(path=rel,blocks=len(records)))
            decisions[rel]=('include with suppression/redaction','Statistics suppressed to prevent recovery from nested samples and coverage rules; N and BH q retained.')
            if rel in base_files:privacy_exceptions[rel]='complementary suppression'
    # Derived displays are withheld rather than allowing plots, tables, ranges or
    # textual ROI tables to restore suppressed statistics. Anatomical maps show
    # voxelwise results and do not display ROI sample means.
    withheld=[]
    for path in sorted(dest.rglob('*')):
        if not path.is_file() or str(path.relative_to(dest)).startswith('code/'):continue
        rel=str(path.relative_to(dest))
        roi_display=(path.suffix in {'.pdf','.png'} and 'anatomical_display/' not in rel
                     and 'representative_run_timeline' not in rel)
        derived_table=rel in ['tables/observed_significant_rois.json','tables/observed_significant_rois.tex',
            'results/glm_volume/reporting_20261012/model_stability.csv']
        roi_report=(path.name=='REPORT.md' and ('/roi' in rel or '/figures/' in rel or '/targeted_sensitivity/' in rel or rel.startswith(JOINT+'/') or rel==IMAGING+'/REPORT.md'))
        summary=rel=='results/reanalysis_20261018_updated_n22/SUMMARY.md'
        if not (roi_display or derived_table or roi_report or summary):continue
        reason='Derived ROI display or report could expose statistics suppressed in the CSVs; withheld in this version.'
        notes.append(dict(path=rel,operation='withholding derived display',source_sha256=sha(path),reason=reason))
        exclusions.append(dict(path=rel,reason=reason,historical=rel in base_files))
        decisions[rel]=('withhold',reason);withheld.append(rel);path.unlink()
        if rel in base_files:privacy_exceptions[rel]='withholding derived display'
    added=[rel for rel in added if (dest/rel).exists()]
    old_figures=json.loads((dest/'FIGURE_REVIEW.json').read_text())
    dump(dest/'FIGURE_REVIEW.json',{rel:r for rel,r in old_figures.items() if (dest/rel).exists()})
    # Keep the original historical verifier byte-identical. Exact BH recomputation
    # needs withheld p values; the new verifier checks admissible BH bounds for
    # affected contracts. All complete families retain the original exact check.
    contract_path=dest/'AGGREGATE_CHECK_SPEC.json'
    contracts=json.loads(contract_path.read_text())
    for contract in contracts['contracts']:
        frame=pd.read_csv(dest/contract['path'])
        if 'n' in frame and 'p' in frame and (frame.n.ge(2)&frame.p.isna()).any():
            contract['suppression_bounds_families']=contract['families'];contract['families']=[]
    contracts['suppression_note']='Exact BH checks use complete families only. Withheld p values are checked by admissible BH bounds in verify_public_summaries_r1.py, with all planned slots retained.'
    dump(contract_path,contracts)
    notes.append(dict(path='AGGREGATE_CHECK_SPEC.json',operation='verification specification update',
                      reason=contracts['suppression_note']))
    # Inherited non-code reports are subject to the runtime-path rule too.
    for path in sorted(dest.rglob('*')):
        if not path.is_file() or str(path.relative_to(dest)).startswith('code/'):continue
        rel=str(path.relative_to(dest));before=sha(path);removed=[]
        if path.suffix=='.json':
            obj=clean_json(json.loads(path.read_text()),removed)
            if removed:dump(path,obj)
        elif path.suffix in {'.md','.txt'}:
            lines=path.read_text().splitlines(keepends=True)
            for i,line in enumerate(lines):
                if PRIVATE.search(line):lines[i]='[Line withheld: private path or identifier.]\n';removed.append(i+1)
            if removed:path.write_text(''.join(lines))
        if removed:
            notes.append(dict(path=rel,operation='redaction',source_sha256=before,release_sha256=sha(path),
                              removed_keys_or_lines=removed,reason='Withhold private runtime paths and identifiers.'))
            if rel in base_files:privacy_exceptions[rel]='private-path redaction'
    disclosure=disclosure_check(dest)
    dump(dest/'AGGREGATION_NOTES.json',notes)
    dump(dest/'RELEASE_EXCLUSIONS.json',exclusions)
    # Coverage always uses the current full source-of-record; absence of the asset report is explicit.
    ledger_bytes=(ROOT/'docs/t9/facts_n22.json').read_bytes()
    ledger=json.loads(ledger_bytes);facts_by_source=collections.defaultdict(list)
    for fact in ledger:facts_by_source[fact['source']].append(fact)
    coverage=[]
    for rel,facts in sorted(facts_by_source.items()):
        if rel not in source_hashes:source_hashes[rel]=sha(ROOT/rel)
        coverage.append(coverage_row(rel,facts,dest,decisions,source_hashes[rel]))
    write_csv(dest/'RELEASE_COVERAGE.csv',coverage)
    aliases=sum(PRIVATE.search(rel) is not None for rel in facts_by_source)
    if aliases:
        notes.append(dict(path='RELEASE_COVERAGE.csv',operation='redaction',
                          source_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
                          release_sha256=sha(dest/'RELEASE_COVERAGE.csv'),
                          fields=['source'],restricted_source_references=aliases,
                          reason='Replace restricted participant source filenames with opaque SHA-256 references; their facts remain withheld.'))
        dump(dest/'AGGREGATION_NOTES.json',notes)
    ready=(ROOT/'docs/review_round1/reports/rev-assets.md').is_file()
    coverage_status=dict(complete=ready,source_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
                         source_files=len(coverage),recorded_values=len(ledger),
                         verifiable_values=sum(r['verifiable_entries'] for r in coverage),
                         withheld_values=sum(r['withheld_entries'] for r in coverage),
                         description='Coverage uses the completed article value inventory.' if ready else 'Refresh coverage after article assets are completed.')
    dump(dest/'COVERAGE_STATUS.json',coverage_status)
    dump(dest/'AGGREGATE_CHECK_SPEC_R1.json',dict(description='Saved aggregate arithmetic only; no fitting or new inference.',
         added_results=sorted(added),suppression=suppressed,helper_imports=imported,missing_helper_functions=[],
         planned_BH_slots=dict(stage=48,joint=192,direct=144,headline=4,behavior=24),
         complementary_suppression=suppression_records,disclosure=disclosure,withheld_derived_displays=withheld))
    release_url='https://github.com/audachang/ids-srtt-hbm/releases/tag/v1.2.0'
    cff=(dest/'CITATION.cff').read_text()
    cff='\n'.join(line for line in cff.splitlines() if not line.startswith(('doi:','commit:')))+'\n'
    cff=re.sub(r'^version:.*$','version: 1.2.0',cff,flags=re.M)
    cff=re.sub(r'^date-released:.*$','date-released: '+date,cff,flags=re.M)
    cff=cff.replace('releases/tag/v1.1.0','releases/tag/v1.2.0')
    cff=re.sub(r'^message:.*$','message: "Please cite the version DOI when available; otherwise cite the release URL."',cff,flags=re.M)
    if args.doi:cff+='doi: "'+args.doi+'"\ncommit: "'+args.commit+'"\n'
    (dest/'CITATION.cff').write_text(cff)
    metadata=json.loads((dest/'.zenodo.json').read_text());metadata.update(version='1.2.0',publication_date=date);metadata.pop('doi',None)
    metadata['description']=('<p>Analysis source code and group-level imaging and behavioral summaries for an interleaved dual-structure serial reaction time fMRI study.</p>'
      '<p>Version 1.2.0 adds sensitivity analyses requested in review: nuisance-model decomposition, spike-regressor censoring, immediate-repetition sensitivity, joint stage correction, condition-mean component balance, participant-level reaction-time intervals, design statistics and motion by condition. These findings remain exploratory. The 22-participant uncensored reference analysis with at least 80% observed regional support remains primary. The overlapping learner-selected 20-participant sample is a sensitivity reported only in the article supplement. Confidence intervals are pointwise. The analysis history and model selection are disclosed in the article Methods.</p>'
      '<p>Participant-level observations, fitted models, MRI images, raw permutation arrays and recoverable individual values are withheld. The complete original run_saved.py is withheld; only its generic direct comparison function and necessary imports are distributed. Code labels and paths refer to restricted inputs. Approved data access and separate dependencies are required to re-estimate models.</p>'
      '<p>Version 1.2.0 applies stricter suppression to overlapping sample and coverage summaries and supersedes version 1.1.0 for citation. Derived ROI displays that could restore suppressed values are withheld.</p><p>Original software is MIT; group summaries, figures and documentation are CC BY 4.0. Version 1.1.0 remains available at doi:10.5281/zenodo.23471923 and version 1.0.0 at doi:10.5281/zenodo.23438462.</p>')
    dump(dest/'.zenodo.json',metadata)
    status=('Version 1.2.0 published on '+date+'.\nVersion DOI: https://doi.org/'+args.doi+'\nFrozen commit: '+args.commit+'\n') if args.doi else (
        'Version 1.2.0 prepared on '+date+'.\nThe version DOI will be linked after archival completes.\n')
    previous='Version 1.1.0: doi:10.5281/zenodo.23471923.\nVersion 1.0.0: doi:10.5281/zenodo.23438462.\n'
    (dest/'RELEASE_STATUS.txt').write_text(status+'GitHub release: '+release_url+'\n'+previous+'Participant-level data are not distributed.\nOriginal software: MIT. Group summaries and documentation: CC BY 4.0.\n')
    (dest/'README.txt').write_text(f'''IDS-SRTT HBM companion — v1.2.0 ({date})

Analysis source code and group summaries only.
Public repository: https://github.com/audachang/ids-srtt-hbm
GitHub release: {release_url}
{status}{previous}
What's new in 1.2.0

Sensitivity analyses requested in review address nuisance-model decomposition,
spike-regressor censoring, immediate-repetition sensitivity, joint stage
correction, condition-mean component balance, participant-level reaction-time
intervals, design statistics and motion by condition. These are exploratory
sensitivities and secondary analyses. Version 1.2.0 applies stricter suppression
to group summaries of overlapping samples and coverage rules, including summaries
inherited from v1.1.0. Version 1.2.0 supersedes v1.1.0 for citation. Earlier code
remains unchanged; affected statistics and derived ROI displays are withheld.
The 22-participant uncensored reference analysis
with at least 80% observed regional support remains primary. The overlapping
learner-selected 20-participant subsample appears only in the supplement.
Intervals are pointwise. Analysis choices followed earlier observations;
see the article Methods for the disclosed model-selection history.

Verify the supplied summaries (NumPy, pandas and SciPy required):
  python -B code/reporting/verify_public_summaries_r1.py .
  python -B code/reporting/verify_public_summaries_r1.py --legacy .
The first command checks new aggregate arithmetic, t intervals, back-transforms,
48/192/144/4-slot imaging and 24-slot behavioral BH families, suppressed cells,
file inventory and coverage totals. It does not fit models or generate new
scientific results. Bootstrap intervals are checked for ordering and their
back-transforms; their estimation requires withheld observations.
Families containing withheld p values are checked against admissible BH bounds;
exact q recomputation is limited to complete families. The saved q values and
planned slots are retained. All CSVs are checked for nested-sample disclosure.
The second runs the unchanged v1.1.0 verifier using its preserved original
coverage table, RELEASE_COVERAGE_V1_1.csv. Its original direct command assumes
the former coverage size and does not support the expanded coverage table.
Its checksum, available arithmetic, code, retained-figure and suppression checks run
against this release. Its family specification assigns incomplete BH families
to the new verifier's bounds check. Neither command verifies the manuscript or model fits.

FILE_DICTIONARY.csv describes every file. COLUMN_DICTIONARY.json explains
statistical units; AGGREGATE_CHECK_SPEC_R1.json specifies the added checks.
RELEASE_COVERAGE.csv accounts for {coverage_status['recorded_values']:,} article
values from {coverage_status['source_files']:,} sources, of which
{coverage_status['verifiable_values']:,} values can be checked against public files.
COVERAGE_STATUS.json records completeness and the source inventory checksum.
Restricted source filenames use opaque references in the coverage table;
their recorded values remain withheld and their source checksums are retained.
Source hashes in saved validation reports describe original outputs;
SHA256_manifest.json gives current release hashes for all files except itself.
AGGREGATION_NOTES.json records redactions and suppression. For N<2, estimates
and test values are blank; N, planned slots and BH q are retained. Blank means
missing. Smaller nested sets differing by one observation, equivalent displays,
and their related ROI components and contrasts have all statistics blanked;
N and BH q remain. Affected ROI figures and derived tables/reports are withheld.
Singleton design-frequency durations and exact individual timing
precision are withheld. Complete realized order tables with singleton patterns
are excluded. The single-run timeline CSV is withheld. The retained timeline
figure is an example run schedule showing stimulus design only, with no responses.

Restricted inputs include raw/preprocessed MRI, participant maps and masks,
confounds, censor vectors, trial logs, individual summaries and random effects,
fitted models, bootstrap identities, raw nulls and participant manifests.
Leave-one-out values are withheld because individual observations are recoverable.
Seven complete original scripts embedding individual observations remain
withheld. The released code/reanalysis/run_saved.py contains only the generic
direct comparison function copied verbatim, with necessary imports. Its source
checksum and line range are in CODE_REVIEW.csv; no other imported function
from that original is missing. Subject labels and paths in code refer to
restricted inputs that are not distributed. Re-estimation requires approved
input access, subject-list configuration, the appropriate SRTT_SOURCE_ROOT,
separate software and atlas/template resources, and any additional restricted
helpers needed for the chosen analysis. Public summaries alone cannot rerun
participant models or the original article asset-generation scripts.

Original software under code/ is MIT. Group summaries, figures, documentation,
environment records and descriptive metadata are CC BY 4.0. See LICENSE.txt
and LICENSES/; third-party resources retain their own terms. Requests for
participant data go to the corresponding authors and depend on participant
consent, ethics approval and institutional requirements. Access is not granted
by this release and is not guaranteed.
''')
    license_text=(dest/'LICENSE.txt').read_text()
    license_text+='\nVersion 1.2.0: original software under code/ is MIT; group summaries, figures, documentation and descriptive metadata are CC BY 4.0. Attribute the authors and version at '+release_url+'.\n'
    (dest/'LICENSE.txt').write_text(license_text)
    columns=json.loads((dest/'COLUMN_DICTIONARY.json').read_text())
    columns.update(q_joint_stages_192='BH across 4 stages x 48 regions per cohort/model/policy; missing p is 1.',
                   q_joint_192='BH across 4 stages x 48 regions per cohort/model/policy.',
                   q_headline_early_4='BH across the four displayed early regions per cohort/model/policy/component.',
                   condition_mean='(structured change + random change)/2, phase-mean coefficient units.',
                   bootstrap_log_ci_low='Saved whole-participant percentile interval lower limit; natural log RT contrast.',
                   n_participants='Participant count contributing to a group design summary; singleton duration cells are blank.',
                   q_bh_24='BH across 24 behavioral contrasts per RT policy; untested slots have p=1.')
    dump(dest/'COLUMN_DICTIONARY.json',columns)
    permitted_changes={'README.txt','RELEASE_STATUS.txt','CITATION.cff','.zenodo.json','LICENSE.txt','COLUMN_DICTIONARY.json',
                       'FILE_DICTIONARY.csv','SHA256_manifest.json','CODE_REVIEW.csv','RELEASE_COVERAGE.csv','RELEASE_EXCLUSIONS.json','AGGREGATION_NOTES.json',
                       'FIGURE_REVIEW.json','AGGREGATE_CHECK_SPEC.json'}
    preserved={rel:digest for rel,digest in base_files.items() if rel not in permitted_changes and rel not in privacy_exceptions}
    assert all((dest/rel).is_file() and sha(dest/rel)==digest for rel,digest in preserved.items()),'Published base changed'
    dump(dest/'BASE_PRESERVATION.json',dict(commit=BASE_COMMIT,base_files=len(base_files),
         permitted_metadata_changes=sorted(permitted_changes),privacy_exceptions=privacy_exceptions,preserved_files=preserved))
    detail={r['path']:r for r in csv.DictReader((dest/'FILE_DICTIONARY.csv').open()) if (dest/r['path']).is_file()}
    for path in sorted(dest.rglob('*')):
        if not path.is_file():continue
        rel=str(path.relative_to(dest))
        if rel in detail and path.suffix=='.csv':detail[rel]['rows']=len(pd.read_csv(path))
        if 'representative_run_timeline.' in rel and path.suffix in {'.png','.pdf'}:
            detail[rel]=dict(path=rel,rows='',units='seconds / stimulus design',description='Example run schedule; stimulus design only, no responses. The single-run timeline CSV is withheld.')
        if rel not in detail:
            rows=len(pd.read_csv(path)) if path.suffix=='.csv' else ''
            detail[rel]=dict(path=rel,rows=rows,units='Source-defined / not applicable',
                description='Exploratory sensitivity group summary, design illustration, source code or release documentation; see README and column definitions.')
    detail['SHA256_manifest.json']=dict(path='SHA256_manifest.json',rows='',units='bytes / SHA-256',description='All release files except the manifest itself.')
    detail['FILE_DICTIONARY.csv']['rows']=len(detail)
    write_csv(dest/'FILE_DICTIONARY.csv',[detail[k] for k in sorted(detail)],['path','rows','units','description'])
    for path in dest.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.json','.csv','.md','.txt','.cff','.tex'} and not str(path.relative_to(dest)).startswith('code/'):
            assert not PRIVATE.search(path.read_text()),'Private content: '+str(path.relative_to(dest))
    manifest=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert all(r['bytes']<100_000_000 for r in manifest)
    dump(dest/'SHA256_manifest.json',manifest)
    print(json.dumps(dict(files=len(manifest)+1,bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
         base_preserved=len(preserved),added_result_files=len(added),excluded_review_source_files=sum(excluded_groups.values()),
         coverage=coverage_status),indent=2))


if __name__=='__main__':main()
