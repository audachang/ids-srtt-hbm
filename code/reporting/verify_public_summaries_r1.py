#!/usr/bin/env python3
"""Verify saved public sensitivity aggregates; never fit participant models.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
Usage: python -B verify_public_summaries_r1.py [--legacy] RELEASE_ROOT
Requires NumPy, pandas, SciPy. Does not write files.
"""
import argparse
import collections
import itertools
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np
import pandas as pd
from scipy import stats
sys.dont_write_bytecode = True

IMAGING='results/glm_volume/review_r1_20261019'
JOINT='results/glm_volume/review_r1_20261019_joint_family'
BEHAVIOR='results/behavioral/review_r1_20261019'
HEADLINE=['LeftSMA','RightSMA','RightPallidum','RightThalamusProper']
FIELDS=['mean','sd','se','ci_low','ci_high','cohen_dz','t','df','p','estimate','probability']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



# Sample policies are nested in this order. Equal-size nested sets are identical,
# so duplicate displays under either policy must receive the same suppression.
POLICY_RANK={'complete100':0,'observed95':1,'observed80':2,'legacy_full_parcel':3,
             'min100':0,'primary':1}
COHORT_RANK={'N20':0,'N22':1}
SLOT_KEYS=['stage','pair','component','phase','metric','contrast','run','condition',
           'displayed_cycle','session','block_kind','block_number','rule_group',
           'transition_category','position','target','previous_target','outer_block_order']


def summary_blocks(rel, frame):
    """Enumerate statistic blocks without using their values as identity keys."""
    for index,row in frame.iterrows():
        for suffix in ['', '_variant','_reference','_spike','_row_deletion']:
            count='n'+suffix
            if suffix=='' and count not in frame:
                count=next((c for c in ['n_subjects','n_participants'] if c in frame),count)
            if count not in frame:continue
            fields=[c+suffix for c in FIELDS if c+suffix in frame]
            if suffix=='':
                fields += [c for c in frame if (c.startswith(('mean_','log_','ratio_','geometric_','bootstrap_log_ci_','bootstrap_ratio_ci_'))
                          or c.endswith(('_mean','_sd','_se','_ci_low','_ci_high')))
                          and c not in SLOT_KEYS and not c.startswith('n_') and '_q' not in c]
                fields += [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])
                          and any(token in c for token in ['mean','difference','_rate','probability','_ratio','_ci_','_sd','_se'])
                          and c not in SLOT_KEYS and not c.startswith(('n_','q_')) and '_q' not in c
                          and c not in ['bootstrap_resamples','bootstrap_seed']]
            fields=sorted(set(fields))
            if not fields:continue
            historical='/reporting_20261012/' in rel or rel.startswith('tables/')
            model=str(row.get('model','censor_hp128' if historical else 'unspecified'))
            if suffix=='_reference':model='reference'
            elif suffix=='_spike':model='censored_spike'
            elif suffix=='_row_deletion':model='censored'
            cohort=str(row.get('cohort','N20' if '/reporting_20261012/' in rel or rel.startswith('tables/') else 'unspecified'))
            policy=str(row.get('policy','observed80' if '/reporting_20261012/' in rel or rel.startswith('tables/') else 'unspecified'))
            if 'roi' in frame:
                # Suppress the entire model/ROI/sample block if any contrast is
                # vulnerable. This also closes linear component/contrast routes.
                slot=('historical_roi' if historical else 'updated_roi',model,str(row.roi))
            else:
                slot=(str(Path(rel).parent),model)+tuple((k,str(row[k])) for k in SLOT_KEYS if k in frame)
            yield dict(path=rel,index=index,suffix=suffix,count=count,n=float(row[count]),
                       cohort=cohort,policy=policy,slot=slot,fields=fields)


def nested(a,b):
    if a['slot']!=b['slot']:return False
    def le(x,y,ranks):return x==y or (x in ranks and y in ranks and ranks[x]<=ranks[y])
    return (le(a['cohort'],b['cohort'],COHORT_RANK) and le(a['policy'],b['policy'],POLICY_RANK))


def disclosure_plan(root):
    """Global key-only plan; includes duplicate and algebraically related CSVs."""
    frames={str(p.relative_to(root)):pd.read_csv(p) for p in sorted(root.rglob('*.csv'))}
    grouped=collections.defaultdict(list)
    for rel,frame in frames.items():
        for b in summary_blocks(rel,frame):grouped[b['slot']].append(b)
    suppressed=set();pairs=0
    for blocks in grouped.values():
        nodes={}
        for b in blocks:
            key=(b['cohort'],b['policy'],b['n'])
            nodes.setdefault(key,b)
        unsafe=set()
        for ka,a in nodes.items():
            if a['n']<2:unsafe.add(ka)
            for kb,b in nodes.items():
                if nested(a,b) and b['n']-a['n']==1:
                    unsafe.add(ka);pairs+=1
        # Equal-size nested policies/cohorts are alternative displays of a set.
        changed=True
        while changed:
            changed=False
            for ka,a in nodes.items():
                if ka in unsafe:continue
                if any(a['n']==nodes[kb]['n'] and (nested(a,nodes[kb]) or nested(nodes[kb],a)) for kb in unsafe):
                    unsafe.add(ka);changed=True
        for b in blocks:
            if (b['cohort'],b['policy'],b['n']) in unsafe:
                suppressed.add((b['path'],b['index'],b['suffix']))
    return frames,suppressed,pairs


def disclosure_check(root):
    frames,plan,pairs=disclosure_plan(root)
    checked=0
    for rel,frame in frames.items():
        for b in summary_blocks(rel,frame):
            if (rel,b['index'],b['suffix']) not in plan:continue
            fields=b['fields']
            # Derived shifts can reveal a suppressed side of a comparison.
            fields += [c for c in frame if c.startswith('mean_shift')]
            if b['suffix']=='':
                fields += [c for c in frame if c.startswith(('structured_','random_','condition_mean_')) and not ('_q' in c)]
                fields += [c for c in ['component_wording','direction'] if c in frame]
            assert frame.loc[b['index'],fields].isna().all(), 'Disclosure: nonblank nested/singleton statistic '+rel+' row '+str(b['index']+2)+' block '+b['suffix']
            checked+=len(fields)
    return dict(statistic_cells_checked=checked,nested_relations=pairs,blocked_csv_blocks=len(plan))


def manifest_check(root):
    manifest=json.loads((root/'SHA256_manifest.json').read_text())
    expected={r['path'] for r in manifest}
    assert len(expected)==len(manifest),'Duplicate manifest entries'
    assert not any(p.is_symlink() for p in root.rglob('*')),'Symlinks not allowed'
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual==expected|{'SHA256_manifest.json'},'Manifest inventory mismatch'
    for row in manifest:
        path=root/row['path']
        assert path.resolve().is_relative_to(root),'Unsafe manifest path'
        assert 0<=row['bytes']<100_000_000
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    return len(manifest)


def legacy_check(root):
    """Run unchanged historical verifier, routing only its coverage read to its old table."""
    namespace=runpy.run_path(str(root/'code/reporting/verify_public_summaries_n22.py'))
    class PandasWithHistoricalCoverage:
        def read_csv(self,path,*args,**kwargs):
            if Path(path).resolve()==root/'RELEASE_COVERAGE.csv':
                path=root/'RELEASE_COVERAGE_V1_1.csv'
            return pd.read_csv(path,*args,**kwargs)
        def __getattr__(self,name):return getattr(pd,name)
    # Its main function retains the original globals mapping returned by __globals__.
    namespace['main'].__globals__['pd']=PandasWithHistoricalCoverage()
    namespace['main'](root)


def main(root):
    manifested=manifest_check(root)
    disclosure=disclosure_check(root)
    arithmetic=0;families={};blanks=0;relations=0;bounded_families=0
    def close(actual,expected,label):
        nonlocal arithmetic
        actual=np.asarray(actual,dtype=float);expected=np.asarray(expected,dtype=float)
        assert np.allclose(actual,expected,rtol=2e-7,atol=2e-9,equal_nan=True),label
        arithmetic+=int(np.isfinite(actual).sum())
    def bh(frame,group,p,q,slots,label):
        subset=frame[frame[q].notna()] if q=='q_headline_early_4' else frame
        for _,part in subset.groupby(group,dropna=False) if group else [(None,subset)]:
            assert len(part)==slots,(label,slots,len(part))
            nonlocal bounded_families
            count='n'+p[1:] if p.startswith('p_') and 'n'+p[1:] in part else 'n'
            hidden=part[p].isna() & part[count].ge(2) if count in part else pd.Series(False,index=part.index)
            if hidden.any():
                # Suppressed p values cannot be reconstructed. Verify the saved
                # q against the full admissible BH range, retaining all slots.
                lower=part[p].fillna(1).copy();upper=lower.copy()
                lower.loc[hidden]=0;upper.loc[hidden]=1
                lo=stats.false_discovery_control(lower,method='bh')
                hi=stats.false_discovery_control(upper,method='bh')
                assert (part[q].to_numpy()>=lo-2e-9).all() and (part[q].to_numpy()<=hi+2e-9).all(),label+' BH bounds '+q
                bounded_families+=1
            else:
                probabilities=part[p].fillna(1).to_numpy()
                close(part[q],stats.false_discovery_control(probabilities,method='bh'),label+' '+q)
            families[slots]=families.get(slots,0)+1
    def arithmetic_block(frame,suffix='',log=False,label=''):
        nonlocal blanks
        n='n'+suffix
        if n not in frame:return
        fields=[field+suffix for field in FIELDS if field+suffix in frame]
        fields += [c for c in frame if c.startswith('mean_shift')]
        if suffix=='':fields += [c for c in frame if c.startswith(('structured_','random_','condition_mean_')) and c.endswith(('_mean','_ci_low','_ci_high','_p'))]
        low=frame[n].lt(2)
        assert frame.loc[low,fields].isna().all().all(),label+' unsuppressed singleton'
        blanks+=int(low.sum())*len(fields)
        mean='log_difference' if log else 'mean'+suffix
        se='se'+suffix;df='df'+suffix;t='t'+suffix
        ci_low='log_ci_low' if log else 'ci_low'+suffix
        ci_high='log_ci_high' if log else 'ci_high'+suffix
        if not {mean,se,df,t,ci_low,ci_high}<=set(frame):return
        valid=frame[[mean,se,df,t,ci_low,ci_high]].notna().all(axis=1)
        part=frame[valid]
        assert part[se].gt(0).all(),label+' SE'
        close(part[df],part[n]-1,label+' df')
        if 'sd'+suffix in part:close(part[se],part['sd'+suffix]/np.sqrt(part[n]),label+' SE arithmetic')
        close(part[t],part[mean]/part[se],label+' t')
        margin=stats.t.ppf(.975,part[df])*part[se]
        close(part[ci_low],part[mean]-margin,label+' CI low')
        close(part[ci_high],part[mean]+margin,label+' CI high')
        p='p_two_sided' if log else 'p'+suffix
        if p in part:close(part[p],2*stats.t.sf(np.abs(part[t]),part[df]),label+' p from t')
    spec=json.loads((root/'AGGREGATE_CHECK_SPEC_R1.json').read_text())
    for rel in spec['added_results']:
        if not rel.endswith('.csv'):continue
        frame=pd.read_csv(root/rel)
        assert not set(frame)&{'subject','participant','sid','sub','subject_id','participant_id','omitted_subject'},rel
        for suffix in ['', '_variant','_reference','_spike','_row_deletion']:
            arithmetic_block(frame,suffix,log='log_difference' in frame,label=rel+suffix)
        if 'log_difference' in frame:
            for source,target in [('log_difference','geometric_rt_ratio'),('log_ci_low','ratio_ci_low'),('log_ci_high','ratio_ci_high'),
                                  ('bootstrap_log_ci_low','bootstrap_ratio_ci_low'),('bootstrap_log_ci_high','bootstrap_ratio_ci_high')]:
                close(frame[target],np.exp(frame[source]),rel+' transform '+source)
            assert frame.bootstrap_log_ci_low.le(frame.bootstrap_log_ci_high).all(),rel+' bootstrap order'
            for prefix,low,high in [('t_log','log_ci_low','log_ci_high'),('bootstrap_log','bootstrap_log_ci_low','bootstrap_log_ci_high'),
                                   ('t_ratio','ratio_ci_low','ratio_ci_high'),('bootstrap_ratio','bootstrap_ratio_ci_low','bootstrap_ratio_ci_high')]:
                lo='lme_log_ci_low' if 'log' in prefix else 'lme_ratio_ci_low'
                hi='lme_log_ci_high' if 'log' in prefix else 'lme_ratio_ci_high'
                close(frame[prefix+'_CI_width_over_lme'],(frame[high]-frame[low])/(frame[hi]-frame[lo]),rel+' interval-width ratio')
        if rel.endswith('descriptive_transition_category_advantage.csv'):
            for source,target in [('mean_log_ratio','geometric_rt_ratio'),('log_ci_low','ratio_ci_low'),('log_ci_high','ratio_ci_high')]:
                close(frame[target],np.exp(frame[source]),rel+' descriptive transform')
            assert frame.log_ci_low.le(frame.log_ci_high).all()
        if rel.startswith(IMAGING+'/roi/'):
            keys=['cohort','model','policy']
            if Path(rel).name in ['stage_results.csv','component_balance_results.csv','phase_component_results.csv']:
                bh(frame,keys+['stage']+(['component'] if 'component' in frame else []),'p','q',48,rel)
            if Path(rel).name=='stage_results.csv':bh(frame,keys,'p','q_joint_stages_192',192,rel)
            if Path(rel).name in ['component_balance_results.csv','headline_early_component_balance.csv']:
                bh(frame,keys+['component'],'p','q_headline_early_4',4,rel)
            if Path(rel).name=='direct_phase_results.csv':
                bh(frame,keys,'p','q_three_pairs_144',144,rel)
                bh(frame,keys+['pair'],'p','q_within_pair_48',48,rel)
            for suffix in ['_variant','_reference','_spike','_row_deletion']:
                if 'q'+suffix in frame:
                    group=[c for c in keys if c in frame]
                    bh(frame,group+['stage'],'p'+suffix,'q'+suffix,48,rel)
                    bh(frame,group,'p'+suffix,'q_joint_stages_192'+suffix,192,rel)
            if 'mean_shift' in frame:close(frame.mean_shift,frame.mean_variant-frame.mean_reference,rel+' mean shift')
            if 'mean_shift_spike_minus_row_deletion' in frame:close(frame.mean_shift_spike_minus_row_deletion,frame.mean_spike-frame.mean_row_deletion,rel+' mean shift')
        if rel==JOINT+'/joint_stage_results.csv':
            keys=['cohort','model','policy']
            bh(frame,keys+['stage'],'p','q_stage_saved',48,rel)
            bh(frame,keys+['stage'],'p','q_stage_recomputed',48,rel)
            bh(frame,keys,'p','q_joint_192',192,rel)
        if rel.endswith('_BH_family_slots.csv'):
            bh(frame,['policy'],'p_for_BH','q_bh_24',24,rel)
            assert frame.loc[~frame.tested,'p_for_BH'].eq(1).all(),rel+' untested slots'
        if Path(rel).name=='fixation_duration_frequencies.csv':
            low=frame.n_participants.lt(2)
            assert frame.loc[low,'nominal_duration_s'].isna().all()
            blanks+=int(low.sum())
        if Path(rel).name=='representative_run_timeline.csv':
            assert not {'logged_start_s','logged_end_s'}&set(frame)
            close(frame.duration_s,frame.end_s-frame.start_s,rel+' durations')
            close(frame.start_s.iloc[1:],frame.end_s.iloc[:-1],rel+' contiguous segments')
            assert frame.duration_s.sum()==400 and frame.start_s.iloc[0]==0 and frame.end_s.iloc[-1]==400
    # Duplicated display/comparison tables must agree with the full released tables.
    keys=['cohort','model','policy','stage','roi']
    stage=pd.read_csv(root/IMAGING/'roi/stage_results.csv')
    components=pd.read_csv(root/IMAGING/'roi/component_balance_results.csv')
    def agrees(subset,full,keys,columns,label):
        nonlocal relations
        assert not full.duplicated(keys).any(),label+' duplicate keys'
        joined=subset.merge(full,on=keys,suffixes=('_display','_full'),validate='one_to_one')
        assert len(joined)==len(subset),label+' missing rows'
        for column in columns:close(joined[column+'_display'],joined[column+'_full'],label+' '+column)
        relations+=len(joined)
    tidy=pd.read_csv(root/IMAGING/'roi/headline_tidy_all_specifications.csv')
    agrees(tidy,stage,keys,[c for c in FIELDS+['q','q_joint_stages_192'] if c in stage], 'headline stage')
    early=pd.read_csv(root/IMAGING/'roi/headline_early_component_balance.csv')
    agrees(early,components,keys+['component'],[c for c in FIELDS+['q','q_headline_early_4'] if c in components],'headline component')
    for component in ['structured','random','condition_mean']:
        source=components[components.component.eq(component)]
        for column in ['mean','ci_low','ci_high','p','q','q_headline_early_4']:
            selected=tidy[tidy.stage.ne('initial')][keys+[component+'_'+column]].rename(columns={component+'_'+column:column})
            agrees(selected,source,keys,[column],component+' display')
    for rel in sorted((root/IMAGING/'roi').glob('*_vs_*.csv')):
        frame=pd.read_csv(rel)
        for suffix,model in [('_variant',None),('_reference','reference'),('_spike','censored_spike'),('_row_deletion','censored')]:
            if 'mean'+suffix not in frame:continue
            selected=frame.copy()
            if model is not None:selected['model']=model
            selected=selected[keys+['mean'+suffix,'ci_low'+suffix,'ci_high'+suffix]].rename(columns={c+suffix:c for c in ['mean','ci_low','ci_high']})
            agrees(selected,stage,keys,['mean','ci_low','ci_high'],rel.name+suffix)
    joint=pd.read_csv(root/JOINT/'joint_stage_results.csv')
    for name in ['joint_only_discoveries.csv','stage_discovery_survival.csv']:
        agrees(pd.read_csv(root/JOINT/name),joint,keys,[c for c in FIELDS+['q_joint_192','q_stage_saved'] if c in joint],name)
    for folder in ['B1_participant_rt','B2_repetition_behavior']:
        path=root/BEHAVIOR/folder
        for policy in ['primary','min100']:
            frame=pd.read_csv(path/f'{policy}_contrast_comparison.csv')
            family=pd.read_csv(path/f'{policy}_BH_family_slots.csv')
            agrees(frame,family,['policy','contrast'],['q_bh_24'],folder+' behavioral family')
            joined=frame.merge(family,on=['policy','contrast'],validate='one_to_one')
            close(joined.p_two_sided,joined.p_for_BH,folder+' behavioral p')
    for number in [48,192,144,4,24]:assert families.get(number,0)>0,('Missing required BH family',number)
    historical=json.loads((root/'AGGREGATE_CHECK_SPEC.json').read_text())
    for contract in historical['contracts']:
        for family in contract.get('suppression_bounds_families',[]):
            bh(pd.read_csv(root/contract['path']),family['group'],family['p'],family['q'],family['slots'],contract['path'])
    # Code review must account for every source file, including the extracted helper.
    review=pd.read_csv(root/'CODE_REVIEW.csv').fillna('')
    assert not review.path.duplicated().any()
    for row in review.itertuples():
        path=root/row.path
        assert row.credentials_or_tokens=='none found',row.path
        if row.decision=='include verbatim':
            assert path.is_file() and sha(path)==row.source_sha256,row.path
            assert row.embedded_participant_data=='none found',row.path
        else:assert not path.exists(),row.path
    helper=root/'code/reanalysis/run_saved.py'
    tree=ast.parse(helper.read_text())
    assert [n.name for n in tree.body if isinstance(n,ast.FunctionDef)]==['direct']
    assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) for n in tree.body)
    row=review[review.path.eq('code/reanalysis/run_saved.py')].iloc[0]
    assert len(row.origin_sha256)==64 and row.origin_line_range=='10-20'
    assert spec['missing_helper_functions']==[]
    for rel,record in json.loads((root/'FIGURE_REVIEW_R1.json').read_text()).items():
        assert sha(root/rel)==record['release_sha256'],rel
    preserved=json.loads((root/'BASE_PRESERVATION.json').read_text())
    for rel,digest in preserved['preserved_files'].items():assert sha(root/rel)==digest,rel+' changed from published base'
    coverage=pd.read_csv(root/'RELEASE_COVERAGE.csv')
    status=json.loads((root/'COVERAGE_STATUS.json').read_text())
    assert not coverage.source.duplicated().any()
    assert len(coverage)==status['source_files']
    assert coverage.ledger_entries.sum()==status['recorded_values']
    assert coverage.verifiable_entries.sum()==status['verifiable_values']
    assert coverage.withheld_entries.sum()==status['withheld_values']
    assert (coverage.ledger_entries==coverage.verifiable_entries+coverage.withheld_entries).all()
    assert coverage[['ledger_entries','verifiable_entries','withheld_entries']].ge(0).all().all()
    for row in coverage.itertuples():
        if row.decision=='withhold':assert not (root/row.source).exists(),row.source
        else:assert row.release_path==row.source and (root/row.source).is_file(),row.source
    inventory=pd.read_csv(root/'FILE_DICTIONARY.csv')
    assert set(inventory.path)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert not inventory.path.duplicated().any()
    for row in inventory.itertuples():
        path=root/row.path
        if path.suffix=='.csv':assert int(row.rows)==len(pd.read_csv(path)),row.path+' dictionary row count'
    print(json.dumps(dict(status='PASS',manifest_files=manifested,arithmetic_values_checked=arithmetic,
        disclosure=disclosure,BH_families_with_suppressed_p_bounds=bounded_families,BH_families_checked={str(k):v for k,v in sorted(families.items())},suppressed_cells_checked=blanks,
        duplicate_summary_rows_checked=relations,published_base_files_preserved=len(preserved['preserved_files']),
        coverage_sources=len(coverage),coverage_values=int(coverage.ledger_entries.sum()),coverage_complete=status['complete'],
        limits='Saved aggregate arithmetic only; no participant fits, bootstrap estimation, raw-data analyses or manuscript build verified.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disclosure-only',action='store_true',help='Check all CSVs for nested-set and singleton disclosure, without other checks.')
    parser.add_argument('--legacy',action='store_true',help='Run the unchanged v1.1.0 verifier with its preserved coverage table.')
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    if args.disclosure_only:print(json.dumps(disclosure_check(root),indent=2))
    elif args.legacy:legacy_check(root)
    else:main(root)
