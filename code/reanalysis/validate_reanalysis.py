#!/usr/bin/env python3
"""Independent numerical, provenance and completeness checks for handback."""
from common import *
import argparse

def independent_stats(values,result,keys,family,qcol='q'):
 rows=[]
 for k,g in values.groupby(keys,sort=False,dropna=False):
  if not isinstance(k,tuple):k=(k,)
  a=g.value.to_numpy();a=a[np.isfinite(a)];n=len(a)
  if n>1 and a.std(ddof=1)>0:
   test=stats.ttest_1samp(a,0);mean=a.mean();se=stats.sem(a);ci=stats.t.interval(.95,n-1,loc=mean,scale=se)
   row=dict(n=n,mean=mean,se=se,p=test.pvalue,ci_low=ci[0],ci_high=ci[1])
  else:row=dict(n=n,mean=a.mean() if n else np.nan,se=np.nan,p=np.nan,ci_low=np.nan,ci_high=np.nan)
  rows.append(dict(zip(keys,k))|row)
 d=pd.DataFrame(rows);d['q_independent']=np.nan
 for _,g in d.groupby(family):
  assert len(g)==48
  # Independent explicit BH implementation, including all planned null slots.
  p=g.p.fillna(1.).to_numpy();order=np.argsort(p);q=np.empty(48);q[order]=np.minimum(1,np.minimum.accumulate((p[order]*48/np.arange(1,49))[::-1])[::-1]);d.loc[g.index,'q_independent']=q
 m=d.merge(result,on=keys,suffixes=('_check','_saved'),validate='one_to_one');assert len(m)==len(result)
 for f in ['n','mean','se','p','ci_low','ci_high']:np.testing.assert_allclose(m[f+'_check'],m[f+'_saved'],atol=1e-9,rtol=1e-8,equal_nan=True)
 np.testing.assert_allclose(m.q_independent,m[qcol],atol=1e-9,rtol=1e-8,equal_nan=True)
 return len(m)

def main():
 p=argparse.ArgumentParser();p.add_argument('--rehash-inputs',action='store_true');args=p.parse_args()
 required=['T2_within_run/anatomical_display','T2_within_run/censored_pooled_maps','T2_within_run/voxelwise_censored_float32','shared_rt_first_level','T7_imaging_shared_rt','new_first_level','T1_reference','T2_within_run','T2_within_run/voxelwise_float32','T3_sample','T4_design','T5_components','T6_robustness','T6_robustness/saved_alternatives_audit','T7_imaging','T8_trend']
 folders=[BASE/n for n in required]+[BEHAV/'T7_anticipation'];checks=[];inputs={};source_issues=[]
 for folder in folders:
  assert all((folder/f).is_file() for f in ['input_manifest.json','validation.json','REPORT.md']),folder
  v=json.loads((folder/'validation.json').read_text());assert v['passed'],folder
  for filename,expected in v['output_sha256'].items():assert digest(folder/filename)==expected,(folder,filename)
  manifest=json.loads((folder/'input_manifest.json').read_text())
  for path,rec in manifest['inputs'].items():
   if path in inputs:assert inputs[path]['sha256']==rec['sha256'],f'Inconsistent input snapshot: {path}'
   inputs[path]=rec
  for path,expected in manifest['sources'].items():
   if not Path(path).is_file() or digest(path)!=expected:source_issues.append(dict(task=str(folder),source=path,expected=expected))
  checks.append(dict(task=str(folder.relative_to(DEST)),status=v['status'],output_hashes=len(v['output_sha256'])))
 # These sources may be added after a task ran; differences are allowed only for
 # unrelated source files. Producer scripts must match or be explicitly archived.
 producer_by_task={'anatomical_display':'visualize_within_run.py','censored_pooled_maps':'build_censored_pooled.py','voxelwise_censored_float32':'run_tfce_censored_float32.py','voxelwise_float32':'run_tfce_float32.py','saved_alternatives_audit':'audit_alternatives.py','new_first_level':'run_refits.py','T1_reference':'run_saved.py','T2_within_run':'run_saved.py','T3_sample':'run_saved.py','T4_design':'run_saved.py','T5_components':'run_saved.py','T8_trend':'run_saved.py','T6_robustness':'summarize_sensitivity.py','T7_imaging':'summarize_sensitivity.py','T7_imaging_shared_rt':'update_t7_shared_rt.py','shared_rt_first_level':'run_shared_rt.py','T7_anticipation':'behavior.py'}
 for issue in source_issues:
  task=Path(issue['task']).name;producer=producer_by_task.get(task)
  if Path(issue['source']).name in ['common.py',producer]:raise AssertionError(f'Producer source changed after task: {issue}')
 if args.rehash_inputs:
  for index,(path,rec) in enumerate(inputs.items(),1):
   assert digest(path)==rec['sha256'],path
   if index%200==0 or index==len(inputs):print(f'Input hashes checked: {index}/{len(inputs)}',flush=True)
 qcs=[]
 for s in SUBJECTS:
  for r in range(1,9):
   p=BASE/'new_first_level/runs'/s/f'run-{r:02d}'/'qc.json';q=json.loads(p.read_text());assert q['script_sha256']==digest(Path(__file__).with_name('run_refits.py'));assert q['common_sha256']==digest(Path(__file__).with_name('common.py'));assert q['reference_max_voxel_difference']<=3e-5
   for path,rec in q['inputs'].items():
    assert str(path) in inputs
   qcs.append(q)
 shared_qcs=[]
 for subject in SUBJECTS:
  for run in range(1,9):
   path=BASE/'shared_rt_first_level/runs'/subject/f'run-{run:02d}'/'qc.json';q=json.loads(path.read_text())
   assert q['script_sha256']==digest(Path(__file__).with_name('run_shared_rt.py')) and q['common_sha256']==digest(Path(__file__).with_name('common.py'))
   assert q['n_rows']==200 and q['rank']==q['n_columns']
   new_design=pd.read_csv(path.parent/'design_matrix.tsv',sep='\t');old_design=pd.read_csv(FIRST/MODELS['reference']/'level1/eventrelated'/subject/f'run-{run:02d}'/'design_matrix.tsv',sep='\t')
   np.testing.assert_allclose(new_design[old_design.columns],old_design,atol=1e-10,rtol=1e-10)
   assert len(new_design.columns)==len(old_design.columns)+1
   shared_qcs.append(q)
 counts={}
 for name,vals,res,keys,family in [
  ('T1','T1_reference/participant_roi_values.csv','T1_reference/all_model_roi_results.csv',['model','policy','stage','roi'],['model','policy','stage']),
  ('T2','T2_within_run/participant_within_run_values.csv','T2_within_run/within_run_roi_results.csv',['model','contrast','roi'],['model','contrast']),
  ('T6','T6_robustness/participant_roi_values.csv','T6_robustness/specification_results.csv',['model','policy','stage','roi'],['model','policy','stage']),
  ('T7','T7_imaging/participant_roi_values.csv','T7_imaging/sensitivity_roi_results.csv',['model','policy','stage','roi'],['model','policy','stage']),
  ('T7_shared','T7_imaging_shared_rt/participant_roi_values.csv','T7_imaging_shared_rt/sensitivity_roi_results.csv',['model','policy','stage','roi'],['model','policy','stage']),
 ('T8','T8_trend/participant_slopes.csv','T8_trend/trend_roi_results.csv',['model','contrast','roi'],['model','contrast'])]:counts[name]=independent_stats(pd.read_csv(BASE/vals),pd.read_csv(BASE/res),keys,family)
 direct=pd.read_csv(BASE/'T1_reference/direct_stage_comparisons.csv')
 for _,g in direct.groupby(['model','policy']):
  assert len(g)==144;np.testing.assert_allclose(stats.false_discovery_control(g.p.fillna(1)),g.q_three_pairs_144,atol=1e-10)
 # Independently validate every accumbens full/LOO q across the 144-test family.
 import warnings
 for folder in ['T6_robustness','T7_imaging_shared_rt']:
  sv=pd.read_csv(BASE/folder/'participant_roi_values.csv')
  acc=pd.read_csv(BASE/folder/'right_accumbens_early_minus_late.csv')
  for model,part in sv.groupby('model'):
   pivot=part.pivot(index=['subject','roi'],columns='stage',values='value')
   blocks=[]
   for a,b in [('early','middle'),('early','late'),('middle','late')]:
    blocks.append((pivot[a]-pivot[b]).unstack('roi').reindex(index=SUBJECTS,columns=ROIS).to_numpy())
   matrix=np.concatenate(blocks,axis=1)
   for omitted in ['none']+SUBJECTS:
    x=matrix if omitted=='none' else matrix[np.array(SUBJECTS)!=omitted]
    with warnings.catch_warnings():
     warnings.simplefilter('ignore');p=stats.ttest_1samp(x,0,axis=0,nan_policy='omit').pvalue
    q=stats.false_discovery_control(np.where(np.isfinite(p),p,1.))
    row=acc[(acc.model==model)&(acc.omitted_subject==omitted)].iloc[0]
    index=48+ROIS.index('RightAccumbensArea')
    np.testing.assert_allclose(q[index],row.q144,atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(np.nanmean(x[:,index]),row['mean'],atol=1e-10,rtol=1e-9)
 coverage=pd.read_csv(BASE/'T1_reference/participant_roi_coverage.csv');assert len(coverage)==20*48
 values=pd.read_csv(BASE/'T1_reference/participant_roi_values.csv');join=values.merge(coverage,on=['subject','roi']);assert join[join.fraction<.8-1e-12].query('policy=="observed80"').value.isna().all()
 assert all(values.groupby(['model','policy','stage']).size()==960)
 # Independently reconstruct both model DiDs and all new within-run weights.
 condition=pd.read_csv(BASE/'T2_within_run/participant_run_condition_values.csv');cp=condition.pivot(index=['model','subject','roi'],columns=['run','condition'],values='value');init=cp[1,'Str1']+cp[2,'Str1']-cp[1,'Ran']-cp[2,'Ran']
 for i,stage in enumerate(STAGES):
  y=init if i==0 else cp[2*i+1,'Str2']+cp[2*i+2,'Str2']-cp[2*i+1,'Ran']-cp[2*i+2,'Ran']-init
  expected=values[(values.policy=='observed80')&(values.stage==stage)].set_index(['model','subject','roi']).value.reindex(cp.index);np.testing.assert_allclose(y,expected,atol=2e-7,rtol=1e-6,equal_nan=True)
 wi=pd.read_csv(BASE/'T2_within_run/participant_within_run_values.csv')
 for name,runs in [('run3',[3]),('run4',[4]),('run5',[5]),('run6',[6]),('early_r34',[3,4]),('middle_r56',[5,6]),('pooled_r36',[3,4,5,6])]:
  y=sum(cp[r,'Str2']-cp[r,'Str1'] for r in runs)/len(runs);exp=wi[wi.contrast==name].set_index(['model','subject','roi']).value.reindex(cp.index);np.testing.assert_allclose(y,exp,atol=1e-10,rtol=1e-9,equal_nan=True)
 for folder in ['voxelwise_float32','voxelwise_censored_float32']:
  voxel=BASE/'T2_within_run'/folder;v=json.loads((voxel/'validation.json').read_text());assert v['n_perm']==10000 and v['n_subjects']==20
  stat=np.asarray(nib.load(voxel/'t.nii.gz').dataobj);logp=np.asarray(nib.load(voxel/'logp_max_tfce.nii.gz').dataobj);signed=np.asarray(nib.load(voxel/'signed_tfce_fwe05_t.nii.gz').dataobj)
  for path in voxel.glob('*.nii.gz'):
   im=nib.load(path);assert im.get_data_dtype()==np.dtype('float32') and im.dataobj.slope==1. and im.dataobj.inter==0.
   assert np.isfinite(np.asarray(im.dataobj)).all()
   common_mask=nib.load(ROOT/'results/glm_volume/whole_brain/common_intersection_mask.nii.gz');assert im.shape==common_mask.shape;np.testing.assert_allclose(im.affine,common_mask.affine,atol=1e-6)
  raw=np.load(voxel/'raw_permutation_arrays.npz');mask=np.asarray(nib.load(ROOT/'results/glm_volume/whole_brain/common_intersection_mask.nii.gz').dataobj)>0
  h0=raw['h0_max_tfce'];assert h0.shape==(1,10000) and np.isfinite(h0).all()
  rank=np.searchsorted(np.sort(h0[0]),np.abs(raw['tfce'][0]),side='left');expected=-np.log10((10001-rank)/10001)
  np.testing.assert_allclose(raw['logp_max_tfce'][0],expected,atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(logp[mask],expected,atol=2e-7,rtol=1e-6)
  np.testing.assert_allclose(stat[mask],raw['t'][0],atol=1e-6,rtol=1e-6)
  # Independently reconstruct the intercept-only group t from subject maps.
  pooled_folder='pooled_subject_maps' if folder=='voxelwise_float32' else 'censored_pooled_maps'
  subject_maps=[np.asarray(nib.load(BASE/'T2_within_run'/pooled_folder/f'{subject}_Str2_minus_Str1_r36_mean.nii.gz').dataobj)[mask] for subject in SUBJECTS]
  independent_t=stats.ttest_1samp(np.asarray(subject_maps,dtype=np.float64),0.,axis=0).statistic
  np.testing.assert_allclose(stat[mask],independent_t,atol=2e-6,rtol=2e-6)

  np.testing.assert_allclose(signed,np.where(logp>=-np.log10(.05),stat,0),atol=1e-6)
  assert int((signed>0).sum())==v['positive_voxels'];assert int((signed<0).sum())==v['negative_voxels']
 # Check raw-vs-cleaned anticipation distinction and explicit GLMM failure state.
 b=BEHAV/'T7_anticipation';raw=pd.read_csv(b/'raw_trials_22.csv');n20=raw[raw.cohort20];ant=n20.p_response_rt.gt(0)&n20.p_response_rt.lt(.1)&~n20.omitted
 assert len(raw)==22*2560 and len(n20)==51200 and ant.sum()==1895 and (ant&n20.correct).sum()==1837
 counts_behavior=json.loads((b/'counts.json').read_text());assert counts_behavior['n_correct_anticipatory_after_saved_cleaning']==1833
 glmm=json.loads((b/'glmm/status.json').read_text());assert glmm['status']=='model_not_identifiable' and not glmm['valid_for_inference']
 missing=pd.read_csv(BASE/'T3_sample/input_availability.csv');assert (~missing.exists).sum()==16 and (~missing[missing.kind=='events'].exists).all()
 ordering=pd.read_csv(BASE/'T4_design/event_ordering.csv');assert len(ordering)==20*8*8 and ordering.n_trials.eq(40).all()
 structured=ordering[~ordering.block.str.contains('Ran')]
 for run,g in structured.groupby('run'):
  expected=0 if run<=2 else 5 if run>=7 else run-2
  assert g.sequence_windows.str.count('Str2').eq(expected).all()
 diag=pd.read_csv(BASE/'T4_design/design_diagnostics.csv');assert len(diag)==320 and (diag['rank']==diag.n_columns).all()
 assert diag.query('model=="reference"').filter(like='ar1_fullmask_mean_unit_variance').notna().sum().sum()>0
 result={'passed':True,'all_available_products_validated':True,'all_planned_analyses_complete':False,'limitations':['T3 N22 imaging blocked by missing event and censor inputs','T7a full GLMM not identifiable; descriptive results only','Original response assignment/randomization and exact learner-test options unavailable','T9 deferred to author review; T10 optional not run'],'task_checks':checks,'independent_numerical_rows':counts,'n_unique_input_hashes':len(inputs),'input_hashes_rechecked':args.rehash_inputs,'new_run_jobs':len(qcs),'additional_shared_rt_run_jobs':len(shared_qcs),'max_reference_voxel_difference':max(q['reference_max_voxel_difference'] for q in qcs),'source_changes_unrelated_to_producer':source_issues,'superseded_export':'T2_within_run/voxelwise: integer-header raw maps; preserved for audit','summary_sha256':digest(TOP/'SUMMARY.md')}
 writej(TOP/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['task_checks','source_changes_unrelated_to_producer']},indent=2))
if __name__=='__main__':main()
