#!/usr/bin/env python3
"""T7c extension: shared run-centered RT PM retains between-condition RT differences.

Added after inspecting the centering scope, before looking at this model's results.
Original within-condition PM remains available. No original outputs are changed.
"""
from common import *
import argparse,concurrent.futures,time,warnings
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix,run_glm
from nilearn.glm.first_level.first_level import mean_scaling

def fit(job):
 subject,run=job;out=BASE/'shared_rt_first_level/runs'/subject/f'run-{run:02d}';marker=out/'qc.json';sha=digest(__file__)
 if marker.exists():
  q=json.loads(marker.read_text());assert q['script_sha256']==sha and q['common_sha256']==digest(Path(__file__).with_name('common.py'));return q
 started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
 bold,mask,conf,event=paths(subject,run);rawp=ROOT/'data/behavioral'/f'{int(subject[-3:]):03d}_run_{run}_srtt_prob_new.csv'
 inputs={str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in [bold,mask,conf,event,rawp]}
 raw=pd.read_csv(rawp);responded=np.isfinite(raw.p_response_rt)&raw.p_response_rt.gt(0)&raw.p_response_rt.lt(1)&~raw.p_response_keys.astype(str).isin(['None','nan',''])
 rt=raw[responded];ev=pd.read_csv(event,sep='\t');ev['trial_type']=ev.trial_type.str.extract('(Ran|Str[12])$')[0];ev['modulation']=1.
 pm=pd.DataFrame({'onset':rt.stim_onset,'duration':0.,'trial_type':'RT_shared_run_centered','modulation':rt.p_response_rt-rt.p_response_rt.mean()})
 ev=pd.concat([ev,pm],ignore_index=True);c=pd.read_csv(conf,sep='\t')[CONFOUNDS].fillna(0).astype(float)
 with warnings.catch_warnings(record=True) as warn:
  warnings.simplefilter('always');masker=NiftiMasker(mask_img=str(mask),smoothing_fwhm=4.,standardize=False,detrend=False,t_r=2.)
  y=masker.fit_transform(str(bold));y,_=mean_scaling(y,axis=0)
  design=make_first_level_design_matrix(np.arange(200)*2.,ev,hrf_model='glover',drift_model='cosine',high_pass=1/128,add_regs=c.to_numpy(),add_reg_names=list(c))
 x=design.to_numpy();rank=int(np.linalg.matrix_rank(x));assert rank==x.shape[1] and np.isfinite(y).all()
 design.to_csv(out/'design_matrix.tsv',sep='\t',index=False)
 labels,fits=run_glm(y,x,noise_model='ar1',bins=100,n_jobs=1)
 ref=nib.load(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz');support=np.ones(ref.shape,bool)
 for r in range(1,9):support&=image(paths(subject,r)[1],ref,True)>0
 native=nib.load(mask);nm=np.asarray(native.dataobj)>0
 rows=[];diagnostics=[]
 task=conditions(run);nuisance=design[[name for name in design if name not in task]].to_numpy();adj=design[task].to_numpy()-nuisance@np.linalg.lstsq(nuisance,design[task],rcond=None)[0]
 for i,condition in enumerate(task):
  beta=np.zeros(y.shape[1]);index=design.columns.get_loc(condition)
  for label,result in fits.items():beta[labels==label]=result.theta[index]
  a=np.zeros(native.shape,dtype=np.float32);a[nm]=beta;im=nib.Nifti1Image(a,native.affine)
  if native.shape!=ref.shape or not np.allclose(native.affine,ref.affine,atol=1e-5):im=resample_to_img(im,ref,interpolation='continuous',force_resample=True,copy_header=True)
  a=np.asarray(im.dataobj,float)
  for roi in ROIS:
   m=image(ROOT/'resources/roi_masks'/f'{roi}.nii.gz',ref,True)>0;v=m&support
   value=float(a[v].mean()) if v.sum()/m.sum()+1e-12>=.8 and v.any() else np.nan
   rows.append(dict(model='rt_shared_modulator',subject=subject,run=run,condition=condition,roi=roi,value=value))
  a=adj[:,i];other=np.delete(adj,i,axis=1);resid=a-other@np.linalg.lstsq(other,a,rcond=None)[0]
  diagnostics.append(dict(subject=subject,run=run,condition=condition,n_rows=len(x),n_columns=x.shape[1],rank=rank,nuisance_adjusted_task_vif=(a@a)/(resid@resid),n_valid_rt=int(responded.sum()),rt_center_seconds=float(rt.p_response_rt.mean())))
 csv(rows,out/'condition_values.csv');csv(diagnostics,out/'design_qc.csv')
 q=dict(subject=subject,run=run,script_sha256=sha,common_sha256=digest(Path(__file__).with_name('common.py')),inputs=inputs,n_rows=len(x),rank=rank,n_columns=x.shape[1],warnings=[str(w.message) for w in warn],seconds=time.monotonic()-started)
 writej(marker,q);return q

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);args=p.parse_args();assert nilearn.__version__=='0.13.1'
 t=Task('shared_rt_first_level');qcs=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  for i,q in enumerate(pool.map(fit,[(s,r) for s in SUBJECTS for r in range(1,9)]),1):
   qcs.append(q);print(f'Completed {i}/160 {q["subject"]} run{q["run"]}',flush=True)
 for name in ['condition_values','design_qc']:
  csv(pd.concat([pd.read_csv(t.path/'runs'/s/f'run-{r:02d}'/f'{name}.csv') for s in SUBJECTS for r in range(1,9)]),t.path/f'{name}.csv')
 for q in qcs:
  t.inputs.update(q['inputs']);folder=t.path/'runs'/q['subject']/f'run-{q["run"]:02d}'
  for p in folder.iterdir():t.record(p)
 t.finish('# T7c shared RT-PM extension\n\n160 full-voxel Glover/AR(1) fits with the reference mask, 4-mm smoothing, basic eight nuisance columns and 128-s high-pass. A single impulse PM spans every valid responded trial (correct and incorrect; 0<RT<1), with RT in seconds centered over the entire run. This retains between-condition RT mean differences in the regressor, unlike the separate within-condition-centered PM. Original 8-/40-s condition boxcars remain.\n\nThis specification was added after inspecting the centering scope and before inspecting its results. Both PM specifications are retained. It models an RT-associated response, not measured cue duration; actual cue offsets/response assignment remain unavailable, and RT is affected by learning. See T7_imaging for comparisons.',dict(passed=True,n_run_fits=160,n_subjects=20,all_full_rank=all(q['rank']==q['n_columns'] for q in qcs)))
if __name__=='__main__':main()
