#!/usr/bin/env python3
"""New full-voxel AR(1) fits for T6 SPM+derivative and T7 nuisance sensitivities.

Loads/smooths each run once. Verifies the reference against saved voxel betas.
Only ROI coefficients, design/QC and support are retained for new models.
"""
from common import *
import argparse,time,warnings,concurrent.futures
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix,run_glm
from nilearn.glm.first_level.first_level import mean_scaling
VARIANTS=['reference_check','spm_derivative','errors_omissions','rt_modulator']
def job(s,run,out):
 started=time.monotonic();folder=Path(out)/'runs'/s/f'run-{run:02d}';marker=folder/'qc.json'
 script_hash=digest(__file__);common_hash=digest(Path(__file__).with_name('common.py'))
 if marker.exists():
  q=json.loads(marker.read_text());assert q['script_sha256']==script_hash and q['common_sha256']==common_hash,'changed fit source: use new version';return q
 folder.mkdir(parents=True,exist_ok=True)
 bold,mask,conf,event=paths(s,run);rawp=ROOT/'data/behavioral'/f'{int(s[-3:]):03d}_run_{run}_srtt_prob_new.csv'
 inputs={str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in [bold,mask,conf,event,rawp]}
 f=pd.read_csv(conf,sep='\t');ev=pd.read_csv(event,sep='\t');ev['trial_type']=ev.trial_type.str.extract('(Ran|Str[12])$')[0];assert ev.trial_type.notna().all()
 raw=pd.read_csv(rawp);raw['omitted']=~np.isfinite(raw.p_response_rt)|raw.p_response_keys.astype(str).isin(['None','nan','']);raw['wrong']=(raw['correct.resp']!=1)&~raw.omitted
 nuisance=f[CONFOUNDS].fillna(0).astype(float)
 with warnings.catch_warnings(record=True) as warn:
  warnings.simplefilter('always');masker=NiftiMasker(mask_img=str(mask),smoothing_fwhm=4.,standardize=False,detrend=False,t_r=2.);y=masker.fit_transform(str(bold))
 y,means=mean_scaling(y,axis=0);assert np.isfinite(y).all()
 ref=nib.load(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz')
 support=np.ones(ref.shape,bool)
 for r in range(1,9):support&=image(paths(s,r)[1],ref,True)>0
 # Native run masks are used for fitting; ROI means are taken only after map
 # interpolation to the historical 3-mm reference grid, preserving support.
 native=nib.load(mask);native_mask=np.asarray(native.dataobj)>0
 roi_masks={r:image(ROOT/'resources/roi_masks'/f'{r}.nii.gz',ref,True)>0 for r in ROIS}
 def arr_on_grid(v):
  a=np.zeros(native.shape,dtype=np.float32);a[native_mask]=v
  im=nib.Nifti1Image(a,native.affine)
  if native.shape!=ref.shape or not np.allclose(native.affine,ref.affine,atol=1e-5):im=resample_to_img(im,ref,interpolation='continuous',force_resample=True,copy_header=True)
  return np.asarray(im.dataobj,float)
 rows=[];diagnostics=[];max_diff=0.;arrows=[]
 for variant in VARIANTS:
  e=ev.copy();hrf='spm + derivative' if variant=='spm_derivative' else 'glover'
  if variant=='errors_omissions':
   extra=[]
   for flag,name in [('wrong','WrongPress'),('omitted','Omission')]:
    d=raw[raw[flag]]
    if len(d):extra.append(pd.DataFrame({'onset':d.stim_onset+(d.p_response_rt if flag=='wrong' else 1.),'duration':0.,'trial_type':name}))
   if extra:e=pd.concat([e,*extra],ignore_index=True)
  if variant=='rt_modulator':
   e['modulation']=1.;extra=[]
   for c,g in raw[(~raw.omitted)&raw.p_response_rt.gt(0)&raw.p_response_rt.lt(1)].groupby('rule_group'):
    extra.append(pd.DataFrame({'onset':g.stim_onset,'duration':0.,'trial_type':'RT_dev_'+c,'modulation':g.p_response_rt-g.p_response_rt.mean()}))
   e=pd.concat([e,*extra],ignore_index=True)
  design=make_first_level_design_matrix(np.arange(200)*2.,e,hrf_model=hrf,drift_model='cosine',high_pass=1/128,add_regs=nuisance.to_numpy(),add_reg_names=list(nuisance))
  x=design.to_numpy();rank=np.linalg.matrix_rank(x);assert rank==x.shape[1],(s,run,variant,rank,x.shape)
  design.to_csv(folder/f'{variant}_design.tsv',sep='\t',index=False)
  saved=pd.read_csv(FIRST/MODELS['reference']/'level1/eventrelated'/s/f'run-{run:02d}'/'design_matrix.tsv',sep='\t')
  if variant=='reference_check':np.testing.assert_allclose(x,saved.to_numpy(),atol=1e-10,rtol=1e-10)
  labels,fits=run_glm(y,x,noise_model='ar1',bins=100,n_jobs=1)
  if variant=='reference_check':
   # Keep the whole-voxel AR histogram for genuinely fitted covariance diagnostics.
   for k,fit in fits.items():arrows.append(dict(subject=s,run=run,rho=float(k),n_voxels=int((labels==k).sum())))
  for c in conditions(run):
   beta=np.zeros(y.shape[1]);i=design.columns.get_loc(c)
   for k,fit in fits.items():beta[labels==k]=fit.theta[i]
   a=arr_on_grid(beta)
   if variant=='reference_check':
    p=FIRST/MODELS['reference']/'level1/eventrelated'/s/f'run-{run:02d}'/f'{c}_effect_size.nii.gz';old=image(p,ref)
    diff=float(np.max(abs(a-old)));max_diff=max(max_diff,diff)
    np.testing.assert_allclose(a,old,atol=3e-5,rtol=3e-5)
   for roi,m in roi_masks.items():
    v=m&support
    val=a[v].mean() if v.sum()/m.sum()+1e-12>=.8 and v.any() else np.nan
    rows.append(dict(model=variant,subject=s,run=run,condition=c,roi=roi,value=val))
  # Nuisance-adjusted task VIF and sparsity before reporting any group effects.
  tasks=conditions(run);z=design[[c for c in design if c not in tasks]].to_numpy();adj=design[tasks].to_numpy()-z@np.linalg.lstsq(z,design[tasks],rcond=None)[0]
  for j,c in enumerate(tasks):
   a=adj[:,j];other=np.delete(adj,j,axis=1);res=a-other@np.linalg.lstsq(other,a,rcond=None)[0]
   diagnostics.append(dict(model=variant,subject=s,run=run,condition=c,n_rows=len(x),n_columns=x.shape[1],rank=rank,n_wrong=int(raw.wrong.sum()),n_omissions=int(raw.omitted.sum()),nuisance_adjusted_task_vif=(a@a)/(res@res),normalized_condition_number=np.linalg.cond(x/np.linalg.norm(x,axis=0))))
 csv(rows,folder/'condition_values.csv');csv(diagnostics,folder/'design_qc.csv');csv(arrows,folder/'reference_ar1_histogram.csv')
 qc=dict(subject=s,run=run,script_sha256=script_hash,common_sha256=common_hash,inputs=inputs,variants=VARIANTS,reference_max_voxel_difference=max_diff,warnings=[str(w.message) for w in warn],elapsed_seconds=time.monotonic()-started,n_voxels=y.shape[1])
 writej(marker,qc);return qc

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--subjects',nargs='+',default=SUBJECTS);p.add_argument('--runs',nargs='+',type=int,default=list(range(1,9)));a=p.parse_args();assert nilearn.__version__=='0.13.1'
 out=BASE/'new_first_level';out.mkdir(parents=True,exist_ok=True)
 writej(out/'execution.json',{'argv':sys.argv,'models':VARIANTS,'subjects':a.subjects,'runs':a.runs,'workers':a.workers,'script_sha256':digest(__file__)})
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs=[pool.submit(job,s,r,out) for s in a.subjects for r in a.runs]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):
   q=f.result();print(json.dumps({'completed':i,'total':len(fs),'subject':q['subject'],'run':q['run'],'seconds':round(q['elapsed_seconds'],2),'max_reference_difference':q['reference_max_voxel_difference']}),flush=True)
 if set(a.subjects)==set(SUBJECTS) and sorted(a.runs)==list(range(1,9)):
  for name in ['condition_values','design_qc','reference_ar1_histogram']:
   csv(pd.concat([pd.read_csv(out/'runs'/s/f'run-{r:02d}'/f'{name}.csv') for s in SUBJECTS for r in range(1,9)]),out/f'{name}.csv')
 print('Completed',flush=True)
if __name__=='__main__':main()
