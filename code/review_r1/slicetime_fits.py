#!/usr/bin/env python3
"""A7 frame-time-aligned reference fits with a zero-shift control gate."""
from slicetime_common import *
import argparse
import concurrent.futures
import warnings
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.first_level.first_level import mean_scaling


def fit(job):
    subject, run, variants = job
    assert set(variants)<= {'reference','slicetime_aligned'}
    assert json.loads((OUT/'metadata/validation.json').read_text())['passed']
    root = OUT/'first_level/runs'/subject/f'run-{run:02d}'
    root.mkdir(parents=True, exist_ok=True)
    todo = []
    for variant in variants:
        marker = root/variant/'qc.json'
        if marker.exists():
            q = json.loads(marker.read_text())
            if q['producer']!=digest(__file__) or q['common']!=digest(Path(__file__).with_name('slicetime_common.py')):
                todo.append(variant)
                continue
            for name, sha in q['outputs'].items():
                assert digest(root/variant/name)==sha
        else:
            todo.append(variant)
    if not todo:
        return dict(subject=subject,run=run,cached=True,elapsed_seconds=0.)
    started=time.monotonic()
    bold, mask, conf, _ = paths(subject,run)
    ep=event_path(subject,run)
    rp=ROOT/'data/behavioral'/f'{subject[-3:]}_run_{run}_srtt_prob_new.csv'
    sidecar=bold.with_name(bold.name.replace('.nii.gz','.json'))
    timing=json.loads(sidecar.read_text());tr=float(timing['RepetitionTime']);start=float(timing['StartTime'])
    assert tr==2. and 0<=start<tr
    inputs=[bold,sidecar,mask,conf,conf.with_suffix('.json'),ep,rp,mask_path(subject)]
    inputs += [paths(subject,r)[1] for r in range(1,9)]
    inputs += [ROOT/'resources/roi_masks'/f'{roi}.nii.gz' for roi in ROIS]
    hashes={str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in inputs}
    e=pd.read_csv(ep,sep='\t');e.trial_type=e.trial_type.str.extract('(Ran|Str[12])$')[0]
    assert not e.trial_type.isna().any()
    raw=pd.read_csv(rp);repeat=repeats(raw)
    confounds=pd.read_csv(conf,sep='\t');metadata=json.loads(conf.with_suffix('.json').read_text())
    components=sorted(k for k,v in metadata.items() if k.startswith('a_comp_cor_') and v.get('Mask')=='combined' and v.get('Retained') is True and k in confounds)[:8]
    assert components  # Identical components[:8] selection to existing producer; one run has five.
    censor=np.loadtxt(mask_path(subject)).reshape(8,200)[run-1]
    assert set(np.unique(censor)) <= {0.,1.}
    ref=nib.load(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz')
    support=np.logical_and.reduce([image(paths(subject,r)[1],ref,True)>0 for r in range(1,9)])
    parcel={roi:image(ROOT/'resources/roi_masks'/f'{roi}.nii.gz',ref,True)>0 for roi in ROIS}
    native=nib.load(mask);native_mask=np.asarray(native.dataobj)>0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        original=NiftiMasker(mask_img=str(mask),smoothing_fwhm=4.,standardize=False,detrend=False,t_r=2.).fit_transform(str(bold))
        assert original.shape[0]==200 and np.isfinite(original).all()
        for variant in todo:
            before=time.monotonic();folder=root/variant;folder.mkdir(exist_ok=True)
            cols=CONFOUNDS
            for col in cols:
                missing=np.flatnonzero(confounds[col].isna())
                assert len(missing)==0 or (col.endswith('_derivative1') and np.array_equal(missing,[0]))
            c=confounds[cols].fillna(0).astype(float).copy()
            ev=e.copy()
            design=make_first_level_design_matrix(np.arange(200)*tr+(start if variant=='slicetime_aligned' else 0.),ev,hrf_model='glover',drift_model='cosine',high_pass=1/128,
                                                  add_regs=c.to_numpy(),add_reg_names=list(c))
            x=design.to_numpy();rank=int(np.linalg.matrix_rank(x));df=200-rank
            if rank!=x.shape[1] or df<20:
                writej(folder/'failure.json',dict(model=variant,rank=rank,n_columns=x.shape[1],residual_df=df,
                                                 handling='assertion failure, identical to existing censored variant'))
                raise AssertionError((subject,run,variant,rank,x.shape[1],df))
            design.to_csv(folder/'design_matrix.tsv',sep='\t',index=True,index_label='frame_time')
            y,_=mean_scaling(original.copy(),axis=0)
            labels,fits=run_glm(y,x,noise_model='ar1',bins=100,n_jobs=1)
            values=[];qc=[]
            for condition in conditions(run):
                beta=np.zeros(y.shape[1]);index=design.columns.get_loc(condition)
                for label,result in fits.items():
                    beta[labels==label]=result.theta[index]
                array=np.zeros(native.shape,dtype=np.float32);array[native_mask]=beta
                im=nib.Nifti1Image(array,native.affine)
                if native.shape!=ref.shape or not np.allclose(native.affine,ref.affine,atol=1e-5):
                    im=resample_to_img(im,ref,interpolation='continuous',force_resample=True,copy_header=True)
                array=np.asarray(im.dataobj,dtype=np.float32)
                if variant=='slicetime_aligned':
                    save32(array,ref,folder/f'{condition}_effect_size.nii.gz')
                for roi,p in parcel.items():
                    observed=p&support
                    v=float(array[observed].mean(dtype=np.float64)) if observed.any() and observed.sum()/p.sum()+1e-12>=.8 else np.nan
                    values.append(dict(model=variant,subject=subject,run=run,condition=condition,roi=roi,value=v))
            task=conditions(run);nuisance=design[[v for v in design if v not in task]].to_numpy()
            adjusted=design[task].to_numpy()-nuisance@np.linalg.lstsq(nuisance,design[task],rcond=None)[0]
            for idx,condition in enumerate(task):
                target=adjusted[:,idx];other=np.delete(adjusted,idx,axis=1)
                resid=target-other@np.linalg.lstsq(other,target,rcond=None)[0]
                qc.append(dict(model=variant,subject=subject,run=run,condition=condition,n_rows=200,n_columns=x.shape[1],rank=rank,
                               residual_df=df,nuisance_adjusted_vif=float((target@target)/(resid@resid)),n_censored=int((censor==0).sum()),
                               n_repeats=int(repeat.sum()),n_spikes=int((censor==0).sum()) if variant=='censored_spike' else 0))
            csv(values,folder/'condition_values.csv');csv(qc,folder/'design_qc.csv')
            q=dict(subject=subject,run=run,model=variant,producer=digest(__file__),common=digest(Path(__file__).with_name('slicetime_common.py')),
                   inputs=hashes,confound_columns=cols,StartTime=start,RepetitionTime=tr,frame_shift=start if variant=='slicetime_aligned' else 0.,acompcor_columns=components,
                   elapsed_seconds=time.monotonic()-before,shared_io_seconds=before-started,
                   warnings=[str(w.message) for w in caught],
                   outputs={p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='qc.json'})
            writej(folder/'qc.json',q)
            del fits,y
    return dict(subject=subject,run=run,models=todo,elapsed_seconds=time.monotonic()-started)


def launch(jobs,workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fit,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            q=f.result();print(f'{i}/{len(jobs)} '+json.dumps(q),flush=True)


def stage_values(data):
    p=data.pivot(index=['model','subject','roi'],columns=['run','condition'],values='value')
    baseline=p[1,'Str1']+p[2,'Str1']-p[1,'Ran']-p[2,'Ran'];rows=[]
    for idx,stage in enumerate(STAGES):
        val=baseline if idx==0 else p[2*idx+1,'Str2']+p[2*idx+2,'Str2']-p[2*idx+1,'Ran']-p[2*idx+2,'Ran']-baseline
        d=val.rename('value').reset_index();d['stage']=stage;rows.append(d)
    return pd.concat(rows,ignore_index=True)


def controls(workers):
    t=ImagingTask('controls')
    launch([(s,r,['reference']) for s,r in [('sub-001',1),('sub-020',6)]],workers)
    reference=pd.read_csv(t.record(EXISTING/'roi_reporting/participant_run_condition_values.csv'))
    reference=reference.query("model=='reference' and policy=='observed80'")
    rows=[]
    for s,r in [('sub-001',1),('sub-020',6)]:
        p=OUT/'first_level/runs'/s/f'run-{r:02d}'/'reference/condition_values.csv';fresh=pd.read_csv(t.record(p))
        z=fresh.merge(reference,on=['model','subject','run','condition','roi'],suffixes=('_new','_old'),validate='one_to_one')
        assert len(z)==len(fresh)
        np.testing.assert_allclose(z.value_new,z.value_old,atol=1e-6,rtol=1e-6,equal_nan=True)
        rows.append(dict(check='reference_condition_values',subject=s,run=r,n_values=len(z),max_difference=float(abs(z.value_new-z.value_old).max())))
    csv(rows,t.path/'reproduction_checks.csv')
    for subject,run in [('sub-001',1),('sub-020',6)]:
        q=json.loads(t.record(OUT/'first_level/runs'/subject/f'run-{run:02d}'/'reference/qc.json').read_text());t.inputs.update(q['inputs'])
    t.finish('# Zero-shift controls before aligned fits\n\nBoth specified reference runs reproduce saved observed80 condition values, including NaN slots, within 1e-6. Reference frame shift is zero. No aligned run was fit before these checks passed.',dict(passed=True,checks=rows))
    launch([(s,r,['slicetime_aligned']) for s,r in [('sub-001',1),('sub-020',6)]],workers)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controls','full']);parser.add_argument('--workers',type=int,default=4);a=parser.parse_args()
    assert nilearn.__version__=='0.13.1' and 1<=a.workers<=10
    if a.mode=='controls':controls(a.workers);return
    assert json.loads((OUT/'controls/validation.json').read_text())['passed']
    t=ImagingTask('first_level');launch([(s,r,NEW_MODELS) for s in N22 for r in range(1,9)],a.workers)
    for name in ['condition_values','design_qc']:
        csv(pd.concat([pd.read_csv(OUT/'first_level/runs'/s/f'run-{r:02d}'/m/f'{name}.csv') for s in N22 for r in range(1,9) for m in NEW_MODELS],ignore_index=True),t.path/f'{name}.csv')
    qcs=[]
    for s in N22:
        for r in range(1,9):
            for m in NEW_MODELS:
                q=json.loads(t.record(t.path/'runs'/s/f'run-{r:02d}'/m/'qc.json').read_text());t.inputs.update(q['inputs']);qcs.append(q)
    qc=pd.read_csv(t.path/'design_qc.csv');assert qc['rank'].eq(qc.n_columns).all() and qc.residual_df.ge(20).all()
    t.finish('# A7 aligned first-level fits\n\n176 fits (22 participants x eight runs). Frame times arange(200)*RepetitionTime+StartTime, from each BOLD JSON. Reference nuisance model, Glover/cosine 1/128 s, AR(1), 100 bins, 4-mm smoothing, all 200 rows and percent mean scaling unchanged. Float32 condition maps are resampled continuously to ROI reference before observed80 extraction using the participant eight-run intersection. All aligned maps retained for voxelwise inference. Saved designs include frame times; QC reports rank, df and nuisance-adjusted VIF.',dict(passed=True,n_fits=len(qcs),n_subjects=22,n_runs=176,min_residual_df=int(qc.residual_df.min()),all_full_rank=True))



if __name__=='__main__':main()
