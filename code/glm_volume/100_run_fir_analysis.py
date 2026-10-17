#!/usr/bin/env python3
"""Fit duration-aware 0–32 s FIR kernels; preserves primary run contrasts."""
from __future__ import annotations
import argparse
import concurrent.futures
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

helper=module('fir_helper','98_run_model_alternatives.py')
roi=module('fir_roi','96_summarize_motion_filter_sensitivity.py')
primary=helper.primary
OUT=helper.FIRST_LEVEL_DIR/'FLA_nilearn_fir_20260921'
REPORT=helper.MANUSCRIPT/'data/fir_20260921'
DELAYS=list(range(16))

def fit_run(job):
    subject,run=job;start=time.monotonic()
    target=OUT/'level1/eventrelated'/subject/f'run-{run:02d}'
    marker=target/'model_qc.json'
    if marker.exists():
        qc=json.loads(marker.read_text())
        assert helper.fit_source_matches(qc['script_sha256'], Path(__file__).name)
        assert all((target/p).exists() for p in qc['output_files'])
        return dict(subject=subject,run=run,skipped=True)
    bold,mask,cpath,epath=primary.input_paths(subject,run)
    events=primary.load_events(epath,primary.CONDITIONS[run])
    confounds=primary.load_confounds(cpath,nib.load(bold).shape[-1])
    model=FirstLevelModel(t_r=2.,hrf_model='fir',fir_delays=DELAYS,
        drift_model='cosine',high_pass=1/128,smoothing_fwhm=4.,signal_scaling=0,
        noise_model='ar1',mask_img=str(mask),standardize=False,minimize_memory=True,n_jobs=1)
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(str(bold),events=events,confounds=confounds)
        design=model.design_matrices_[0];x=design.to_numpy()
        rank=int(np.linalg.matrix_rank(x))
        assert np.isfinite(x).all() and rank==x.shape[1]
        assert x.shape[0]-rank>=10
        columns=list(design)
        task_columns=[f'{cond}_delay_{delay}' for cond in primary.CONDITIONS[run] for delay in DELAYS]
        task_indices=[columns.index(name) for name in task_columns]
        nuisance_indices=[i for i,c in enumerate(columns) if c not in task_columns]
        task=x[:,task_indices];nuisance=x[:,nuisance_indices]
        residualized=task-nuisance@np.linalg.lstsq(nuisance,task,rcond=None)[0]
        task_rank=int(np.linalg.matrix_rank(residualized));assert task_rank==len(task_indices)
        # Recover exactly the fitted coefficients without thousands of redundant contrasts.
        beta=np.empty((len(task_indices),len(model.labels_[0])),dtype=np.float64)
        for label,res in model.results_[0].items():
            beta[:,model.labels_[0]==label]=res.theta[task_indices,:]
        images=model.masker_.inverse_transform(beta)
        # Independently verify one extracted coefficient for each condition.
        for cond in primary.CONDITIONS[run]:
            idx=task_columns.index(f'{cond}_delay_0')
            check=model.compute_contrast(f'{cond}_delay_0',output_type='effect_size')
            assert np.allclose(np.asarray(images.dataobj)[...,idx],np.asarray(check.dataobj),rtol=1e-7,atol=1e-7)
        reference=primary.canonical_reference()
        if primary.grid_signature(images)!=primary.grid_signature(reference):
            images=resample_to_img(images,reference,interpolation='continuous',force_resample=True,copy_header=True)
        data=np.asarray(images.dataobj,dtype=np.float32)
        assert np.isfinite(data).all()
        target.mkdir(parents=True,exist_ok=True)
        design.to_csv(target/'design_matrix.tsv',sep='\t',index=False)
        primary.as_float32_image(images).to_filename(target/'fir_coefficients.nii.gz')
        helper.write_json(target/'fir_columns.json',task_columns)
        rois=roi.read_rois();masks=roi.load_masks(rois)
        records=[];outputs=['fir_coefficients.nii.gz','fir_columns.json','roi_kernels.csv','design_matrix.tsv']
        for name,maskimg in masks.items():
            assert roi.same_grid(reference,maskimg)
            means=data[np.asarray(maskimg.dataobj)>0,:].mean(axis=0,dtype=np.float64)
            for cond in primary.CONDITIONS[run]:
                for delay in DELAYS:
                    records.append(dict(subject=subject,run=run,roi=name,condition=cond,delay=delay,
                        time_seconds=2*delay,effect=float(means[task_columns.index(f'{cond}_delay_{delay}')])) )
        pd.DataFrame(records).to_csv(target/'roi_kernels.csv',index=False)
        for cond in primary.CONDITIONS[run]:
            inds=[task_columns.index(f'{cond}_delay_{delay}') for delay in DELAYS]
            summed=data[...,inds].sum(axis=-1,dtype=np.float64)
            name=f'{cond}_effect_size.nii.gz'
            primary.as_float32_image(reference,summed).to_filename(target/name);outputs.append(name)
        qc=dict(subject=subject,run=run,n_rows=x.shape[0],n_columns=x.shape[1],rank=rank,
            residual_df=x.shape[0]-rank,task_columns=len(task_indices),task_rank_after_nuisance=task_rank,
            normalized_condition_number=float(np.linalg.cond(x/np.linalg.norm(x,axis=0))),
            residualized_task_condition_number=float(np.linalg.cond(residualized/np.linalg.norm(residualized,axis=0))),
            warnings=[str(w.message) for w in caught],output_files=outputs,
            script_sha256=helper.digest(__file__),events_sha256=helper.digest(epath),confounds_sha256=helper.digest(cpath))
        helper.write_json(marker,qc)
    return dict(subject=subject,run=run,elapsed_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+');p.add_argument('--runs',nargs='+',type=int,default=list(range(1,9)));p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    assert helper.nilearn.__version__=='0.13.1'
    subjects=args.subjects or primary.read_subjects();assert set(subjects)<=set(primary.read_subjects())
    helper.write_json(REPORT/'execution.json',dict(subjects=subjects,runs=args.runs,delays=DELAYS,
        nilearn=helper.nilearn.__version__,script_sha256=helper.digest(__file__),specification_sha256=helper.digest(REPORT/'ANALYSIS_SPEC.md')))
    jobs=[(s,r) for s in subjects for r in args.runs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fit_run,j) for j in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            print(json.dumps(dict(completed=i,total=len(jobs),**f.result())),flush=True)
    if sorted(args.runs)==list(range(1,9)):
        for subject in subjects:
            primary.build_subject_contrasts(subject,OUT)
            print('Built integrated contrasts: '+subject,flush=True)

if __name__=='__main__':main()
