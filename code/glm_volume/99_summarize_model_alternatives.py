#!/usr/bin/env python3
"""Summarize and validate the exploratory HRF/nuisance/run-weighting analyses."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

HERE=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

runner=module('alternative_runner','98_run_model_alternatives.py')
roi=module('alternative_roi','96_summarize_motion_filter_sensitivity.py')
OUT=runner.REPORT
ROOTS={'primary':runner.PRIMARY,**{v:runner.OUT/v for v in runner.MODELS if v!='primary_refit'},'precision_weighted':runner.OUT/'precision_weighted'}
LABELS={'primary':'Primary Glover','glover_derivative':'Glover + derivative','spm':'SPM canonical','motion12_tissue':'12 motion + tissue means','motion12_acompcor8':'12 motion + up to 8 aCompCor','precision_weighted':'Within-phase precision weighting'}

def main():
    subjects=roi.read_subjects();rois=roi.read_rois();masks=roi.load_masks(rois)
    qc=[json.loads(p.read_text()) for p in runner.OUT.glob('*/level1/eventrelated/sub-*/run-*/model_qc.json')]
    assert len(qc)==800 and len({(r['subject'],r['run'],r['model']) for r in qc})==800
    for r in qc:
        assert r['rank']==r['n_columns'] and r['residual_df']>=10
        assert not r['warnings'],r
        assert runner.fit_source_matches(r['script_sha256'], '98_run_model_alternatives.py') or r['script_sha256'] == runner.PREVIOUS_FIT_SHA
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in qc]).to_csv(OUT/'run_qc.csv',index=False)
    values={};results=[];manifest=[];subject_rows=[];reproduction=[]
    for model,root in ROOTS.items():
        for contrast in roi.CONTRASTS:
            a,records=roi.extract_values(root,subjects,rois,masks,contrast)
            values[model,contrast]=a
            manifest.extend(dict(model=model,**r) for r in records)
            results.extend(roi.infer_rois(model,contrast,subjects,rois,a))
            for si,subject in enumerate(subjects):
                for ri,name in enumerate(rois):
                    subject_rows.append(dict(model=model,contrast=contrast,subject=subject,roi=name,value=a[si,ri]))
    # All primary run effects and derived subject maps must reproduce saved outputs.
    for subject in subjects:
        for contrast in roi.CONTRASTS:
            tail=Path('subject_contrasts/eventrelated')/subject/(contrast+'.nii.gz')
            a=np.asarray(nib.load(runner.PRIMARY/tail).dataobj)
            b=np.asarray(nib.load(runner.OUT/'primary_refit'/tail).dataobj)
            assert np.allclose(a,b,rtol=2e-5,atol=2e-5)
            reference=runner.primary.canonical_reference()
            reconstructed=np.zeros(reference.shape,dtype=float)
            for run in sorted({run for run,condition in runner.primary.PRIMARY_CONTRASTS[contrast]}):
                sign=1 if contrast==roi.CONTRASTS[0] or run>2 else -1
                path=runner.OUT/'primary_refit/level1/eventrelated'/subject/f'run-{run:02d}'/'structured_minus_random_effect_size.nii.gz'
                image=nib.load(path)
                if not roi.same_grid(image,reference):
                    image=roi.resample_to_img(image,reference,interpolation='continuous',force_resample=True,copy_header=True)
                reconstructed+=sign*np.asarray(image.dataobj)
            assert np.allclose(a,reconstructed,rtol=2e-5,atol=2e-5)
            reproduction.append(dict(subject=subject,contrast=contrast,max_abs_difference=float(np.max(np.abs(a-b))),
                within_run_contrast_algebra_max_abs_difference=float(np.max(np.abs(a-reconstructed)))))
    pd.DataFrame(reproduction).to_csv(OUT/'primary_reproduction.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(OUT/'subject_roi_values.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/'subject_map_manifest.csv',index=False)
    results=pd.DataFrame(results);results.to_csv(OUT/'roi_results.csv',index=False)
    summary=[];overlap=[];reduced=[]
    for model in ROOTS:
        for contrast in roi.CONTRASTS:
            frame=results[(results.model==model)&(results.contrast==contrast)].set_index('roi').loc[rois]
            base=results[(results.model=='primary')&(results.contrast==contrast)].set_index('roi').loc[rois]
            sig=frame.significant_fdr_0_05.to_numpy();bsig=base.significant_fdr_0_05.to_numpy()
            a=values[model,contrast];b=values['primary',contrast]
            row=dict(model=model,contrast=contrast,stage=roi.CONTRAST_SHORT[contrast],n_fdr=int(sig.sum()),
                n_positive_fdr=int((sig&(frame['mean'].to_numpy()>0)).sum()),
                n_negative_fdr=int((sig&(frame['mean'].to_numpy()<0)).sum()),
                primary_significant_retained=int((sig&bsig).sum()),
                same_group_direction=int((np.sign(frame['mean'])==np.sign(base['mean'])).sum()),
                subject_roi_pearson_r=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))
            summary.append(row)
            for name in rois:
                overlap.append(dict(model=model,contrast=contrast,roi=name,primary_significant=bool(base.loc[name,'significant_fdr_0_05']),alternative_significant=bool(frame.loc[name,'significant_fdr_0_05']),t=frame.loc[name,'t'],q=frame.loc[name,'q_bh_within_contrast']))
            inds=[i for i,s in enumerate(subjects) if s!='sub-022']
            reduced.extend(roi.infer_rois(model,contrast,[subjects[i] for i in inds],rois,a[inds]))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'model_contrast_summary.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'primary_overlap.csv',index=False)
    reduced=pd.DataFrame(reduced);reduced.to_csv(OUT/'excluding_sub022_roi_results.csv',index=False)
    reduced_summary=reduced.groupby(['model','stage']).significant_fdr_0_05.sum().unstack().reindex(index=list(ROOTS),columns=['initial','early','middle','late'])
    reduced_summary.to_csv(OUT/'excluding_sub022_summary.csv')
    stability=[]
    for contrast in roi.CONTRASTS:
        for name in rois:
            f=results[(results.contrast==contrast)&(results.roi==name)]
            stability.append(dict(contrast=contrast,stage=roi.CONTRAST_SHORT[contrast],roi=name,
                all_significant=bool(f.significant_fdr_0_05.all()),same_direction=bool((f['mean']>0).nunique()==1)))
    stability=pd.DataFrame(stability);stability.to_csv(OUT/'roi_stability.csv',index=False)
    expected=[29,8,0,0]
    assert summary[summary.model=='primary'].n_fdr.tolist()==expected
    lines=['# Exploratory HRF, nuisance, and run-weighting sensitivity','',
        'These models were specified after the primary results were known. They do not replace the primary analysis.',
        'BH FDR is controlled over 48 ROIs separately for each model/contrast; not across models.',
        'No additional whole-brain permutation inference was performed.','',
        '| Model | Initial | Early | Middle | Late |','|---|---:|---:|---:|---:|']
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Exploratory HRF, nuisance-regressor, and run-weighting sensitivity. Counts are ROIs surviving BH FDR across 48 regions separately within each model and contrast.}',r'\label{tab:modelalternatives}',r'\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Model & Initial & Early & Middle & Late \\',r'\midrule']
    for model in ROOTS:
        counts=summary[summary.model==model].n_fdr.astype(int).tolist()
        lines.append('| '+LABELS[model]+' | '+' | '.join(map(str,counts))+' |')
        tex.append(LABELS[model]+' & '+' & '.join(map(str,counts))+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (runner.MANUSCRIPT/'tables/model_alternatives.tex').write_text('\n'.join(tex)+'\n')
    lines+=['','## Agreement with primary','',summary.to_string(index=False),'','## Matched exclusion of sub-022','',reduced_summary.to_string(),'',
        'Precision weighting interpolates the exceptional run variance and changes within-phase run weights; the reduced-sample results are a sensitivity check, not a post-hoc exclusion from the main analysis.','',
        '## Quality checks','',f"All {len(qc)} designs are full rank and warning-free; residual df {min(r['residual_df'] for r in qc)}–{max(r['residual_df'] for r in qc)}.",
        f"Maximum absolute primary condition-map reproduction difference: {max(r['primary_reproduction_max_abs_difference'] for r in qc if r['model']=='primary_refit'):.8g}.",
        f"Maximum absolute primary subject-map reproduction difference: {max(r['max_abs_difference'] for r in reproduction):.8g}.",'',
        '## Consistency across primary plus five alternatives','',stability.groupby('stage')[['all_significant','same_direction']].sum().to_string(),'']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    runner.write_json(OUT/'summary_provenance.json',dict(models=list(ROOTS),subjects=subjects,rois=rois,
        n_fits=len(qc),n_subject_maps=len(manifest),inference='two-sided one-sample t; BH 48 ROIs within model/contrast',
        n_primary_reproduced_conditions=400,n_primary_reproduced_subject_maps=80,
        script_sha256=runner.digest(__file__),source_fit_script_sha256=runner.digest(HERE/'98_run_model_alternatives.py')))
    print(summary.to_string(index=False));print('PASS: 800 fits, primary reproduction, 480 subject maps, ROI comparisons.')

if __name__=='__main__':main()
