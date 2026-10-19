#!/usr/bin/env python3
"""Validate FIR fits, summarize signed integrals, and test full ROI time courses."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fir_runner',HERE/'100_run_fir_analysis.py')
fir=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fir;spec.loader.exec_module(fir)
roi=fir.roi;helper=fir.helper;OUT=fir.REPORT
STAGES=['initial','early','middle','late'];N_PERM=10000;SEED=20260921

def label(name):
    return name.replace('Left','L ').replace('Right','R ').replace('DLPFC46','BA46').replace('DLPFC9','BA9').replace('SMA','BA6').replace('PrimaryMotor','BA4').replace('ThalamusProper','Thalamus').replace('Cerebellum','Cerebellar cortex')

def permutation_tests(data):
    n=data.shape[0];flat=data.reshape(n,-1)
    mean=flat.mean(axis=0);ss=(flat*flat).sum(axis=0)
    def studentized(means):
        variance=np.maximum((ss-n*means*means)/(n-1),np.finfo(float).tiny)
        return means/np.sqrt(variance/n)
    observed=studentized(mean).reshape(data.shape[1:])
    assert np.allclose(observed,stats.ttest_1samp(data,0,axis=0).statistic,rtol=1e-10,atol=1e-10)
    global_counts=np.zeros(observed.shape,dtype=int)
    roi_counts=np.zeros(observed.shape[:-1],dtype=int)
    raw_counts=np.zeros(observed.shape,dtype=int)
    omnibus=np.max(np.abs(observed),axis=-1)
    rng=np.random.default_rng(SEED)
    max_distribution=[]
    for start in range(0,N_PERM,100):
        signs=rng.choice([-1.,1.],size=(min(100,N_PERM-start),n))
        perm=studentized(signs@flat/n).reshape((-1,)+observed.shape)
        maxima=np.max(np.abs(perm),axis=(1,2,3))
        max_distribution.extend(maxima.tolist())
        global_counts+=(maxima[:,None,None,None]>=np.abs(observed)[None,...]).sum(axis=0)
        roi_counts+=(np.max(np.abs(perm),axis=-1)>=omnibus[None,...]).sum(axis=0)
        raw_counts+=(np.abs(perm)>=np.abs(observed)[None,...]).sum(axis=0)
    return observed,(global_counts+1)/(N_PERM+1),(roi_counts+1)/(N_PERM+1),(raw_counts+1)/(N_PERM+1),np.array(max_distribution)

def main():
    subjects=roi.read_subjects();rois=roi.read_rois();contrasts=list(fir.primary.PRIMARY_CONTRASTS)
    paths=sorted(fir.OUT.glob('level1/eventrelated/sub-*/run-*/model_qc.json'))
    assert len(paths)==160
    qcs=[json.loads(p.read_text()) for p in paths]
    assert len({(r['subject'],r['run']) for r in qcs})==160
    for r in qcs:
        assert r['rank']==r['n_columns'] and r['task_rank_after_nuisance']==r['task_columns']
        assert not r['warnings'],r
        assert helper.fit_source_matches(r['script_sha256'], '100_run_fir_analysis.py')
    pd.DataFrame(qcs).to_csv(OUT/'run_qc.csv',index=False)
    run_tables=[]
    for p in paths:
        run_tables.append(pd.read_csv(p.parent/'roi_kernels.csv'))
        stack=nib.load(p.parent/'fir_coefficients.nii.gz')
        a=np.asarray(stack.dataobj);cols=json.loads((p.parent/'fir_columns.json').read_text())
        assert stack.header.get_data_dtype()==np.dtype('float32') and a.shape[-1]==len(cols) and np.isfinite(a).all()
        for cond in fir.primary.CONDITIONS[json.loads(p.read_text())['run']]:
            inds=[cols.index(f'{cond}_delay_{d}') for d in fir.DELAYS]
            saved=np.asarray(nib.load(p.parent/f'{cond}_effect_size.nii.gz').dataobj)
            assert np.allclose(a[...,inds].sum(axis=-1,dtype=float),saved,rtol=2e-5,atol=2e-5)
    runs=pd.concat(run_tables,ignore_index=True)
    runs.to_csv(OUT/'run_condition_roi_kernels.csv',index=False)
    indexed=runs.set_index(['subject','run','roi','condition','delay']).effect
    data=np.zeros((20,4,48,16),dtype=float)
    for si,subject in enumerate(subjects):
        for ci,contrast in enumerate(contrasts):
            for (run,cond),weight in fir.primary.PRIMARY_CONTRASTS[contrast].items():
                for ri,name in enumerate(rois):
                    data[si,ci,ri,:]+=weight*np.array([indexed.loc[(subject,run,name,cond,d)] for d in fir.DELAYS])
    np.savez_compressed(OUT/'subject_roi_kernels.npz',values=data,subjects=subjects,rois=rois,contrasts=contrasts,delays=fir.DELAYS)
    records=[]
    for si,s in enumerate(subjects):
        for ci,c in enumerate(contrasts):
            for ri,r in enumerate(rois):
                for d in fir.DELAYS:
                    records.append(dict(subject=s,contrast=c,stage=STAGES[ci],roi=r,delay=d,time_seconds=2*d,effect=data[si,ci,ri,d]))
    pd.DataFrame(records).to_csv(OUT/'subject_roi_kernels.csv',index=False)
    masks=roi.load_masks(rois);integrated=[];manifests=[];summary=[]
    for ci,c in enumerate(contrasts):
        a,m=roi.extract_values(fir.OUT,subjects,rois,masks,c)
        assert np.allclose(a,data[:,ci].sum(axis=-1),rtol=2e-5,atol=2e-5)
        integrated.extend(roi.infer_rois('fir_integrated',c,subjects,rois,a));manifests+=m
    integrated=pd.DataFrame(integrated);integrated.to_csv(OUT/'integrated_roi_results.csv',index=False)
    pd.DataFrame(manifests).to_csv(OUT/'subject_map_manifest.csv',index=False)
    t,pfwe,pomni,praw,maxdist=permutation_tests(data)
    assert np.all(pfwe>=praw) and np.all(pomni>=np.min(praw,axis=-1))
    np.save(OUT/'permutation_max_abs_t.npy',maxdist)
    means=data.mean(axis=0);se=stats.sem(data,axis=0);ci95=stats.t.ppf(.975,19)*se
    lagrows=[];omnirows=[]
    for ci,c in enumerate(contrasts):
        q=roi.bh_adjust(pomni[ci])
        for ri,r in enumerate(rois):
            omnirows.append(dict(contrast=c,stage=STAGES[ci],roi=r,max_abs_t=float(np.abs(t[ci,ri]).max()),
                p_temporal_max_t=pomni[ci,ri],q_bh_within_contrast=q[ri],significant_fdr_0_05=bool(q[ri]<=.05)))
            for d in fir.DELAYS:
                lagrows.append(dict(contrast=c,stage=STAGES[ci],roi=r,delay=d,time_seconds=2*d,
                    mean=means[ci,ri,d],se=se[ci,ri,d],ci95_lower=means[ci,ri,d]-ci95[ci,ri,d],ci95_upper=means[ci,ri,d]+ci95[ci,ri,d],
                    t=t[ci,ri,d],p_signflip=praw[ci,ri,d],p_fwe_all_rois_lags_contrasts=pfwe[ci,ri,d],significant_fwe_0_05=bool(pfwe[ci,ri,d]<=.05)))
        f=integrated[integrated.contrast==c];sig=f.significant_fdr_0_05
        summary.append(dict(contrast=c,stage=STAGES[ci],integrated_n_fdr=int(sig.sum()),
            integrated_positive=int((sig&(f['mean']>0)).sum()),integrated_negative=int((sig&(f['mean']<0)).sum()),
            temporal_omnibus_n_fdr=int((q<=.05).sum()),global_fwe_bins=int((pfwe[ci]<=.05).sum()),global_fwe_rois=int((pfwe[ci]<=.05).any(axis=-1).sum())))
    pd.DataFrame(lagrows).to_csv(OUT/'lag_roi_results.csv',index=False)
    pd.DataFrame(omnirows).to_csv(OUT/'temporal_omnibus_roi_results.csv',index=False)
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    # Complete, unselected visualization of every ROI and contrast.
    vmax=max(3.,float(np.ceil(np.max(np.abs(t)))))
    fig,axes=plt.subplots(2,2,figsize=(14,16),layout='constrained')
    for ci,ax in enumerate(axes.flat):
        im=ax.imshow(t[ci],aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
        yy,xx=np.where(pfwe[ci]<=.05);ax.scatter(xx,yy,s=11,c='black',marker='o')
        ax.set_title(STAGES[ci].title(),fontsize=12);ax.set_yticks(range(48),[label(r) for r in rois],fontsize=6)
        ax.set_xticks(range(0,16,2),[str(2*d) for d in range(0,16,2)]);ax.set_xlabel('FIR kernel lag (s; 2-s bins)')
    fig.colorbar(im,ax=axes,label='One-sample t(19)',shrink=.6)
    fig.suptitle('Duration-adjusted FIR contrast profiles\nDots: max-|t| FWE p ≤ .05 across 48 ROIs × 16 bins × 4 contrasts',fontsize=13)
    fpath=helper.MANUSCRIPT/'figures/fir_roi_lag_heatmap.png';fig.savefig(fpath,dpi=300);plt.close(fig)
    colors=['#2166ac','#b2182b','#4d9221','#762a83'];times=np.arange(16)*2
    pdfpath=OUT/'fir_all_roi_timecourses.pdf'
    with PdfPages(pdfpath) as pdf:
        for start in range(0,48,6):
            fig,axes=plt.subplots(3,2,figsize=(11.7,8.3),layout='constrained')
            for ax,ri in zip(axes.flat,range(start,start+6)):
                for ci in range(4):
                    ax.plot(times,means[ci,ri],color=colors[ci],label=STAGES[ci].title(),lw=1.2)
                    ax.fill_between(times,means[ci,ri]-ci95[ci,ri],means[ci,ri]+ci95[ci,ri],color=colors[ci],alpha=.09)
                ax.axhline(0,color='0.5',lw=.6);ax.set_title(label(rois[ri]),fontsize=10)
                ax.set_xlabel('Kernel lag (s)');ax.set_ylabel('FIR coefficient contrast');ax.tick_params(labelsize=8)
            axes[0,0].legend(fontsize=7,ncol=2)
            fig.suptitle('Duration-adjusted FIR kernel profiles: mean ± pointwise 95% CI\nIntervals are descriptive; corrected tests are in the accompanying tables.',fontsize=12)
            pdf.savefig(fig);plt.close(fig)
    lines=['# Exploratory duration-aware FIR analysis','',
        '16 two-second kernel bins spanning 0–32 s; original event durations retained.',
        'These are fitted kernel profiles, not event-triggered BOLD averages.',
        'All 48 ROIs and four contrast time courses are reported; no ROI selected by FIR significance.','',summary.to_string(index=False),'',
        f"160 full-rank designs; residual df {min(r['residual_df'] for r in qcs)}–{max(r['residual_df'] for r in qcs)}.",
        f"Residualized task-design condition numbers {min(r['residualized_task_condition_number'] for r in qcs):.3f}–{max(r['residualized_task_condition_number'] for r in qcs):.3f}.",
        '','Integrated-kernel estimates sum all bins and use two-sided one-sample t tests with BH over 48 ROIs within each contrast.',
        'Temporal omnibus tests use max-|t| across 16 bins within ROI, then BH over 48 ROIs within each contrast.',
        'Binwise FWE p uses the maximum absolute t across all 3,072 ROI/bin/contrast tests in each of 10,000 shared participant sign flips (seed 20260921).',
        'No correction across integrated and temporal families or across exploratory model variants.',
        'The FIR integral is a different estimand from a canonical HRF coefficient; significance differences do not establish differences between models.',
        'No whole-brain voxelwise inference was performed for FIR.',
        '','Files: `fir_all_roi_timecourses.pdf`, `lag_roi_results.csv`, `temporal_omnibus_roi_results.csv`, `integrated_roi_results.csv`, `subject_roi_kernels.npz`.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Exploratory FIR ROI results. Integrated estimates use BH FDR across 48 ROIs within contrast. Temporal omnibus tests first account for 16 bins within each ROI, then apply BH across 48 ROIs. Binwise FWE is controlled jointly across all 3,072 ROI/bin/contrast tests.}',r'\label{tab:fir}',r'\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Contrast & Integral ROIs & Omnibus ROIs & FWE ROIs & FWE bins \\',r'\midrule']
    for r in summary.to_dict('records'):
        tex.append(f"{r['stage'].title()} & {r['integrated_n_fdr']} & {r['temporal_omnibus_n_fdr']} & {r['global_fwe_rois']} & {r['global_fwe_bins']}"+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);(helper.MANUSCRIPT/'tables/fir_results.tex').write_text('\n'.join(tex)+'\n')
    helper.write_json(OUT/'provenance.json',dict(n_subjects=20,n_rois=48,n_fits=160,n_delays=16,n_permutations=N_PERM,seed=SEED,
        family_size=3072,script_sha256=helper.digest(__file__),fit_script_sha256=helper.digest(HERE/'100_run_fir_analysis.py'),
        figure_sha256=helper.digest(fpath),profiles_pdf_sha256=helper.digest(pdfpath),specification_sha256=helper.digest(OUT/'ANALYSIS_SPEC.md')))
    print(summary.to_string(index=False));print('PASS: FIR coefficients, 160 full-rank designs, integrated-map algebra, shared sign-flip inference, figures.')

if __name__=='__main__':main()
