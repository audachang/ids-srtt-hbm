#!/usr/bin/env python3
"""Coverage, uncertainty, model robustness and paired-stage reporting from saved maps.

No first-level fit or historical product is overwritten. All new extractions use
the participant's eight-run mask intersection, fixed across every comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import nilearn
from nilearn.image import resample_to_img
import numpy as np
import pandas as pd
import scipy
from scipy import stats

STAGES = ('initial','early','middle','late')
CONTRASTS = ('Str1_minus_Ran_r12','Switch_r34_vs_r12_did',
             'Switch_r56_vs_r12_did','Switch_r78_vs_r12_did')
POLICIES = {'legacy_full_parcel':0.,'observed80':.8,'observed95':.95,'complete100':1.}
MODELS = {
    'nocensor_hp128_primary':'FLA_nilearn_float32_20260814',
    'censor_hp128':'FLA_nilearn_sensitivity_20260921/censor_hp128',
    'nocensor_bandpass':'FLA_nilearn_sensitivity_20260921/nocensor_bandpass',
    'censor_bandpass':'FLA_nilearn_sensitivity_20260921/censor_bandpass',
    'glover_derivative':'FLA_nilearn_model_alternatives_20260921/glover_derivative',
    'spm':'FLA_nilearn_model_alternatives_20260921/spm',
    'motion12_tissue':'FLA_nilearn_model_alternatives_20260921/motion12_tissue',
    'motion12_acompcor8':'FLA_nilearn_model_alternatives_20260921/motion12_acompcor8',
    'precision_weighted':'FLA_nilearn_model_alternatives_20260921/precision_weighted',
}
MODEL_LABELS = ('Uncensored high-pass','Censored high-pass','Uncensored band-pass',
                'Censored band-pass','Glover + derivative','SPM HRF',
                'Motion derivatives + tissue','Motion derivatives + aCompCor','Precision weighting')
EXAMPLES = ('LeftSMA','RightDLPFC9','RightInferiorParietal','LeftThalamusProper',
            'LeftHippocampus','LeftAccumbensArea')


def label(roi):
    side = 'Left ' if roi.startswith('Left') else 'Right '
    name = roi.removeprefix('Left').removeprefix('Right')
    names = {'SMA':'BA6','DLPFC9':'BA9','DLPFC46':'BA46','Cerebellum':'cerebellar cortex',
             'PrimaryMotor':'BA4','AccumbensArea':'accumbens','ThalamusProper':'thalamus'}
    import re
    return side + names.get(name,re.sub(r'(?<!^)([A-Z])',r' \1',name).lower())


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def bh(p):
    values=np.asarray(p,float)
    return stats.false_discovery_control(np.where(np.isfinite(values),values,1.),method='bh')


def summarize(x):
    sample=np.asarray(x,float)
    sample=sample[np.isfinite(sample)]
    n=len(sample)
    result=dict(n=n,mean=np.mean(sample) if n else np.nan,sd=np.nan,se=np.nan,
                ci_low=np.nan,ci_high=np.nan,cohen_dz=np.nan,t=np.nan,df=n-1,p=np.nan)
    if n<2: return result
    sd=sample.std(ddof=1)
    se=sd/np.sqrt(n)
    if sd==0: return result | dict(sd=0.,se=0.)
    t=stats.ttest_1samp(sample,0.)
    half=stats.t.ppf(.975,n-1)*se
    return result | dict(sd=sd,se=se,ci_low=sample.mean()-half,
        ci_high=sample.mean()+half,cohen_dz=sample.mean()/sd,t=float(t.statistic),
        p=float(t.pvalue))


def observed_mean(data,roi,support,minimum):
    """Missing tissue stays missing; a genuinely observed zero stays zero."""
    n=int(roi.sum())
    valid=roi & support & np.isfinite(data)
    fraction=int(valid.sum())/n
    if fraction+1e-12 < minimum or not valid.any(): return np.nan
    return float(data[valid].mean())


def same_grid(a,b):
    return a.shape[:3]==b.shape[:3] and np.allclose(a.affine,b.affine,atol=1e-5)


def image_on_grid(path,reference,nearest=False):
    im=nib.load(path)
    if not same_grid(im,reference):
        im=resample_to_img(im,reference,interpolation='nearest' if nearest else 'continuous',
                           force_resample=True,copy_header=True)
    return np.asarray(im.dataobj,dtype=float)


def csv(frame,path):
    frame.to_csv(path,index=False,float_format='%.12g')


def infer(values,keys):
    rows=[]
    # values index: model, policy, stage, subject, roi
    for (model,policy,stage,roi),part in values.groupby(keys,sort=False):
        rows.append(dict(model=model,policy=policy,stage=stage,roi=roi,
                         **summarize(part.value)))
    result=pd.DataFrame(rows)
    result['q_within_contrast']=np.nan
    result['q_joint_stages']=np.nan
    result['q_enumerated_models_stages']=np.nan
    for _,part in result.groupby(['model','policy','stage']):
        assert len(part)==48
        result.loc[part.index,'q_within_contrast']=bh(part.p)
    for _,part in result.groupby(['model','policy']):
        assert len(part)==192
        result.loc[part.index,'q_joint_stages']=bh(part.p)
    for _,part in result.groupby('policy'):
        assert len(part)==1728
        result.loc[part.index,'q_enumerated_models_stages']=bh(part.p)
    result['significant_within_contrast']=result.q_within_contrast.le(.05)&result.p.notna()
    result['significant_joint_stages']=result.q_joint_stages.le(.05)&result.p.notna()
    result['significant_enumerated_family']=result.q_enumerated_models_stages.le(.05)&result.p.notna()
    result['direction']=np.where(result['mean'].gt(0),'positive',np.where(result['mean'].lt(0),'negative','undefined'))
    return result


def make_table(path,caption,columns,rows,tablelabel=None,note=''):
    text=['\\small','\\begin{longtable}{'+'l'*len(columns)+'}',
          '\\caption{'+caption+'}'+('\\label{'+tablelabel+'}' if tablelabel else '')+r'\\',
          r'\toprule',' & '.join(columns)+r' \\',r'\midrule',r'\endfirsthead',
          r'\toprule',' & '.join(columns)+r' \\',r'\midrule',r'\endhead',
          r'\bottomrule',r'\endfoot']
    text+=[' & '.join(row)+r' \\' for row in rows]
    text += [r'\end{longtable}',note,'']
    path.write_text('\n'.join(text))


def fmt(x,d=3):
    if not np.isfinite(x): return '--'
    return f'{x:.{d}f}'


def figures(result,conditions,coverage,out,figdir,rois):
    colors=('#607284','#8050a3','#218c72','#d37736')
    # All models are compared in their original signed coefficient units.
    focus=result[(result.policy=='observed80')&result.roi.isin(EXAMPLES)]
    fig,axes=plt.subplots(3,2,figsize=(11,11),sharex=True)
    for ax,roi in zip(axes.flat,EXAMPLES):
        for j,stage in enumerate(STAGES):
            d=focus[(focus.roi==roi)&(focus.stage==stage)].set_index('model').reindex(MODELS)
            y=np.arange(9)+(j-1.5)*.17
            ax.errorbar(d['mean'],y,xerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),
                        fmt='o',ms=3,linewidth=.8,color=colors[j],label=stage.title())
        ax.axvline(0,color='.65',lw=.8)
        ax.set_yticks(np.arange(9),MODEL_LABELS,fontsize=7)
        ax.invert_yaxis();ax.set_title(label(roi),fontsize=11)
        ax.grid(axis='x',alpha=.16)
    axes[0,0].legend(ncol=2,fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('Summed contrast estimate (95% CI)')
    fig.suptitle('Model sensitivity on the same observed ROI support',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97));savefig(fig,figdir/'model_effect_intervals')

    # Run condition coefficients -> phase means -> DiD, maintaining same support.
    allconditions=conditions[(conditions.policy=='observed80')]
    phase_rows=[]
    for (s,roi),p in allconditions.groupby(['subject','roi']):
        for i,stage in enumerate(STAGES):
            cond='Str1' if i==0 else 'Str2'
            runs=[i*2+1,i*2+2]
            pair=p[p.run.isin(runs)]
            a=pair[pair.condition==cond].value.to_numpy()
            b=pair[pair.condition=='Ran'].value.to_numpy()
            # Require both runs, rather than pandas silently averaging one.
            am=np.mean(a) if len(a)==2 and np.isfinite(a).all() else np.nan
            bm=np.mean(b) if len(b)==2 and np.isfinite(b).all() else np.nan
            phase_rows.append(dict(subject=s,roi=roi,stage=stage,structured=am,random=bm,
                                   sequence_minus_random=am-bm))
    phase=pd.DataFrame(phase_rows)
    csv(phase,out/'participant_phase_components.csv')
    component_summary=[]
    for (roi,stage),p in phase.groupby(['roi','stage']):
        for component in ['structured','random','sequence_minus_random']:
            component_summary.append(dict(roi=roi,stage=stage,component=component,
                                          **summarize(p[component])))
    cs=pd.DataFrame(component_summary)
    csv(cs,out/'phase_component_results.csv')
    primary=result[(result.model=='censor_hp128')&(result.policy=='observed80')]
    def panel(ax,roi,components=False):
        if components:
            for col,c,name in [('structured','#8050a3','Structured'),('random','.4','Random')]:
                a=cs[(cs.roi==roi)&(cs.component==col)].set_index('stage').reindex(STAGES)
                ax.errorbar(np.arange(4),a['mean'],yerr=np.vstack([a['mean']-a.ci_low,a.ci_high-a['mean']]),
                            fmt='o-',color=c,ms=4,lw=1,label=name)
            ax.set_ylabel('Mean run coefficient')
        else:
            a=primary[primary.roi==roi].set_index('stage').reindex(STAGES)
            ax.errorbar(np.arange(4),a['mean'],yerr=np.vstack([a['mean']-a.ci_low,a.ci_high-a['mean']]),
                        fmt='o',color='#218c72',ms=4,lw=1)
            ax.plot(np.arange(1,4),a['mean'].to_numpy()[1:],color='#218c72',lw=1)
            ax.axvline(.5,color='.75',ls=':',lw=.8)
            ax.set_ylabel('Initial / summed DiD')
        ax.axhline(0,color='.7',lw=.7)
        ax.set_xticks(range(4),['Initial','Early','Middle','Late'],fontsize=8)
        ax.grid(alpha=.15);ax.set_title(label(roi),fontsize=10)
    fig,axes=plt.subplots(3,4,figsize=(13,10))
    for i,roi in enumerate(EXAMPLES):
        r=i//2;c=(i%2)*2
        panel(axes[r,c],roi,True);panel(axes[r,c+1],roi,False)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Conditions and difference-in-differences (pointwise 95% CI)',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97));savefig(fig,figdir/'condition_did_components')
    with PdfPages(out/'all_roi_condition_profiles.pdf') as pdf:
        for offset in range(0,48,6):
            fig,axes=plt.subplots(3,4,figsize=(13,10))
            for i,roi in enumerate(rois[offset:offset+6]):
                r=i//2;c=(i%2)*2
                panel(axes[r,c],roi,True);panel(axes[r,c+1],roi,False)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,12))
    matrix=coverage.pivot(index='roi',columns='subject',values='fraction').reindex(rois)
    im=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(48),[label(x) for x in rois],fontsize=7)
    ax.set_xticks(range(20),matrix.columns,rotation=90,fontsize=7)
    ax.set_title('ROI coverage: intersection of all eight fitting masks')
    fig.colorbar(im,ax=ax,label='Fraction of original parcel observed',pad=.015)
    fig.tight_layout();savefig(fig,figdir/'roi_coverage')


def savefig(fig,path):
    fig.savefig(str(path)+'.png',dpi=240,bbox_inches='tight')
    fig.savefig(str(path)+'.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[2])
    args=p.parse_args();root=args.source_root.resolve();dest=args.output_root.resolve()
    out=dest/'results/glm_volume/reporting_20261012';out.mkdir(parents=True,exist_ok=True)
    figdir=dest/'manuscript/figures';figdir.mkdir(parents=True,exist_ok=True)
    tables=dest/'manuscript/tables';tables.mkdir(parents=True,exist_ok=True)
    subjects=(root/'code/replication/config/subjects.txt').read_text().split()
    rois=(root/'code/replication/config/rois.txt').read_text().split()
    assert len(subjects)==20 and len(set(subjects))==20 and len(rois)==48
    spec=json.loads((Path(__file__).parent/'reporting_spec.json').read_text())
    (out/'analysis_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    manifest={}
    def record(path):
        path=Path(path)
        if str(path) not in manifest: manifest[str(path)]=digest(path)
        return path
    ref=nib.load(record(root/'resources/roi_masks'/f'{rois[0]}.nii.gz'))
    masks={r:image_on_grid(record(root/'resources/roi_masks'/f'{r}.nii.gz'),ref,True)>0 for r in rois}
    supports={};coverage_rows=[];run_coverage=[]
    for s in subjects:
        support=np.ones(ref.shape,bool)
        for run in range(1,9):
            maskpath=root/'data/derivatives/fmriprep'/s/'func/Masks'/f'{s}_task-srttprob_run-{run:02d}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
            rm=image_on_grid(record(maskpath),ref,True)>0
            support &= rm
            for r,m in masks.items():
                run_coverage.append(dict(subject=s,run=run,roi=r,n_roi_voxels=int(m.sum()),
                                         n_observed_voxels=int((rm&m).sum()),fraction=float(rm[m].mean())))
        supports[s]=support
        for r,m in masks.items():
            coverage_rows.append(dict(subject=s,roi=r,n_roi_voxels=int(m.sum()),
                n_observed_voxels=int((support&m).sum()),fraction=float(support[m].mean())))
    coverage=pd.DataFrame(coverage_rows)
    csv(coverage,out/'participant_roi_coverage.csv');csv(pd.DataFrame(run_coverage),out/'run_roi_coverage.csv')
    covsummary=[]
    for r in rois:
        a=coverage[coverage.roi==r]
        covsummary.append(dict(roi=r,n_nominal=20,min_fraction=a.fraction.min(),
            median_fraction=a.fraction.median(),n_observed80=int(a.fraction.ge(.8).sum()),
            n_observed95=int(a.fraction.ge(.95).sum()),n_complete100=int(a.fraction.eq(1).sum())))
    covsummary=pd.DataFrame(covsummary);csv(covsummary,out/'roi_coverage_summary.csv')
    rows=[]
    for model,relative in MODELS.items():
        print('Extracting',model,flush=True)
        modelroot=root/'results/glm_volume/first_level'/relative
        for stage,contrast in zip(STAGES,CONTRASTS):
            for s in subjects:
                path=modelroot/'subject_contrasts/eventrelated'/s/f'{contrast}.nii.gz'
                data=image_on_grid(record(path),ref)
                if not np.isfinite(data).all(): raise ValueError(f'Nonfinite map {path}')
                for r,m in masks.items():
                    for policy,cut in POLICIES.items():
                        value=float(data[m].mean()) if policy=='legacy_full_parcel' else observed_mean(data,m,supports[s],cut)
                        rows.append(dict(model=model,policy=policy,stage=stage,subject=s,roi=r,value=value))
    values=pd.DataFrame(rows);csv(values,out/'participant_roi_values.csv')
    result=infer(values,['model','policy','stage','roi'])
    csv(result,out/'all_model_roi_results.csv')
    primary=result[(result.model=='censor_hp128')&(result.policy=='observed80')]
    csv(primary,out/'reported_roi_results.csv')
    # Independent reproduction of all historical one-sample means and statistics.
    prior=pd.read_csv(record(root/'results/glm_volume/sensitivity/roi_results.csv'))
    chk=result[(result.policy=='legacy_full_parcel')&result.model.isin(prior.model.unique())]
    merged=chk.merge(prior,on=['model','roi'],suffixes=('_new','_saved'))
    merged=merged[merged.stage_new==merged.stage_saved]
    assert len(merged)==768
    for field,old in [('mean','mean'),('sd','sd'),('t','t'),('p','p_two_sided'),('q_within_contrast','q_bh_within_contrast')]:
        if not np.allclose(merged[field+'_new' if field in ['mean','sd','t'] else field],
                           merged[old+'_saved' if old in ['mean','sd','t'] else old],rtol=2e-7,atol=2e-9):
            raise AssertionError('Historical reproduction failed: '+field)
    altprior=pd.read_csv(record(root/'results/glm_volume/model_alternatives_20260921/roi_results.csv'))
    altcheck=result[(result.policy=='legacy_full_parcel')&result.model.isin(altprior.model.unique())]
    altmerged=altcheck.merge(altprior,on=['model','roi','stage'],suffixes=('_new','_saved'))
    assert len(altmerged)==960
    for field in ('mean','sd','t'):
        assert np.allclose(altmerged[field+'_new'],altmerged[field+'_saved'],rtol=2e-7,atol=2e-9)
    assert np.allclose(altmerged.q_within_contrast,altmerged.q_bh_within_contrast,rtol=2e-7,atol=2e-9)
    comparisons=[]
    selected=values[values.model=='censor_hp128']
    for policy in POLICIES:
        pivot=selected[selected.policy==policy].pivot(index=['subject','roi'],columns='stage',values='value')
        for a,b in [('early','middle'),('early','late'),('middle','late')]:
            for r in rois:
                part=pivot.xs(r,level='roi')
                comparisons.append(dict(policy=policy,pair=f'{a}_minus_{b}',roi=r,
                                        **summarize(part[a]-part[b])))
    comparisons=pd.DataFrame(comparisons)
    comparisons['q_three_pairs_144']=np.nan;comparisons['q_within_pair_48']=np.nan
    for _,part in comparisons.groupby('policy'):
        assert len(part)==144
        comparisons.loc[part.index,'q_three_pairs_144']=bh(part.p)
    for _,part in comparisons.groupby(['policy','pair']):
        comparisons.loc[part.index,'q_within_pair_48']=bh(part.p)
    csv(comparisons,out/'direct_stage_comparisons.csv')
    conditionrows=[]
    modelroot=root/'results/glm_volume/first_level'/MODELS['censor_hp128']
    for s in subjects:
        for run in range(1,9):
            conditions=['Ran','Str1'] if run<=2 else ['Ran','Str2'] if run>=7 else ['Ran','Str1','Str2']
            for condition in conditions:
                path=modelroot/'level1/eventrelated'/s/f'run-{run:02d}'/f'{condition}_effect_size.nii.gz'
                data=image_on_grid(record(path),ref)
                for r,m in masks.items():
                    for policy,cut in POLICIES.items():
                        val=float(data[m].mean()) if policy=='legacy_full_parcel' else observed_mean(data,m,supports[s],cut)
                        conditionrows.append(dict(subject=s,run=run,condition=condition,roi=r,policy=policy,value=val))
    conditions=pd.DataFrame(conditionrows);csv(conditions,out/'participant_run_condition_values.csv')
    # Reconstruct summed DiD from independently extracted condition maps.
    maxerrors={}
    for policy in POLICIES:
        cp=conditions[conditions.policy==policy].pivot(index=['subject','roi'],columns=['run','condition'],values='value')
        initial=cp[1,'Str1']+cp[2,'Str1']-cp[1,'Ran']-cp[2,'Ran']
        for i,stage in enumerate(STAGES):
            actual=initial if i==0 else cp[i*2+1,'Str2']+cp[i*2+2,'Str2']-cp[i*2+1,'Ran']-cp[i*2+2,'Ran']-initial
            expected=selected[(selected.policy==policy)&(selected.stage==stage)].set_index(['subject','roi']).value.reindex(actual.index)
            assert np.array_equal(np.isfinite(actual),np.isfinite(expected))
            error=float(np.nanmax(np.abs(actual-expected)))
            if error>2e-7: raise AssertionError(f'Condition algebra mismatch {policy} {stage}: {error}')
            maxerrors[policy+'_'+stage]=error
    # Summary distinguishes significance from direction stability and precision.
    modelsummary=[]
    for (model,policy,stage),part in result.groupby(['model','policy','stage'],sort=False):
        modelsummary.append(dict(model=model,policy=policy,stage=stage,
            n_testable=int(part.p.notna().sum()),n_within48=int(part.significant_within_contrast.sum()),
            n_joint192=int(part.significant_joint_stages.sum()),n_enumerated1728=int(part.significant_enumerated_family.sum()),
            min_n=int(part.n.min()),max_n=int(part.n.max())))
    csv(pd.DataFrame(modelsummary),out/'model_stage_summary.csv')
    stability=[]
    obs=result[result.policy=='observed80']
    for (stage,r),part in obs.groupby(['stage','roi']):
        stability.append(dict(stage=stage,roi=r,n_models=9,n_testable=int(part.p.notna().sum()),
            same_direction=bool(part['mean'].notna().all() and (part['mean'].gt(0).all() or part['mean'].lt(0).all())),
            n_significant_within48=int(part.significant_within_contrast.sum()),
            mean_min=part['mean'].min(),mean_max=part['mean'].max(),
            ci_width_min=(part.ci_high-part.ci_low).min(),ci_width_max=(part.ci_high-part.ci_low).max()))
    csv(pd.DataFrame(stability),out/'model_stability.csv')
    figures(result,conditions,coverage,out,figdir,rois)
    rows=[]
    for stage in STAGES:
        for r in rois:
            a=primary[(primary.stage==stage)&(primary.roi==r)].iloc[0]
            rows.append([stage.title(),label(r),str(a.n),fmt(a['mean']),fmt(a.sd),fmt(a.se),
                         '['+fmt(a.ci_low)+', '+fmt(a.ci_high)+']',fmt(a.cohen_dz),fmt(a.t),
                         fmt(a.p),fmt(a.q_within_contrast),fmt(a.q_joint_stages)])
    # Landscape longtable keeps all 192 results, including null/untestable slots.
    make_table(tables/'coverage_all_roi_results.tex',
       'All ROI results on observed support (80\\% coverage requirement). Pointwise intervals are unadjusted.',
       ['Stage','ROI','$N$','Mean','SD','SE','95\\% CI','$d_z$','$t$','$p$','$q_{48}$','$q_{192}$'],rows,
       'tab:allcoverage',note='The 48 planned ROI slots are retained in BH adjustment; untestable slots enter correction as $p=1$, without inventing a measured effect. Contrast estimates sum two run coefficients. $q_{192}$ additionally adjusts the four stages jointly. Intervals and $d_z$ remain conditional on the inspected model and sample.')
    rows=[]
    for _,a in covsummary.iterrows():
        rows.append([label(a.roi),fmt(100*a.min_fraction,1),fmt(100*a.median_fraction,1),
                     str(a.n_observed80),str(a.n_observed95),str(a.n_complete100)])
    make_table(tables/'coverage_summary.tex','ROI support across all eight fitting masks.',
       ['ROI','Minimum \\%','Median \\%','$N_{80}$','$N_{95}$','$N_{100}$'],rows,'tab:coverage')
    rows=[]
    for _,a in comparisons[comparisons.policy=='observed80'].iterrows():
        pair=a.pair.replace('_minus_',' -- ').title()
        rows.append([pair,label(a.roi),str(a.n),fmt(a['mean']),
                     '['+fmt(a.ci_low)+', '+fmt(a.ci_high)+']',fmt(a.cohen_dz),fmt(a.t),fmt(a.p),fmt(a.q_three_pairs_144)])
    make_table(tables/'direct_stage_results.tex','Direct within-participant comparisons of sequence-change phases.',
        ['Comparison','ROI','$N$','Mean difference','95\\% CI','$d_z$','$t$','$p$','$q_{144}$'],rows,'tab:directstage')
    summary={'historical_reproduction_rows':1728,'condition_algebra_max_errors':maxerrors,
             'models':list(MODELS),'subjects':subjects,'policies':list(POLICIES),
             'selected_model_counts':{},'direct_stage_counts':{},'n_inputs':len(manifest)}
    for policy in POLICIES:
        a=result[(result.model=='censor_hp128')&(result.policy==policy)]
        summary['selected_model_counts'][policy]={stage:int(a[a.stage==stage].significant_within_contrast.sum()) for stage in STAGES}
        a=comparisons[comparisons.policy==policy]
        summary['direct_stage_counts'][policy]={pair:int(((a.pair==pair)&(a.q_three_pairs_144<=.05)&a.p.notna()).sum()) for pair in a.pair.unique()}
    summary['software']={'nilearn':nilearn.__version__,'numpy':np.__version__,'scipy':scipy.__version__}
    summary['source_root']=str(root)
    summary['script_sha256']=digest(__file__)
    summary['spec_sha256']=digest(Path(__file__).parent/'reporting_spec.json')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary['selected_model_counts'],indent=2),flush=True)
    print('Direct stage counts',summary['direct_stage_counts'],flush=True)


if __name__=='__main__': main()
