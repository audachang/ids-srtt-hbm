#!/usr/bin/env python3
"""Complete the canonical SPM, aCompCor and run-precision T6 updates."""
from updated_common import *
import concurrent.futures
from nilearn.glm.first_level import FirstLevelModel
from run_saved import direct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ALT=FIRST/'FLA_nilearn_model_alternatives_20260921'
NEW=['spm','motion12_acompcor8','precision_weighted']
MOTION=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']


def fit(job):
    subject,run=job;out=BASE/'targeted_sensitivity/runs'/subject/f'run-{run:02d}'
    marker=out/'qc.json'
    if marker.exists():
        q=json.loads(marker.read_text());assert q['producer']==digest(__file__)
        for p,sha in q['outputs'].items():assert digest(out/p)==sha
        return q
    bold,mask,cpath,_=paths(subject,run);epath=event_path(subject,run)
    e=pd.read_csv(epath,sep='\t');e.trial_type=e.trial_type.str.extract('(Ran|Str[12])$')[0]
    c=pd.read_csv(cpath,sep='\t');metadata=json.loads(cpath.with_suffix('.json').read_text())
    components=sorted(k for k,v in metadata.items() if k.startswith('a_comp_cor_') and v.get('Mask')=='combined' and v.get('Retained') is True and k in c)
    assert components
    inputs=[bold,mask,cpath,cpath.with_suffix('.json'),epath];checks=[];diagnostics=[]
    for variant in ['reference_variance','spm','motion12_acompcor8']:
        cols=MOTION+[v+'_derivative1' for v in MOTION]+components[:8] if variant=='motion12_acompcor8' else CONFOUNDS
        for col in cols:
            missing=np.flatnonzero(c[col].isna());assert len(missing)==0 or (col.endswith('_derivative1') and np.array_equal(missing,[0]))
        model=FirstLevelModel(t_r=2.,hrf_model='spm' if variant=='spm' else 'glover',drift_model='cosine',high_pass=1/128,
            smoothing_fwhm=4.,signal_scaling=0,noise_model='ar1',mask_img=str(mask),standardize=False,minimize_memory=True,n_jobs=1)
        model.fit(str(bold),events=e,confounds=c[cols].fillna(0).astype(float))
        folder=out/variant;folder.mkdir(parents=True,exist_ok=True);design=model.design_matrices_[0]
        assert np.linalg.matrix_rank(design)==design.shape[1];design.to_csv(folder/'design_matrix.tsv',sep='\t',index=False)
        diagnostics.append(dict(model=variant,subject=subject,run=run,rank=design.shape[1],n_rows=len(design),confounds=list(cols)))
        if variant=='reference_variance':
            condition='Str1' if run<=2 else 'Str2';result=model.compute_contrast(condition+' - Ran',output_type='all')
            for key in ['effect_size','effect_variance']:save32(np.asarray(result[key].dataobj),result[key],folder/f'structured_minus_random_{key}.nii.gz')
            fresh=BASE/'first_level/runs'/subject/f'run-{run:02d}/reference'
            for cond in conditions(run):
                arr=np.asarray(model.compute_contrast(cond,output_type='effect_size').dataobj)
                prior=np.asarray(nib.load(fresh/f'{cond}_effect_size.nii.gz').dataobj)
                np.testing.assert_allclose(arr,prior,atol=2e-5,rtol=2e-5)
                checks.append(float(abs(arr-prior).max()))
        else:
            for condition in conditions(run):
                im=model.compute_contrast(condition,output_type='effect_size');save32(np.asarray(im.dataobj),im,folder/f'{condition}_effect_size.nii.gz')
        # Two unchanged controls also reproduce the historical alternative maps.
        if (subject,run) in [('sub-001',1),('sub-020',6)]:
            old=ALT/('primary_refit' if variant=='reference_variance' else variant)/'level1/eventrelated'/subject/f'run-{run:02d}'
            for path in folder.glob('*.nii.gz'):
                expected=np.asarray(nib.load(old/path.name).dataobj);actual=np.asarray(nib.load(path).dataobj)
                np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5);checks.append(float(abs(actual-expected).max()));inputs.append(old/path.name)
    q=dict(subject=subject,run=run,producer=digest(__file__),updated_common_sha256=digest(Path(__file__).with_name('updated_common.py')),
        inputs={str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in inputs},outputs={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},diagnostics=diagnostics,max_reproduction_difference=max(checks))
    writej(marker,q);return q


def main():
    t=Task('targeted_sensitivity');inventory=pd.read_csv(t.record(OLD_BASE/'T3_timing_repair/event_inventory.csv'))
    changed=set(zip(inventory[inventory.changed].subject,inventory[inventory.changed].run))
    refreshed={(s,r) for s in N22 for r in range(1,9) if s not in N20 or (s,r) in changed}
    jobs=sorted(refreshed|{('sub-001',1),('sub-020',6)})
    qcs=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for q in pool.map(fit,jobs):qcs.append(q);print(q['subject'],q['run'],'fits passed',flush=True)
    for q in qcs:
        for path in q['inputs']:t.record(path)
        t.record(t.path/'runs'/q['subject']/f'run-{q["run"]:02d}'/'qc.json')
    ref=nib.load(t.record(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz'))
    parcels={roi:image(t.record(ROOT/'resources/roi_masks'/f'{roi}.nii.gz'),ref,True)>0 for roi in ROIS}
    rows=[];weight_stats=[]
    affected={s for s,r in refreshed}
    for subject in N22:
        support=np.logical_and.reduce([image(t.record(paths(subject,r)[1]),ref,True)>0 for r in range(1,9)])
        for variant in NEW:
            if subject not in affected:
                for stage,contrast in zip(STAGES,CONTRASTS):
                    p=t.record(ALT/variant/'subject_contrasts/eventrelated'/subject/f'{contrast}.nii.gz');d=image(p,ref)
                    for roi,value in extract(d,subject,parcels,{subject:support}).items():rows.append(dict(model=variant,policy='observed80',subject=subject,roi=roi,stage=stage,value=value))
                continue
            phases=[]
            for a,b in [(1,2),(3,4),(5,6),(7,8)]:
                effects=[];variances=[]
                for run in [a,b]:
                    fresh=(subject,run) in refreshed
                    folder=t.path/'runs'/subject/f'run-{run:02d}'/('reference_variance' if variant=='precision_weighted' else variant) if fresh else ALT/('primary_refit' if variant=='precision_weighted' else variant)/'level1/eventrelated'/subject/f'run-{run:02d}'
                    if variant=='precision_weighted':
                        effects.append(image(t.record(folder/'structured_minus_random_effect_size.nii.gz'),ref))
                        variances.append(image(t.record(folder/'structured_minus_random_effect_variance.nii.gz'),ref))
                    else:
                        condition='Str1' if run<=2 else 'Str2'
                        effects.append(image(t.record(folder/f'{condition}_effect_size.nii.gz'),ref)-image(t.record(folder/'Ran_effect_size.nii.gz'),ref))
                if variant=='precision_weighted':
                    e=np.array(effects);v=np.array(variances);precision=np.divide(1,v,out=np.zeros_like(v),where=np.isfinite(v)&(v>0));total=precision.sum(axis=0)
                    weights=np.divide(precision,total,out=np.zeros_like(v),where=total>0);phase=2*(weights*e).sum(axis=0)
                    weight_stats.append(dict(subject=subject,pair=f'{a}{b}',both_valid=int((v>0).all(axis=0).sum()),neither_valid=int((total<=0).sum())))
                else:phase=sum(effects)
                phases.append(phase)
            for idx,stage in enumerate(STAGES):
                data=phases[0] if idx==0 else phases[idx]-phases[0]
                folder=t.path/'subject_contrasts'/variant/subject;folder.mkdir(parents=True,exist_ok=True);save32(data,ref,folder/f'{stage}.nii.gz')
                # Extraction from the saved float32 contrast matches the historical convention.
                for roi,value in extract(np.asarray(nib.load(folder/f'{stage}.nii.gz').dataobj),subject,parcels,{subject:support}).items():rows.append(dict(model=variant,policy='observed80',subject=subject,roi=roi,stage=stage,value=value))
    participant=pd.DataFrame(rows);allparticipants=[]
    for cohort,subjects in [('N20',N20),('N22',N22)]:
        d=participant[participant.subject.isin(subjects)].copy();d['cohort']=cohort;allparticipants.append(d)
    participant=pd.concat(allparticipants);csv(participant,t.path/'participant_stage_values.csv');csv(weight_stats,t.path/'precision_pair_support.csv')
    result=infer(participant,['cohort','model','policy','stage','roi'],['cohort','model','policy','stage']);csv(result,t.path/'stage_results.csv')
    mainvalues=pd.read_csv(t.record(BASE/'roi_reporting/participant_stage_values.csv')).query('policy=="observed80"')
    combined=pd.concat([mainvalues,participant]);csv(combined,t.path/'all_participant_stage_values.csv')
    stages=infer(combined,['cohort','model','policy','stage','roi'],['cohort','model','policy','stage']);csv(stages,t.path/'all_stage_results.csv')
    dr=[];influence=[]
    for (cohort,model),g in combined.groupby(['cohort','model']):
        d=direct(g);d['cohort']=cohort;dr.append(d)
        for omitted in ['none']+(N20 if cohort=='N20' else N22):
            part=g if omitted=='none' else g[g.subject!=omitted];r=direct(part)
            row=r[(r.roi=='RightAccumbensArea')&(r.pair=='early_minus_late')].iloc[0].to_dict();influence.append(dict(cohort=cohort,omitted_subject=omitted,**row))
    csv(pd.concat(dr),t.path/'all_direct_phase_results.csv');csv(influence,t.path/'all_right_accumbens_influence.csv')
    with PdfPages(t.path/'all_N22_specifications.pdf') as pdf:
        for roi in ROIS:
            fig,axes=plt.subplots(1,4,figsize=(14,4),sharey=True)
            for ax,stage in zip(axes,STAGES):
                d=stages[(stages.cohort=='N22')&(stages.stage==stage)&(stages.roi==roi)].set_index('model').reindex(VARIANTS+NEW)
                ax.errorbar(d['mean'],range(9),xerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=3)
                ax.axvline(0,color='.7');ax.set(title=stage,yticks=range(9),yticklabels=VARIANTS+NEW)
            axes[0].invert_yaxis();fig.suptitle(roi+'; N22; pointwise 95% CI');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    t.finish('# Completed targeted T6 specifications\n\nCanonical SPM HRF, Glover with motion12 plus up to eight retained combined-mask aCompCor components selected by the original metadata rule, and voxelwise inverse-run-variance combination are updated for both samples. SPM+temporal derivative, censor on/off, errors/omissions and two RT adjustments are in roi_reporting. Together these give nine observed80 specifications. Only added/changed runs are refitted; two unchanged control runs reproduce the historical alternative maps and all refreshed Glover fits reproduce the current reference betas. Unchanged historical stage/condition/variance maps are reused with pinned hashes. Precision weighting uses twice the inverse-variance-weighted two-run phase mean, preserving original sum units; DiD subtracts the weighted initial phase. It changes run weights rather than introducing a new participant-level sampling model. Full planned 48-stage and 144-direct families are retained. Right accumbens early−late is recomputed after every participant deletion across all nine specifications and both samples. The primary significant-slot Cook-style diagnostics remain in roi_reporting. Band-pass variants are not counted as robustness evidence; their historical diagnostic flags remain in the previous audit.\n',dict(passed=True,n_refitted_runs=len(jobs),n_models_per_run=3,n_stage_slots=len(result),n_all_stage_slots=len(stages),n_direct_slots=sum(len(x) for x in dr),n_all_models=9,n_control_runs=2,max_control_or_reference_reproduction_difference=max(q['max_reproduction_difference'] for q in qcs)))


if __name__=='__main__':main()
