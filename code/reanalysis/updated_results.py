#!/usr/bin/env python3
"""Matched N20/N22 reporting and subject maps from corrected timing inputs."""
from updated_common import *
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from run_saved import direct
from polish_figures import label


def map_path(model, subject, run, condition):
    fresh = BASE/'first_level/runs'/subject/f'run-{run:02d}'/model/f'{condition}_effect_size.nii.gz'
    if fresh.exists():
        return fresh
    assert model in MODELS and subject in N20
    inventory = pd.read_csv(OLD_BASE/'T3_timing_repair/event_inventory.csv')
    assert not inventory[(inventory.subject==subject)&(inventory.run==run)].changed.iloc[0]
    return FIRST/MODELS[model]/'level1/eventrelated'/subject/f'run-{run:02d}'/f'{condition}_effect_size.nii.gz'


def main():
    t = Task('roi_reporting')
    fit_validation = json.loads(t.record(BASE/'first_level/validation.json').read_text())
    assert fit_validation['passed']
    old = pd.read_csv(t.record(OLD_BASE/'T2_within_run/participant_run_condition_values.csv'))
    old = old[['model','subject','run','condition','roi','value']]
    extras = pd.read_csv(t.record(OLD_BASE/'new_first_level/condition_values.csv'))
    extras = extras[extras.model.isin(['spm_derivative','errors_omissions','rt_modulator'])]
    shared = pd.read_csv(t.record(OLD_BASE/'shared_rt_first_level/condition_values.csv'))
    cached = pd.concat([old,extras,shared],ignore_index=True)
    fresh = pd.read_csv(t.record(BASE/'first_level/condition_values.csv'))
    keys = ['model','subject','run','condition','roi']
    overlap = cached.merge(fresh,on=keys,suffixes=('_cached','_fresh'))
    inventory = pd.read_csv(t.record(OLD_BASE/'T3_timing_repair/event_inventory.csv'))
    changed = set(zip(inventory[inventory.changed].subject,inventory[inventory.changed].run))
    stable = overlap[[((s,r) not in changed) for s,r in zip(overlap.subject,overlap.run)]]
    np.testing.assert_allclose(stable.value_cached,stable.value_fresh,atol=1e-6,rtol=1e-6,equal_nan=True)
    replaced = pd.MultiIndex.from_frame(fresh[keys])
    cached = cached[~pd.MultiIndex.from_frame(cached[keys]).isin(replaced)]
    values80 = pd.concat([cached,fresh],ignore_index=True)
    assert not values80.duplicated(keys).any() and len(values80)==len(VARIANTS)*22*20*48
    values80['policy']='observed80'
    ref = nib.load(t.record(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz'))
    parcels = {roi:image(t.record(ROOT/'resources/roi_masks'/f'{roi}.nii.gz'),ref,True)>0 for roi in ROIS}
    supports={};indices={};coverage=[]
    for subject in N22:
        support=np.logical_and.reduce([image(t.record(paths(subject,r)[1]),ref,True)>0 for r in range(1,9)])
        supports[subject]=support
        for roi,parcel in parcels.items():
            observed=parcel&support;fraction=observed.sum()/parcel.sum()
            indices[subject,roi]=(np.flatnonzero(observed),fraction)
            coverage.append(dict(subject=subject,roi=roi,n_roi=int(parcel.sum()),n_observed=int(observed.sum()),fraction=fraction))
    csv(coverage,t.path/'participant_roi_coverage.csv')
    additional=[];extracted80=[]
    for model in MODELS:
        for subject in N22:
            for run in range(1,9):
                for condition in conditions(run):
                    data=image(t.record(map_path(model,subject,run,condition)),ref).ravel()
                    assert np.isfinite(data).all()
                    for roi in ROIS:
                        index,fraction=indices[subject,roi]
                        mean=float(data[index].mean()) if len(index) else np.nan
                        for policy,cut in POLICIES.items():
                            row=dict(model=model,subject=subject,run=run,condition=condition,roi=roi,policy=policy,
                                     value=mean if fraction+1e-12>=cut else np.nan)
                            (extracted80 if policy=='observed80' else additional).append(row)
    check=pd.DataFrame(extracted80).merge(values80,on=keys+['policy'],suffixes=('_independent','_assembled'),validate='one_to_one')
    np.testing.assert_allclose(check.value_independent,check.value_assembled,atol=1e-9,rtol=1e-9,equal_nan=True)
    conditions_all=pd.concat([values80,pd.DataFrame(additional)],ignore_index=True)
    csv(conditions_all,t.path/'participant_run_condition_values.csv')
    stage_values=[];within_values=[];components=[];slopes=[];trajectories=[];trend_covariance=[]
    for cohort,subjects in [('N20',N20),('N22',N22)]:
        data=conditions_all[conditions_all.subject.isin(subjects)].copy()
        data['cohort']=cohort
        p=data.pivot(index=['cohort','model','policy','subject','roi'],columns=['run','condition'],values='value')
        baseline=p[1,'Str1']+p[2,'Str1']-p[1,'Ran']-p[2,'Ran']
        for i,stage in enumerate(STAGES):
            value=baseline if i==0 else p[2*i+1,'Str2']+p[2*i+2,'Str2']-p[2*i+1,'Ran']-p[2*i+2,'Ran']-baseline
            block=value.rename('value').reset_index();block['stage']=stage;stage_values.append(block)
            r1,r2=2*i+1,2*i+2;c='Str1' if i==0 else 'Str2'
            block=pd.DataFrame(dict(structured=(p[r1,c]+p[r2,c])/2,random=(p[r1,'Ran']+p[r2,'Ran'])/2)).reset_index()
            block['stage']=stage;block['sequence_minus_random']=block.structured-block.random;components.append(block)
        for contrast,runs in [('run3',[3]),('run4',[4]),('run5',[5]),('run6',[6]),('early_r34',[3,4]),('middle_r56',[5,6]),('pooled_r36',[3,4,5,6])]:
            block=(sum(p[r,'Str2']-p[r,'Str1'] for r in runs)/len(runs)).rename('value').reset_index()
            block['contrast']=contrast;within_values.append(block)
        for target,other in [('Str2_minus_Str1','Str1'),('Str2_minus_Ran','Ran')]:
            block=sum(w*(p[r,'Str2']-p[r,other]) for w,r in zip([-.3,-.1,.1,.3],[3,4,5,6])).rename('value').reset_index()
            block['contrast']=target;slopes.append(block)
            runs=pd.DataFrame({r:p[r,'Str2']-p[r,other] for r in [3,4,5,6]})
            for key,group in runs.groupby(level=['cohort','model','policy','roi']):
                complete=group.dropna();n=len(complete);weights=np.array([-.3,-.1,.1,.3])
                covariance=complete.cov().to_numpy() if n>1 else np.full((4,4),np.nan)
                propagated=float(weights@covariance@weights/n) if n>1 else np.nan
                check=complete.to_numpy()@weights
                np.testing.assert_allclose(propagated,np.var(check,ddof=1)/n if n>1 else np.nan,atol=1e-12,equal_nan=True)
                trend_covariance.append(dict(zip(['cohort','model','policy','roi'],key))|dict(contrast=target,n=n,
                    propagated_group_slope_variance=propagated,**{f'cov_run{r}_run{s}':covariance[i,j] for i,r in enumerate([3,4,5,6]) for j,s in enumerate([3,4,5,6])}))
        trajectories.append(infer(data,['cohort','model','policy','run','condition','roi'],['cohort','model','policy','run','condition']))
    sv=pd.concat(stage_values,ignore_index=True);wv=pd.concat(within_values,ignore_index=True);pv=pd.concat(components,ignore_index=True);tv=pd.concat(slopes,ignore_index=True)
    sr=infer(sv,['cohort','model','policy','stage','roi'],['cohort','model','policy','stage'])
    wr=infer(wv,['cohort','model','policy','contrast','roi'],['cohort','model','policy','contrast'])
    tr=infer(tv,['cohort','model','policy','contrast','roi'],['cohort','model','policy','contrast'])
    cv=pd.DataFrame(trend_covariance)
    check=cv.merge(tr,on=['cohort','model','policy','roi','contrast'],validate='one_to_one')
    np.testing.assert_allclose(check.propagated_group_slope_variance,check.se**2,atol=1e-12,equal_nan=True)
    csv(cv,t.path/'run_covariance_propagation.csv')
    for data,name in [(sv,'participant_stage_values'),(wv,'participant_within_run_values'),(pv,'participant_phase_components'),
                      (tv,'participant_slopes'),(sr,'stage_results'),(wr,'within_run_results'),(tr,'trend_results'),
                      (pd.concat(trajectories),'trajectory_results')]:csv(data,t.path/f'{name}.csv')
    direct_rows=[]
    for cohort,group in sv.groupby('cohort'):
        out=direct(group);out['cohort']=cohort;direct_rows.append(out)
    dr=pd.concat(direct_rows,ignore_index=True);csv(dr,t.path/'direct_phase_results.csv')
    components_summary=[];covariances=[];changes=[]
    for key,group in pv.groupby(['cohort','model','policy','stage','roi']):
        clean=group.dropna(subset=['structured','random']);n=len(clean)
        for component in ['structured','random','sequence_minus_random']:
            components_summary.append(dict(zip(['cohort','model','policy','stage','roi'],key))|dict(component=component,**summarize(clean[component])))
        covariance=clean[['structured','random']].cov().iloc[0,1] if n>1 else np.nan
        covariances.append(dict(zip(['cohort','model','policy','stage','roi'],key))|dict(n=n,structured_random_covariance=covariance,covariance_of_means=covariance/n if n else np.nan))
    for key,group in pv.groupby(['cohort','model','policy','roi']):
        for component in ['structured','random','sequence_minus_random']:
            p=group.pivot(index='subject',columns='stage',values=component)
            for stage in STAGES[1:]:changes.append(dict(zip(['cohort','model','policy','roi'],key))|dict(stage=stage,component=component,**summarize(p[stage]-p.initial)))
    cs=pd.DataFrame(components_summary)
    csv(cs,t.path/'phase_component_results.csv');csv(covariances,t.path/'phase_component_covariance.csv');csv(changes,t.path/'phase_component_changes.csv')
    compare=sr[sr.cohort=='N20'].drop(columns='cohort').merge(sr[sr.cohort=='N22'].drop(columns='cohort'),on=['model','policy','stage','roi'],suffixes=('_N20','_N22'))
    compare['mean_shift_N22_minus_N20']=compare.mean_N22-compare.mean_N20
    csv(compare,t.path/'N20_vs_N22_stage_comparison.csv')
    previous=pd.read_csv(t.record(OLD_BASE/'T1_reference/all_model_roi_results.csv'))
    timing=sr[(sr.cohort=='N20')&sr.model.isin(MODELS)].drop(columns='cohort').merge(previous,on=['model','policy','stage','roi'],suffixes=('_repaired','_historical'))
    timing['mean_shift_timing_repair']=timing.mean_repaired-timing.mean_historical
    csv(timing,t.path/'N20_timing_repair_comparison.csv')
    influence=[]
    for (cohort,model),group in sv[sv.policy=='observed80'].groupby(['cohort','model']):
        subjects=N20 if cohort=='N20' else N22
        for omitted in ['none']+subjects:
            part=group if omitted=='none' else group[group.subject!=omitted]
            d=direct(part)
            row=d[(d.roi=='RightAccumbensArea')&(d.pair=='early_minus_late')].iloc[0].to_dict()
            influence.append(dict(cohort=cohort,omitted_subject=omitted,**row))
    csv(influence,t.path/'right_accumbens_early_minus_late.csv')
    significance=sr[(sr.cohort=='N22')&(sr.model=='reference')&(sr.policy=='observed80')&sr.q.le(.05)]
    loo=[]
    data=sv[(sv.cohort=='N22')&(sv.model=='reference')&(sv.policy=='observed80')]
    for _,row in significance.iterrows():
        x=data[(data.stage==row.stage)&(data.roi==row.roi)].set_index('subject').value
        for omitted in N22:
            z=summarize(x.drop(omitted));cook=row.n*(z['mean']-row['mean'])**2/row.sd**2 if row.sd>0 else np.nan
            loo.append(dict(stage=row.stage,roi=row.roi,omitted_subject=omitted,cook_style_mean_influence=cook,**z))
    csv(loo,t.path/'primary_significant_slot_influence.csv')
    focus=['LeftSMA','RightAccumbensArea','LeftPutamen','LeftCaudate','LeftHippocampus','LeftCerebellum']
    fig,axes=plt.subplots(1,4,figsize=(14,4.8),sharey=True)
    for ax,stage in zip(axes,STAGES):
        for cohort,offset,color in [('N20',-.12,'#8d969d'),('N22',.12,'#286a9c')]:
            a=sr[(sr.model=='reference')&(sr.policy=='observed80')&(sr.cohort==cohort)&(sr.stage==stage)].set_index('roi').reindex(focus)
            ax.errorbar(a['mean'],np.arange(6)+offset,xerr=np.vstack([a['mean']-a.ci_low,a.ci_high-a['mean']]),fmt='o',ms=4,color=color,label=cohort)
        ax.axvline(0,color='.7',lw=.7);ax.set_title(stage);ax.set_yticks(range(6),[label(r) for r in focus])
    axes[0].invert_yaxis();axes[0].legend();fig.suptitle('Corrected timing: reference N20/N22; pointwise 95% CI');fig.tight_layout();savefig(fig,t.path/'sample_comparison')
    with PdfPages(t.path/'updated_model_specifications.pdf') as pdf:
        for roi in ROIS:
            fig,axes=plt.subplots(1,4,figsize=(13,3.5),sharey=True)
            for ax,stage in zip(axes,STAGES):
                a=sr[(sr.cohort=='N22')&(sr.policy=='observed80')&(sr.stage==stage)&(sr.roi==roi)].set_index('model').reindex(VARIANTS)
                ax.errorbar(a['mean'],range(6),xerr=np.vstack([a['mean']-a.ci_low,a.ci_high-a['mean']]),fmt='o',ms=3,color='#286a9c')
                ax.axvline(0,color='.7',lw=.7);ax.set_title(stage);ax.set_yticks(range(6),VARIANTS)
            axes[0].invert_yaxis();fig.suptitle(label(roi)+'; N22; pointwise 95% CI');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    t.finish('# Updated ROI reporting: matched N20/N22\n\nSix updated specifications use the same corrected timing and participant-specific observed eight-run support. Reference/censored results additionally include 95% and complete-coverage sensitivities; other specifications retain observed80. All planned 48 slots remain in each stage/within-run/trend BH family; direct phase comparisons retain 144 slots per cohort/model/policy. Pointwise t CIs are not simultaneous or corrected for model selection. Initial and DiD coefficients use historical two-run sums; pooled within-run contrasts and component curves use equal-run means. No absent condition is imputed. The two samples overlap: N22-minus-N20 mean shifts are descriptive and are not tested as independent groups. Historical N20 versus corrected N20 is reported separately. Both error and RT sensitivities are statistical adjustments, not measured cue duration or causal controls. Updated specifications do not silently inherit results from the additional historical motion/HRF/weighting alternatives that were not refitted for N22.\n',
             dict(passed=True,n_subjects_N20=20,n_subjects_N22=22,n_stage_slots=len(sr),n_within_run_slots=len(wr),
                  n_trend_slots=len(tr),n_direct_slots=len(dr),n_fresh_condition_values=len(fresh),
                  unchanged_fit_roi_checks=len(stable),independent_primary_extraction_rows=len(check),
                  all_reference_control_checks_passed=True))
    # Prepare one shared N22 mask and maps for matched voxelwise comparisons.
    maps=Task('voxelwise_inputs')
    old_mask=nib.load(maps.record(ROOT/'results/glm_volume/whole_brain/common_intersection_mask.nii.gz'))
    assert old_mask.shape==ref.shape;np.testing.assert_allclose(old_mask.affine,ref.affine,atol=1e-5)
    common=np.asarray(old_mask.dataobj)>0
    for subject in N22:
        common &= supports[subject]
        for run in range(1,9):maps.record(paths(subject,run)[1])
    assert common.sum()>40000
    save32(common.astype(float),ref,maps.path/'common_N22_mask.nii.gz')
    manifest=[]
    for model in ['reference','censored','errors_omissions','rt_shared_modulator']:
        for subject in N22:
            pool=np.zeros(ref.shape,dtype=float)
            by_run={}
            selected=range(1,9) if model in MODELS else range(3,7)
            for run in selected:
                by_run[run]={c:image(maps.record(map_path(model,subject,run,c)),ref) for c in conditions(run)}
                if run in [3,4,5,6]:pool+=(by_run[run]['Str2']-by_run[run]['Str1'])/4
            folder=maps.path/model/subject;folder.mkdir(parents=True,exist_ok=True)
            path=folder/'pooled_r36_Str2_minus_Str1.nii.gz';save32(pool,ref,path)
            manifest.append(dict(model=model,subject=subject,contrast='pooled_r36_Str2_minus_Str1',path=str(path),sha256=digest(path)))
            if model in MODELS:
                baseline=sum(by_run[r]['Str1']-by_run[r]['Ran'] for r in [1,2])
                for i,stage in enumerate(STAGES):
                    data=baseline if i==0 else sum(by_run[r]['Str2']-by_run[r]['Ran'] for r in [2*i+1,2*i+2])-baseline
                    path=folder/f'{stage}.nii.gz';save32(data,ref,path)
                    manifest.append(dict(model=model,subject=subject,contrast=stage,path=str(path),sha256=digest(path)))
    csv(manifest,maps.path/'subject_map_manifest.csv')
    maps.finish('# Updated voxelwise inputs\n\nAll four historical stage contrasts for reference/censored models and pooled within-run Str2−Str1 for reference/censored/error/shared-RT specifications. The shared search mask is the historical N20 common mask intersected with observed support in every run of all 22 participants. Both N20 and N22 inference use this same mask; historical full-mask counts cannot be compared as if the family were unchanged. Explicit unscaled float32 subject maps and their hashes are retained. No significance inference is performed by this input-preparation task.\n',
                dict(passed=True,n_maps=len(manifest),n_common_voxels=int(common.sum()),
                     n_historical_mask_voxels=int((np.asarray(old_mask.dataobj)>0).sum()),shared_mask_for_both_samples=True))


if __name__=='__main__':main()
