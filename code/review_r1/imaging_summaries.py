#!/usr/bin/env python3
"""Fixed R1 ROI families, paired component balance, D1 counts and motion burden."""
from imaging_common import *
from imaging_fits import stage_values
from run_saved import direct
import argparse


def event_diagnostics():
    t=ImagingTask('event_motion_diagnostics');counts=[];motion=[]
    for subject in N22:
        masks=np.loadtxt(t.record(mask_path(subject))).reshape(8,200)
        for run in range(1,9):
            raw=pd.read_csv(t.record(ROOT/'data/behavioral'/f'{subject[-3:]}_run_{run}_srtt_prob_new.csv'))
            repeat=repeats(raw);boundary=raw.block.ne(raw.block.shift())
            any_previous=raw.keypress.eq(raw.keypress.shift())
            for condition,g in raw.groupby('rule_group',sort=False):
                index=g.index
                counts.append(dict(subject=subject,run=run,condition=condition,n_trials=len(g),n_repeats=int(repeat.loc[index].sum()),
                                   n_repeats_cross_block=int((repeat&boundary).loc[index].sum()),
                                   n_equal_previous_excluded_gap=int((any_previous&~raw.stim_onset.diff().eq(1.)).loc[index].sum())))
            ep=t.record(event_path(subject,run));events=pd.read_csv(ep,sep='\t')
            fd=pd.read_csv(t.record(paths(subject,run)[2]),sep='\t').framewise_displacement.to_numpy()
            assert len(fd)==200
            assigned=np.full(200,'fixation_or_gap',dtype=object);clock=np.arange(200)*2.
            for e in events.itertuples():
                condition='random' if e.trial_type.endswith('_Ran') else 'structured'
                select=(clock>=e.onset)&(clock<e.onset+e.duration)
                assert np.all((assigned[select]=='fixation_or_gap')|(assigned[select]==condition))
                assigned[select]=condition
            for condition in ['structured','random']:
                select=assigned==condition;assert select.any()
                motion.append(dict(subject=subject,run=run,phase=STAGES[(run-1)//2],condition=condition,
                                   n_volumes=int(select.sum()),n_fd_defined=int(np.isfinite(fd[select]).sum()),
                                   mean_fd=float(np.nanmean(fd[select])),censored_fraction=float((masks[run-1][select]==0).mean())))
    counts=pd.DataFrame(counts);motion=pd.DataFrame(motion)
    csv(counts,t.path/'participant_repeat_counts.csv');csv(motion,t.path/'participant_run_motion.csv')
    group_counts=[];phase_values=[];paired_values=[];summaries=[]
    for cohort,subjects in [('N22',N22),('N20',N20)]:
        d=counts[counts.subject.isin(subjects)]
        grouped=d.groupby(['run','condition'],as_index=False)[['n_trials','n_repeats','n_repeats_cross_block','n_equal_previous_excluded_gap']].sum()
        grouped['cohort']=cohort;grouped['repeat_rate']=grouped.n_repeats/grouped.n_trials;group_counts.append(grouped)
        # Each participant contributes the equal-run mean within each two-run phase.
        m=motion[motion.subject.isin(subjects)].groupby(['subject','phase','condition'],as_index=False)[['mean_fd','censored_fraction']].mean()
        m['cohort']=cohort;phase_values.append(m)
        for metric in ['mean_fd','censored_fraction']:
            p=m.pivot(index=['subject','phase'],columns='condition',values=metric)
            delta=(p.structured-p.random).rename('value').reset_index();delta['metric']=metric;delta['cohort']=cohort;paired_values.append(delta)
            for phase in STAGES:
                for condition in ['structured','random']:
                    summaries.append(dict(cohort=cohort,phase=phase,metric=metric,contrast=condition,**summarize(m[(m.phase==phase)&(m.condition==condition)][metric])))
                summaries.append(dict(cohort=cohort,phase=phase,metric=metric,contrast='structured_minus_random',**summarize(delta[delta.phase==phase].value)))
    gc=pd.concat(group_counts,ignore_index=True);csv(gc,t.path/'repeat_counts_by_run_condition.csv')
    all_counts=gc.groupby(['cohort','condition'],as_index=False)[['n_trials','n_repeats','n_repeats_cross_block','n_equal_previous_excluded_gap']].sum()
    all_counts['repeat_rate']=all_counts.n_repeats/all_counts.n_trials;csv(all_counts,t.path/'repeat_counts_by_condition.csv')
    csv(pd.concat(phase_values),t.path/'participant_phase_motion.csv');csv(pd.concat(paired_values),t.path/'participant_phase_motion_differences.csv');csv(summaries,t.path/'phase_motion_summary.csv')
    assert len(motion)==22*8*2 and len(counts)==22*20
    report='# D1 repeats and A6 motion burden\n\nD1, fixed by the coordinator before A5/B2 results: current keypress target equals the immediately preceding trial slot in the same run, and stim_onset difference is exactly 1 s. Condition changes and block boundaries do not reset detection; fixation/trigger gaps exclude pairs. First trial has no preceding slot. Counts include errors/omissions because targets, not observed responses, define repeats. The preflight 48 gap-separated Random boundary pairs are excluded.\n\nCondition counts (N20 is supplement-only):\n\n'+'```text\n'+all_counts.to_string(index=False)+'\n```'+'\n\nA6 assigns frame acquisition times 0,2,…398 s to corrected event boxcars using onset≤time<onset+duration. Str1 and Str2 are combined as structured; times outside boxcars are omitted. FD uses fMRIPrep framewise_displacement with missing values omitted, separately reporting n_fd_defined. Censored fractions use all assigned volumes and the reconstructed v8-rule masks; these masks are distinct from an FD threshold. Run values are equally averaged within participant over the two runs of each phase, then participants are equally weighted. Paired structured−random differences are tested against zero using participant phase values and pointwise t 95% CIs; these descriptive p values have no multiplicity adjustment. Participant observations remain internal.\n'
    t.finish(report,dict(passed=True,n_runs=176,n_run_condition_rows=len(motion),n_repeat_condition_rows=len(counts),
                         D1_exact_one_second=True,counts=all_counts.to_dict('records')))


def roi_summaries():
    t=ImagingTask('roi');assert json.loads((OUT/'first_level/validation.json').read_text())['passed']
    fresh=pd.read_csv(t.record(OUT/'first_level/condition_values.csv'));fresh['policy']='observed80'
    prior=pd.read_csv(t.record(EXISTING/'roi_reporting/participant_run_condition_values.csv'))
    prior=prior[prior.model.isin(['reference','censored'])&prior.policy.eq('observed80')]
    data=pd.concat([prior,fresh],ignore_index=True)
    keys=['model','subject','run','condition','roi'];assert not data.duplicated(keys).any()
    assert len(data)==7*22*20*48
    csv(data,t.path/'participant_run_condition_values.csv')
    sv=stage_values(data);sv['policy']='observed80'
    # All-participant expanded-cell comparison follows the pre-full-run controls.
    old=pd.read_csv(t.record(EXISTING/'targeted_sensitivity/participant_stage_values.csv')).query("cohort=='N22' and model=='motion12_acompcor8'")
    check=sv[sv.model.eq('motion12_acompcor8')].merge(old,on=['model','policy','subject','roi','stage'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==22*192
    np.testing.assert_allclose(check.value_new,check.value_old,atol=2e-6,rtol=2e-6,equal_nan=True)
    check['difference']=check.value_new-check.value_old;csv(check,t.path/'participant_acompcor_stage_reproduction.csv')
    stage_all=[];components=[];phase_all=[]
    for cohort,subjects in [('N22',N22),('N20',N20)]:
        s=sv[sv.subject.isin(subjects)].copy();s['cohort']=cohort;stage_all.append(s)
        d=data[data.subject.isin(subjects)]
        p=d.pivot(index=['model','policy','subject','roi'],columns=['run','condition'],values='value')
        S0=(p[1,'Str1']+p[2,'Str1'])/2;R0=(p[1,'Ran']+p[2,'Ran'])/2
        for i,stage in enumerate(STAGES):
            r1,r2=2*i+1,2*i+2;c='Str1' if i==0 else 'Str2'
            S=(p[r1,c]+p[r2,c])/2;R=(p[r1,'Ran']+p[r2,'Ran'])/2
            phase=pd.DataFrame(dict(structured=S,random=R,sequence_minus_random=S-R)).reset_index();phase['stage']=stage;phase['cohort']=cohort;phase_all.append(phase)
            if i:
                deltaS=S-S0;deltaR=R-R0
                v=pd.DataFrame(dict(structured=deltaS,random=deltaR,condition_mean=(deltaS+deltaR)/2,did_half=deltaS-deltaR)).reset_index()
                v['stage']=stage;v['cohort']=cohort;components.append(v)
    sv=pd.concat(stage_all,ignore_index=True);pv=pd.concat(phase_all,ignore_index=True);cv=pd.concat(components,ignore_index=True)
    for v,name in [(sv,'participant_stage_values'),(pv,'participant_phase_components'),(cv,'participant_component_changes')]:csv(v,t.path/f'{name}.csv')
    result=infer(sv,['cohort','model','policy','stage','roi'],['cohort','model','policy','stage'])
    result['q_joint_stages_192']=np.nan
    for _,g in result.groupby(['cohort','model','policy']):
        assert len(g)==192;result.loc[g.index,'q_joint_stages_192']=bh(g.p)
    csv(result,t.path/'stage_results.csv')
    dr=[]
    for cohort,g in sv.groupby('cohort'):
        out=direct(g);out['cohort']=cohort;dr.append(out)
    dr=pd.concat(dr,ignore_index=True);csv(dr,t.path/'direct_phase_results.csv')
    long=cv.melt(id_vars=['cohort','model','policy','subject','roi','stage'],value_vars=['structured','random','condition_mean','did_half'],var_name='component',value_name='value')
    cr=infer(long,['cohort','model','policy','stage','roi','component'],['cohort','model','policy','stage','component'])
    cr['q_headline_early_4']=np.nan
    for _,g in cr[cr.stage.eq('early')&cr.roi.isin(HEADLINE)].groupby(['cohort','model','policy','component']):
        assert len(g)==4;cr.loc[g.index,'q_headline_early_4']=bh(g.p)
    csv(cr,t.path/'component_balance_results.csv');csv(cr[cr.stage.eq('early')&cr.roi.isin(HEADLINE)],t.path/'headline_early_component_balance.csv')
    phase_long=pv.melt(id_vars=['cohort','model','policy','subject','roi','stage'],value_vars=['structured','random','sequence_minus_random'],var_name='component',value_name='value')
    pr=infer(phase_long,['cohort','model','policy','stage','roi','component'],['cohort','model','policy','stage','component'])
    csv(pr,t.path/'phase_component_results.csv')
    identity=cv.merge(sv,on=['cohort','model','policy','subject','roi','stage'],validate='one_to_one')
    np.testing.assert_allclose(identity.did_half*2,identity.value,atol=1e-12,rtol=1e-12,equal_nan=True)
    headline=result[result.roi.isin(HEADLINE)].copy()
    for component in ['structured','random','condition_mean']:
        c=cr[cr.component.eq(component)][['cohort','model','policy','stage','roi','mean','ci_low','ci_high','p','q','q_headline_early_4']]
        c=c.rename(columns={col:f'{component}_{col}' for col in ['mean','ci_low','ci_high','p','q','q_headline_early_4']})
        headline=headline.merge(c,on=['cohort','model','policy','stage','roi'],how='left',validate='one_to_one')
    headline['component_wording']=np.where(headline.stage.eq('initial'),'not_a_change_stage',np.where(headline.condition_mean_ci_high.lt(0),'mainly_reflected_declining_random_responses','relative_contributions_not_resolved'))
    csv(headline,t.path/'headline_tidy_all_specifications.csv')
    oldref=pd.read_csv(t.record(EXISTING/'roi_reporting/stage_results.csv'))
    oldref=oldref[oldref.model.isin(['reference','censored'])&oldref.policy.eq('observed80')]
    z=result.merge(oldref,on=['cohort','model','policy','stage','roi'],suffixes=('_new','_old'),validate='one_to_one')
    for col in ['mean','ci_low','ci_high','p','q']:
        np.testing.assert_allclose(z[col+'_new'],z[col+'_old'],atol=2e-9,rtol=2e-8,equal_nan=True)
    for model in ['motion12_tissue','motion6_acompcor8','motion12_acompcor8','censored_spike','repeat_regressor']:
        c=result[result.model.eq(model)].drop(columns='model').merge(result[result.model.eq('reference')].drop(columns='model'),on=['cohort','policy','stage','roi'],suffixes=('_variant','_reference'))
        c['model']=model;c['mean_shift']=c.mean_variant-c.mean_reference;csv(c,t.path/f'{model}_vs_reference.csv')
    c=result[result.model.eq('censored_spike')].drop(columns='model').merge(result[result.model.eq('censored')].drop(columns='model'),on=['cohort','policy','stage','roi'],suffixes=('_spike','_row_deletion'))
    c['mean_shift_spike_minus_row_deletion']=c.mean_spike-c.mean_row_deletion;csv(c,t.path/'censored_spike_vs_row_deletion.csv')
    focus=headline[headline.cohort.eq('N22')&headline.stage.eq('early')]
    cols=['model','roi','mean','ci_low','ci_high','q','q_joint_stages_192','structured_mean','random_mean','condition_mean_mean','condition_mean_ci_low','condition_mean_ci_high','condition_mean_q_headline_early_4','component_wording']
    report='# R1 ROI inference and A2 component balance\n\nN=22 uncensored reference remains primary; all new fits are sensitivity/secondary analyses, with N20 only for the supplement. Reference and row-deletion censored values reuse existing saved observed80 condition values. The five R1 variants use their new fits. No models, ROIs, stages or families were selected by outcomes. All cohort/model combinations are reported.\n\nStage contrasts retain two-run sums: initial = (Str1−Ran) runs1+2; change stages = (Str2−Ran) in the two stage runs minus initial. One-sample t, df=n−1, pointwise 95% CIs; 48-slot BH separately by stage, and joint 192-slot BH per cohort/model. Untestable slots remain at p=1 in BH. Direct early−middle/early−late/middle−late tests use run_saved.direct and its fixed 144-slot family; within-pair 48-slot q is additionally labeled. No project-wide correction across all sensitivities is claimed.\n\nA2 uses two-run means: ΔS = structured stage mean minus initial Str1 mean; ΔR = random stage mean minus initial Random mean. M=(ΔS+ΔR)/2, and DiD/2=ΔS−ΔR. All participant pairs are retained for paired uncertainty; M SE comes from participant M values, not unpaired component SEs. Full 48×3 change-stage results use BH within stage separately for each component. The predesignated early four (left/right BA6, right pallidum, right thalamus) additionally use four-slot BH, separately for M and for each displayed component. The wording rule depends only on M pointwise CI: declining-random attribution is permitted only if M CI lies entirely below zero; otherwise relative contributions are unresolved. These pointwise intervals are not selection-adjusted.\n\nHistorical ROI keys LeftSMA/RightSMA denote whole BA6 parcels, not SMA-specific localization. Condition coefficient units follow Nilearn percent mean scaling; summed stages are twice the phase-mean contrast. Expanded nuisance changes both motion derivatives and/or tissue treatment as labeled; attenuation does not establish causality. Repeat regression is exploratory and does not isolate longer-term predictability from every task confound. Spike adjustment retains temporal spacing and all rows but does not assert complete elimination of motion bias.\n\nN22 early headline table:\n\n'+'```text\n'+focus[cols].to_string(index=False)+'\n```'+'\n\nAll-participant expanded-nuisance stage reproduction maximum difference: '+str(float(abs(check.difference).max()))+'. Reference/row-deletion group estimates, CIs, p and 48-slot q reproduce saved results within the stated tolerance. Component identity was checked for every participant and change-stage slot.\n'
    t.finish(report,dict(passed=True,n_models=7,n_stage_slots=len(result),n_direct_slots=len(dr),n_component_slots=len(cr),
                         all_stage_families_48=True,all_joint_families_192=True,all_direct_families_144=True,
                         max_expanded_stage_reproduction_difference=float(abs(check.difference).max()),
                         reference_censored_saved_results_reproduced=True,paired_component_identity_checked=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['diagnostics','roi']);a=p.parse_args()
    {'diagnostics':event_diagnostics,'roi':roi_summaries}[a.mode]()
