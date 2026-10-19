from slicetime_common import *
from slicetime_fits import stage_values
from run_saved import direct

def roi_summaries():
    t=ImagingTask('roi');assert json.loads((OUT/'first_level/validation.json').read_text())['passed']
    fresh=pd.read_csv(t.record(OUT/'first_level/condition_values.csv'));fresh['policy']='observed80'
    prior=pd.read_csv(t.record(EXISTING/'roi_reporting/participant_run_condition_values.csv'))
    prior=prior[prior.model.isin(['reference'])&prior.policy.eq('observed80')]
    data=pd.concat([prior,fresh],ignore_index=True)
    keys=['model','subject','run','condition','roi'];assert not data.duplicated(keys).any()
    assert len(data)==2*22*20*48
    csv(data,t.path/'participant_run_condition_values.csv')
    sv=stage_values(data);sv['policy']='observed80'
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
    oldref=oldref[oldref.model.isin(['reference'])&oldref.policy.eq('observed80')]
    z=result.merge(oldref,on=['cohort','model','policy','stage','roi'],suffixes=('_new','_old'),validate='one_to_one')
    for col in ['mean','ci_low','ci_high','p','q']:
        np.testing.assert_allclose(z[col+'_new'],z[col+'_old'],atol=2e-9,rtol=2e-8,equal_nan=True)
    model='slicetime_aligned'
    comparison=result[result.model.eq(model)].drop(columns='model').merge(result[result.model.eq('reference')].drop(columns='model'),on=['cohort','policy','stage','roi'],suffixes=('_aligned','_reference'),validate='one_to_one')
    comparison['mean_shift']=comparison.mean_aligned-comparison.mean_reference
    csv(comparison,t.path/'slicetime_aligned_vs_reference.csv')
    focus=comparison[(comparison.stage.eq('early')&comparison.roi.isin(HEADLINE)) | (comparison.stage.eq('initial')&comparison.roi.isin(['LeftAccumbensArea','RightAccumbensArea']))]
    assert len(focus)==12
    csv(focus,t.path/'headline_and_accumbens_comparison.csv')
    t.finish('# A7 ROI inference and component balance\n\nN22 unaligned reference remains primary. Aligned sensitivity and reference use observed80 with participant eight-run support; N20 is supplement-only. All 48 slots retained. Stage effects use two-run sums, components use two-run means. One-sample t, df=n-1, pointwise 95% CIs; BH48 per stage/component, joint BH192, direct phase BH144 and additional labelled within-pair BH48. Four early headline components additionally receive BH4. M=(deltaS+deltaR)/2; declining-random wording only when pointwise M CI is entirely below zero. Untestable p enters BH as 1. Historical LeftSMA/RightSMA keys denote complete BA6 parcels. Comparisons are descriptive and do not test model differences.\n\n'+focus.to_string(index=False),dict(passed=True,n_models=2,n_stage_slots=len(result),n_direct_slots=len(dr),n_component_slots=len(cr),all_stage_families_48=True,all_joint_families_192=True,all_direct_families_144=True,reference_saved_results_reproduced=True,paired_component_identity_checked=True))


if __name__=='__main__':roi_summaries()
