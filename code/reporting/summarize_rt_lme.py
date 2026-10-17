#!/usr/bin/env python3
"""Plot and independently validate the mixed-model RT estimates and contrasts."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from augment_roi_reporting import digest, savefig, fmt, make_table
from validate_reporting import independent_bh
from make_reporting_tables import probability


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
    root=a.root.resolve();out=root/'results/behavioral/fmri_rt_20261012/lme'
    figures=root/'manuscript/figures';tables=root/'manuscript/tables'
    statuses={}
    for policy in ['primary','min100']:
        s=json.loads((out/f'{policy}_status.json').read_text());statuses[policy]=s
        assert s['status']=='completed' and s['n_subjects']==20 and s['design_rank']==20
        co=pd.read_csv(out/f'{policy}_fixed_coefficients.csv').set_index('term')['Value']
        cov=pd.read_csv(out/f'{policy}_fixed_covariance.csv',index_col=0).loc[co.index,co.index]
        weights=pd.read_csv(out/f'{policy}_contrast_weights.csv').set_index('contrast').loc[:,co.index]
        d=pd.read_csv(out/f'{policy}_contrasts.csv').set_index('contrast').loc[weights.index]
        estimate=weights.to_numpy()@co.to_numpy()
        se=np.sqrt(np.einsum('ij,jk,ik->i',weights,cov,weights))
        assert len(d)==24 and (d.df==19).all()
        np.testing.assert_allclose(d.log_difference,estimate,rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(d.se,se,rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(d.p_two_sided,2*stats.t.sf(abs(estimate/se),19),rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(d.q_bh_24,independent_bh(d.p_two_sided),rtol=2e-9,atol=1e-11)
        half=stats.t.ppf(.975,19)*se
        np.testing.assert_allclose(d.ratio_ci_low,np.exp(estimate-half),rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(d.ratio_ci_high,np.exp(estimate+half),rtol=2e-9,atol=1e-11)
        cells=pd.read_csv(out/f'{policy}_cell_estimates.csv')
        assert len(cells)==20
        for _,r in cells.iterrows():
            term='cell'+r.cell
            np.testing.assert_allclose(r.log_rt,co[term],rtol=2e-9)
            np.testing.assert_allclose(r.se,np.sqrt(cov.loc[term,term]),rtol=2e-9)
        rows=[]
        for name,r in d.iterrows():
            name=name.replace('_',' ').replace('minus','--')
            rows.append([name,fmt(r.log_difference),fmt(r.se),fmt(r.t,2),probability(r.p_two_sided),probability(r.q_bh_24),
                fmt(r.geometric_rt_ratio),'['+fmt(r.ratio_ci_low)+', '+fmt(r.ratio_ci_high)+']'])
        make_table(tables/f'rt_lme_{policy}.tex',
            ('Primary' if policy=='primary' else '100-ms sensitivity')+' trial-level RT LME contrasts. All tests use conservative participant-based $df=19$.',
            ['Contrast','Log difference','SE','$t$','$p$','$q_{24}$','RT ratio',r'95\% ratio CI'],rows,
            tablelabel='tab:rt'+policy,
            note='Ratios describe model-based geometric RTs or ratios of ratios for DiD terms. They are not arithmetic mean differences in milliseconds. Pointwise intervals are unadjusted. BH covers all 24 listed contrasts jointly within this policy.')
    # Display the >=100-ms fit with participant arithmetic means on the same
    # response-validity policy; the caption distinguishes the two estimands.
    sens=pd.read_csv(out/'min100_cell_estimates.csv')
    sens['run']=sens.cell.str.extract(r'r(\d+)')[0].astype(int)
    sens['condition']=sens.cell.str.extract(r'_(.+)')[0]
    participants=pd.read_csv(out.parent/'participant_run_rt_min100.csv')
    clean=pd.read_csv(out.parent/'cleaned_trials.csv')
    expected=(clean[clean.p_response_rt.ge(.1)]
              .groupby(['subject','session','rule_group']).p_response_rt.mean()*1000)
    observed=participants.set_index(['subject','session','rule_group']).rt_ms
    assert expected.index.equals(observed.sort_index().index)
    np.testing.assert_allclose(observed.sort_index(),expected,rtol=2e-9)
    assert len(participants)==400 and participants.subject.nunique()==20
    assert participants.groupby(['session','rule_group']).subject.nunique().eq(20).all()
    assert set(zip(participants.session,participants.rule_group))==set(zip(sens.run,sens.condition))

    colors={'Ran':'#616873','Str1':'#7546BD','Str2':'#078773'}
    labels={'Ran':'Random','Str1':'Structure 1','Str2':'Structure 2'}
    offsets={'Ran':-.22,'Str1':0.,'Str2':.22}
    # One fixed offset per participant keeps horizontal spread deterministic.
    jitter=dict(zip(sorted(participants.subject.unique()),np.linspace(-.065,.065,20)))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,
                         'axes.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,ax=plt.subplots(figsize=(10,6.2))
        fig.subplots_adjust(left=.10,right=.985,bottom=.17,top=.77)
        ax.axvspan(2.5,6.5,color='#F5F6F8',zorder=0)
        for boundary in [2.5,6.5]:
            ax.axvline(boundary,color='#D8DCE2',lw=.9,ls=(0,(3,4)),zorder=1)
        handles=[]
        for condition,color in colors.items():
            points=participants[participants.rule_group==condition]
            x=points.session+offsets[condition]+points.subject.map(jitter)
            ax.scatter(x,points.rt_ms,s=25,color=color,alpha=.23,
                       edgecolors='none',zorder=2)
            estimates=sens[sens.condition==condition].sort_values('run')
            x=estimates.run+offsets[condition]
            y=estimates.geometric_rt_ms
            handle=ax.errorbar(x,y,
                yerr=np.vstack([y-estimates.geometric_ci_low_ms,
                                estimates.geometric_ci_high_ms-y]),
                fmt='o-',color=color,ms=9,mec='white',mew=1.1,lw=1.8,
                elinewidth=1.45,capsize=3.5,capthick=1.25,
                label=labels[condition],zorder=4)
            handles.append(handle)
        ax.set(xlabel='Run',ylabel='Reaction time (ms)',xlim=(.5,8.5),ylim=(170,570))
        ax.set_xticks(range(1,9));ax.set_yticks(range(200,551,50))
        ax.grid(axis='y',color='#E5E8ED',lw=.7,zorder=0)
        ax.set_axisbelow(True)
        for side in ['top','right']:
            ax.spines[side].set_visible(False)
        for side in ['left','bottom']:
            ax.spines[side].set_color('#AAB0B9');ax.spines[side].set_linewidth(.8)
        ax.tick_params(axis='both',length=3,color='#AAB0B9',labelcolor='#424750')
        for center,label in [(1.5,'Initial sequence'),(4.5,'Interleaved sequences'),
                             (7.5,'New sequence')]:
            ax.text(center,1.025,label,transform=ax.get_xaxis_transform(),
                    ha='center',va='bottom',color='#676D77',fontsize=10.5)
        fig.suptitle('Reaction times across sequence replacement',
                     y=.975,fontsize=16,fontweight='bold',color='#262B33')
        fig.text(.5,.923,'20 participants  ·  Correct-response RT ≥ 100 ms',
                 ha='center',fontsize=11,color='#676D77')
        fig.legend(handles=handles,labels=list(labels.values()),loc='upper center',
                   bbox_to_anchor=(.5,.892),ncol=3,frameon=False,
                   handlelength=2.3,columnspacing=2.5)
        fig.text(.5,.04,'Large points: LME estimates (95% CI)     '
                 'Faint points: participant mean RTs',ha='center',fontsize=10,
                 color='#676D77')
        savefig(fig,figures/'behavioral_lme_estimates')
    metadata=dict(policy='min100',n_subjects=20,n_model_points=len(sens),
                  n_participant_points=len(participants),
                  model_summary='Back-transformed fixed effects at random effects zero; geometric RTs',
                  participant_summary='Arithmetic mean correct RT within participant/run/condition after trimming and RT >= 100 ms',
                  confidence_intervals='Pointwise 95%; denominator df capped at 19',
                  condition_offsets=offsets,participant_jitter='Fixed participant offsets from -0.065 to 0.065 runs',
                  model_input_sha256=digest(out/'min100_cell_estimates.csv'),
                  participant_input_sha256=digest(out.parent/'participant_run_rt_min100.csv'),
                  script_sha256=digest(Path(__file__)),
                  png_sha256=digest(figures/'behavioral_lme_estimates.png'),
                  pdf_sha256=digest(figures/'behavioral_lme_estimates.pdf'))
    (figures/'behavioral_lme_estimates_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    comparisons=pd.read_csv(out/'primary_contrasts.csv').merge(pd.read_csv(out/'min100_contrasts.csv'),
                    on='contrast',suffixes=('_primary','_min100'))
    comparisons['same_direction']=np.sign(comparisons.log_difference_primary)==np.sign(comparisons.log_difference_min100)
    comparisons['both_fdr']=comparisons.q_bh_24_primary.le(.05)&comparisons.q_bh_24_min100.le(.05)
    comparisons.to_csv(out/'policy_comparison.csv',index=False,float_format='%.12g')
    record=dict(status='passed',model='Trial-level log-RT LME; subject random intercept/condition slopes, run/block intercepts, AR1, condition-specific residual SD',
                policies=statuses,n_contrasts_each=24,n_same_direction=int(comparisons.same_direction.sum()),
                n_both_fdr=int(comparisons.both_fdr.sum()),
                independent_covariance_and_contrast_checks=True,
                fit_script_sha256=digest(Path(__file__).with_name('fit_fmri_rt_lme.R')),
                cleaned_input_sha256=digest(out.parent/'cleaned_trials.csv'),
                analysis_spec_sha256=digest(out/'analysis_spec.json'))
    (out/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    if (root/'results/behavioral/block_errors_20261013/group_run_error_rates.csv').is_file():
        from make_error_rate_figure import make_figures
        make_figures(root)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
