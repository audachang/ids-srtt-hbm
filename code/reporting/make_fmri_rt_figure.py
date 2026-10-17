#!/usr/bin/env python3
"""Plot the imaging cohort using the original Exp3 repetition convention."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from augment_roi_reporting import digest, summarize, csv, savefig


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--original-behavior-root',type=Path,
                   default=Path('/home/fmri2404/chiaoen/Chiaoen_AllExpData/Exp3/fmri_beh_V2/SRTT'))
    args=p.parse_args();root=args.source_root.resolve();dest=args.output_root.resolve()
    out=dest/'results/behavioral/fmri_rt_20261012';figdir=dest/'manuscript/figures'
    subjects=(root/'code/replication/config/subjects.txt').read_text().split()
    cycles=pd.read_csv(out/'participant_cycles.csv')
    runs=pd.read_csv(out/'participant_run_rt.csv')
    minimum100=pd.read_csv(out/'participant_run_rt_min100.csv')
    accuracy=pd.read_csv(out/'participant_accuracy.csv')
    clean=pd.read_csv(out/'cleaned_trials.csv')
    assert set(cycles.subject)==set(subjects)==set(runs.subject)==set(accuracy.subject)
    assert len(subjects)==20 and len(accuracy[['subject','session']].drop_duplicates())==160
    expected={1:{'Ran','Str1'},2:{'Ran','Str1'},3:{'Ran','Str1','Str2'},
              4:{'Ran','Str1','Str2'},5:{'Ran','Str1','Str2'},6:{'Ran','Str1','Str2'},
              7:{'Ran','Str2'},8:{'Ran','Str2'}}
    for (s,run),part in runs.groupby(['subject','session']):
        assert set(part.rule_group)==expected[run]
    group=[]
    for (condition,stage,cycle),part in cycles.groupby(['rule_group','stage','displayed_cycle']):
        assert not part.subject.duplicated().any()
        group.append(dict(condition=condition,stage=stage,displayed_cycle=cycle,
                          **summarize(part.rt_ms)))
    group=pd.DataFrame(group).drop(columns=['cohen_dz','t','df','p'])
    csv(group,out/'group_cycles.csv')
    summaries=[]
    for name,data,column in [('rt',runs,'rt_ms'),('rt_min100',minimum100,'rt_ms'),
                             ('accuracy',accuracy,'accuracy'),('omission',accuracy,'omission_rate')]:
        for (run,condition),part in data.groupby(['session','rule_group']):
            summaries.append(dict(measure=name,run=run,condition=condition,**summarize(part[column])))
    summary=pd.DataFrame(summaries).drop(columns=['cohen_dz','t','df','p'])
    csv(summary,out/'run_summary.csv')
    benefit=[]
    for s in subjects:
        for run,condition in [(2,'Str1'),(8,'Str2')]:
            d=clean[(clean.subject==s)&(clean.session==run)]
            r=d[(d.rule_group=='Ran')&d.block.str.endswith('Ran2')].p_response_rt
            a=d[d.rule_group==condition].p_response_rt
            assert len(a)>0 and len(r)>0
            benefit.append(dict(subject=s,run=run,condition=condition,
                                value=(r.mean()-a.mean())*1000))
    benefits=pd.DataFrame(benefit);csv(benefits,out/'participant_learning_benefits.csv')
    bsummary=[]
    for condition,part in benefits.groupby('condition'):
        bsummary.append(dict(condition=condition,**summarize(part.value)))
    bsummary=pd.DataFrame(bsummary);csv(bsummary,out/'learning_benefit_summary.csv')
    colors={('Str1','Early'):'#713ab2',('Str1','Late'):'#b74b91',
            ('Str2','Early'):'#198463',('Str2','Late'):'#31a0ae'}
    fig,ax=plt.subplots(figsize=(10,6))
    for key,c in colors.items():
        d=group[(group.condition==key[0])&(group.stage==key[1])].sort_values('displayed_cycle')
        x=d.displayed_cycle.to_numpy();y=d['mean'].to_numpy()
        ax.errorbar(x,y,yerr=d.se.to_numpy(),fmt='o',ms=3,mfc='white',mec=c,
                    ecolor=c,elinewidth=.65,alpha=.75,label=f'{key[0]}: {key[1].lower()}')
    ax.axvline(60.5,color='.6',ls=':',lw=1)
    ax.axvline(120.5,color='.6',ls=':',lw=1)
    ax.set(xlabel='Sequence repetition (Str2 offset by 60)',ylabel='Reaction time (ms)',
           title='IDS-SRTT reaction times in the fMRI cohort (20 participants)')
    ax.set_xticks([1,30,60,90,120,150,180]);ax.grid(alpha=.15)
    ax.legend(ncol=2,loc='upper right',frameon=False,fontsize=10)
    fig.tight_layout();savefig(fig,figdir/'behavioral_rt_cycles')
    fig,axes=plt.subplots(1,3,figsize=(13,4.1))
    cmap={'Ran':'#555555','Str1':'#713ab2','Str2':'#198463'}
    for ax,measure,title,scale in [(axes[0],'rt','Correct-response RT',1),
                                   (axes[1],'accuracy','Accuracy',100),
                                   (axes[2],'omission','Omissions',100)]:
        for condition,c in cmap.items():
            d=summary[(summary.measure==measure)&(summary.condition==condition)].sort_values('run')
            ax.errorbar(d.run,d['mean']*scale,
                yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']])*scale,
                fmt='o-',color=c,ms=4,lw=1,label=condition)
        ax.set_title(title);ax.set_xlabel('Run');ax.set_xticks(range(1,9));ax.grid(alpha=.15)
    axes[0].set_ylabel('RT (ms); pointwise 95% CI')
    axes[1].set_ylabel('Percent; pointwise 95% CI');axes[2].set_ylabel('Percent; pointwise 95% CI')
    axes[0].legend(frameon=False,fontsize=9)
    fig.tight_layout();savefig(fig,figdir/'behavioral_run_summary')
    # Plot the requested 20 individual trajectories, allowing inspection of spread.
    fig,axes=plt.subplots(5,4,figsize=(13,13),sharex=True,sharey=True)
    for ax,s in zip(axes.flat,subjects):
        for condition,c in cmap.items():
            d=runs[(runs.subject==s)&(runs.rule_group==condition)].sort_values('session')
            ax.plot(d.session,d.rt_ms,'o-',color=c,ms=3,lw=.8,label=condition)
        ax.set_title(s,fontsize=10);ax.set_xticks(range(1,9));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.supxlabel('Run');fig.supylabel('Correct-response RT (ms)')
    fig.tight_layout();savefig(fig,figdir/'behavioral_individual_rt')
    rawpaths=pd.read_csv(out/'raw_input_files.csv').path.tolist()
    originals=[args.original_behavior_root/'fmri_srtt_beh_new2_plot.R',
               args.original_behavior_root/'fmri_srtt_beh_new2_analysis.R']
    manifest={str(path):digest(path) for path in rawpaths+list(originals)}
    agreement=[]
    for path in rawpaths:
        logged=pd.read_csv(path)
        participant=int(logged.participant.iloc[0]);run=int(logged.session.iloc[0])
        candidates=list((args.original_behavior_root/'Import_data').glob(f'{participant}_{run}_*.csv'))
        if len(candidates)!=1:
            raise ValueError(f'Expected one original Exp3 file for {participant}/{run}: {len(candidates)}')
        raw=pd.read_csv(candidates[0]).dropna(subset=['keypress']).reset_index(drop=True)
        assert len(raw)==len(logged)==320
        r1=pd.to_numeric(raw['Stimuli_key_resp.rt'],errors='coerce')
        r2=pd.to_numeric(raw['Stimuli_key_resp_2.rt'],errors='coerce')
        response_rt=r1.combine_first(r2)
        k1=pd.to_numeric(raw['Stimuli_key_resp.keys'],errors='coerce')
        k2=pd.to_numeric(raw['Stimuli_key_resp_2.keys'],errors='coerce')
        response_key=k1.combine_first(k2)
        assert np.allclose(response_rt,logged.p_response_rt,equal_nan=True,atol=1e-10,rtol=1e-10)
        assert np.array_equal(raw.keypress.to_numpy(),logged.keypress.to_numpy())
        assert np.array_equal(raw.rule.to_numpy(),logged.rule.to_numpy())
        assert np.array_equal((response_key==raw.keypress).astype(int),logged['correct.resp'])
        agreement.append(dict(subject=f'sub-{participant:03d}',run=run,
            original=str(candidates[0]),logged=path,n_trials=320,
            max_rt_error_s=float(np.nanmax(np.abs(response_rt-logged.p_response_rt))),
            correctness_matches=True,condition_order_matches=True))
        manifest[str(candidates[0])]=digest(candidates[0])
    csv(pd.DataFrame(agreement),out/'original_raw_agreement.csv')
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    m=dict(n_subjects=20,subjects=subjects,n_runs=160,n_logged_trials=51200,
           original_plot=str(originals[0]),original_analysis=str(originals[1]),
           cycle_convention='Assign eight-trial repetitions before cleaning; Str1=1..120; Str2=61..180.',
           figure_error_bars='SEM across participant cycle means; contributing N exported for every point',
           min_cycle_n=int(group.n.min()),max_cycle_n=int(group.n.max()),
           preprocessing='trimr 1.1.1 modifiedRecursive reconstruction on correct finite 0<RT<1-second trials within participant/block. Original custom My_modifiedRecursive helper not located.',
           fits='Cycle figure shows descriptive observations only. RT inference uses the separately specified trial-level linear mixed-effects model.',
           min100_sensitivity='Descriptive exclusion of RT<100 ms after the same cleaning; separate run summaries.',
           sample_match_verified=True,original_raw_runs_verified=160,
           source_root=str(root),script_sha256=digest(__file__))
    (out/'summary.json').write_text(json.dumps(m,indent=2)+'\n')
    (figdir/'behavioral_rt_cycles_metadata.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m,indent=2));print(bsummary.to_string(index=False))


if __name__=='__main__': main()
