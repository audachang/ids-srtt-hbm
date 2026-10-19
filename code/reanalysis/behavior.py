#!/usr/bin/env python3
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def prepare():
 t=Task('T7_anticipation',True);frames=[];timing=[];eligibility=[]
 for i in [int(s[-3:]) for s in SUBJECTS]+[5,23]:
  for run in range(1,9):
   p=t.record(ROOT/'data/behavioral'/f'{i:03d}_run_{run}_srtt_prob_new.csv');d=pd.read_csv(p)
   assert len(d)==320 and (d.participant==i).all() and (d.session==run).all()
   d['subject']=f'sub-{i:03d}';d['trial_in_run']=np.arange(1,321);d['position8']=d.groupby('block',sort=False).cumcount()%8+1
   d['omitted']=~np.isfinite(d.p_response_rt)|d.p_response_keys.astype(str).isin(['None','nan',''])
   d['responded']=~d.omitted;d['correct']=d['correct.resp'].eq(1)&d.responded
   d['anticipatory']=d.p_response_rt.gt(0)&d.p_response_rt.lt(.1)&d.responded
   d['cohort20']=d.subject.isin(SUBJECTS)
   # Compare target response coding, without asserting unavailable keyboard timing.
   numeric=pd.to_numeric(d.p_response_keys,errors='coerce');coded=(numeric==d.keypress)&d.responded
   correct_mismatch=int((coded!=d.correct).sum())
   nominal=d.stim_onset.diff().to_numpy();clock=d.stim_started.diff().to_numpy();within=d.block.eq(d.block.shift())
   timing.append(dict(subject=d.subject.iloc[0],run=run,n_nonpositive_rt=int((d.p_response_rt<=0).sum()),n_rt_ge1=int((d.p_response_rt>=1).sum()),n_correct_keycode_disagreements=correct_mismatch,n_nominal_within_block_intervals=int(within.sum()),nominal_min_step=float(nominal[within].min()),nominal_max_step=float(nominal[within].max()),actual_median_step=float(np.nanmedian(clock[within])),actual_min_step=float(np.nanmin(clock[within])),actual_max_step=float(np.nanmax(clock[within])),max_abs_actual_minus_nominal_step=float(np.nanmax(abs(clock[within]-nominal[within])))))
   frames.append(d)
 raw=pd.concat(frames,ignore_index=True);csv(timing,t.path/'timing_checks.csv')
 csv(raw,t.path/'raw_trials_22.csv')
 block=[]
 for (s,run,b,c),g in raw[raw.cohort20].groupby(['subject','session','block','rule_group']):
  block.append(dict(subject=s,session=run,block=b,rule_group=c,n_trials=len(g),n_anticipatory=int(g.anticipatory.sum()),n_correct_anticipatory=int((g.anticipatory&g.correct).sum()),n_errors=int((~g.correct).sum()),n_omissions=int(g.omitted.sum())))
 csv(block,t.path/'participant_block_anticipation.csv')
 desc=[]
 for cohort,subjects in [('N20',SUBJECTS),('N22',SUBJECTS+['sub-005','sub-023'])]:
  d=raw[raw.subject.isin(subjects)]
  for (run,c,pos),g in d.groupby(['session','rule_group','position8']):
   agg=g.groupby('subject').agg(n=('anticipatory','size'),n_ant=('anticipatory','sum'))
   desc.append(dict(cohort=cohort,run=run,condition=c,position8=pos,n_trials=len(g),n_anticipatory=int(g.anticipatory.sum()),n_correct_anticipatory=int((g.correct&g.anticipatory).sum()),anticipatory_accuracy=float(g[g.anticipatory].correct.mean()),**summarize(agg.n_ant/agg.n)))
 desc=pd.DataFrame(desc);csv(desc,t.path/'anticipation_by_run_condition_position.csv')
 cell=[]
 for (s,run,c),g in raw.groupby(['subject','session','rule_group']):cell.append(dict(subject=s,run=run,condition=c,cohort20=s in SUBJECTS,n_trials=len(g),n_anticipatory=int(g.anticipatory.sum()),n_correct_anticipatory=int((g.correct&g.anticipatory).sum()),rate=g.anticipatory.mean(),n_omissions=int(g.omitted.sum())))
 cell=pd.DataFrame(cell);csv(cell,t.path/'participant_run_anticipation.csv')
 # Learner criterion documented but original custom helper and t-test options absent.
 # Export transparent variants, retaining continuous index and trial-level CI caveat.
 learning=[]
 cleaned=pd.read_csv(t.record(ROOT/'results/behavioral/fmri_rt_20261012/cleaned_trials.csv'))
 for s,g in raw[raw.session==2].groupby('subject'):
  for policy in ['correct_valid_all','correct_valid_min100','saved_trimr_cleaned']:
   d=g[g.correct&g.p_response_rt.gt(0)&g.p_response_rt.lt(1)]
   if policy=='correct_valid_min100':d=d[d.p_response_rt>=.1]
   if policy=='saved_trimr_cleaned':
    if s not in SUBJECTS:continue
    d=cleaned[(cleaned.subject==s)&(cleaned.session==2)]
   a=d[d.block=='Ses2Str_6'].p_response_rt.to_numpy()*1000;b=d[d.block=='Ses2Ran2'].p_response_rt.to_numpy()*1000
   assert len(a)>1 and len(b)>1
   for equalvar in [False,True]:
    res=ttest_ind(a,b,equal_var=equalvar);va=a.var(ddof=1);vb=b.var(ddof=1)
    if equalvar:df=len(a)+len(b)-2;se=np.sqrt(((len(a)-1)*va+(len(b)-1)*vb)/df*(1/len(a)+1/len(b)))
    else:
     se=np.sqrt(va/len(a)+vb/len(b));df=(va/len(a)+vb/len(b))**2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    diff=b.mean()-a.mean();half=stats.t.ppf(.975,df)*se
    learning.append(dict(subject=s,cohort20=s in SUBJECTS,cleaning=policy,test='pooled_variance' if equalvar else 'Welch',n_str=len(a),n_random=len(b),benefit_random_minus_str_ms=diff,ci_low=diff-half,ci_high=diff+half,df=df,t_str_minus_random=res.statistic,p_two_sided=res.pvalue,p_one_sided_faster=res.pvalue/2 if res.statistic<0 else 1-res.pvalue/2,criterion_two_sided_faster=bool(res.pvalue<.05 and res.statistic<0)))
 csv(learning,t.path/'participant_run2_learning.csv')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for c,color in [('Ran','gray'),('Str1','#4866a4'),('Str2','#a44d79')]:
  d=cell[(cell.cohort20)&(cell.condition==c)];a=[]
  for run,g in d.groupby('run'):a.append(dict(run=run,**summarize(g.rate)))
  a=pd.DataFrame(a);axes[0].errorbar(a.run,100*a['mean'],yerr=100*np.vstack([a['mean']-a.ci_low,a.ci_high-a['mean']]),fmt='o-',label=c,color=color)
  a=desc[(desc.cohort=='N20')&(desc.condition==c)].groupby('position8').agg(n=('n_trials','sum'),anticipatory=('n_anticipatory','sum'))
  axes[1].plot(a.index,100*a.anticipatory/a.n,marker='o',label=c,color=color)
 axes[0].set_xlabel('Run');axes[0].set_ylabel('Anticipatory responses (% all trials; pointwise 95% CI)');axes[0].legend();axes[1].set_xlabel('Position in logged 8-trial window');axes[1].set_ylabel('Pooled anticipatory responses (%)');fig.tight_layout();savefig(fig,t.path/'anticipatory_responses')
 n20=raw[raw.cohort20];counts=dict(n_trials=len(n20),n_anticipatory=int(n20.anticipatory.sum()),n_correct_anticipatory=int((n20.correct&n20.anticipatory).sum()),anticipatory_accuracy=float(n20[n20.anticipatory].correct.mean()),n_correct_anticipatory_after_saved_cleaning=int((cleaned.p_response_rt<.1).sum()))
 writej(t.path/'counts.json',counts)
 t.finish('# T7a — anticipation and learning diagnostics\n\nAnticipation is any logged response with 0 < RT < 100 ms, including incorrect presses. Rates use all logged trials; accuracy is reported among anticipatory responses. Position is assigned from the original order, before filtering, within each 8-trial sequence window (random blocks are segmented into comparable windows).\n\nCounts: `'+json.dumps(counts)+'`. The anticipated 1,833 count is checked against the saved cleaned trials separately; raw logged counts are not silently substituted for it.\n\n`participant_run2_learning.csv` gives random-minus-final-structured benefit, pointwise Welch and pooled-variance CIs, one-/two-sided p values for transparent cleaning variants, and the existing saved trimr cleaning for N=20. These within-person trial-level CIs assume independent trials and may be too narrow under serial dependence. They are descriptive continuous learning indices, not a certified reconstruction of the original exclusion test. The thesis states final structured block followed by random in run 2 and p<.05, but original custom cleaning/test options and the exclusion log were unavailable.\n\n`timing_checks.csv` checks nominal/actual onset increments, RT ranges and target-key coding. The underlying keyboard-event stream, buffer-clearing code and direct response timestamps are absent. Response assignment and cue offset timing therefore remain **unverified**. The logs alone cannot establish that every <100-ms response is correctly attributed. Anticipation GLMM results are added by fit_anticipation.R with the same fixed/random hierarchy as the historical error model. Its completion is separately validated.',dict(passed=True,n20_counts=counts,n22_trials=len(raw),raw_rows_validated=True,keyboard_assignment_verified=False,glmm_status='separate fit pending'),status='descriptives_complete_glmm_pending')
if __name__=='__main__':prepare()
