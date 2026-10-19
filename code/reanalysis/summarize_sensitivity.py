#!/usr/bin/env python3
"""T6 and T7b/c summaries with signed estimates, uncertainty and influence."""
from common import *
from run_saved import direct
from polish_figures import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def stage_values(cond):
 p=cond.pivot(index=['model','subject','roi'],columns=['run','condition'],values='value')
 initial=p[1,'Str1']+p[2,'Str1']-p[1,'Ran']-p[2,'Ran'];frames=[]
 for i,stage in enumerate(STAGES):
  v=initial if i==0 else p[2*i+1,'Str2']+p[2*i+2,'Str2']-p[2*i+1,'Ran']-p[2*i+2,'Ran']-initial
  d=v.rename('value').reset_index();d['stage']=stage;d['policy']='observed80';frames.append(d)
 return pd.concat(frames,ignore_index=True)

def summary():
 t=Task('T6_robustness');newroot=BASE/'new_first_level';path=t.record(newroot/'condition_values.csv');cond=pd.read_csv(path)
 assert len(cond)==20*20*48*4
 for s in SUBJECTS:
  for r in range(1,9):
   q=json.loads(t.record(newroot/'runs'/s/f'run-{r:02d}'/'qc.json').read_text());t.inputs.update(q['inputs'])
   for design_path in (newroot/'runs'/s/f'run-{r:02d}').glob('*_design.tsv'):t.record(design_path)
   t.record(newroot/'runs'/s/f'run-{r:02d}'/'condition_values.csv')
 t.record(newroot/'design_qc.csv');t.record(newroot/'execution.json')
 fit_task=Task('new_first_level');fit_task.inputs.update(t.inputs)
 fit_task.finish('# Full-voxel fit execution\n\n160 retained runs, four AR(1) fits each: reference reproduction, SPM+temporal derivative, Glover+error/omission events, Glover+RT modulation. All new run jobs completed. Original input hashes, exact designs and per-voxel reference reproduction checks are recorded under runs/. New fits are summarized as ROI means, not saved as full new coefficient volumes.',dict(passed=True,n_run_jobs=160,n_full_voxel_fits=640,nilearn_version=nilearn.__version__,all_jobs_complete=True))
 values=stage_values(cond)
 ref=pd.read_csv(t.record(BASE/'T1_reference/participant_roi_values.csv')).query('model=="reference" and policy=="observed80"')
 ch=values[values.model=='reference_check'].merge(ref,on=['subject','roi','policy','stage'],suffixes=('_new','_saved'),validate='one_to_one')
 assert len(ch)==3840;np.testing.assert_allclose(ch.value_new,ch.value_saved,atol=1e-7,rtol=1e-6,equal_nan=True)
 # Reuse existing alternative full-voxel fits on the identical observed support.
 old=pd.concat([pd.read_csv(t.record(BASE/'T1_reference/participant_roi_values.csv')).query('policy=="observed80"'),pd.read_csv(t.record(t.path/'saved_alternatives_audit/participant_roi_values.csv'))],ignore_index=True)
 t.record(t.path/'saved_alternatives_audit/input_manifest.json');t.record(t.path/'saved_alternatives_audit/validation.json')
 mapping={'nocensor_hp128_primary':'reference','censor_hp128':'censored'}
 old['model']=old.model.replace(mapping)
 excluded=old[old.model.str.contains('bandpass')];csv(excluded,t.path/'flagged_bandpass_values.csv')
 old=old[~old.model.str.contains('bandpass')]
 extra=values[values.model=='spm_derivative'];allvalues=pd.concat([old,extra],ignore_index=True)
 models=list(allvalues.model.unique());assert len(models)==8
 results=infer(allvalues,['model','policy','stage','roi'],['model','policy','stage']);csv(results,t.path/'specification_results.csv');csv(allvalues,t.path/'participant_roi_values.csv')
 dd=direct(allvalues);csv(dd,t.path/'direct_stage_results.csv')
 # Every significant reference ROI-stage plus right accumbens early-late in all models.
 main=results[(results.model=='reference')&(results.q<=.05)];keys=set(zip(main.stage,main.roi))
 influence=[]
 for (model,stage,roi),g in allvalues.groupby(['model','stage','roi']):
  if (stage,roi) not in keys:continue
  g=g[np.isfinite(g.value)];full=summarize(g.value);n=len(g)
  for _,row in g.iterrows():
   loo=summarize(g[g.subject!=row.subject].value)
   influence.append(dict(model=model,stage=stage,roi=roi,omitted_subject=row.subject,full_mean=full['mean'],full_ci_low=full['ci_low'],full_ci_high=full['ci_high'],loo_mean=loo['mean'],loo_ci_low=loo['ci_low'],loo_ci_high=loo['ci_high'],loo_p=loo['p'],mean_shift=loo['mean']-full['mean'],shift_in_full_se=(loo['mean']-full['mean'])/full['se'],cook_intercept=n*(loo['mean']-full['mean'])**2/full['sd']**2,sign_changed=bool(np.sign(loo['mean'])!=np.sign(full['mean']))))
 influence=pd.DataFrame(influence);csv(influence,t.path/'leave_one_out_influence.csv')
 acc=[]
 for model,g in allvalues.groupby('model'):
  p=g.pivot(index=['subject','roi'],columns='stage',values='value');d=p.early-p.late
  a=d.xs('RightAccumbensArea',level='roi');s=summarize(a)
  fullq=dd[(dd.model==model)&(dd.policy=='observed80')&(dd.roi=='RightAccumbensArea')&(dd.pair=='early_minus_late')].iloc[0].q_three_pairs_144
  acc.append(dict(model=model,omitted_subject='none',q144=fullq,**s))
  for subject in SUBJECTS:
   subset=g[g.subject!=subject];ld=direct(subset);q=ld[(ld.roi=='RightAccumbensArea')&(ld.pair=='early_minus_late')].iloc[0].q_three_pairs_144
   ss=summarize(a.drop(subject));acc.append(dict(model=model,omitted_subject=subject,q144=q,cook_intercept=len(a.dropna())*(ss['mean']-s['mean'])**2/s['sd']**2,**ss))
 acc=pd.DataFrame(acc);csv(acc,t.path/'right_accumbens_early_minus_late.csv')
 # Retain accumbens diagnostics for band-pass alternatives without counting them.
 bandrows=[]
 for model,g in excluded.groupby('model'):
  for subject in ['none']+SUBJECTS:
   h=g if subject=='none' else g[g.subject!=subject]
   d=direct(h);row=d[(d.roi=='RightAccumbensArea')&(d.pair=='early_minus_late')].iloc[0]
   bandrows.append(dict(model=model,omitted_subject=subject,diagnostically_flagged=True,mean=row['mean'],ci_low=row.ci_low,ci_high=row.ci_high,q144=row.q_three_pairs_144))
 csv(bandrows,t.path/'right_accumbens_flagged_bandpass.csv')
 # Full all-ROI specification plots, separate row per stage and model.
 colors=['#4866a4','#a44d79','#16846d','#c6852d']
 with PdfPages(t.path/'specification_plot.pdf') as pdf:
  for roi in ROIS:
   fig,axes=plt.subplots(1,4,figsize=(15,3.8),sharey=True)
   for ax,stage,color in zip(axes,STAGES,colors):
    d=results[(results.roi==roi)&(results.stage==stage)].set_index('model').reindex(models)
    ax.errorbar(d['mean'],range(len(models)),xerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=3,color=color)
    ax.axvline(0,c='.65',lw=.7);ax.set_title(stage);ax.set_yticks(range(len(models)),models,fontsize=7);ax.set_ylim(len(models)-.5,-.5);ax.grid(axis='x',alpha=.2)
   fig.suptitle(label(roi)+' — signed summed estimates, pointwise 95% CI');fig.tight_layout();pdf.savefig(fig)
   if roi=='RightAccumbensArea':savefig(fig,t.path/'right_accumbens_specification')
   else:plt.close(fig)
 # Targeted accumbens direct-phase influence visual with full and LOO ranges.
 fig,ax=plt.subplots(figsize=(9,5))
 for i,model in enumerate(models):
  d=acc[acc.model==model];full=d[d.omitted_subject=='none'].iloc[0];loo=d[d.omitted_subject!='none']
  ax.hlines(i,loo['mean'].min(),loo['mean'].max(),color='#a44d79',lw=4);ax.errorbar(full['mean'],i,xerr=[[full['mean']-full.ci_low],[full.ci_high-full['mean']]],fmt='o',color='#4866a4',ms=4)
 ax.axvline(0,c='.65',lw=.7);ax.set_yticks(range(len(models)),models);ax.invert_yaxis();ax.set_xlabel('Right accumbens: early − late (pointwise 95% CI; colored bar = LOO means)');fig.tight_layout();savefig(fig,t.path/'accumbens_robustness')
 reference_acc=acc[(acc.model=='reference')&(acc.omitted_subject=='none')].iloc[0]
 acc_note=f'\nReference right accumbens early−late: {reference_acc["mean"]:.4f}, pointwise 95% CI [{reference_acc.ci_low:.4f}, {reference_acc.ci_high:.4f}], q144={reference_acc.q144:.4g}. Maximum reference Cook distance across its significant ROI-stage slots: {influence[influence.model=="reference"].cook_intercept.max():.4f}. Flagged band-pass accumbens results and LOO diagnostics are exported separately.\n'
 t.finish('# T6 — specification and influence analysis\n\nEight model specifications are evaluated on the same participant-specific observed80 support: reference, censoring, Glover+derivative, SPM, SPM+derivative (new full-voxel fit), motion12+tissue, motion12+aCompCor (up to eight retained combined-mask components), and precision weighting. Existing alternatives are reused with their recorded history; the aCompCor specification also adds motion derivatives. Precision weighting changes the within-phase estimand and is not an equal-run comparison.\n\n`specification_plot.pdf` gives all 48 ROIs × four stages as signed estimates and pointwise 95% CIs. The two band-pass fits are saved only as flagged diagnostics; their previous residual/filter problems mean they are not tallied as evidence of robustness.\n\n`leave_one_out_influence.csv` covers every ROI-stage significant under the new reference model, across all eight alternatives. Cook distance uses the intercept-only one-sample formula n×(mean_LOO−mean_full)²/s². It diagnoses influence; it is not a pretext to remove participants.\n\n`right_accumbens_early_minus_late.csv` includes full and all 20 LOO results for each model; BH is recomputed over the complete 144 direct-test family in each LOO sample. The full estimates, intervals and correction matter more than a pass/fail count.\n\nThe independently refitted reference is compared against the original extracted participant values before any sensitivity inference.'+acc_note,dict(passed=True,n_models=8,n_roi_stage_results=len(results),n_influence_rows=len(influence),n_reference_significant_slots=len(keys),n_accumbens_rows=len(acc),reference_value_max_abs_difference=float(np.nanmax(abs(ch.value_new-ch.value_saved))),n_completed_new_run_jobs=160))
 # T7 error and RT sensitivity keeps primary event duration and intended conditions.
 t7=Task('T7_imaging');t7.inputs.update(t.inputs)
 v=values[values.model.isin(['errors_omissions','rt_modulator'])];v=pd.concat([ref,v],ignore_index=True)
 res=infer(v,['model','policy','stage','roi'],['model','policy','stage']);csv(res,t7.path/'sensitivity_roi_results.csv');csv(v,t7.path/'participant_roi_values.csv')
 base=res[res.model=='reference'].drop(columns='model')
 comp=res[res.model!='reference'].merge(base,on=['policy','stage','roi'],suffixes=('_alternative','_reference'));comp['estimate_shift']=comp.mean_alternative-comp.mean_reference;csv(comp,t7.path/'estimate_shifts.csv')
 shift=[]
 for model,g in v[v.model!='reference'].groupby('model'):
  d=g.merge(ref,on=['subject','stage','policy','roi'],suffixes=('_alt','_ref'));d['value']=d.value_alt-d.value_ref
  a=infer(d,['model_alt','stage','roi'],['model_alt','stage']);shift.append(a)
 csv(pd.concat(shift),t7.path/'paired_estimate_shift_intervals.csv')
 qc=pd.read_csv(newroot/'design_qc.csv');csv(qc,t7.path/'run_sparsity_and_collinearity.csv')
 shift_note='\nLargest absolute group estimate shifts (summed units):\n'
 for model,g in comp.groupby('model'):
  row=g.loc[g.estimate_shift.abs().idxmax()];shift_note+=f'- {model}: {row.roi}, {row.stage}, shift={row.estimate_shift:.4f}; alternative {row.mean_alternative:.4f} [{row.ci_low_alternative:.4f}, {row.ci_high_alternative:.4f}], reference {row.mean_reference:.4f} [{row.ci_low_reference:.4f}, {row.ci_high_reference:.4f}].\n'
 t7.finish('# T7b/c — error, omission and RT sensitivities\n\nAll 160 runs were fit at full voxel resolution with Glover HRF, AR(1), 4-mm smoothing, six motion + CSF + WM, 128-s cosine high-pass and no censoring. Task boxcars retain the original 8-/40-s duration. Intended-condition events include all trials as in the reference.\n\n**Errors/omissions:** additional impulse regressors place incorrect responded trials at logged stimulus onset + RT (response time); omissions occur at stimulus onset + 1 s (response deadline). An empty event category is omitted, not replaced by an all-zero column. Sparsity and nuisance-adjusted task VIF/rank are exported for every run. These timings are modeling choices based on the 1-s trial logs; direct keyboard assignment remains unverified.\n\n**RT adjustment:** add separate condition-specific impulse regressors at responded trial onsets, modulated by within-run/condition mean-centered RT in seconds (valid 0<RT<1; correct and incorrect responses). Original task columns remain unchanged. This RT-parametric sensitivity is not a direct measurement of cue duration, since exact cue-offset events are unavailable. RT is affected by learning, so adjusting for it changes the scientific question.\n\nGroup estimates, pointwise CIs, BH q, and shifts from the reference are in `estimate_shifts.csv`; `paired_estimate_shift_intervals.csv` additionally gives CIs for within-participant shifts. No new sensitivity whole-brain claims are made. New voxel fits are reduced to ROI coefficients; the actual designs, full-voxel AR bins and reproduction QC are retained under `../new_first_level`.'+shift_note,dict(passed=True,n_sensitivity_models=2,n_new_sensitivity_run_fits=320,n_reference_reproduction_run_fits=160,n_roi_stage_results=len(res),all_designs_full_rank=bool((qc['rank']==qc.n_columns).all()),n_sparse_zero_error_runs=int(qc[(qc.model=='errors_omissions')&(qc.condition=='Ran')].n_wrong.eq(0).sum()),n_sparse_zero_omission_runs=int(qc[(qc.model=='errors_omissions')&(qc.condition=='Ran')].n_omissions.eq(0).sum()),max_nuisance_adjusted_task_vif=float(qc.nuisance_adjusted_task_vif.max())))
if __name__=='__main__':summary()
