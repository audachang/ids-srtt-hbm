#!/usr/bin/env python3
"""Combine the retained T7 specifications with the shared RT-PM extension."""
from common import *
from summarize_sensitivity import stage_values
from run_saved import direct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
 t=Task('T7_imaging_shared_rt');prior=BASE/'T7_imaging';shared=BASE/'shared_rt_first_level'
 manifest=json.loads((prior/'input_manifest.json').read_text());t.inputs.update(manifest['inputs'])
 snapshot=t.path/'original_T7_input_manifest.json';snapshot.write_bytes((prior/'input_manifest.json').read_bytes());t.record(snapshot);old=pd.read_csv(t.record(prior/'participant_roi_values.csv'))
 cond=pd.read_csv(t.record(shared/'condition_values.csv'));assert len(cond)==20*20*48
 for s in SUBJECTS:
  for run in range(1,9):
   folder=shared/'runs'/s/f'run-{run:02d}';q=json.loads(t.record(folder/'qc.json').read_text());t.inputs.update(q['inputs']);t.record(folder/'design_matrix.tsv')
 v=pd.concat([old,stage_values(cond)],ignore_index=True);assert v.model.nunique()==4
 csv(v,t.path/'participant_roi_values.csv');res=infer(v,['model','policy','stage','roi'],['model','policy','stage']);csv(res,t.path/'sensitivity_roi_results.csv')
 ref=res[res.model=='reference'].drop(columns='model');comp=res[res.model!='reference'].merge(ref,on=['policy','stage','roi'],suffixes=('_alternative','_reference'));comp['estimate_shift']=comp.mean_alternative-comp.mean_reference;csv(comp,t.path/'estimate_shifts.csv')
 base=v[v.model=='reference'];paired=[]
 for model,g in v[v.model!='reference'].groupby('model'):
  d=g.merge(base,on=['subject','stage','policy','roi'],suffixes=('_alt','_ref'));d['value']=d.value_alt-d.value_ref
  paired.append(infer(d,['model_alt','stage','roi'],['model_alt','stage']))
 csv(pd.concat(paired),t.path/'paired_estimate_shift_intervals.csv')
 qc=pd.read_csv(t.record(shared/'design_qc.csv'));csv(qc,t.path/'shared_rt_design_qc.csv')
 dd=direct(v);csv(dd,t.path/'direct_stage_comparisons.csv');acc=[]
 for model,g in v.groupby('model'):
  pivot=g.pivot(index=['subject','roi'],columns='stage',values='value');a=(pivot.early-pivot.late).xs('RightAccumbensArea',level='roi')
  for omitted in ['none']+SUBJECTS:
   d=dd[dd.model==model] if omitted=='none' else direct(g[g.subject!=omitted]);r=d[(d.roi=='RightAccumbensArea')&(d.pair=='early_minus_late')].iloc[0]
   acc.append(dict(model=model,omitted_subject=omitted,q144=r.q_three_pairs_144,**summarize(a if omitted=='none' else a.drop(omitted))))
 csv(acc,t.path/'right_accumbens_early_minus_late.csv')
 fig,axes=plt.subplots(1,4,figsize=(14,4),sharey=True);models=['reference','errors_omissions','rt_modulator','rt_shared_modulator']
 for ax,stage in zip(axes,STAGES):
  d=res[(res.roi=='LeftSMA')&(res.stage==stage)].set_index('model').reindex(models)
  ax.errorbar(d['mean'],range(4),xerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=4,color='#4866a4');ax.axvline(0,c='.7',lw=.7);ax.set_yticks(range(4),models);ax.set_ylim(3.5,-.5);ax.set_title(stage)
 fig.suptitle('Left BA6: error and two RT specifications, pointwise 95% CI');fig.tight_layout();savefig(fig,t.path/'ba6_error_rt_specifications')
 report='''# T7 imaging extension — shared run-centered RT modulator

The earlier within-condition RT-PM is retained in the same comparison. That PM captures **within-cell** RT variation and does not remove condition-average RT/duration differences. Its stable BA6 estimates cannot by themselves exclude the exposure-duration concern.

The additional model uses one shared Glover-convolved impulse PM across all valid responded trials (correct and incorrect, 0<RT<1), centered over the whole run. Between-condition RT mean differences remain in this PM. Original condition boxcars, masks, 4-mm smoothing, eight basic nuisance columns, 128-s high-pass and full-voxel AR(1) fitting are fixed. The model was added after examining the centering scope, before examining its results, and is an additional exploratory specification. Separate 48-slot BH families per model/stage are retained; no correction across model searches is claimed.

This is an RT-associated BOLD adjustment, not measured visual exposure. Exact cue offsets and response assignment remain unverified. RT is affected by learning, so adjusting for it changes the question and can remove learning-related variance. All four conditions/specifications and paired estimate-shift CIs are exported. `right_accumbens_early_minus_late.csv` includes full and all 20 LOO direct tests, each with the 144-slot correction.

Reference plus three sensitivities are included: errors/omissions, within-condition RT-PM, and shared run-centered RT-PM. Original T7 outputs are preserved in `../T7_imaging`.
'''
 t.finish(report,dict(passed=True,n_subjects=20,n_shared_rt_run_fits=160,n_total_sensitivity_models=3,n_roi_stage_results=len(res),all_shared_designs_full_rank=bool((qc['rank']==qc.n_columns).all()),max_shared_nuisance_adjusted_task_vif=float(qc.nuisance_adjusted_task_vif.max()),n_accumbens_full_loo_rows=len(acc)))
