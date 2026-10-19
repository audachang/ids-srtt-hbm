#!/usr/bin/env python3
"""Updated design diagnostics, fitted AR-bin propagation, and component plots."""
from updated_common import *
import concurrent.futures
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import mean_scaling, _yule_walker
from nilearn.glm.regression import OLSModel
from polish_figures import label


def ar_hist(job):
    subject,run=job
    bold,mask,conf,_=paths(subject,run)
    events=pd.read_csv(event_path(subject,run),sep='\t');events.trial_type=events.trial_type.str.extract('(Ran|Str[12])$')[0]
    nuisance=pd.read_csv(conf,sep='\t')[CONFOUNDS].fillna(0).astype(float)
    design=make_first_level_design_matrix(np.arange(200)*2.,events,hrf_model='glover',drift_model='cosine',high_pass=1/128,
        add_regs=nuisance.to_numpy(),add_reg_names=list(nuisance))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');y=NiftiMasker(mask_img=str(mask),smoothing_fwhm=4.,standardize=False,detrend=False,t_r=2.).fit_transform(str(bold))
    y,_=mean_scaling(y,axis=0)
    coefficient=_yule_walker(OLSModel(design.to_numpy()).fit(y).residuals.T,1)[:,0]
    bins=(coefficient*100).astype(int)/100.
    values,counts=np.unique(bins,return_counts=True)
    return [dict(subject=subject,run=run,rho=value,n_voxels=count) for value,count in zip(values,counts)]


def main():
    t=Task('design_diagnostics')
    inventory=pd.read_csv(t.record(OLD_BASE/'T3_timing_repair/event_inventory.csv'))
    changed=set(zip(inventory[inventory.changed].subject,inventory[inventory.changed].run))
    refreshed={(s,r) for s in N22 for r in range(1,9) if s not in N20 or (s,r) in changed}
    controls={('sub-001',1),('sub-020',6)}
    old=pd.read_csv(t.record(OLD_BASE/'new_first_level/reference_ar1_histogram.csv'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        new=pd.DataFrame([row for group in pool.map(ar_hist,sorted(refreshed|controls)) for row in group])
    for subject,run in controls:
        a=new[(new.subject==subject)&(new.run==run)].groupby('rho').n_voxels.sum().sort_index()
        b=old[(old.subject==subject)&(old.run==run)].groupby('rho').n_voxels.sum().sort_index()
        pd.testing.assert_series_equal(a,b,check_dtype=False)
    old=old[[((s,r) not in refreshed|controls) for s,r in zip(old.subject,old.run)]]
    hist=pd.concat([old,new],ignore_index=True);csv(hist,t.path/'reference_ar1_histogram.csv')
    for subject,run in refreshed|controls:
        for path in [*paths(subject,run)[:3],event_path(subject,run)]:t.record(path)
    diagnostics=[]
    for model in MODELS:
        for subject in N22:
            for run in range(1,9):
                path=BASE/'first_level/runs'/subject/f'run-{run:02d}'/model/'design_matrix.tsv'
                if not path.exists():path=FIRST/MODELS[model]/'level1/eventrelated'/subject/f'run-{run:02d}'/'design_matrix.tsv'
                design=pd.read_csv(t.record(path),sep='\t',float_precision='round_trip');x=design.to_numpy();names=list(design);tasks=conditions(run)
                inverse=np.linalg.pinv(x);covariance=inverse@inverse.T
                nuisance=design[[c for c in names if c not in tasks]].to_numpy()
                adjusted=design[tasks].to_numpy()-nuisance@np.linalg.lstsq(nuisance,design[tasks],rcond=None)[0]
                row=dict(model=model,subject=subject,run=run,n_rows=len(x),n_columns=x.shape[1],rank=int(np.linalg.matrix_rank(x)))
                assert row['rank']==row['n_columns']
                for i,condition in enumerate(tasks):
                    y=adjusted[:,i];other=np.delete(adjusted,i,axis=1);residual=y-other@np.linalg.lstsq(other,y,rcond=None)[0]
                    row[condition+'_nuisance_adjusted_vif']=(y@y)/(residual@residual)
                if run in [3,4,5,6]:row['adjusted_Str1_Str2_correlation']=np.corrcoef(adjusted[:,tasks.index('Str1')],adjusted[:,tasks.index('Str2')])[0,1]
                contrasts={}
                for a,b in [('Str2','Str1'),('Str2','Ran'),('Str1','Ran')]:
                    if a in names and b in names:
                        c=np.zeros(len(names));c[names.index(a)]=1;c[names.index(b)]=-1
                        contrasts[a+'_minus_'+b]=c;row[a+'_minus_'+b+'_iid_unit_variance']=float(c@covariance@c)
                if model=='reference':
                    h=hist[(hist.subject==subject)&(hist.run==run)]
                    assert len(h)>0
                    ar_variance={key:0. for key in contrasts}
                    for _,binrow in h.iterrows():
                        whitened=x.copy();whitened[1:]-=binrow.rho*x[:-1]
                        inverse=np.linalg.pinv(whitened);covariance=inverse@inverse.T
                        for key,c in contrasts.items():ar_variance[key]+=binrow.n_voxels*(c@covariance@c)
                    for key,value in ar_variance.items():row[key+'_fullmask_AR_unit_variance']=value/h.n_voxels.sum()
                diagnostics.append(row)
    diagnostics=pd.DataFrame(diagnostics);csv(diagnostics,t.path/'design_diagnostics.csv')
    means=[]
    for cohort,subjects in [('N20',N20),('N22',N22)]:
        part=diagnostics[diagnostics.subject.isin(subjects)].groupby(['model','run']).median(numeric_only=True).reset_index()
        part['cohort']=cohort;means.append(part)
    summary=pd.concat(means);csv(summary,t.path/'median_diagnostics_by_run.csv')
    did=[]
    for (model,subject),d in diagnostics.groupby(['model','subject']):
        d=d.set_index('run')
        for i,stage in enumerate(STAGES[1:],1):
            for method in ['iid_unit_variance','fullmask_AR_unit_variance']:
                if model=='censored' and method.startswith('fullmask'):continue
                variance=sum(d.loc[r,'Str1_minus_Ran_'+method] for r in [1,2])+sum(d.loc[r,'Str2_minus_Ran_'+method] for r in [2*i+1,2*i+2])
                did.append(dict(model=model,subject=subject,stage=stage,method=method,unit_variance=variance,efficiency=1/variance))
    csv(did,t.path/'did_unit_variance.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,target in zip(axes,['Str2_minus_Str1','Str2_minus_Ran']):
        for method,color in [('iid_unit_variance','#6f8797'),('fullmask_AR_unit_variance','#27658f')]:
            d=summary[(summary.cohort=='N22')&(summary.model=='reference')]
            ax.plot(d.run,d[target+'_'+method],marker='o',label=method,color=color)
        ax.set(title=target,xlabel='Run',ylabel='Unit-noise contrast variance');ax.legend(fontsize=7)
    fig.tight_layout();savefig(fig,t.path/'contrast_variance_by_run')
    t.finish('# Matched-sample design diagnostics\n\nAll 352 reference/censored designs for N22 are checked for full rank, nuisance-adjusted task VIF, within-run sequence correlation and IID unit-noise contrast variance. Separate N20/N22 medians are exported. For reference designs, full-voxel fitted AR-bin histograms are regenerated for added/changed runs using the exact native OLS residual/Yule-Walker/binning rule; two unchanged controls must exactly match the archived full-fit histograms. Unchanged-run histograms are reused. Averaging whitened-design unit-innovation-noise contrast variances over the actual run-mask AR bins gives the reference AR diagnostic, not an ROI sampling variance or SE. DiD unit variances add independent-run contributions with the original sum weights. Censored diagnostics use their actual retained rows; full-mask AR-bin diagnostics are explicitly limited to reference fits.\n\nObserved event lists and recovered source generation rules are documented in source_discovery; the original RNG seed/state remains unavailable. These diagnostics do not isolate prevalence from time/practice/familiarity.\n',
             dict(passed=True,n_designs=len(diagnostics),n_regenerated_AR_runs=len(refreshed|controls),
                  n_exact_AR_control_runs=2,all_designs_full_rank=True))
    # Updated trajectories and structured/random components, preserving units.
    f=Task('figures')
    sr=pd.read_csv(f.record(BASE/'roi_reporting/stage_results.csv'))
    cs=pd.read_csv(f.record(BASE/'roi_reporting/phase_component_results.csv'))
    trajectories=pd.read_csv(f.record(BASE/'roi_reporting/trajectory_results.csv'))
    focus=['LeftSMA','RightAccumbensArea','LeftPutamen','LeftCaudate','LeftHippocampus','LeftCerebellum']
    fig,axes=plt.subplots(3,4,figsize=(14,10))
    for i,roi in enumerate(focus):
        left=axes[i//2,(i%2)*2];right=axes[i//2,(i%2)*2+1]
        for component,color in [('structured','#326b9d'),('random','#858585')]:
            d=cs[(cs.cohort=='N22')&(cs.model=='reference')&(cs.policy=='observed80')&(cs.roi==roi)&(cs.component==component)].set_index('stage').reindex(STAGES)
            left.errorbar(range(4),d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o-',ms=3,label=component,color=color)
        for cohort,color,offset in [('N20','#8d969d',-.07),('N22','#268378',.07)]:
            d=sr[(sr.cohort==cohort)&(sr.model=='reference')&(sr.policy=='observed80')&(sr.roi==roi)].set_index('stage').reindex(STAGES)
            right.errorbar(np.arange(4)+offset,d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=3,label=cohort,color=color)
        right.axvline(.5,color='.7',ls=':');left.set_ylabel('Two-run mean coefficient');right.set_ylabel('Initial / summed DiD')
        for ax in [left,right]:ax.axhline(0,color='.7',lw=.6);ax.set_xticks(range(4),STAGES,fontsize=8);ax.set_title(label(roi),fontsize=10)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7);fig.suptitle('Corrected timing: N22 condition components and N20/N22 contrasts; pointwise 95% CI');fig.tight_layout();savefig(fig,f.path/'condition_did_components')
    planned=[r for r in ROIS if any(term in r for term in ['SMA','PrimaryMotor','Precentral','Caudate','Putamen','Pallidum','Accumbens','Hippocampus','Cerebellum'])]
    with PdfPages(f.path/'sequence_trajectories.pdf') as pdf:
        for roi in planned:
            fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
            for ax,model in zip(axes,['reference','censored']):
                for cohort,style,alpha in [('N20','s--',.4),('N22','o-',1.)]:
                    for condition,color in [('Ran','#777777'),('Str1','#356ba3'),('Str2','#aa5279')]:
                        d=trajectories[(trajectories.cohort==cohort)&(trajectories.model==model)&(trajectories.policy=='observed80')&(trajectories.roi==roi)&(trajectories.condition==condition)].sort_values('run')
                        ax.errorbar(d.run,d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt=style,ms=3,color=color,alpha=alpha,label=cohort+' '+condition)
                ax.axhline(0,color='.7',lw=.6);ax.set(title=model,xlabel='Run',xticks=range(1,9))
            axes[0].set_ylabel('Coefficient vs implicit fitted baseline');axes[1].legend(fontsize=6,ncol=2);fig.suptitle(label(roi)+'; pointwise 95% CI');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    f.finish('# Updated component and trajectory figures\n\nThe component figure shows N22 structured/random coefficients as two-run means relative to the implicit fitted baseline, and N20/N22 initial/DiD contrasts in their historical two-run sum units. Initial structured means Str1; later structured means Str2. No global random baseline is used. The trajectory PDF includes the anatomical ROI set requested in the plan, both primary/sensitivity models and both samples. All intervals are pointwise, not simultaneous. Absent sequences are not imputed.\n',dict(passed=True,n_trajectory_ROIs=len(planned),figure_samples=['N20','N22']))


if __name__=='__main__':main()
