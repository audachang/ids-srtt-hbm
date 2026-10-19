#!/usr/bin/env python3
"""Summarize matched behavioral fits without writing manuscript files."""
from updated_common import *
from scipy import special
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def marginal(beta, v, conditions, nodes=80):
    beta=np.atleast_2d(beta);v=np.atleast_2d(v)
    variance=v[:,0,None]+v[:,1,None]*(conditions=='Str1')+v[:,2,None]*(conditions=='Str2')
    x,w=special.roots_hermitenorm(nodes)
    return (special.expit(beta[:,:,None]+np.sqrt(variance[:,:,None])*x)*w/np.sqrt(2*np.pi)).sum(axis=2)


def rt():
    allrows=[]
    for cohort,n in [('N20',20),('N22',22)]:
        t=Task('rt_lme_'+cohort,behavior=True)
        t.record(Path(__file__).with_name('updated_rt_lme.R'))
        t.record(BEHAV/'source_replay'/cohort/'cleaned_trials.csv')
        t.record(BEHAV/'source_replay/validation.json')
        statuses={}
        for policy in ['primary','min100']:
            status=json.loads((t.path/f'{policy}_status.json').read_text())
            assert status['status']=='completed' and status['n_subjects']==n and status['design_rank']==20
            statuses[policy]=status
            coef=pd.read_csv(t.path/f'{policy}_fixed_coefficients.csv').set_index('term')
            cov=pd.read_csv(t.path/f'{policy}_fixed_covariance.csv',index_col=0).loc[coef.index,coef.index].to_numpy()
            w=pd.read_csv(t.path/f'{policy}_contrast_weights.csv').set_index('contrast').loc[:,coef.index]
            r=pd.read_csv(t.path/f'{policy}_contrasts.csv')
            np.testing.assert_allclose(r.log_difference,w.to_numpy()@coef.Value,atol=1e-11)
            np.testing.assert_allclose(r.se,np.sqrt(np.einsum('ij,jk,ik->i',w,cov,w)),atol=1e-11)
            np.testing.assert_allclose(r.q_bh_24,bh(2*stats.t.sf(abs(r.t),r.df)),atol=1e-11)
            assert len(r)==24 and (r.df<=n-1).all()
            r.insert(0,'cohort',cohort);allrows.append(r)
        if cohort=='N20':
            for policy in ['primary','min100']:
                t.record(ROOT/'results/behavioral/fmri_rt_20261012/lme'/f'{policy}_model.rds')
            t.record(BEHAV/'source_replay/historical_model_reuse_checks.json')
        report=f'# {cohort} reaction-time mixed model\n\nThe source-replayed custom block cleaner supplies the correct responded trials. Log RT uses the unchanged 20 observed run-condition cells, independent participant intercept/sequence slopes and nested run/block intercepts; REML, within-block AR(1) indexed by original trial order, condition-specific residual variance. The primary policy retains RT>0; the sensitivity requires RT≥100ms. The same 24 contrast vectors are used. Tests use approximate t degrees of freedom capped at {n-1}, with BH across 24 per policy; pointwise model-based 95% intervals and geometric ratios are exported. The df cap is not an exact or guaranteed conservative calibration.\n\n'
        for policy,status in statuses.items():report+=f'- {policy}: {status["n_trials"]} trials, AR1={status["AR1"]}, {status["n_fdr_significant"]}/24 BH findings.\n'
        report+='\nN20 saved models were reused only after an exact saved-data check; its trial keys and RT values also reproduce the historical cleaned dataset. N22 was refitted. Primary and min100 estimates answer different response-selection questions.\n'
        t.finish(report,dict(passed=True,n_subjects=n,n_contrasts_per_policy=24,statuses=statuses,contrast_algebra_and_BH_checked=True))
    t=Task('rt_comparison',behavior=True)
    for cohort in ['N20','N22']:t.record(BEHAV/f'rt_lme_{cohort}/validation.json')
    csv(pd.concat(allrows),t.path/'matched_contrasts.csv')
    t.finish('# Matched reaction-time comparison\n\nN20 and N22 use identical cleaning, contrast definitions and mixed-model structure. Estimates and pointwise intervals are compared descriptively because the samples overlap; no independent-group test is performed.\n',dict(passed=True,n_rows=96))


def errors():
    task=Task('error_models',behavior=True);rows=[];cells=[];statuses={}
    for cohort,n in [('N20',20),('N22',22)]:
        source=ROOT/'results/behavioral/block_errors_20261013/mixed_model' if n==20 else BEHAV/'source_replay/N22/mixed_model'
        out=task.path/cohort;out.mkdir(exist_ok=True)
        fit=json.loads(task.record(source/'fit_status.json').read_text());statuses[cohort]=fit
        assert fit['status']=='completed' and fit['n_subjects']==n and fit['bootstrap_requested']==999
        for model in ['binomial','gaussian']:assert fit[f'bootstrap_{model}_valid']>=.95*999
        if (source/'compute_resume_20261018.json').exists():
            resumed=json.loads(task.record(source/'compute_resume_20261018.json').read_text())
            assert digest(source/'bootstrap_subject_draws.csv')==resumed['bootstrap_draws_before_resume_sha256']
        weights=pd.read_csv(task.record(source/'contrast_weights.csv')).set_index('contrast')
        meta=pd.read_csv(task.record(source/'cell_metadata.csv')).sort_values('cell')
        counts=pd.read_csv(task.record(BEHAV/'source_replay'/cohort/'participant_block_error_rates.csv'))
        assert counts.n_trials.sum()==n*2560 and counts.n_errors.sum()==fit['n_errors']
        for file in ['analysis_spec.json','bootstrap_status.csv','bootstrap_subject_draws.csv','optimizer_agreement.json','binomial_model.rds','gaussian_model.rds','R_session_info.txt','gaussian_full_hierarchy_model.rds','gaussian_full_variance_components.csv','binomial_variance_components.csv','gaussian_variance_components.csv','bootstrap_messages.json']:
            task.record(source/file)
        agreement=json.loads((source/'optimizer_agreement.json').read_text())
        assert agreement['max_beta_difference']<.003 and agreement['log_likelihood_difference']<.001
        for model in ['binomial','gaussian']:
            coef=pd.read_csv(task.record(source/f'{model}_fixed_coefficients.csv')).set_index('term');terms=coef.index
            assert [v.removeprefix('cell') for v in terms]==meta.cell.tolist()
            beta=coef.estimate.to_numpy();w=weights.loc[:,terms].to_numpy()
            native=pd.read_csv(task.record(source/f'{model}_fixed_covariance_native.csv'),index_col=0).loc[terms,terms].to_numpy()
            boot=pd.read_csv(task.record(source/f'{model}_bootstrap_fixed_coefficients.csv')).set_index('replicate')[terms]
            assert len(boot)==fit[f'bootstrap_{model}_valid'] and np.isfinite(boot).all().all()
            cov=np.cov(boot,rowvar=False,ddof=1)
            pd.DataFrame(cov,index=terms,columns=terms).to_csv(out/f'{model}_fixed_covariance_bootstrap.csv')
            estimates=w@beta;se=np.sqrt(np.einsum('ij,jk,ik->i',w,cov,w));assert (se>0).all()
            low,high=np.quantile(boot.to_numpy()@w.T,[.025,.975],axis=0)
            statistic=estimates/se;p=2*stats.t.sf(abs(statistic),n-1)
            native_se=np.sqrt(np.einsum('ij,jk,ik->i',w,native,w))
            native_p=2*(stats.norm.sf(abs(estimates/native_se)) if model=='binomial' else stats.t.sf(abs(estimates/native_se),n-1))
            result=pd.DataFrame(dict(cohort=cohort,model=model,contrast=weights.index,estimate=estimates,se_bootstrap=se,
                df=n-1,t=statistic,p_two_sided=p,q_bh_24=bh(p),ci_low=low,ci_high=high,native_se=native_se,native_p=native_p,native_q_bh_24=bh(native_p)))
            if model=='binomial':
                result['odds_ratio']=np.exp(estimates);result['odds_ratio_ci_low']=np.exp(low);result['odds_ratio_ci_high']=np.exp(high)
            else:
                result['difference_pp']=100*estimates;result['se_pp']=100*se;result['ci_low_pp']=100*low;result['ci_high_pp']=100*high
            csv(result,out/f'{model}_contrasts.csv');rows.append(result)
            cell=meta.copy();cell['cohort']=cohort;cell['model']=model;cell['estimate']=beta
            cell['ci_low'],cell['ci_high']=np.quantile(boot,[.025,.975],axis=0)
            if model=='binomial':
                names=['intercept_variance','str1_variance','str2_variance']
                v=pd.read_csv(task.record(source/'binomial_marginal_variances.csv')).set_index('component').variance.loc[names].to_numpy()
                bv=pd.read_csv(task.record(source/'binomial_bootstrap_marginal_variances.csv')).set_index('replicate').loc[boot.index,names].to_numpy()
                condition=meta.rule_group.to_numpy();m=marginal(beta,v,condition)[0]
                np.testing.assert_allclose(m,marginal(beta,v,condition,160)[0],rtol=1e-8,atol=1e-10)
                bm=marginal(boot.to_numpy(),bv,condition)
                cell['marginal_probability']=m;cell['marginal_ci_low'],cell['marginal_ci_high']=np.quantile(bm,[.025,.975],axis=0)
                csv(pd.DataFrame(bm,index=boot.index,columns=meta.cell).reset_index(),out/'binomial_bootstrap_marginal_probabilities.csv')
            else:cell['probability']=beta
            csv(cell,out/f'{model}_cells.csv');cells.append(cell)
            if n==20:
                old=pd.read_csv(task.record(source/f'{model}_contrasts.csv'))
                for col in ['estimate','se_bootstrap','q_bh_24','ci_low','ci_high']:np.testing.assert_allclose(result[col],old[col],rtol=2e-9,atol=1e-11)
        # Temporal residual check retains original within-block trial order.
        trials=pd.read_csv(task.record(BEHAV/'source_replay'/cohort/'raw_trials.csv'))
        pred=pd.read_csv(task.record(source/'binomial_block_diagnostics.csv'))
        trials=trials.merge(pred[['subject','session','block','rule_group','conditional_prediction']],on=['subject','session','block','rule_group'],validate='many_to_one').sort_values(['subject','session','trial_in_run'])
        prob=trials.conditional_prediction
        trials['pearson']=((1-trials.correct_response)-prob)/np.sqrt(prob*(1-prob))
        trials['next']=trials.groupby(['subject','session','block']).pearson.shift(-1)
        pair=trials.dropna(subset=['next']);assert len(pair)==n*(2560-64)
        gp=pd.read_csv(task.record(source/'gaussian_block_diagnostics.csv'))
        writej(out/'diagnostics.json',dict(pooled_trial_lag1=float(np.corrcoef(pair.pearson,pair.next)[0,1]),
            n_adjacent_pairs=len(pair),naive_dispersion=float(sum(pred.pearson_residual**2)/(len(pred)-20)),
            gaussian_predictions_outside_0_1=int((~gp.conditional_prediction.between(0,1)).sum()),
            note='Descriptive residual checks; dispersion denominator omits effective random-effect df.'))
    allrows=pd.concat(rows);csv(allrows,task.path/'matched_contrasts.csv');csv(pd.concat(cells),task.path/'matched_cell_estimates.csv')
    for cohort,fit in statuses.items():writej(task.path/cohort/'fit_status.json',fit)
    c=pd.concat(cells);fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,cohort in zip(axes,['N20','N22']):
        for condition,color in [('Ran','#777777'),('Str1','#356ba3'),('Str2','#aa5279')]:
            d=c[(c.cohort==cohort)&(c.model=='binomial')&(c.rule_group==condition)].sort_values('session')
            ax.errorbar(d.session,100*d.marginal_probability,yerr=100*np.vstack([d.marginal_probability-d.marginal_ci_low,d.marginal_ci_high-d.marginal_probability]),fmt='o-',color=color,label=condition,ms=3)
        ax.set(title=cohort,xlabel='Run',xticks=range(1,9))
    axes[0].set_ylabel('Population-average error probability (%)');axes[1].legend();fig.suptitle('Pointwise participant-bootstrap 95% intervals');fig.tight_layout();savefig(fig,task.path/'marginal_error_probabilities')
    task.record(Path(__file__).with_name('updated_error_glmm.R'))
    task.record(BEHAV/'source_replay/historical_model_reuse_checks.json')
    report='# Matched N20/N22 error models\n\nErrors include incorrect responses and omissions among all logged trials, with no RT exclusions. Binomial logit GLMM uses 20 observed run-condition cells, independent participant intercept/sequence slopes, and nested run/block intercepts. The Gaussian sensitivity models trial-denominator-weighted block proportions; its full hierarchy is retained unless the block variance is numerically zero and equivalence is verified. N20 source counts match exactly, allowing historical model/bootstrap reuse; N22 is refitted.\n\nThe 999 complete-participant bootstrap draws keep observed within-person dependence. Singular variance estimates are permitted and counted; numerical nonconvergence is excluded, with at least 95% valid fits required. Contrast tests use bootstrap covariance, approximate t(19)/t(21), and separate BH families of 24 per cohort/model. Percentile 95% intervals are pointwise. Native z (binomial) or approximate t (Gaussian) results are exported separately. Conditional odds ratios differ from population-average error probabilities, which integrate random-effect distributions with quadrature checked at 80 versus 160 nodes. No independent-group test compares the overlapping samples.\n\n'
    for cohort,fit in statuses.items():
        report+=f'- {cohort}: {fit["n_trials"]} trials, {fit["n_errors"]} errors; valid binomial/Gaussian bootstrap refits {fit["bootstrap_binomial_valid"]}/{fit["bootstrap_gaussian_valid"]} of 999.\n'
    report+='\n| Sample | Contrast | Conditional OR [pointwise 95% CI] | BH q |\n|---|---|---:|---:|\n'
    for _,r in allrows[(allrows.model=='binomial')&allrows.contrast.str.contains('run3_Str2_minus_Str1|early|middle|late')].iterrows():
        report+=f'| {r.cohort} | {r.contrast} | {r.odds_ratio:.3f} [{r.odds_ratio_ci_low:.3f}, {r.odds_ratio_ci_high:.3f}] | {r.q_bh_24:.4g} |\n'
    task.finish(report,dict(passed=True,n_contrast_rows=len(allrows),bootstrap_replicates=999,N20_source_reuse_verified=True,quadrature_80_160_checked=True,statuses=statuses))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rt-only',action='store_true');a=parser.parse_args()
    rt()
    if not a.rt_only:errors()
