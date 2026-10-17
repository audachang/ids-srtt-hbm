#!/usr/bin/env python3
"""Independently summarize error mixed models and participant-bootstrap inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, special, stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_bh(p):
    p=np.asarray(p,float);order=np.argsort(p);q=np.empty(len(p))
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q


def marginalized(beta,variances,conditions,nodes=80):
    beta=np.asarray(beta,float);v=np.asarray(variances,float)
    if beta.ndim==1:beta=beta[None,:]
    if v.ndim==1:v=v[None,:]
    variance=v[:,0,None]+v[:,1,None]*(conditions=='Str1')+v[:,2,None]*(conditions=='Str2')
    x,w=special.roots_hermitenorm(nodes);w=w/np.sqrt(2*np.pi)
    return (special.expit(beta[:,:,None]+np.sqrt(variance[:,:,None])*x)*w).sum(axis=2)


def probability(x):
    return '<.001' if x<.001 else f'{x:.3f}'


def make_table(path,title,label,rows,binomial):
    headers=['Contrast','Log odds ratio','SE','Odds ratio',r'95\% OR CI','$q_{24}$'] if binomial else [
        'Contrast','Difference (pp)','SE (pp)',r'95\% difference CI','$q_{24}$']
    text=[r'\small',r'\begin{longtable}{p{.28\linewidth}'+('rrrrr' if binomial else 'rrrr')+'}',
          r'\caption{'+title+r'}\label{'+label+r'}\\',r'\toprule',
          ' & '.join(headers)+r' \\',r'\midrule',r'\endfirsthead',
          r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead',
          r'\bottomrule',r'\endfoot']
    for _,r in rows.iterrows():
        name=r.contrast.replace('_minus_',' -- ').replace('_',' ').replace('Str1','Structure 1').replace('Str2','Structure 2')
        if binomial:
            values=[name,f'{r.estimate:.3f}',f'{r.se_bootstrap:.3f}',f'{r.odds_ratio:.3f}',
                    f'[{r.odds_ratio_ci_low:.3f}, {r.odds_ratio_ci_high:.3f}]',probability(r.q_bh_24)]
        else:
            values=[name,f'{r.difference_pp:.3f}',f'{r.se_pp:.3f}',
                    f'[{r.ci_low_pp:.3f}, {r.ci_high_pp:.3f}]',probability(r.q_bh_24)]
        text.append(' & '.join(values)+r' \\')
    text += [r'\end{longtable}',
        'Intervals are pointwise participant-bootstrap percentile intervals and are unadjusted for multiplicity. '
        'Tests use bootstrap covariance and an approximate participant-based $t(19)$ reference; BH covers all 24 contrasts separately for each model. '
        +('Adjusted-change and phase-pair odds ratios are ratios of conditional odds ratios. The plotted probabilities integrate random effects.' if binomial else
          'The Gaussian sensitivity fits block proportions with trial-denominator weights. Differences are in percentage points.'),'']
    path.write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    root=parser.parse_args().root.resolve();out=root/'results/behavioral/block_errors_20261013/mixed_model'
    fit=json.loads((out/'fit_status.json').read_text())
    assert fit['status']=='completed' and fit['n_subjects']==20 and fit['n_trials']==51200
    assert fit['bootstrap_binomial_valid']>=.95*fit['bootstrap_requested']
    assert fit['bootstrap_gaussian_valid']>=.95*fit['bootstrap_requested']
    raw_weights=pd.read_csv(out/'contrast_weights.csv').set_index('contrast')
    rt_weights=pd.read_csv(root/'results/behavioral/fmri_rt_20261012/lme/min100_contrast_weights.csv').set_index('contrast')
    pd.testing.assert_frame_equal(raw_weights,rt_weights)
    meta=pd.read_csv(out/'cell_metadata.csv').sort_values('cell')
    run=meta.session.to_numpy();conditions=meta.rule_group.to_numpy();assert len(meta)==20
    design=np.column_stack([np.ones(20)]+[(run==r).astype(float) for r in range(2,9)]+
                           [(conditions==c).astype(float) for c in ['Str1','Str2']])
    K=linalg.null_space(design.T).T;assert K.shape==(10,20)
    outputs={};bootstrap={};model_beta={};omnibus=[];covariance_ranks={}
    for model in ['binomial','gaussian']:
        coef=pd.read_csv(out/f'{model}_fixed_coefficients.csv').set_index('term')
        terms=coef.index.tolist();beta=coef.estimate.to_numpy();model_beta[model]=beta
        assert [t.removeprefix('cell') for t in terms]==meta.cell.tolist()
        native=pd.read_csv(out/f'{model}_fixed_covariance_native.csv',index_col=0).loc[terms,terms].to_numpy()
        np.testing.assert_allclose(np.sqrt(np.diag(native)),coef.native_se,rtol=2e-9)
        boot=pd.read_csv(out/f'{model}_bootstrap_fixed_coefficients.csv').set_index('replicate')[terms]
        assert len(boot)==fit[f'bootstrap_{model}_valid'] and np.isfinite(boot).all().all()
        bootstrap[model]=boot
        covariance=np.cov(boot.to_numpy(),rowvar=False,ddof=1)
        eigenvalues=np.linalg.eigvalsh(covariance)
        assert eigenvalues.min()>-1e-12
        covariance_ranks[model]=int(np.linalg.matrix_rank(covariance,tol=eigenvalues.max()*1e-10))
        # Gaussian fixed cell means are linear averages of 20 participant vectors;
        # their bootstrap covariance can have rank 19. Requested contrasts are
        # checked on their own variances rather than adding a numerical ridge.
        pd.DataFrame(covariance,index=terms,columns=terms).to_csv(out/f'{model}_fixed_covariance_bootstrap.csv')
        weights=raw_weights.loc[:,terms].to_numpy()
        estimate=weights@beta
        contrast_variances=np.einsum('ij,jk,ik->i',weights,covariance,weights)
        assert (contrast_variances>0).all()
        se=np.sqrt(contrast_variances)
        native_se=np.sqrt(np.einsum('ij,jk,ik->i',weights,native,weights))
        samples=boot.to_numpy()@weights.T
        low,high=np.quantile(samples,[.025,.975],axis=0)
        statistic=estimate/se;p=2*stats.t.sf(np.abs(statistic),19)
        q=stats.false_discovery_control(p,method='bh');np.testing.assert_allclose(q,independent_bh(p))
        native_p=2*(stats.norm.sf(np.abs(estimate/native_se)) if model=='binomial' else stats.t.sf(np.abs(estimate/native_se),19))
        result=pd.DataFrame(dict(model=model,contrast=raw_weights.index,estimate=estimate,
            se_bootstrap=se,df=19,t=statistic,p_two_sided=p,q_bh_24=q,
            ci_low=low,ci_high=high,native_se=native_se,native_p=native_p,
            native_q_bh_24=stats.false_discovery_control(native_p,method='bh')))
        if model=='binomial':
            result['odds_ratio']=np.exp(estimate);result['odds_ratio_ci_low']=np.exp(low);result['odds_ratio_ci_high']=np.exp(high)
        else:
            result['difference_pp']=100*estimate;result['se_pp']=100*se
            result['ci_low_pp']=100*low;result['ci_high_pp']=100*high
        result.to_csv(out/f'{model}_contrasts.csv',index=False,float_format='%.12g');outputs[model]=result
        cells=meta.copy();cells['model']=model;cells['estimate']=beta
        cells['se_bootstrap']=np.sqrt(np.diag(covariance))
        cells['ci_low'],cells['ci_high']=np.quantile(boot.to_numpy(),[.025,.975],axis=0)
        if model=='binomial':
            v=pd.read_csv(out/'binomial_marginal_variances.csv').set_index('component').variance.loc[
                ['intercept_variance','str1_variance','str2_variance']].to_numpy()
            bv=pd.read_csv(out/'binomial_bootstrap_marginal_variances.csv').set_index('replicate').loc[boot.index,
                ['intercept_variance','str1_variance','str2_variance']].to_numpy()
            marginal=marginalized(beta,v,conditions)[0]
            np.testing.assert_allclose(marginal,marginalized(beta,v,conditions,nodes=160)[0],rtol=1e-8,atol=1e-10)
            boot_marginal=marginalized(boot.to_numpy(),bv,conditions)
            assert np.isfinite(boot_marginal).all() and (boot_marginal>=0).all() and (boot_marginal<=1).all()
            cells['conditional_probability']=special.expit(beta)
            cells['conditional_ci_low']=special.expit(cells.ci_low);cells['conditional_ci_high']=special.expit(cells.ci_high)
            cells['marginal_error_probability']=marginal
            cells['marginal_ci_low'],cells['marginal_ci_high']=np.quantile(boot_marginal,[.025,.975],axis=0)
            pd.DataFrame(boot_marginal,index=boot.index,columns=meta.cell).to_csv(out/'binomial_bootstrap_marginal_probabilities.csv')
        else:
            cells['error_probability']=beta
        cells.to_csv(out/f'{model}_cell_estimates.csv',index=False,float_format='%.12g')
        adjusted=K@beta;covK=K@covariance@K.T
        assert np.linalg.matrix_rank(covK)==10 and np.linalg.eigvalsh(covK).min()>0
        F=adjusted@np.linalg.solve(covK,adjusted)/10
        omnibus.append(dict(model=model,numerator_df=10,denominator_df=19,F=F,
                            p=stats.f.sf(F,10,19),method='Joint Wald F using participant-bootstrap covariance; approximate participant df'))
        title='Binomial mixed-model error contrasts.' if model=='binomial' else 'Denominator-weighted Gaussian LMM error contrasts.'
        make_table(root/'manuscript/tables'/f'error_{model}_contrasts.tex',title,'tab:error'+model,result,model=='binomial')
    pd.DataFrame(omnibus).to_csv(out/'run_condition_wald.csv',index=False,float_format='%.12g')
    comparison=outputs['binomial'].merge(outputs['gaussian'],on='contrast',suffixes=('_binomial','_gaussian'))
    comparison['same_direction']=np.sign(comparison.estimate_binomial)==np.sign(comparison.estimate_gaussian)
    comparison['both_fdr']=comparison.q_bh_24_binomial.le(.05)&comparison.q_bh_24_gaussian.le(.05)
    comparison.to_csv(out/'model_comparison.csv',index=False,float_format='%.12g')
    # Trial-level temporal diagnostics use actual trial order, without treating pairs as subjects.
    trials=pd.read_csv(out.parent/'trial_outcomes.csv')
    predictions=pd.read_csv(out/'binomial_block_diagnostics.csv')
    join=['subject','session','block','rule_group']
    trials=trials.merge(predictions[join+['conditional_prediction']],on=join,validate='many_to_one')
    trials=trials.sort_values(['subject','session','trial_in_run'])
    prob=trials.conditional_prediction.to_numpy()
    trials['pearson']=(trials.error-prob)/np.sqrt(prob*(1-prob))
    trials['next_pearson']=trials.groupby(['subject','session','block']).pearson.shift(-1)
    paired=trials.dropna(subset=['next_pearson'])
    assert len(paired)==51200-1280
    pooled=np.corrcoef(paired.pearson,paired.next_pearson)[0,1]
    pearson_block=predictions.pearson_residual.to_numpy()
    diagnostics=dict(pooled_within_block_trial_lag1=float(pooled),n_adjacent_trial_pairs=len(paired),
        naive_pearson_dispersion=float(np.sum(pearson_block**2)/(len(predictions)-20)),
        interpretation='Descriptive only; the dispersion denominator does not account for random-effect effective degrees of freedom',
        gaussian_predictions_outside_probability_range=int(((pd.read_csv(out/'gaussian_block_diagnostics.csv').conditional_prediction<0)|
            (pd.read_csv(out/'gaussian_block_diagnostics.csv').conditional_prediction>1)).sum()))
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    axes[0].scatter(predictions.conditional_prediction*100,predictions.pearson_residual,s=5,alpha=.18)
    axes[0].axhline(0,color='.5',lw=1);axes[0].set(xlabel='Conditional error probability (%)',ylabel='Block Pearson residual')
    stats.probplot(pd.read_csv(out/'gaussian_block_diagnostics.csv').pearson_residual,dist='norm',plot=axes[1])
    axes[1].set_title('Gaussian sensitivity residuals')
    fig.tight_layout();fig.savefig(out/'model_diagnostics.pdf');plt.close(fig)
    # Independently re-read exported tables and reconstruct their estimates and standard errors.
    for model in ['binomial','gaussian']:
        r=pd.read_csv(out/f'{model}_contrasts.csv')
        cov=pd.read_csv(out/f'{model}_fixed_covariance_bootstrap.csv',index_col=0).to_numpy()
        W=raw_weights.loc[:,bootstrap[model].columns].to_numpy()
        np.testing.assert_allclose(r.estimate,W@model_beta[model],rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(r.se_bootstrap,np.sqrt(np.diag(W@cov@W.T)),rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(r.q_bh_24,independent_bh(2*stats.t.sf(np.abs(r.estimate/r.se_bootstrap),19)),rtol=2e-9,atol=1e-11)
    inputs=[out.parent/'participant_block_error_rates.csv',out.parent/'trial_outcomes.csv',
            out/'binomial_fixed_coefficients.csv',out/'gaussian_fixed_coefficients.csv',
            out/'binomial_bootstrap_fixed_coefficients.csv',out/'gaussian_bootstrap_fixed_coefficients.csv',
            out/'binomial_bootstrap_marginal_variances.csv',out/'contrast_weights.csv',out/'analysis_spec.json']
    validation=dict(status='passed',n_subjects=20,n_trials=51200,n_errors=1425,n_contrasts_each=24,
        bootstrap_covariance_ranks=covariance_ranks,
        bootstrap_requested=fit['bootstrap_requested'],bootstrap_binomial_valid=fit['bootstrap_binomial_valid'],
        bootstrap_gaussian_valid=fit['bootstrap_gaussian_valid'],n_same_direction=int(comparison.same_direction.sum()),
        n_binomial_fdr=int(outputs['binomial'].q_bh_24.le(.05).sum()),n_gaussian_fdr=int(outputs['gaussian'].q_bh_24.le(.05).sum()),
        contrast_covariance_and_bh_independently_checked=True,marginal_quadrature_80_vs_160_checked=True,
        fit_script_sha256=digest(root/'code/reporting/fit_error_mixed_models.R'),summary_script_sha256=digest(Path(__file__)),
        input_sha256={str(path):digest(path) for path in inputs})
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report=['# Mixed-effects analysis of error rates','',
        'The primary binomial logit GLMM models errors/omissions among all 51,200 trials from 20 imaging participants, using 1,760 block/condition count observations. No RT exclusions are applied. Twenty observed run/condition cells use the same 24 contrasts as the RT model. Independent participant intercept/sequence slopes and nested participant/run/block intercepts model clustering. The Gaussian LMM sensitivity uses trial-denominator-weighted block proportions and REML; its zero block-variance component is omitted in the equivalent reduced model.','',
        f'Whole-participant bootstrap: {fit["bootstrap_binomial_valid"]}/{fit["bootstrap_requested"]} valid binomial and {fit["bootstrap_gaussian_valid"]}/{fit["bootstrap_requested"]} valid Gaussian refits. Bootstrap covariance gives approximate t(19) tests; BH covers 24 contrasts separately within model. Pointwise percentile intervals are not simultaneous, and t calibration is approximate. Native model-based inference is also exported.','',
        'Binomial contrast effects are conditional odds ratios; phase effects are ratios of odds ratios. Plotted absolute probabilities average over the fitted normal random-effect distributions. These plotted marginal probabilities are distinct from conditional odds-ratio contrasts. The Gaussian sensitivity gives percentage-point differences.','',
        '| Contrast | Odds ratio | Pointwise 95% bootstrap CI | BH q | Gaussian difference (pp) | Gaussian BH q |',
        '|---|---:|---|---:|---:|---:|']
    gauss=outputs['gaussian'].set_index('contrast')
    for _,r in outputs['binomial'].iterrows():
        g=gauss.loc[r.contrast]
        report.append(f'| {r.contrast} | {r.odds_ratio:.3f} | [{r.odds_ratio_ci_low:.3f}, {r.odds_ratio_ci_high:.3f}] | {r.q_bh_24:.4g} | {g.difference_pp:.3f} | {g.q_bh_24:.4g} |')
    report += ['',f'Joint run/condition Wald tests: {omnibus}.', '',
        f'The two models agree on {comparison.same_direction.sum()}/24 contrast directions. Binomial and Gaussian BH families have {validation["n_binomial_fdr"]} and {validation["n_gaussian_fdr"]} findings, respectively. These models estimate different scales and need not yield identical effect magnitudes or thresholds.', '',
        f'Descriptive pooled trial residual lag-1 correlation within blocks: {pooled:.4f}. Gaussian conditional predictions outside [0,1]: {diagnostics["gaussian_predictions_outside_probability_range"]}/{len(predictions)}. The binomial model has no separate AR(1) process; whole-participant bootstrap retains actual within-person trial dependence.', '',
        'The analysis is exploratory. Intermixing, prevalence, familiarity, and practice change together, and the cohort is selected for initial Structure 1 learning. Bootstrap variance estimates may reach boundaries; all singular/nonconverged refits and optimizer messages are retained in the analysis records.','']
    (out/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(validation,indent=2))
    print(outputs['binomial'][['contrast','odds_ratio','odds_ratio_ci_low','odds_ratio_ci_high','q_bh_24']].to_string(index=False))


if __name__=='__main__':main()
