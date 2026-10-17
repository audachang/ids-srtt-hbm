#!/usr/bin/env python3
"""Independently verify extraction, uncertainty, corrections, sample and TeX inputs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from augment_roi_reporting import digest, observed_mean


class CoverageRegressionTests(unittest.TestCase):
    def test_unobserved_zero_is_not_an_observation(self):
        x=np.array([2.,0.]);mask=np.array([True,True]);support=np.array([True,False])
        self.assertEqual(observed_mean(x,mask,support,.5),2.)
        self.assertTrue(np.isnan(observed_mean(x,mask,support,.8)))

    def test_observed_zero_is_retained(self):
        x=np.array([2.,0.]);mask=np.array([True,True]);support=mask.copy()
        self.assertEqual(observed_mean(x,mask,support,1.),1.)

    def test_nonfinite_tissue_reduces_coverage(self):
        x=np.array([2.,np.nan]);mask=np.array([True,True])
        self.assertTrue(np.isnan(observed_mean(x,mask,mask,1.)))

    def test_direct_stage_difference_cancels_initial(self):
        initial=np.array([-.3,.8,.1]);early=np.array([.2,.4,.9]);late=np.array([.5,.7,.4])
        np.testing.assert_allclose((early-initial)-(late-initial),early-late)


def independent_bh(p):
    x=np.asarray(p,float);x=np.where(np.isfinite(x),x,1.)
    order=np.argsort(x);n=len(x)
    q=np.minimum.accumulate((x[order]*n/np.arange(1,n+1))[::-1])[::-1]
    result=np.empty(n);result[order]=np.minimum(q,1.)
    return result


def verify_stats(values,results,keys,value='value'):
    indexed=results.set_index(keys)
    for key,part in values.groupby(keys,sort=False):
        x=part[value].dropna().to_numpy();row=indexed.loc[key]
        assert int(row.n)==len(x)
        if len(x)==0:
            assert pd.isna(row['mean']);continue
        np.testing.assert_allclose(row['mean'],np.mean(x),rtol=2e-9,atol=2e-11)
        if len(x)<2: assert pd.isna(row.p);continue
        se=x.std(ddof=1)/np.sqrt(len(x))
        t=x.mean()/se
        np.testing.assert_allclose([row.sd,row.se,row.t,row.p,row.cohen_dz],
            [x.std(ddof=1),se,t,2*stats.t.sf(abs(t),len(x)-1),x.mean()/x.std(ddof=1)],rtol=2e-9,atol=2e-11)
        np.testing.assert_allclose([row.ci_low,row.ci_high],
            x.mean()+np.array([-1,1])*stats.t.isf(.025,len(x)-1)*se,rtol=2e-9,atol=2e-11)



def validate_error_mixed_models(root):
    out=root/'results/behavioral/block_errors_20261013/mixed_model'
    path=out/'validation.json'
    if not path.is_file():
        return None
    record=json.loads(path.read_text())
    assert record['status']=='passed' and record['n_subjects']==20 and record['n_contrasts_each']==24
    assert record['fit_script_sha256']==digest(root/'code/reporting/fit_error_mixed_models.R')
    assert record['summary_script_sha256']==digest(root/'code/reporting/summarize_error_mixed_models.py')
    for source,sha in record['input_sha256'].items():
        assert digest(source)==sha,source
    weights=pd.read_csv(out/'contrast_weights.csv').set_index('contrast')
    for model in ['binomial','gaussian']:
        coefficients=pd.read_csv(out/f'{model}_fixed_coefficients.csv').set_index('term').estimate
        bootstrap=pd.read_csv(out/f'{model}_bootstrap_fixed_coefficients.csv').set_index('replicate').loc[:,coefficients.index]
        covariance=pd.read_csv(out/f'{model}_fixed_covariance_bootstrap.csv',index_col=0).loc[coefficients.index,coefficients.index]
        np.testing.assert_allclose(covariance,np.cov(bootstrap,rowvar=False,ddof=1),rtol=2e-9,atol=1e-11)
        W=weights.loc[:,coefficients.index].to_numpy()
        contrasts=pd.read_csv(out/f'{model}_contrasts.csv').set_index('contrast').loc[weights.index]
        estimate=W@coefficients.to_numpy()
        se=np.sqrt(np.einsum('ij,jk,ik->i',W,covariance,W))
        np.testing.assert_allclose(contrasts.estimate,estimate,rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(contrasts.se_bootstrap,se,rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(contrasts.q_bh_24,independent_bh(2*stats.t.sf(abs(estimate/se),19)),rtol=2e-9,atol=1e-11)
    figure=json.loads((root/'manuscript/figures/behavioral_rt_and_errors_metadata.json').read_text())
    assert figure['error_mixed_model'] and figure['n_error_participant_points']==400
    for source,sha in figure['input_sha256'].items():
        assert digest(source)==sha,source
    for name,artifacts in figure['artifacts'].items():
        for extension,sha in artifacts.items():
            assert digest(root/'manuscript/figures'/(name+'.'+extension))==sha
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--historical',action='store_true')
    a=p.parse_args();root=a.root.resolve();out=root/'results/glm_volume/reporting_20261012'
    tests=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(CoverageRegressionTests))
    assert tests.wasSuccessful()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['historical_reproduction_rows']==1728
    assert summary['script_sha256']==digest(Path(__file__).with_name('augment_roi_reporting.py'))
    assert max(summary['condition_algebra_max_errors'].values())<2e-7
    manifests=[out/'input_manifest.json',root/'results/behavioral/fmri_rt_20261012/input_manifest.json']
    for manifest in manifests:
        for path,sha in json.loads(manifest.read_text()).items(): assert digest(path)==sha,path
    values=pd.read_csv(out/'participant_roi_values.csv');results=pd.read_csv(out/'all_model_roi_results.csv')
    assert len(values)==9*4*4*20*48 and len(results)==9*4*4*48
    keys=['model','policy','stage','roi'];assert not results.duplicated(keys).any()
    verify_stats(values,results,keys)
    for _,part in results.groupby(['model','policy','stage']):
        np.testing.assert_allclose(part.q_within_contrast,independent_bh(part.p),rtol=2e-9,atol=2e-11)
    for _,part in results.groupby(['model','policy']):
        np.testing.assert_allclose(part.q_joint_stages,independent_bh(part.p),rtol=2e-9,atol=2e-11)
    for _,part in results.groupby('policy'):
        np.testing.assert_allclose(part.q_enumerated_models_stages,independent_bh(part.p),rtol=2e-9,atol=2e-11)
    contrasts=pd.read_csv(out/'direct_stage_comparisons.csv');direct=[]
    sel=values[values.model=='censor_hp128']
    for policy,part in sel.groupby('policy'):
        pv=part.pivot(index=['subject','roi'],columns='stage',values='value')
        for e,l in [('early','middle'),('early','late'),('middle','late')]:
            diff=(pv[e]-pv[l]).reset_index(name='value');diff['policy']=policy
            diff['pair']=e+'_minus_'+l;direct.append(diff)
    verify_stats(pd.concat(direct),contrasts,['policy','pair','roi'])
    for _,part in contrasts.groupby('policy'):
        np.testing.assert_allclose(part.q_three_pairs_144,independent_bh(part.p),rtol=2e-9,atol=2e-11)
    coverage=pd.read_csv(out/'participant_roi_coverage.csv')
    obs=sel[sel.policy=='observed80'].merge(coverage,on=['subject','roi'])
    assert np.array_equal(obs.value.notna(),obs.fraction.ge(.8))
    assert obs.groupby('roi').apply(lambda x:x.groupby('stage').value.count().nunique(),include_groups=False).eq(1).all()
    # Verify re-extraction on all subjects for a complete ROI, a partial cerebellar
    # ROI, and a scarcely covered frontal-pole ROI, independently of stored means.
    masks={r:np.asarray(nib.load(a.source_root/'resources/roi_masks'/f'{r}.nii.gz').dataobj)>0
           for r in ['LeftAccumbensArea','LeftCerebellum','LeftFrontalPole']}
    for s in summary['subjects']:
        for stage,contrast in zip(['initial','early','middle','late'],
            ['Str1_minus_Ran_r12','Switch_r34_vs_r12_did','Switch_r56_vs_r12_did','Switch_r78_vs_r12_did']):
            im=nib.load(a.source_root/'results/glm_volume/first_level/FLA_nilearn_sensitivity_20260921/censor_hp128/subject_contrasts/eventrelated'/s/f'{contrast}.nii.gz')
            data=np.asarray(im.dataobj,float)
            for r,m in masks.items():
                val=data[m].sum()/m.sum()
                saved=sel[(sel.subject==s)&(sel.stage==stage)&(sel.roi==r)&(sel.policy=='legacy_full_parcel')].value.iloc[0]
                np.testing.assert_allclose(val,saved,rtol=2e-9,atol=2e-11)
    figuremeta=json.loads((root/'manuscript/figures/observed_roi_metadata.json').read_text())
    assert figuremeta['symmetric_cbar'] and figuremeta['color_scale']==[-8,8]
    assert figuremeta['script_sha256']==digest(Path(__file__).with_name('make_reporting_tables.py'))
    assert figuremeta['source_csv_sha256']==digest(out/'reported_roi_results.csv')
    for extension in ['png','pdf']:
        assert figuremeta[extension+'_sha256']==digest(root/'manuscript/figures'/('observed_roi_axial.'+extension))
    assert figuremeta['counts']=={'initial':29,'early':15,'middle':5,'late':0}
    behavior=root/'results/behavioral/fmri_rt_20261012';bm=json.loads((behavior/'summary.json').read_text())
    assert bm['script_sha256']==digest(Path(__file__).with_name('make_fmri_rt_figure.py'))
    assert bm['n_subjects']==20 and bm['original_raw_runs_verified']==160
    rawchecks=pd.read_csv(behavior/'original_raw_agreement.csv')
    assert len(rawchecks)==160 and rawchecks.n_trials.sum()==51200 and rawchecks.max_rt_error_s.max()<1e-10
    cycles=pd.read_csv(behavior/'participant_cycles.csv');assert set(cycles.subject)==set(summary['subjects'])
    group=pd.read_csv(behavior/'group_cycles.csv')
    for key,part in cycles.groupby(['rule_group','stage','displayed_cycle']):
        row=group[(group.condition==key[0])&(group.stage==key[1])&(group.displayed_cycle==key[2])].iloc[0]
        assert int(row.n)==len(part)
        np.testing.assert_allclose([row['mean'],row.se],[part.rt_ms.mean(),part.rt_ms.std(ddof=1)/np.sqrt(len(part))],rtol=2e-9)
    lme=behavior/'lme'
    lmevalidation=json.loads((lme/'validation.json').read_text())
    assert lmevalidation['status']=='passed' and lmevalidation['independent_covariance_and_contrast_checks']
    assert lmevalidation['fit_script_sha256']==digest(Path(__file__).with_name('fit_fmri_rt_lme.R'))
    assert lmevalidation['cleaned_input_sha256']==digest(behavior/'cleaned_trials.csv')
    assert lmevalidation['analysis_spec_sha256']==digest(lme/'analysis_spec.json')
    for policy in ['primary','min100']:
        status=json.loads((lme/f'{policy}_status.json').read_text())
        assert status['status']=='completed' and status['n_subjects']==20 and status['design_rank']==20
        coefficients=pd.read_csv(lme/f'{policy}_fixed_coefficients.csv').set_index('term')['Value']
        covariance=pd.read_csv(lme/f'{policy}_fixed_covariance.csv',index_col=0).loc[coefficients.index,coefficients.index]
        weights=pd.read_csv(lme/f'{policy}_contrast_weights.csv').set_index('contrast').loc[:,coefficients.index]
        contrasts=pd.read_csv(lme/f'{policy}_contrasts.csv').set_index('contrast').loc[weights.index]
        x=weights.to_numpy()@coefficients.to_numpy()
        se=np.sqrt(np.einsum('ij,jk,ik->i',weights,covariance,weights))
        np.testing.assert_allclose(contrasts.log_difference,x,rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(contrasts.se,se,rtol=2e-9,atol=1e-11)
        np.testing.assert_allclose(contrasts.q_bh_24,independent_bh(2*stats.t.sf(abs(x/se),19)),rtol=2e-9,atol=1e-11)
    error_models=validate_error_mixed_models(root)
    # Resolve all local TeX inputs/graphics/citations independently of the frozen
    # validator's expectations about the earlier manuscript layout and counts.
    sources=list((root/'manuscript').rglob('*.tex'))
    for tex in sources:
        text=re.sub(r'(?<!\\)%[^\n]*','',tex.read_text())
        for name in re.findall(r'\\input\{([^}]+)\}',text):
            target=root/'manuscript'/name
            assert target.is_file() or target.with_suffix('.tex').is_file(),(tex,name)
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
            assert (root/'manuscript'/name).is_file(),(tex,name)
    if a.historical:
        # Preserve source-of-record numerical checks while current text and RT
        # metadata are validated above rather than against the old 42-person plot.
        modulepath=a.source_root/'code/glm_volume/validate_manuscript.py'
        sys.path.insert(0,str(modulepath.parent))
        spec=importlib.util.spec_from_file_location('historical_validation',modulepath)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        subjects,_=mod.validate_float32_roi_sources();mod.validate_first_level_qc(subjects)
        mask,common=mod.validate_mask_and_subject_maps(subjects)
        mod.validate_group_inference(mask,common);mod.validate_sensitivity_analysis()
        mod.validate_reported_censored_products()
    record=dict(status='passed',all_model_rows=len(results),historical_reproduced_rows=1728,
                direct_stage_tests=144,rt_subjects=20,raw_runs_verified=160,
                input_manifests_verified=True,historical_checks=a.historical,
                lme_policies_verified=2,lme_contrasts_each=24)
    if error_models is not None:
        record['error_mixed_models_verified']=2
        record['error_contrasts_each']=24
        record['error_bootstrap_requested']=error_models['bootstrap_requested']
    (out/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
