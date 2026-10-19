#!/usr/bin/env python3
"""Validate completed R1 imaging artifacts against participant observations and inputs."""
from imaging_common import *
import argparse


def assert_close(a,b,atol=2e-10,rtol=2e-9):
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,equal_nan=True)


def check_summary(participant,results,keys,value='value'):
    groups=participant.groupby(keys,dropna=False)
    result=results.set_index(keys)
    assert len(groups)==len(result)
    checks=0
    for key,g in groups:
        z=np.asarray(g[value],float);z=z[np.isfinite(z)];row=result.loc[key];n=len(z)
        assert row['n']==n and row['df']==n-1
        if not n:assert np.isnan(row['mean']);continue
        assert_close(row['mean'],z.mean());checks+=1
        if n>=2 and z.std(ddof=1)>0:
            se=z.std(ddof=1)/np.sqrt(n);half=stats.t.ppf(.975,n-1)*se
            test=stats.ttest_1samp(z,0.)
            assert_close([row['se'],row['ci_low'],row['ci_high'],row['t'],row['p']],
                         [se,z.mean()-half,z.mean()+half,test.statistic,test.pvalue]);checks+=5
    return checks


def main():
    t=ImagingTask('validation');assert nilearn.__version__=='0.13.1'
    for name in ['controls','first_level','event_motion_diagnostics','roi','voxelwise_runtime_gate','voxelwise_inputs','voxelwise']:
        p=t.record(OUT/name/'validation.json');q=json.loads(p.read_text());assert q['passed'],name
    roi=OUT/'roi'
    sv=pd.read_csv(t.record(roi/'participant_stage_values.csv'));sr=pd.read_csv(t.record(roi/'stage_results.csv'))
    cv=pd.read_csv(t.record(roi/'participant_component_changes.csv'));cr=pd.read_csv(t.record(roi/'component_balance_results.csv'))
    n_arithmetic=check_summary(sv,sr,['cohort','model','policy','stage','roi'])
    long=cv.melt(id_vars=['cohort','model','policy','subject','roi','stage'],value_vars=['structured','random','condition_mean','did_half'],var_name='component',value_name='value')
    n_arithmetic+=check_summary(long,cr,['cohort','model','policy','stage','roi','component'])
    for family in ['stage','joint','component']:
        table=cr if family=='component' else sr
        keys=['cohort','model','policy']+(['stage','component'] if family=='component' else ['stage'] if family=='stage' else [])
        col='q_joint_stages_192' if family=='joint' else 'q';n=192 if family=='joint' else 48
        for _,g in table.groupby(keys):
            assert len(g)==n;assert_close(g[col],stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    for _,g in cr[cr.stage.eq('early')&cr.roi.isin(HEADLINE)].groupby(['cohort','model','policy','component']):
        assert len(g)==4;assert_close(g.q_headline_early_4,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    assert_close(cv.condition_mean,(cv.structured+cv.random)/2)
    assert_close(cv.did_half,cv.structured-cv.random)
    z=cv.merge(sv,on=['cohort','model','policy','subject','roi','stage'],validate='one_to_one');assert_close(2*z.did_half,z.value)
    direct=pd.read_csv(t.record(roi/'direct_phase_results.csv'));pairs=[]
    for cohort in ['N20','N22']:
        values=sv[sv.cohort.eq(cohort)].pivot(index=['model','policy','subject','roi'],columns='stage',values='value')
        for a,b in [('early','middle'),('early','late'),('middle','late')]:
            d=(values[a]-values[b]).rename('value').reset_index();d['pair']=a+'_minus_'+b;d['cohort']=cohort;pairs.append(d)
    n_arithmetic+=check_summary(pd.concat(pairs),direct,['cohort','model','policy','pair','roi'])
    for _,g in direct.groupby(['cohort','model','policy']):
        assert len(g)==144;assert_close(g.q_three_pairs_144,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    for _,g in direct.groupby(['cohort','model','policy','pair']):
        assert len(g)==48;assert_close(g.q_within_pair_48,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    qc=pd.read_csv(t.record(OUT/'first_level/design_qc.csv'))
    assert qc['rank'].eq(qc.n_columns).all() and qc.residual_df.ge(20).all() and qc.n_rows.eq(200).all()
    assert np.isfinite(qc.nuisance_adjusted_vif).all()
    counts=pd.read_csv(t.record(OUT/'event_motion_diagnostics/participant_repeat_counts.csv'))
    manifest_inputs={};n_qc=0;n_output_hashes=0;short_components=[]
    for subject in N22:
        for run in range(1,9):
            for model in NEW_MODELS:
                folder=OUT/'first_level/runs'/subject/f'run-{run:02d}'/model
                q=json.loads((folder/'qc.json').read_text());n_qc+=1
                assert q['producer']==digest(ROOT/'code/review_r1/imaging_fits.py')
                assert q['common']==digest(ROOT/'code/review_r1/imaging_common.py')
                manifest_inputs.update(q['inputs'])
                for name,sha in q['outputs'].items():
                    assert digest(folder/name)==sha;n_output_hashes+=1
                if model in ['motion6_acompcor8','motion12_acompcor8']:
                    assert len(q['acompcor_columns'])== (5 if (subject,run)==('sub-020',5) else 8)
                    if len(q['acompcor_columns'])<8:short_components.append(dict(subject=subject,run=run,model=model,n_components=len(q['acompcor_columns'])))
                if model=='repeat_regressor':
                    raw=pd.read_csv(ROOT/'data/behavioral'/f'{subject[-3:]}_run_{run}_srtt_prob_new.csv')
                    flag=raw.keypress.eq(raw.keypress.shift())&raw.stim_onset.diff().eq(1.)
                    saved=counts[(counts.subject==subject)&(counts.run==run)]
                    assert int(saved.n_repeats.sum())==int(flag.sum())
                    design=pd.read_csv(folder/'design_matrix.tsv',sep='\t');assert 'ImmediateRepeat' in design
                if model=='censored_spike':
                    design=pd.read_csv(folder/'design_matrix.tsv',sep='\t');columns=[v for v in design if v.startswith('censor_volume_')]
                    mask=np.loadtxt(mask_path(subject)).reshape(8,200)[run-1];indices=np.flatnonzero(mask==0)
                    assert len(columns)==len(indices)
                    for i,col in zip(indices,columns):np.testing.assert_array_equal(design[col],np.eye(200)[:,i])
    # Hash each unique actual input once, including BOLDs and all eight-run masks.
    for path,record in manifest_inputs.items():
        assert digest(path)==record['sha256'],path
    csv([dict(path=path,**record) for path,record in manifest_inputs.items()],t.path/'verified_first_level_input_hashes.csv')
    perms=pd.read_csv(t.record(OUT/'voxelwise/permutation_summary.csv'))
    assert len(perms)==10 and perms.n_perm.eq(10000).all() and perms.seed.eq(20261018).all()
    assert perms.n_mask_voxels.eq(58550).all()
    assert perms[perms.contrast.isin(STAGES)].alpha.eq(.0125).all()
    assert perms[~perms.contrast.isin(STAGES)].alpha.eq(.05).all()
    for cohort in ['N20','N22']:
        for contrast in STAGES+['pooled_r36_Str2_minus_Str1']:
            folder=OUT/'voxelwise'/cohort/'censored_spike'/contrast
            q=json.loads((folder/'validation.json').read_text());assert q['passed']
            assert q['null_FWE_exact'] and q['native_observed_TFCE_checked']
            for name,sha in q['output_sha256'].items():assert digest(folder/name)==sha;n_output_hashes+=1
    checks=dict(passed=True,n_arithmetic_checks=n_arithmetic,n_first_level_qc=n_qc,n_unique_first_level_inputs=len(manifest_inputs),
                n_output_hashes_verified=n_output_hashes,stage_families_48=True,joint_families_192=True,direct_families_144=True,
                headline_families_4=True,paired_component_identity=True,D1_repeat_counts_verified=True,
                spike_indicators_verified=True,all_designs_full_rank=True,min_residual_df=int(qc.residual_df.min()),
                n_permutation_analyses=10,aCompCor_short_component_runs=short_components)
    t.finish('# R1 imaging validation\n\nSaved participant values independently reproduce all stage, direct-phase and component estimates, pointwise t CIs and p values. All BH families are independently recomputed with untestable p=1, including 48-slot stage/components,192-slot joint stages,144-slot direct phases and4-slot early headline components. Component identities are checked participant by participant. Every first-level design is full rank with ≥20 residual df; exact spike indicator columns match reconstructed masks; D1 counts match raw trial slots. Every unique first-level input is rehashed and per-run output hashes verified. All ten permutation reports and outputs are checked, with exact FWE/native observed-TFCE checks recorded per analysis. The auditor remains responsible for independent project-wide protected-file comparison; this validator does not substitute for that audit.\n\n'+json.dumps(checks,indent=2),checks)


if __name__=='__main__':main()
