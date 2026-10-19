"""Validate A7 from sidecars, nuisance inputs, saved maps and participant values."""
from slicetime_common import *
from imaging_validate import assert_close,check_summary
from nilearn.glm.first_level import make_first_level_design_matrix
import warnings


def main():
    t=ImagingTask('validation');assert nilearn.__version__=='0.13.1'
    for name in ['metadata','controls','first_level','roi','voxelwise_inputs','voxelwise']:
        assert json.loads(t.record(OUT/name/'validation.json').read_text())['passed']
    controls=pd.read_csv(t.record(OUT/'controls/reproduction_checks.csv'))
    assert len(controls)==2 and controls.max_difference.eq(0).all() and controls.n_values.sum()==240
    for subject,run in [('sub-001',1),('sub-020',6)]:
        folder=OUT/'first_level/runs'/subject/f'run-{run:02d}'/'reference'
        q=json.loads(t.record(folder/'qc.json').read_text());assert q['frame_shift']==0.
        design=pd.read_csv(t.record(folder/'design_matrix.tsv'),sep='\t')
        assert_close(design.frame_time,np.arange(200)*2.)
    sr=pd.read_csv(t.record(OUT/'roi/stage_results.csv'));sv=pd.read_csv(t.record(OUT/'roi/participant_stage_values.csv'))
    cv=pd.read_csv(t.record(OUT/'roi/participant_component_changes.csv'));cr=pd.read_csv(t.record(OUT/'roi/component_balance_results.csv'))
    n=check_summary(sv,sr,['cohort','model','policy','stage','roi'])
    long=cv.melt(id_vars=['cohort','model','policy','subject','roi','stage'],value_vars=['structured','random','condition_mean','did_half'],var_name='component',value_name='value')
    n+=check_summary(long,cr,['cohort','model','policy','stage','roi','component'])
    for family in ['stage','joint','component']:
        table=cr if family=='component' else sr
        keys=['cohort','model','policy']+(['stage','component'] if family=='component' else ['stage'] if family=='stage' else [])
        col='q_joint_stages_192' if family=='joint' else 'q';slots=192 if family=='joint' else 48
        for _,g in table.groupby(keys):
            assert len(g)==slots;assert_close(g[col],stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    for _,g in cr[cr.stage.eq('early')&cr.roi.isin(HEADLINE)].groupby(['cohort','model','policy','component']):
        assert len(g)==4;assert_close(g.q_headline_early_4,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    assert_close(cv.condition_mean,(cv.structured+cv.random)/2);assert_close(cv.did_half,cv.structured-cv.random)
    z=cv.merge(sv,on=['cohort','model','policy','subject','roi','stage'],validate='one_to_one');assert_close(2*z.did_half,z.value)
    direct=pd.read_csv(t.record(OUT/'roi/direct_phase_results.csv'));pairs=[]
    for cohort in ['N20','N22']:
        v=sv[sv.cohort.eq(cohort)].pivot(index=['model','policy','subject','roi'],columns='stage',values='value')
        for a,b in [('early','middle'),('early','late'),('middle','late')]:
            d=(v[a]-v[b]).rename('value').reset_index();d['pair']=a+'_minus_'+b;d['cohort']=cohort;pairs.append(d)
    n+=check_summary(pd.concat(pairs),direct,['cohort','model','policy','pair','roi'])
    for _,g in direct.groupby(['cohort','model','policy']):
        assert len(g)==144;assert_close(g.q_three_pairs_144,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    for _,g in direct.groupby(['cohort','model','policy','pair']):
        assert len(g)==48;assert_close(g.q_within_pair_48,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    phase=pd.read_csv(t.record(OUT/'roi/participant_phase_components.csv'));pr=pd.read_csv(t.record(OUT/'roi/phase_component_results.csv'))
    pl=phase.melt(id_vars=['cohort','model','policy','subject','roi','stage'],value_vars=['structured','random','sequence_minus_random'],var_name='component',value_name='value')
    n+=check_summary(pl,pr,['cohort','model','policy','stage','roi','component'])
    for _,g in pr.groupby(['cohort','model','policy','stage','component']):
        assert len(g)==48;assert_close(g.q,stats.false_discovery_control(g.p.fillna(1).to_numpy(),method='bh'))
    qc=pd.read_csv(t.record(OUT/'first_level/design_qc.csv'));assert len(qc)==440
    assert qc['rank'].eq(qc.n_columns).all() and qc.residual_df.ge(20).all() and qc.n_rows.eq(200).all()
    assert np.isfinite(qc.nuisance_adjusted_vif).all()
    values=pd.read_csv(t.record(OUT/'first_level/condition_values.csv')).set_index(['subject','run','condition','roi'])
    meta=pd.read_csv(t.record(OUT/'metadata/participant_run_slice_timing.csv')).set_index(['subject','run']);assert len(meta)==176
    ref=nib.load(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz');parcels={r:image(ROOT/'resources/roi_masks'/f'{r}.nii.gz',ref,True)>0 for r in ROIS}
    inputs={};output_checks=0;map_slots=0;design_checks=0;max_design_difference=0.;coverage=[]
    for subject in N22:
        support=np.logical_and.reduce([image(paths(subject,r)[1],ref,True)>0 for r in range(1,9)])
        masks={roi:p&support for roi,p in parcels.items()}
        eligible={roi:masks[roi].any() and masks[roi].sum()/p.sum()+1e-12>=.8 for roi,p in parcels.items()}
        for roi,p in parcels.items():coverage.append(dict(subject=subject,roi=roi,n_roi=int(p.sum()),n_observed=int(masks[roi].sum()),fraction=float(masks[roi].sum()/p.sum()),eligible=bool(eligible[roi])))
        runmaps={}
        for run in range(1,9):
            folder=OUT/'first_level/runs'/subject/f'run-{run:02d}'/'slicetime_aligned';q=json.loads(t.record(folder/'qc.json').read_text());inputs.update(q['inputs'])
            assert q['producer']==digest(ROOT/'code/review_r1/slicetime_fits.py') and q['common']==digest(ROOT/'code/review_r1/slicetime_common.py')
            for name,sha in q['outputs'].items():assert digest(folder/name)==sha;output_checks+=1
            bold,mask,conf,_=paths(subject,run);sidecar=bold.with_name(bold.name.replace('.nii.gz','.json'));source=json.loads(sidecar.read_text())
            start=float(source['StartTime']);tr=float(source['RepetitionTime']);assert q['frame_shift']==start and q['StartTime']==start and q['RepetitionTime']==tr
            assert_close([meta.loc[(subject,run),'StartTime'],meta.loc[(subject,run),'RepetitionTime']],[start,tr]);assert digest(sidecar)==meta.loc[(subject,run),'sha256']
            design=pd.read_csv(folder/'design_matrix.tsv',sep='\t').set_index('frame_time');assert_close(design.index,np.arange(200)*tr+start)
            e=pd.read_csv(event_path(subject,run),sep='\t');e.trial_type=e.trial_type.str.extract('(Ran|Str[12])$')[0];c=pd.read_csv(conf,sep='\t')[CONFOUNDS]
            assert q['confound_columns']==CONFOUNDS and not c.isna().any().any()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');fresh=make_first_level_design_matrix(np.arange(200)*tr+start,e,hrf_model='glover',drift_model='cosine',high_pass=1/128,add_regs=c.to_numpy(),add_reg_names=list(c))
            assert list(design)==list(fresh);assert_close(design,fresh,atol=1e-12,rtol=1e-12);design_checks+=1
            max_design_difference=max(max_design_difference,float(np.max(abs(design.to_numpy()-fresh.to_numpy()))))
            runmaps[run]={}
            for condition in conditions(run):
                path=folder/f'{condition}_effect_size.nii.gz';im=nib.load(path);assert im.get_data_dtype()==np.dtype('float32') and im.dataobj.slope==1. and im.dataobj.inter==0.
                array=image(path,ref);runmaps[run][condition]=array
                for roi in ROIS:
                    expected=array[masks[roi]].mean() if eligible[roi] else np.nan
                    assert_close(values.loc[(subject,run,condition,roi),'value'],expected);map_slots+=1
        baseline=sum(runmaps[r]['Str1']-runmaps[r]['Ran'] for r in [1,2])
        for i,stage in enumerate(STAGES):
            expected=baseline if i==0 else sum(runmaps[r]['Str2']-runmaps[r]['Ran'] for r in [2*i+1,2*i+2])-baseline
            saved=image(OUT/'voxelwise_inputs/slicetime_aligned'/subject/f'{stage}.nii.gz',ref)
            np.testing.assert_array_equal(saved,expected.astype('float32'))
        pooled=sum((runmaps[r]['Str2']-runmaps[r]['Str1'])/4 for r in [3,4,5,6])
        np.testing.assert_array_equal(image(OUT/'voxelwise_inputs/slicetime_aligned'/subject/'pooled_r36_Str2_minus_Str1.nii.gz',ref),pooled.astype('float32'))
    csv(coverage,t.path/'participant_roi_coverage.csv')
    for path,record in inputs.items():assert digest(path)==record['sha256'],path
    t.inputs.update(inputs)
    csv([dict(path=path,**record) for path,record in inputs.items()],t.path/'verified_first_level_input_hashes.csv')
    perm=pd.read_csv(t.record(OUT/'voxelwise/permutation_summary.csv'));assert len(perm)==10 and perm.n_perm.eq(10000).all() and perm.seed.eq(20261018).all() and perm.n_mask_voxels.eq(58550).all()
    assert perm[perm.contrast.isin(STAGES)].alpha.eq(.0125).all() and perm[~perm.contrast.isin(STAGES)].alpha.eq(.05).all()
    for row in perm.itertuples():
        folder=OUT/'voxelwise'/row.cohort/'slicetime_aligned'/row.contrast;q=json.loads((folder/'validation.json').read_text());assert q['passed'] and q['null_FWE_exact'] and q['native_observed_TFCE_checked']
        for name,sha in q['output_sha256'].items():assert digest(folder/name)==sha;output_checks+=1
        signed=np.asarray(nib.load(folder/'signed_tfce_corrected_t.nii.gz').dataobj)
        assert int((signed>0).sum())==row.positive_voxels and int((signed<0).sum())==row.negative_voxels
    baseline=json.loads(t.record('/tmp/r2-r2-slicetime/protected_before.json').read_text());changes=[]
    for path,r in baseline.items():
        if not Path(path).is_file() or digest(path)!=r['sha256']:changes.append(path)
    writej(t.path/'preservation_check.json',dict(n_files=len(baseline),changed=changes,passed=not changes))
    assert not changes,changes
    checks=dict(passed=True,n_arithmetic_checks=n,n_shifted_designs_reconstructed=design_checks,max_design_difference=max_design_difference,n_observed80_map_slots_checked=map_slots,n_subject_maps_checked=110,n_unique_first_level_inputs=len(inputs),n_output_hashes_verified=output_checks,n_protected_files_unchanged=len(baseline),stage_BH48=True,joint_BH192=True,component_BH48=True,headline_BH4=True,direct_BH144=True,n_permutation_analyses=10,n_permutations_each=10000,seed=20261018,min_residual_df=int(qc.residual_df.min()))
    t.finish('# A7 validation\n\nRegenerated all 176 shifted designs from JSON timing, corrected events and original nuisance TSVs; checked saved frame times, every observed80 map extraction including coverage-ineligible NaNs, and all 110 voxelwise subject maps. Recomputed group estimates, pointwise t intervals, p values and all specified BH families from participant values, including component identities and phase components. Rehashed every unique first-level input and saved fit/permutation output. Per-analysis voxelwise validation independently checks observed t, native TFCE, exact empirical FWE and map casts. Recounted signed corrected voxels. A before/after preservation hash baseline checks existing code/docs/manuscript/resources and specified updated/review results without following directory symlinks. No manuscript/UI/build or scientific identification is validated.\n\n'+json.dumps(checks,indent=2),checks)

if __name__=='__main__':main()
