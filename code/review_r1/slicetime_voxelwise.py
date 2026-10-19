#!/usr/bin/env python3
"""A7 fixed-mask slice-timing-aligned TFCE using the existing validated backend."""
from slicetime_common import *
import argparse
import importlib
from scipy.ndimage import label, generate_binary_structure
from nilearn.glm.second_level import non_parametric_inference
from tfce_direct import union_tfce, direct_chunk, LIBRARY
from tfce_sparse import NATIVE_TFCE


def prepare():
    assert json.loads((OUT/'first_level/validation.json').read_text())['passed']
    t=ImagingTask('voxelwise_inputs');mask=t.record(EXISTING/'voxelwise_inputs/common_N22_mask.nii.gz')
    ref=nib.load(mask);manifest=[]
    for subject in N22:
        byrun={}
        for run in range(1,9):
            byrun[run]={c:image(t.record(OUT/'first_level/runs'/subject/f'run-{run:02d}'/'slicetime_aligned'/f'{c}_effect_size.nii.gz'),ref) for c in conditions(run)}
        baseline=sum(byrun[r]['Str1']-byrun[r]['Ran'] for r in [1,2])
        out=t.path/'slicetime_aligned'/subject;out.mkdir(parents=True,exist_ok=True)
        for i,stage in enumerate(STAGES):
            value=baseline if i==0 else sum(byrun[r]['Str2']-byrun[r]['Ran'] for r in [2*i+1,2*i+2])-baseline
            path=out/f'{stage}.nii.gz';save32(value,ref,path)
            manifest.append(dict(subject=subject,model='slicetime_aligned',contrast=stage,path=str(path),sha256=digest(path)))
        pooled=sum((byrun[r]['Str2']-byrun[r]['Str1'])/4 for r in [3,4,5,6])
        path=out/'pooled_r36_Str2_minus_Str1.nii.gz';save32(pooled,ref,path)
        manifest.append(dict(subject=subject,model='slicetime_aligned',contrast='pooled_r36_Str2_minus_Str1',path=str(path),sha256=digest(path)))
    csv(manifest,t.path/'subject_map_manifest.csv')
    t.finish('# Slice-timing-aligned voxelwise inputs\n\nFour summed stage maps and equal-run-mean pooled Str2−Str1 across runs3–6 for all 22 participants. The N20 sample is a subset. All maps are explicitly unscaled float32, derived from reference-space condition maps, with native-to-reference interpolation already applied at first level. Both cohorts use the unchanged existing N22 search mask. No statistical selection is performed.',dict(passed=True,n_maps=len(manifest),n_mask_voxels=int((np.asarray(ref.dataobj)>0).sum())))


def run_one(cohort,contrast):
    t=ImagingTask(f'voxelwise/{cohort}/slicetime_aligned/{contrast}')
    marker=t.path/'validation.json'
    if marker.exists():
        q=json.loads(marker.read_text());assert q['passed']
        for name,sha in q['output_sha256'].items():assert digest(t.path/name)==sha
        return q['summary']
    for name in ['tfce_direct.py','tfce_sparse.py','tfce_union.py','tfce_sparse.cpp','tfce_union.cpp','libtfce_sparse.so','libtfce_union.so','updated_tfce.py','postprocess_updated_tfce.py']:
        t.record(ROOT/'code/reanalysis'/name)
    q=json.loads(t.record(EXISTING/'permutation_backend_validation_direct/validation.json').read_text());assert q['passed']
    mask=t.record(EXISTING/'voxelwise_inputs/common_N22_mask.nii.gz');ref=nib.load(mask);m=np.asarray(ref.dataobj)>0
    subjects=N20 if cohort=='N20' else N22
    maps=[t.record(OUT/'voxelwise_inputs/slicetime_aligned'/s/f'{contrast}.nii.gz') for s in subjects]
    alpha=.05/4 if contrast in STAGES else .05
    os.environ['SRTT_TFCE_MASK']=str(mask);os.environ['SRTT_TFCE_STRICT']='1'
    design=pd.DataFrame({'intercept':np.ones(len(subjects))});csv(design,t.path/'design_matrix.csv')
    rawpath=t.path/'raw_permutation_arrays.npz';permutation_seconds=None
    if not rawpath.exists():
        pl=importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
        sl=importlib.import_module('nilearn.glm.second_level.second_level')
        old_chunk,old_tfce,old_call=pl._permuted_ols_on_chunk,pl.calculate_tfce,sl.permuted_ols
        def capture(*args,**kwargs):
            result=old_call(*args,**kwargs)
            np.savez_compressed(rawpath,**result)
            return result
        pl._permuted_ols_on_chunk=direct_chunk;pl.calculate_tfce=union_tfce;sl.permuted_ols=capture
        start=time.monotonic()
        try:
            non_parametric_inference([str(p) for p in maps],design_matrix=design,second_level_contrast='intercept',
                mask=str(mask),model_intercept=True,n_perm=10000,two_sided_test=True,random_state=20261018,n_jobs=8,tfce=True,verbose=0)
        finally:
            pl._permuted_ols_on_chunk,pl.calculate_tfce,sl.permuted_ols=old_chunk,old_tfce,old_call
        permutation_seconds=time.monotonic()-start
        writej(t.path/'timing.json',dict(permutation_seconds=permutation_seconds))
    elif (t.path/'timing.json').exists():
        permutation_seconds=json.loads((t.path/'timing.json').read_text())['permutation_seconds']
    raw=np.load(rawpath)
    for key in ['h0_max_t','h0_max_tfce']:
        assert raw[key].shape==(1,10000) and np.isfinite(raw[key]).all()
    for key,score in [('logp_max_t','t'),('logp_max_tfce','tfce')]:
        rank=np.searchsorted(np.sort(raw['h0_max_'+score][0]),abs(raw[score][0]),side='left')
        logp=-np.log10((10001-rank)/10001)
        np.testing.assert_array_equal(logp,raw[key][0])
    field=np.zeros(m.shape+(1,),dtype=raw['t'].dtype);field[...,0][m]=raw['t'][0]
    native=NATIVE_TFCE(field,generate_binary_structure(3,1))[...,0][m]
    np.testing.assert_allclose(native,raw['tfce'][0],atol=1e-9,rtol=1e-12)
    data=np.stack([np.asarray(nib.load(p).dataobj,dtype=float)[m] for p in maps])
    independent=stats.ttest_1samp(data,0,axis=0).statistic
    np.testing.assert_allclose(independent,raw['t'][0],atol=5e-6,rtol=2e-6)
    for key in ['t','tfce','logp_max_t','logp_max_tfce']:
        full=np.zeros(m.shape);full[m]=raw[key][0];save32(full,ref,t.path/f'{key}.nii.gz')
        np.testing.assert_array_equal(np.asarray(nib.load(t.path/f'{key}.nii.gz').dataobj)[m],raw[key][0].astype('float32'))
    keep=raw['logp_max_tfce'][0]>=-np.log10(alpha)
    signed=np.zeros(m.shape);signed[m]=np.where(keep,raw['t'][0],0)
    save32(signed,ref,t.path/'signed_tfce_corrected_t.nii.gz')
    stat=np.zeros(m.shape);stat[m]=raw['t'][0]
    mean=data.mean(axis=0);se=data.std(axis=0,ddof=1)/np.sqrt(len(subjects));half=stats.t.ppf(.975,len(subjects)-1)*se
    for name,v in [('mean_effect',mean),('standard_error',se),('pointwise_ci_low',mean-half),('pointwise_ci_high',mean+half)]:
        full=np.zeros(m.shape);full[m]=v;save32(full,ref,t.path/f'{name}.nii.gz')
    components=[]
    for sign in [-1,1]:
        lab,n=label(sign*signed>0,generate_binary_structure(3,3))
        for idx in range(1,n+1):
            points=np.argwhere(lab==idx);values=stat[lab==idx];peak=points[np.argmax(sign*values)];xyz=nib.affines.apply_affine(ref.affine,peak)
            full_logp=np.zeros(m.shape);full_logp[m]=raw['logp_max_tfce'][0]
            components.append(dict(sign=sign,n_voxels=len(points),peak_t=float(values[np.argmax(sign*values)]),
                                   peak_x=float(xyz[0]),peak_y=float(xyz[1]),peak_z=float(xyz[2]),
                                   peak_tfce_fwe_p=float(10**(-full_logp[tuple(peak)]))))
    csv(pd.DataFrame(components,columns=['sign','n_voxels','peak_t','peak_x','peak_y','peak_z','peak_tfce_fwe_p']),t.path/'components.csv')
    summary=dict(cohort=cohort,model='slicetime_aligned',contrast=contrast,n_subjects=len(subjects),alpha=alpha,n_perm=10000,seed=20261018,
                 positive_voxels=int((signed>0).sum()),negative_voxels=int((signed<0).sum()),n_mask_voxels=int(m.sum()),
                 permutation_seconds=permutation_seconds)
    t.finish(f'# A7 slice-timing-aligned voxelwise inference\n\n{cohort}, {contrast}. Same settings as existing reference inference: 10,000 two-sided sign-flip permutations; seed 20261018; eight chunks; fixed 58,550-voxel mask. The A7-specific same-seed instruction takes precedence over the general R1 seed. The existing validated native one-sample/direct component-tree backend is imported unchanged (hashes recorded). TFCE uses E=.5/H=2, 100 native Nilearn thresholds with its integration/sign rules, and six-neighbor connectivity. Descriptive components use 26-neighbor connectivity and are not cluster-level inference. The four summed historical stages have FWE .05/4 within each cohort/model; pooled runs3–6 Str2−Str1 is a separate FWE .05 family. Positive pooled means Str2>Str1.\n\nAll null maxima are preserved. Both max-t and TFCE FWE log-p are independently reconstructed exactly from raw arrays. Observed TFCE is checked against native Nilearn; t is independently recomputed from participant maps within the existing float32-normalization tolerance (atol5e-6/rtol2e-6). Exported images exactly match raw arrays cast to unscaled float32. Effect CIs are pointwise and not adjusted for selecting peaks. Detection-count differences from unaligned reference are not tests of model differences.\n\n'+json.dumps(summary,indent=2),
             dict(passed=True,summary=summary,null_FWE_exact=True,native_observed_TFCE_checked=True,
                  independent_t_max_abs_difference=float(abs(independent-raw['t'][0]).max()),all_images_unscaled_float32=True))
    return summary


def inference():
    assert json.loads((OUT/'voxelwise_inputs/validation.json').read_text())['passed']
    t=ImagingTask('voxelwise');summaries=[]
    for cohort in ['N20','N22']:
        for contrast in STAGES+['pooled_r36_Str2_minus_Str1']:
            summaries.append(run_one(cohort,contrast));csv(summaries,t.path/'permutation_summary.csv')
            print(json.dumps(summaries[-1]),flush=True)
    old=pd.read_csv(t.record(EXISTING/'voxelwise/permutation_summary.csv'));old=old[old.model.eq('reference')]
    new=pd.DataFrame(summaries)
    compare=new.merge(old,on=['cohort','contrast','n_subjects','alpha','n_perm','n_mask_voxels'],suffixes=('_aligned','_reference'),validate='one_to_one')
    csv(compare,t.path/'aligned_vs_reference_counts.csv')
    for cohort in ['N20','N22']:
        for contrast in STAGES+['pooled_r36_Str2_minus_Str1']:
            t.record(t.path/cohort/'slicetime_aligned'/contrast/'validation.json')
            t.record(t.path/cohort/'slicetime_aligned'/contrast/'input_manifest.json')
    t.finish('# A7 complete slice-timing-aligned rerun\n\nTen complete matched-mask analyses (four historical stages plus the pooled within-run contrast, each in N20/N22), 10,000 two-sided permutations each. N20 remains supplement-only. See per-analysis reports for exact multiplicity families and validated raw nulls. Matched unaligned reference counts are provided descriptively; count differences do not test model differences.\n\n'+new.to_string(index=False),dict(passed=True,n_analyses=len(summaries),n_permutations_each=10000,summary=summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
    {'prepare':prepare,'run':inference}[a.mode]()
