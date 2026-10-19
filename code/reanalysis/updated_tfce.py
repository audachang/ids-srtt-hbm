#!/usr/bin/env python3
"""Matched N20/N22 two-sided 10,000-permutation voxelwise analyses."""
from updated_common import *
import argparse
import importlib
import concurrent.futures
import multiprocessing
import time
from scipy.ndimage import label as connected_label, generate_binary_structure
from nilearn.glm.second_level import non_parametric_inference
from tfce_direct import union_tfce as sparse_tfce, direct_chunk as sparse_chunk, LIBRARY


def jobs():
    result=[]
    for cohort in ['N20','N22']:
        for model in ['reference','censored']:
            for contrast in STAGES:
                result.append((cohort,model,contrast,.05/4))
        for model in ['reference','censored','errors_omissions','rt_shared_modulator']:
            result.append((cohort,model,'pooled_r36_Str2_minus_Str1',.05))
    return result


def run(job):
    cohort,model,contrast,alpha=job
    name=f'voxelwise/{cohort}/{model}/{contrast}'
    t=Task(name)
    marker=t.path/'validation.json'
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['passed']
        producer=json.loads((t.path/'input_manifest.json').read_text())['sources'][str(Path(__file__).resolve())]
        assert producer==digest(__file__)
        for file,sha in saved['output_sha256'].items():assert digest(t.path/file)==sha
        return saved['summary']
    backend=t.record(BASE/'permutation_backend_validation_direct/validation.json')
    assert json.loads(backend.read_text())['passed']
    for path in [*[Path(__file__).with_name(name) for name in ['tfce_sparse.cpp','tfce_sparse.py','libtfce_sparse.so','tfce_union.cpp','tfce_union.py','tfce_direct.py']],LIBRARY]:t.record(path)
    mask=t.record(BASE/'voxelwise_inputs/common_N22_mask.nii.gz')
    subjects=N20 if cohort=='N20' else N22
    paths20=[t.record(BASE/'voxelwise_inputs'/model/s/f'{contrast}.nii.gz') for s in subjects]
    os.environ['SRTT_TFCE_MASK']=str(mask);os.environ['SRTT_TFCE_STRICT']='1'
    pl=importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
    sl=importlib.import_module('nilearn.glm.second_level.second_level')
    old_chunk,old_tfce,old_call=pl._permuted_ols_on_chunk,pl.calculate_tfce,sl.permuted_ols
    captured={}
    def capture(*args,**kwargs):
        result=old_call(*args,**kwargs);captured.update(result)
        np.savez_compressed(t.path/'raw_permutation_arrays.npz',**result)
        return result
    pl._permuted_ols_on_chunk=sparse_chunk;pl.calculate_tfce=sparse_tfce;sl.permuted_ols=capture
    design=pd.DataFrame({'intercept':np.ones(len(subjects))});csv(design,t.path/'design_matrix.csv')
    started=time.monotonic()
    try:
        result=non_parametric_inference([str(p) for p in paths20],design_matrix=design,second_level_contrast='intercept',
            mask=str(mask),model_intercept=True,n_perm=10000,two_sided_test=True,random_state=20261018,n_jobs=8,tfce=True,verbose=0)
    finally:
        pl._permuted_ols_on_chunk,pl.calculate_tfce,sl.permuted_ols=old_chunk,old_tfce,old_call
    reference=nib.load(mask);m=np.asarray(reference.dataobj)>0
    for key,im in result.items():save32(np.asarray(im.dataobj),reference,t.path/f'{key}.nii.gz')
    h0=captured['h0_max_tfce'];assert h0.shape==(1,10000)
    rank=np.searchsorted(np.sort(h0[0]),np.abs(captured['tfce'][0]),side='left')
    logp=-np.log10((10001-rank)/10001)
    np.testing.assert_allclose(logp,captured['logp_max_tfce'][0],atol=1e-12,rtol=1e-12)
    stat=np.asarray(result['t'].dataobj);p=np.asarray(result['logp_max_tfce'].dataobj)
    np.testing.assert_allclose(p[m],logp,atol=1e-12,rtol=1e-12)
    data=np.stack([np.asarray(nib.load(path).dataobj,dtype=float)[m] for path in paths20])
    independent=stats.ttest_1samp(data,0,axis=0).statistic
    np.testing.assert_allclose(stat[m],independent,atol=1e-8,rtol=1e-8)
    keep=p>=-np.log10(alpha);signed=np.where(keep,stat,0.)
    save32(signed,reference,t.path/'signed_tfce_corrected_t.nii.gz')
    mean=data.mean(axis=0);se=data.std(axis=0,ddof=1)/np.sqrt(len(subjects));half=stats.t.ppf(.975,len(subjects)-1)*se
    for key,values in [('mean_effect',mean),('standard_error',se),('pointwise_ci_low',mean-half),('pointwise_ci_high',mean+half)]:
        full=np.zeros(m.shape);full[m]=values;save32(full,reference,t.path/f'{key}.nii.gz')
    components=[]
    for sign in [-1,1]:
        labels,n=connected_label(sign*signed>0,structure=generate_binary_structure(3,3))
        for index in range(1,n+1):
            points=np.argwhere(labels==index);values=stat[labels==index];peak=points[np.argmax(sign*values)]
            xyz=nib.affines.apply_affine(reference.affine,peak)
            components.append(dict(sign=sign,n_voxels=len(points),peak_t=values[np.argmax(sign*values)],peak_x=xyz[0],peak_y=xyz[1],peak_z=xyz[2]))
    csv(pd.DataFrame(components,columns=['sign','n_voxels','peak_t','peak_x','peak_y','peak_z']),t.path/'components.csv')
    summary=dict(cohort=cohort,model=model,contrast=contrast,n_subjects=len(subjects),alpha=alpha,n_perm=10000,
                  positive_voxels=int((signed>0).sum()),negative_voxels=int((signed<0).sum()),n_mask_voxels=int(m.sum()),
                  elapsed_seconds=time.monotonic()-started)
    t.finish(f'# Updated voxelwise inference\n\n{cohort}, {model}, {contrast}. 10,000 two-sided sign-flip permutations; seed 20261018; eight chunks; validated component-tree TFCE and native one-sample sign-flip logic. Corrected threshold {alpha:g}. Historical stage contrasts retain Bonferroni across their four stages (.05/4) separately within each cohort/model. The novel pooled within-run contrast has its own single-contrast .05 family per cohort/model. There is no project-wide correction across all cohorts/models/families. Both samples share the N22 intersection mask. Signed effect orientation follows contrast weights; positive pooled = Str2 > Str1.\n\n{summary["positive_voxels"]} positive and {summary["negative_voxels"]} negative corrected voxels. Component sizes are descriptive 26-connected sets, not cluster-level inference. All images are unscaled float32. Raw null distributions are saved; FWE p values are independently reconstructed and group t values independently recomputed from participant maps. Exported effect CIs are pointwise and are not selection-adjusted intervals for chosen peaks.\n',
             dict(passed=True,summary=summary,native_backend_prevalidated=True,null_fwe_reconstructed=True,
                  independent_group_t_checked=True,all_image_outputs_unscaled_float32=True))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parallel-contrasts',type=int,default=2)
    args=parser.parse_args()
    for name in ['voxelwise_inputs','permutation_backend_validation_direct']:
        assert json.loads((BASE/name/'validation.json').read_text())['passed']
    t=Task('voxelwise');results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_contrasts,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run,job) for job in jobs()]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result);csv(results,t.path/'permutation_summary.csv')
            print(json.dumps(result),flush=True)
    for cohort,model,contrast,_ in jobs():
        folder=t.path/cohort/model/contrast
        t.record(folder/'validation.json');t.record(folder/'input_manifest.json')
    t.finish('# Matched updated-sample voxelwise analyses\n\nTwenty-four complete 10,000-permutation analyses: historical four stages for reference/censored models and novel pooled within-run contrast for reference/censored/error/shared-RT specifications, each in N20 and N22. Mask and model definitions are held fixed across sample comparisons. See each analysis for its explicitly separate multiplicity family and effect-size maps.\n',dict(passed=True,n_analyses=len(results),n_permutations_per_analysis=10000,n_full_permutation_runs=24))


if __name__=='__main__':main()
