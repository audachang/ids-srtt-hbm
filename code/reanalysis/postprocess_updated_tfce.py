#!/usr/bin/env python3
"""Audit/export saved full permutations using native float32 score precision."""
from updated_common import *
from updated_tfce import jobs
from tfce_sparse import NATIVE_TFCE
from scipy.ndimage import label,generate_binary_structure
import time


def one(job):
    cohort,model,contrast,alpha=job;t=Task(f'voxelwise/{cohort}/{model}/{contrast}')
    rawpath=t.path/'raw_permutation_arrays.npz'
    if not rawpath.exists():return None
    if (t.path/'validation.json').exists():
        old=json.loads((t.path/'validation.json').read_text());assert old['passed'];return old['summary']
    r=np.load(t.record(rawpath));mask=t.record(BASE/'voxelwise_inputs/common_N22_mask.nii.gz')
    ref=nib.load(mask);m=np.asarray(ref.dataobj)>0;n=int(cohort[1:]);subjects=N20 if n==20 else N22
    paths20=[t.record(BASE/'voxelwise_inputs'/model/s/f'{contrast}.nii.gz') for s in subjects]
    t.record(BASE/'permutation_backend_validation_direct/validation.json')
    t.record(Path(__file__).with_name('updated_tfce.py'))
    for key in ['h0_max_tfce','h0_max_t']:assert r[key].shape==(1,10000) and np.isfinite(r[key]).all()
    for key,score in [('logp_max_tfce','tfce'),('logp_max_t','t')]:
        rank=np.searchsorted(np.sort(r['h0_max_'+score][0]),abs(r[score][0]),side='left')
        logp=-np.log10((10001-rank)/10001)
        np.testing.assert_array_equal(logp,r[key][0])
    # Reproduce the native observed TFCE on the original saved score precision.
    field=np.zeros(m.shape+(1,),dtype=r['t'].dtype);field[...,0][m]=r['t'][0]
    expected=NATIVE_TFCE(field,generate_binary_structure(3,1))[...,0][m]
    np.testing.assert_allclose(expected,r['tfce'][0],atol=1e-9,rtol=1e-12)
    data=np.stack([np.asarray(nib.load(p).dataobj,dtype=float)[m] for p in paths20])
    independent=stats.ttest_1samp(data,0,axis=0).statistic
    # Native masker/normalization uses float32; independent SciPy recomputation
    # is float64. The tolerance covers that precision difference only.
    np.testing.assert_allclose(r['t'][0],independent,atol=5e-6,rtol=2e-6)
    stat=np.zeros(m.shape);stat[m]=r['t'][0]
    keep=r['logp_max_tfce'][0]>=-np.log10(alpha)
    signed=np.zeros(m.shape);signed[m]=np.where(keep,r['t'][0],0)
    for key in ['t','tfce','logp_max_t','logp_max_tfce']:
        full=np.zeros(m.shape);full[m]=r[key][0];save32(full,ref,t.path/f'{key}.nii.gz')
        np.testing.assert_array_equal(np.asarray(nib.load(t.path/f'{key}.nii.gz').dataobj)[m],r[key][0].astype('float32'))
    save32(signed,ref,t.path/'signed_tfce_corrected_t.nii.gz')
    mean=data.mean(axis=0);se=data.std(axis=0,ddof=1)/np.sqrt(n);half=stats.t.ppf(.975,n-1)*se
    for key,v in [('mean_effect',mean),('standard_error',se),('pointwise_ci_low',mean-half),('pointwise_ci_high',mean+half)]:
        full=np.zeros(m.shape);full[m]=v;save32(full,ref,t.path/f'{key}.nii.gz')
    components=[]
    for sign in [-1,1]:
        lab,count=label(signed*sign>0,generate_binary_structure(3,3))
        for idx in range(1,count+1):
            points=np.argwhere(lab==idx);v=stat[lab==idx];peak=points[np.argmax(sign*v)];xyz=nib.affines.apply_affine(ref.affine,peak)
            components.append(dict(sign=sign,n_voxels=len(points),peak_t=v[np.argmax(sign*v)],peak_x=xyz[0],peak_y=xyz[1],peak_z=xyz[2]))
    csv(pd.DataFrame(components,columns=['sign','n_voxels','peak_t','peak_x','peak_y','peak_z']),t.path/'components.csv')
    summary=dict(cohort=cohort,model=model,contrast=contrast,n_subjects=n,alpha=alpha,n_perm=10000,
        positive_voxels=int((signed>0).sum()),negative_voxels=int((signed<0).sum()),n_mask_voxels=int(m.sum()))
    t.finish(f'# Updated voxelwise inference\n\n{cohort}, {model}, {contrast}: 10,000 two-sided sign flips, seed 20261018, eight chunks. Native score/TFCE definitions with the independently validated component-tree backend. Historical stages use .05/4 within each cohort/model; pooled Str2−Str1 across runs 3–6 uses its separate single-contrast FWE .05 family. No project-wide correction across these separate families. Positive pooled effects mean Str2>Str1. All samples/models use the same 58,550-voxel N22 intersection mask.\n\n{summary["positive_voxels"]} positive and {summary["negative_voxels"]} negative corrected voxels. Components are descriptive 26-connected sets, not cluster inference. Effects and 95% CIs are pointwise, not intervals adjusted for selecting peaks.\n\nRaw null distributions are preserved and both t and TFCE FWE maps reconstructed exactly. Native TFCE is recomputed independently on the saved observed-score precision. SciPy t values from float64 participant effects agree within the expected native float32 normalization tolerance (atol 5e-6, rtol 2e-6); exported float32 images must exactly equal cast raw values. The initial launcher used a float64-only postprocessing tolerance, so its postprocessing failed after saving the full permutations. This separate exporter audits and completes those saved runs without rerunning or altering their permutations.\n',
        dict(passed=True,summary=summary,native_observed_TFCE_checked=True,null_FWE_exact=True,
             independent_t_max_abs_difference=float(abs(independent-r['t'][0]).max()),
             independent_t_atol=5e-6,independent_t_rtol=2e-6,all_images_unscaled_float32=True))
    return summary


def main():
    summaries=[]
    for job in jobs():
        result=one(job)
        if result:summaries.append(result)
    t=Task('voxelwise');csv(summaries,t.path/'permutation_summary.csv')
    if len(summaries)==24:
        for cohort,model,contrast,_ in jobs():
            for file in ['input_manifest.json','validation.json']:t.record(t.path/cohort/model/contrast/file)
        t.finish('# Matched-sample voxelwise analysis\n\nAll 24 analyses have complete 10,000-permutation raw null distributions and independently audited native TFCE/FWE results. Separate correction families are stated per analysis. Both samples share the 58,550-voxel N22 intersection mask.\n',dict(passed=True,n_analyses=24,n_permutations_per_analysis=10000,n_full_permutation_runs=24))
    print(f'Validated {len(summaries)}/24 saved permutation analyses',flush=True)


if __name__=='__main__':main()
