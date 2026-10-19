#!/usr/bin/env python3
"""Full-precision rerun of T2 TFCE, preserving raw null arrays and image dtype.

Same inputs, seed, 8 chunks and 10,000 permutations as the first run. The prior
integer-header maps are retained as a superseded export. Only memory layout is
optimized; native-vs-contiguous TFCE equality is tested before inference.
"""
from common import *
import argparse,time,importlib,warnings
from scipy.ndimage import label,generate_binary_structure
from nilearn.glm.second_level import non_parametric_inference
from nilearn.maskers import NiftiMasker
from tfce_backend import NATIVE_TFCE,contiguous_tfce,contiguous_chunk
pl=importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
sl=importlib.import_module('nilearn.glm.second_level.second_level')

def save32(img,path):
 values=np.asarray(img.dataobj,dtype=np.float32);header=img.header.copy();header.set_data_dtype(np.float32);header.set_slope_inter(1.,0.)
 out=nib.Nifti1Image(values,img.affine,header);out.to_filename(path);check=nib.load(path)
 assert check.get_data_dtype()==np.dtype('float32') and check.dataobj.slope==1. and check.dataobj.inter==0.
 np.testing.assert_array_equal(np.asarray(check.dataobj),values)
 return check

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=8);args=parser.parse_args();assert args.jobs==8,'Keep the original chunk/seed schedule'
 t=Task('T2_within_run/voxelwise_float32');paths=[t.record(BASE/'T2_within_run/pooled_subject_maps'/f'{s}_Str2_minus_Str1_r36_mean.nii.gz') for s in SUBJECTS]
 mask=t.record(ROOT/'results/glm_volume/whole_brain/common_intersection_mask.nii.gz');ref=nib.load(mask);m=np.asarray(ref.dataobj)>0
 masker=NiftiMasker(mask_img=str(mask)).fit();y=masker.transform([str(p) for p in paths]);rng=np.random.default_rng(20261018)
 equality=[]
 for i in range(4):
  z=y if i==0 else y*rng.choice([-1,1],size=(20,1));v=z.mean(axis=0)/(z.std(axis=0,ddof=1)/np.sqrt(20))
  array=np.zeros(m.shape+(1,),order='F');array[...,0][m]=v
  native=NATIVE_TFCE(array,generate_binary_structure(3,1),two_sided_test=True)
  optimized=contiguous_tfce(array,generate_binary_structure(3,1),two_sided_test=True)
  np.testing.assert_array_equal(native,optimized);equality.append(float(np.max(abs(native-optimized))))
 writej(t.path/'memory_layout_equality.json',{'passed':True,'observed_and_three_null_fields_max_differences':equality,'change':'C-contiguous copy before calling unchanged native TFCE; no value/threshold/connectivity change'})
 print('Native/contiguous TFCE equality passed for all four fields.',flush=True)
 old_chunk=pl._permuted_ols_on_chunk;old_tfce=pl.calculate_tfce;old_call=sl.permuted_ols
 def capture(*a,**kw):
  result=old_call(*a,**kw);np.savez_compressed(t.path/'raw_permutation_arrays.npz',**result);return result
 pl._permuted_ols_on_chunk=contiguous_chunk;pl.calculate_tfce=contiguous_tfce;sl.permuted_ols=capture
 design=pd.DataFrame({'intercept':np.ones(20)});csv(design,t.path/'second_level_design.csv');started=time.monotonic()
 try:
  output=non_parametric_inference([str(p) for p in paths],design_matrix=design,second_level_contrast='intercept',mask=str(mask),model_intercept=True,n_perm=10000,two_sided_test=True,random_state=20261018,n_jobs=8,tfce=True,verbose=1)
 finally:pl._permuted_ols_on_chunk=old_chunk;pl.calculate_tfce=old_tfce;sl.permuted_ols=old_call
 # Save every returned map with a float32 header, including probability maps.
 for name,im in output.items():save32(im,t.path/f'{name}.nii.gz')
 stat=np.asarray(output['t'].dataobj);logp=np.asarray(output['logp_max_tfce'].dataobj);keep=logp>=-np.log10(.05);signed=np.where(keep,stat,0.)
 save32(nib.Nifti1Image(signed,ref.affine),t.path/'signed_tfce_fwe05_t.nii.gz')
 raw=np.load(t.path/'raw_permutation_arrays.npz');h0=raw['h0_max_tfce'];assert h0.shape==(1,10000)
 ranks=np.searchsorted(np.sort(h0[0]),np.abs(raw['tfce'][0]),side='left');expected=-np.log10((10001-ranks)/10001)
 np.testing.assert_allclose(expected,raw['logp_max_tfce'][0],atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(expected,logp[m],atol=1e-12,rtol=1e-12)
 npos=int((signed>0).sum());nneg=int((signed<0).sum());rows=[]
 for sign in [-1,1]:
  labels,n=label(signed*sign>0,structure=generate_binary_structure(3,3))
  for k in range(1,n+1):
   idx=np.argwhere(labels==k);v=stat[labels==k];peak=idx[np.argmax(sign*v)];xyz=nib.affines.apply_affine(ref.affine,peak)
   rows.append(dict(sign=sign,n_voxels=len(v),peak_t=v[np.argmax(sign*v)],peak_x=xyz[0],peak_y=xyz[1],peak_z=xyz[2]))
 csv(pd.DataFrame(rows,columns=['sign','n_voxels','peak_t','peak_x','peak_y','peak_z']),t.path/'components.csv')
 # Compare the full-precision signed inference saved before the first export bug.
 previous=BASE/'T2_within_run/voxelwise/signed_tfce_fwe05_t.nii.gz';old=np.asarray(nib.load(t.record(previous)).dataobj)
 np.testing.assert_allclose(signed,old,atol=1e-6,rtol=1e-6)
 t.finish(f'# T2 pooled TFCE — full-precision export\n\n20 maps; 58,611-voxel common mask; 10,000 two-sided permutations; seed 20261018; eight chunks. Positive means mean Str2−Str1 across runs 3–6 >0. TFCE voxelwise FWE p≤.05 covers this single new contrast.\n\nCorrected voxels: {npos} positive, {nneg} negative. All maps are unscaled float32. The raw permutation arrays, including 10,000 null TFCE maxima, are retained. FWE log-p values are independently reconstructed from the saved null distribution.\n\nThe preceding export inherited a uint8 mask header for raw t/log-p images. It is retained under ../voxelwise for audit and superseded by this directory. Its separate float32 signed map agrees with this rerun. Computation is the unchanged native TFCE after a contiguous copy; equality is verified on the observed and three null fields. Inputs, seed, chunk count, permutations, connectivity and thresholds are unchanged. Runtime {time.monotonic()-started:.1f} s. Connected-component summaries are descriptive, not additional cluster inference.',dict(passed=True,n_subjects=20,n_perm=10000,n_jobs=8,seed=20261018,positive_voxels=npos,negative_voxels=nneg,n_mask_voxels=int(m.sum()),all_image_outputs_unscaled_float32=True,null_fwe_reconstruction_passed=True,layout_equality_passed=True,previous_signed_map_reproduced=True))
if __name__=='__main__':main()
