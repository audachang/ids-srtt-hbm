#!/usr/bin/env python3
from common import *
from nilearn.glm.second_level import non_parametric_inference
from nilearn.maskers import NiftiMasker
from scipy.ndimage import label,generate_binary_structure
import argparse,time
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=8);p.add_argument('--permutations',type=int,default=10000);a=p.parse_args()
 assert a.permutations>=10000,'Submission inference requires >=10000 permutations'
 t=Task('T2_within_run/voxelwise');maps=[t.record(BASE/'T2_within_run/pooled_subject_maps'/f'{s}_Str2_minus_Str1_r36_mean.nii.gz') for s in SUBJECTS]
 mask=t.record(ROOT/'results/glm_volume/whole_brain/common_intersection_mask.nii.gz');design=pd.DataFrame({'intercept':np.ones(20)})
 csv(design,t.path/'second_level_design.csv');started=time.monotonic()
 output=non_parametric_inference([str(p) for p in maps],design_matrix=design,second_level_contrast='intercept',mask=str(mask),model_intercept=True,n_perm=a.permutations,two_sided_test=True,random_state=20261018,n_jobs=a.jobs,tfce=True,verbose=1)
 for name,im in output.items():im.to_filename(t.path/f'{name}.nii.gz')
 stat=np.asarray(output['t'].dataobj);logp=np.asarray(output['logp_max_tfce'].dataobj);keep=logp>=-np.log10(.05);signed=np.where(keep,stat,0.)
 ref=nib.load(mask);h=ref.header.copy();h.set_data_dtype('float32');nib.Nifti1Image(signed.astype('float32'),ref.affine,h).to_filename(t.path/'signed_tfce_fwe05_t.nii.gz')
 # Descriptive connected sets, with no new cluster-level test.
 rows=[]
 for sign in [-1,1]:
  lab,n=label(signed*sign>0,structure=generate_binary_structure(3,3))
  for k in range(1,n+1):
   indices=np.argwhere(lab==k);vals=stat[lab==k];peak=indices[np.argmax(vals*sign)];xyz=nib.affines.apply_affine(ref.affine,peak)
   rows.append(dict(sign=sign,n_voxels=len(vals),peak_t=float(vals[np.argmax(vals*sign)]),peak_x=xyz[0],peak_y=xyz[1],peak_z=xyz[2]))
 csv(rows,t.path/'components.csv')
 npos=int((signed>0).sum());nneg=int((signed<0).sum())
 t.finish(f'# T2 — pooled within-run voxelwise inference\n\nTwenty reference-model maps are the mean of Str2−Str1 across runs 3–6 (positive = Str2 > Str1). Reuses the historical 58,611-voxel common intersection mask and a group intercept. Nilearn {nilearn.__version__}; {a.permutations:,} two-sided TFCE permutations; seed 20261018. TFCE maximum-statistic FWE threshold p≤.05 covers this **single new pooled contrast**, without claiming correction across historical/new families.\n\nCorrected voxels: {npos} positive, {nneg} negative. `components.csv` describes 26-connected sets; it does not add cluster-level inference. Runtime {time.monotonic()-started:.1f} s. Raw t, TFCE and corrected log-p maps are retained.',dict(passed=True,n_subjects=20,n_perm=a.permutations,seed=20261018,positive_voxels=npos,negative_voxels=nneg,n_mask_voxels=int(np.asarray(ref.dataobj).sum()),finite_maps=bool(np.isfinite(stat).all() and np.isfinite(logp).all())))
