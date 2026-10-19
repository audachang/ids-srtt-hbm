#!/usr/bin/env python3
from common import *
if __name__=='__main__':
 t=Task('T2_within_run/censored_pooled_maps');ref=nib.load(t.record(ROOT/'results/glm_volume/whole_brain/common_intersection_mask.nii.gz'))
 for subject in SUBJECTS:
  data=np.zeros(ref.shape)
  for run in [3,4,5,6]:
   folder=FIRST/MODELS['censored']/'level1/eventrelated'/subject/f'run-{run:02d}'
   data+=(image(t.record(folder/'Str2_effect_size.nii.gz'),ref)-image(t.record(folder/'Str1_effect_size.nii.gz'),ref))/4
  header=ref.header.copy();header.set_data_dtype('float32');header.set_slope_inter(1.,0.)
  nib.Nifti1Image(data.astype('float32'),ref.affine,header).to_filename(t.path/f'{subject}_Str2_minus_Str1_r36_mean.nii.gz')
 t.finish('# Censored pooled subject contrasts\n\nEqual-run mean of Str2−Str1 across runs 3–6, weights +1/4 for each Str2 and −1/4 for each Str1. Uses archived censor_hp128 first-level estimates. Positive = Str2 > Str1. The common reference grid and whole-brain search mask match the uncensored analysis.',dict(passed=True,n_subject_maps=20,n_condition_input_maps=160,weights_documented=True))
