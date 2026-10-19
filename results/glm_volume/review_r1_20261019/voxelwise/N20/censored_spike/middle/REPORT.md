# A4 censored-spike voxelwise inference

[Line withheld: participant identifier or private path.]

All null maxima are preserved. Both max-t and TFCE FWE log-p are independently reconstructed exactly from raw arrays. Observed TFCE is checked against native Nilearn; t is independently recomputed from participant maps within the existing float32-normalization tolerance (atol5e-6/rtol2e-6). Exported images exactly match raw arrays cast to unscaled float32. Effect CIs are pointwise and not adjusted for selecting peaks. Detection-count differences from row deletion are not tests of model differences.

{
  "cohort": "N20",
  "model": "censored_spike",
  "contrast": "middle",
  "n_subjects": 20,
  "alpha": 0.0125,
  "n_perm": 10000,
  "seed": 20261018,
  "positive_voxels": 0,
  "negative_voxels": 0,
  "n_mask_voxels": 58550,
  "permutation_seconds": 25.69077660702169
}
