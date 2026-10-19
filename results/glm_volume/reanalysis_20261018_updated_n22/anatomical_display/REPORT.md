# Updated descriptive anatomy and displays

All 24 corrected maps receive atlas and original-48-ROI overlap summaries and component peak labels. Atlas labels are nearest-neighbor assignments at native 3-mm analysis voxel centers. Unassigned voxels remain unassigned and are not assigned tissue by proximity. Components are descriptive 26-connected sets, not cluster-level inference; label overlap does not establish a mechanism or introduce a new ROI test.

The pooled-contrast PDF shows every occupied axial level across six maps: N20 reference/censored and N22 reference/censored/errors/shared-RT. Each uses its separate single-contrast voxelwise FWE .05 family, the same N22 search mask and sign convention Str2>Str1. Native statistics are displayed with nearest-neighbor resampling onto stored MNI152NLin2009cAsym 1-mm anatomy; neurological orientation. Differences in threshold patterns are not tests of model or cohort differences.

| Sample | Model | Positive | Negative | In original ROI union |
|---|---|---:|---:|---:|
| N20 | reference | 279 | 0 | 5/279 |
| N20 | censored | 424 | 0 | 17/424 |
| N20 | errors_omissions | 1 | 0 | 0/1 |
| N20 | rt_shared_modulator | 0 | 0 | 0/0 |
| N22 | reference | 420 | 0 | 26/420 |
| N22 | censored | 577 | 0 | 39/577 |
| N22 | errors_omissions | 23 | 0 | 1/23 |
| N22 | rt_shared_modulator | 0 | 0 | 0/0 |

