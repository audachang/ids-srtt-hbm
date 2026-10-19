# A7 validation

Regenerated all 176 shifted designs from JSON timing, corrected events and original nuisance TSVs; checked saved frame times, every observed80 map extraction including coverage-ineligible NaNs, and all 110 voxelwise subject maps. Recomputed group estimates, pointwise t intervals, p values and all specified BH families from participant values, including component identities and phase components. Rehashed every unique first-level input and saved fit/permutation output. Per-analysis voxelwise validation independently checks observed t, native TFCE, exact empirical FWE and map casts. Recounted signed corrected voxels. A before/after preservation hash baseline checks existing code/docs/manuscript/resources and specified updated/review results without following directory symlinks. No manuscript/UI/build or scientific identification is validated.

{
  "passed": true,
  "n_arithmetic_checks": 34348,
  "n_shifted_designs_reconstructed": 176,
  "max_design_difference": 2.220446049250313e-16,
  "n_observed80_map_slots_checked": 21120,
  "n_subject_maps_checked": 110,
  "n_unique_first_level_inputs": 1302,
  "n_output_hashes_verified": 1118,
  "n_protected_files_unchanged": 10924,
  "stage_BH48": true,
  "joint_BH192": true,
  "component_BH48": true,
  "headline_BH4": true,
  "direct_BH144": true,
  "n_permutation_analyses": 10,
  "n_permutations_each": 10000,
  "seed": 20261018,
  "min_residual_df": 182
}
