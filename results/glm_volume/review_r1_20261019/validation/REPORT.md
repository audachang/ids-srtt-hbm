# R1 imaging validation

Saved participant values independently reproduce all stage, direct-phase and component estimates, pointwise t CIs and p values. All BH families are independently recomputed with untestable p=1, including 48-slot stage/components,192-slot joint stages,144-slot direct phases and4-slot early headline components. Component identities are checked participant by participant. Every first-level design is full rank with ≥20 residual df; exact spike indicator columns match reconstructed masks; D1 counts match raw trial slots. Every unique first-level input is rehashed and per-run output hashes verified. All ten permutation reports and outputs are checked, with exact FWE/native observed-TFCE checks recorded per analysis. The auditor remains responsible for independent project-wide protected-file comparison; this validator does not substitute for that audit.

{
  "passed": true,
  "n_arithmetic_checks": 73682,
  "n_first_level_qc": 880,
  "n_unique_first_level_inputs": 1126,
  "n_output_hashes_verified": 3230,
  "stage_families_48": true,
  "joint_families_192": true,
  "direct_families_144": true,
  "headline_families_4": true,
  "paired_component_identity": true,
  "D1_repeat_counts_verified": true,
  "spike_indicators_verified": true,
  "all_designs_full_rank": true,
  "min_residual_df": 134,
  "n_permutation_analyses": 10,
  "aCompCor_short_component_runs": [
    {
[Line withheld: participant identifier or private path.]
      "run": 5,
      "model": "motion6_acompcor8",
      "n_components": 5
    },
    {
[Line withheld: participant identifier or private path.]
      "run": 5,
      "model": "motion12_acompcor8",
      "n_components": 5
    }
  ]
}
