# B1: participant-level RT contrast validation

N=22 only. The original custom modified-recursive cleaner was replayed in R 4.4.2 (trimr 1.1.1), using minRT=0, participant × original block labels and correct responses. Retained trial identities and RTs match the saved primary LME input exactly: 53,959 trials; the subsequent RT≥100-ms policy retains 52,126. The LME was not refitted.

[Line withheld: participant identifier or private path.]

Exponentiating the mean log contrast and both interval endpoints produces geometric RT ratios (ratios of ratios for change contrasts), as in Table 1. These are participant summary estimates; the saved LME estimates are fixed effects with random effects at zero. Equal participant/cell weighting differs from trial-level model weighting. Both log-scale and ratio-scale CI-width ratios (participant / saved LME) are exported, separately for t and bootstrap intervals. Agreement flags mean only matching sign or matching q≤.05 status, not equivalence or independent replication.

| policy | contrast | geometric_rt_ratio | ratio_ci_low | ratio_ci_high | bootstrap_ratio_ci_low | bootstrap_ratio_ci_high | q_bh_24 | lme_q_bh_24 | t_log_CI_width_over_lme |
|---|---|---|---|---|---|---|---|---|---|
| primary | run2_Str1_minus_Ran | 0.685555 | 0.592983 | 0.792578 | 0.597207 | 0.779459 | 0.000295719 | 8.89225e-06 | 1.19705 |
| primary | run8_Str2_minus_Ran | 0.660048 | 0.54328 | 0.801913 | 0.54581 | 0.783192 | 0.000659716 | 6.14613e-06 | 1.54418 |
| primary | early_adjusted_change | 1.24632 | 1.13863 | 1.36421 | 1.15298 | 1.35954 | 0.000360889 | 0.031062 | 0.533993 |
| primary | early_minus_late | 1.34875 | 1.14177 | 1.59325 | 1.16797 | 1.59064 | 0.00209738 | 4.634e-09 | 3.21102 |
| min100 | run2_Str1_minus_Ran | 0.759212 | 0.697558 | 0.826314 | 0.701431 | 0.820871 | 8.68691e-06 | 2.13366e-07 | 1.24178 |
| min100 | run8_Str2_minus_Ran | 0.758848 | 0.68531 | 0.840279 | 0.689092 | 0.832685 | 3.67085e-05 | 3.63255e-07 | 1.43389 |
| min100 | early_adjusted_change | 1.18299 | 1.11841 | 1.25129 | 1.12508 | 1.24833 | 1.54888e-05 | 0.00262522 | 0.603908 |
| min100 | early_minus_late | 1.22008 | 1.11553 | 1.33442 | 1.12576 | 1.32903 | 0.000237657 | 7.74653e-09 | 2.49617 |

All 48 policy × contrast results are reported regardless of outcome. CIs are pointwise, not simultaneous; resampling 22 participants cannot establish a correct trial-level covariance model. No manuscript or UI validation was performed.

Deviations: none for B1.

