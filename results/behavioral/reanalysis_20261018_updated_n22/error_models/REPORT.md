# Matched N20/N22 error models

Errors include incorrect responses and omissions among all logged trials, with no RT exclusions. Binomial logit GLMM uses 20 observed run-condition cells, independent participant intercept/sequence slopes, and nested run/block intercepts. The Gaussian sensitivity models trial-denominator-weighted block proportions; its full hierarchy is retained unless the block variance is numerically zero and equivalence is verified. N20 source counts match exactly, allowing historical model/bootstrap reuse; N22 is refitted.

The 999 complete-participant bootstrap draws keep observed within-person dependence. Singular variance estimates are permitted and counted; numerical nonconvergence is excluded, with at least 95% valid fits required. Contrast tests use bootstrap covariance, approximate t(19)/t(21), and separate BH families of 24 per cohort/model. Percentile 95% intervals are pointwise. Native z (binomial) or approximate t (Gaussian) results are exported separately. Conditional odds ratios differ from population-average error probabilities, which integrate random-effect distributions with quadrature checked at 80 versus 160 nodes. No independent-group test compares the overlapping samples.

- N20: 51200 trials, 1425 errors; valid binomial/Gaussian bootstrap refits 999/999 of 999.
- N22: 56320 trials, 1583 errors; valid binomial/Gaussian bootstrap refits 999/999 of 999.

| Sample | Contrast | Conditional OR [pointwise 95% CI] | BH q |
|---|---|---:|---:|
| N20 | run3_Str2_minus_Str1 | 3.191 [1.908, 5.568] | 0.01061 |
| N20 | early_adjusted_change | 2.475 [1.601, 3.986] | 0.01061 |
| N20 | middle_adjusted_change | 1.652 [1.118, 2.429] | 0.06819 |
| N20 | late_adjusted_change | 1.203 [0.880, 1.551] | 0.2779 |
| N20 | early_minus_middle | 1.498 [1.008, 2.326] | 0.1294 |
| N20 | early_minus_late | 2.057 [1.391, 3.261] | 0.01468 |
| N20 | middle_minus_late | 1.373 [1.011, 1.930] | 0.1294 |
| N22 | run3_Str2_minus_Str1 | 3.122 [1.845, 5.123] | 0.005031 |
| N22 | early_adjusted_change | 2.249 [1.458, 3.490] | 0.01048 |
| N22 | middle_adjusted_change | 1.625 [1.145, 2.246] | 0.04305 |
| N22 | late_adjusted_change | 1.208 [0.866, 1.640] | 0.3228 |
| N22 | early_minus_middle | 1.384 [0.943, 2.083] | 0.2043 |
| N22 | early_minus_late | 1.862 [1.239, 2.796] | 0.03147 |
| N22 | middle_minus_late | 1.346 [1.030, 1.850] | 0.1425 |

