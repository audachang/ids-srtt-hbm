# Corrected-timing N20/N22 reanalysis for HBM

[Line withheld: participant identifier or private path.]

## Main findings and interpretation

Left BA6 (the original `LeftSMA` mask) retains a negative initial contrast and positive early DiD in both samples. Estimates use the historical two-run sum scale, with pointwise 95% t intervals and BH across 48 ROIs per stage. Later reference contrasts remain uncertain.

| cohort | stage | mean | ci_low | ci_high | q |
| --- | --- | --- | --- | --- | --- |
| N20 | initial | -0.201106 | -0.259437 | -0.142774 | 3.02425e-05 |
| N20 | early | 0.178465 | 0.0821347 | 0.274795 | 0.0162129 |
| N20 | middle | 0.0668049 | -0.023213 | 0.156823 | 0.386405 |
| N20 | late | 0.0509279 | -0.0457943 | 0.14765 | 0.620077 |
| N22 | initial | -0.195985 | -0.249165 | -0.142805 | 4.65448e-06 |
| N22 | early | 0.165788 | 0.0770979 | 0.254479 | 0.0136015 |
| N22 | middle | 0.0515982 | -0.0341641 | 0.13736 | 0.578445 |
| N22 | late | 0.0455057 | -0.0420117 | 0.133023 | 0.657911 |

The same early BA6 estimate across all nine N22 specifications is listed here to show signed sensitivity, without using threshold counts to select a model:

| model | mean | ci_low | ci_high | q |
| --- | --- | --- | --- | --- |
| censored | 0.159548 | 0.0775124 | 0.241583 | 0.0101814 |
| errors_omissions | 0.166043 | 0.0757824 | 0.256303 | 0.0157589 |
| reference | 0.165788 | 0.0770979 | 0.254479 | 0.0136015 |
| rt_modulator | 0.165887 | 0.0721456 | 0.259628 | 0.022276 |
| rt_shared_modulator | 0.142548 | 0.0489996 | 0.236097 | 0.0463238 |
| spm_derivative | 0.202333 | 0.0704617 | 0.334205 | 0.0527568 |
| spm | 0.203393 | 0.0679742 | 0.338812 | 0.0616457 |
| motion12_acompcor8 | 0.0920739 | 0.012278 | 0.17187 | 0.337239 |
| precision_weighted | 0.164325 | 0.0785329 | 0.250117 | 0.0147047 |

The early BA6 result combines changes in structured and random coefficients; it should not be described as an isolated increase in the new-sequence response. The condition coefficients are relative to the implicit fitted baseline. Early-minus-initial changes are two-run **means**, so twice their sequence-minus-random change equals the summed neural DiD:

| component | mean | ci_low | ci_high |
| --- | --- | --- | --- |
| structured | 0.0209693 | -0.021586 | 0.0635246 |
| random | -0.0619249 | -0.108114 | -0.015736 |
| sequence_minus_random | 0.0828942 | 0.038549 | 0.127239 |

The time-matched within-run contrast is the equal-run mean of Str2−Str1 over runs 3–6 (positive means Str2>Str1). N22 reference yields 0/48 ROI findings at BH q≤.05; the whole-brain analysis uses a distinct search family. ROI and whole-brain threshold patterns should not be treated as contradictory or interchangeable.

Right accumbens early−late is sensitive to censoring. The direct family retains all 144 planned ROI×phase-pair tests; overlapping N20/N22 samples are compared descriptively.

| cohort | model | mean | ci_low | ci_high | q_three_pairs_144 |
| --- | --- | --- | --- | --- | --- |
| N20 | censored | 0.32697 | 0.176578 | 0.477362 | 0.0314869 |
| N20 | reference | 0.420667 | 0.173632 | 0.667701 | 0.292892 |
| N22 | censored | 0.294756 | 0.151402 | 0.43811 | 0.04836 |
| N22 | reference | 0.380447 | 0.150353 | 0.610541 | 0.140526 |

[Line withheld: participant identifier or private path.]

Compared with the current manuscript, the headline censored N20 counts (29/15/5/0) should be replaced by estimates from the transparently revised reporting model, rather than transferred to N22. For audit only, N22 reference stage counts are 28/4/0/0; these counts do not select the model. The early BA6 estimate is now about .166 (reference N22), versus .179 in the current censored N20 component example.

## Whole-brain results

[Line withheld: participant identifier or private path.]

| cohort | model | contrast | alpha | positive_voxels | negative_voxels |
| --- | --- | --- | --- | --- | --- |
| N20 | reference | initial | 0.0125 | 0 | 7861 |
| N20 | reference | early | 0.0125 | 10 | 0 |
| N20 | reference | middle | 0.0125 | 0 | 0 |
| N20 | reference | late | 0.0125 | 0 | 0 |
| N20 | censored | initial | 0.0125 | 0 | 8741 |
| N20 | censored | early | 0.0125 | 7 | 0 |
| N20 | censored | middle | 0.0125 | 0 | 0 |
| N20 | censored | late | 0.0125 | 0 | 0 |
| N20 | reference | pooled_r36_Str2_minus_Str1 | 0.05 | 279 | 0 |
| N20 | censored | pooled_r36_Str2_minus_Str1 | 0.05 | 424 | 0 |
| N20 | errors_omissions | pooled_r36_Str2_minus_Str1 | 0.05 | 1 | 0 |
| N20 | rt_shared_modulator | pooled_r36_Str2_minus_Str1 | 0.05 | 0 | 0 |
| N22 | reference | initial | 0.0125 | 0 | 6297 |
| N22 | reference | early | 0.0125 | 8 | 0 |
| N22 | reference | middle | 0.0125 | 0 | 0 |
| N22 | reference | late | 0.0125 | 0 | 0 |
| N22 | censored | initial | 0.0125 | 0 | 7144 |
| N22 | censored | early | 0.0125 | 0 | 0 |
| N22 | censored | middle | 0.0125 | 0 | 0 |
| N22 | censored | late | 0.0125 | 0 | 0 |
| N22 | reference | pooled_r36_Str2_minus_Str1 | 0.05 | 420 | 0 |
| N22 | censored | pooled_r36_Str2_minus_Str1 | 0.05 | 577 | 0 |
| N22 | errors_omissions | pooled_r36_Str2_minus_Str1 | 0.05 | 23 | 0 |
| N22 | rt_shared_modulator | pooled_r36_Str2_minus_Str1 | 0.05 | 0 | 0 |

For the N22 pooled within-run contrast, reference/censored/error-adjusted/shared-RT models yield 420/577/23/0 corrected voxels, respectively. These are separate model-specific tests, not a test of differences between models. Treat sensitivity to error/RT adjustment as a constraint on interpretation; RT adjustment changes the question and is not a causal demonstration that a neural effect is explained by response time.

[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

N22 reference overlap with the original ROI union is 26/420 corrected voxels. FS atlas assignments are Unassigned: 204, ctx-lh-superiorparietal: 132, ctx-lh-supramarginal: 50, ctx-lh-postcentral: 10, ctx-lh-precentral: 9, ctx-lh-caudalmiddlefrontal: 6, ctx-lh-inferiorparietal: 5, ctx-lh-superiorfrontal: 4. These center-based labels are descriptive; unassigned voxels are not assumed to be cortex or white matter.

## Behavior

The recovered custom recursive cleaner (minRT=0, block-wise, default error omission) exactly matches the reconstructed stock block-cleaning trial set. N20 cleaned trial identities/RT values and error block counts exactly match historical model inputs, permitting verified reuse. N22 has 56,320 logged trials, 53,959 primary-cleaned correct RT observations and 52,126 observations under RT≥100ms. RT uses the unchanged mixed-model structure, AR(1) within original trial order and condition-specific variance; approximate t df are capped at 19/21 and BH covers 24 contrasts per cohort/policy.

| cohort | contrast | geometric_rt_ratio | ratio_ci_low | ratio_ci_high | q_bh_24 |
| --- | --- | --- | --- | --- | --- |
| N20 | run2_Str1_minus_Ran | 0.66491 | 0.58397 | 0.75707 | 1.1456e-05 |
| N20 | run3_Str2_minus_Str1 | 1.2049 | 1.0089 | 1.439 | 0.057308 |
| N20 | early_adjusted_change | 1.2507 | 1.0435 | 1.4989 | 0.029015 |
| N20 | early_minus_late | 1.3706 | 1.2953 | 1.4502 | 9.7199e-09 |
| N22 | run2_Str1_minus_Ran | 0.68723 | 0.60879 | 0.77576 | 8.8923e-06 |
| N22 | run3_Str2_minus_Str1 | 1.1922 | 1.0099 | 1.4074 | 0.054897 |
| N22 | early_adjusted_change | 1.2288 | 1.0374 | 1.4554 | 0.031062 |
| N22 | early_minus_late | 1.3283 | 1.2611 | 1.399 | 4.634e-09 |

[Line withheld: participant identifier or private path.]

Errors/omissions total 1,583/56,320 trials in N22. The N22 error GLMM and denominator-weighted Gaussian sensitivity have 999 and 999 valid whole-participant refits out of 999. The original full hierarchy is retained unless its block variance is numerically zero and equivalence to the reduced Gaussian model is verified. Bootstrap covariance gives approximate t(21) contrast tests; percentile intervals are pointwise, with separate BH-24 families per cohort/model.

| cohort | contrast | odds_ratio | odds_ratio_ci_low | odds_ratio_ci_high | q_bh_24 |
| --- | --- | --- | --- | --- | --- |
| N20 | run3_Str2_minus_Str1 | 3.1905 | 1.9085 | 5.5678 | 0.010609 |
| N20 | early_adjusted_change | 2.4752 | 1.6009 | 3.9862 | 0.010609 |
| N20 | early_minus_late | 2.0567 | 1.391 | 3.2614 | 0.014677 |
| N22 | run3_Str2_minus_Str1 | 3.1219 | 1.8453 | 5.1227 | 0.0050314 |
| N22 | early_adjusted_change | 2.2492 | 1.4579 | 3.4902 | 0.010481 |
| N22 | early_minus_late | 1.8624 | 1.2391 | 2.7961 | 0.031474 |

[Line withheld: participant identifier or private path.]

Anticipation includes all responded 0<RT<100ms trials, including wrong responses. Counts and accuracy are:

| cohort | n_trials | n_anticipatory | n_correct_anticipatory | anticipatory_accuracy |
| --- | --- | --- | --- | --- |
| N20 | 51200 | 1895 | 1837 | 0.969393139842 |
| N22 | 56320 | 1900 | 1837 | 0.966842105263 |

[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

## Recovered sources and input corrections

[Line withheld: participant identifier or private path.]
[Line withheld: participant identifier or private path.]
- Original cleaner and analysis: `My_modifiedRecursive.R` and `fmri_srtt_beh_new2_analysis.R` alongside those native CSVs.
[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

## Design, sensitivity, and manuscript implications

All 352 N22 reference/censored designs are full rank. Full-mask fitted AR-bin diagnostics are regenerated for every added/changed run and exactly match two archived controls. N22 reference run medians:

| run | Str1_nuisance_adjusted_vif | Str2_nuisance_adjusted_vif | adjusted_Str1_Str2_correlation | Str2_minus_Str1_fullmask_AR_unit_variance | Str2_minus_Ran_fullmask_AR_unit_variance |
| --- | --- | --- | --- | --- | --- |
| 3 | 1.7804 | 1.7328 | -0.63935 | 0.035244 | 0.11511 |
| 4 | 2.135 | 2.0893 | -0.7133 | 0.023929 | 0.1003 |
| 5 | 2.159 | 2.1752 | -0.71932 | 0.023978 | 0.093302 |
| 6 | 1.8186 | 1.8296 | -0.64596 | 0.03543 | 0.088273 |

[Line withheld: participant identifier or private path.]

[Line withheld: participant identifier or private path.]

Error sensitivity uses impulses at a wrong-response time and the omission deadline (trial onset+1s). RT sensitivities use trial-slot onset with a centered RT amplitude, either centered within condition or shared across conditions. Neither is measured cue duration; RT is affected by learning, so adjustment changes the estimand. The trend across runs 3–6 uses [−.3,−.1,.1,.3] weights with the full participant run covariance propagated. Time, prevalence, practice and familiarity covary; call it an exploratory run trend.

For the manuscript, lead with signed effects and uncertainty; distinguish negative initial effects, within-run sequence differences and baseline-adjusted DiD. Keep the reporting-model selection history visible. Do not reselect censoring, sample or covariates because a new specification crosses a threshold. N22 primary versus sensitivity status remains an author decision. T9 prose edits are deferred until author review, as required by the plan; optional T10 brain–behavior analysis and newly found cognitive covariates were not added.

## Deliverables and validation

| task | status |
| --- | --- |
| T1 | complete: matched reference/censored ROI results and N22 motion audit |
| T2 | complete: within-run ROI, trajectories and pooled voxelwise inference |
| T3 | complete: corrected matched N20/N22 imaging, RT, error models and learner table |
| T4 | complete: recovered ordering rules, measured VIF/correlation/variance and fitted AR-bin diagnostics |
| T5 | complete: condition means, CIs, covariance and component/DiD figures |
| T6 | complete: nine updated specifications and participant influence |
| T7 | sensitivities and descriptive anticipation complete; saturated anticipation GLMM nonidentifiable; response/cue audit incomplete because of partial log coverage and timestamp flags |
| T8 | complete: exploratory run trends with covariance propagation |
| T9 | deferred: author review before manuscript edits |
| T10 | optional; not run |

[Line withheld: participant identifier or private path.]
