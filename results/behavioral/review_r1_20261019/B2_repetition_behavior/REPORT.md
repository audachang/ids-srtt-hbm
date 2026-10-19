# B2: immediate-repetition exclusion and transition descriptions

**D1, coordinator clarification, 2026-10-19 19:10 Asia/Taipei (before A5/B2 results):** A repeat has the same cued target as the immediately preceding trial slot in the same run, with logged onsets exactly 1 s apart, regardless of condition or block boundary. Targets come from `keypress` (the cued target, not `p_response_keys`); trial order is `trial_in_run`, time is `stim_onset`. Labels are computed from all raw trials before RT/error exclusions; errors do not redefine the preceding stimulus. Fixation/trigger-gap matches are excluded from the repeat definition.

| rule_group | n_trials | immediate_repeats_D1 | repeats_crossing_block | gap_separated_target_matches_excluded |
|---|---|---|---|---|
| Ran | 14080 | 0 | 0 | 48 |
| Str1 | 21120 | 2640 | 0 | 29 |
| Str2 | 21120 | 2638 | 0 | 20 |

[Line withheld: participant identifier or private path.]

| policy | contrast | geometric_rt_ratio | ratio_ci_low | ratio_ci_high | bootstrap_ratio_ci_low | bootstrap_ratio_ci_high | q_bh_24 | all_trials_participant_ratio |
|---|---|---|---|---|---|---|---|---|
| primary | run2_Str1_minus_Ran | 0.688303 | 0.597717 | 0.792618 | 0.601901 | 0.780128 | 0.000441336 | 0.685555 |
| primary | run8_Str2_minus_Ran | 0.661176 | 0.545769 | 0.800987 | 0.548553 | 0.78285 | 0.00162877 | 0.660048 |
| primary | early_adjusted_change | 1.24437 | 1.13633 | 1.36267 | 1.15068 | 1.35858 | 0.000709024 | 1.24632 |
| primary | early_minus_late | 1.34672 | 1.14221 | 1.58785 | 1.16929 | 1.58618 | 0.00693635 | 1.34875 |
| min100 | run2_Str1_minus_Ran | 0.760749 | 0.70051 | 0.826168 | 0.704074 | 0.820984 | 1.97044e-05 | 0.759212 |
| min100 | run8_Str2_minus_Ran | 0.757646 | 0.685214 | 0.837734 | 0.688677 | 0.830559 | 8.47529e-05 | 0.758848 |
| min100 | early_adjusted_change | 1.18129 | 1.11617 | 1.25022 | 1.12362 | 1.24729 | 5.528e-05 | 1.18299 |
| min100 | early_minus_late | 1.22161 | 1.11665 | 1.33644 | 1.12783 | 1.33101 | 0.000857645 | 1.22008 |

Secondary descriptions use the same nonrepeat trials. The printed physical sequences are A=(3,1,1,2,3,4,2,4), B=(3,4,2,1,1,3,2,4). Their cyclic pair sets include the final→first transition. A preceding transition is shared if in both pair sets; diagnostic if in only the trial’s physical sequence. Actual preceding adjacent trial targets determine the class, including uninterrupted eight-trial sequence boundaries. Run starts and trials after gaps have undefined class and are omitted from these descriptions. For each participant/run/condition/category, subtract the mean log RT of all nonrepeat Random trials in that run; average the two run contrasts equally within phase and exponentiate the participant mean. The Random baseline is not transition-matched, and these descriptive categories do not match every local statistic or identify a causal sequence effect. Pointwise t intervals are descriptive; no p/q is assigned. Absent sequence phases are not imputed. Participant values stay internal.

Deviation/clarification: D1 replaces the original expectation that repeats occur only in structured conditions; the exact D1 definition is implemented. No other deviation.

