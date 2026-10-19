# D1 repeats and A6 motion burden

D1, fixed by the coordinator before A5/B2 results: current keypress target equals the immediately preceding trial slot in the same run, and stim_onset difference is exactly 1 s. Condition changes and block boundaries do not reset detection; fixation/trigger gaps exclude pairs. First trial has no preceding slot. Counts include errors/omissions because targets, not observed responses, define repeats. The preflight 48 gap-separated Random boundary pairs are excluded.

Condition counts (N20 is supplement-only):

```text
cohort condition  n_trials  n_repeats  n_repeats_cross_block  n_equal_previous_excluded_gap  repeat_rate
   N20       Ran     12800          0                      0                             41     0.000000
   N20      Str1     19200       2400                      0                             28     0.125000
   N20      Str2     19200       2399                      0                             19     0.124948
   N22       Ran     14080          0                      0                             48     0.000000
   N22      Str1     21120       2640                      0                             29     0.125000
   N22      Str2     21120       2638                      0                             20     0.124905
```

A6 assigns frame acquisition times 0,2,…398 s to corrected event boxcars using onset≤time<onset+duration. Str1 and Str2 are combined as structured; times outside boxcars are omitted. FD uses fMRIPrep framewise_displacement with missing values omitted, separately reporting n_fd_defined. Censored fractions use all assigned volumes and the reconstructed v8-rule masks; these masks are distinct from an FD threshold. Run values are equally averaged within participant over the two runs of each phase, then participants are equally weighted. Paired structured−random differences are tested against zero using participant phase values and pointwise t 95% CIs; these descriptive p values have no multiplicity adjustment. Participant observations remain internal.

