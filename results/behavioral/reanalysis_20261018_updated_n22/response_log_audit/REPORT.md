# Native response-log consistency audit

Available native PsychoPy logs are matched by exact native CSV filename. For each valid logged 0<RT<1-s response, the response-clock start plus native RT predicts an experiment-clock keypress time. The nearest DATA Keypress event of the same key is checked against a declared 50-ms tolerance; duplicate nearest-event uses are separately flagged. Anticipatory responses are audited with the same rule. Missing logs are explicit. No trial is removed or reassigned and no RT is changed.

{
  "n_cohort_runs": 176,
  "n_runs_with_logs": 136,
  "n_valid_responses_checked": 43333,
  "n_response_matches_within_50ms": 43294,
  "n_duplicate_nearest_event_assignments": 0,
  "n_anticipatory_checked": 1203,
  "n_anticipatory_unique_matches": 1169
}

[Line withheld: participant identifier or private path.]

