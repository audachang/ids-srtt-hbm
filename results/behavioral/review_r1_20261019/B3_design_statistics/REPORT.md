# B3: realized target statistics, sequence overlap and task order

N=22, all 56,320 logged trials before response cleaning. Only group-level design outputs are exported here. Targets are `keypress`; physical sequence identity is `rule` (StrA/StrB), counterbalanced condition is `rule_group` (Str1/Str2). All 1,408 original 40-trial condition lists match logged targets and physical rules exactly, as do the 176 canonical and native trial files. All 1,584 fixation values match their source lists. Trial onsets reproduce rounded native stimulus start minus the first fixation start for every trial. Source paths and hashes are recorded in the input manifest; no individual sequence/timing data table is exported.

Target frequencies retain four targets per run/condition (80 rows). First-order transitions are consecutive target slots **within each 40-trial condition list**, assigned to the current trial condition, retaining all 16 ordered pairs per run/condition (320 rows, including zeros). Trial-order transitions across an internal trigger gap remain part of the list’s realized design; `n_onset_exactly_1s` identifies the temporally adjacent subset. Fixation-separated block boundaries are excluded. Pooled pair frequencies, row-normalized conditional matrices and the distribution of participant frequencies are exported. D1 repeat rates use exactly-1-s onset differences, irrespective of block labels, and separately report gap matches and cross-block repeats. Repeat fractions use both all trials and eligible adjacent pairs as denominators.

| condition | n_trials | n_immediate_repeats_D1 | n_repeats_crossing_block | n_gap_separated_target_matches_excluded |
|---|---|---|---|---|
| Ran | 14080 | 0 | 0 | 48 |
| Str1 | 21120 | 2640 | 0 | 29 |
| Str2 | 21120 | 2638 | 0 | 20 |

The physical cyclic sequences are A=3-1-1-2-3-4-2-4 and B=3-4-2-1-1-3-2-4. Pair and triplet enumerations include wrap-around; all 4²+4³ words are retained, with multiplicities per eight-item cycle, frequencies and shared/diagnostic/neither labels. Shared pairs/triplets:

| order | word | count_per_cycle_StrA | count_per_cycle_StrB |
|---|---|---|---|
| 2 | 1-1 | 1 | 1 |
| 2 | 2-4 | 1 | 1 |
| 2 | 3-4 | 1 | 1 |
| 2 | 4-2 | 1 | 1 |
| 2 | 4-3 | 1 | 1 |
| 3 | 2-4-3 | 1 | 1 |
| 3 | 3-4-2 | 1 | 1 |

The outer block order is fixed in all 176 runs: Random1 → Structured1–6 → Random2. Each structured block comprises five eight-trial repetitions; relative Str1/Str2 counts per block are 5/0, 5/0, 4/1, 3/2, 2/3, 1/4, 0/5, 0/5 across runs 1–8. Odd/even participant assignment swaps A/B as Str1/Str2. The source shuffles physical repetition order within those counts, prohibits more than two consecutive minority sequences inside a block, and forbids identical adjacent structured-block lists in runs 3–6; all logged lists satisfy those constraints. Exported group pattern frequencies distinguish physical A/B order, Str1/Str2 order and nine-fixation duration patterns. Distinct whole-run role orders and per-participant diversity are aggregated by run:

| run | n_outer_block_orders | n_whole_run_role_orders | n_fixation_duration_patterns | distinct_subblock_orders_per_participant_min | distinct_subblock_orders_per_participant_max |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 22 | 1 | 1 |
| 2 | 1 | 1 | 22 | 1 | 1 |
| 3 | 1 | 22 | 22 | 3 | 5 |
| 4 | 1 | 22 | 22 | 4 | 6 |
| 5 | 1 | 22 | 22 | 4 | 6 |
| 6 | 1 | 22 | 22 | 3 | 5 |
| 7 | 1 | 1 | 22 | 1 | 1 |
| 8 | 1 | 1 | 22 | 1 | 1 |

The representative timeline is selected deterministically as the first sorted participant’s first interleaved run (run 3), independent of behavioral outcomes. It displays all 30 structured repetitions, both random blocks, nine fixations, trigger waits and the remaining acquisition window. Times are rounded native routine times/trial onsets relative to first fixation start, matching the imaging event clock. The timeline tiles 0–400 s without overlap or gaps; its breakdown is:

| kind | duration_s |
|---|---|
| fixation | 72 |
| remaining_acquisition_window | 4 |
| task_slots | 320 |
| trigger_wait | 4 |

The generator programs each fixation at nominal 6/8/10 s minus .02 s; the nine nominal durations sum to 72 s in every run (programmed sum 71.82 s). 320 one-second task slots plus nominal fixations account for 392 s; trigger waits and time following the logged task explain the rest of this example. Counts vary with observed trigger alignment, so this example is not asserted to be every run’s timeline. Fixation routine timing is not visual cue exposure, and the remaining acquisition window is not labeled as fixation. RNG seed/state used to generate the original lists is unavailable; realized lists, rather than a newly simulated generator, supply these statistics. No causal separation of prevalence, familiarity, practice or local transition structure is claimed.

Deviation/clarification: D1 supplies the temporal repeat definition. No other deviation.

