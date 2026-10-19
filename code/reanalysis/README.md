# HBM reanalysis, 2026-10-18

The work order is `docs/REANALYSIS_PLAN_2026-10-18.md`. Historical inputs are read from `/media/data3/chiaoen_srtt`; all new outputs are written into this checkout, under `results/{glm_volume,behavioral}/reanalysis_20261018` and `results/reanalysis_20261018`.

Use the original pinned environment and single-thread BLAS. Do not run the old manuscript Makefile targets for this new version.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ANALYSIS_PYTHON=/media/data3/chiaoen_srtt/envs/nilearn-0131/bin/python
$ANALYSIS_PYTHON code/reanalysis/run_saved.py
$ANALYSIS_PYTHON code/reanalysis/behavior.py
Rscript code/reanalysis/fit_anticipation.R results/behavioral/reanalysis_20261018/T7_anticipation /media/data3/chiaoen_srtt 999 6
$ANALYSIS_PYTHON code/reanalysis/run_refits.py --workers 6
$ANALYSIS_PYTHON code/reanalysis/audit_alternatives.py
$ANALYSIS_PYTHON code/reanalysis/run_tfce_float32.py --jobs 8
$ANALYSIS_PYTHON code/reanalysis/build_censored_pooled.py
$ANALYSIS_PYTHON code/reanalysis/run_tfce_censored_float32.py --jobs 8
$ANALYSIS_PYTHON code/reanalysis/visualize_within_run.py
$ANALYSIS_PYTHON code/reanalysis/run_saved.py --tasks diagnostics
$ANALYSIS_PYTHON code/reanalysis/summarize_sensitivity.py
$ANALYSIS_PYTHON code/reanalysis/run_shared_rt.py --workers 6
$ANALYSIS_PYTHON code/reanalysis/update_t7_shared_rt.py
$ANALYSIS_PYTHON code/reanalysis/finalize.py
$ANALYSIS_PYTHON code/reanalysis/validate_reanalysis.py --rehash-inputs
```

The full run needs substantial compute time. Run-wise refits resume only if the exact numerical script and common module hashes match. The full anticipation GLMM has separated zero-event cells and is deliberately rejected for inference; a successful process exit documents this scientific failure. The 999 requested bootstrap resamples are not accepted/executed to completion for an invalid full-model estimator. Descriptive probability differences instead use 20,000 participant resamples and are explicitly separate from GLMM inference.

`run_saved.py` re-extracts both reference/censored maps, recomputes coverage/FDR, within-run comparisons, trajectories, phase components and trends, checks N=22 inputs, and measures the design. `run_refits.py` performs four full-voxel AR(1) fits per run (reference reproduction, SPM+derivative, errors/omissions, RT modulation); only new ROI coefficients and design/QC are retained. Preserved alternative fits are independently re-extracted by `audit_alternatives.py`. `run_tfce_float32.py` and `run_tfce_censored_float32.py` perform the authoritative 10,000-permutation pooled within-run analyses, with raw null arrays and explicit float32 exports.

T3 produces the work-order-permitted missing-input report, rather than an N=22 fit. T9 manuscript edits are deferred for author review. Optional T10 was not run. No manuscript or historical analysis file is overwritten.

`SUMMARY.md` is the author handback. Every task has inputs/hashes, validation and a report. `validate_reanalysis.py` independently checks one-sample statistics, BH families, contrast algebra, masks, map thresholding, trial counts, missing-data states and first-level reproduction. `--rehash-inputs` also rereads all input files, including the original BOLD images.

The additional shared RT-PM was added after inspecting the original PM centering scope, before inspecting its results. Both variants are retained. The within-condition PM cannot remove condition-average RT/duration differences; the shared run-centered PM retains those differences in its regressor. Neither measures actual cue duration.

The original TFCE export in T2_within_run/voxelwise is superseded because Nilearn inherited a uint8 mask header for raw t/log-p maps. The authoritative voxelwise_float32 rerun explicitly writes unscaled float32 and retains raw null distributions. A contiguous copy before native TFCE improves runtime without changing values; observed/three-null equality and the first signed map are checked. Use the float32 runner above.
