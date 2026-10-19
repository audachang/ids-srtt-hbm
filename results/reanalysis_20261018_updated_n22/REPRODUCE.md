# Reproduction and immutable output roots

[Line withheld: participant identifier or private path.]

The timing/censor preparation must already have passed its controls. Execute dependencies in this order, with R behavior replay and fits before final behavior summaries:

```bash
[Line withheld: participant identifier or private path.]
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
g++ -O3 -fno-fast-math -ffp-contract=off -shared -fPIC code/reanalysis/tfce_sparse.cpp -o code/reanalysis/libtfce_sparse.so
g++ -O3 -fno-fast-math -ffp-contract=off -shared -fPIC code/reanalysis/tfce_union.cpp -o code/reanalysis/libtfce_union.so
[Line withheld: participant identifier or private path.]
[Line withheld: participant identifier or private path.]
Rscript code/reanalysis/updated_rt_lme.R results/behavioral/reanalysis_20261018_updated_n22/source_replay/N20 results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N20 --reuse
Rscript code/reanalysis/updated_rt_lme.R results/behavioral/reanalysis_20261018_updated_n22/source_replay/N22 results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22 --reuse
Rscript code/reanalysis/updated_error_glmm.R results/behavioral/reanalysis_20261018_updated_n22/source_replay/N22 999 12
```

Python stages (run only after each dependency is ready):

- `prepare_n22_inputs.py`
- `updated_first_level.py --workers 4`
- `updated_results.py`
- `updated_design_figures.py`
- `updated_motion.py`
- `updated_targeted_sensitivity.py`
- `validate_direct_backend.py`
- `run_updated_permutations.py`
- `postprocess_updated_tfce.py`
- `audit_response_logs.py`
- `finalize_updated_behavior.py`
- `updated_anatomical_display.py`
- `report_updated_analysis.py`
- `validate_updated_analysis.py`

The saved full raw permutations are authoritative; `postprocess_updated_tfce.py` completes exports and native-precision checks. Use `run_updated_permutations.py` to resume/run: it audits existing raw arrays and completes the independently validated native-precision export after each new run. The original launcher used an overly strict export tolerance; do not discard/recompute completed raw arrays solely for that exception. The final independent validator checks reused input/output and producer hashes. Use a fresh dated namespace if scientific inputs or model definitions change.

N20 RT/error fits are reused only after exact input checks; source paths and raw bootstrap draws are pinned in the task manifests. N22 fits are new. The final summary is written before independent global validation; its SHA256 is included in validation.json. No manuscript table-generation routines are called.
