IDS-SRTT HBM companion — v1.0.0 (2026-10-17)

Scope: analysis source code and group summaries only.
Public repository: https://github.com/audachang/ids-srtt-hbm
Archival version: v1.0.0. Its Zenodo version DOI will be linked from the GitHub
release page after automatic archival completes.
Licenses: original software under code/ is MIT; group summaries, derived tables,
documentation and environment records are CC BY 4.0. See LICENSE.txt and LICENSES/.

Quick start (from this folder; Python standard library only):
  python code/reporting/coverage_significant_table.py results/glm_volume/reporting_20261012/reported_roi_results.csv reproduced/observed_significant_rois.tex
This produces the 49 selected estimates grouped into 31 ROI entries, counts 29/15/5/0.
It does not require a LaTeX installation to generate the table source and CSV.

Aggregate verification (NumPy, pandas, SciPy required; versions in environment/):
  python code/reporting/verify_public_summaries.py .
Checks the manifest, imaging t-test arithmetic, CI and BH families, RT contrasts
from group fixed effects/covariance, and behavioral contrast corrections.
This verifies supplied aggregate outputs, not the underlying participant models.

Data guide: FILE_DICTIONARY.csv describes file units and scope;
COLUMN_DICTIONARY.json explains common statistical columns;
results/glm_volume/reporting_20261012/analysis_spec.json defines estimands/families.
N<2 estimates are suppressed in release CSVs and profiles, with a record in
AGGREGATION_NOTES.json. N, missing values, and 48 planned correction slots remain.
The significant table is unchanged by this suppression. Blank CSV entries are missing,
not zero. A q value for an untestable slot does not imply that a test was performed.

Restricted inputs: raw/preprocessed MRI, participant contrast/condition maps, run
masks/confounds/censor vectors, subject lists, trial logs, participant summaries,
random effects, fitted R model objects and bootstrap subject identities are excluded.
Individual trajectories/coverage heatmaps and their participant labels are excluded.
No MRI images, raw participant paths, participant input manifests or third-party
atlases/templates are distributed. Atlas/template resources must be obtained under
their own terms; AFNI, fMRIPrep, R and Workbench require separate installations.

Full re-estimation requires approved participant-data access. Requests go to the
corresponding authors and depend on consent/ethics/institutional requirements;
approval is not guaranteed. For an approved local copy, recreate the role-based
data/, resources/, manuscript/data/sensitivity_inputs/afni_v8_censor/, and config
subject-list inputs, set SRTT_ROOT to that copy, and consult code/common/srtt_paths.py.
First-level reference: code/glm_volume/90_run_float32_eventrelated.py;
censor/filter variants: 95_run_motion_filter_sensitivity.py;
ROI support/inference: code/reporting/augment_roi_reporting.py;
RT: prepare_fmri_rt.R and fit_fmri_rt_lme.R;
errors: analyze_block_errors.py and fit_error_mixed_models.R.
Some provenance validators also need internal archived sources/manifests; they are
supplied for transparency, not advertised as executable using this public subset.
Historical script identifiers/docstrings (e.g. primary or prespecified) do not
establish preregistration. The censored high-pass reporting model was selected
post hoc based on significant-result counts. Q values and pointwise intervals
do not account for this selection. Whole-brain summaries use the uncensored
reference model and are not independent confirmation of censored ROI results.

Archival workflow: enable this repository in the Zenodo GitHub integration,
publish the GitHub release, and cite the verified Zenodo version DOI. Metadata
for automatic archival are supplied in .zenodo.json; CITATION.cff supports GitHub
citation export. The file-level license map in LICENSE.txt governs reuse.
GitHub integration does not require creating or sharing a personal access token.
Do not upload the existing private whole project tree or participant-level data.
