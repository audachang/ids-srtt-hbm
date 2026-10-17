#!/usr/bin/env python3
"""Build an allowlisted local release candidate; never upload or publish."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from coverage_significant_table import render, roi_name

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'submissions/hbm_public_release_candidate_2026-10-17'
REDACTIONS = []
INVENTORY = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(relative):
    source = ROOT / relative
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def group_csv(relative, units, description):
    source = ROOT / relative
    frame = pd.read_csv(source)
    forbidden = {'subject', 'participant', 'participant_id', 'subject_id', 'filename', 'file_path', 'path'}
    assert not (set(frame.columns.str.lower()) & forbidden), relative
    # A mean with N=1 is an individual value, even in a group-summary file.
    if 'n' in frame and 'mean' in frame:
        mask = frame.n < 2
        fields = [x for x in ('mean', 'sd', 'se', 'ci_low', 'ci_high', 'cohen_dz', 't', 'p') if x in frame]
        modified = int(frame.loc[mask, fields].notna().sum().sum())
        frame.loc[mask, fields] = np.nan
        if modified:
            REDACTIONS.append(dict(path=relative, source_sha256=sha(source), rows=int(mask.sum()),
                                   cells_suppressed=modified, reason='Suppress estimates with fewer than two participants; retain N, planned slots and BH q values.'))
    if relative.endswith('/model_stability.csv'):
        mask = frame.n_testable == 0
        fields = ['mean_min', 'mean_max', 'ci_width_min', 'ci_width_max', 'same_direction']
        modified = int(frame.loc[mask, fields].notna().sum().sum())
        frame['same_direction'] = frame.same_direction.astype(object)
        frame.loc[mask, fields] = np.nan
        REDACTIONS.append(dict(path=relative, source_sha256=sha(source), rows=int(mask.sum()),
                               cells_suppressed=modified, reason='Suppress estimate ranges and direction for parcels untestable in every model, including singleton-based summaries.'))
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    INVENTORY.append(dict(path=relative, rows=len(frame), units=units, description=description))


def profiles():
    base = DEST / 'results/glm_volume/reporting_20261012'
    data = pd.read_csv(base / 'reported_roi_results.csv')
    component = pd.read_csv(base / 'phase_component_results.csv')
    names = list(data.roi.unique()); stages = ['initial', 'early', 'middle', 'late']
    target = base / 'all_roi_condition_profiles.pdf'
    with PdfPages(target) as pdf:
        for start in range(0, len(names), 4):
            fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.5))
            for row, roi in enumerate(names[start:start + 4]):
                for col, source in enumerate((component, data)):
                    ax = axes[row, col]
                    if col == 0:
                        series = [(kind, color) for kind, color in [('structured', '#8050a3'), ('random', '#555555')]]
                    else:
                        series = [(None, '#218c72')]
                    observed = False
                    for kind, color in series:
                        x = source[source.roi == roi]
                        if kind is not None:
                            x = x[x.component == kind]
                        x = x.set_index('stage').reindex(stages)
                        if x['mean'].notna().any():
                            observed = True
                            ax.errorbar(range(4), x['mean'],
                                        yerr=np.vstack([x['mean'] - x.ci_low, x.ci_high - x['mean']]),
                                        fmt='o-', color=color, label=kind, ms=3, lw=.8)
                    if not observed:
                        ax.text(.5, .5, 'No estimable group summary', ha='center', transform=ax.transAxes, fontsize=9)
                    side, name = roi_name(roi)
                    ax.set_title(side + ' ' + name, fontsize=10)
                    ax.set_xticks(range(4), [s.title() for s in stages], fontsize=8)
                    ax.tick_params(axis='y', labelsize=8); ax.axhline(0, lw=.5, color='.7')
                    ax.set_ylabel('Two-run mean coefficient' if col == 0 else 'Initial / summed DiD', fontsize=8)
                    if col == 0 and observed:
                        ax.legend(fontsize=7, frameon=False)
            fig.suptitle('Observed support, 80% coverage; pointwise 95% CI\nN < 2 estimates suppressed; model selected post hoc', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .95)); pdf.savefig(fig); plt.close(fig)


def main():
    assert not DEST.exists(), 'Keep existing snapshots; archive the prior candidate before rebuilding.'
    DEST.mkdir(parents=True)
    # Source code only: no data directories, subject list, credentials, models or archives.
    for folder in ('code/reporting', 'code/glm_volume', 'code/common'):
        for path in sorted((ROOT / folder).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.R', '.json') and path.name != 'migration_source_hashes.json':
                copy(str(path.relative_to(ROOT)))
    copy('code/replication/config/rois.txt')
    copy('code/replication/requirements-nilearn.txt')
    imaging = 'results/glm_volume/reporting_20261012/'
    for name in ('reported_roi_results', 'all_model_roi_results', 'direct_stage_comparisons',
                 'phase_component_results', 'model_stage_summary', 'model_stability', 'roi_coverage_summary'):
        group_csv(imaging + name + '.csv',
                  'Percent-signal-scaled coefficient units; two-run sums for contrasts, two-run means for phase components; statistics and coverage fractions dimensionless.',
                  'Aggregate ROI inference, coverage or sensitivity; no participant rows.')
    group_csv('results/glm_volume/whole_brain/components_peaks.csv',
              'Coordinates in MNI mm; extents in 3-mm voxels; test statistics dimensionless.',
              'Connected components of TFCE/FWE-corrected voxels from the uncensored reference model; not the selected censored ROI model.')
    rt = 'results/behavioral/fmri_rt_20261012/'
    for name in ('group_cycles', 'run_summary', 'learning_benefit_summary', 'preprocessing_summary'):
        group_csv(rt + name + '.csv', 'RT in milliseconds; accuracy/omission measures as proportions; counts as labeled.', 'Imaging-cohort group behavioral summaries.')
    for policy in ('primary', 'min100'):
        for suffix in ('cell_estimates', 'fixed_coefficients', 'fixed_covariance', 'contrast_weights', 'contrasts', 'run_condition_wald'):
            group_csv(rt + f'lme/{policy}_{suffix}.csv',
                      'RT model on natural-log seconds; geometric RT columns in ms; exponentiated contrasts are ratios.',
                      'Population fixed effects, covariance, contrasts and omnibus test; participant random effects excluded.')
    group_csv(rt + 'lme/cell_metadata.csv', 'Run/condition labels.', 'Design-cell metadata without participant identifiers.')
    error = 'results/behavioral/block_errors_20261013/'
    for name in ('group_block_error_rates', 'group_run_error_rates', 'paired_contrasts'):
        group_csv(error + name + '.csv', 'Rates as proportions; difference_pp and CI_pp in percentage points.', 'Group error/omission summaries and paired contrasts.')
    for name in ('binomial_contrasts', 'gaussian_contrasts', 'binomial_cell_estimates', 'gaussian_cell_estimates',
                 'binomial_fixed_coefficients', 'gaussian_fixed_coefficients', 'binomial_fixed_covariance_bootstrap',
                 'gaussian_fixed_covariance_bootstrap', 'contrast_weights', 'cell_metadata', 'run_condition_wald'):
        group_csv(error + 'mixed_model/' + name + '.csv',
                  'Binomial coefficients on log-odds scale, odds_ratio dimensionless, marginal probabilities as proportions; Gaussian coefficients as proportions, pp columns as percentage points.',
                  'Population model summaries; no participant diagnostics, resampling identities or fitted model objects.')
    environment = DEST / 'environment'; environment.mkdir()
    for source, name in [('results/glm_volume/model_diagnostics_20260921/full_0131/environment.freeze.txt', 'python_environment.freeze.txt'),
                         (rt + 'lme/R_session_info.txt', 'rt_R_session_info.txt'),
                         (error + 'mixed_model/R_session_info.txt', 'error_R_session_info.txt')]:
        shutil.copy2(ROOT / source, environment / name)
    spec = json.loads((ROOT / imaging / 'analysis_spec.json').read_text())
    spec.pop('source_root_default', None)
    (DEST / imaging / 'analysis_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    render(DEST / imaging / 'reported_roi_results.csv', DEST / 'tables/observed_significant_rois.tex')
    profiles()
    (DEST / 'AGGREGATION_NOTES.json').write_text(json.dumps(REDACTIONS, indent=2) + '\n')
    with (DEST / 'FILE_DICTIONARY.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'rows', 'units', 'description'])
        writer.writeheader(); writer.writerows(INVENTORY)
    common = {'n': 'Actual eligible participant count for this ROI or summary.', 'df': 'Degrees of freedom; one-sample imaging tests use N-1.',
              'mean': 'Arithmetic group mean, suppressed for N<2; see file dictionary for units.',
              'sd': 'Sample SD (N-1 denominator).', 'se': 'Standard error of group mean.',
              'ci_low': 'Lower pointwise 95% interval; see model-specific scale in file dictionary.',
              'ci_high': 'Upper pointwise 95% interval; see model-specific scale in file dictionary.',
              'cohen_dz': 'Paired/one-sample standardized effect, mean/SD.', 't': 't statistic.', 'p': 'Unadjusted two-sided probability.',
              'q_within_contrast': 'BH across 48 planned slots within model, coverage policy and stage.',
              'q_joint_stages': 'BH across 192 planned slots within model and coverage policy.',
              'q_enumerated_models_stages': 'Exploratory BH across 1728 enumerated slots within coverage policy; not a correction for model/ROI/policy selection.',
              'q_three_pairs_144': 'BH across three direct stage pairs x 48 planned slots within policy.',
              'q_within_pair_48': 'Supplementary BH across 48 slots for one direct pair and policy.',
              'q_bh_24': 'BH across the 24 planned behavioral model contrasts within model/policy.',
              'model': 'Model specification identifier; legacy primary/nocensor naming denotes the uncensored reference, not the current ROI reporting model.',
              'policy': 'Coverage or RT validity policy; consult analysis specification.',
              'roi': 'Atlas parcel identifier. SMA legacy names denote the whole BA6 parcel, not an SMA-specific subdivision.',
              'stage': 'Initial, early, middle or late contrast; the initial contrast is a within-phase condition difference.',
              'pair': 'Direct subtraction of the named phases.', 'direction': 'Sign of the group estimate.'}
    (DEST / 'COLUMN_DICTIONARY.json').write_text(json.dumps(common, indent=2) + '\n')
    (DEST / 'README.txt').write_text('''IDS-SRTT HBM companion — local release candidate 0.1.0-rc1 (2026-10-17)

Scope: analysis source code and group summaries only. This folder has NOT been published.
Recommended deposit: Zenodo for a versioned archival record; optionally GitHub for maintained code.

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

Before public release: confirm author/rights-holder licenses, create the repository
record, verify metadata, reserve and insert the version DOI, and replace the pending
availability statements in the manuscript. Suggested licenses for author approval:
MIT for original software and CC BY 4.0 for group data/documentation. No license has
been applied by this preparation. Do not upload the existing whole project tree.
''')
    (DEST / 'LICENSE_PENDING.txt').write_text('Local review candidate only. No new reuse license has been granted. Author/rights-holder confirmation is needed before a public deposit; suggested MIT for original software and CC BY 4.0 for group summaries/documentation. Third-party dependencies retain their own licenses.\n')
    metadata = dict(title='Analysis code and group summaries for Neural responses during motor statistical learning and gradual sequence updating',
                    version='0.1.0-rc1', publication_status='Local unpublished release candidate',
                    creators=[dict(name='Chan, Chiao-En'), dict(name='Chang, Erik Chihhung', orcid='0000-0002-1813-9663'), dict(name='Wu, Denise Hsien')],
                    doi=None, public_url=None, license='Pending author/rights-holder confirmation',
                    scope='Original source code and group summaries; participant-level data excluded',
                    keywords=['motor sequence learning', 'statistical learning', 'fMRI', 'coverage-aware ROI', 'IDS-SRTT'])
    (DEST / 'DEPOSIT_METADATA_DRAFT.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest = [dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(DEST.rglob('*')) if p.is_file()]
    (DEST / 'SHA256_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                handle.write(p, str(Path(DEST.name) / p.relative_to(DEST)))
    print(json.dumps(dict(folder=str(DEST), archive=str(archive), files=len(manifest) + 1,
                          group_csv_files=len(INVENTORY), singleton_redactions=REDACTIONS), indent=2))


if __name__ == '__main__':
    main()
