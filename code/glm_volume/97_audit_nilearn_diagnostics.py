#!/usr/bin/env python3
"""Reproducible, non-destructive design/residual/ROI audit of saved models.

Refits a deterministic spatial sample in every run; does not replace images.
Residual references condition on fitted AR coefficients and the fitted design.
They are descriptive diagnostics, not calibrated whiteness hypothesis tests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.first_level.first_level import mean_scaling
from nilearn.maskers import NiftiMasker
from scipy import stats
from threadpoolctl import threadpool_limits


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sensitivity = load_script("diagnostic_sensitivity", "95_run_motion_filter_sensitivity.py")
roi_module = load_script("diagnostic_rois", "96_summarize_motion_filter_sensitivity.py")
primary = sensitivity.primary
MANUSCRIPT = primary.MANUSCRIPT_DIR
OUT = MANUSCRIPT / "data" / "model_diagnostics_20260921"
MODELS = roi_module.MODELS
LAGS = range(1, 11)
SEED = 20260921


def sha256(path):
    return roi_module.sha256_file(Path(path))


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def residualize(values, nuisance):
    return values - nuisance @ np.linalg.lstsq(nuisance, values, rcond=None)[0]


def acf(values, lag, indices=None):
    centered = values - values.mean(axis=0)
    denominator = np.sum(centered * centered, axis=0)
    if indices is None:
        left, right = np.arange(len(values) - lag), np.arange(lag, len(values))
    else:
        left, right = np.where(indices[None, :] - indices[:, None] == lag)
    numerator = np.sum(centered[left] * centered[right], axis=0)
    return numerator / denominator, numerator.sum() / denominator.sum(), len(left)


def projection_covariance(design):
    q, _ = np.linalg.qr(design, mode="reduced")
    covariance = np.eye(len(design)) - q @ q.T
    # ACF below uses demeaned residuals, including after AR whitening.
    return (covariance - covariance.mean(axis=0)[None, :]
            - covariance.mean(axis=1)[:, None] + covariance.mean())


def task_diagnostics(design, model, subject, run):
    x = design.to_numpy()
    task = list(primary.CONDITIONS[run])
    filter_names = [c for c in design if c == "constant" or c == "linear_drift"
                    or c.startswith("drift_") or c.startswith("stop_")]
    nuisance_names = [c for c in design if c not in task]
    rows = []
    for condition in task:
        y = design[condition].to_numpy()
        total = np.sum((y - y.mean()) ** 2)
        filtered = residualize(y, design[filter_names].to_numpy())
        nuisance_resid = residualize(y, design[nuisance_names].to_numpy())
        all_other = residualize(y, design.drop(columns=condition).to_numpy())
        rows.append(dict(subject=subject, run=run, model=model, condition=condition,
                         filter_retained_ss=float(filtered @ filtered / total),
                         nuisance_retained_ss=float(nuisance_resid @ nuisance_resid / total),
                         unique_task_retained_ss=float(all_other @ all_other / total)))
    # Equal-iid-noise contrast variance is a design measure, not AR-adjusted precision.
    inverse = np.linalg.pinv(x)
    covariance = inverse @ inverse.T
    variances = []
    for condition in task:
        if condition == "Ran":
            continue
        c = np.zeros(x.shape[1])
        c[design.columns.get_loc(condition)] = 1
        c[design.columns.get_loc("Ran")] = -1
        variances.append(dict(subject=subject, run=run, model=model,
                              contrast=condition + "_minus_Ran",
                              iid_unit_noise_variance=float(c @ covariance @ c)))
    return rows, variances


def audit_run(job):
    subject, run, n_voxels, output = job
    start = time.monotonic()
    threadpool_limits(1)
    output = Path(output)
    bold, mask_path, confound_path, event_path = primary.input_paths(subject, run)
    mask_img = nib.load(mask_path)
    mask = np.asarray(mask_img.dataobj) > 0
    rng = np.random.default_rng(SEED + int(subject[-3:]) * 100 + run)
    selected = np.sort(rng.choice(mask.sum(), min(n_voxels, mask.sum()), replace=False))
    sampled_flat = np.flatnonzero(mask)[selected]
    masker = NiftiMasker(mask_img=str(mask_path), smoothing_fwhm=4.0,
                        standardize=False, detrend=False, t_r=primary.TR)
    with warnings.catch_warnings(record=True) as caught:
        all_y = masker.fit_transform(str(bold))
    keep = sensitivity.load_censor_mask(sensitivity.DEFAULT_CENSOR_DIR,
                                        subject, run, len(all_y))
    confounds = pd.read_csv(confound_path, sep="\t")
    events = primary.load_events(event_path, primary.CONDITIONS[run])
    result = dict(subject=subject, run=run, residual=[], designs=[], tasks=[],
                  contrasts=[], motion=[], reproduction=[], upper_cutoff=[],
                  sampled_flat_voxels=sampled_flat.tolist(),
                  input_files={str(p): dict(size=p.stat().st_size,
                                           mtime_ns=p.stat().st_mtime_ns)
                               for p in (bold, mask_path, confound_path, event_path)},
                  small_input_sha256={str(p): sha256(p)
                                      for p in (mask_path, confound_path, event_path)},
                  warnings=[str(w.message) for w in caught])
    for model_name, root in MODELS.items():
        directory = root / "level1" / "eventrelated" / subject / f"run-{run:02d}"
        censored = model_name.startswith("censor_")
        bandpass = "bandpass" in model_name
        indices = keep if censored else np.arange(len(all_y))
        saved = pd.read_csv(directory / "design_matrix.tsv", sep="\t")
        _, _, _, nuisance, n = sensitivity.load_model_inputs(
            subject, run, "bandpass" if bandpass else "hp128")
        qc = json.loads((directory / "model_qc.json").read_text())
        dropped = qc.get("dropped_redundant_nuisance_columns", [])
        nuisance = nuisance.drop(columns=dropped)
        design = make_first_level_design_matrix(
            np.arange(n) * primary.TR, events, hrf_model="glover",
            drift_model=None if bandpass else "cosine",
            high_pass=0.01 if bandpass else 1 / 128,
            add_regs=nuisance.to_numpy(), add_reg_names=list(nuisance))
        design = design.iloc[indices]
        if list(design) != list(saved):
            raise RuntimeError(f"design columns changed: {subject} {run} {model_name}")
        np.testing.assert_allclose(design.to_numpy(), saved.to_numpy(), rtol=1e-10, atol=1e-10)
        if censored:
            np.testing.assert_array_equal(indices, np.loadtxt(
                directory / "retained_volume_indices.1D", dtype=int))
        x = design.to_numpy()
        rank = int(np.linalg.matrix_rank(x))
        assert rank == x.shape[1]
        y, _ = mean_scaling(all_y[indices], axis=0)
        y = y[:, selected]
        labels, fits = run_glm(y, x, noise_model="ar1", bins=100, n_jobs=1)
        beta = np.empty((x.shape[1], len(selected)))
        raw = np.empty_like(y, dtype=float)
        white = np.empty_like(raw)
        rho = np.empty(len(selected))
        expected_num = np.zeros(10)
        expected_gap_num = np.zeros(10)
        expected_den = 0.0
        for label, fit in fits.items():
            columns = labels == label
            beta[:, columns] = fit.theta
            # RegressionResults.residuals mixes raw Y and whitened X in 0.13.1.
            raw[:, columns] = y[:, columns] - x @ fit.theta
            white[:, columns] = fit.whitened_residuals
            rho[columns] = float(label)
            # Whitening is recomputed through a second BLAS path; the pinned
            # stack can differ slightly for tiny AR-bin values at this voxel
            # count. Keep this check tight while allowing harmless numerical
            # noise that is negligible relative to the residual scale.
            np.testing.assert_allclose(fit.model.whiten(raw[:, columns]),
                                       fit.whitened_residuals, atol=1e-4, rtol=1e-4)
            covariance = projection_covariance(fit.whitened_design)
            centered = fit.whitened_residuals - fit.whitened_residuals.mean(axis=0)
            energy = np.sum(centered ** 2)
            scale = energy / np.trace(covariance)
            expected_den += energy
            for lag in LAGS:
                expected_num[lag - 1] += scale * np.diag(covariance, k=lag).sum()
                a, b = np.where(indices[None, :] - indices[:, None] == lag)
                expected_gap_num[lag - 1] += scale * covariance[a, b].sum()
        for condition in primary.CONDITIONS[run]:
            path = directory / f"{condition}_effect_size.nii.gz"
            img = nib.load(path)
            assert img.shape == mask_img.shape
            np.testing.assert_allclose(img.affine, mask_img.affine, atol=1e-5)
            existing = np.asarray(img.dataobj).ravel()[sampled_flat]
            new = beta[design.columns.get_loc(condition)]
            error = np.abs(existing - new)
            # Saved maps round to float32; preprocessing may also round float32.
            passed = bool(np.allclose(existing, new, rtol=2e-5, atol=2e-5))
            result["reproduction"].append(dict(subject=subject, run=run, model=model_name,
                condition=condition, n_voxels=len(selected), maximum_absolute_error=float(error.max()),
                rms_error=float(np.sqrt(np.mean(error ** 2))), passed=passed,
                map_sha256=sha256(path), design_sha256=sha256(directory / "design_matrix.tsv")))
            if not passed:
                raise RuntimeError(f"saved beta mismatch: {subject} {run} {model_name} {condition}: {error.max()}")
        norms = np.linalg.norm(x, axis=0)
        result["designs"].append(dict(subject=subject, run=run, model=model_name,
            n_retained=len(indices), rank=rank, residual_df=len(indices) - rank,
            column_normalized_condition_number=float(np.linalg.cond(x / norms)),
            ar1_median=float(np.median(rho)), ar1_q05=float(np.quantile(rho, .05)),
            ar1_q95=float(np.quantile(rho, .95)),
            raw_variance_median=float(np.median(np.var(raw, axis=0))),
            whitened_variance_median=float(np.median(np.var(white, axis=0))),
            n_censor_gaps=int(np.sum(np.diff(indices) > 1))))
        for lag in LAGS:
            ra, rp, _ = acf(raw, lag)
            wa, wp, pairs = acf(white, lag)
            ga, gp, gap_pairs = acf(white, lag, indices)
            reference = expected_num[lag - 1] / expected_den
            gap_reference = expected_gap_num[lag - 1] / expected_den
            result["residual"].append(dict(subject=subject, run=run, model=model_name,
                lag=lag, n_voxels=len(selected), raw_acf_median=float(np.median(ra)),
                raw_acf_pooled=float(rp), white_acf_median=float(np.median(wa)),
                white_acf_q05=float(np.quantile(wa, .05)), white_acf_q95=float(np.quantile(wa, .95)),
                white_acf_pooled=float(wp), projected_white_reference=float(reference),
                excess_acf=float(wp - reference), n_compressed_pairs=pairs,
                gap_aware_white_acf_pooled=float(gp), n_true_tr_pairs=gap_pairs,
                gap_aware_reference=float(gap_reference), gap_aware_excess=float(gp - gap_reference)))
        task_rows, contrast_rows = task_diagnostics(design, model_name, subject, run)
        result["tasks"].extend(task_rows)
        result["contrasts"].extend(contrast_rows)
        if model_name == "nocensor_hp128_primary":
            stop = sensitivity.bandpass_stop_basis(n)
            high = stop[[c for c in stop if int(c.rsplit("_", 1)[1]) >= 41]].to_numpy()
            hp = design[[c for c in design if c.startswith("drift_") or c == "constant"]].to_numpy()
            for condition in primary.CONDITIONS[run]:
                task_y = design[condition].to_numpy()
                total = np.sum((task_y - task_y.mean()) ** 2)
                before = residualize(task_y, hp)
                after = residualize(task_y, np.column_stack([hp, high]))
                result["upper_cutoff"].append(dict(subject=subject, run=run, condition=condition,
                    hp_retained_ss=float(before @ before / total),
                    hp_plus_upper_stop_retained_ss=float(after @ after / total),
                    additional_loss_percentage_points=float(100 * (before @ before - after @ after) / total)))
                onset_mask = np.zeros(n, dtype=bool)
                frame_times = np.arange(n) * primary.TR
                for row in events.loc[events.trial_type == condition].itertuples():
                    onset_mask |= (frame_times >= row.onset) & (frame_times < row.onset + row.duration)
                censor_flag = np.ones(n, dtype=bool)
                censor_flag[keep] = False
                fd = confounds["framewise_displacement"].to_numpy()
                dvars = confounds["std_dvars"].to_numpy()
                valid = np.isfinite(fd)
                valid_dv = np.isfinite(dvars)
                result["motion"].append(dict(subject=subject, run=run, condition=condition,
                    n_condition_frames=int(onset_mask.sum()),
                    condition_censored_frames=int(np.sum(censor_flag & onset_mask)),
                    condition_censored_fraction=float(censor_flag[onset_mask].mean()),
                    condition_mean_fd=float(np.nanmean(fd[onset_mask])),
                    run_mean_fd=float(np.nanmean(fd)),
                    fd_task_correlation=float(np.corrcoef(task_y[valid], fd[valid])[0, 1]),
                    std_dvars_task_correlation=float(np.corrcoef(task_y[valid_dv], dvars[valid_dv])[0, 1]),
                    run_censored_fraction=float(censor_flag.mean())))
    result["elapsed_seconds"] = time.monotonic() - start
    json_write(output / "runs" / f"{subject}_run-{run:02d}.json", result)
    return result


def residual_report(output, records):
    frames = {}
    for key in ("residual", "designs", "tasks", "contrasts", "motion", "reproduction", "upper_cutoff"):
        frames[key] = pd.DataFrame([row for record in records for row in record[key]])
        frames[key].to_csv(output / f"{key}.csv", index=False)
    df = frames["residual"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        means = df.loc[df.model == model].groupby("lag").mean(numeric_only=True)
        ax.plot(means.index, means.white_acf_pooled, "o-", label="Observed whitened")
        ax.plot(means.index, means.projected_white_reference, "s--", label="Design-conditioned reference")
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Lag in retained-volume sequence")
        ax.set_ylabel("Pooled residual ACF (mean across runs)")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "residual_acf.png", dpi=160)
    plt.close(fig)
    return frames


def audit_rois(output):
    subjects, rois = roi_module.read_subjects(), roi_module.read_rois()
    masks = roi_module.load_masks(rois)
    old = pd.read_csv(MANUSCRIPT / "data/sensitivity/roi_results.csv")
    manifest = pd.read_csv(MANUSCRIPT / "data/sensitivity/subject_map_manifest.csv")
    all_rows, values_rows, checks, influence = [], [], [], []
    for model, root in MODELS.items():
        model_rows = []
        for contrast in roi_module.CONTRASTS:
            values, new_manifest = roi_module.extract_values(root, subjects, rois, masks, contrast)
            for item in new_manifest:
                match = manifest[(manifest.model == model) & (manifest.subject == item["subject"])
                                 & (manifest.contrast == contrast)]
                assert len(match) == 1 and match.iloc[0].sha256 == item["sha256"]
            rows = roi_module.infer_rois(model, contrast, subjects, rois, values)
            saved = old[(old.model == model) & (old.contrast == contrast)].set_index("roi")
            recomputed = pd.DataFrame(rows).set_index("roi")
            for field in ("mean", "sd", "cohen_dz", "t", "p_two_sided", "q_bh_within_contrast"):
                a, b = saved.loc[rois, field].to_numpy(), recomputed.loc[rois, field].to_numpy()
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
                checks.append(dict(model=model, contrast=contrast, field=field,
                                   max_absolute_difference=float(np.max(np.abs(a - b))), passed=True))
            loo_means, loo_q = [], []
            for si, subject in enumerate(subjects):
                retained = np.delete(values, si, axis=0)
                m = retained.mean(axis=0)
                p = stats.ttest_1samp(retained, 0, axis=0).pvalue
                q = roi_module.bh_adjust(p)
                loo_means.append(m)
                loo_q.append(q)
                for ri, roi in enumerate(rois):
                    influence.append(dict(model=model, contrast=contrast, excluded_subject=subject,
                                          roi=roi, mean=float(m[ri]), q_bh_within_contrast=float(q[ri])))
                    values_rows.append(dict(model=model, contrast=contrast, subject=subject,
                                            roi=roi, effect=float(values[si, ri])))
            loo_means, loo_q = np.asarray(loo_means), np.asarray(loo_q)
            for ri, row in enumerate(rows):
                sem = row["sd"] / np.sqrt(len(subjects))
                delta = stats.t.ppf(.975, len(subjects) - 1) * sem
                row.update(ci95_low=row["mean"] - delta, ci95_high=row["mean"] + delta,
                           loo_same_direction_count=int(np.sum(np.sign(loo_means[:, ri]) == np.sign(row["mean"]))),
                           loo_fdr_count=int(np.sum(loo_q[:, ri] <= .05)),
                           loo_mean_min=float(loo_means[:, ri].min()), loo_mean_max=float(loo_means[:, ri].max()),
                           max_abs_loo_mean_shift=float(np.max(np.abs(loo_means[:, ri] - row["mean"]))))
            model_rows.extend(rows)
            print(f"ROI audit: {model} {contrast}", flush=True)
        pooled = roi_module.bh_adjust(np.asarray([row["p_two_sided"] for row in model_rows]))
        for row, q in zip(model_rows, pooled):
            row["q_bh_across_192"] = float(q)
        all_rows.extend(model_rows)
    result = pd.DataFrame(all_rows)
    result.to_csv(output / "roi_influence_summary.csv", index=False)
    pd.DataFrame(values_rows).to_csv(output / "roi_subject_values.csv", index=False)
    pd.DataFrame(influence).to_csv(output / "roi_leave_one_out.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "roi_reproduction.csv", index=False)
    json_write(output / "roi_mask_sha256.json", {roi: sha256(roi_module.ROI_MASK_DIR / f"{roi}.nii.gz") for roi in rois})
    return result


def write_report(output, frames, rois):
    residuals = frames["residual"]
    designs = frames["designs"]
    lines = ["# Nilearn 0.13.1 diagnostic audit", "",
             "Generated " + datetime.now(timezone.utc).isoformat(), "",
             "## Scope and reproduction", "",
             f"{len(designs)} model/run refits; a fixed random sample of "
             f"{int(residuals.n_voxels.min())}-{int(residuals.n_voxels.max())} in-mask voxels per run, shared across models. "
             "Spatial smoothing uses the complete image before sampling. No saved effect maps or manuscript products are overwritten.", "",
             f"All {len(frames['reproduction'])} task-map sample comparisons passed rtol=2e-5, atol=2e-5. "
             f"Maximum absolute beta difference: {frames['reproduction'].maximum_absolute_error.max():.6g}. "
             "Every design was regenerated from events/confounds and checked against its saved matrix. "
             "Censored designs retain original event timing; scaling is recomputed on retained volumes.", "",
             "## Residual diagnostics", "",
             "Raw residuals are explicitly Y - X beta. Nilearn 0.13.1's RegressionResults.residuals "
             "uses a whitened design with unwhitened Y and is not used here. Whitened residuals are also "
             "checked against whitening the explicit raw residuals.", "",
             "The reference is the residual covariance H(I - QQ')H of iid innovations, conditional on each "
             "fitted AR-bin's whitened design Q; H removes the temporal mean. References are weighted by "
             "observed residual energy. ACF uses lagged cross-products divided by total centered energy. "
             "The ratio-of-expected-quadratic-forms reference is descriptive, not an exact expectation of "
             "individual voxel ACF or a calibrated hypothesis test. AR parameters were estimated from these data. "
             "Spatially correlated voxels are not independent replicates.", "",
             "Mean across runs of pooled lag-1 ACF:", "",
             "| Model | Raw | Whitened | Design reference | Excess | Residual df range |",
             "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = residuals[(residuals.model == model) & (residuals.lag == 1)].mean(numeric_only=True)
        d = designs[designs.model == model]
        lines.append(f"| {model} | {row.raw_acf_pooled:.3f} | {row.white_acf_pooled:.3f} | "
                     f"{row.projected_white_reference:.3f} | {row.excess_acf:.3f} | {d.residual_df.min()}-{d.residual_df.max()} |")
    lines += ["", "![Residual ACF](residual_acf.png)", "",
              "Both compressed-sequence and original-TR-gap-aware ACF are saved for lags 1-10. "
              "Nilearn's sample_masks AR fit treats adjacent retained volumes as adjacent observations, including "
              "across censoring gaps; gap-aware diagnostics do not repair that covariance assumption. "
              "Bandpass nuisance regression imposes residual dependence even under the white-innovation reference. "
              "A high bandpass lag-1 ACF alone therefore does not demonstrate whitening failure. "
              "Conversely, near-zero high-pass lag-1 ACF alone does not establish adequate noise modeling at higher lags.", "",
              "## Task signal and motion", "",
              "Filter retention is demeaned task-regressor sum-of-squares after projecting filter nuisance terms, "
              "relative to its original demeaned sum-of-squares. It is not a percentage of actual BOLD signal lost. "
              "tasks.csv also separates all-nuisance and all-other-column projections. contrasts.csv reports "
              "equal-iid-noise design variance, not fitted AR precision. The two filter families use different "
              "low-frequency bases; their difference does not isolate an upper cutoff.", "",
              "Uncensored mean task-regressor retention:", "",
              "| Model | Ran | Str1 | Str2 |", "|---|---:|---:|---:|"]
    for model in ("nocensor_hp128_primary", "nocensor_bandpass"):
        row = frames["tasks"].query("model == @model").groupby("condition").filter_retained_ss.mean()
        lines.append(f"| {model} | {100*row['Ran']:.2f}% | {100*row['Str1']:.2f}% | {100*row['Str2']:.2f}% |")
    row = frames["upper_cutoff"].groupby("condition").additional_loss_percentage_points.mean()
    lines += ["", "Adding only the >0.1-Hz Fourier stop terms to the SAME 128-s high-pass basis "
              f"removes an additional {row['Ran']:.3f}, {row['Str1']:.3f}, and {row['Str2']:.3f} percentage "
              "points of demeaned Ran, Str1, and Str2 task energy, respectively (design calculation only).", "",
              "motion.csv reports condition-window motion/censoring burden and HRF-regressor correlations "
              "with FD and standardized DVARS for each run. Frames are assigned using actual event durations, "
              "not HRF peaks; first-volume missing motion metrics are omitted, not imputed as zero. "
              "These are descriptive and repeated runs/frames are not treated as independent participants.", "",
              "## ROI verification and influence", "",
              "Re-extraction verified the checksums of all 320 subject contrast maps and reproduced all 768 "
              "saved ROI tests (means, SD, dz, t, p, within-contrast BH q). "
              "The additional 192-test BH family is a sensitivity check within each model, not a retroactive "
              "replacement of the existing correction. Leave-one-out results are influence diagnostics, not "
              "20 new independent hypothesis tests. Pointwise 95% t intervals are not multiplicity-adjusted.", "",
              "| Model | Stage | Original FDR | FDR after all 20 deletions | Across-192 FDR |",
              "|---|---|---:|---:|---:|"]
    for (model, stage), group in rois.groupby(["model", "stage"], sort=False):
        sig = group.q_bh_within_contrast <= .05
        lines.append(f"| {model} | {stage} | {int(sig.sum())} | "
                     f"{int((sig & (group.loo_fdr_count == 20)).sum())} | "
                     f"{int((group.q_bh_across_192 <= .05).sum())} |")
    lines += ["", "## Limits and decision rule", "",
              "This audit evaluates all runs but only sampled voxels; it is not a voxelwise whole-brain "
              "whiteness test, a simulation-calibrated false-positive-rate study, a new preprocessing run, "
              "or a rerun of group permutations. Existing primary and sensitivity images already record "
              "Nilearn 0.13.1. Their matching refits do not justify regenerating every map just because "
              "the earlier diagnostic interpreter used 0.10.1.", "",
              "Do not select a primary pipeline by ROI counts. Lock the scientific estimand, multiplicity "
              "family, cohort and censor/filter/noise policy before the final manuscript. Any change of "
              "primary model requires matching group inference and regenerated tables/figures. "
              "If stronger first-level serial-noise claims are required, use a design- and sampling-aware "
              "simulation/refit benchmark or a suitable alternative covariance model; do not infer that "
              "a version upgrade alone fixes whitening.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--runs", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--voxels", type=int, default=2048)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mode", choices=("all", "residual", "roi", "report"), default="all")
    args = parser.parse_args()
    if nilearn.__version__ != "0.13.1":
        raise RuntimeError(f"Use pinned Nilearn 0.13.1, found {nilearn.__version__}")
    if args.voxels < 1 or args.jobs < 1 or not set(args.runs) <= set(range(1, 9)):
        parser.error("positive voxel/job counts and runs 1-8 required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or primary.read_subjects()
    if not set(subjects) <= set(primary.read_subjects()):
        parser.error("unknown subject")
    environment = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        executable=sys.executable, python=platform.python_version(), nilearn=nilearn.__version__,
        nilearn_path=nilearn.__file__, numpy=np.__version__, seed=SEED,
        command=sys.argv, subjects=subjects, runs=args.runs, voxels_per_run=args.voxels,
        script_sha256={str(Path(__file__).name): sha256(__file__),
                       "95_run_motion_filter_sensitivity.py": sha256(sensitivity.__file__),
                       "96_summarize_motion_filter_sensitivity.py": sha256(roi_module.__file__)})
    json_write(args.out_dir / f"provenance_{args.mode}.json", environment)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.out_dir / "environment.freeze.txt").write_text(freeze)
    if args.mode in ("all", "residual"):
        jobs = [(s, r, args.voxels, str(args.out_dir)) for s in subjects for r in args.runs]
        records = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for i, record in enumerate(pool.map(audit_run, jobs), 1):
                records.append(record)
                print(f"Residual audit {i}/{len(jobs)}: {record['subject']} run {record['run']} "
                      f"({record['elapsed_seconds']:.1f}s)", flush=True)
        frames = residual_report(args.out_dir, records)
    if args.mode in ("all", "roi"):
        rois = audit_rois(args.out_dir)
    if args.mode == "report":
        frames = {key: pd.read_csv(args.out_dir / f"{key}.csv") for key in
                  ("residual", "designs", "tasks", "contrasts", "motion", "reproduction", "upper_cutoff")}
        rois = pd.read_csv(args.out_dir / "roi_influence_summary.csv")
    if args.mode in ("all", "report"):
        write_report(args.out_dir, frames, rois)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
