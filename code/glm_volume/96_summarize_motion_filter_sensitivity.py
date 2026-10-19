#!/usr/bin/env python3
"""Summarize ROI robustness across the 2x2 censoring/filtering models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from nilearn.image import resample_to_img
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, FIRST_LEVEL_DIR, ROI_MASK_DIR

from _layout import FLA_DIR, read_rois, read_subjects  # noqa: E402


PRIMARY_ROOT = (
    FIRST_LEVEL_DIR / "FLA_nilearn_float32_20260814"
)
SENSITIVITY_BASE = (
    FIRST_LEVEL_DIR / "FLA_nilearn_sensitivity_20260921"
)
DEFAULT_OUT_DIR = MANUSCRIPT_DIR / "data" / "sensitivity"
# ROI_MASK_DIR comes from the canonical layout.

MODELS = {
    "nocensor_hp128_primary": PRIMARY_ROOT,
    "censor_hp128": SENSITIVITY_BASE / "censor_hp128",
    "nocensor_bandpass": SENSITIVITY_BASE / "nocensor_bandpass",
    "censor_bandpass": SENSITIVITY_BASE / "censor_bandpass",
}
CONTRASTS = (
    "Str1_minus_Ran_r12",
    "Switch_r34_vs_r12_did",
    "Switch_r56_vs_r12_did",
    "Switch_r78_vs_r12_did",
)
CONTRAST_SHORT = {
    "Str1_minus_Ran_r12": "initial",
    "Switch_r34_vs_r12_did": "early",
    "Switch_r56_vs_r12_did": "middle",
    "Switch_r78_vs_r12_did": "late",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-5)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def load_masks(rois: list[str]) -> dict[str, nib.Nifti1Image]:
    result: dict[str, nib.Nifti1Image] = {}
    for roi in rois:
        path = ROI_MASK_DIR / f"{roi}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(path)
        if not np.any(np.asarray(image.dataobj) > 0):
            raise RuntimeError(f"empty ROI mask: {path}")
        result[roi] = image
    return result


def extract_values(
    root: Path,
    subjects: list[str],
    rois: list[str],
    masks: dict[str, nib.Nifti1Image],
    contrast: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = np.empty((len(subjects), len(rois)), dtype=np.float64)
    manifest: list[dict[str, Any]] = []
    for si, subject in enumerate(subjects):
        path = (
            root
            / "subject_contrasts"
            / "eventrelated"
            / subject
            / f"{contrast}.nii.gz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(path)
        if image.header.get_data_dtype() != np.dtype("float32"):
            raise RuntimeError(f"not float32: {path}")
        native = np.asarray(image.dataobj, dtype=np.float64)
        if not np.isfinite(native).all():
            raise RuntimeError(f"non-finite map: {path}")
        cache: dict[tuple[Any, ...], np.ndarray] = {}
        for ri, roi in enumerate(rois):
            mask_img = masks[roi]
            key = (mask_img.shape[:3], tuple(np.round(mask_img.affine.ravel(), 5)))
            if key not in cache:
                if same_grid(image, mask_img):
                    cache[key] = native
                else:
                    resampled = resample_to_img(
                        image,
                        mask_img,
                        interpolation="continuous",
                        copy_header=True,
                        force_resample=True,
                    )
                    cache[key] = np.asarray(resampled.dataobj, dtype=np.float64)
            mask = np.asarray(mask_img.dataobj) > 0
            values[si, ri] = float(np.mean(cache[key][mask]))
        manifest.append(
            {
                "subject": subject,
                "contrast": contrast,
                "path": str(path.relative_to(PACKAGE_ROOT)),
                "sha256": sha256_file(path),
            }
        )
    return values, manifest


def infer_rois(
    model: str,
    contrast: str,
    subjects: list[str],
    rois: list[str],
    values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for ri, roi in enumerate(rois):
        sample = values[:, ri]
        result = stats.ttest_1samp(sample, 0.0, alternative="two-sided")
        sd = float(np.std(sample, ddof=1))
        row = {
            "model": model,
            "contrast": contrast,
            "stage": CONTRAST_SHORT[contrast],
            "roi": roi,
            "n": len(subjects),
            "mean": float(np.mean(sample)),
            "sd": sd,
            "cohen_dz": float(np.mean(sample) / sd) if sd > 0 else float("nan"),
            "t": float(result.statistic),
            "df": len(subjects) - 1,
            "p_two_sided": float(result.pvalue),
        }
        rows.append(row)
        p_values.append(float(result.pvalue))
    q_values = bh_adjust(np.asarray(p_values))
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_bh_within_contrast"] = float(q_value)
        row["significant_fdr_0_05"] = bool(q_value <= 0.05)
        row["direction"] = "positive" if row["mean"] > 0 else "negative"
    return rows


def write_report(
    out_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    censor_qc: dict[str, Any],
) -> None:
    lines = [
        "# Nilearn motion-censoring and temporal-filter sensitivity",
        "",
        f"Generated: **{utc_now()}**.",
        "",
        "This is a prespecified 2×2 robustness analysis. The validated manuscript",
        "model is the no-censor/128-s-high-pass cell. The three sensitivity cells",
        "change only the saved AFNI v8 censor mask and/or temporal filter. The",
        "band-pass cells use a 0.0078–0.1-Hz Fourier nuisance basis spanning the",
        "same subspace as AFNI `1dBport`; all other first-level settings and all",
        "four subject-level contrasts are held fixed.",
        "",
        "## Censor-mask QC",
        "",
        f"The archived AFNI masks censor **{censor_qc['n_censored']} of "
        f"{censor_qc['n_total']} volumes ({100*censor_qc['fraction_censored']:.2f}%)**. "
        f"The minimum retained in any run is **{censor_qc['minimum_retained_per_run']}/200**.",
        "",
        "## ROI results",
        "",
        "FDR-significant ROI counts (BH across 48 ROIs within each model and contrast):",
        "",
        "| Model | Initial | Early | Middle | Late |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = summary[summary["model"] == model].set_index("stage")
        lines.append(
            f"| {model} | {int(row.loc['initial','n_fdr'])} | "
            f"{int(row.loc['early','n_fdr'])} | {int(row.loc['middle','n_fdr'])} | "
            f"{int(row.loc['late','n_fdr'])} |"
        )
    lines.extend(
        [
            "",
            "A sensitivity-stable ROI has the same effect direction in all four",
            "cells and survives within-contrast FDR in all four. Counts:",
            "",
            "| Stage | Same direction | FDR in all four |",
            "|---|---:|---:|",
        ]
    )
    for stage in ("initial", "early", "middle", "late"):
        frame = stability[stability["stage"] == stage]
        lines.append(
            f"| {stage.title()} | {int(frame['same_direction_all_models'].sum())} | "
            f"{int(frame['fdr_all_models'].sum())} |"
        )
    lines.extend(
        [
            "",
            "This robustness rule is descriptive and does not replace the inferential",
            "correction within each model. Whole-brain permutation inference remains",
            "the validated primary analysis; this sensitivity report evaluates ROI",
            "stability before any decision to rerun expensive whole-brain permutations.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    marker = out_dir / "provenance.json"
    if marker.exists() and not args.overwrite:
        raise SystemExit(f"{marker} exists; use --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = read_subjects()
    rois = read_rois()
    masks = load_masks(rois)
    all_values: dict[tuple[str, str], np.ndarray] = {}
    manifests: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for model, root in MODELS.items():
        for contrast in CONTRASTS:
            values, manifest = extract_values(root, subjects, rois, masks, contrast)
            all_values[(model, contrast)] = values
            manifests.extend({"model": model, **row} for row in manifest)
            result_rows.extend(infer_rois(model, contrast, subjects, rois, values))

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "roi_results.csv", index=False, float_format="%.12g")
    pd.DataFrame(manifests).to_csv(out_dir / "subject_map_manifest.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for contrast in CONTRASTS:
            frame = results[(results.model == model) & (results.contrast == contrast)]
            summary_rows.append(
                {
                    "model": model,
                    "contrast": contrast,
                    "stage": CONTRAST_SHORT[contrast],
                    "n_fdr": int(frame.significant_fdr_0_05.sum()),
                    "n_positive_fdr": int(
                        (frame.significant_fdr_0_05 & (frame["mean"] > 0)).sum()
                    ),
                    "n_negative_fdr": int(
                        (frame.significant_fdr_0_05 & (frame["mean"] < 0)).sum()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "model_contrast_summary.csv", index=False)

    stability_rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for roi in rois:
            frame = results[(results.contrast == contrast) & (results.roi == roi)].set_index(
                "model"
            )
            signs = np.sign(frame.loc[list(MODELS), "mean"].to_numpy())
            sig = frame.loc[list(MODELS), "significant_fdr_0_05"].to_numpy(bool)
            stability_rows.append(
                {
                    "contrast": contrast,
                    "stage": CONTRAST_SHORT[contrast],
                    "roi": roi,
                    "same_direction_all_models": bool(np.all(signs == signs[0])),
                    "fdr_all_models": bool(np.all(sig) and np.all(signs == signs[0])),
                    "n_models_fdr": int(sig.sum()),
                }
            )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out_dir / "roi_stability.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    pairs = {
        "censor_effect_hp128": ("nocensor_hp128_primary", "censor_hp128"),
        "filter_effect_nocensor": ("nocensor_hp128_primary", "nocensor_bandpass"),
        "censor_effect_bandpass": ("nocensor_bandpass", "censor_bandpass"),
        "filter_effect_censored": ("censor_hp128", "censor_bandpass"),
    }
    for label, (left, right) in pairs.items():
        for contrast in CONTRASTS:
            a = all_values[(left, contrast)].ravel()
            b = all_values[(right, contrast)].ravel()
            pair_rows.append(
                {
                    "comparison": label,
                    "left_model": left,
                    "right_model": right,
                    "contrast": contrast,
                    "stage": CONTRAST_SHORT[contrast],
                    "subject_roi_pearson_r": float(stats.pearsonr(a, b).statistic),
                    "mean_absolute_difference": float(np.mean(np.abs(a - b))),
                    "median_absolute_difference": float(np.median(np.abs(a - b))),
                }
            )
    pd.DataFrame(pair_rows).to_csv(out_dir / "pairwise_agreement.csv", index=False)

    censor_counts: list[int] = []
    censor_manifest: list[dict[str, Any]] = []
    for subject in subjects:
        path = (
            MANUSCRIPT_DIR
            / "data"
            / "sensitivity_inputs"
            / "afni_v8_censor"
            / f"{subject}_afni_v8_censor.1D"
        )
        values = np.loadtxt(path).reshape(8, 200)
        subject_censored = int(np.count_nonzero(values == 0))
        censor_counts.extend(int(np.count_nonzero(run == 0)) for run in values)
        censor_manifest.append(
            {
                "subject": subject,
                "path": str(path.resolve().relative_to(PACKAGE_ROOT)),
                "sha256": sha256_file(path),
                "n_values": int(values.size),
                "n_censored": subject_censored,
                "n_retained": int(values.size - subject_censored),
            }
        )
    pd.DataFrame(censor_manifest).to_csv(
        out_dir / "censor_mask_manifest.csv", index=False
    )
    censor_qc = {
        "n_total": len(subjects) * 8 * 200,
        "n_censored": int(sum(censor_counts)),
        "fraction_censored": sum(censor_counts) / (len(subjects) * 8 * 200),
        "minimum_retained_per_run": int(200 - max(censor_counts)),
    }
    write_report(out_dir, results, summary, stability, censor_qc)

    provenance = {
        "created_utc": utc_now(),
        "analysis": "nilearn_2x2_motion_censor_temporal_filter_roi_sensitivity",
        "models": {name: str(path.relative_to(PACKAGE_ROOT)) for name, path in MODELS.items()},
        "subjects": subjects,
        "n_subjects": len(subjects),
        "rois": rois,
        "n_rois": len(rois),
        "contrasts": list(CONTRASTS),
        "censor_qc": censor_qc,
        "censor_mask_manifest": str(
            (out_dir / "censor_mask_manifest.csv").resolve().relative_to(PACKAGE_ROOT)
        ),
        "multiplicity": "BH FDR across 48 ROIs separately within model and contrast",
        "sensitivity_stability_rule": "same direction and q<=0.05 in all four 2x2 cells",
        "software": {
            "python": platform.python_version(),
            "nilearn": nilearn.__version__,
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    marker.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
