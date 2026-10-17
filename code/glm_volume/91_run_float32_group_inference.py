#!/usr/bin/env python3
"""Run the audited ROI and whole-brain group inference on float32 maps.

This program is the group-level companion to
``90_run_float32_eventrelated.py``.  It deliberately refuses the older,
quantized uint8 contrast maps.  The analysis is fixed to the 20-subject,
four-contrast event-related family and performs the following operations:

1. Intersect all 20 x 8 fMRIPrep run masks after nearest-neighbour
   resampling to the fixed sub-001 run-01 3-mm grid.
2. Validate that every subject contrast is an unscaled, finite float32 image
   on that exact grid.
3. Fit an explicit-mask intercept-only parametric second-level model and a
   two-sided TFCE permutation model (Nilearn 0.13.1).
4. Threshold the signed t statistic by TFCE voxelwise FWE p < .05/4.
5. Re-extract the 48 prespecified ROI means and apply two-sided one-sample
   t tests with Benjamini-Hochberg FDR separately within each contrast.
6. Write input manifests, QC/provenance, and descriptive 26-connected
   component/peak summaries.

The default output is ``results/glm_volume/whole_brain``.  A quick smoke
test can use ``--n-perm 10 --out-dir /tmp/srtt_group_smoke``; the manuscript
analysis should use the default 10,000 permutations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from nibabel.affines import apply_affine
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import resample_to_img
from scipy import stats
from scipy.ndimage import generate_binary_structure, label


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, FIRST_LEVEL_DIR, ROI_MASK_DIR

from _layout import (  # noqa: E402
    FLA_DIR,
    FMRIPREP_DIR,
    SPACE,
    TASK_NAME,
    read_rois,
    read_subjects,
)


N_RUNS = 8
EXPECTED_N_SUBJECTS = 20
EXPECTED_N_ROIS = 48
EXPECTED_REFERENCE_SHAPE = (53, 65, 56)
EXPECTED_REFERENCE_ZOOMS_MM = (3.0, 3.0, 3.0)
RANDOM_SEED = 20260814
FAMILY_ALPHA = 0.05
FAMILY_N_CONTRASTS = 4
BONFERRONI_ALPHA = FAMILY_ALPHA / FAMILY_N_CONTRASTS
BONFERRONI_LOGP = -math.log10(BONFERRONI_ALPHA)
EXPECTED_NILEARN_VERSION = "0.13.1"

CONTRASTS = (
    "Str1_minus_Ran_r12",
    "Switch_r34_vs_r12_did",
    "Switch_r56_vs_r12_did",
    "Switch_r78_vs_r12_did",
)

CONTRAST_LABELS = {
    "Str1_minus_Ran_r12": "Initial structured-minus-random effect (runs 1-2)",
    "Switch_r34_vs_r12_did": "Early switch difference-in-differences (runs 3-4 vs 1-2)",
    "Switch_r56_vs_r12_did": "Middle switch difference-in-differences (runs 5-6 vs 1-2)",
    "Switch_r78_vs_r12_did": "Late switch difference-in-differences (runs 7-8 vs 1-2)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer relocation-safe project-relative paths in manifests."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PACKAGE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def grid_signature(img: nib.spatialimages.SpatialImage) -> tuple[Any, ...]:
    return (
        tuple(int(value) for value in img.shape[:3]),
        tuple(float(value) for value in np.round(img.header.get_zooms()[:3], 6)),
        tuple(float(value) for value in np.round(img.affine.ravel(), 6)),
    )


def grid_description(img: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in img.shape[:3]],
        "zooms_mm": [float(value) for value in img.header.get_zooms()[:3]],
        "affine": np.asarray(img.affine, dtype=float).tolist(),
        "orientation": list(nib.aff2axcodes(img.affine)),
    }


def same_grid(
    left: nib.spatialimages.SpatialImage,
    right: nib.spatialimages.SpatialImage,
) -> bool:
    return left.shape[:3] == right.shape[:3] and np.allclose(
        left.affine, right.affine, rtol=0.0, atol=1e-5
    )


def ensure_3d_data(
    img: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    label_text: str,
) -> np.ndarray:
    data = np.asarray(img.dataobj)
    if data.shape == reference.shape[:3] + (1,):
        data = data[..., 0]
    if data.shape != reference.shape[:3]:
        raise RuntimeError(
            f"{label_text}: expected shape {reference.shape[:3]}, found {data.shape}"
        )
    if not np.isfinite(data).all():
        raise RuntimeError(f"{label_text}: non-finite output values")
    return np.asarray(data, dtype=np.float64)


def save_image(
    reference: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    path: Path,
    dtype: np.dtype | type = np.float32,
) -> None:
    values = np.asarray(data, dtype=dtype)
    if values.shape != reference.shape[:3]:
        raise RuntimeError(f"{path}: invalid output shape {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path}: refusing to save non-finite values")
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header["cal_min"] = float(np.min(values))
    header["cal_max"] = float(np.max(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(values, reference.affine, header=header).to_filename(path)

    written = nib.load(path)
    expected_dtype = np.dtype(dtype)
    if written.header.get_data_dtype() != expected_dtype:
        raise RuntimeError(
            f"{path}: wrote {written.header.get_data_dtype()}, expected {expected_dtype}"
        )
    if not same_grid(written, reference):
        raise RuntimeError(f"{path}: output grid changed on write")
    proxy_slope = float(getattr(written.dataobj, "slope", 1.0))
    proxy_inter = float(getattr(written.dataobj, "inter", 0.0))
    if not np.isclose(proxy_slope, 1.0) or not np.isclose(proxy_inter, 0.0):
        raise RuntimeError(
            f"{path}: unexpected NIfTI scaling slope={proxy_slope}, inter={proxy_inter}"
        )


def reference_mask_path(fmriprep_dir: Path) -> Path:
    return (
        fmriprep_dir
        / "sub-001"
        / "func"
        / "Masks"
        / (
            f"sub-001_task-{TASK_NAME}_run-01_space-{SPACE}"
            "_desc-brain_mask.nii.gz"
        )
    )


def run_mask_path(fmriprep_dir: Path, subject: str, run: int) -> Path:
    return (
        fmriprep_dir
        / subject
        / "func"
        / "Masks"
        / (
            f"{subject}_task-{TASK_NAME}_run-{run:02d}_space-{SPACE}"
            "_desc-brain_mask.nii.gz"
        )
    )


def validate_reference(reference_path: Path) -> nib.Nifti1Image:
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = nib.load(reference_path)
    if len(reference.shape) != 3:
        raise RuntimeError(f"reference mask is not 3D: {reference_path}")
    if reference.shape != EXPECTED_REFERENCE_SHAPE:
        raise RuntimeError(
            f"reference shape {reference.shape} != fixed {EXPECTED_REFERENCE_SHAPE}"
        )
    zooms = tuple(float(value) for value in reference.header.get_zooms()[:3])
    if not np.allclose(zooms, EXPECTED_REFERENCE_ZOOMS_MM, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            f"reference zooms {zooms} != fixed 3-mm isotropic grid"
        )
    if not np.isfinite(reference.affine).all() or abs(np.linalg.det(reference.affine[:3, :3])) == 0:
        raise RuntimeError("reference affine is non-finite or singular")
    return reference


def build_common_mask(
    subjects: list[str],
    fmriprep_dir: Path,
    output_dir: Path,
) -> tuple[nib.Nifti1Image, Path, dict[str, Any]]:
    """Build the strict intersection of all 160 run masks."""
    source_reference_path = reference_mask_path(fmriprep_dir)
    reference = validate_reference(source_reference_path)
    common = np.ones(reference.shape[:3], dtype=bool)
    manifest_rows: list[dict[str, Any]] = []

    for subject in subjects:
        for run in range(1, N_RUNS + 1):
            path = run_mask_path(fmriprep_dir, subject, run)
            if not path.is_file():
                raise FileNotFoundError(path)
            image = nib.load(path)
            if len(image.shape) != 3:
                raise RuntimeError(f"{path}: mask is not 3D")
            raw = np.asarray(image.dataobj)
            if not np.isfinite(raw).all():
                raise RuntimeError(f"{path}: non-finite mask values")
            unique = np.unique(raw)
            if not np.all(np.isin(unique, (0, 1))):
                raise RuntimeError(f"{path}: expected a binary mask, found {unique[:10]}")
            was_resampled = not same_grid(image, reference)
            if was_resampled:
                on_reference = resample_to_img(
                    image,
                    reference,
                    interpolation="nearest",
                    copy_header=True,
                    force_resample=True,
                )
            else:
                on_reference = image
            values = np.asarray(on_reference.dataobj)
            if values.shape != reference.shape[:3] or not np.isfinite(values).all():
                raise RuntimeError(f"{path}: invalid resampled mask")
            binary = values > 0.5
            if not binary.any():
                raise RuntimeError(f"{path}: empty mask after reference-grid resampling")
            common &= binary
            manifest_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "path": display_path(path),
                    "sha256": sha256_file(path),
                    "source_shape": "x".join(str(v) for v in image.shape),
                    "source_zooms_mm": "x".join(
                        f"{float(v):.6g}" for v in image.header.get_zooms()[:3]
                    ),
                    "source_voxels": int(np.count_nonzero(raw)),
                    "resampled_to_reference": bool(was_resampled),
                    "reference_grid_voxels": int(np.count_nonzero(binary)),
                    "running_intersection_voxels": int(np.count_nonzero(common)),
                }
            )

    if len(manifest_rows) != EXPECTED_N_SUBJECTS * N_RUNS:
        raise RuntimeError(
            f"expected 160 input masks, processed {len(manifest_rows)}"
        )
    common_count = int(np.count_nonzero(common))
    if common_count < 1000:
        raise RuntimeError(f"implausibly small common mask: {common_count} voxels")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "common_intersection_mask.nii.gz"
    save_image(reference, common.astype(np.uint8), mask_path, np.uint8)
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "mask_input_qc.csv", index=False, lineterminator="\n"
    )

    written = nib.load(mask_path)
    written_binary = np.asarray(written.dataobj) > 0.5
    if not np.array_equal(common, written_binary):
        raise RuntimeError("common mask changed on write")
    voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
    qc = {
        "reference_mask": display_path(source_reference_path),
        "reference_mask_sha256": sha256_file(source_reference_path),
        "reference_grid": grid_description(reference),
        "n_subjects": len(subjects),
        "runs_per_subject": N_RUNS,
        "n_input_masks": len(manifest_rows),
        "n_resampled_masks": int(
            sum(bool(row["resampled_to_reference"]) for row in manifest_rows)
        ),
        "intersection_voxels": common_count,
        "voxel_volume_mm3": voxel_volume,
        "intersection_volume_mm3": common_count * voxel_volume,
        "reference_grid_fraction": common_count / float(np.prod(reference.shape[:3])),
        "mask_output": display_path(mask_path),
        "mask_output_sha256": sha256_file(mask_path),
    }
    (output_dir / "common_mask_qc.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n"
    )
    return written, mask_path, qc


def subject_map_path(input_root: Path, subject: str, contrast: str) -> Path:
    return (
        input_root
        / "subject_contrasts"
        / "eventrelated"
        / subject
        / f"{contrast}.nii.gz"
    )


def validate_subject_maps(
    subjects: list[str],
    contrasts: list[str],
    input_root: Path,
    reference: nib.Nifti1Image,
    common_mask: np.ndarray,
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Validate all maps before any inferential model is run."""
    result: dict[str, list[Path]] = {}
    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        paths: list[Path] = []
        for subject in subjects:
            path = subject_map_path(input_root, subject, contrast)
            if not path.is_file():
                raise FileNotFoundError(path)
            image = nib.load(path)
            if len(image.shape) != 3:
                raise RuntimeError(f"{path}: subject contrast is not 3D")
            if not same_grid(image, reference):
                raise RuntimeError(
                    f"{path}: subject contrast is not on the fixed sub-001 run-01 grid"
                )
            dtype = image.header.get_data_dtype()
            if dtype != np.dtype("float32"):
                raise RuntimeError(
                    f"{path}: expected regenerated float32 data, found {dtype}"
                )
            slope = float(getattr(image.dataobj, "slope", 1.0))
            inter = float(getattr(image.dataobj, "inter", 0.0))
            if not np.isclose(slope, 1.0) or not np.isclose(inter, 0.0):
                raise RuntimeError(
                    f"{path}: scaled data are forbidden (slope={slope}, inter={inter})"
                )
            data = np.asarray(image.dataobj, dtype=np.float64)
            if not np.isfinite(data).all():
                raise RuntimeError(f"{path}: non-finite values")
            unique_count = int(np.unique(data).size)
            if unique_count < 1000:
                raise RuntimeError(
                    f"{path}: only {unique_count} distinct values; possible quantization"
                )
            masked = data[common_mask]
            masked_sd = float(np.std(masked, ddof=1))
            if not np.isfinite(masked_sd) or masked_sd <= 0:
                raise RuntimeError(f"{path}: constant or invalid data in common mask")
            rows.append(
                {
                    "subject": subject,
                    "contrast": contrast,
                    "path": display_path(path),
                    "sha256": sha256_file(path),
                    "dtype": str(dtype),
                    "proxy_slope": slope,
                    "proxy_inter": inter,
                    "shape": "x".join(str(v) for v in image.shape),
                    "unique_values": unique_count,
                    "whole_image_min": float(np.min(data)),
                    "whole_image_max": float(np.max(data)),
                    "common_mask_mean": float(np.mean(masked)),
                    "common_mask_sd": masked_sd,
                    "common_mask_nonzero_voxels": int(np.count_nonzero(masked)),
                    "outside_common_mask_nonzero_voxels": int(
                        np.count_nonzero(data[~common_mask])
                    ),
                }
            )
            paths.append(path)
        result[contrast] = paths

    expected = len(subjects) * len(contrasts)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} maps, validated {len(rows)}")
    pd.DataFrame(rows).to_csv(
        output_dir / "subject_map_qc.csv", index=False, lineterminator="\n"
    )
    return result


def load_roi_masks(
    rois: list[str],
    roi_mask_dir: Path,
    output_dir: Path,
) -> dict[str, nib.Nifti1Image]:
    masks: dict[str, nib.Nifti1Image] = {}
    rows: list[dict[str, Any]] = []
    for roi in rois:
        path = roi_mask_dir / f"{roi}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(path)
        if len(image.shape) != 3:
            raise RuntimeError(f"{path}: ROI mask is not 3D")
        data = np.asarray(image.dataobj)
        if not np.isfinite(data).all():
            raise RuntimeError(f"{path}: ROI mask has non-finite values")
        binary = data > 0
        count = int(np.count_nonzero(binary))
        if count == 0:
            raise RuntimeError(f"{path}: ROI mask is empty")
        masks[roi] = image
        rows.append(
            {
                "roi": roi,
                "path": display_path(path),
                "sha256": sha256_file(path),
                "dtype": str(image.header.get_data_dtype()),
                "shape": "x".join(str(v) for v in image.shape),
                "zooms_mm": "x".join(
                    f"{float(v):.6g}" for v in image.header.get_zooms()[:3]
                ),
                "mask_voxels_gt_zero": count,
                "mask_volume_mm3": count
                * float(abs(np.linalg.det(image.affine[:3, :3]))),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "roi_mask_qc.csv", index=False, lineterminator="\n"
    )
    return masks


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values, preserving input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("BH correction received invalid p values")
    if np.any((values < 0) | (values > 1)):
        raise RuntimeError("BH correction received p values outside [0, 1]")
    n_tests = len(values)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * n_tests / np.arange(1, n_tests + 1)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def roi_inference(
    subjects: list[str],
    rois: list[str],
    contrasts: list[str],
    map_paths: dict[str, list[Path]],
    roi_masks: dict[str, nib.Nifti1Image],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reproduce Stage 44 extraction on the regenerated float32 maps."""
    roi_dir = output_dir / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    tidy_rows: list[dict[str, Any]] = []
    significant_counts: dict[str, int] = {}

    for contrast in contrasts:
        values = np.empty((len(subjects), len(rois)), dtype=np.float64)
        for subject_index, path in enumerate(map_paths[contrast]):
            contrast_img = nib.load(path)
            resampled_cache: dict[tuple[Any, ...], np.ndarray] = {}
            native_data = np.asarray(contrast_img.dataobj, dtype=np.float64)
            for roi_index, roi in enumerate(rois):
                mask_img = roi_masks[roi]
                key = grid_signature(mask_img)
                if key not in resampled_cache:
                    if same_grid(contrast_img, mask_img):
                        on_mask = native_data
                    else:
                        resampled = resample_to_img(
                            contrast_img,
                            mask_img,
                            interpolation="continuous",
                            copy_header=True,
                            force_resample=True,
                        )
                        on_mask = np.asarray(resampled.dataobj, dtype=np.float64)
                    if not np.isfinite(on_mask).all():
                        raise RuntimeError(
                            f"{path}: non-finite values after ROI-grid resampling"
                        )
                    resampled_cache[key] = on_mask
                mask = np.asarray(mask_img.dataobj) > 0
                roi_values = resampled_cache[key][mask]
                if roi_values.size == 0 or not np.isfinite(roi_values).all():
                    raise RuntimeError(f"{subject} {contrast} {roi}: invalid ROI data")
                values[subject_index, roi_index] = float(np.mean(roi_values))

        if not np.isfinite(values).all():
            raise RuntimeError(f"{contrast}: non-finite ROI mean matrix")
        wide = pd.DataFrame(values, columns=rois)
        wide.insert(0, "subject", subjects)
        wide_path = roi_dir / f"All_eventrelated_{contrast}_Coef.tsv"
        wide.to_csv(
            wide_path,
            sep="\t",
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )

        per_roi: list[dict[str, Any]] = []
        p_values: list[float] = []
        for roi_index, roi in enumerate(rois):
            sample = values[:, roi_index]
            n = int(sample.size)
            mean = float(np.mean(sample))
            sd = float(np.std(sample, ddof=1))
            sem = sd / math.sqrt(n)
            test = stats.ttest_1samp(sample, popmean=0.0, alternative="two-sided")
            t_value = float(test.statistic)
            p_value = float(test.pvalue)
            if not np.isfinite(t_value) or not np.isfinite(p_value):
                raise RuntimeError(f"{contrast} {roi}: invalid one-sample t test")
            critical = float(stats.t.ppf(0.975, df=n - 1))
            row = {
                "contrast": contrast,
                "contrast_label": CONTRAST_LABELS[contrast],
                "roi": roi,
                "n": n,
                "mean": mean,
                "sd": sd,
                "sem": sem,
                "cohen_dz": mean / sd if sd > 0 else float("nan"),
                "ci95_low": mean - critical * sem,
                "ci95_high": mean + critical * sem,
                "t": t_value,
                "df": n - 1,
                "p_two_sided": p_value,
            }
            per_roi.append(row)
            p_values.append(p_value)

        q_values = bh_adjust(np.asarray(p_values))
        for row, q_value in zip(per_roi, q_values, strict=True):
            row["q_bh_within_contrast"] = float(q_value)
            row["significant_fdr_0_05"] = bool(q_value <= 0.05)
            tidy_rows.append(row)
        significant_counts[contrast] = int(np.count_nonzero(q_values <= 0.05))

    results = pd.DataFrame(tidy_rows)
    results.to_csv(
        roi_dir / "roi_results.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return tidy_rows, significant_counts


def output_array(
    output: dict[str, nib.spatialimages.SpatialImage],
    key: str,
    reference: nib.Nifti1Image,
) -> np.ndarray:
    if key not in output:
        raise RuntimeError(
            f"non_parametric_inference omitted required '{key}' output; "
            f"available keys: {sorted(output)}"
        )
    image = output[key]
    if not same_grid(image, reference):
        raise RuntimeError(f"permutation output {key} is not on the common-mask grid")
    return ensure_3d_data(image, reference, f"permutation output {key}")


def component_table_rows(
    contrast: str,
    signed_t: np.ndarray,
    logp_tfce: np.ndarray,
    affine: np.ndarray,
) -> list[dict[str, Any]]:
    """Describe positive and negative 26-connected suprathreshold sets."""
    structure = generate_binary_structure(rank=3, connectivity=3)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    rows: list[dict[str, Any]] = []

    for direction, selected in (
        ("positive", signed_t > 0),
        ("negative", signed_t < 0),
    ):
        labels, n_components = label(selected, structure=structure)
        provisional: list[dict[str, Any]] = []
        for component_id in range(1, n_components + 1):
            coordinates = np.argwhere(labels == component_id)
            t_values = signed_t[labels == component_id]
            logp_values = logp_tfce[labels == component_id]
            if coordinates.size == 0:
                continue
            t_peak_index = int(np.argmax(np.abs(t_values)))
            logp_peak_index = int(np.argmax(logp_values))
            t_peak_ijk = coordinates[t_peak_index]
            logp_peak_ijk = coordinates[logp_peak_index]
            t_peak_xyz = apply_affine(affine, t_peak_ijk)
            logp_peak_xyz = apply_affine(affine, logp_peak_ijk)
            peak_logp = float(logp_values[logp_peak_index])
            provisional.append(
                {
                    "contrast": contrast,
                    "contrast_label": CONTRAST_LABELS[contrast],
                    "direction": direction,
                    "component_id": component_id,
                    "n_voxels": int(coordinates.shape[0]),
                    "volume_mm3": float(coordinates.shape[0] * voxel_volume),
                    "peak_t": float(t_values[t_peak_index]),
                    "peak_abs_t": float(abs(t_values[t_peak_index])),
                    "tfce_logp_at_t_peak": float(logp_values[t_peak_index]),
                    "t_peak_i": int(t_peak_ijk[0]),
                    "t_peak_j": int(t_peak_ijk[1]),
                    "t_peak_k": int(t_peak_ijk[2]),
                    "t_peak_x_mm": float(t_peak_xyz[0]),
                    "t_peak_y_mm": float(t_peak_xyz[1]),
                    "t_peak_z_mm": float(t_peak_xyz[2]),
                    "max_tfce_logp": peak_logp,
                    "min_tfce_fwe_p": float(10.0 ** (-peak_logp)),
                    "tfce_peak_i": int(logp_peak_ijk[0]),
                    "tfce_peak_j": int(logp_peak_ijk[1]),
                    "tfce_peak_k": int(logp_peak_ijk[2]),
                    "tfce_peak_x_mm": float(logp_peak_xyz[0]),
                    "tfce_peak_y_mm": float(logp_peak_xyz[1]),
                    "tfce_peak_z_mm": float(logp_peak_xyz[2]),
                    "connectivity": 26,
                    "threshold": "TFCE voxelwise FWE p <= .05, then Bonferroni /4",
                }
            )
        provisional.sort(key=lambda row: (-row["n_voxels"], -row["peak_abs_t"]))
        for rank_value, row in enumerate(provisional, start=1):
            row["extent_rank_within_direction"] = rank_value
            rows.append(row)
    return rows


def run_group_models(
    subjects: list[str],
    contrasts: list[str],
    map_paths: dict[str, list[Path]],
    common_mask_img: nib.Nifti1Image,
    common_mask_path: Path,
    output_dir: Path,
    n_perm: int,
    n_jobs: int,
    verbose: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    design_matrix = pd.DataFrame({"intercept": np.ones(len(subjects), dtype=float)})
    design_matrix.insert(0, "subject", subjects)
    design_matrix.to_csv(
        output_dir / "second_level_design.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    model_design = design_matrix.loc[:, ["intercept"]]
    common_mask = np.asarray(common_mask_img.dataobj) > 0.5
    group_summary: dict[str, Any] = {}
    all_components: list[dict[str, Any]] = []
    captured_warnings: list[str] = []

    for contrast in contrasts:
        print(f"\n=== group inference: {contrast} ===", flush=True)
        contrast_dir = output_dir / "group" / contrast
        contrast_dir.mkdir(parents=True, exist_ok=True)
        inputs = [str(path) for path in map_paths[contrast]]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SecondLevelModel(
                mask_img=common_mask_img,
                smoothing_fwhm=None,
                n_jobs=n_jobs,
                minimize_memory=True,
                verbose=verbose,
            ).fit(second_level_input=inputs, design_matrix=model_design)
            t_img = model.compute_contrast("intercept", output_type="stat")
            z_img = model.compute_contrast("intercept", output_type="z_score")
        captured_warnings.extend(
            f"{contrast} parametric: {item.message}" for item in caught
        )
        t_data = ensure_3d_data(t_img, common_mask_img, f"{contrast} t map")
        z_data = ensure_3d_data(z_img, common_mask_img, f"{contrast} z map")
        if not np.isfinite(t_data[common_mask]).all() or not np.isfinite(
            z_data[common_mask]
        ).all():
            raise RuntimeError(f"{contrast}: invalid parametric statistics")
        save_image(
            common_mask_img,
            t_data,
            contrast_dir / "one_sample_t_map.nii.gz",
        )
        save_image(
            common_mask_img,
            z_data,
            contrast_dir / "one_sample_z_map.nii.gz",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            permutation = non_parametric_inference(
                second_level_input=inputs,
                design_matrix=model_design,
                second_level_contrast="intercept",
                mask=common_mask_img,
                smoothing_fwhm=None,
                model_intercept=False,
                n_perm=n_perm,
                two_sided_test=True,
                random_state=RANDOM_SEED,
                n_jobs=n_jobs,
                verbose=verbose,
                threshold=None,
                tfce=True,
            )
        captured_warnings.extend(
            f"{contrast} permutation: {item.message}" for item in caught
        )
        if not isinstance(permutation, dict):
            raise RuntimeError(
                f"{contrast}: TFCE inference should return a dict, found "
                f"{type(permutation).__name__}"
            )

        permutation_t = output_array(permutation, "t", common_mask_img)
        logp_max_t = output_array(permutation, "logp_max_t", common_mask_img)
        tfce_score = output_array(permutation, "tfce", common_mask_img)
        logp_max_tfce = output_array(
            permutation, "logp_max_tfce", common_mask_img
        )
        for name, data in (
            ("logp_max_t", logp_max_t),
            ("logp_max_tfce", logp_max_tfce),
        ):
            if np.any(data[common_mask] < 0):
                raise RuntimeError(f"{contrast}: negative values in {name}")

        t_difference = np.abs(permutation_t[common_mask] - t_data[common_mask])
        max_t_difference = float(np.max(t_difference))
        t_correlation = float(
            np.corrcoef(permutation_t[common_mask], t_data[common_mask])[0, 1]
        )
        if not np.isfinite(t_correlation) or t_correlation < 0.999999:
            raise RuntimeError(
                f"{contrast}: parametric/permutation t mismatch "
                f"(r={t_correlation}, max diff={max_t_difference})"
            )

        significant = common_mask & (logp_max_tfce >= BONFERRONI_LOGP)
        signed_t = np.where(significant, t_data, 0.0)
        save_image(
            common_mask_img,
            logp_max_tfce,
            contrast_dir / "logp_max_tfce.nii.gz",
        )
        save_image(
            common_mask_img,
            logp_max_t,
            contrast_dir / "logp_max_t.nii.gz",
        )
        save_image(
            common_mask_img,
            tfce_score,
            contrast_dir / "tfce_score.nii.gz",
        )
        signed_path = contrast_dir / "signed_t_tfce_fwe_bonferroni4.nii.gz"
        save_image(common_mask_img, signed_t, signed_path)

        components = component_table_rows(
            contrast, signed_t, logp_max_tfce, common_mask_img.affine
        )
        all_components.extend(components)
        positive_voxels = int(np.count_nonzero(signed_t > 0))
        negative_voxels = int(np.count_nonzero(signed_t < 0))
        group_summary[contrast] = {
            "label": CONTRAST_LABELS[contrast],
            "n_subjects": len(subjects),
            "parametric_t_min_in_mask": float(np.min(t_data[common_mask])),
            "parametric_t_max_in_mask": float(np.max(t_data[common_mask])),
            "parametric_z_min_in_mask": float(np.min(z_data[common_mask])),
            "parametric_z_max_in_mask": float(np.max(z_data[common_mask])),
            "max_logp_tfce_in_mask": float(np.max(logp_max_tfce[common_mask])),
            "min_tfce_fwe_p_in_mask": float(
                10.0 ** (-np.max(logp_max_tfce[common_mask]))
            ),
            "max_logp_max_t_in_mask": float(np.max(logp_max_t[common_mask])),
            "positive_suprathreshold_voxels": positive_voxels,
            "negative_suprathreshold_voxels": negative_voxels,
            "total_suprathreshold_voxels": positive_voxels + negative_voxels,
            "positive_components_26_connected": int(
                sum(row["direction"] == "positive" for row in components)
            ),
            "negative_components_26_connected": int(
                sum(row["direction"] == "negative" for row in components)
            ),
            "parametric_vs_permutation_t_correlation": t_correlation,
            "parametric_vs_permutation_t_max_abs_difference": max_t_difference,
            "maps": {
                "t": display_path(contrast_dir / "one_sample_t_map.nii.gz"),
                "z": display_path(contrast_dir / "one_sample_z_map.nii.gz"),
                "logp_max_tfce": display_path(
                    contrast_dir / "logp_max_tfce.nii.gz"
                ),
                "logp_max_t": display_path(contrast_dir / "logp_max_t.nii.gz"),
                "tfce_score": display_path(contrast_dir / "tfce_score.nii.gz"),
                "signed_thresholded_t": display_path(signed_path),
            },
        }
        print(
            f"  TFCE/Bonferroni surviving voxels: "
            f"+{positive_voxels} / -{negative_voxels}",
            flush=True,
        )

    component_columns = [
        "contrast",
        "contrast_label",
        "direction",
        "component_id",
        "extent_rank_within_direction",
        "n_voxels",
        "volume_mm3",
        "peak_t",
        "peak_abs_t",
        "tfce_logp_at_t_peak",
        "t_peak_i",
        "t_peak_j",
        "t_peak_k",
        "t_peak_x_mm",
        "t_peak_y_mm",
        "t_peak_z_mm",
        "max_tfce_logp",
        "min_tfce_fwe_p",
        "tfce_peak_i",
        "tfce_peak_j",
        "tfce_peak_k",
        "tfce_peak_x_mm",
        "tfce_peak_y_mm",
        "tfce_peak_z_mm",
        "connectivity",
        "threshold",
    ]
    pd.DataFrame(all_components, columns=component_columns).to_csv(
        output_dir / "components_peaks.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return group_summary, all_components, captured_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=FIRST_LEVEL_DIR / "FLA_nilearn_float32_20260814",
        help="Root produced by 90_run_float32_eventrelated.py.",
    )
    parser.add_argument(
        "--fmriprep-dir",
        type=Path,
        default=FMRIPREP_DIR,
        help="fMRIPrep derivative root containing run-specific Masks.",
    )
    parser.add_argument(
        "--roi-mask-dir",
        type=Path,
        default=FLA_DIR / "fsl" / "roi_masks",
        help="Directory containing the exact 48 study ROI NIfTI masks.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MANUSCRIPT_DIR / "data" / "whole_brain",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        choices=CONTRASTS,
        default=list(CONTRASTS),
        help="Subset for testing; the primary run uses all four (default).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10_000,
        help="TFCE permutations (10,000 for the manuscript; 10 is smoke-test only).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Nilearn worker count; -1 uses all available CPUs.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this script's named outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if nilearn.__version__ != EXPECTED_NILEARN_VERSION:
        raise RuntimeError(
            f"this audited analysis requires Nilearn {EXPECTED_NILEARN_VERSION}; "
            f"found {nilearn.__version__}"
        )
    if args.n_perm < 1:
        raise SystemExit("--n-perm must be at least 1")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be zero")
    contrasts = list(dict.fromkeys(args.contrasts))

    subjects = read_subjects()
    rois = read_rois()
    if len(subjects) != EXPECTED_N_SUBJECTS or len(set(subjects)) != len(subjects):
        raise RuntimeError(
            f"expected {EXPECTED_N_SUBJECTS} unique subjects, found {len(subjects)}"
        )
    if len(rois) != EXPECTED_N_ROIS or len(set(rois)) != len(rois):
        raise RuntimeError(
            f"expected {EXPECTED_N_ROIS} unique ROIs, found {len(rois)}"
        )

    output_dir = args.out_dir.resolve()
    provenance_path = output_dir / "provenance.json"
    if provenance_path.exists() and not args.overwrite:
        raise SystemExit(
            f"output already contains {provenance_path}; rerun with --overwrite "
            "or choose a new --out-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building strict 160-run common mask...", flush=True)
    common_mask_img, common_mask_path, mask_qc = build_common_mask(
        subjects, args.fmriprep_dir.resolve(), output_dir
    )
    common_mask = np.asarray(common_mask_img.dataobj) > 0.5

    print("Validating regenerated float32 subject maps...", flush=True)
    map_paths = validate_subject_maps(
        subjects,
        contrasts,
        args.input_root.resolve(),
        common_mask_img,
        common_mask,
        output_dir,
    )

    print("Loading and validating 48 ROI masks...", flush=True)
    roi_masks = load_roi_masks(rois, args.roi_mask_dir.resolve(), output_dir)
    print("Extracting subject-level ROI means and testing ROI effects...", flush=True)
    _, roi_significant_counts = roi_inference(
        subjects, rois, contrasts, map_paths, roi_masks, output_dir
    )

    group_summary, components, captured_warnings = run_group_models(
        subjects,
        contrasts,
        map_paths,
        common_mask_img,
        common_mask_path,
        output_dir,
        args.n_perm,
        args.n_jobs,
        args.verbose,
    )
    for contrast in contrasts:
        group_summary[contrast]["significant_rois_bh_fdr_0_05"] = (
            roi_significant_counts[contrast]
        )

    minimum_attainable_p = 1.0 / (args.n_perm + 1.0)
    provenance = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "analysis": "audited_float32_eventrelated_roi_and_whole_brain_group_inference",
        "subjects": subjects,
        "n_subjects": len(subjects),
        "contrasts_run": contrasts,
        "planned_contrast_family": list(CONTRASTS),
        "contrast_labels": CONTRAST_LABELS,
        "input_root": display_path(args.input_root),
        "fmriprep_dir": display_path(args.fmriprep_dir),
        "roi_mask_dir": display_path(args.roi_mask_dir),
        "output_dir": display_path(output_dir),
        "common_mask": mask_qc,
        "whole_brain_model": {
            "design": "intercept-only one-sample model",
            "design_column": "intercept",
            "explicit_mask": display_path(common_mask_path),
            "additional_second_level_smoothing_fwhm_mm": None,
            "n_perm": args.n_perm,
            "minimum_attainable_permutation_p": minimum_attainable_p,
            "two_sided_test": True,
            "tfce": True,
            "cluster_forming_threshold": None,
            "model_intercept": False,
            "random_seed": RANDOM_SEED,
            "n_jobs": args.n_jobs,
            "primary_error_control": "voxelwise TFCE FWE within contrast, Bonferroni across four contrasts",
            "family_alpha": FAMILY_ALPHA,
            "number_of_planned_contrasts": FAMILY_N_CONTRASTS,
            "per_contrast_alpha": BONFERRONI_ALPHA,
            "negative_log10_p_threshold": BONFERRONI_LOGP,
            "permutations_resolve_primary_threshold": minimum_attainable_p
            <= BONFERRONI_ALPHA,
        },
        "roi_model": {
            "n_prespecified_rois": len(rois),
            "rois_in_config_order": rois,
            "extraction": "subject contrast resampled continuously to each ROI grid; arithmetic mean over mask > 0",
            "test": "two-sided one-sample t test against zero",
            "multiplicity": "Benjamini-Hochberg FDR across 48 ROIs separately within each contrast",
            "fdr_alpha": 0.05,
        },
        "descriptive_components": {
            "source": "signed parametric t map surviving the primary TFCE/Bonferroni threshold",
            "positive_and_negative_labeled_separately": True,
            "voxel_connectivity": 26,
            "inferential_role": "descriptive only; voxelwise TFCE FWE supplies inference",
        },
        "software": {
            "python": platform.python_version(),
            "nilearn": nilearn.__version__,
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "captured_warnings": sorted(set(captured_warnings)),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "n_subjects": len(subjects),
        "n_rois": len(rois),
        "n_input_run_masks": EXPECTED_N_SUBJECTS * N_RUNS,
        "common_mask_voxels": mask_qc["intersection_voxels"],
        "common_mask_volume_mm3": mask_qc["intersection_volume_mm3"],
        "n_perm": args.n_perm,
        "random_seed": RANDOM_SEED,
        "primary_tfce_logp_threshold": BONFERRONI_LOGP,
        "primary_per_contrast_alpha": BONFERRONI_ALPHA,
        "contrasts": group_summary,
        "total_descriptive_components": len(components),
        "roi_results": display_path(output_dir / "roi" / "roi_results.csv"),
        "component_results": display_path(output_dir / "components_peaks.csv"),
        "provenance": display_path(provenance_path),
    }
    (output_dir / "group_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    output_manifest_rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "output_manifest.csv":
            output_manifest_rows.append(
                {
                    "path": display_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(output_manifest_rows).to_csv(
        output_dir / "output_manifest.csv", index=False, lineterminator="\n"
    )
    print(f"\nComplete: {output_dir}", flush=True)
    print(
        f"Primary threshold: log10-FWE >= {BONFERRONI_LOGP:.6f} "
        f"(p <= {BONFERRONI_ALPHA:.4g})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
