#!/usr/bin/env python3
"""Validate the audited float32 ROI, whole-brain, figure, and TeX bundle.

This program is deliberately non-mutating. It independently recomputes the
ROI tests, inspects every first-level QC record and subject contrast, rebuilds
the primary signed whole-brain maps from their t and corrected-p inputs, and
checks the checksummed analysis/figure manifests used by the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label
from scipy.stats import ttest_1samp


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, FIRST_LEVEL_DIR, ROI_MASK_DIR, recorded_path, source_matches
MAIN_TEX = MANUSCRIPT_DIR / "manuscript.tex"
BIB_FILE = MANUSCRIPT_DIR / "references.bib"
WHOLE_BRAIN_DIR = MANUSCRIPT_DIR / "data" / "whole_brain"
ROI_DIR = WHOLE_BRAIN_DIR / "roi"
FLOAT32_ROOT = (
    FIRST_LEVEL_DIR / "FLA_nilearn_float32_20260814"
)
FIGURE_DIR = MANUSCRIPT_DIR / "figures"
SENSITIVITY_DIR = MANUSCRIPT_DIR / "data" / "sensitivity"
SENSITIVITY_INPUT_DIR = MANUSCRIPT_DIR / "data" / "sensitivity_inputs" / "afni_v8_censor"
SENSITIVITY_FLA_ROOT = (
    FIRST_LEVEL_DIR / "FLA_nilearn_sensitivity_20260921"
)
EXPECTED_SENSITIVITY_COUNTS = {
    "nocensor_hp128_primary": (29, 8, 0, 0),
    "censor_hp128": (31, 15, 5, 0),
    "nocensor_bandpass": (15, 0, 2, 0),
    "censor_bandpass": (14, 4, 4, 0),
}

EXPECTED_N_SUBJECTS = 20
EXPECTED_N_RUNS = 8
EXPECTED_N_ROIS = 48
EXPECTED_MASK_VOXELS = 58_611
EXPECTED_SHAPE = (53, 65, 56)
EXPECTED_ZOOMS = (3.0, 3.0, 3.0)
EXPECTED_N_PERM = 10_000
EXPECTED_SEED = 20260814
FAMILY_ALPHA = 0.05
N_PLANNED_CONTRASTS = 4
PER_CONTRAST_ALPHA = FAMILY_ALPHA / N_PLANNED_CONTRASTS
TFCE_LOGP_THRESHOLD = -math.log10(PER_CONTRAST_ALPHA)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

CONTRASTS = (
    "Str1_minus_Ran_r12",
    "Switch_r34_vs_r12_did",
    "Switch_r56_vs_r12_did",
    "Switch_r78_vs_r12_did",
)

CONTRAST_FILES = {
    "Str1_minus_Ran_r12": "All_eventrelated_Str1_minus_Ran_r12_Coef.tsv",
    "Switch_r34_vs_r12_did": "All_eventrelated_Switch_r34_vs_r12_did_Coef.tsv",
    "Switch_r56_vs_r12_did": "All_eventrelated_Switch_r56_vs_r12_did_Coef.tsv",
    "Switch_r78_vs_r12_did": "All_eventrelated_Switch_r78_vs_r12_did_Coef.tsv",
}

EXPECTED_ROI_COUNTS = {
    "Str1_minus_Ran_r12": (29, 1, 28),
    "Switch_r34_vs_r12_did": (8, 8, 0),
    "Switch_r56_vs_r12_did": (0, 0, 0),
    "Switch_r78_vs_r12_did": (0, 0, 0),
}

EXPECTED_EARLY_ROIS = {
    "LeftCerebellum",
    "LeftSMA",
    "LeftThalamusProper",
    "RightDLPFC9",
    "RightPallidum",
    "RightSMA",
    "RightSuperiorFrontal",
    "RightThalamusProper",
}

GROUP_MAP_FILES = {
    "t": "one_sample_t_map.nii.gz",
    "z": "one_sample_z_map.nii.gz",
    "logp_max_tfce": "logp_max_tfce.nii.gz",
    "logp_max_t": "logp_max_t.nii.gz",
    "tfce_score": "tfce_score.nii.gz",
    "signed_thresholded_t": "signed_t_tfce_fwe_bonferroni4.nii.gz",
}

EXPECTED_GRAPHICS = {"figures/behavioral_rt_cycles.png", "figures/censored_roi_axial.png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Missing JSON file: {path}"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), f"Expected a JSON object: {path}"
    return value


def read_csv_rows(
    path: Path,
    delimiter: str = ",",
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    assert path.is_file(), f"Missing table: {path}"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    assert rows or allow_empty, f"Empty table: {path}"
    return rows


def package_path(value: str) -> Path:
    path = Path(value)
    return recorded_path(value)


def same_grid(
    left: nib.spatialimages.SpatialImage,
    right: nib.spatialimages.SpatialImage,
) -> bool:
    return left.shape[:3] == right.shape[:3] and np.allclose(
        left.affine, right.affine, rtol=0.0, atol=1e-5
    )


def finite_3d(
    image: nib.spatialimages.SpatialImage,
    label_text: str,
) -> np.ndarray:
    values = np.asarray(image.dataobj)
    if values.shape == image.shape[:3] + (1,):
        values = values[..., 0]
    assert values.ndim == 3, f"{label_text}: expected 3D, found {values.shape}"
    assert np.isfinite(values).all(), f"{label_text}: non-finite values"
    return np.asarray(values, dtype=np.float64)


def strip_tex_comments(text: str) -> str:
    retained: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding += 1
                cursor -= 1
            if preceding % 2 == 0:
                cut = index
                break
        retained.append(line[:cut])
    return "\n".join(retained)


def assert_balanced_tex(path: Path) -> None:
    text = strip_tex_comments(path.read_text(encoding="utf-8"))
    depth = 0
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            assert depth >= 0, f"Extra closing brace in {path.name}"
        index += 1
    assert depth == 0, f"Unbalanced braces in {path.name}: depth={depth}"

    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        else:
            assert stack, f"Unmatched end of {environment} in {path.name}"
            opened = stack.pop()
            assert opened == environment, (
                f"Environment mismatch in {path.name}: "
                f"began {opened}, ended {environment}"
            )
    assert not stack, f"Unclosed environments in {path.name}: {stack}"


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty_like(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted[order] = np.minimum(
        np.minimum.accumulate(ranked[::-1])[::-1], 1.0
    )
    return adjusted


def load_roi_table(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows = read_csv_rows(path, delimiter="\t")
    assert "subject" in rows[0], f"Missing subject column: {path}"
    roi_names = [name for name in rows[0] if name != "subject"]
    subjects = [row["subject"] for row in rows]
    values = np.asarray(
        [[float(row[name]) for name in roi_names] for row in rows], dtype=float
    )
    return subjects, roi_names, values


def parse_bool(value: str, label_text: str) -> bool:
    normalized = str(value).strip().lower()
    assert normalized in {"true", "false", "1", "0"}, (
        f"{label_text}: invalid Boolean {value!r}"
    )
    return normalized in {"true", "1"}


def validate_float32_roi_sources() -> tuple[list[str], dict[str, set[str]]]:
    tidy_rows = read_csv_rows(ROI_DIR / "roi_results.csv")
    assert len(tidy_rows) == len(CONTRASTS) * EXPECTED_N_ROIS
    tidy = {(row["contrast"], row["roi"]): row for row in tidy_rows}
    assert len(tidy) == len(tidy_rows), "Duplicate contrast/ROI rows"

    common_subjects: list[str] | None = None
    significant_sets: dict[str, set[str]] = {}
    for contrast in CONTRASTS:
        subjects, roi_names, values = load_roi_table(
            ROI_DIR / CONTRAST_FILES[contrast]
        )
        assert len(subjects) == EXPECTED_N_SUBJECTS
        assert len(set(subjects)) == EXPECTED_N_SUBJECTS
        assert len(roi_names) == EXPECTED_N_ROIS
        assert values.shape == (EXPECTED_N_SUBJECTS, EXPECTED_N_ROIS)
        assert np.isfinite(values).all()
        if common_subjects is None:
            common_subjects = subjects
        else:
            assert subjects == common_subjects, "ROI subject order changed by contrast"

        test = ttest_1samp(values, 0.0, axis=0)
        t_values = np.asarray(test.statistic, dtype=float)
        p_values = np.asarray(test.pvalue, dtype=float)
        q_values = bh_fdr(p_values)
        means = np.mean(values, axis=0)
        sds = np.std(values, axis=0, ddof=1)
        significant = q_values <= 0.05
        significant_rois = {
            roi_names[index] for index in np.flatnonzero(significant)
        }
        significant_sets[contrast] = significant_rois

        expected_total, expected_positive, expected_negative = (
            EXPECTED_ROI_COUNTS[contrast]
        )
        assert int(np.count_nonzero(significant)) == expected_total
        assert int(np.count_nonzero(significant & (means > 0))) == expected_positive
        assert int(np.count_nonzero(significant & (means < 0))) == expected_negative

        for index, roi in enumerate(roi_names):
            row = tidy[(contrast, roi)]
            assert int(row["n"]) == EXPECTED_N_SUBJECTS
            assert int(row["df"]) == EXPECTED_N_SUBJECTS - 1
            comparisons = {
                "mean": means[index],
                "sd": sds[index],
                "t": t_values[index],
                "p_two_sided": p_values[index],
                "q_bh_within_contrast": q_values[index],
            }
            for column, recomputed in comparisons.items():
                stored = float(row[column])
                assert np.isclose(stored, recomputed, rtol=2e-10, atol=2e-12), (
                    f"{contrast} {roi}: stored {column}={stored} does not "
                    f"match recomputed {recomputed}"
                )
            assert parse_bool(
                row["significant_fdr_0_05"], f"{contrast} {roi}"
            ) == bool(significant[index])

    assert common_subjects is not None
    assert significant_sets["Switch_r34_vs_r12_did"] == EXPECTED_EARLY_ROIS
    assert "LeftCaudalAnteriorCingulate" not in EXPECTED_EARLY_ROIS
    initial_positive = {
        row["roi"]
        for row in tidy_rows
        if row["contrast"] == "Str1_minus_Ran_r12"
        and parse_bool(row["significant_fdr_0_05"], "initial ROI flag")
        and float(row["mean"]) > 0
    }
    assert initial_positive == {"LeftAccumbensArea"}
    assert (
        EXPECTED_EARLY_ROIS - {"RightPallidum"}
    ) <= significant_sets["Str1_minus_Ran_r12"]
    return common_subjects, significant_sets


def validate_first_level_qc(subjects: list[str]) -> None:
    seen: set[tuple[str, int]] = set()
    for subject in subjects:
        provenance = read_json(FLOAT32_ROOT / "provenance" / f"{subject}.json")
        assert provenance["subject"] == subject
        assert provenance["output_dtype"] == "float32"
        assert set(provenance["primary_contrasts"]) == set(CONTRASTS)
        for run in range(1, EXPECTED_N_RUNS + 1):
            path = (
                FLOAT32_ROOT
                / "level1"
                / "eventrelated"
                / subject
                / f"run-{run:02d}"
                / "model_qc.json"
            )
            qc = read_json(path)
            assert qc["subject"] == subject and int(qc["run"]) == run
            assert int(qc["matrix_rank"]) == int(qc["n_columns"])
            assert int(qc["n_timepoints"]) > int(qc["n_columns"])
            assert math.isfinite(float(qc["condition_number"]))
            assert qc["warnings"] == [], f"Model warning in {path}: {qc['warnings']}"
            counts = qc.get("event_counts", {})
            assert counts and all(int(value) > 0 for value in counts.values())
            seen.add((subject, run))
    assert len(seen) == EXPECTED_N_SUBJECTS * EXPECTED_N_RUNS


def validate_mask_and_subject_maps(
    subjects: list[str],
) -> tuple[nib.Nifti1Image, np.ndarray]:
    mask_path = WHOLE_BRAIN_DIR / "common_intersection_mask.nii.gz"
    mask_img = nib.load(mask_path)
    mask_data = finite_3d(mask_img, "common intersection mask")
    assert mask_img.shape == EXPECTED_SHAPE
    assert np.allclose(mask_img.header.get_zooms()[:3], EXPECTED_ZOOMS)
    assert np.all(np.isin(np.unique(mask_data), (0, 1)))
    common_mask = mask_data > 0.5
    assert int(np.count_nonzero(common_mask)) == EXPECTED_MASK_VOXELS

    mask_qc = read_json(WHOLE_BRAIN_DIR / "common_mask_qc.json")
    assert int(mask_qc["n_subjects"]) == EXPECTED_N_SUBJECTS
    assert int(mask_qc["runs_per_subject"]) == EXPECTED_N_RUNS
    assert int(mask_qc["n_input_masks"]) == EXPECTED_N_SUBJECTS * EXPECTED_N_RUNS
    assert int(mask_qc["n_resampled_masks"]) == 1
    assert int(mask_qc["intersection_voxels"]) == EXPECTED_MASK_VOXELS
    assert np.isclose(float(mask_qc["voxel_volume_mm3"]), 27.0)
    assert mask_qc["mask_output_sha256"] == sha256_file(mask_path)

    mask_rows = read_csv_rows(WHOLE_BRAIN_DIR / "mask_input_qc.csv")
    assert len(mask_rows) == EXPECTED_N_SUBJECTS * EXPECTED_N_RUNS
    assert len({(row["subject"], int(row["run"])) for row in mask_rows}) == len(mask_rows)
    assert int(mask_rows[-1]["running_intersection_voxels"]) == EXPECTED_MASK_VOXELS

    manifest_rows = read_csv_rows(WHOLE_BRAIN_DIR / "subject_map_qc.csv")
    manifest = {
        (row["subject"], row["contrast"]): row for row in manifest_rows
    }
    assert len(manifest_rows) == EXPECTED_N_SUBJECTS * len(CONTRASTS)
    assert len(manifest) == len(manifest_rows)

    seen = 0
    for subject in subjects:
        for contrast in CONTRASTS:
            path = (
                FLOAT32_ROOT
                / "subject_contrasts"
                / "eventrelated"
                / subject
                / f"{contrast}.nii.gz"
            )
            assert path.is_file(), f"Missing float32 subject map: {path}"
            image = nib.load(path)
            assert image.header.get_data_dtype() == np.dtype("float32")
            assert same_grid(image, mask_img), f"Grid mismatch: {path}"
            slope = float(getattr(image.dataobj, "slope", 1.0))
            inter = float(getattr(image.dataobj, "inter", 0.0))
            assert np.isclose(slope, 1.0) and np.isclose(inter, 0.0)
            data = finite_3d(image, str(path))
            unique_count = int(np.unique(data).size)
            assert unique_count >= 1000, f"Possible quantization: {path}"
            assert float(np.std(data[common_mask], ddof=1)) > 0
            row = manifest[(subject, contrast)]
            assert row["dtype"] == "float32"
            assert row["shape"] == "53x65x56"
            assert int(row["unique_values"]) == unique_count
            assert row["sha256"] == sha256_file(path)
            seen += 1
    assert seen == 80
    return mask_img, common_mask


def validate_component_rows(
    contrast: str,
    signed: np.ndarray,
    t_map: np.ndarray,
    logp_map: np.ndarray,
    rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> None:
    contrast_rows = [row for row in rows if row["contrast"] == contrast]
    structure = generate_binary_structure(rank=3, connectivity=3)
    for direction, selected in (
        ("positive", signed > 0),
        ("negative", signed < 0),
    ):
        labeled, n_components = label(selected, structure=structure)
        direction_rows = [
            row for row in contrast_rows if row["direction"] == direction
        ]
        expected_key = f"{direction}_components_26_connected"
        assert len(direction_rows) == n_components == int(summary[expected_key])
        assert sum(int(row["n_voxels"]) for row in direction_rows) == int(
            np.count_nonzero(selected)
        )
        actual_sizes = np.bincount(labeled.ravel())[1:]
        assert sorted(int(row["n_voxels"]) for row in direction_rows) == sorted(
            int(value) for value in actual_sizes
        )
        for row in direction_rows:
            assert int(row["connectivity"]) == 26
            assert np.isclose(
                float(row["volume_mm3"]), int(row["n_voxels"]) * 27.0
            )
            peak_t = float(row["peak_t"])
            assert (peak_t > 0) == (direction == "positive")
            t_ijk = tuple(int(row[f"t_peak_{axis}"]) for axis in "ijk")
            p_ijk = tuple(int(row[f"tfce_peak_{axis}"]) for axis in "ijk")
            assert all(
                0 <= value < signed.shape[index]
                for index, value in enumerate(t_ijk)
            )
            assert all(
                0 <= value < signed.shape[index]
                for index, value in enumerate(p_ijk)
            )
            assert np.isclose(t_map[t_ijk], peak_t, rtol=1e-6, atol=1e-6)
            assert (signed[t_ijk] > 0) == (direction == "positive")
            max_logp = float(row["max_tfce_logp"])
            assert np.isclose(logp_map[p_ijk], max_logp, rtol=1e-6, atol=1e-6)
            assert np.isclose(
                float(row["min_tfce_fwe_p"]),
                10.0 ** (-max_logp),
                rtol=1e-9,
            )


def validate_group_inference(
    mask_img: nib.Nifti1Image,
    common_mask: np.ndarray,
) -> dict[str, tuple[int, int]]:
    summary_path = WHOLE_BRAIN_DIR / "group_summary.json"
    provenance_path = WHOLE_BRAIN_DIR / "provenance.json"
    component_path = WHOLE_BRAIN_DIR / "components_peaks.csv"
    summary = read_json(summary_path)
    provenance = read_json(provenance_path)
    assert int(summary["n_subjects"]) == EXPECTED_N_SUBJECTS
    assert int(summary["n_rois"]) == EXPECTED_N_ROIS
    assert int(summary["n_input_run_masks"]) == 160
    assert int(summary["common_mask_voxels"]) == EXPECTED_MASK_VOXELS
    assert int(summary["n_perm"]) == EXPECTED_N_PERM
    assert int(summary["random_seed"]) == EXPECTED_SEED
    assert np.isclose(
        float(summary["primary_per_contrast_alpha"]), PER_CONTRAST_ALPHA
    )
    assert np.isclose(
        float(summary["primary_tfce_logp_threshold"]), TFCE_LOGP_THRESHOLD
    )
    assert set(summary["contrasts"]) == set(CONTRASTS)

    assert int(provenance["n_subjects"]) == EXPECTED_N_SUBJECTS
    assert provenance["contrasts_run"] == list(CONTRASTS)
    assert provenance["planned_contrast_family"] == list(CONTRASTS)
    model = provenance["whole_brain_model"]
    expected_model = {
        "n_perm": EXPECTED_N_PERM,
        "two_sided_test": True,
        "tfce": True,
        "random_seed": EXPECTED_SEED,
        "number_of_planned_contrasts": N_PLANNED_CONTRASTS,
    }
    for key, value in expected_model.items():
        assert model[key] == value, f"whole_brain_model.{key}={model[key]!r}"
    assert model["cluster_forming_threshold"] is None
    assert model["additional_second_level_smoothing_fwhm_mm"] is None
    assert model["primary_error_control"] == (
        "voxelwise TFCE FWE within contrast, Bonferroni across four contrasts"
    )
    assert np.isclose(float(model["family_alpha"]), FAMILY_ALPHA)
    assert np.isclose(float(model["per_contrast_alpha"]), PER_CONTRAST_ALPHA)
    assert np.isclose(
        float(model["negative_log10_p_threshold"]), TFCE_LOGP_THRESHOLD
    )
    assert bool(model["permutations_resolve_primary_threshold"])
    roi_model = provenance["roi_model"]
    assert int(roi_model["n_prespecified_rois"]) == EXPECTED_N_ROIS
    assert roi_model["multiplicity"] == (
        "Benjamini-Hochberg FDR across 48 ROIs separately within each contrast"
    )

    component_rows = read_csv_rows(component_path, allow_empty=True)
    counts: dict[str, tuple[int, int]] = {}
    for contrast in CONTRASTS:
        contrast_dir = WHOLE_BRAIN_DIR / "group" / contrast
        arrays: dict[str, np.ndarray] = {}
        contrast_summary = summary["contrasts"][contrast]
        assert int(contrast_summary["n_subjects"]) == EXPECTED_N_SUBJECTS
        assert int(contrast_summary["significant_rois_bh_fdr_0_05"]) == (
            EXPECTED_ROI_COUNTS[contrast][0]
        )
        assert (
            float(
                contrast_summary["parametric_vs_permutation_t_correlation"]
            )
            >= 0.999999
        )
        assert math.isfinite(
            float(
                contrast_summary[
                    "parametric_vs_permutation_t_max_abs_difference"
                ]
            )
        )
        for key, filename in GROUP_MAP_FILES.items():
            path = contrast_dir / filename
            assert path.is_file(), f"Missing group map: {path}"
            image = nib.load(path)
            assert image.header.get_data_dtype() == np.dtype("float32")
            assert same_grid(image, mask_img), f"Group map grid mismatch: {path}"
            values = finite_3d(image, str(path))
            assert np.all(values[~common_mask] == 0), (
                f"Nonzero values outside mask: {path}"
            )
            arrays[key] = values
            recorded = package_path(contrast_summary["maps"][key])
            assert recorded.resolve() == path.resolve()

        assert np.all(arrays["logp_max_tfce"][common_mask] >= 0)
        assert np.all(arrays["logp_max_t"][common_mask] >= 0)
        significant = common_mask & (
            arrays["logp_max_tfce"] >= TFCE_LOGP_THRESHOLD
        )
        reconstructed = np.where(significant, arrays["t"], 0.0)
        assert np.allclose(
            reconstructed,
            arrays["signed_thresholded_t"],
            rtol=1e-6,
            atol=1e-6,
        ), f"Signed-map reconstruction failed: {contrast}"
        positive = int(np.count_nonzero(reconstructed > 0))
        negative = int(np.count_nonzero(reconstructed < 0))
        counts[contrast] = (positive, negative)
        assert positive == int(
            contrast_summary["positive_suprathreshold_voxels"]
        )
        assert negative == int(
            contrast_summary["negative_suprathreshold_voxels"]
        )
        assert positive + negative == int(
            contrast_summary["total_suprathreshold_voxels"]
        )
        assert np.isclose(
            float(contrast_summary["max_logp_tfce_in_mask"]),
            float(np.max(arrays["logp_max_tfce"][common_mask])),
            rtol=1e-6,
            atol=1e-6,
        )
        validate_component_rows(
            contrast,
            arrays["signed_thresholded_t"],
            arrays["t"],
            arrays["logp_max_tfce"],
            component_rows,
            contrast_summary,
        )

    assert len(component_rows) == int(summary["total_descriptive_components"])

    manifest_rows = read_csv_rows(WHOLE_BRAIN_DIR / "output_manifest.csv")
    assert len({row["path"] for row in manifest_rows}) == len(manifest_rows)
    for row in manifest_rows:
        path = package_path(row["path"])
        assert path.is_file(), f"Manifest output missing: {path}"
        assert int(row["bytes"]) == path.stat().st_size
        assert row["sha256"] == sha256_file(path), f"Checksum mismatch: {path}"
    required_manifest_paths = {
        summary_path.resolve(),
        provenance_path.resolve(),
        component_path.resolve(),
        (ROI_DIR / "roi_results.csv").resolve(),
    }
    manifest_paths = {
        package_path(row["path"]).resolve() for row in manifest_rows
    }
    assert required_manifest_paths <= manifest_paths
    return counts


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    assert header[:8] == PNG_SIGNATURE, f"Invalid PNG signature: {path}"
    assert header[12:16] == b"IHDR", f"Missing PNG IHDR: {path}"
    return struct.unpack(">II", header[16:24])


def validate_figures(group_counts: dict[str, tuple[int, int]]) -> None:
    metadata = read_json(FIGURE_DIR / "brain_figure_metadata.json")
    assert source_matches(metadata["script"]["sha256"], "92_make_brain_figures.py")
    assert metadata["analysis_role"] == (
        "visualization of validated ROI-mean and volume-space inference; "
        "no additional statistical inference"
    )
    rules = metadata["statistical_display_rules"]
    assert int(rules["n_subjects"]) == EXPECTED_N_SUBJECTS
    assert int(rules["df"]) == EXPECTED_N_SUBJECTS - 1
    assert rules["roi"]["voxel_or_vertex_inference"] is False
    whole = rules["whole_brain"]
    assert whole["two_sided"] is True and whole["tfce"] is True
    assert int(whole["n_permutations"]) == EXPECTED_N_PERM
    assert whole["voxelwise_fwe_within_contrast"] is True
    assert int(whole["bonferroni_across_planned_contrasts"]) == 4
    assert np.isclose(
        float(whole["per_contrast_alpha"]), PER_CONTRAST_ALPHA
    )
    assert np.isclose(
        float(whole["negative_log10_p_threshold"]), TFCE_LOGP_THRESHOLD
    )
    assert int(metadata["common_mask"]["voxels"]) == EXPECTED_MASK_VOXELS

    display = metadata["volume_visualization"]
    assert display["plane"] == "axial"
    assert display["orientation"] == "neurological"
    assert display["left_side_of_slice"] == "left hemisphere"
    assert display["template"] == "MNI152NLin2009cAsym"
    assert display["anatomical_grid_orientation"] == ["R", "A", "S"]
    assert display["anatomical_voxel_size_mm"] == [1.0, 1.0, 1.0]
    assert display["statistical_interpolation"] == "nearest"
    assert display["display_smoothing"] is None
    assert display["spatial_dilation"] is False
    assert display["surface_projection"] is False
    for key in ("anatomy", "anatomical_brain_mask"):
        record = display[key]
        path = package_path(record["path"])
        assert path.is_file() and path.stat().st_size == int(record["bytes"])
        assert sha256_file(path) == record["sha256"]
    cuts = [-54, -36, -18, -6, 6, 18, 30, 36, 48, 57, 60, 72]
    assert display["axial_cut_coords_mm"] == cuts
    for key in ("roi_figure", "whole_brain_figure"):
        assert metadata[key]["axial_cut_coords_mm"] == cuts
    assert metadata["roi_figure"]["axial_slices_include_noncortical_rois"] is True
    detail = metadata["whole_brain_figure"]["early_detail"]
    early_img = nib.load(WHOLE_BRAIN_DIR / "group" / CONTRASTS[1] /
                         GROUP_MAP_FILES["signed_thresholded_t"])
    occupied = np.argwhere(np.asarray(early_img.dataobj) != 0)
    world = nib.affines.apply_affine(early_img.affine, occupied)
    occupied_z = sorted(float(z) for z in np.unique(world[:, 2]))
    assert detail["axial_cut_coords_mm"] == occupied_z
    counts = [int(np.count_nonzero(np.isclose(world[:, 2], z)))
              for z in occupied_z]
    assert detail["native_voxels_per_slice"] == counts
    assert detail["total_covered_native_voxels"] == sum(counts) == 7
    assert detail["spatial_dilation"] is False
    xmin, xmax, ymin, ymax = detail["bounds_xy_mm"]
    assert np.all((world[:, 0] >= xmin) & (world[:, 0] <= xmax) &
                  (world[:, 1] >= ymin) & (world[:, 1] <= ymax))

    roi_figure = metadata["roi_figure"]
    assert int(
        roi_figure["contrasts"]["Str1_minus_Ran_r12"]["n_significant_rois"]
    ) == 29
    assert int(
        roi_figure["contrasts"]["Switch_r34_vs_r12_did"][
            "n_significant_rois"
        ]
    ) == 8
    whole_figure = metadata["whole_brain_figure"]
    assert whole_figure["displayed_contrasts"] == list(CONTRASTS)
    assert whole_figure["null_contrasts_retained_and_labeled"] is True
    for contrast, (positive, negative) in group_counts.items():
        row = whole_figure["contrasts"][contrast]
        assert int(row["positive_suprathreshold_voxels"]) == positive
        assert int(row["negative_suprathreshold_voxels"]) == negative
        assert int(row["total_suprathreshold_voxels"]) == positive + negative
        assert bool(row["is_null"]) == (positive + negative == 0)
        assert (
            float(row["signed_map_reconstruction_max_abs_difference"]) <= 1e-6
        )

    for key, filename in (
        ("roi_figure", "roi_surface_and_volume.png"),
        ("whole_brain_figure", "whole_brain_surface_and_volume.png"),
    ):
        path = FIGURE_DIR / filename
        assert path.is_file() and path.stat().st_size >= 10_000
        record = metadata[key]["output"]
        assert package_path(record["path"]).resolve() == path.resolve()
        assert int(record["bytes"]) == path.stat().st_size
        assert record["sha256"] == sha256_file(path)
        assert int(metadata[key]["dpi"]) >= 300
        assert list(png_dimensions(path)) == metadata[key][
            "expected_pixel_dimensions"
        ]


def validate_sensitivity_analysis() -> None:
    required = [
        SENSITIVITY_DIR / "ANALYSIS_SPEC.md",
        SENSITIVITY_DIR / "REPORT.md",
        SENSITIVITY_DIR / "censor_mask_manifest.csv",
        SENSITIVITY_DIR / "model_contrast_summary.csv",
        SENSITIVITY_DIR / "pairwise_agreement.csv",
        SENSITIVITY_DIR / "roi_results.csv",
        SENSITIVITY_DIR / "roi_stability.csv",
        SENSITIVITY_DIR / "subject_map_manifest.csv",
        SENSITIVITY_DIR / "provenance.json",
        MANUSCRIPT_DIR / "tables" / "sensitivity_roi_counts.tex",
    ]
    assert all(path.is_file() for path in required)

    provenance = read_json(SENSITIVITY_DIR / "provenance.json")
    assert provenance["analysis"] == "nilearn_2x2_motion_censor_temporal_filter_roi_sensitivity"
    assert int(provenance["n_subjects"]) == 20
    assert int(provenance["n_rois"]) == 48
    censor_qc = provenance["censor_qc"]
    assert int(censor_qc["n_total"]) == 32_000
    assert int(censor_qc["n_censored"]) == 759
    assert int(censor_qc["minimum_retained_per_run"]) == 152

    censor_files = sorted(SENSITIVITY_INPUT_DIR.glob("sub-*_afni_v8_censor.1D"))
    assert len(censor_files) == 20
    assert sum(int(np.count_nonzero(np.loadtxt(path) == 0)) for path in censor_files) == 759
    censor_manifest = pd.read_csv(SENSITIVITY_DIR / "censor_mask_manifest.csv")
    assert len(censor_manifest) == 20
    for row in censor_manifest.to_dict("records"):
        path = (MANUSCRIPT_DIR / row["path"]) if row["path"].startswith("data/") else recorded_path(row["path"])
        assert path.is_file()
        assert row["sha256"] == sha256_file(path)
        assert int(row["n_values"]) == 1600
        assert int(row["n_censored"]) + int(row["n_retained"]) == 1600

    summary = pd.read_csv(SENSITIVITY_DIR / "model_contrast_summary.csv")
    assert len(summary) == 16
    stages = ("initial", "early", "middle", "late")
    for model, expected in EXPECTED_SENSITIVITY_COUNTS.items():
        rows = summary[summary["model"] == model].set_index("stage")
        assert tuple(int(rows.loc[stage, "n_fdr"]) for stage in stages) == expected

    roi_results = pd.read_csv(SENSITIVITY_DIR / "roi_results.csv")
    assert len(roi_results) == 4 * 4 * 48
    assert set(roi_results["n"].astype(int)) == {20}
    assert np.isfinite(
        roi_results[["mean", "sd", "t", "p_two_sided", "q_bh_within_contrast"]]
        .to_numpy(dtype=float)
    ).all()

    stability = pd.read_csv(SENSITIVITY_DIR / "roi_stability.csv")
    assert len(stability) == 4 * 48
    same_direction = tuple(
        int(stability.loc[stability.stage == stage, "same_direction_all_models"].sum())
        for stage in stages
    )
    stable_fdr = tuple(
        int(stability.loc[stability.stage == stage, "fdr_all_models"].sum())
        for stage in stages
    )
    assert same_direction == (42, 46, 37, 29)
    assert stable_fdr == (14, 0, 0, 0)

    qc_paths = sorted(
        SENSITIVITY_FLA_ROOT.glob("*/level1/eventrelated/sub-*/run-*/model_qc.json")
    )
    map_paths = sorted(
        SENSITIVITY_FLA_ROOT.glob(
            "*/subject_contrasts/eventrelated/sub-*/*.nii.gz"
        )
    )
    assert len(qc_paths) == 3 * 20 * 8
    assert len(map_paths) == 3 * 20 * 4
    dropped: list[tuple[str, list[str]]] = []
    for path in qc_paths:
        qc = read_json(path)
        assert int(qc["matrix_rank"]) == int(qc["n_columns"])
        assert int(qc["residual_df"]) >= 10
        assert qc["warnings"] == []
        if qc.get("dropped_redundant_nuisance_columns"):
            dropped.append((str(path), qc["dropped_redundant_nuisance_columns"]))
    assert len(dropped) == 1
    assert "sub-012/run-06" in dropped[0][0]
    assert dropped[0][1] == ["stop_cos_099"]
    for path in map_paths:
        image = nib.load(path)
        assert image.header.get_data_dtype() == np.dtype("float32")
        assert np.isfinite(np.asarray(image.dataobj)).all()


def validate_reported_censored_products() -> None:
    """Reproduce reported ROI tests and check their displayed products, without fitting."""
    script = Path(__file__).with_name("103_make_censored_roi_figure.py")
    spec = importlib.util.spec_from_file_location("srtt_reported_roi", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, selected, _, input_records = module.validate_inputs()
    metadata = read_json(FIGURE_DIR / "censored_roi_metadata.json")
    assert metadata["model"] == "censor_hp128"
    assert metadata["n_subjects"] == 20 and metadata["df"] == 19
    assert metadata["roi_counts"] == [31, 15, 5, 0]
    assert metadata["analysis_role"] == "visualization of ROI-average inference; no voxelwise inference"
    assert metadata["inputs"][:len(input_records)] == input_records
    for contrast in CONTRASTS:
        row = metadata["contrasts"][contrast]
        assert row["rois"] == selected[contrast]
        assert row["n_significant_rois"] == len(selected[contrast])
    assert metadata["plane"] == "axial" and metadata["orientation"] == "neurological"
    assert metadata["axial_cut_coords_mm"] == [-54, -36, -18, -6, 6, 18, 30, 36, 48, 57, 60, 72]
    assert metadata["statistical_interpolation"] == "nearest"
    assert metadata["display_smoothing"] is None and metadata["spatial_dilation"] is False
    records = metadata["inputs"] + list(metadata["tables"].values())
    records += [metadata[key] for key in ("script", "display_helpers", "anatomy", "anatomical_brain_mask", "output")]
    for record in records:
        path = package_path(record["path"])
        assert path.is_file() and path.stat().st_size == record["bytes"], path
        assert sha256_file(path) == record["sha256"], path
    assert metadata["script"]["sha256"] == sha256_file(script)
    assert set(metadata["tables"]) == {"censored_nilearn.tex", "censored_significant_rois.tex"}
    assert metadata["dpi"] >= 300
    assert list(png_dimensions(FIGURE_DIR / "censored_roi_axial.png")) == metadata["expected_pixel_dimensions"]


def validate_behavioral_rt_figure() -> None:
    """Check reconstructed RT summaries, held-out predictions, and displayed fits."""
    metadata = read_json(FIGURE_DIR / "behavioral_rt_cycles_metadata.json")
    assert metadata["n_learners"] == 42 and metadata["trials_per_cycle"] == 8
    assert metadata["expected_pixel_dimensions"] == [2850, 1950]
    assert list(png_dimensions(FIGURE_DIR / "behavioral_rt_cycles.png")) == [2850, 1950]
    for record in metadata["inputs"] + metadata["scripts"] + metadata["outputs"]:
        path = package_path(record["path"])
        assert path.is_file() and path.stat().st_size == record["bytes"], path
        assert sha256_file(path) == record["sha256"], path
    directory = PACKAGE_ROOT / "results/behavioral/rt_curve_comparison_20261008"
    participants = pd.read_csv(directory / "participant_cycles.csv")
    group = pd.read_csv(directory / "group_cycles.csv")
    assert participants.participant.nunique() == 42 and len(group) == 240
    assert participants.n_trials.between(1, 8).all()
    assert not participants.duplicated(["participant", "structure", "structure_cycle"]).any()
    keys = ["structure", "stage", "phase_cycle", "displayed_cycle"]
    calculated = participants.groupby(keys).rt_ms.agg(["mean", "std", "count"]).reset_index()
    comparison = group.merge(calculated, on=keys, validate="one_to_one")
    assert len(comparison) == 240
    assert np.allclose(comparison.rt_ms, comparison["mean"], rtol=0, atol=1e-9)
    assert np.allclose(comparison.sem_ms, comparison["std"] / np.sqrt(comparison["count"]), rtol=0, atol=1e-9)
    assert np.array_equal(comparison.n_participants, comparison["count"])
    scores = pd.read_csv(directory / "model_comparison.csv")
    predictions = pd.read_csv(directory / "cycle_predictions.csv")
    folds = pd.read_csv(directory / "fold_scores.csv")
    assert len(scores) == 16 and len(predictions) == 960 and len(folds) == 80
    assert set(scores.model) == {"Linear", "Logarithmic", "Exponential", "Power law"}
    for row in scores.itertuples():
        selection = ((predictions.structure == row.structure) & (predictions.stage == row.stage)
                     & (predictions.model == row.model))
        d = predictions[selection].sort_values("phase_cycle")
        assert np.array_equal(d.phase_cycle, np.arange(1, 61))
        observed = group[(group.structure == row.structure) & (group.stage == row.stage)].sort_values("phase_cycle")
        assert np.allclose(d.observed_rt_ms, observed.rt_ms, rtol=0, atol=1e-9)
        held_out_rmse = np.sqrt(np.mean((d.observed_rt_ms - d.held_out_rt_ms) ** 2))
        training_rmse = np.sqrt(np.mean((d.observed_rt_ms - d.fitted_rt_ms) ** 2))
        assert np.isclose(held_out_rmse, row.blocked_cv_rmse_ms, rtol=0, atol=1e-9)
        assert np.isclose(training_rmse, row.training_rmse_ms, rtol=0, atol=1e-9)
        t = d.phase_cycle.to_numpy(float)
        if row.model == "Linear": feature = t
        elif row.model == "Logarithmic": feature = np.log(t)
        elif row.model == "Exponential": feature = np.exp(-t / row.shape_parameter)
        else: feature = t ** (-row.shape_parameter)
        assert np.allclose(row.a + row.b * feature, d.fitted_rt_ms, rtol=0, atol=1e-8)
        f = folds[(folds.structure == row.structure) & (folds.stage == row.stage)
                  & (folds.model == row.model)].sort_values("fold")
        assert np.array_equal(f.held_out_first_cycle, [1, 13, 25, 37, 49])
        assert np.array_equal(f.held_out_last_cycle, [12, 24, 36, 48, 60])
        assert np.isclose(np.sqrt(np.mean(f.rmse_ms ** 2)), held_out_rmse, rtol=0, atol=1e-9)
    ranks = scores.groupby(["structure", "stage"]).blocked_cv_rmse_ms.rank(method="min").astype(int)
    assert np.array_equal(ranks, scores.cv_rank)
    winners = scores[scores.cv_rank == 1]
    assert len(winners) == 4
    assert winners[["structure", "stage", "model"]].to_dict("records") == [
        {key: row[key] for key in ["structure", "stage", "model"]} for row in metadata["best_models"]]
    for record in read_json_list(directory / "raw_input_hashes.json"):
        path = Path(record["path"])
        assert path.is_file() and sha256_file(path) == record["sha256"], path


def read_json_list(path: Path) -> list:
    value = json.loads(path.read_text())
    assert isinstance(value, list)
    return value


def resolve_graphic(source: str) -> Path:
    path = MANUSCRIPT_DIR / source
    if path.suffix:
        return path
    candidates = [
        path.with_suffix(extension)
        for extension in (".pdf", ".png", ".jpg", ".jpeg")
    ]
    matches = [candidate for candidate in candidates if candidate.is_file()]
    assert len(matches) == 1, f"Could not resolve graphic {source!r}: {matches}"
    return matches[0]


def read_tex_document(path: Path, stack: tuple[Path, ...] = ()) -> tuple[str, list[Path]]:
    """Expand nested TeX inputs relative to the manuscript compilation directory."""
    path = path.resolve()
    assert path not in stack, f"Circular TeX input: {(*stack, path)}"
    assert path.is_file(), f"Missing input file: {path}"
    assert_balanced_tex(path)
    text = strip_tex_comments(path.read_text(encoding="utf-8"))
    inputs: list[Path] = []

    def expand(match: re.Match[str]) -> str:
        child = MANUSCRIPT_DIR / match.group(1)
        if not child.suffix:
            child = child.with_suffix(".tex")
        child_text, nested = read_tex_document(child, (*stack, path))
        inputs.append(child.resolve())
        inputs.extend(nested)
        return child_text

    return re.sub(r"\\input\s*\{([^}]+)\}", expand, text), inputs


def validate_manuscript_structure() -> tuple[int, int, int]:
    required = [
        MAIN_TEX,
        BIB_FILE,
        MANUSCRIPT_DIR / "supplement.tex",
        MANUSCRIPT_DIR / "tables" / "censored_nilearn.tex",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"Missing required files: {missing}"

    main_text, input_paths = read_tex_document(MAIN_TEX)
    citation_groups = re.findall(
        r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^{}]+)\}", main_text
    )
    citation_keys = {
        key.strip()
        for group in citation_groups
        for key in group.split(",")
        if key.strip()
    }
    bibliography_keys = set(
        re.findall(
            r"@\w+\s*\{\s*([^,\s]+)\s*,",
            BIB_FILE.read_text(encoding="utf-8"),
        )
    )
    assert citation_keys <= bibliography_keys, (
        f"Missing bibliography keys: {sorted(citation_keys - bibliography_keys)}"
    )


    graphic_sources = re.findall(
        r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", main_text
    )
    graphic_paths = [resolve_graphic(source) for source in graphic_sources]
    relative_graphics = {
        str(path.relative_to(MANUSCRIPT_DIR)) for path in graphic_paths
    }
    assert len(graphic_paths) == len(EXPECTED_GRAPHICS)
    assert relative_graphics == EXPECTED_GRAPHICS, (
        f"Expected manuscript graphics {sorted(EXPECTED_GRAPHICS)}, found "
        f"{sorted(relative_graphics)}"
    )
    assert r"\usepackage{graphicx}" in main_text

    forbidden = [
        "Chan (2025)",
        "Chan_2025",
        "cross-pipeline",
        "four-pipeline",
        "robust consensus",
        "consensus analysis",
        "pipeline dependence",
        "pipeline robustness",
        "spm25",
        "blocked-model",
        r"\input{supplement}",
    ]
    supplement_path = MANUSCRIPT_DIR / "supplement.tex"
    assert supplement_path.resolve() not in input_paths, "Supplement must remain a separate document"
    supplement_text, supplement_inputs = read_tex_document(supplement_path)
    supplement_citations = {
        key.strip()
        for group in re.findall(
            r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^{}]+)\}", supplement_text
        )
        for key in group.split(",")
    }
    assert supplement_citations <= bibliography_keys
    assert bibliography_keys <= citation_keys | supplement_citations, (
        f"Unused bibliography keys: {sorted(bibliography_keys - (citation_keys | supplement_citations))}"
    )
    supplement_graphics = re.findall(
        r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", supplement_text
    )
    assert supplement_graphics == ["figures/whole_brain_surface_and_volume.png"]
    assert resolve_graphic(supplement_graphics[0]).is_file()
    methods = main_text.split(r"\section{Methods}", 1)[1].split(r"\section{Results}", 1)[0]
    method_titles = re.findall(r"\\subsection\{([^}]+)\}", methods)
    assert method_titles[5:7] == [
        "Sensitivity analyses", "First-level residual and task-regressor diagnostics"
    ], "Main Methods 2.6/2.7 must retain the requested concise structure"
    assert r"Supplementary Methods S1--S2" in main_text
    assert "Supplementary Methods S4" in main_text
    assert len(re.findall(r"\\section\{", supplement_text)) == 4
    for label in ("sec:s1", "sec:s2", "sec:s3", "sec:s4", "tab:sub022", "fig:referencewholebrain", "tab:modelalternatives", "tab:censoredfull", "tab:sensitivity", "tab:wholebrain"):
        assert r"\label{" + label + "}" in supplement_text
    for table in ("model_alternatives", "sensitivity_roi_counts", "whole_brain_components", "censored_significant_rois"):
        path = (MANUSCRIPT_DIR / "tables" / (table + ".tex")).resolve()
        assert path in supplement_inputs and path not in input_paths
    assert (MANUSCRIPT_DIR / "tables/censored_nilearn.tex").resolve() in input_paths
    assert (MANUSCRIPT_DIR / "tables/primary_nilearn.tex").resolve() not in input_paths
    assert "Supplementary Methods S3" in main_text
    for expanded in (main_text, supplement_text):
        assert not re.search(r"\bFIR\b|finite[- ]impulse", expanded, re.IGNORECASE)
    results = main_text.split(r"\section{Results}", 1)[1].split(r"\section{Discussion}", 1)[0]
    assert "censored" in results and r"\label{tab:reported}" in results
    assert "exploratory" in main_text and "do not account for model selection" in main_text
    # Check every local reference within each independently compiled document.
    for expanded in (main_text, supplement_text):
        labels = re.findall(r"\\label\{([^}]+)\}", expanded)
        references = set(re.findall(r"\\(?:ref|pageref|eqref)\{([^}]+)\}", expanded))
        assert len(labels) == len(set(labels)), "Duplicate TeX labels"
        assert references <= set(labels), f"Undefined TeX references: {references - set(labels)}"

    lower_main = main_text.lower()
    for phrase in forbidden:
        assert phrase.lower() not in lower_main, (
            f"Removed analysis framing remains in manuscript: {phrase}"
        )
    assert "nine positive FDR-significant" not in main_text
    return main_text.count(r"\todo{"), len(citation_keys), len(graphic_paths)


def main() -> None:
    subjects, _ = validate_float32_roi_sources()
    validate_first_level_qc(subjects)
    mask_img, common_mask = validate_mask_and_subject_maps(subjects)
    group_counts = validate_group_inference(mask_img, common_mask)
    validate_figures(group_counts)
    validate_sensitivity_analysis()
    validate_reported_censored_products()
    validate_behavioral_rt_figure()
    todo_count, citation_count, graphic_count = validate_manuscript_structure()
    formatted_counts = ", ".join(
        f"{contrast}=+{positive}/-{negative}"
        for contrast, (positive, negative) in group_counts.items()
    )
    print(
        "PASS: exact float32 ROI tests (29/8/0/0), 160 full-rank warning-free "
        "run models, 80 unscaled float32 subject maps, the 58,611-voxel strict "
        "common mask, primary 10,000-permutation two-sided TFCE/FWE inference, "
        "component summaries, checksummed figures, all 480 full-rank motion/filter "
        "sensitivity fits, their 240 subject maps and ROI summaries, and TeX "
        "dependencies, including the reported censored ROI tests (31/15/5/0) "
        "and reconstructed RT summaries, held-out predictions and curve figures, are internally consistent."
    )
    print(f"Whole-brain suprathreshold voxel counts: {formatted_counts}")
    print(
        f"TeX citations: {citation_count}; graphics: {graphic_count}; "
        f"author-facing TODO markers retained: {todo_count}"
    )


if __name__ == "__main__":
    main()
