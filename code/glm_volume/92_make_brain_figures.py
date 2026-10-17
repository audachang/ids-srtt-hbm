#!/usr/bin/env python3
"""Create the manuscript ROI and corrected whole-brain brain figures.

This program is intentionally a visualization-only companion to
91_run_float32_group_inference.py. It consumes the exact float32 ROI results
and the already thresholded, volume-space TFCE/FWE maps. It never performs a
new statistical test and never substitutes an uncorrected map when
the corrected result is null.

The two publication figures are:

* figures/roi_surface_and_volume.png
* figures/whole_brain_surface_and_volume.png

The corresponding inputs, checksums, thresholds, axial slice settings, selected
ROIs, map reconstruction checks, and output checksums are recorded in
figures/brain_figure_metadata.json.
"""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import nilearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from nilearn.image import resample_to_img  # noqa: E402


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, FIRST_LEVEL_DIR, ROI_MASK_DIR, recorded_path, source_matches

EXPECTED_NILEARN_VERSION = "0.13.1"
EXPECTED_N_SUBJECTS = 20
EXPECTED_DF = 19
EXPECTED_N_ROIS = 48
EXPECTED_N_PERMUTATIONS = 10_000
FAMILY_ALPHA = 0.05
N_PLANNED_CONTRASTS = 4
PER_CONTRAST_ALPHA = FAMILY_ALPHA / N_PLANNED_CONTRASTS
TFCE_LOGP_THRESHOLD = -math.log10(PER_CONTRAST_ALPHA)
FDR_ALPHA = 0.05
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

CONTRASTS = (
    "Str1_minus_Ran_r12",
    "Switch_r34_vs_r12_did",
    "Switch_r56_vs_r12_did",
    "Switch_r78_vs_r12_did",
)

SHORT_LABELS = {
    "Str1_minus_Ran_r12": "Initial: Structure 1 - random",
    "Switch_r34_vs_r12_did": "Early switch: difference-in-differences",
    "Switch_r56_vs_r12_did": "Middle switch: difference-in-differences",
    "Switch_r78_vs_r12_did": "Late switch: difference-in-differences",
}

ROI_FIGURE_CONTRASTS = CONTRASTS[:2]

# These study ROIs are not cortex and are therefore deliberately excluded from
# cortical classification recorded in ROI provenance. All parcels are shown
# directly on axial slices, including the subcortical and cerebellar ROIs.
NONCORTICAL_ROI_TOKENS = (
    "Accumbens",
    "Amygdala",
    "Caudate",
    "Cerebellum",
    "Hippocampus",
    "Pallidum",
    "Putamen",
    "Thalamus",
)

# ColorBrewer-style, color-vision-accessible blue-white-red endpoints.
BLUE_WHITE_RED = LinearSegmentedColormap.from_list(
    "defensible_blue_white_red",
    ("#2166AC", "#F7F7F7", "#B2182B"),
    N=256,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PACKAGE_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": display_path(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def same_grid(
    left: nib.spatialimages.SpatialImage,
    right: nib.spatialimages.SpatialImage,
) -> bool:
    return left.shape[:3] == right.shape[:3] and np.allclose(
        left.affine, right.affine, rtol=0.0, atol=1e-5
    )


def finite_3d(
    image: nib.spatialimages.SpatialImage,
    label: str,
) -> np.ndarray:
    data = np.asarray(image.dataobj)
    if data.shape == image.shape[:3] + (1,):
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"{label}: expected a 3D image, found {data.shape}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"{label}: non-finite values")
    return np.asarray(data, dtype=np.float64)


def image_from_reference(
    reference: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    *,
    dtype: np.dtype | type = np.float32,
) -> nib.Nifti1Image:
    values = np.asarray(data, dtype=dtype)
    if values.shape != reference.shape[:3]:
        raise RuntimeError(
            f"image construction: {values.shape} != {reference.shape[:3]}"
        )
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    return nib.Nifti1Image(values, reference.affine, header=header)


def rounded_symmetric_limit(values: list[np.ndarray]) -> float:
    maxima = [
        float(np.max(np.abs(value[np.isfinite(value)])))
        for value in values
        if value.size and np.isfinite(value).any()
    ]
    raw = max(maxima, default=0.0)
    if raw <= 0:
        return 1.0
    return max(1.0, math.ceil(raw * 2.0) / 2.0)


def parse_boolean(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    if not set(normalized.unique()).issubset(allowed):
        raise RuntimeError(f"{name}: invalid Boolean values {sorted(set(normalized))}")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def validate_group_inputs(
    group_root: Path,
) -> tuple[
    nib.Nifti1Image,
    np.ndarray,
    dict[str, dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Validate and reconstruct every corrected whole-brain display map."""
    summary_path = group_root / "group_summary.json"
    provenance_path = group_root / "provenance.json"
    mask_path = group_root / "common_intersection_mask.nii.gz"
    for path in (summary_path, provenance_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if int(summary.get("n_subjects", -1)) != EXPECTED_N_SUBJECTS:
        raise RuntimeError("group summary is not the fixed 20-subject analysis")
    if int(summary.get("n_perm", -1)) != EXPECTED_N_PERMUTATIONS:
        raise RuntimeError(
            "refusing non-primary inference: group summary must report "
            f"{EXPECTED_N_PERMUTATIONS:,} permutations"
        )
    if int(summary.get("random_seed", -1)) != 20260814:
        raise RuntimeError("group summary does not report the fixed random seed")
    reported_threshold = float(summary.get("primary_tfce_logp_threshold", math.nan))
    if not np.isclose(
        reported_threshold, TFCE_LOGP_THRESHOLD, rtol=0.0, atol=1e-10
    ):
        raise RuntimeError(
            f"group threshold {reported_threshold} != {TFCE_LOGP_THRESHOLD}"
        )

    model = provenance.get("whole_brain_model", {})
    required_model_values = {
        "n_perm": EXPECTED_N_PERMUTATIONS,
        "two_sided_test": True,
        "tfce": True,
        "random_seed": 20260814,
        "number_of_planned_contrasts": N_PLANNED_CONTRASTS,
    }
    for key, expected in required_model_values.items():
        if model.get(key) != expected:
            raise RuntimeError(
                f"provenance whole_brain_model.{key}={model.get(key)!r}; "
                f"expected {expected!r}"
            )
    if not np.isclose(
        float(model.get("per_contrast_alpha", math.nan)),
        PER_CONTRAST_ALPHA,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("provenance per-contrast alpha is not .05/4")

    mask_img = nib.load(mask_path)
    mask_raw = finite_3d(mask_img, "common intersection mask")
    unique_mask = np.unique(mask_raw)
    if not np.all(np.isin(unique_mask, (0, 1))):
        raise RuntimeError(f"common mask is not binary: {unique_mask[:10]}")
    common_mask = mask_raw > 0.5
    mask_voxels = int(np.count_nonzero(common_mask))
    if mask_voxels < 1000:
        raise RuntimeError(f"implausibly small common mask ({mask_voxels} voxels)")
    if mask_voxels != int(summary.get("common_mask_voxels", -1)):
        raise RuntimeError("common-mask voxel count disagrees with group summary")

    group_contrasts = summary.get("contrasts", {})
    if set(group_contrasts) != set(CONTRASTS):
        raise RuntimeError(
            "group summary must contain exactly the four planned contrasts"
        )

    maps: dict[str, dict[str, Any]] = {}
    input_records = [
        file_record(summary_path),
        file_record(provenance_path),
        file_record(mask_path),
    ]
    for contrast in CONTRASTS:
        contrast_dir = group_root / "group" / contrast
        paths = {
            "t": contrast_dir / "one_sample_t_map.nii.gz",
            "logp_tfce": contrast_dir / "logp_max_tfce.nii.gz",
            "signed_thresholded_t": (
                contrast_dir / "signed_t_tfce_fwe_bonferroni4.nii.gz"
            ),
        }
        images = {name: nib.load(path) for name, path in paths.items()}
        for name, image in images.items():
            if not same_grid(image, mask_img):
                raise RuntimeError(
                    f"{contrast} {name}: grid differs from common mask"
                )
        t_data = finite_3d(images["t"], f"{contrast} t map")
        logp_data = finite_3d(images["logp_tfce"], f"{contrast} TFCE logp map")
        signed_data = finite_3d(
            images["signed_thresholded_t"], f"{contrast} signed thresholded map"
        )
        if np.any(logp_data[common_mask] < 0):
            raise RuntimeError(f"{contrast}: negative -log10 corrected p values")
        significant = common_mask & (logp_data >= TFCE_LOGP_THRESHOLD)
        reconstructed = np.where(significant, t_data, 0.0)
        max_difference = float(np.max(np.abs(reconstructed - signed_data)))
        if not np.allclose(
            reconstructed, signed_data, rtol=1e-6, atol=1e-6, equal_nan=False
        ):
            raise RuntimeError(
                f"{contrast}: signed map does not reconstruct from t and TFCE "
                f"logp maps (max difference {max_difference:.6g})"
            )
        if np.any(signed_data[~common_mask] != 0):
            raise RuntimeError(f"{contrast}: signed map extends outside common mask")

        positive = int(np.count_nonzero(signed_data > 0))
        negative = int(np.count_nonzero(signed_data < 0))
        contrast_summary = group_contrasts[contrast]
        expected_counts = (
            int(contrast_summary["positive_suprathreshold_voxels"]),
            int(contrast_summary["negative_suprathreshold_voxels"]),
        )
        if (positive, negative) != expected_counts:
            raise RuntimeError(
                f"{contrast}: map counts {(positive, negative)} != summary "
                f"{expected_counts}"
            )
        if int(contrast_summary.get("n_subjects", -1)) != EXPECTED_N_SUBJECTS:
            raise RuntimeError(f"{contrast}: invalid group sample size")

        display_mask = np.abs(signed_data) > 0
        maps[contrast] = {
            "paths": paths,
            "t": t_data,
            "logp_tfce": logp_data,
            "signed_t": signed_data,
            "display_image": images["signed_thresholded_t"],
            "display_mask_image": image_from_reference(
                mask_img, display_mask.astype(np.uint8), dtype=np.uint8
            ),
            "positive_voxels": positive,
            "negative_voxels": negative,
            "total_voxels": positive + negative,
            "reconstruction_max_abs_difference": max_difference,
        }
        input_records.extend(file_record(path) for path in paths.values())

    return mask_img, common_mask, maps, summary, input_records


def validate_roi_results(
    roi_results_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    if not roi_results_path.is_file():
        raise FileNotFoundError(roi_results_path)
    table = pd.read_csv(roi_results_path)
    required = {
        "contrast",
        "roi",
        "n",
        "t",
        "df",
        "q_bh_within_contrast",
        "significant_fdr_0_05",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"ROI results missing columns: {missing}")
    if len(table) != len(CONTRASTS) * EXPECTED_N_ROIS:
        raise RuntimeError(
            f"ROI results contain {len(table)} rows; expected "
            f"{len(CONTRASTS) * EXPECTED_N_ROIS}"
        )
    if set(table["contrast"]) != set(CONTRASTS):
        raise RuntimeError("ROI results do not contain exactly four contrasts")
    if table.duplicated(["contrast", "roi"]).any():
        raise RuntimeError("ROI results contain duplicate contrast/ROI rows")

    numeric_columns = ("n", "t", "df", "q_bh_within_contrast")
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f"ROI results contain non-finite {column} values")
    if not np.all(table["n"].to_numpy(dtype=int) == EXPECTED_N_SUBJECTS):
        raise RuntimeError("ROI results are not based on 20 participants")
    if not np.all(table["df"].to_numpy(dtype=int) == EXPECTED_DF):
        raise RuntimeError("ROI results do not report df=19")
    if np.any(
        (table["q_bh_within_contrast"] < 0)
        | (table["q_bh_within_contrast"] > 1)
    ):
        raise RuntimeError("ROI q values are outside [0, 1]")

    recomputed_flag = table["q_bh_within_contrast"].to_numpy() <= FDR_ALPHA
    stored_flag = parse_boolean(
        table["significant_fdr_0_05"], "significant_fdr_0_05"
    )
    if not np.array_equal(recomputed_flag, stored_flag):
        raise RuntimeError("stored ROI significance flags disagree with q <= .05")

    selected: dict[str, list[dict[str, Any]]] = {}
    for contrast in CONTRASTS:
        contrast_rows = table.loc[table["contrast"] == contrast].copy()
        if len(contrast_rows) != EXPECTED_N_ROIS:
            raise RuntimeError(f"{contrast}: expected 48 ROI rows")
        sig = contrast_rows.loc[
            contrast_rows["q_bh_within_contrast"] <= FDR_ALPHA
        ].sort_values(["t", "roi"])
        selected[contrast] = [
            {
                "roi": str(row.roi),
                "t": float(row.t),
                "q_bh_within_contrast": float(row.q_bh_within_contrast),
                "cortical": not any(
                    token in str(row.roi) for token in NONCORTICAL_ROI_TOKENS
                ),
            }
            for row in sig.itertuples(index=False)
        ]

    if selected[CONTRASTS[2]] or selected[CONTRASTS[3]]:
        raise RuntimeError(
            "ROI figure specification expects no middle or late ROI to survive FDR"
        )
    if not selected[CONTRASTS[0]] or not selected[CONTRASTS[1]]:
        raise RuntimeError("initial and early ROI panels unexpectedly contain no ROIs")
    return table, selected


def build_roi_display_volumes(
    selected: dict[str, list[dict[str, Any]]],
    reference: nib.Nifti1Image,
    roi_mask_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    outputs: dict[str, dict[str, Any]] = {}
    records_by_path: dict[Path, dict[str, Any]] = {}
    for contrast in ROI_FIGURE_CONTRASTS:
        all_values = np.zeros(reference.shape[:3], dtype=np.float64)
        cortical_values = np.zeros(reference.shape[:3], dtype=np.float64)
        all_union = np.zeros(reference.shape[:3], dtype=bool)
        cortical_union = np.zeros(reference.shape[:3], dtype=bool)
        roi_qc: list[dict[str, Any]] = []

        for row in selected[contrast]:
            mask_path = roi_mask_dir / f"{row['roi']}.nii.gz"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            records_by_path.setdefault(mask_path.resolve(), file_record(mask_path))
            native = nib.load(mask_path)
            native_data = finite_3d(native, f"ROI mask {row['roi']}")
            if not np.any(native_data > 0):
                raise RuntimeError(f"empty ROI mask: {mask_path}")
            if same_grid(native, reference):
                on_reference = native
            else:
                on_reference = resample_to_img(
                    native,
                    reference,
                    interpolation="nearest",
                    copy_header=True,
                    force_resample=True,
                )
            mask = finite_3d(on_reference, f"resampled ROI mask {row['roi']}") > 0
            n_voxels = int(np.count_nonzero(mask))
            if n_voxels == 0:
                raise RuntimeError(f"{row['roi']}: empty after reference resampling")

            # Deterministic overlap rule: the ROI with the larger absolute
            # one-sample t value supplies the displayed color.
            replace_all = mask & (
                (all_values == 0) | (abs(row["t"]) > np.abs(all_values))
            )
            all_values[replace_all] = row["t"]
            all_union |= mask
            if row["cortical"]:
                replace_cortex = mask & (
                    (cortical_values == 0)
                    | (abs(row["t"]) > np.abs(cortical_values))
                )
                cortical_values[replace_cortex] = row["t"]
                cortical_union |= mask
            roi_qc.append(
                {
                    **row,
                    "mask": display_path(mask_path),
                    "mask_sha256": records_by_path[mask_path.resolve()]["sha256"],
                    "native_voxels": int(np.count_nonzero(native_data > 0)),
                    "reference_grid_voxels": n_voxels,
                    "resampled_to_reference": not same_grid(native, reference),
                }
            )

        outputs[contrast] = {
            "all_values": all_values,
            "cortical_values": cortical_values,
            "all_image": image_from_reference(reference, all_values),
            "cortical_image": image_from_reference(reference, cortical_values),
            "all_mask_image": image_from_reference(
                reference, all_union.astype(np.uint8), dtype=np.uint8
            ),
            "cortical_mask_image": image_from_reference(
                reference, cortical_union.astype(np.uint8), dtype=np.uint8
            ),
            "all_display_voxels": int(np.count_nonzero(all_union)),
            "cortical_display_voxels": int(np.count_nonzero(cortical_union)),
            "roi_qc": roi_qc,
        }
    return outputs, list(records_by_path.values())


def save_png(fig: plt.Figure, path: Path, dpi: int, title: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
        metadata={
            "Title": title,
            "Software": (
                f"Matplotlib {matplotlib.__version__}; Nilearn {nilearn.__version__}"
            ),
        },
    )
    plt.close(fig)
    with path.open("rb") as stream:
        signature = stream.read(8)
    if signature != PNG_SIGNATURE:
        raise RuntimeError(f"{path}: invalid PNG signature")
    if path.stat().st_size < 10_000:
        raise RuntimeError(f"{path}: implausibly small figure file")
    return file_record(path)


AXIAL_CUTS_MM = (-54, -36, -18, -6, 6, 18, 30, 36, 48, 57, 60, 72)
EARLY_DETAIL_CUTS_MM = (30, 33, 36, 48, 57, 60)
EARLY_DETAIL_BOUNDS_MM = (12, 66, -30, 42)


def load_axial_anatomy(anatomy_path: Path, brain_mask_path: Path):
    """Load matching 1-mm MNI anatomy; never alter the statistical maps."""
    anatomy = nib.as_closest_canonical(nib.load(anatomy_path))
    brain_mask = nib.as_closest_canonical(nib.load(brain_mask_path))
    if not same_grid(anatomy, brain_mask):
        raise RuntimeError("anatomical template and its brain mask differ in grid")
    if not np.allclose(anatomy.affine[:3, :3], np.eye(3), atol=1e-6):
        raise RuntimeError("axial montage requires the RAS, 1-mm template grid")
    background = finite_3d(anatomy, "MNI T1 anatomy")
    mask = finite_3d(brain_mask, "MNI brain mask") > 0
    background = np.where(mask, background, 0.0)
    indices = np.argwhere(mask)
    low, high = indices.min(axis=0), indices.max(axis=0)
    origin = anatomy.affine[:3, 3]
    x = np.arange(anatomy.shape[0]) + origin[0]
    y = np.arange(anatomy.shape[1]) + origin[1]
    return {
        "image": anatomy,
        "data": background,
        "x": x,
        "y": y,
        "extent": (x[0] - .5, x[-1] + .5, y[0] - .5, y[-1] + .5),
        "bounds": (x[low[0]] - 5, x[high[0]] + 5,
                   y[low[1]] - 5, y[high[1]] + 5),
        "gray_max": float(np.percentile(background[mask], 99.5)),
    }


def axial_plane(image, anatomy, z_mm):
    """Sample a native statistical plane with nearest-neighbour interpolation."""
    from scipy.ndimage import map_coordinates
    x, y = np.meshgrid(anatomy["x"], anatomy["y"], indexing="ij")
    xyz = np.stack((x, y, np.full_like(x, z_mm)), axis=0)
    inverse = np.linalg.inv(image.affine)
    voxel = np.einsum("ij,jxy->ixy", inverse[:3, :3], xyz)
    voxel += inverse[:3, 3, None, None]
    return map_coordinates(np.asarray(image.dataobj), voxel, order=0,
                           mode="constant", cval=0.0, prefilter=False)


def axial_axis(fig, cell, image, anatomy, z_mm, vmax, *, detail=False):
    axis = fig.add_subplot(cell)
    axis.set_facecolor("black")
    z_index = int(round(z_mm - anatomy["image"].affine[2, 3]))
    if not 0 <= z_index < anatomy["image"].shape[2]:
        raise RuntimeError(f"axial slice outside anatomical template: z={z_mm}")
    axis.imshow(anatomy["data"][:, :, z_index].T, origin="lower", cmap="gray",
                vmin=0, vmax=anatomy["gray_max"], extent=anatomy["extent"],
                interpolation="nearest", rasterized=True)
    plane = axial_plane(image, anatomy, z_mm)
    axis.imshow(np.ma.masked_where(plane == 0, plane).T, origin="lower",
                cmap=BLUE_WHITE_RED, norm=Normalize(-vmax, vmax),
                extent=anatomy["extent"], interpolation="nearest", rasterized=True)
    bounds = EARLY_DETAIL_BOUNDS_MM if detail else anatomy["bounds"]
    axis.set_xlim(bounds[0], bounds[1]); axis.set_ylim(bounds[2], bounds[3])
    axis.set_aspect("equal"); axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(.5, -.025, f"z = {z_mm:+d} mm", ha="center", va="top",
              color="white", fontsize=15, transform=axis.transAxes)
    if detail:
        from matplotlib import patheffects
        axis.plot([bounds[0] + 4, bounds[0] + 19], [bounds[2] + 4] * 2,
                  color="white", linewidth=1.5,
                  path_effects=[patheffects.Stroke(linewidth=3.5, foreground="black"),
                                patheffects.Normal()])
        axis.text(bounds[0] + 11.5, bounds[2] + 7, "15 mm", color="white",
                  ha="center", fontsize=11, bbox={"facecolor": "black", "alpha": .65, "pad": 1, "edgecolor": "none"})
    else:
        for xpos, label in ((.025, "L"), (.975, "R")):
            axis.text(xpos, .97, label, ha="left" if xpos < .5 else "right",
                      va="top", color="white", fontsize=13,
                      transform=axis.transAxes)


def axial_strip(fig, grid, row_index, image, anatomy, cuts, vmax, *, detail=False):
    for column, z_mm in enumerate(cuts):
        axial_axis(fig, grid[row_index, column], image, anatomy, z_mm, vmax,
                   detail=detail)


def strip_label(fig, grid, row_index, label):
    axis = fig.add_subplot(grid[row_index, :])
    axis.axis("off")
    axis.text(0, .5, label, color="white", fontsize=17,
              fontweight="semibold", ha="left", va="center",
              transform=axis.transAxes)


def axial_colorbar(fig, vmax, label, bottom):
    axis = fig.add_axes((.34, bottom, .32, .018))
    scalar = ScalarMappable(norm=Normalize(-vmax, vmax), cmap=BLUE_WHITE_RED)
    scalar.set_array([])
    bar = fig.colorbar(scalar, cax=axis, orientation="horizontal",
                       ticks=(-vmax, 0, vmax), extend="both")
    bar.ax.tick_params(labelsize=13, colors="white", length=2)
    bar.outline.set_edgecolor("white")
    bar.set_label(label, color="white", fontsize=14, labelpad=2)


def make_roi_figure(roi_volumes, anatomy, output_path, dpi):
    vmax = rounded_symmetric_limit([roi_volumes[c]["all_values"]
                                    for c in ROI_FIGURE_CONTRASTS])
    width, height = 14.0, 8.0
    fig = plt.figure(figsize=(width, height), facecolor="black")
    grid = fig.add_gridspec(6, 6, left=.035, right=.98, top=.855, bottom=.185,
                           height_ratios=(.22, 1, 1, .22, 1, 1),
                           hspace=.30, wspace=.02)
    fig.suptitle("Primary ROI results: horizontal slices", color="white",
                 fontsize=22, fontweight="bold", x=.5, y=.985)
    fig.text(.5, .920, "ROI means surviving BH FDR; MNI152NLin2009cAsym anatomy; L is on the left",
             color="white", fontsize=14, ha="center")
    for index, contrast in enumerate(ROI_FIGURE_CONTRASTS):
        row = 3 * index
        label = "A  Initial: Structure 1 - random" if index == 0 else "B  Early sequence change: DiD"
        strip_label(fig, grid, row, f"{label}  |  {len(roi_volumes[contrast]['roi_qc'])} ROIs")
        for offset in range(2):
            axial_strip(fig, grid, row + 1 + offset, roi_volumes[contrast]["all_image"],
                        anatomy, AXIAL_CUTS_MM[offset * 6:(offset + 1) * 6], vmax)
    fig.text(.5, .118, "Colors represent whole-ROI mean tests, not voxelwise significance.\n"
             "Overlapping parcels show the larger absolute t. Middle and late: no ROIs survived FDR.",
             color="white", fontsize=13, ha="center", va="center")
    axial_colorbar(fig, vmax, "ROI one-sample t(19): negative (blue), positive (red)", .060)
    output = save_png(fig, output_path, dpi, "Primary ROI results: horizontal slices")
    return {"output": output, "dpi": dpi, "figure_size_inches": [width, height],
            "expected_pixel_dimensions": [round(width * dpi), round(height * dpi)],
            "color_scale_t": [-vmax, vmax], "axial_cut_coords_mm": list(AXIAL_CUTS_MM)}


def make_whole_brain_figure(maps, anatomy, output_path, dpi, n_permutations):
    vmax = rounded_symmetric_limit([maps[c]["signed_t"] for c in CONTRASTS])
    width, height = 14.0, 12.0
    fig = plt.figure(figsize=(width, height), facecolor="black")
    grid = fig.add_gridspec(10, 6, left=.035, right=.98, top=.89, bottom=.14,
                           height_ratios=(.22, 1, 1, .22, 1, 1, .22, .95, .19, .19),
                           hspace=.30, wspace=.02)
    fig.suptitle("Primary whole-brain results: horizontal slices", color="white",
                 fontsize=22, fontweight="bold", x=.5, y=.985)
    fig.text(.5, .938, f"Two-sided TFCE/FWE p <= {PER_CONTRAST_ALPHA:.4f}; "
             f"{n_permutations:,} permutations; neurological orientation (L on the left)",
             color="white", fontsize=14, ha="center")
    for index, contrast in enumerate(CONTRASTS[:2]):
        row = 3 * index
        label = "A  Initial: Structure 1 - random" if index == 0 else "B  Early sequence change: DiD"
        strip_label(fig, grid, row, f"{label}  |  +{maps[contrast]['positive_voxels']} / "
                    f"-{maps[contrast]['negative_voxels']} corrected voxels")
        for offset in range(2):
            axial_strip(fig, grid, row + 1 + offset, maps[contrast]["display_image"],
                        anatomy, AXIAL_CUTS_MM[offset * 6:(offset + 1) * 6], vmax)
    early = maps[CONTRASTS[1]]
    strip_label(fig, grid, 6, "Early detail: right frontal crop at all six occupied axial levels")
    axial_strip(fig, grid, 7, early["display_image"], anatomy,
                EARLY_DETAIL_CUTS_MM, vmax, detail=True)
    occupied = np.argwhere(early["signed_t"] != 0)
    world = nib.affines.apply_affine(early["display_image"].affine, occupied)
    counts = [int(np.count_nonzero(np.isclose(world[:, 2], z)))
              for z in EARLY_DETAIL_CUTS_MM]
    if sum(counts) != early["total_voxels"]:
        raise RuntimeError("early axial detail fails to include all corrected voxels")
    xmin, xmax, ymin, ymax = EARLY_DETAIL_BOUNDS_MM
    if not np.all((world[:, 0] >= xmin) & (world[:, 0] <= xmax) &
                  (world[:, 1] >= ymin) & (world[:, 1] <= ymax)):
        raise RuntimeError("early corrected voxels extend outside the detail crop")
    for row, contrast, label in ((8, CONTRASTS[2], "C  Middle sequence change"),
                                 (9, CONTRASTS[3], "D  Late sequence change")):
        if maps[contrast]["total_voxels"] != 0:
            raise RuntimeError("null-row layout requires the validated corrected null")
        axis = fig.add_subplot(grid[row, :]); axis.axis("off")
        axis.text(.01, .5, f"{label}: no voxels survived the corrected threshold",
                  color="white", fontsize=16, ha="left", va="center",
                  transform=axis.transAxes)
    fig.text(.5, .101, "Native 3-mm statistical voxels; nearest-neighbour display; "
             "no smoothing or spatial dilation.\nEarly detail magnifies the same corrected map; "
             "all seven early voxels are represented.", color="white", fontsize=13,
             ha="center", va="center")
    axial_colorbar(fig, vmax, "One-sample t(19), shown only at corrected voxels", .049)
    output = save_png(fig, output_path, dpi, "Primary whole-brain results: horizontal slices")
    return {"output": output, "dpi": dpi, "figure_size_inches": [width, height],
            "expected_pixel_dimensions": [round(width * dpi), round(height * dpi)],
            "color_scale_t": [-vmax, vmax], "axial_cut_coords_mm": list(AXIAL_CUTS_MM),
            "early_detail": {"axial_cut_coords_mm": list(EARLY_DETAIL_CUTS_MM),
                             "bounds_xy_mm": list(EARLY_DETAIL_BOUNDS_MM),
                             "native_voxels_per_slice": counts,
                             "total_covered_native_voxels": sum(counts),
                             "spatial_dilation": False}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--group-root",
        type=Path,
        default=MANUSCRIPT_DIR / "data" / "whole_brain",
        help="Output directory produced by 91_run_float32_group_inference.py.",
    )
    parser.add_argument(
        "--roi-results",
        type=Path,
        default=None,
        help="Tidy exact-float32 ROI CSV (default: GROUP_ROOT/roi/roi_results.csv).",
    )
    parser.add_argument(
        "--roi-mask-dir",
        type=Path,
        default=(
            ROI_MASK_DIR
        ),
        help="Directory containing the 48 study-defined ROI NIfTI masks.",
    )
    parser.add_argument(
        "--anatomy", type=Path,
        default=PACKAGE_ROOT / "data/templates/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
        help="Matching 1-mm MNI152NLin2009cAsym T1 template for axial display.",
    )
    parser.add_argument(
        "--anatomy-mask", type=Path,
        default=PACKAGE_ROOT / "data/templates/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz",
        help="Brain mask distributed with the anatomical template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MANUSCRIPT_DIR / "figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=360,
        help="Raster resolution; must be at least 300 (default: 360).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing named figure outputs and metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if nilearn.__version__ != EXPECTED_NILEARN_VERSION:
        raise RuntimeError(
            f"this figure audit requires Nilearn {EXPECTED_NILEARN_VERSION}; "
            f"found {nilearn.__version__}"
        )
    if args.dpi < 300:
        raise SystemExit("--dpi must be at least 300 for manuscript figures")

    group_root = args.group_root.resolve()
    roi_results_path = (
        args.roi_results.resolve()
        if args.roi_results is not None
        else group_root / "roi" / "roi_results.csv"
    )
    roi_mask_dir = args.roi_mask_dir.resolve()
    anatomy_path = args.anatomy.resolve()
    anatomy_mask_path = args.anatomy_mask.resolve()
    output_dir = args.output_dir.resolve()
    roi_figure_path = output_dir / "roi_surface_and_volume.png"
    whole_brain_path = output_dir / "whole_brain_surface_and_volume.png"
    metadata_path = output_dir / "brain_figure_metadata.json"
    named_outputs = (roi_figure_path, whole_brain_path, metadata_path)
    existing = [path for path in named_outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "refusing to replace existing outputs without --overwrite: "
            + ", ".join(display_path(path) for path in existing)
        )

    print("Validating primary group inference and thresholded maps...", flush=True)
    mask_img, common_mask, maps, group_summary, input_records = (
        validate_group_inputs(group_root)
    )
    print("Validating exact float32 ROI results...", flush=True)
    _, selected = validate_roi_results(roi_results_path)
    input_records.append(file_record(roi_results_path))
    print("Loading matching MNI anatomy for axial slices...", flush=True)
    anatomy = load_axial_anatomy(anatomy_path, anatomy_mask_path)
    input_records.extend([file_record(anatomy_path), file_record(anatomy_mask_path)])
    print("Constructing study-ROI display volumes...", flush=True)
    roi_volumes, roi_mask_records = build_roi_display_volumes(
        selected, mask_img, roi_mask_dir
    )
    input_records.extend(roi_mask_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Rendering ROI horizontal-slice figure...", flush=True)
    roi_figure = make_roi_figure(
        roi_volumes, anatomy, roi_figure_path, args.dpi
    )
    print("Rendering corrected whole-brain horizontal-slice figure...", flush=True)
    whole_brain_figure = make_whole_brain_figure(
        maps,
        anatomy,
        whole_brain_path,
        args.dpi,
        int(group_summary["n_perm"]),
    )

    unique_inputs: dict[str, dict[str, Any]] = {}
    for record in input_records:
        unique_inputs[record["path"]] = record
    metadata = {
        "schema_version": "2.0",
        "created_utc": utc_now(),
        "analysis_role": (
            "visualization of validated ROI-mean and volume-space inference; "
            "no additional statistical inference"
        ),
        "command": [sys.executable, *sys.argv],
        "script": file_record(Path(__file__)),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "nilearn": nilearn.__version__,
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "statistical_display_rules": {
            "n_subjects": EXPECTED_N_SUBJECTS,
            "df": EXPECTED_DF,
            "roi": {
                "source": display_path(roi_results_path),
                "criterion": "q_bh_within_contrast <= 0.05",
                "multiplicity": (
                    "Benjamini-Hochberg FDR across 48 study-defined ROIs "
                    "separately within each contrast"
                ),
                "inferential_unit": "mean across the entire ROI mask",
                "voxel_or_vertex_inference": False,
            },
            "whole_brain": {
                "source": display_path(group_root),
                "two_sided": True,
                "tfce": True,
                "n_permutations": int(group_summary["n_perm"]),
                "voxelwise_fwe_within_contrast": True,
                "bonferroni_across_planned_contrasts": N_PLANNED_CONTRASTS,
                "family_alpha": FAMILY_ALPHA,
                "per_contrast_alpha": PER_CONTRAST_ALPHA,
                "negative_log10_p_threshold": TFCE_LOGP_THRESHOLD,
                "threshold_comparison": ">=",
                "surface_inference": False,
            },
        },
        "common_mask": {
            "shape": [int(value) for value in mask_img.shape[:3]],
            "affine": np.asarray(mask_img.affine, dtype=float).tolist(),
            "orientation": list(nib.aff2axcodes(mask_img.affine)),
            "voxels": int(np.count_nonzero(common_mask)),
        },
        "volume_visualization": {
            "plane": "axial",
            "orientation": "neurological",
            "left_side_of_slice": "left hemisphere",
            "template": "MNI152NLin2009cAsym",
            "anatomy": file_record(anatomy_path),
            "anatomical_brain_mask": file_record(anatomy_mask_path),
            "anatomical_grid_orientation": list(nib.aff2axcodes(anatomy["image"].affine)),
            "anatomical_voxel_size_mm": [float(v) for v in anatomy["image"].header.get_zooms()[:3]],
            "background": "native 1-mm brain-masked T1 anatomy",
            "statistical_interpolation": "nearest",
            "display_smoothing": None,
            "spatial_dilation": False,
            "surface_projection": False,
            "axial_cut_coords_mm": list(AXIAL_CUTS_MM),
            "slice_selection": "common descriptive whole-brain levels; enlarged early views include every occupied axial level",
            "color_map": {"name": BLUE_WHITE_RED.name, "negative": "#2166AC",
                          "zero": "#F7F7F7", "positive": "#B2182B"},
        },
        "roi_figure": {
            **roi_figure,
            "displayed_contrasts": list(ROI_FIGURE_CONTRASTS),
            "middle_and_late_explicitly_reported_null": True,
            "whole_mask_colored_from_roi_mean_test": True,
            "axial_slices_include_noncortical_rois": True,
            "overlap_rule": (
                "where significant ROI masks overlap after nearest-neighbour "
                "resampling, display the ROI with the largest absolute t"
            ),
            "contrasts": {
                contrast: {
                    "n_significant_rois": len(selected[contrast]),
                    "n_cortical_rois": sum(
                        bool(row["cortical"]) for row in selected[contrast]
                    ),
                    "n_noncortical_rois": sum(
                        not bool(row["cortical"]) for row in selected[contrast]
                    ),
                    "all_display_voxels": roi_volumes[contrast][
                        "all_display_voxels"
                    ],
                    "cortical_display_voxels": roi_volumes[contrast][
                        "cortical_display_voxels"
                    ],
                    "rois": roi_volumes[contrast]["roi_qc"],
                }
                for contrast in ROI_FIGURE_CONTRASTS
            },
        },
        "whole_brain_figure": {
            **whole_brain_figure,
            "displayed_contrasts": list(CONTRASTS),
            "null_contrasts_retained_and_labeled": True,
            "volume_source": (
                "signed t map thresholded in volume at the primary corrected "
                "TFCE/FWE threshold"
            ),
            "contrasts": {
                contrast: {
                    "positive_suprathreshold_voxels": maps[contrast][
                        "positive_voxels"
                    ],
                    "negative_suprathreshold_voxels": maps[contrast][
                        "negative_voxels"
                    ],
                    "total_suprathreshold_voxels": maps[contrast]["total_voxels"],
                    "is_null": maps[contrast]["total_voxels"] == 0,
                    "signed_map_reconstruction_max_abs_difference": maps[contrast][
                        "reconstruction_max_abs_difference"
                    ],
                    "map_paths": {
                        key: display_path(path)
                        for key, path in maps[contrast]["paths"].items()
                    },
                }
                for contrast in CONTRASTS
            },
        },
        "inputs": [unique_inputs[key] for key in sorted(unique_inputs)],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Complete: {roi_figure_path}", flush=True)
    print(f"Complete: {whole_brain_path}", flush=True)
    print(f"Provenance: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
