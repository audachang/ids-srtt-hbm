#!/usr/bin/env python3
"""Refit the primary event-related Nilearn model with float32 image outputs.

The earlier retained maps inherited a uint8 mask header and were quantized on
write. This program reproduces the manuscript's run-wise model but explicitly
writes floating-point condition estimates and the four primary subject
contrasts needed for ROI and whole-brain inference.

Run from manuscript/ with the pinned environment:

    python scripts/90_run_float32_eventrelated.py --subjects sub-001

The default output is the canonical project directory
results/glm_volume/first_level/FLA_nilearn_float32_20260814.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, FIRST_LEVEL_DIR, ROI_MASK_DIR

from _layout import BIDS_DIR, FMRIPREP_DIR, SPACE, TASK_NAME, read_subjects  # noqa: E402


TR = 2.0
N_RUNS = 8
CONFOUND_COLS = (
    "csf",
    "white_matter",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
)
CONDITIONS = {
    1: ("Ran", "Str1"),
    2: ("Ran", "Str1"),
    3: ("Ran", "Str1", "Str2"),
    4: ("Ran", "Str1", "Str2"),
    5: ("Ran", "Str1", "Str2"),
    6: ("Ran", "Str1", "Str2"),
    7: ("Ran", "Str2"),
    8: ("Ran", "Str2"),
}
PRIMARY_CONTRASTS = {
    "Str1_minus_Ran_r12": {
        (1, "Str1"): 1,
        (2, "Str1"): 1,
        (1, "Ran"): -1,
        (2, "Ran"): -1,
    },
    "Switch_r34_vs_r12_did": {
        (3, "Str2"): 1,
        (4, "Str2"): 1,
        (1, "Str1"): -1,
        (2, "Str1"): -1,
        (3, "Ran"): -1,
        (4, "Ran"): -1,
        (1, "Ran"): 1,
        (2, "Ran"): 1,
    },
    "Switch_r56_vs_r12_did": {
        (5, "Str2"): 1,
        (6, "Str2"): 1,
        (1, "Str1"): -1,
        (2, "Str1"): -1,
        (5, "Ran"): -1,
        (6, "Ran"): -1,
        (1, "Ran"): 1,
        (2, "Ran"): 1,
    },
    "Switch_r78_vs_r12_did": {
        (7, "Str2"): 1,
        (8, "Str2"): 1,
        (1, "Str1"): -1,
        (2, "Str1"): -1,
        (7, "Ran"): -1,
        (8, "Ran"): -1,
        (1, "Ran"): 1,
        (2, "Ran"): 1,
    },
}


def grid_signature(img: nib.spatialimages.SpatialImage) -> tuple:
    return (
        img.shape[:3],
        tuple(np.round(img.header.get_zooms()[:3], 5)),
        tuple(np.round(img.affine.ravel(), 5)),
    )


def as_float32_image(
    img: nib.spatialimages.SpatialImage,
    data: np.ndarray | None = None,
) -> nib.Nifti1Image:
    """Return an unscaled float32 NIfTI while preserving spatial metadata."""
    values = (
        np.asarray(img.dataobj, dtype=np.float32)
        if data is None
        else np.asarray(data, dtype=np.float32)
    )
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header["cal_min"] = float(np.nanmin(values))
    header["cal_max"] = float(np.nanmax(values))
    return nib.Nifti1Image(values, img.affine, header=header)


def load_events(path: Path, expected: tuple[str, ...]) -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t")

    def collapse(raw: str) -> str | None:
        for condition in sorted(expected, key=len, reverse=True):
            if condition in raw:
                return condition
        return None

    events["trial_type"] = events["trial_type"].map(collapse)
    events = events[events["trial_type"].isin(expected)]
    return events.loc[:, ["onset", "duration", "trial_type"]].reset_index(drop=True)


def load_confounds(path: Path, n_volumes: int) -> pd.DataFrame:
    full = pd.read_csv(path, sep="\t")
    missing = [name for name in CONFOUND_COLS if name not in full]
    if missing:
        raise RuntimeError(f"{path}: missing confounds {missing}")
    confounds = full.loc[:, CONFOUND_COLS].fillna(0.0).astype(float)
    if len(confounds) != n_volumes:
        raise RuntimeError(
            f"{path}: {len(confounds)} confound rows for {n_volumes} volumes"
        )
    return confounds


def input_paths(subject: str, run: int) -> tuple[Path, Path, Path, Path]:
    rr = f"{run:02d}"
    func = FMRIPREP_DIR / subject / "func"
    bold = (
        func
        / "MNI_space"
        / f"{subject}_task-{TASK_NAME}_run-{rr}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )
    mask = (
        func
        / "Masks"
        / f"{subject}_task-{TASK_NAME}_run-{rr}_space-{SPACE}_desc-brain_mask.nii.gz"
    )
    confounds = (
        func
        / "Confounds"
        / f"{subject}_task-{TASK_NAME}_run-{rr}_desc-confounds_timeseries.tsv"
    )
    events = (
        BIDS_DIR
        / subject
        / "func"
        / f"{subject}_task-{TASK_NAME}_run-{rr}_eventrelated.tsv"
    )
    for path in (bold, mask, confounds, events):
        if not path.is_file():
            raise FileNotFoundError(path)
    return bold, mask, confounds, events


def canonical_reference() -> nib.Nifti1Image:
    """Return the fixed 3-mm master grid used for every subject contrast."""
    _, mask, _, _ = input_paths("sub-001", 1)
    return nib.load(mask)


def fit_run(subject: str, run: int, out_root: Path, overwrite: bool) -> None:
    expected = CONDITIONS[run]
    run_dir = out_root / "level1" / "eventrelated" / subject / f"run-{run:02d}"
    expected_outputs = [run_dir / f"{condition}_effect_size.nii.gz" for condition in expected]
    if not overwrite and all(path.is_file() for path in expected_outputs):
        for path in expected_outputs:
            if nib.load(path).header.get_data_dtype() != np.dtype("float32"):
                raise RuntimeError(f"existing output is not float32: {path}")
        print(f"  skip run-{run:02d}: complete float32 outputs")
        return

    bold, mask, confounds_path, events_path = input_paths(subject, run)
    bold_img = nib.load(bold)
    events = load_events(events_path, expected)
    confounds = load_confounds(confounds_path, bold_img.shape[-1])

    model = FirstLevelModel(
        t_r=TR,
        hrf_model="glover",
        drift_model="cosine",
        high_pass=1.0 / 128.0,
        smoothing_fwhm=4.0,
        signal_scaling=0,
        noise_model="ar1",
        mask_img=str(mask),
        standardize=False,
        minimize_memory=True,
        verbose=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(run_imgs=str(bold), events=events, confounds=confounds)

    run_dir.mkdir(parents=True, exist_ok=True)
    design = model.design_matrices_[0]
    design_values = design.to_numpy(dtype=float)
    if not np.isfinite(design_values).all():
        raise RuntimeError(f"{subject} run-{run:02d}: non-finite design values")
    matrix_rank = int(np.linalg.matrix_rank(design_values))
    n_columns = int(design_values.shape[1])
    if matrix_rank != n_columns:
        raise RuntimeError(
            f"{subject} run-{run:02d}: design rank {matrix_rank}/{n_columns}"
        )
    condition_number = float(np.linalg.cond(design_values))
    if not np.isfinite(condition_number):
        raise RuntimeError(f"{subject} run-{run:02d}: non-finite condition number")
    design.to_csv(run_dir / "design_matrix.tsv", sep="\t", index=False)
    qc = {
        "subject": subject,
        "run": run,
        "event_counts": {
            condition: int((events["trial_type"] == condition).sum())
            for condition in expected
        },
        "n_timepoints": int(design_values.shape[0]),
        "n_columns": n_columns,
        "matrix_rank": matrix_rank,
        "condition_number": condition_number,
        "warnings": [str(item.message) for item in caught],
    }
    (run_dir / "model_qc.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n"
    )
    for condition in expected:
        if condition not in design.columns:
            raise RuntimeError(f"{subject} run-{run:02d}: missing {condition} regressor")
        vector = np.zeros(design.shape[1], dtype=float)
        vector[list(design.columns).index(condition)] = 1.0
        effect = model.compute_contrast(vector, output_type="effect_size")
        output = run_dir / f"{condition}_effect_size.nii.gz"
        as_float32_image(effect).to_filename(output)
        written = nib.load(output)
        if written.header.get_data_dtype() != np.dtype("float32"):
            raise RuntimeError(f"failed to preserve float32 dtype: {output}")


def build_subject_contrasts(subject: str, out_root: Path) -> None:
    level1 = out_root / "level1" / "eventrelated" / subject
    output_dir = out_root / "subject_contrasts" / "eventrelated" / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    for contrast, weights in PRIMARY_CONTRASTS.items():
        entries: list[tuple[float, nib.Nifti1Image]] = []
        for (run, condition), weight in weights.items():
            path = level1 / f"run-{run:02d}" / f"{condition}_effect_size.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append((float(weight), nib.load(path)))

        reference = canonical_reference()
        reference_signature = grid_signature(reference)
        summed = np.zeros(reference.shape[:3], dtype=np.float32)
        for weight, image in entries:
            if grid_signature(image) != reference_signature:
                image = resample_to_img(
                    image,
                    reference,
                    interpolation="continuous",
                    copy_header=True,
                    force_resample=True,
                )
            summed += np.asarray(image.dataobj, dtype=np.float32) * weight

        output = output_dir / f"{contrast}.nii.gz"
        as_float32_image(reference, summed).to_filename(output)
        written = nib.load(output)
        unique = np.unique(np.asarray(written.dataobj))
        if written.header.get_data_dtype() != np.dtype("float32") or len(unique) < 1000:
            raise RuntimeError(
                f"contrast output failed precision check: {output} ({len(unique)} values)"
            )


def write_subject_provenance(subject: str, out_root: Path) -> None:
    payload = {
        "subject": subject,
        "nilearn": nilearn.__version__,
        "nibabel": nib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "model": {
            "tr_seconds": TR,
            "hrf_model": "glover",
            "drift_model": "cosine",
            "high_pass_hz": 1.0 / 128.0,
            "smoothing_fwhm_mm": 4.0,
            "signal_scaling": 0,
            "noise_model": "ar1",
            "standardize": False,
            "confounds": list(CONFOUND_COLS),
            "scrubbing_or_censoring": False,
        },
        "primary_contrasts": {
            name: {
                f"run-{run:02d}_{condition}": weight
                for (run, condition), weight in weights.items()
            }
            for name, weights in PRIMARY_CONTRASTS.items()
        },
        "output_dtype": "float32",
    }
    path = out_root / "provenance" / f"{subject}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=FIRST_LEVEL_DIR / "FLA_nilearn_float32_20260814",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = args.subjects or read_subjects()
    unknown = sorted(set(subjects) - set(read_subjects()))
    if unknown:
        raise SystemExit(f"unknown subjects: {unknown}")

    for subject in subjects:
        print(f"\n=== {subject} ===", flush=True)
        for run in range(1, N_RUNS + 1):
            fit_run(subject, run, args.out_root, args.overwrite)
            print(f"  fit run-{run:02d}", flush=True)
        build_subject_contrasts(subject, args.out_root)
        write_subject_provenance(subject, args.out_root)
        print("  wrote four primary contrasts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
