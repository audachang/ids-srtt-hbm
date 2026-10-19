#!/usr/bin/env python3
"""Fit prespecified Nilearn motion-censoring/filtering sensitivity models.

This script keeps the manuscript's run-wise Nilearn model, Glover HRF,
4-mm smoothing, AR(1), signal scaling, eight basic nuisance regressors, and
four subject contrasts fixed.  It changes only two factors:

1. whether the exact saved AFNI v8 motion/outlier censor mask is applied; and
2. whether temporal filtering uses the manuscript's 128-s high-pass or the
   thesis AFNI model's 0.0078--0.1-Hz band-pass nuisance basis.

The three non-primary cells are named ``censor_hp128``,
``nocensor_bandpass``, and ``censor_bandpass``.  The existing validated
``nocensor_hp128`` primary model is not overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy.linalg
from nilearn.glm.first_level import FirstLevelModel


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, FIRST_LEVEL_DIR, ROI_MASK_DIR, recorded_path, source_matches
PRIMARY_SCRIPT = Path(__file__).with_name("90_run_float32_eventrelated.py")
DEFAULT_OUT_BASE = (
    FIRST_LEVEL_DIR / "FLA_nilearn_sensitivity_20260921"
)
DEFAULT_CENSOR_DIR = MANUSCRIPT_DIR / "data" / "sensitivity_inputs" / "afni_v8_censor"


def load_primary_module():
    spec = importlib.util.spec_from_file_location("primary_float32", PRIMARY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PRIMARY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


primary = load_primary_module()

TR = primary.TR
N_RUNS = primary.N_RUNS
CONFOUND_COLS = primary.CONFOUND_COLS
CONDITIONS = primary.CONDITIONS
PRIMARY_CONTRASTS = primary.PRIMARY_CONTRASTS

LOW_HZ = 0.0078
HIGH_HZ = 0.1
MIN_RESIDUAL_DF = 10
EXPECTED_BANDPASS_COLUMNS = 123
EXPECTED_NILEARN_VERSION = "0.13.1"

VARIANTS = {
    "censor_hp128": {"censor": True, "filter": "hp128"},
    "nocensor_bandpass": {"censor": False, "filter": "bandpass"},
    "censor_bandpass": {"censor": True, "filter": "bandpass"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bandpass_stop_basis(n_volumes: int) -> pd.DataFrame:
    """Return the AFNI 1dBport-equivalent stop-band Fourier nuisance basis.

    AFNI selects a Fourier frequency when its half-bin interval lies wholly
    outside the retained band.  At 200 volumes and TR=2 s this selects bins
    1, 2, and 41--100, producing 123 columns, exactly as in the saved v8
    model.  Column signs do not affect the nuisance subspace.
    """
    duration = n_volumes * TR
    half_bin_hz = 0.5 / duration
    timepoints = np.arange(n_volumes, dtype=float)
    columns: dict[str, np.ndarray] = {}
    selected_bins: list[int] = []
    for index in range(1, n_volumes // 2 + 1):
        frequency = index / duration
        outside = (
            frequency + half_bin_hz < LOW_HZ
            or frequency - half_bin_hz > HIGH_HZ
        )
        if not outside:
            continue
        selected_bins.append(index)
        angle = 2.0 * np.pi * index * timepoints / n_volumes
        columns[f"stop_cos_{index:03d}"] = np.cos(angle)
        if index < n_volumes / 2:
            columns[f"stop_sin_{index:03d}"] = np.sin(angle)
    basis = pd.DataFrame(columns)
    if n_volumes == 200:
        expected_bins = [1, 2, *range(41, 101)]
        if selected_bins != expected_bins or basis.shape[1] != EXPECTED_BANDPASS_COLUMNS:
            raise RuntimeError(
                f"unexpected band-pass basis: bins={selected_bins}, shape={basis.shape}"
            )
    return basis


def load_censor_mask(censor_dir: Path, subject: str, run: int, n_volumes: int) -> np.ndarray:
    path = censor_dir / f"{subject}_afni_v8_censor.1D"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, dtype=float).reshape(-1)
    expected = N_RUNS * n_volumes
    if len(values) != expected or not np.isin(values, (0.0, 1.0)).all():
        raise RuntimeError(f"{path}: expected {expected} binary values")
    start = (run - 1) * n_volumes
    keep = np.flatnonzero(values[start : start + n_volumes] == 1.0)
    if len(keep) == 0:
        raise RuntimeError(f"{subject} run-{run:02d}: censor mask removes every volume")
    return keep


def load_model_inputs(
    subject: str,
    run: int,
    filter_name: str,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame, int]:
    bold, mask, confounds_path, events_path = primary.input_paths(subject, run)
    bold_img = nib.load(bold)
    n_volumes = int(bold_img.shape[-1])
    events = primary.load_events(events_path, CONDITIONS[run])
    confounds = primary.load_confounds(confounds_path, n_volumes)
    if filter_name == "bandpass":
        stop_basis = bandpass_stop_basis(n_volumes)
        linear = np.linspace(-1.0, 1.0, n_volumes)
        confounds = pd.concat(
            [
                confounds.reset_index(drop=True),
                pd.DataFrame({"linear_drift": linear}),
                stop_basis,
            ],
            axis=1,
        )
    return bold, mask, events, confounds, n_volumes


def fit_run(
    subject: str,
    run: int,
    variant: str,
    out_root: Path,
    censor_dir: Path,
    overwrite: bool,
) -> None:
    settings = VARIANTS[variant]
    expected = CONDITIONS[run]
    run_dir = out_root / "level1" / "eventrelated" / subject / f"run-{run:02d}"
    outputs = [run_dir / f"{condition}_effect_size.nii.gz" for condition in expected]
    qc_path = run_dir / "model_qc.json"
    if not overwrite and qc_path.is_file() and all(path.is_file() for path in outputs):
        print(f"  {variant} run-{run:02d}: skip complete outputs", flush=True)
        return

    bold, mask, events, confounds, n_volumes = load_model_inputs(
        subject, run, settings["filter"]
    )
    sample_mask = (
        load_censor_mask(censor_dir, subject, run, n_volumes)
        if settings["censor"]
        else None
    )
    n_retained = n_volumes if sample_mask is None else int(len(sample_mask))

    bandpass = settings["filter"] == "bandpass"

    def make_model() -> FirstLevelModel:
        return FirstLevelModel(
            t_r=TR,
            hrf_model="glover",
            drift_model=None if bandpass else "cosine",
            high_pass=None if bandpass else 1.0 / 128.0,
            smoothing_fwhm=4.0,
            signal_scaling=0,
            noise_model="ar1",
            mask_img=str(mask),
            standardize=False,
            minimize_memory=True,
            verbose=0,
        )

    def fit(current_confounds: pd.DataFrame) -> tuple[FirstLevelModel, list[str]]:
        fitted = make_model()
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            fitted.fit(
                run_imgs=str(bold),
                events=events,
                confounds=current_confounds,
                sample_masks=sample_mask,
            )
        return fitted, [str(item.message) for item in fit_warnings]

    model, caught = fit(confounds)

    run_dir.mkdir(parents=True, exist_ok=True)
    design = model.design_matrices_[0]
    design_values = design.to_numpy(dtype=float)
    if design_values.shape[0] != n_retained:
        raise RuntimeError(
            f"{subject} {variant} run-{run:02d}: design rows "
            f"{design_values.shape[0]} != retained volumes {n_retained}"
        )
    if not np.isfinite(design_values).all():
        raise RuntimeError(f"{subject} {variant} run-{run:02d}: non-finite design")
    rank = int(np.linalg.matrix_rank(design_values))
    n_columns = int(design_values.shape[1])
    dropped_redundant: list[str] = []
    if rank != n_columns:
        _, _, pivot = scipy.linalg.qr(design_values, mode="economic", pivoting=True)
        dropped_redundant = [str(design.columns[i]) for i in pivot[rank:]]
        if not dropped_redundant or not set(dropped_redundant) <= set(confounds.columns):
            raise RuntimeError(
                f"{subject} {variant} run-{run:02d}: design rank {rank}/{n_columns}; "
                f"cannot safely remove redundant columns {dropped_redundant}"
            )
        confounds = confounds.drop(columns=dropped_redundant)
        model, refit_warnings = fit(confounds)
        caught.extend(refit_warnings)
        design = model.design_matrices_[0]
        design_values = design.to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(design_values))
        n_columns = int(design_values.shape[1])
        if rank != n_columns:
            raise RuntimeError(
                f"{subject} {variant} run-{run:02d}: refit rank {rank}/{n_columns}"
            )
    residual_df = n_retained - rank
    if residual_df < MIN_RESIDUAL_DF:
        raise RuntimeError(
            f"{subject} {variant} run-{run:02d}: only {residual_df} residual df"
        )
    condition_number = float(np.linalg.cond(design_values))
    if not np.isfinite(condition_number):
        raise RuntimeError(f"{subject} {variant} run-{run:02d}: invalid condition number")

    design.to_csv(run_dir / "design_matrix.tsv", sep="\t", index=False)
    if sample_mask is not None:
        np.savetxt(run_dir / "retained_volume_indices.1D", sample_mask, fmt="%d")
    qc = {
        "subject": subject,
        "run": run,
        "variant": variant,
        "censoring": settings["censor"],
        "filter": settings["filter"],
        "n_timepoints_original": n_volumes,
        "n_timepoints_retained": n_retained,
        "n_timepoints_censored": n_volumes - n_retained,
        "retained_fraction": n_retained / n_volumes,
        "n_columns": n_columns,
        "matrix_rank": rank,
        "residual_df": residual_df,
        "condition_number": condition_number,
        "event_counts": {
            condition: int((events["trial_type"] == condition).sum())
            for condition in expected
        },
        "bandpass_stop_columns": EXPECTED_BANDPASS_COLUMNS if bandpass else 0,
        "dropped_redundant_nuisance_columns": dropped_redundant,
        "warnings": caught,
    }
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")

    for condition in expected:
        if condition not in design.columns:
            raise RuntimeError(f"{subject} run-{run:02d}: missing {condition}")
        vector = np.zeros(n_columns, dtype=float)
        vector[list(design.columns).index(condition)] = 1.0
        effect = model.compute_contrast(vector, output_type="effect_size")
        output = run_dir / f"{condition}_effect_size.nii.gz"
        primary.as_float32_image(effect).to_filename(output)
        if nib.load(output).header.get_data_dtype() != np.dtype("float32"):
            raise RuntimeError(f"output is not float32: {output}")


def write_subject_provenance(subject: str, variant: str, out_root: Path) -> None:
    settings = VARIANTS[variant]
    payload = {
        "created_utc": utc_now(),
        "subject": subject,
        "variant": variant,
        "software": {
            "python": platform.python_version(),
            "nilearn": nilearn.__version__,
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fixed_model": {
            "tr_seconds": TR,
            "hrf_model": "glover",
            "smoothing_fwhm_mm": 4.0,
            "signal_scaling": 0,
            "noise_model": "ar1",
            "standardize": False,
            "confounds": list(CONFOUND_COLS),
            "runwise_models": True,
        },
        "sensitivity_factors": {
            "censoring": (
                "saved AFNI v8 combined motion 0.3/outlier 0.05 mask"
                if settings["censor"]
                else "none"
            ),
            "filtering": (
                "0.0078-0.1 Hz AFNI-equivalent Fourier stop-band nuisance basis; linear drift"
                if settings["filter"] == "bandpass"
                else "cosine drift, 128-second high-pass"
            ),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--censor-dir", type=Path, default=DEFAULT_CENSOR_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if nilearn.__version__ != EXPECTED_NILEARN_VERSION:
        raise RuntimeError(
            f"requires Nilearn {EXPECTED_NILEARN_VERSION}; found {nilearn.__version__}"
        )
    all_subjects = primary.read_subjects()
    subjects = args.subjects or all_subjects
    unknown = sorted(set(subjects) - set(all_subjects))
    if unknown:
        raise SystemExit(f"unknown subjects: {unknown}")
    variants = list(dict.fromkeys(args.variants))

    for variant in variants:
        out_root = args.out_base.resolve() / variant
        print(f"\n##### {variant} #####", flush=True)
        for subject in subjects:
            print(f"\n=== {subject} ===", flush=True)
            for run in range(1, N_RUNS + 1):
                fit_run(
                    subject,
                    run,
                    variant,
                    out_root,
                    args.censor_dir.resolve(),
                    args.overwrite,
                )
                print(f"  fit run-{run:02d}", flush=True)
            primary.build_subject_contrasts(subject, out_root)
            write_subject_provenance(subject, variant, out_root)
            print("  wrote four subject contrasts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
