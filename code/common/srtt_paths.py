"""Canonical locations in the IDS-SRTT project tree (role-based layout, 2026-10-03).

Analysis code under ``code/`` imports its inputs and output roots from here:

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from srtt_paths import BIDS_DIR, FMRIPREP_DIR, ...

Set ``SRTT_ROOT`` to point the same code at another copy of the tree. Run this
file directly to list every location and whether it exists.

The manuscript scripts in ``code/glm_volume`` use these canonical paths via
``code/glm_volume/_layout.py``. That adapter also resolves historical manifest
paths and preserves verification of archived executed sources.
``code/replication`` keeps its environment-driven ``config/paths.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("SRTT_ROOT", Path(__file__).resolve().parents[2]))

# Inputs
DATA_DIR = PROJECT_ROOT / "data"
BIDS_DIR = DATA_DIR / "bids"
BEHAV_DIR = DATA_DIR / "behavioral"
DERIV_DIR = DATA_DIR / "derivatives"
FMRIPREP_DIR = DERIV_DIR / "fmriprep"
FREESURFER_DIR = DERIV_DIR / "freesurfer"

# Atlases, masks and templates
RESOURCES_DIR = PROJECT_ROOT / "resources"
ROI_MASK_DIR = RESOURCES_DIR / "roi_masks"
ATLAS_DIR = RESOURCES_DIR / "atlases"
FSLR32K_DIR = RESOURCES_DIR / "fs_LR_32"

# Outputs, one root per analysis
RESULTS_DIR = PROJECT_ROOT / "results"
GLM_VOLUME_DIR = RESULTS_DIR / "glm_volume"
GLM_FIRST_LEVEL_DIR = GLM_VOLUME_DIR / "first_level"
REPLICATION_DIR = RESULTS_DIR / "replication"
FLA_DIR = REPLICATION_DIR / "FLA_afni_out_v8_20250707"
SLA_DIR = REPLICATION_DIR / "SLA_v8_20260721"
SURFACE_DIR = RESULTS_DIR / "surface"
DFC_DIR = RESULTS_DIR / "dfc"

# Code, configuration and software
CODE_DIR = PROJECT_ROOT / "code"
CONFIG_DIR = CODE_DIR / "replication" / "config"
SUBJECT_LIST = CONFIG_DIR / "subjects.txt"
ROI_LIST = CONFIG_DIR / "rois.txt"
SECOND_LEVEL_DESIGN = GLM_VOLUME_DIR / "whole_brain" / "second_level_design.tsv"
ENVS_DIR = PROJECT_ROOT / "envs"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"

TASK_NAME = "srttprob"
SPACE = "MNI152NLin2009cAsym"
N_RUNS = 8
TR = 2.0


def read_subjects() -> list[str]:
    return [s.strip() for s in SUBJECT_LIST.read_text().splitlines() if s.strip()]


def read_rois() -> list[str]:
    return [s.strip() for s in ROI_LIST.read_text().splitlines() if s.strip()]


if __name__ == "__main__":
    missing = 0
    for name, value in sorted(globals().items()):
        if isinstance(value, Path):
            exists = value.exists()
            missing += not exists
            print(f"{'ok     ' if exists else 'MISSING'} {name:22s} {value}")
    raise SystemExit(1 if missing else 0)
