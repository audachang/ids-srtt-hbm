"""Canonical manuscript paths and read-only resolution of historical provenance.

Saved manifests and executed sources are never rewritten by this module.
"""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from srtt_paths import (PROJECT_ROOT, MANUSCRIPT_DIR, BIDS_DIR, FMRIPREP_DIR,
    GLM_VOLUME_DIR, GLM_FIRST_LEVEL_DIR, REPLICATION_DIR, ROI_MASK_DIR,
    CONFIG_DIR, SUBJECT_LIST, ROI_LIST, SPACE, TASK_NAME, read_subjects, read_rois)

ARCHIVE_ROOT = PROJECT_ROOT / "archive/srtt_self_contained_20260813"
OLD_PACKAGE_ROOT = PROJECT_ROOT / "srtt_self_contained_20260813"
PACKAGE_ROOT = PROJECT_ROOT  # New manifests use project-relative paths.
FIRST_LEVEL_DIR = GLM_FIRST_LEVEL_DIR
FLA_DIR = REPLICATION_DIR / "FLA_afni_out_v8_20250707"


def recorded_path(value):
    """Map an old package-relative/absolute path to its preserved physical file."""
    path = Path(value)
    if path.is_absolute():
        for root in (OLD_PACKAGE_ROOT, ARCHIVE_ROOT):
            try:
                path = path.relative_to(root)
                break
            except ValueError:
                pass
        else:
            return path
    parts = path.parts
    if parts and parts[0] == OLD_PACKAGE_ROOT.name:
        path = Path(*parts[1:])
    text = path.as_posix()
    redirects = [
        ("manuscript_new/scripts", ARCHIVE_ROOT / "manuscript_new/scripts"),
        ("manuscript_new/data", GLM_VOLUME_DIR),
        ("manuscript_new", MANUSCRIPT_DIR),
        ("Nifti/derivatives/fMRIPrep", FMRIPREP_DIR),
        ("Nifti/derivatives/FLA_nilearn_float32_20260814", FIRST_LEVEL_DIR / "FLA_nilearn_float32_20260814"),
        ("Nifti/derivatives/FLA_nilearn_sensitivity_20260921", FIRST_LEVEL_DIR / "FLA_nilearn_sensitivity_20260921"),
        ("Nifti/derivatives/FLA_nilearn_model_alternatives_20260921", FIRST_LEVEL_DIR / "FLA_nilearn_model_alternatives_20260921"),
        ("Nifti/derivatives/FLA_nilearn_fir_20260921", FIRST_LEVEL_DIR / "FLA_nilearn_fir_20260921"),
        ("Nifti/derivatives", REPLICATION_DIR),
        ("Nifti", BIDS_DIR),
        ("Behavioral_data", PROJECT_ROOT / "data/behavioral"),
        ("replication_results", REPLICATION_DIR / "summary"),
        ("analysis", ARCHIVE_ROOT / "analysis"),
        ("provenance", ARCHIVE_ROOT / "provenance"),
    ]
    for prefix, destination in redirects:
        if text == prefix or text.startswith(prefix + "/"):
            return destination / path.relative_to(prefix)
    return PROJECT_ROOT / path


def source_matches(recorded_sha256, filename):
    """Verify the exact current or preserved executed source, not a substituted hash."""
    for path in (Path(__file__).parent / filename,
                 ARCHIVE_ROOT / "manuscript_new/scripts" / filename):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == recorded_sha256:
            return True
    return False


def fit_source_matches(recorded_sha256, filename):
    """Allow archived fit reuse only for the checksum-pinned layout-only migration.

Any later edit to the active numerical scripts, layout or shared paths removes
this historical exception; the runner then requires a fresh fit/version.
"""
    here = Path(__file__).parent
    current = here / filename
    if hashlib.sha256(current.read_bytes()).hexdigest() == recorded_sha256:
        return True
    manifest = json.loads((here / "migration_source_hashes.json").read_text())
    for relative, expected in manifest["adapted_sources"].items():
        if hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() != expected:
            return False
    original = ARCHIVE_ROOT / "manuscript_new/scripts" / filename
    return hashlib.sha256(original.read_bytes()).hexdigest() == recorded_sha256
