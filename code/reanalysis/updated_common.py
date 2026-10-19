"""Separate output namespace for corrected-timing N20/N22 analyses."""
from common import *
import common as previous

OLD_BASE = previous.BASE
OLD_BEHAV = previous.BEHAV
BASE = DEST / 'results/glm_volume/reanalysis_20261018_updated_n22'
BEHAV = DEST / 'results/behavioral/reanalysis_20261018_updated_n22'
TOP = DEST / 'results/reanalysis_20261018_updated_n22'
N20 = list(SUBJECTS)
N22 = N20 + ['sub-005', 'sub-023']
EVENTS = OLD_BASE / 'T3_timing_repair/events'
CENSORS = OLD_BASE / 'T3_censor_reconstruction/masks'
VARIANTS = ['reference', 'censored', 'spm_derivative', 'errors_omissions', 'rt_modulator', 'rt_shared_modulator']


class Task(previous.Task):
    def __init__(self, name, behavior=False):
        self.path = (BEHAV if behavior else BASE) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.inputs = {}

    def finish(self, report, validation, status='complete'):
        for name in ['T3_timing_repair', 'T3_censor_reconstruction', 'source_discovery']:
            self.record(OLD_BASE / name / 'input_manifest.json')
            self.record(OLD_BASE / name / 'validation.json')
        super().finish(report, validation, status)


def event_path(subject, run):
    return EVENTS / subject / f'{subject}_task-srttprob_run-{run:02d}_eventrelated.tsv'


def mask_path(subject):
    return CENSORS / f'{subject}_afni_v8_rule_reconstructed_censor.1D'


def save32(values, ref, path):
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1., 0.)
    nib.Nifti1Image(np.asarray(values, dtype=np.float32), ref.affine, header).to_filename(path)
    check = nib.load(path)
    assert check.get_data_dtype() == np.dtype('float32') and check.dataobj.slope == 1. and check.dataobj.inter == 0.
    return check
