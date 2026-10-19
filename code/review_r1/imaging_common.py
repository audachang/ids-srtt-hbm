"""R1 imaging namespace and provenance; historical helpers are read-only."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reanalysis'))
from updated_common import *
import platform
import time

EXISTING = BASE
OUT = ROOT / 'results/glm_volume/review_r1_20261019'
NEW_MODELS = ['motion12_tissue', 'motion6_acompcor8', 'motion12_acompcor8', 'censored_spike', 'repeat_regressor']
MOTION = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
HEADLINE = ['LeftSMA','RightSMA','RightPallidum','RightThalamusProper']
PLAN = ROOT / 'docs/REVIEW_R1_ANALYSIS_PLAN_2026-10-19.md'


class ImagingTask:
    def __init__(self, name):
        self.path = OUT / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.inputs = {}
        self.started = time.monotonic()

    def record(self, path):
        path = Path(path)
        if str(path) not in self.inputs:
            self.inputs[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
        return path

    def finish(self, report, validation):
        for path in [PLAN, ROOT/'code/replication/config/subjects.txt', ROOT/'code/replication/config/rois.txt']:
            self.record(path)
        sources = list((ROOT/'code/review_r1').glob('imaging_*.py'))
        sources += [ROOT/'code/reanalysis'/n for n in ['common.py','updated_common.py','updated_first_level.py',
                    'updated_targeted_sensitivity.py','run_saved.py']]
        writej(self.path/'input_manifest.json', dict(inputs=self.inputs,
               sources={str(p):digest(p) for p in sources}, argv=sys.argv,
               software=dict(python=platform.python_version(),nilearn=nilearn.__version__,numpy=np.__version__,
                             pandas=pd.__version__,scipy=scipy.__version__,nibabel=nib.__version__),
               thread_limits=dict(OPENBLAS_NUM_THREADS=os.environ['OPENBLAS_NUM_THREADS'],OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'])))
        (self.path/'REPORT.md').write_text(report+'\n')
        outputs={str(p.relative_to(self.path)):digest(p) for p in self.path.rglob('*') if p.is_file() and p.name!='validation.json'}
        writej(self.path/'validation.json', dict(status='complete',elapsed_seconds=time.monotonic()-self.started,
                                               **validation,output_sha256=outputs))
        print(str(self.path), 'complete', flush=True)


def repeats(raw):
    """D1: consecutive trial slots in the same run, exact 1-second onset gap."""
    return raw.keypress.eq(raw.keypress.shift()) & raw.stim_onset.diff().eq(1.)
