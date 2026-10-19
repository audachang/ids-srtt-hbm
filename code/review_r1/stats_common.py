"""Owned IO for the fixed review-R1 statistics analyses; no historical writes."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/reanalysis'))
from common import bh, summarize, digest, writej
import json, platform, time
import numpy as np
import pandas as pd
import scipy, nilearn
assert nilearn.__version__ == '0.13.1'

BEHAV = ROOT / 'results/behavioral/review_r1_20261019'
JOINT = ROOT / 'results/glm_volume/review_r1_20261019_joint_family'
OLD_B = ROOT / 'results/behavioral/reanalysis_20261018_updated_n22'
OLD_I = ROOT / 'results/glm_volume/reanalysis_20261018_updated_n22'
SEED, NBOOT = 20261019, 10000
STAGES = ['initial', 'early', 'middle', 'late']
KEYS = ['run2_Str1_minus_Ran', 'run8_Str2_minus_Ran',
        'early_adjusted_change', 'early_minus_late']

def save(d, path):
    pd.DataFrame(d).to_csv(path, index=False, float_format='%.16g')

class Task:
    def __init__(self, path):
        self.path = Path(path)
        assert self.path == JOINT or self.path == BEHAV or BEHAV in self.path.parents
        self.path.mkdir(parents=True, exist_ok=True)
        self.inputs = {}
        self.started = time.perf_counter()
        for p in ['AGENTS.md', 'docs/MANUSCRIPT_STATUS.md', 'docs/REVISION_LOG.md',
                  'docs/REVIEW_R1_ANALYSIS_PLAN_2026-10-19.md',
                  'docs/review_round1/briefs/R1_COMMON.md',
                  'docs/review_round1/briefs/r1-stats.md',
                  'docs/review_round1/reviewer-1.md', 'docs/review_round1/reviewer-2.md',
                  'code/reanalysis/common.py', 'code/replication/config/subjects.txt',
                  'code/replication/config/rois.txt']:
            self.record(ROOT / p)
        for p in sorted((ROOT / 'code/review_r1').glob('stats_*')):
            if p.is_file(): self.record(p)

    def record(self, path):
        p = Path(path).absolute()
        if str(p) not in self.inputs:
            self.inputs[str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        return p

    def read(self, path, **kwargs):
        return pd.read_csv(self.record(path), **kwargs)

    def finish(self, report, checks):
        # Every source consumed by this task must still have its recorded bytes.
        changed = [p for p, m in self.inputs.items() if digest(p) != m['sha256']]
        assert not changed, changed
        manifest = dict(inputs=self.inputs, argv=sys.argv, seed=SEED,
                        software=dict(python=platform.python_version(), numpy=np.__version__,
                                      pandas=pd.__version__, scipy=scipy.__version__,
                                      nilearn=nilearn.__version__), workers=1)
        writej(self.path / 'input_manifest.json', manifest)
        (self.path / 'REPORT.md').write_text(report + '\n')
        writej(self.path / 'validation.json', dict(status='complete', passed=True,
               elapsed_seconds=time.perf_counter()-self.started,
               checks=checks, consumed_inputs_unchanged=True,
               output_sha256={str(p.relative_to(self.path)): digest(p)
                   for p in sorted(self.path.rglob('*')) if p.is_file()
                   and p.name not in ['validation.json', 'input_manifest.json']}))
        print(self.path.relative_to(ROOT), 'PASS', flush=True)

def boot_draws():
    return np.random.default_rng(SEED).integers(0, 22, size=(NBOOT, 22))

def phase(run):
    return STAGES[(int(run)-1)//2]
