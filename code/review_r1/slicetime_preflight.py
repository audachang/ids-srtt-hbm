"""A7 required metadata gate and read-only preservation baseline."""
from slicetime_common import *


def main():
    assert nilearn.__version__=='0.13.1'
    assert not OUT.exists(), 'A7 result directory must be new at initial preflight'
    t=ImagingTask('metadata');rows=[]
    for subject in N22:
        for run in range(1,9):
            bold=paths(subject,run)[0];sidecar=bold.with_name(bold.name.replace('.nii.gz','.json'))
            q=json.loads(t.record(sidecar).read_text())
            start=float(q['StartTime']);tr=float(q['RepetitionTime'])
            assert np.isfinite(start) and np.isfinite(tr) and 0<=start<tr and tr==2.
            assert nib.load(bold).shape[3]==200
            rows.append(dict(subject=subject,run=run,StartTime=start,RepetitionTime=tr,slice_time_ref=start/tr,bold=str(bold),sidecar=str(sidecar),sha256=digest(sidecar)))
    csv(rows,t.path/'participant_run_slice_timing.csv')
    for path in ['AGENTS.md','docs/MANUSCRIPT_STATUS.md','docs/REVISION_LOG.md','docs/review_round1/briefs/R1_COMMON.md','docs/review_round1/briefs/r2-slicetime.md','docs/review_round1/reviewer-1.md','docs/review_round1/reviewer-2.md','docs/review_round1/reports/r1-imaging.md','docs/review_round1/reports/r1-audit.md']:
        t.record(ROOT/path)
    t.finish('# A7 timing metadata gate\n\nAll 176 included BOLD JSON files have finite StartTime and RepetitionTime; TR=2 s and 200 volumes. Values and hashes recorded per run. The result directory did not exist before this preflight. No missing value was substituted.',dict(passed=True,n_runs=176,min_StartTime=min(r['StartTime'] for r in rows),max_StartTime=max(r['StartTime'] for r in rows),result_directory_absent_before_start=True))
    preservation_baseline()


def preservation_baseline():
    protected={}
    parents=['code','docs','manuscript','resources','results/glm_volume/reanalysis_20261018_updated_n22','results/glm_volume/review_r1_20261019','results/glm_volume/review_r1_20261019_joint_family','results/reanalysis_20261018_updated_n22','results/behavioral/reanalysis_20261018_updated_n22','results/behavioral/review_r1_20261019']
    for parent in parents:
        for base,dirs,files in os.walk(ROOT/parent,followlinks=False):
            dirs[:]=[d for d in dirs if d!='__pycache__']
            for name in files:
                path=Path(base)/name
                if not path.is_file() or path.is_symlink() or path.name.startswith('slicetime_') or path==ROOT/'docs/review_round1/reports/r2-slicetime.md':continue
                protected[str(path)]=dict(sha256=digest(path),bytes=path.stat().st_size)
    writej('/tmp/r2-r2-slicetime/protected_before.json',protected)
    print('Protected baseline',len(protected),flush=True)

if __name__=='__main__':
    if '--baseline-only' in sys.argv:preservation_baseline()
    else:main()
