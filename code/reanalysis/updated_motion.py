#!/usr/bin/env python3
from updated_common import *


def main():
    t=Task('motion_quality');rows=[]
    censor_runs=pd.read_csv(t.record(OLD_BASE/'T3_censor_reconstruction/run_censor_counts.csv'))
    for subject in N22:
        masks=np.loadtxt(t.record(mask_path(subject))).reshape(8,200)
        for run in range(1,9):
            f=pd.read_csv(t.record(paths(subject,run)[2]),sep='\t');fd=f.framewise_displacement.to_numpy();assert len(fd)==200
            row=dict(subject=subject,run=run,mean_fd=np.nanmean(fd),median_fd=np.nanmedian(fd),max_fd=np.nanmax(fd),
                n_fd_defined=int(np.isfinite(fd).sum()),n_fd_gt02=int((fd>.2).sum()),n_fd_gt05=int((fd>.5).sum()),
                fraction_fd_gt05=float((fd>.5).sum()/np.isfinite(fd).sum()),n_censored=int((masks[run-1]==0).sum()),censored_fraction=float((masks[run-1]==0).mean()))
            for col in ['std_dvars','rmsd']:
                if col in f:row['mean_'+col]=f[col].mean();row['max_'+col]=f[col].max()
            for prefix in ['motion_outlier','non_steady_state_outlier']:
                cols=[c for c in f if c.startswith(prefix)];row['n_'+prefix+'_frames']=int(f[cols].fillna(0).any(axis=1).sum()) if cols else 0
            rows.append(row)
    motion=pd.DataFrame(rows);csv(motion,t.path/'run_motion_quality.csv');summary=[]
    for cohort,subjects in [('N20',N20),('N22',N22)]:
        d=motion[motion.subject.isin(subjects)]
        summary.append(dict(cohort=cohort,n_runs=len(d),median_run_mean_fd=d.mean_fd.median(),max_run_mean_fd=d.mean_fd.max(),
            n_run_mean_fd_gt02=int((d.mean_fd>.2).sum()),n_run_mean_fd_gt05=int((d.mean_fd>.5).sum()),
            n_runs_gt20pct_fd_gt05=int((d.fraction_fd_gt05>.2).sum()),n_censored=int(d.n_censored.sum()),n_total_frames=len(d)*200,max_run_censored_fraction=d.censored_fraction.max()))
    csv(summary,t.path/'cohort_motion_summary.csv')
    results=pd.read_csv(t.record(BASE/'roi_reporting/stage_results.csv'))
    ref=results[results.model=='reference'].drop(columns='model');cens=results[results.model=='censored'].drop(columns='model')
    comparison=ref.merge(cens,on=['cohort','policy','stage','roi'],suffixes=('_reference','_censored'),validate='one_to_one')
    csv(comparison,t.path/'reference_vs_censored.csv')
    report='# Updated motion and censoring audit\n\nFD and fMRIPrep outlier flags are descriptive quality screens, not new exclusions. The AFNI censor rule combines original motion-enorm and BOLD-outlier thresholds and is distinct from fMRIPrep FD. Reconstructed masks reproduced all 32,000 historical N20 entries; added-participant counts and exact reconstruction controls are documented in T3_censor_reconstruction.\n\n'
    for s in summary:report+=f'- {s["cohort"]}: median run-mean FD {s["median_run_mean_fd"]:.4f}mm, maximum {s["max_run_mean_fd"]:.4f}mm; {s["n_run_mean_fd_gt02"]}/{s["n_runs"]} runs have mean FD>.2mm, {s["n_run_mean_fd_gt05"]} have mean FD>.5mm; {s["n_runs_gt20pct_fd_gt05"]} have >20% of defined FD values>.5mm. Censoring removes {s["n_censored"]}/{s["n_total_frames"]} frames, maximum run fraction {s["max_run_censored_fraction"]:.3f}.\n'
    report+='\nThe uncensored model is the transparently revised reporting reference under the plan. These quality measures and the matched censored estimates assess sensitivity; they do not establish that retaining all volumes is unbiased. The complete reference/censored effect, CI and q table is exported.\n'
    t.finish(report,dict(passed=True,n_runs=len(motion),n_comparison_rows=len(comparison),summaries=summary))


if __name__=='__main__':main()
