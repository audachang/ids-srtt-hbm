#!/usr/bin/env python3
"""Bounded native CSV/keyboard-log consistency audit; no trial reassignment."""
from updated_common import *
import re


def main():
    t=Task('response_log_audit',behavior=True)
    source=BEHAV/'source_replay'
    native_files=pd.read_csv(t.record(source/'native_input_files.csv')).path
    log_root=Path('/home/fmri2404/chiaoen/Chiaoen_AllExpData/Exp3/fmri_new_V2/實驗程式/myproject_new_prob_v2/data')
    summaries=[];details=[]
    pattern=re.compile(r'^\s*([0-9.eE+-]+)\s+DATA\s+Keypress:\s*([1-4])\s*$')
    for file in native_files:
        path=t.record(file);native=pd.read_csv(path);native=native[native.keypress.notna()].reset_index(drop=True)
        assert len(native)==320
        subject=f'sub-{int(native.participant.iloc[0]):03d}';run=int(native.session.iloc[0])
        log=log_root/path.with_suffix('.log').name
        if not log.exists():
            summaries.append(dict(subject=subject,run=run,log_available=False,n_native_trials=320,log_path=str(log)))
            continue
        t.record(log);events={key:[] for key in [1,2,3,4]}
        for line in log.read_text(errors='replace').splitlines():
            hit=pattern.match(line)
            if hit:events[int(hit.group(2))].append(float(hit.group(1)))
        events={key:np.asarray(values) for key,values in events.items()}
        rt1=pd.to_numeric(native['Stimuli_key_resp.rt'],errors='coerce')
        rt2=pd.to_numeric(native['Stimuli_key_resp_2.rt'],errors='coerce')
        key1=pd.to_numeric(native['Stimuli_key_resp.keys'],errors='coerce')
        key2=pd.to_numeric(native['Stimuli_key_resp_2.keys'],errors='coerce')
        clock1=pd.to_numeric(native['Stimuli_key_resp.started'],errors='coerce')
        clock2=pd.to_numeric(native['Stimuli_key_resp_2.started'],errors='coerce')
        local=[]
        for i in range(320):
            first=i%2==0;rt=rt1.iloc[i] if first else rt2.iloc[i];key=key1.iloc[i] if first else key2.iloc[i];clock=clock1.iloc[i] if first else clock2.iloc[i]
            if not (np.isfinite(rt) and 0<rt<1 and np.isfinite(key) and key in events and np.isfinite(clock)):continue
            expected=clock+rt;times=events[int(key)];j=np.searchsorted(times,expected)
            candidates=[k for k in [j-1,j] if 0<=k<len(times)]
            chosen=min(candidates,key=lambda k:abs(times[k]-expected)) if candidates else -1
            actual=times[chosen] if chosen>=0 else np.nan
            delta=actual-expected
            local.append(dict(subject=subject,run=run,trial_in_run=i+1,key=int(key),rt=rt,
                              response_clock_start=clock,predicted_key_time=expected,log_key_time=actual,
                              log_minus_predicted_seconds=delta,event_index=chosen,
                              within_50ms=bool(np.isfinite(delta) and abs(delta)<=.05),anticipatory=rt<.1))
        d=pd.DataFrame(local)
        duplicate=d.duplicated(['key','event_index'],keep=False)&d.within_50ms
        d['duplicate_nearest_event']=duplicate
        details.extend(d.to_dict('records'))
        ant=d[d.anticipatory]
        summaries.append(dict(subject=subject,run=run,log_available=True,log_path=str(log),n_native_trials=320,
                              n_valid_responses=len(d),n_within_50ms=int(d.within_50ms.sum()),
                              n_duplicate_event_assignments=int(duplicate.sum()),n_anticipatory=len(ant),
                              n_anticipatory_within_50ms=int(ant.within_50ms.sum()),
                              n_anticipatory_unique_matches=int((ant.within_50ms&~ant.duplicate_nearest_event).sum()),
                              median_log_minus_predicted_seconds=float(d.log_minus_predicted_seconds.median()),
                              max_abs_log_minus_predicted_seconds=float(d.log_minus_predicted_seconds.abs().max())))
    summaries=pd.DataFrame(summaries);details=pd.DataFrame(details)
    csv(summaries,t.path/'run_log_coverage.csv');csv(details,t.path/'response_event_checks.csv')
    csv(details[~details.within_50ms|details.duplicate_nearest_event],t.path/'responses_requiring_review.csv')
    available=summaries[summaries.log_available]
    totals=dict(n_cohort_runs=176,n_runs_with_logs=len(available),n_valid_responses_checked=int(available.n_valid_responses.sum()),
                n_response_matches_within_50ms=int(available.n_within_50ms.sum()),
                n_duplicate_nearest_event_assignments=int(available.n_duplicate_event_assignments.sum()),
                n_anticipatory_checked=int(available.n_anticipatory.sum()),
                n_anticipatory_unique_matches=int(available.n_anticipatory_unique_matches.sum()))
    writej(t.path/'summary.json',totals)
    t.finish('# Native response-log consistency audit\n\nAvailable native PsychoPy logs are matched by exact native CSV filename. For each valid logged 0<RT<1-s response, the response-clock start plus native RT predicts an experiment-clock keypress time. The nearest DATA Keypress event of the same key is checked against a declared 50-ms tolerance; duplicate nearest-event uses are separately flagged. Anticipatory responses are audited with the same rule. Missing logs are explicit. No trial is removed or reassigned and no RT is changed.\n\n'+json.dumps(totals,indent=2)+'\n\nThis tests consistency of the saved CSV and log streams. Log timestamps can reflect polling/emission latency, so a match within 50 ms is not an independent measurement of sub-100-ms latency, physical cue offset or hardware timing. Mismatches/duplicates require inspection rather than automatic correction. Because some logs are absent, global response-to-trial assignment remains only partially audited; cue exposure still cannot be equated with RT.\n',
             dict(passed=True,**totals,all_trial_assignment_verified=False,cue_duration_verified=False))


if __name__=='__main__':main()
