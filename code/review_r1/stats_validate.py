#!/usr/bin/env python3
"""Independent saved-output checks for r1-stats, with a complete owned-file handoff."""
from stats_common import *
from scipy import stats
from datetime import datetime


def assertclose(a,b):
    np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-12,equal_nan=True)


def manual_bh(p):
    p=np.where(np.isfinite(p),p,1.)
    order=np.argsort(p,kind='stable')
    adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.clip(adjusted,0,1)
    return out


def check_joint():
    d=pd.read_csv(JOINT/'joint_stage_results.csv')
    for _,g in d.groupby(['cohort','model','policy','stage']):
        assert len(g)==48
        assertclose(manual_bh(g.p.to_numpy()),g.q_stage_recomputed)
    for _,g in d.groupby(['cohort','model','policy']):
        assert len(g)==192
        assertclose(manual_bh(g.p.to_numpy()),g.q_joint_192)
    assert d.stage_discovery.eq(d.q_stage_recomputed.le(.05)).all()
    assert d.joint_discovery.eq(d.q_joint_192.le(.05)).all()
    return dict(passed=True,n_slots=len(d),manual_BH_stage_families=104,manual_BH_joint_families=26)


def independent_repeats(raw):
    result=[]
    for _,g in raw.groupby(['subject','session'],sort=True):
        g=g.sort_values('trial_in_run')
        repeat=np.zeros(len(g),dtype=bool)
        repeat[1:]=(g.keypress.to_numpy()[1:]==g.keypress.to_numpy()[:-1]) & (np.diff(g.stim_onset.to_numpy())==1)
        q=g[['subject','session','trial_in_run']].copy();q['independent_repeat']=repeat;result.append(q)
    return pd.concat(result)


def check_rt(path,clean,repeat_flags,subset):
    errors=[]
    for policy in ['primary','min100']:
        d=clean.copy()
        if policy=='min100':d=d[d.p_response_rt.ge(.1)]
        if subset:
            d=d.merge(repeat_flags,on=['subject','session','trial_in_run'],validate='one_to_one')
            d=d[~d.independent_repeat]
        d=d.copy();d['log_rt']=np.log(1000*d.p_response_rt)
        d['cell']=[f'cellr{int(run):02d}_{cond}' for run,cond in zip(d.session,d.rule_group)]
        matrix=d.groupby(['subject','cell']).log_rt.mean().unstack('cell').sort_index()
        w=pd.read_csv(OLD_B/f'rt_lme_N22/{policy}_contrast_weights.csv').set_index('contrast')
        if subset:w=w.loc[KEYS]
        values=matrix[w.columns].to_numpy()@w.to_numpy().T
        saved=pd.read_csv(path/f'participant_{policy}_contrasts.csv').pivot(index='subject',columns='contrast',values='log_difference')
        assertclose(values,saved.reindex(index=matrix.index,columns=w.index))
        assertclose(values.mean(0),pd.read_csv(path/f'{policy}_contrast_comparison.csv').set_index('contrast').loc[w.index].log_difference)
        draws=np.random.default_rng(20261019).integers(0,22,(10000,22))
        assert np.array_equal(draws,np.load(path/f'participant_{policy}_bootstrap_indices.npy'))
        bootstrap=values[draws].mean(1)
        result=pd.read_csv(path/f'{policy}_contrast_comparison.csv').set_index('contrast').loc[w.index]
        tresult=stats.ttest_1samp(values,0,axis=0)
        assertclose(tresult.statistic,result.t);assertclose(tresult.pvalue,result.p_two_sided)
        half=stats.t.ppf(.975,21)*values.std(0,ddof=1)/np.sqrt(22)
        assertclose(values.mean(0)-half,result.log_ci_low);assertclose(values.mean(0)+half,result.log_ci_high)
        quantiles=np.percentile(bootstrap,[2.5,97.5],axis=0,method='linear')
        assertclose(quantiles[0],result.bootstrap_log_ci_low);assertclose(quantiles[1],result.bootstrap_log_ci_high)
        assertclose(np.exp(result.log_difference),result.geometric_rt_ratio)
        assertclose(np.exp(result.log_ci_low),result.ratio_ci_low)
        assertclose(np.exp(result.log_ci_high),result.ratio_ci_high)
        assertclose(np.exp(quantiles[0]),result.bootstrap_ratio_ci_low)
        assertclose(np.exp(quantiles[1]),result.bootstrap_ratio_ci_high)
        family=pd.read_csv(path/f'{policy}_BH_family_slots.csv').set_index('contrast')
        assert len(family)==24
        assertclose(manual_bh(family.p_for_BH.to_numpy()),family.q_bh_24)
        assertclose(family.loc[w.index].q_bh_24,result.q_bh_24)
        if subset:assert family[~family.tested].p_for_BH.eq(1).all()
        lme=pd.read_csv(OLD_B/f'rt_lme_N22/{policy}_contrasts.csv').set_index('contrast').loc[w.index]
        assertclose((result.log_ci_high-result.log_ci_low)/(lme.log_ci_high-lme.log_ci_low),result.t_log_CI_width_over_lme)
        assertclose((quantiles[1]-quantiles[0])/(lme.log_ci_high-lme.log_ci_low),result.bootstrap_log_CI_width_over_lme)
        assertclose((result.ratio_ci_high-result.ratio_ci_low)/(lme.ratio_ci_high-lme.ratio_ci_low),result.t_ratio_CI_width_over_lme)
        assertclose((np.exp(quantiles[1])-np.exp(quantiles[0]))/(lme.ratio_ci_high-lme.ratio_ci_low),result.bootstrap_ratio_CI_width_over_lme)
        errors.append(float(np.max(abs(values-saved.reindex(index=matrix.index,columns=w.index).to_numpy()))))
    return dict(passed=True,n_policies=2,n_contrasts_per_policy=4 if subset else 24,
        participant_values_recomputed_from_clean_trials=True,whole_participant_bootstrap_reproduced=True,
        all_t_tests_CIs_ratios_width_ratios_and_BH_reproduced=True,seed=20261019,resamples=10000,
        max_participant_contrast_difference=max(errors))


def check_design(raw,flags):
    path=BEHAV/'B3_design_statistics'
    a=pd.read_csv(BEHAV/'B2_repetition_behavior/participant_trial_annotations.csv')
    q=a.merge(flags,on=['subject','session','trial_in_run'],validate='one_to_one')
    assert q.immediate_repeat_D1.eq(q.independent_repeat).all()
    r=pd.read_csv(path/'D1_repeat_rates_by_condition_run.csv')
    b=pd.read_csv(BEHAV/'B2_repetition_behavior/D1_repeat_counts_by_condition_run.csv')
    assertclose(r.n_immediate_repeats_D1,b.immediate_repeats_D1)
    targets=pd.read_csv(path/'target_frequencies_by_condition_run.csv')
    transitions=pd.read_csv(path/'transition_frequencies_by_condition_run.csv')
    for row in targets.itertuples():
        g=raw[(raw.session==row.run)&(raw.rule_group==row.condition)]
        assert len(g)==row.n_trials and g.keypress.eq(row.target).sum()==row.n_target
    for (run,condition),g in raw.groupby(['session','rule_group']):
        s=transitions[(transitions.run==run)&(transitions.condition==condition)]
        # Independent per-list counts; groupby on full raw order before condition filtering
        expected=np.zeros((4,4),int)
        for _,block in raw[raw.session.eq(run)].groupby(['subject','block']):
            block=block.sort_values('trial_in_run')
            for i in range(1,len(block)):
                if block.rule_group.iloc[i]==condition:
                    expected[int(block.keypress.iloc[i-1])-1,int(block.keypress.iloc[i])-1]+=1
        for row in s.itertuples():assert expected[row.previous_target-1,row.target-1]==row.n_transition
    words=pd.read_csv(path/'cyclic_pair_triplet_statistics.csv')
    for order in [2,3]:
        for name,seq in [('StrA',[3,1,1,2,3,4,2,4]),('StrB',[3,4,2,1,1,3,2,4])]:
            expanded=seq+seq[:order-1]
            observed=['-'.join(map(str,expanded[i:i+order])) for i in range(8)]
            for row in words[words.order.eq(order)].itertuples():
                assert getattr(row,'count_per_cycle_'+name)==observed.count(row.word)
    timeline=pd.read_csv(path/'representative_run_timeline.csv')
    assert timeline.start_s.iloc[0]==0 and timeline.end_s.iloc[-1]==400
    assert np.array_equal(timeline.start_s.to_numpy()[1:],timeline.end_s.to_numpy()[:-1])
    assert timeline.duration_s.sum()==400
    for p in path.glob('*.csv'):
        columns=pd.read_csv(p,nrows=0).columns
        assert not set(columns)&{'subject','participant','source_file'},p
    return dict(passed=True,D1_labels_independently_recomputed=56320,repeat_counts_agree_between_B2_B3=True,
        all_80_target_counts_and_320_transition_counts_recomputed=True,cyclic_pairs_triplets_recomputed=True,
        timeline_tiles_400s=True,no_participant_id_columns_in_B3_CSVs=True)


def check_manifests():
    manifests=[JOINT/'input_manifest.json']+list(BEHAV.glob('**/input_manifest.json'))
    inputs={};n_outputs=0
    for p in manifests:
        m=json.loads(p.read_text())
        inputs.update(m['inputs'])
        validation=json.loads((p.parent/'validation.json').read_text())
        assert validation['passed']
        for rel,sha in validation['output_sha256'].items():
            assert digest(p.parent/rel)==sha,(p,rel)
            n_outputs+=1
    for p,meta in inputs.items():assert digest(p)==meta['sha256'],p
    # Read-only check against the coordinator's pre-analysis baseline.
    baselines=sorted((ROOT/'archive').glob('review_r1_preflight_*/project_sha256.txt'))
    assert len(baselines)==1
    checked=0;mismatches=[]
    for baseline in [baselines[0],baselines[0].with_name('results_sha256.txt')]:
        for line in baseline.read_text().splitlines():
            sha,rel=line.split('  ',1)
            if digest(ROOT/rel)!=sha:mismatches.append(rel)
            checked+=1
    return dict(passed=not mismatches,unique_consumed_input_hashes_checked=len(inputs),
        recorded_output_hashes_checked=n_outputs,preflight_protected_files_checked=checked,
        preflight_mismatches=mismatches)


def main():
    started=time.perf_counter()
    raw=pd.read_csv(OLD_B/'source_replay/raw_trials_N22.csv')
    clean=pd.read_csv(OLD_B/'source_replay/N22/cleaned_trials.csv')
    flags=independent_repeats(raw)
    checks=dict(A1=check_joint(),B1=check_rt(BEHAV/'B1_participant_rt',clean,flags,False),
                B2=check_rt(BEHAV/'B2_repetition_behavior',clean,flags,True),B3=check_design(raw,flags),
                integrity=check_manifests())
    assert checks['integrity']['passed'],checks['integrity']
    runtime=time.perf_counter()-started
    output=BEHAV/'independent_validation.json'
    writej(output,dict(passed=True,verification_source=str(Path(__file__).relative_to(ROOT)),
                      verification_source_sha256=digest(__file__),elapsed_seconds=runtime,checks=checks))
    report=ROOT/'docs/review_round1/reports/r1-stats.md'
    files=sorted([p for root in [BEHAV,JOINT] for p in root.rglob('*') if p.is_file()]+
                 [p for p in (ROOT/'code/review_r1').glob('stats_*') if p.is_file()])
    inventory=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p),bytes=p.stat().st_size) for p in files]
    save(inventory,BEHAV/'owned_file_inventory.csv')
    inventory.append(dict(path=str((BEHAV/'owned_file_inventory.csv').relative_to(ROOT)),
        sha256=digest(BEHAV/'owned_file_inventory.csv'),bytes=(BEHAV/'owned_file_inventory.csv').stat().st_size))
    durations={name:json.loads((path/'validation.json').read_text())['elapsed_seconds'] for name,path in
        [('A1',JOINT),('B1',BEHAV/'B1_participant_rt'),('B2',BEHAV/'B2_repetition_behavior'),('B3',BEHAV/'B3_design_statistics')]}
    b1=pd.read_csv(BEHAV/'B1_participant_rt/all_contrast_comparisons.csv')
    early_late=b1.query('policy=="primary" and contrast=="early_minus_late"').iloc[0]
    summary=(
        '# r1-stats handoff — 2026-10-19 (Asia/Taipei)\n\n'
        'Request: implement A1, B1, B2 and B3 from the fixed review-R1 plan, with coordinator '
        'D1 (19:10, before B2 outcomes). Work is complete in the owned paths listed below. '
        'The roots and role report did not exist at start (record: /tmp/r1-r1-stats/start.json). '
        'Only new stats code, the two assigned result roots, this report and assigned /tmp '
        'scratch were written. No Git writes, network calls, installs, model refits, manuscript '
        'edits, status/log edits, publication or external messages were performed.\n\n'
        '## Neutral results\n\n'
        '- A1: all 104 original 48-slot stage BH families reproduced; 26 joint 192-slot '
        'families exported across nine specifications, two cohorts and the already available '
        'coverage policies. Maximum stage-q discrepancy is below 5e-11. Primary N22 reference '
        'observed80: all four early headline ROIs survive jointly (left BA6 .010881, right '
        'BA6 .010346, right pallidum .002211, right thalamus .025118). Initial discoveries '
        'change from 28 to 24; early remains 4, while joint-only discoveries appear in middle '
        '(2) and late (1). BH pooling can produce both losses and gains because its rejection '
        'threshold depends on the whole p distribution. Every slot and gain/loss is saved; '
        'this is not correction for the earlier analysis search. N20 remains supplement-only.\n'
        '- B1: original R cleaning replay exactly matched 53,959 primary / 52,126 min100 '
        'trials. Both policies retain all 440 required participant cells; all tests use '
        'df=21 and the same 24 contrast definitions/family as the saved LME. Primary '
        'participant-summary run-2 Str1/Random = .6856 [.5930,.7926], run-8 Str2/Random = '
        '.6600 [.5433,.8019], early adjusted ratio = 1.2463 [1.1386,1.3642], and early–late '
        '= 1.3487 [1.1418,1.5932] (pointwise t CIs). Early–late log-CI width is '
        f'{early_late.t_log_CI_width_over_lme:.4f} times the saved LME width. Bootstrap '
        'early–late CI is [1.1680,1.5906]. Full comparisons include estimates, t/bootstrap '
        'CIs, log- and ratio-scale CI-width ratios, and q. Participant-summary BH detections '
        'are 21/24 primary and 22/24 min100; q≤.05 agreement with LME is 19/24 and 20/24. '
        'Different weighting/estimands limit interpretation of agreement.\n'
        '- B2: D1 yields 0 Random, 2,640 Str1 and 2,638 Str2 immediate-repeat trials; '
        'none crosses a block boundary. Gap-separated target matches (48 Random, 29 Str1, '
        '20 Str2) are explicitly excluded from D1. All four key contrasts retain q≤.05 '
        'under both RT policies after repeat exclusion. Primary nonrepeat ratios are '
        '.6883, .6612, 1.2444 and 1.3467 in the same key-contrast order. Shared/diagnostic '
        'transition phase descriptions are saved without p/q claims, with same-run Random '
        'means as the reference; this reference is not transition-matched.\n'
        '- B3: all 1,408 condition lists, 176 native/canonical runs, 56,320 trials and 1,584 '
        'fixation values match. Full target frequencies, 4×4 transition matrices and D1 '
        'repeat rates are saved by run/condition. Cyclic shared pairs are 1–1, 2–4, 3–4, '
        '4–2, 4–3; shared triplets are 2–4–3 and 3–4–2. Outer block order is fixed: first '
        'Random, six structured blocks, final Random. Interleaved runs have distinct '
        'realized within-run sequence orders for all 22 participants. All original '
        'ordering constraints checked pass. A deterministic logged run-3 timeline (CSV, '
        'PDF, PNG) shows 320 s task slots + 72 s fixation + 4 s trigger wait + 4 s remaining '
        'acquisition window, with no inferred cue exposure. All B3 tables are group-level.\n\n'
        '## Decisions, implementation and deviations\n\n'
        'User hierarchy preserved: N22 uncensored observed80 reference primary; A1 N20 '
        'is a supplement-only subset sensitivity. No specification or ROI was selected '
        'from its outcome. All intervals are pointwise. Coordinator D1 is the sole dated '
        'plan clarification/deviation adopted (2026-10-19 19:10): target equality must '
        'occur across consecutive slots exactly 1 s apart in the same run. The earlier '
        'statement about structured-only repeats was an expectation, not a condition '
        'restriction. Under D1 no eligible Random repeats remain.\n\n'
        'B2 keeps the original 24-slot LME BH family with the four requested tests and '
        '20 untested slots at p=1; all slots are explicit. No four-slot replacement '
        'family was introduced. Shared/diagnostic descriptions use the nonrepeat data, '
        'actual preceding target pairs including cyclic wraps, same-run Random means '
        'and equal-run phase averages. Empty required cells would have stopped the '
        'analysis; none occurred. No further scientific decision was needed.\n\n'
        '## Commands, runtimes and checks\n\n'
        'Environment: /media/data3/chiaoen_srtt/envs/nilearn-0131/bin/python, nilearn '
        '0.13.1 asserted; /usr/bin/Rscript, R 4.4.2, trimr 1.1.1. One worker, BLAS/OpenMP '
        'thread limits 1 throughout; no imaging workers spawned. Main command:\n\n'
        '```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
        'PYTHONDONTWRITEBYTECODE=1 /media/data3/chiaoen_srtt/envs/nilearn-0131/bin/python '
        '-B code/review_r1/stats_run.py\n'
        '```\n\nThe runner invokes:\n\n'
        '```bash\n/usr/bin/Rscript code/review_r1/stats_cleaning_replay.R '
        '/media/data3/chiaoen_srtt /media/data3/chiaoen_srtt/results/behavioral/review_r1_20261019/B1_participant_rt\n'
        '```\n\nSaved-output verification command:\n\n'
        '```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '
        '/media/data3/chiaoen_srtt/envs/nilearn-0131/bin/python -B code/review_r1/stats_validate.py\n```\n\n'
        'Runtimes from final successful run (seconds): '+', '.join(f'{k}={v:.3f}' for k,v in durations.items())+
        f'; saved-output/integrity verification={runtime:.3f}. '
        'An initial run stopped on a DataFrame index-name mismatch in the independently '
        'constructed weight-check table; the index name was corrected before inference. '
        'A full successful run followed. A final reporting update preserved small q values '
        'in scientific notation and added the timeline figure, followed by a full rerun. '
        'Only owned outputs were replaced.\n\n'
        'Production and independent checks passed: original cleaner identities/RTs; '
        'source contrast weights; family sizes; manual sorted-rank BH; participant values '
        'reconstructed from cleaned trials; SciPy one-sample t tests, df=21 CIs and '
        'back-transforms; all bootstrap indices/percentiles; all CI-width ratios; D1 '
        'labels independently recomputed for 56,320 trials; all group target/transition '
        'counts; cyclic pair/triplet enumeration; timeline coverage; source/list/fixation '
        'matching and ordering constraints; B3 group-only columns. '
        f'Integrity: {checks["integrity"]["unique_consumed_input_hashes_checked"]} unique consumed inputs and '
        f'{checks["integrity"]["recorded_output_hashes_checked"]} recorded output hashes rechecked; '
        f'all {checks["integrity"]["preflight_protected_files_checked"]} coordinator-preflight protected '
        'file hashes unchanged (0 mismatches). Evidence: each analysis validation.json, '
        'input_manifest.json and results/behavioral/review_r1_20261019/independent_validation.json.\n\n'
        'Limits: this is sensitivity/secondary evidence with retrospective dataset history; '
        'participant summaries do not validate the full LME covariance specification; '
        'bootstrap uses only 22 independent participants; repetition exclusion does not '
        'match all local statistics; descriptive transition categories do not isolate '
        'whole-sequence representations; task-slot durations are not measured cue exposure. '
        'No manuscript/UI/build validation is claimed. Historical scientific outputs were '
        'preserved. Next: coordinator and independent r1-audit review these files against '
        'the fixed plan; manuscript integration awaits user review.\n\n'
        '## Files created and SHA-256\n\n'
        'The machine-readable inventory is results/behavioral/review_r1_20261019/'
        'owned_file_inventory.csv. It excludes itself to avoid a recursive hash. '
        'This role report is likewise not self-hashed. All other owned files, plus the '
        'inventory itself, are listed here:\n\n| Path | SHA-256 |\n|---|---|\n'
        +'\n'.join(f'| `{r["path"]}` | `{r["sha256"]}` |' for r in inventory)+'\n')
    report.write_text(summary)
    print(json.dumps(dict(passed=True,checks=checks,runtime_seconds=runtime,report=str(report)),indent=2))


if __name__=='__main__':main()
