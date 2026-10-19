#!/usr/bin/env python3
"""A1, B1 and B2 from the fixed 2026-10-19 review-R1 plan (including D1)."""
from stats_common import *
import subprocess
from scipy import stats


def table(frame, columns):
    def fmt(value):
        if isinstance(value,(float,np.floating)):
            return format(value,'.6g') if np.isfinite(value) else 'NA'
        return str(value)
    return '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---']*len(columns)) + '|\n' + '\n'.join(
        '| ' + ' | '.join(fmt(r[c]) for c in columns) + ' |' for _, r in frame.iterrows())


def joint_family():
    t = Task(JOINT)
    a = t.read(OLD_I / 'roi_reporting/stage_results.csv')
    b = t.read(OLD_I / 'targeted_sensitivity/all_stage_results.csv')
    keys = ['cohort', 'model', 'policy', 'stage', 'roi']
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    overlap = a.merge(b, on=keys, suffixes=('_roi', '_targeted'))
    comparisons = []
    for col in [c for c in a if c not in keys]:
        x, y = overlap[col+'_roi'].to_numpy(), overlap[col+'_targeted'].to_numpy()
        assert np.allclose(x, y, atol=1e-9, rtol=1e-9, equal_nan=True), col
        comparisons.append(dict(column=col, n_duplicate_slots=len(overlap),
                                max_absolute_difference=float(np.nanmax(abs(x-y)))))
    save(comparisons, t.path/'duplicate_source_checks.csv')
    # Use roi_reporting values on overlap; add the three additional specifications.
    existing = set(map(tuple, a[keys].to_numpy()))
    extra = b[[tuple(x) not in existing for x in b[keys].to_numpy()]]
    d = pd.concat([a, extra], ignore_index=True).rename(columns={'q':'q_stage_saved'})
    rois = set((ROOT/'code/replication/config/rois.txt').read_text().split())
    errors = []
    for key, g in d.groupby(['cohort','model','policy','stage']):
        assert len(g)==48 and set(g.roi)==rois, key
        q = bh(g.p.to_numpy())
        error = float(np.max(abs(q-g.q_stage_saved.to_numpy())))
        assert error < 5e-11, (key,error)
        errors.append(error)
        d.loc[g.index,'q_stage_recomputed'] = q
    for key, g in d.groupby(['cohort','model','policy']):
        assert len(g)==192 and set(g.stage)==set(STAGES), key
        d.loc[g.index,'q_joint_192'] = bh(g.p.to_numpy())
    d['stage_discovery'] = d.q_stage_recomputed.le(.05)
    d['joint_discovery'] = d.q_joint_192.le(.05)
    d['survives_jointly'] = d.stage_discovery & d.joint_discovery
    d['primary'] = (d.cohort.eq('N22') & d.model.eq('reference') & d.policy.eq('observed80'))
    d['analysis_role'] = np.where(d.primary,'N22_uncensored_reference_primary',
        np.where(d.cohort.eq('N20'),'supplement_only_subset_sensitivity','sensitivity'))
    d = d.assign(_c=d.cohort.map({'N22':0,'N20':1}),
        _m=np.where(d.model.eq('reference'),0,1),
        _p=np.where(d.policy.eq('observed80'),0,1),_s=d.stage.map(dict(zip(STAGES,range(4)))))
    d = d.sort_values(['_c','_m','_p','model','policy','_s','roi']).drop(columns=['_c','_m','_p','_s'])
    save(d,t.path/'joint_stage_results.csv')
    save(d[d.stage_discovery],t.path/'stage_discovery_survival.csv')
    save(d[d.joint_discovery & ~d.stage_discovery],t.path/'joint_only_discoveries.csv')
    counts = d.groupby(['cohort','model','policy','stage'],sort=False).agg(
        slots=('roi','size'),testable=('p',lambda x:int(np.isfinite(x).sum())),
        stage_discoveries=('stage_discovery','sum'),joint_discoveries=('joint_discovery','sum'),
        stage_discoveries_surviving=('survives_jointly','sum')).reset_index()
    save(counts,t.path/'family_discovery_counts.csv')
    p = counts.query('cohort=="N22" and model=="reference" and policy=="observed80"')
    headline = d.query('primary and stage=="early" and roi in ["LeftSMA","RightSMA","RightPallidum","RightThalamusProper"]')
    t.finish('# A1: joint stage-by-ROI BH sensitivity\n\n'
        'Primary: N=22 uncensored reference, observed80. No model was refitted. '
        'All nine existing specifications are included for N22 and the supplement-only N20 subset. '
        'The reference/censored observed95 and complete100 policies already present in roi_reporting '
        'are also retained, giving 26 cohort × model × policy families. Each joint family has '
        '4 stages × 48 ROI slots = 192. Nonfinite p values enter BH as 1, using common.bh. '
        'The original 48-slot stage families were reproduced before joint correction. '
        'Overlapping source rows agree to numerical precision; roi_reporting is authoritative on overlap.\n\n'
        +table(p,['stage','slots','stage_discoveries','joint_discoveries','stage_discoveries_surviving'])
        +'\n\nFour headline early ROIs (historical SMA filenames refer to the complete BA6 parcels):\n\n'
        +table(headline,['roi','mean','q_stage_recomputed','q_joint_192','survives_jointly'])
        +'\n\nFull slots, discovery survival and any joint-only discoveries are exported. '
        'This sensitivity addresses the stage-family choice; it does not account for prior analysis '
        'selection or make the findings confirmatory. Pointwise CIs are copied unchanged. '
        'N20 and N22 overlap and are not independent replications. No manuscript or UI was changed.\n\n'
        'Deviations: none for A1.\n',
        dict(stage_families=len(errors),joint_families=len(d)//192,n_slots=len(d),
             max_stage_q_absolute_error=max(errors),duplicate_slots_checked=len(overlap),
             all_stage_families_48=True,all_joint_families_192=True,n_models_observed80=9))


def intended_weights(columns):
    def vec(terms):
        result = pd.Series(0.,index=columns)
        for run, cond, weight in terms:
            result[f'cellr{run:02d}_{cond}'] += weight
        return result
    vectors = {}
    for run in range(1,9):
        for cond in ['Str1','Str2']:
            if f'cellr{run:02d}_{cond}' in columns:
                vectors[f'run{run}_{cond}_minus_Ran'] = vec([(run,cond,1),(run,'Ran',-1)])
    vectors['Str1_run2_minus_run1'] = vec([(2,'Str1',1),(1,'Str1',-1)])
    vectors['Str2_run8_minus_run3'] = vec([(8,'Str2',1),(3,'Str2',-1)])
    for run in range(3,7):
        vectors[f'run{run}_Str2_minus_Str1'] = vec([(run,'Str2',1),(run,'Str1',-1)])
    base = vec([(1,'Str1',.5),(2,'Str1',.5),(1,'Ran',-.5),(2,'Ran',-.5)])
    phases = {}
    for name, run in zip(STAGES[1:],[3,5,7]):
        phases[name] = vec([(run,'Str2',.5),(run+1,'Str2',.5),(run,'Ran',-.5),(run+1,'Ran',-.5)])-base
        vectors[name+'_adjusted_change'] = phases[name]
    for a,b in [('early','middle'),('early','late'),('middle','late')]:
        vectors[a+'_minus_'+b] = phases[a]-phases[b]
    result = pd.DataFrame(vectors).T
    result.index.name = 'contrast'
    return result


def participant_analysis(t, d, weights, lme, policy, subset=False):
    d = d.copy()
    d['log_rt'] = np.log(d.p_response_rt*1000)
    d['cell'] = [f'cellr{int(r):02d}_{c}' for r,c in zip(d.session,d.rule_group)]
    cells = d.groupby(['subject','session','rule_group','cell']).agg(
        mean_log_rt=('log_rt','mean'),n_trials=('log_rt','size')).reset_index()
    subjects = sorted(d.subject.unique())
    assert len(subjects)==22
    matrix = cells.pivot(index='subject',columns='cell',values='mean_log_rt').reindex(
        index=subjects,columns=weights.columns)
    # Empty required cells are never imputed or assigned zero. Stop if any exist.
    assert matrix.shape==(22,20) and np.isfinite(matrix.to_numpy()).all(), 'Empty required RT cell: decision needed'
    selected = weights.loc[KEYS] if subset else weights
    values = matrix.to_numpy() @ selected.to_numpy().T
    draws = boot_draws()
    distribution = values[draws].mean(axis=1)
    save(cells,t.path/f'participant_{policy}_cell_log_rt.csv')
    contrasts = pd.DataFrame(values,index=pd.Index(subjects,name='subject'),columns=selected.index).reset_index().melt(
        id_vars='subject',var_name='contrast',value_name='log_difference')
    contrasts['policy'] = policy
    save(contrasts,t.path/f'participant_{policy}_contrasts.csv')
    np.save(t.path/f'participant_{policy}_bootstrap_indices.npy',draws)
    save(pd.DataFrame({'bootstrap_index':range(22),'subject':subjects}),
         t.path/f'participant_{policy}_bootstrap_subject_order.csv')
    rows=[]
    for i, contrast in enumerate(selected.index):
        s=summarize(values[:,i]);lo,hi=np.quantile(distribution[:,i],[.025,.975],method='linear')
        ref=lme.loc[contrast]
        row=dict(policy=policy,contrast=contrast,n=s['n'],df=s['df'],log_difference=s['mean'],
            sd=s['sd'],se=s['se'],t=s['t'],p_two_sided=s['p'],log_ci_low=s['ci_low'],log_ci_high=s['ci_high'],
            bootstrap_log_ci_low=lo,bootstrap_log_ci_high=hi,bootstrap_resamples=NBOOT,bootstrap_seed=SEED,
            geometric_rt_ratio=np.exp(s['mean']),ratio_ci_low=np.exp(s['ci_low']),ratio_ci_high=np.exp(s['ci_high']),
            bootstrap_ratio_ci_low=np.exp(lo),bootstrap_ratio_ci_high=np.exp(hi),
            lme_log_difference=ref.log_difference,lme_log_ci_low=ref.log_ci_low,lme_log_ci_high=ref.log_ci_high,
            lme_geometric_rt_ratio=ref.geometric_rt_ratio,lme_ratio_ci_low=ref.ratio_ci_low,
            lme_ratio_ci_high=ref.ratio_ci_high,lme_q_bh_24=ref.q_bh_24,
            estimate_difference_from_lme=s['mean']-ref.log_difference,
            same_estimate_sign_as_lme=bool(np.sign(s['mean'])==np.sign(ref.log_difference)),
            t_log_CI_width_over_lme=(s['ci_high']-s['ci_low'])/(ref.log_ci_high-ref.log_ci_low),
            bootstrap_log_CI_width_over_lme=(hi-lo)/(ref.log_ci_high-ref.log_ci_low),
            t_ratio_CI_width_over_lme=(np.exp(s['ci_high'])-np.exp(s['ci_low']))/(ref.ratio_ci_high-ref.ratio_ci_low),
            bootstrap_ratio_CI_width_over_lme=(np.exp(hi)-np.exp(lo))/(ref.ratio_ci_high-ref.ratio_ci_low))
        assert s['df']==21
        rows.append(row)
    result=pd.DataFrame(rows)
    # B2 keeps the original 24-slot family. Only the four requested tests receive
    # measured p values; the other twenty untested slots have p=1.
    pvalues=pd.Series(1.,index=weights.index)
    pvalues.loc[result.contrast] = result.p_two_sided.to_numpy()
    qvalues=pd.Series(bh(pvalues.to_numpy()),index=pvalues.index)
    result['q_bh_24'] = result.contrast.map(qvalues)
    result['same_q05_decision_as_lme'] = result.q_bh_24.le(.05).eq(result.lme_q_bh_24.le(.05))
    family=pd.DataFrame({'policy':policy,'contrast':pvalues.index,'p_for_BH':pvalues.to_numpy(),
                         'q_bh_24':qvalues.to_numpy(),'tested':pvalues.index.isin(result.contrast)})
    save(family,t.path/f'{policy}_BH_family_slots.csv')
    save(result,t.path/f'{policy}_contrast_comparison.csv')
    return result


def load_behavior(t):
    raw=t.read(OLD_B/'source_replay/raw_trials_N22.csv')
    clean=t.read(OLD_B/'source_replay/N22/cleaned_trials.csv')
    assert len(raw)==56320 and len(clean)==53959
    assert raw.groupby(['subject','session']).size().eq(320).all()
    assert not raw.duplicated(['subject','session','trial_in_run']).any()
    assert clean.correct_response.eq(1).all() and clean.p_response_rt.gt(0).all() and clean.p_response_rt.lt(1).all()
    cols=['subject','session','trial_in_run','rule_group','p_response_rt']
    expected=raw.merge(clean[['subject','session','trial_in_run']],on=['subject','session','trial_in_run'],validate='one_to_one')
    sort=lambda d:d[cols].sort_values(cols[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sort(expected),sort(clean),check_exact=True)
    weights={};lmes={}
    for policy in ['primary','min100']:
        weights[policy]=t.read(OLD_B/f'rt_lme_N22/{policy}_contrast_weights.csv').set_index('contrast')
        lmes[policy]=t.read(OLD_B/f'rt_lme_N22/{policy}_contrasts.csv').set_index('contrast')
        pd.testing.assert_frame_equal(weights[policy],intended_weights(weights[policy].columns),check_dtype=False)
        assert np.max(abs(bh(lmes[policy].p_two_sided)-lmes[policy].q_bh_24))<1e-12
    pd.testing.assert_frame_equal(weights['primary'],weights['min100'])
    return raw,clean,weights,lmes


def rt_validation():
    t=Task(BEHAV/'B1_participant_rt')
    for p in ['code/reanalysis/updated_rt_lme.R','code/reanalysis/replay_updated_behavior.R',
              'results/behavioral/reanalysis_20261018_updated_n22/source_replay/validation.json',
              'results/behavioral/reanalysis_20261018_updated_n22/rt_lme_N22/analysis_spec.json']:
        t.record(ROOT/p)
    t.record('/media/data2/pinwei/Joanne_SRT_2/For_teamNCU_Joanne/My_modifiedRecursive.R')
    replay=subprocess.run(['/usr/bin/Rscript',str(ROOT/'code/review_r1/stats_cleaning_replay.R'),str(ROOT),str(t.path)],
                          capture_output=True,text=True,check=True)
    (t.path/'cleaning_replay.log').write_text(replay.stdout+replay.stderr)
    raw,clean,weights,lmes=load_behavior(t)
    results=[]
    for policy in ['primary','min100']:
        d=clean if policy=='primary' else clean[clean.p_response_rt.ge(.1)]
        status=json.loads(t.record(OLD_B/f'rt_lme_N22/{policy}_status.json').read_text())
        assert len(d)==status['n_trials']
        results.append(participant_analysis(t,d,weights[policy],lmes[policy],policy))
    allresults=pd.concat(results,ignore_index=True)
    save(allresults,t.path/'all_contrast_comparisons.csv')
    brief=allresults[allresults.contrast.isin(KEYS)]
    t.finish('# B1: participant-level RT contrast validation\n\n'
        'N=22 only. The original custom modified-recursive cleaner was replayed in R 4.4.2 '
        '(trimr 1.1.1), using minRT=0, participant × original block labels and correct responses. '
        'Retained trial identities and RTs match the saved primary LME input exactly: 53,959 trials; '
        'the subsequent RT≥100-ms policy retains 52,126. The LME was not refitted.\n\n'
        'For each participant × run × condition, average natural log RT in milliseconds over '
        'retained trials. Apply the saved 24 LME weight vectors to these 20 observed cells. '
        'The weights are independently checked against the source definitions (equal-run phase '
        'means and baseline subtraction). Absent design conditions are not fabricated. '
        'An empty required cell would stop this analysis; none occurred in either policy '
        '(440/440 required cells each). All 22 participants enter every contrast. '
        'One-sample t tests and pointwise 95% t intervals use df=21. BH spans all 24 contrasts '
        'separately per cleaning policy, exactly the LME family. Bootstrap intervals use '
        '10,000 whole-participant percentile resamples of the mean log contrast, NumPy PCG64 '
        'default_rng(20261019), linear .025/.975 quantiles; the same draws serve all contrasts '
        'and both policies. Indices and participant contrast values are saved internally. '
        'Bootstrap intervals have no separate p or q values.\n\n'
        'Exponentiating the mean log contrast and both interval endpoints produces geometric '
        'RT ratios (ratios of ratios for change contrasts), as in Table 1. These are participant '
        'summary estimates; the saved LME estimates are fixed effects with random effects at zero. '
        'Equal participant/cell weighting differs from trial-level model weighting. '
        'Both log-scale and ratio-scale CI-width ratios (participant / saved LME) are exported, '
        'separately for t and bootstrap intervals. Agreement flags mean only matching sign or '
        'matching q≤.05 status, not equivalence or independent replication.\n\n'
        +table(brief,['policy','contrast','geometric_rt_ratio','ratio_ci_low','ratio_ci_high',
            'bootstrap_ratio_ci_low','bootstrap_ratio_ci_high','q_bh_24','lme_q_bh_24',
            't_log_CI_width_over_lme'])
        +'\n\nAll 48 policy × contrast results are reported regardless of outcome. '
        'CIs are pointwise, not simultaneous; resampling 22 participants cannot establish '
        'a correct trial-level covariance model. No manuscript or UI validation was performed.\n\n'
        'Deviations: none for B1.\n',dict(n_contrast_rows=48,participants=22,required_cells_each_policy=440,
            empty_cells=0,all_df_21=True,bootstrap_resamples=NBOOT,bootstrap_seed=SEED,
            contrast_weights_match_source=True,LME_BH_reproduced=True,
            cleaning_replay=json.loads((t.path/'cleaning_replay_validation.json').read_text()),
            q05_discoveries={r.policy.iloc[0]:int(r.q_bh_24.le(.05).sum()) for r in results},
            q05_agreement_counts={r.policy.iloc[0]:int(r.same_q05_decision_as_lme.sum()) for r in results}))


SEQUENCES={'StrA': [3,1,1,2,3,4,2,4], 'StrB':[3,4,2,1,1,3,2,4]}

def cyclic_words(sequence,order):
    return [tuple(sequence[(i+j)%len(sequence)] for j in range(order)) for i in range(len(sequence))]


def annotate(raw):
    d=raw.sort_values(['subject','session','trial_in_run']).copy()
    g=d.groupby(['subject','session'])
    d['previous_target']=g.keypress.shift()
    d['previous_onset']=g.stim_onset.shift()
    d['previous_block']=g.block.shift()
    d['onset_difference_s']=d.stim_onset-d.previous_onset
    d['adjacent_1s']=d.onset_difference_s.eq(1)
    d['cross_block_boundary']=d.previous_block.notna() & d.block.ne(d.previous_block)
    d['same_target_previous_slot']=d.keypress.eq(d.previous_target)
    d['immediate_repeat_D1']=d.same_target_previous_slot & d.adjacent_1s
    pairs={k:set(cyclic_words(v,2)) for k,v in SEQUENCES.items()}
    shared=pairs['StrA']&pairs['StrB']
    def classify(row):
        if not row.adjacent_1s: return 'undefined_after_gap_or_run_start'
        pair=(int(row.previous_target),int(row.keypress))
        if pair in shared: return 'shared'
        if pair in pairs['StrA']-pairs['StrB']: return 'diagnostic_StrA'
        if pair in pairs['StrB']-pairs['StrA']: return 'diagnostic_StrB'
        return 'neither_sequence'
    d['transition_class']=[classify(row) for row in d.itertuples()]
    d['phase']=d.session.map(phase)
    return d


def repetition_behavior():
    t=Task(BEHAV/'B2_repetition_behavior')
    raw,clean,weights,lmes=load_behavior(t)
    t.record(ROOT/'results/glm_volume/reanalysis_20261018/source_discovery/source_snapshots/stimulus_generator.py')
    annotated=annotate(raw)
    fields=['subject','session','trial_in_run','block','rule','rule_group','keypress','stim_onset','previous_target',
            'previous_onset','onset_difference_s','cross_block_boundary','same_target_previous_slot',
            'adjacent_1s','immediate_repeat_D1','transition_class','phase']
    save(annotated[fields],t.path/'participant_trial_annotations.csv')
    counts=annotated.groupby(['session','rule_group']).agg(n_trials=('keypress','size'),
        adjacent_1s_opportunities=('adjacent_1s','sum'),same_target_previous_slot=('same_target_previous_slot','sum'),
        immediate_repeats_D1=('immediate_repeat_D1','sum')).reset_index()
    cross=annotated.assign(cross_repeat=annotated.immediate_repeat_D1 & annotated.cross_block_boundary,
        gap_match=annotated.same_target_previous_slot & ~annotated.adjacent_1s).groupby(['session','rule_group']).agg(
        repeats_crossing_block=('cross_repeat','sum'),gap_separated_target_matches_excluded=('gap_match','sum')).reset_index()
    counts=counts.merge(cross,on=['session','rule_group'])
    save(counts,t.path/'D1_repeat_counts_by_condition_run.csv')
    totals=counts.groupby('rule_group').sum(numeric_only=True).drop(columns='session').reset_index()
    save(totals,t.path/'D1_repeat_counts_by_condition.csv')
    keys=['subject','session','trial_in_run']
    joined=clean.merge(annotated[keys+['immediate_repeat_D1','transition_class','phase']],on=keys,validate='one_to_one')
    nonrepeat=joined[~joined.immediate_repeat_D1].copy()
    results=[];descriptions=[];participant_descriptions=[];retention=[]
    for policy in ['primary','min100']:
        d=nonrepeat if policy=='primary' else nonrepeat[nonrepeat.p_response_rt.ge(.1)]
        ref=joined if policy=='primary' else joined[joined.p_response_rt.ge(.1)]
        retention.append(dict(policy=policy,n_cleaned=len(ref),repeats_excluded=int(ref.immediate_repeat_D1.sum()),
                              n_nonrepeat=len(d)))
        result=participant_analysis(t,d,weights[policy],lmes[policy],policy,subset=True)
        b1=t.read(BEHAV/f'B1_participant_rt/{policy}_contrast_comparison.csv').set_index('contrast')
        result['all_trials_participant_log_difference']=result.contrast.map(b1.log_difference)
        result['all_trials_participant_ratio']=result.contrast.map(b1.geometric_rt_ratio)
        result['all_trials_participant_q_bh_24']=result.contrast.map(b1.q_bh_24)
        result['nonrepeat_minus_all_trials_log_difference']=result.log_difference-result.all_trials_participant_log_difference
        save(result,t.path/f'{policy}_contrast_comparison.csv')
        results.append(result)
        d=d.copy();d['log_rt']=np.log(d.p_response_rt*1000)
        random=d[d.rule_group.eq('Ran')].groupby(['subject','session']).log_rt.mean()
        s=d[d.rule_group.ne('Ran')].copy()
        assert s.transition_class.isin(['shared','diagnostic_StrA','diagnostic_StrB','undefined_after_gap_or_run_start']).all()
        s=s[s.transition_class.ne('undefined_after_gap_or_run_start')].copy()
        s['transition_category']=np.where(s.transition_class.eq('shared'),'shared','diagnostic')
        cells=s.groupby(['subject','session','phase','rule_group','transition_category']).agg(
            structured_log_rt=('log_rt','mean'),n_structured_trials=('log_rt','size')).reset_index()
        cells['random_log_rt']=[random.loc[(row.subject,row.session)] for row in cells.itertuples()]
        cells['log_ratio']=cells.structured_log_rt-cells.random_log_rt
        phase_values=cells.groupby(['subject','phase','rule_group','transition_category']).agg(
            log_ratio=('log_ratio','mean'),n_runs=('session','size'),n_structured_trials=('n_structured_trials','sum')).reset_index()
        assert phase_values.n_runs.eq(2).all()
        phase_values['policy']=policy;participant_descriptions.append(phase_values)
        for key,g in phase_values.groupby(['phase','rule_group','transition_category']):
            z=summarize(g.log_ratio)
            descriptions.append(dict(policy=policy,phase=key[0],condition=key[1],transition_category=key[2],
                n=z['n'],mean_log_ratio=z['mean'],log_ci_low=z['ci_low'],log_ci_high=z['ci_high'],
                geometric_rt_ratio=np.exp(z['mean']),ratio_ci_low=np.exp(z['ci_low']),ratio_ci_high=np.exp(z['ci_high']),
                n_structured_trials=int(g.n_structured_trials.sum()),random_reference='all_nonrepeat_random_trials_same_run',
                inference='descriptive_pointwise_t_CI_no_p_or_q'))
    save(retention,t.path/'trial_retention.csv')
    save(pd.concat(results),t.path/'key_contrast_comparisons.csv')
    save(pd.concat(participant_descriptions),t.path/'participant_transition_category_phase_values.csv')
    save(descriptions,t.path/'descriptive_transition_category_advantage.csv')
    t.finish('# B2: immediate-repetition exclusion and transition descriptions\n\n'
        '**D1, coordinator clarification, 2026-10-19 19:10 Asia/Taipei (before A5/B2 results):** '
        'A repeat has the same cued target as the immediately preceding trial slot in the same '
        'run, with logged onsets exactly 1 s apart, regardless of condition or block boundary. '
        'Targets come from `keypress` (the cued target, not `p_response_keys`); trial order is '
        '`trial_in_run`, time is `stim_onset`. Labels are computed from all raw trials before '
        'RT/error exclusions; errors do not redefine the preceding stimulus. Fixation/trigger-gap '
        'matches are excluded from the repeat definition.\n\n'
        +table(totals,['rule_group','n_trials','immediate_repeats_D1','repeats_crossing_block',
                       'gap_separated_target_matches_excluded'])
        +'\n\nStarting from the exact B1 cleaned trials, exclude D1 repeats without rerunning '
        'the outlier cleaner. Repeat the four fixed key contrasts for both RT policies, using '
        'the same participant means, df=21 t tests, pointwise t CIs and 10,000 whole-participant '
        'bootstrap draws (seed 20261019). Required cells remain complete. To retain the original '
        'LME family, BH uses 24 slots per policy: the four tested p values and 20 untested slots '
        'at p=1; every slot is exported. Saved LME and all-trials participant results are shown '
        'for comparison, with no LME refit.\n\n'
        +table(pd.concat(results),['policy','contrast','geometric_rt_ratio','ratio_ci_low',
            'ratio_ci_high','bootstrap_ratio_ci_low','bootstrap_ratio_ci_high','q_bh_24',
            'all_trials_participant_ratio'])
        +'\n\nSecondary descriptions use the same nonrepeat trials. The printed physical '
        'sequences are A=(3,1,1,2,3,4,2,4), B=(3,4,2,1,1,3,2,4). Their cyclic pair sets include '
        'the final→first transition. A preceding transition is shared if in both pair sets; '
        'diagnostic if in only the trial’s physical sequence. Actual preceding adjacent trial '
        'targets determine the class, including uninterrupted eight-trial sequence boundaries. '
        'Run starts and trials after gaps have undefined class and are omitted from these '
        'descriptions. For each participant/run/condition/category, subtract the mean log RT '
        'of all nonrepeat Random trials in that run; average the two run contrasts equally '
        'within phase and exponentiate the participant mean. The Random baseline is not '
        'transition-matched, and these descriptive categories do not match every local statistic '
        'or identify a causal sequence effect. Pointwise t intervals are descriptive; no p/q '
        'is assigned. Absent sequence phases are not imputed. Participant values stay internal.\n\n'
        'Deviation/clarification: D1 replaces the original expectation that repeats occur only '
        'in structured conditions; the exact D1 definition is implemented. No other deviation.\n',
        dict(participants=22,n_key_tests_per_policy=4,BH_slots_per_policy=24,untested_slots_at_p1=20,
             empty_required_cells=0,bootstrap_seed=SEED,bootstrap_resamples=NBOOT,
             repeat_definition='D1_same_run_previous_trial_slot_equal_target_exactly_1s_onset_difference',
             repeat_counts=totals.to_dict('records'),n_secondary_description_rows=len(descriptions),
             retained_trials=retention,all_description_phase_run_counts_equal_two=True))


def main():
    joint_family()
    rt_validation()
    repetition_behavior()
    from stats_design import design_statistics
    design_statistics()
    t=Task(BEHAV)
    checks={}
    for subdir in ['B1_participant_rt','B2_repetition_behavior','B3_design_statistics']:
        validation=json.loads(t.record(BEHAV/subdir/'validation.json').read_text())
        assert validation['passed']
        t.record(BEHAV/subdir/'input_manifest.json');t.record(BEHAV/subdir/'REPORT.md')
        checks[subdir]=dict(passed=True,elapsed_seconds=validation['elapsed_seconds'])
    t.finish('# Review-R1 statistics and behavior outputs\n\n'
        'B1_participant_rt: all 24 fixed participant-level RT contrasts, both cleaning policies, '
        'df=21 t tests and 10,000 participant bootstrap resamples.\n\n'
        'B2_repetition_behavior: the four key contrasts after D1 repeat exclusion, with '
        'shared/diagnostic-transition descriptions and explicit repeat/gap counts.\n\n'
        'B3_design_statistics: realized group target/transition/repeat statistics, cyclic '
        'pair/triplet overlap, a logged timeline and ordering variability.\n\n'
        'N22 is the behavioral reference cohort; no LME was refitted. Each subdirectory contains '
        'its own CSVs, REPORT.md, validation.json and SHA-256 input manifest. Participant-prefixed '
        'files are internal and must not be included in a public release without authorization. '
        'A1 joint ROI-stage families are in results/glm_volume/review_r1_20261019_joint_family/. '
        'The only plan clarification implemented is coordinator D1; see B2. '
        'No protected outputs, manuscript, status/log, Git state or UI were written.\n',checks)


if __name__=='__main__':
    main()
