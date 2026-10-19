"""B3 realized design statistics; all exported B3 observations are group-level."""
from stats_common import *
from stats_run import annotate, cyclic_words, SEQUENCES, table
from itertools import product, groupby

SOURCE_LISTS = Path('/home/fmri2404/chiaoen/Chiaoen_AllExpData/Exp3/實驗程式')


def frequency_tables(t, d):
    targets=[];transitions=[];repeats=[]
    within_block=d.block.eq(d.previous_block)
    paired=d[within_block].copy()
    participants=sorted(d.subject.unique())
    for (run,condition),g in d.groupby(['session','rule_group']):
        n_by=g.groupby('subject').size().reindex(participants)
        assert n_by.notna().all()
        for target in range(1,5):
            counts=g[g.keypress.eq(target)].groupby('subject').size().reindex(participants,fill_value=0)
            fractions=counts/n_by
            targets.append(dict(run=run,condition=condition,target=target,n_participants=22,
                n_target=int(counts.sum()),n_trials=int(n_by.sum()),pooled_frequency=float(counts.sum()/n_by.sum()),
                participant_frequency_mean=fractions.mean(),participant_frequency_sd=fractions.std(ddof=1),
                participant_frequency_min=fractions.min(),participant_frequency_max=fractions.max()))
        pairs=paired[(paired.session==run)&(paired.rule_group==condition)]
        n_pairs=pairs.groupby('subject').size().reindex(participants,fill_value=0)
        for origin,destination in product(range(1,5),repeat=2):
            p=pairs[pairs.previous_target.eq(origin)&pairs.keypress.eq(destination)]
            counts=p.groupby('subject').size().reindex(participants,fill_value=0)
            frac=counts/n_pairs
            from_count=int(pairs.previous_target.eq(origin).sum())
            transitions.append(dict(run=run,condition=condition,previous_target=origin,target=destination,
                n_participants=22,n_transition=int(len(p)),n_pairs=int(n_pairs.sum()),
                n_onset_exactly_1s=int(p.adjacent_1s.sum()),
                pooled_frequency=len(p)/n_pairs.sum(),conditional_probability_given_previous=(len(p)/from_count if from_count else np.nan),
                participant_frequency_mean=frac.mean(),participant_frequency_sd=frac.std(ddof=1),
                participant_frequency_min=frac.min(),participant_frequency_max=frac.max()))
        same=g.same_target_previous_slot
        rep=g.immediate_repeat_D1
        eligible=g.adjacent_1s
        repeats.append(dict(run=run,condition=condition,n_participants=22,n_trials=len(g),
            n_adjacent_1s=int(eligible.sum()),n_immediate_repeats_D1=int(rep.sum()),
            n_repeats_crossing_block=int((rep & g.cross_block_boundary).sum()),
            n_gap_separated_target_matches_excluded=int((same & ~eligible).sum()),
            repeat_fraction_of_trials=float(rep.sum()/len(g)),
            repeat_fraction_of_adjacent_1s=float(rep.sum()/eligible.sum())))
    targets=pd.DataFrame(targets);transitions=pd.DataFrame(transitions);repeats=pd.DataFrame(repeats)
    assert len(targets)==80 and len(transitions)==320 and len(repeats)==20
    assert np.allclose(targets.groupby(['run','condition']).pooled_frequency.sum(),1)
    assert np.allclose(transitions.groupby(['run','condition']).pooled_frequency.sum(),1)
    save(targets,t.path/'target_frequencies_by_condition_run.csv')
    save(transitions,t.path/'transition_frequencies_by_condition_run.csv')
    save(repeats,t.path/'D1_repeat_rates_by_condition_run.csv')
    # Row-normalized 4x4 matrices are also exported in a wide format.
    wide=transitions.pivot(index=['run','condition','previous_target'],columns='target',
                           values='conditional_probability_given_previous').reset_index()
    wide.columns=['run','condition','previous_target','P_target1','P_target2','P_target3','P_target4']
    save(wide,t.path/'transition_probability_matrices.csv')
    return targets,transitions,repeats


def word_statistics(t):
    rows=[]
    for order in [2,3]:
        words={k:cyclic_words(v,order) for k,v in SEQUENCES.items()}
        for word in product(range(1,5),repeat=order):
            a,b=words['StrA'].count(word),words['StrB'].count(word)
            membership='shared' if a and b else 'diagnostic_StrA' if a else 'diagnostic_StrB' if b else 'neither'
            rows.append(dict(order=order,word='-'.join(map(str,word)),count_per_cycle_StrA=a,
                count_per_cycle_StrB=b,frequency_StrA=a/8,frequency_StrB=b/8,membership=membership,
                includes_cyclic_wrap_in_definition=True))
    result=pd.DataFrame(rows)
    save(result,t.path/'cyclic_pair_triplet_statistics.csv')
    return result


def verify_sources_and_order(t, d):
    checks=[];orders=[];outer=[];fixations=[];native_cache={}
    for (subject,run),g in d.groupby(['subject','session'],sort=True):
        canonical=t.read(ROOT/f'data/behavioral/{subject[-3:]}_run_{run}_srtt_prob_new.csv')
        assert np.array_equal(canonical.keypress,g.keypress)
        assert np.array_equal(canonical.rule,g.rule)
        source_paths=g.source_file.unique();assert len(source_paths)==1
        native=t.read(source_paths[0]);native_trials=native[native.keypress.notna()]
        assert len(native_trials)==320
        assert np.array_equal(native_trials.keypress,g.keypress)
        assert np.array_equal(native_trials.rule,g.rule)
        origin=float(native['Fixation1.started'].dropna().iloc[0])
        # Validate normalized trial onset clock against all native records.
        assert np.array_equal(np.rint(g.stim_started-origin),g.stim_onset)
        fix=t.read(SOURCE_LISTS/f'fixdur_sub{int(subject[-3:])}_Ses{run}.csv')
        durations=[]
        for index in range(1,10):
            listed=fix[f'fixdur{index}'].dropna().unique()
            logged=native[f'fixdur{index}'].dropna().unique()
            assert len(listed)==len(logged)==1 and abs(listed[0]-logged[0])<1e-9
            assert np.isclose(listed[0],[5.98,7.98,9.98]).any()
            start=native[f'Fixation{index}.started'].dropna().to_numpy()
            end=native[f'Fixation{index}.stopped'].dropna().to_numpy()
            assert len(start)==len(end)==1 and end[0]>start[0]
            durations.append(float(listed[0]))
            fixations.append(dict(subject=subject,run=run,position=index,nominal_duration_s=listed[0]+.02,
                programmed_duration_s=listed[0],logged_routine_duration_s=end[0]-start[0]))
        assert np.isclose(sum(durations),71.82)
        blocks=[];runpatterns=[]
        for block,z in g.groupby('block',sort=False):
            listing=t.read(SOURCE_LISTS/f'condlist_sub{int(subject[-3:])}_{block}.csv')
            assert len(listing)==len(z)==40
            assert np.array_equal(listing.keypress,z.keypress)
            assert np.array_equal(listing.rule,z.rule)
            if 'Ran' in block:
                assert z.rule.eq('Ran').all() and (np.diff(z.keypress)!=0).all()
                blocks.append('Random1' if block.endswith('Ran1') else 'Random2')
            else:
                blockindex=int(block.rsplit('_',1)[1]);blocks.append(f'Structured{blockindex}')
                physical=[];role=[]
                for i in range(5):
                    seq=z.iloc[i*8:(i+1)*8];rule=seq.rule.unique();cond=seq.rule_group.unique()
                    assert len(rule)==len(cond)==1
                    assert seq.keypress.tolist()==SEQUENCES[rule[0]]
                    physical.append(rule[0][-1]);role.append(cond[0])
                assert role.count('Str1')==[5,5,4,3,2,1,0,0][run-1]
                assert role.count('Str2')==[0,0,1,2,3,4,5,5][run-1]
                if run in [3,4,5,6]:
                    minority='Str1' if role.count('Str1')<role.count('Str2') else 'Str2'
                    assert max([len(list(x)) for k,x in groupby(role) if k==minority])<=2
                p=''.join(physical);rp='-'.join(role)
                runpatterns.append(rp)
                orders.append(dict(subject=subject,run=run,structured_block=blockindex,
                    physical_sequence_order=p,role_sequence_order=rp))
        expected=['Random1']+[f'Structured{i}' for i in range(1,7)]+['Random2']
        assert blocks==expected
        if run in [3,4,5,6]:
            assert all(a!=b for a,b in zip(runpatterns[:-1],runpatterns[1:]))
        outer.append(dict(subject=subject,run=run,outer_block_order=' > '.join(blocks),
            full_role_order=' | '.join(runpatterns),n_distinct_structured_orders=len(set(runpatterns)),
            n_adjacent_identical_structured_orders=sum(a==b for a,b in zip(runpatterns[:-1],runpatterns[1:])),
            fixation_nominal_pattern='-'.join(str(int(x+.02)) for x in durations)))
        checks.append(dict(run=run,n_participants=1,n_blocks=8,n_target_rule_trial_matches=320,
                           n_fixation_values_matching=9,n_native_trial_matches=320))
        # Representative selected before behavior results: first sorted participant, run 3.
        if subject==sorted(d.subject.unique())[0] and run==3:
            native_cache=dict(native=native,raw=g.copy(),origin=origin)
    group_checks=pd.DataFrame(checks).groupby('run').sum().reset_index()
    save(group_checks,t.path/'source_match_checks_by_run.csv')
    order=pd.DataFrame(orders);outer=pd.DataFrame(outer);fixations=pd.DataFrame(fixations)
    for col,name in [('physical_sequence_order','physical_subblock_order_frequencies.csv'),
                     ('role_sequence_order','role_subblock_order_frequencies.csv')]:
        summary=order.groupby(['run','structured_block',col]).size().rename('n_participants').reset_index()
        summary['fraction_of_22_participants']=summary.n_participants/22
        save(summary,t.path/name)
    patterns=outer.groupby(['run','outer_block_order']).size().rename('n_participant_runs').reset_index()
    save(patterns,t.path/'outer_block_order_frequencies.csv')
    for col,name in [('full_role_order','whole_run_role_order_frequencies.csv'),
                     ('fixation_nominal_pattern','fixation_pattern_frequencies.csv')]:
        save(outer.groupby(['run',col]).size().rename('n_participants').reset_index(),t.path/name)
    summaries=[]
    for run,g in outer.groupby('run'):
        summaries.append(dict(run=run,n_participants=22,n_outer_block_orders=g.outer_block_order.nunique(),
            n_whole_run_role_orders=g.full_role_order.nunique(),n_fixation_duration_patterns=g.fixation_nominal_pattern.nunique(),
            distinct_subblock_orders_per_participant_mean=g.n_distinct_structured_orders.mean(),
            distinct_subblock_orders_per_participant_min=g.n_distinct_structured_orders.min(),
            distinct_subblock_orders_per_participant_max=g.n_distinct_structured_orders.max(),
            adjacent_identical_structured_orders_total=g.n_adjacent_identical_structured_orders.sum()))
    summaries=pd.DataFrame(summaries);save(summaries,t.path/'order_variability_by_run.csv')
    fixsum=fixations.groupby(['run','position']).agg(n_participants=('nominal_duration_s','size'),
        nominal_duration_mean_s=('nominal_duration_s','mean'),nominal_duration_min_s=('nominal_duration_s','min'),
        nominal_duration_max_s=('nominal_duration_s','max'),programmed_duration_mean_s=('programmed_duration_s','mean'),
        logged_routine_duration_mean_s=('logged_routine_duration_s','mean')).reset_index()
    save(fixsum,t.path/'fixation_duration_summary.csv')
    save(fixations.groupby(['run','position','nominal_duration_s']).size().rename('n_participants').reset_index(),
         t.path/'fixation_duration_frequencies.csv')
    return native_cache,group_checks,summaries


def representative_timeline(t, cache):
    native,d,z=cache['native'],cache['raw'],cache['origin']
    rows=[]
    for i in range(1,10):
        start=float(native[f'Fixation{i}.started'].dropna().iloc[0])-z
        end=float(native[f'Fixation{i}.stopped'].dropna().iloc[0])-z
        nominal=float(native[f'fixdur{i}'].dropna().iloc[0])+.02
        rows.append(dict(kind='fixation',label=f'Fixation {i}',condition='',
            start_s=round(start),end_s=round(end),logged_start_s=start,logged_end_s=end,
            nominal_fixation_s=nominal,basis='logged routine times rounded to nearest second'))
    for block,g in d.groupby('block',sort=False):
        groups=[g] if g.rule_group.eq('Ran').all() else [g.iloc[i:i+8] for i in range(0,40,8)]
        for j,c in enumerate(groups,1):
            breaks=np.r_[0,np.flatnonzero(np.diff(c.stim_onset.to_numpy())!=1)+1,len(c)]
            for startidx,endidx in zip(breaks[:-1],breaks[1:]):
                section=c.iloc[startidx:endidx]
                label=('Random 1' if block.endswith('Ran1') else 'Random 2') if g.rule_group.eq('Ran').all() else (
                    f'Structured {block.rsplit("_",1)[1]}, sequence repetition {j}')
                rows.append(dict(kind='task_slots',label=label,condition=section.rule_group.iloc[0],
                    start_s=int(section.stim_onset.iloc[0]),end_s=int(section.stim_onset.iloc[-1])+1,
                    logged_start_s=float(section.stim_started.iloc[0])-z,logged_end_s=np.nan,
                    nominal_fixation_s=np.nan,basis='logged target onset rounded to second, one-second trial slots'))
    trials=native[native.keypress.notna()]
    for _,row in trials.iterrows():
        start=float(row['Trig_press.started'])-z;end=float(row['Trig_press.stopped'])-z
        if end-start>.1 and round(end)>round(start):
            rows.append(dict(kind='trigger_wait',label='Trigger wait',condition='',start_s=round(start),end_s=round(end),
                logged_start_s=start,logged_end_s=end,nominal_fixation_s=np.nan,basis='logged trigger routine, rounded to second'))
    rows=sorted(rows,key=lambda x:x['start_s'])
    assert rows[0]['start_s']==0
    assert all(a['end_s']==b['start_s'] for a,b in zip(rows[:-1],rows[1:])), rows
    last=rows[-1]['end_s']
    assert last<=400
    rows.append(dict(kind='remaining_acquisition_window',label='After logged task/final fixation',condition='',
        start_s=last,end_s=400,logged_start_s=np.nan,logged_end_s=np.nan,nominal_fixation_s=np.nan,
        basis='400-second acquisition boundary; no inferred stimulus or fixation'))
    for row in rows:row['duration_s']=row['end_s']-row['start_s'];row['example_run']=3
    out=pd.DataFrame(rows)
    assert out.duration_s.sum()==400
    assert out[out.kind.eq('task_slots')].duration_s.sum()==320
    assert out[out.kind.eq('fixation')].duration_s.sum()==72
    save(out,t.path/'representative_run_timeline.csv')
    totals=out.groupby('kind').duration_s.sum().reset_index()
    save(totals,t.path/'representative_run_timeline_totals.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors={'Ran':'#858585','Str1':'#3876a8','Str2':'#b65b7a',
            'fixation':'#dfdfdf','trigger_wait':'#e1a438','remaining_acquisition_window':'#fafafa'}
    fig,ax=plt.subplots(figsize=(13,2.7))
    for row in out.itertuples():
        color=colors[row.condition] if row.kind=='task_slots' else colors[row.kind]
        ax.broken_barh([(row.start_s,row.duration_s)],(0,1),facecolors=color,edgecolors='white',linewidth=.45)
    ax.set(xlim=(0,400),ylim=(-.08,1.08),xlabel='Seconds from first fixation start',yticks=[],
        title='Representative interleaved run (run 3): logged task slots, fixation and trigger waits')
    ax.set_xticks(np.arange(0,401,20));ax.grid(axis='x',alpha=.15)
    ax.spines[['top','left','right']].set_visible(False)
    labels={'Ran':'Random','Str1':'Structure 1','Str2':'Structure 2','fixation':'Fixation',
            'trigger_wait':'Trigger wait','remaining_acquisition_window':'Remaining acquisition window'}
    ax.legend(handles=[Patch(facecolor=colors[k],label=v) for k,v in labels.items()],
        loc='upper center',bbox_to_anchor=(.5,-.37),ncol=6,frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(t.path/'representative_run_timeline.pdf',bbox_inches='tight',metadata={'CreationDate':None})
    fig.savefig(t.path/'representative_run_timeline.png',bbox_inches='tight',dpi=180)
    plt.close(fig)
    return totals


def design_statistics():
    t=Task(BEHAV/'B3_design_statistics')
    for name in ['stimulus_generator.py','program_archive_executable.py','time_conversion.R']:
        t.record(ROOT/'results/glm_volume/reanalysis_20261018/source_discovery/source_snapshots'/name)
    d=annotate(t.read(OLD_B/'source_replay/raw_trials_N22.csv'))
    targets,transitions,repeats=frequency_tables(t,d)
    words=word_statistics(t)
    cache,source_checks,variability=verify_sources_and_order(t,d)
    totals=representative_timeline(t,cache)
    shared=words[words.membership.eq('shared')]
    counts=repeats.groupby('condition').sum(numeric_only=True).reset_index()
    t.finish('# B3: realized target statistics, sequence overlap and task order\n\n'
        'N=22, all 56,320 logged trials before response cleaning. Only group-level design '
        'outputs are exported here. Targets are `keypress`; physical sequence identity is '
        '`rule` (StrA/StrB), counterbalanced condition is `rule_group` (Str1/Str2). '
        'All 1,408 original 40-trial condition lists match logged targets and physical rules '
        'exactly, as do the 176 canonical and native trial files. All 1,584 fixation values '
        'match their source lists. Trial onsets reproduce rounded native stimulus start '
        'minus the first fixation start for every trial. Source paths and hashes are recorded '
        'in the input manifest; no individual sequence/timing data table is exported.\n\n'
        'Target frequencies retain four targets per run/condition (80 rows). First-order '
        'transitions are consecutive target slots **within each 40-trial condition list**, '
        'assigned to the current trial condition, retaining all 16 ordered pairs per '
        'run/condition (320 rows, including zeros). Trial-order transitions across an internal '
        'trigger gap remain part of the list’s realized design; `n_onset_exactly_1s` identifies '
        'the temporally adjacent subset. Fixation-separated block boundaries are excluded. '
        'Pooled pair frequencies, row-normalized conditional matrices and the distribution '
        'of participant frequencies are exported. D1 repeat rates use exactly-1-s onset '
        'differences, irrespective of block labels, and separately report gap matches and '
        'cross-block repeats. Repeat fractions use both all trials and eligible adjacent '
        'pairs as denominators.\n\n'
        +table(counts,['condition','n_trials','n_immediate_repeats_D1','n_repeats_crossing_block',
                       'n_gap_separated_target_matches_excluded'])
        +'\n\nThe physical cyclic sequences are A=3-1-1-2-3-4-2-4 and '
        'B=3-4-2-1-1-3-2-4. Pair and triplet enumerations include wrap-around; all '
        '4²+4³ words are retained, with multiplicities per eight-item cycle, frequencies '
        'and shared/diagnostic/neither labels. Shared pairs/triplets:\n\n'
        +table(shared,['order','word','count_per_cycle_StrA','count_per_cycle_StrB'])
        +'\n\nThe outer block order is fixed in all 176 runs: Random1 → Structured1–6 → Random2. '
        'Each structured block comprises five eight-trial repetitions; relative Str1/Str2 '
        'counts per block are 5/0, 5/0, 4/1, 3/2, 2/3, 1/4, 0/5, 0/5 across runs 1–8. '
        'Odd/even participant assignment swaps A/B as Str1/Str2. The source shuffles physical '
        'repetition order within those counts, prohibits more than two consecutive minority '
        'sequences inside a block, and forbids identical adjacent structured-block lists in '
        'runs 3–6; all logged lists satisfy those constraints. Exported group pattern '
        'frequencies distinguish physical A/B order, Str1/Str2 order and nine-fixation '
        'duration patterns. Distinct whole-run role orders and per-participant diversity '
        'are aggregated by run:\n\n'
        +table(variability,['run','n_outer_block_orders','n_whole_run_role_orders',
            'n_fixation_duration_patterns','distinct_subblock_orders_per_participant_min',
            'distinct_subblock_orders_per_participant_max'])
        +'\n\nThe representative timeline is selected deterministically as the first sorted '
        'participant’s first interleaved run (run 3), independent of behavioral outcomes. '
        'It displays all 30 structured repetitions, both random blocks, nine fixations, '
        'trigger waits and the remaining acquisition window. Times are rounded native '
        'routine times/trial onsets relative to first fixation start, matching the imaging '
        'event clock. The timeline tiles 0–400 s without overlap or gaps; its breakdown is:\n\n'
        +table(totals,['kind','duration_s'])
        +'\n\nThe generator programs each fixation at nominal 6/8/10 s minus .02 s; '
        'the nine nominal durations sum to 72 s in every run (programmed sum 71.82 s). '
        '320 one-second task slots plus nominal fixations account for 392 s; trigger waits '
        'and time following the logged task explain the rest of this example. Counts vary '
        'with observed trigger alignment, so this example is not asserted to be every '
        'run’s timeline. Fixation routine timing is not visual cue exposure, and the '
        'remaining acquisition window is not labeled as fixation. RNG seed/state used '
        'to generate the original lists is unavailable; realized lists, rather than '
        'a newly simulated generator, supply these statistics. No causal separation '
        'of prevalence, familiarity, practice or local transition structure is claimed.\n\n'
        'Deviation/clarification: D1 supplies the temporal repeat definition. No other deviation.\n',
        dict(n_participants=22,n_runs=176,n_condition_lists=1408,n_trials_matched=56320,
            n_native_trials_matched=56320,n_fixation_values_matched=1584,
            n_target_frequency_rows=len(targets),n_transition_frequency_rows=len(transitions),
            n_cyclic_word_rows=len(words),all_original_order_constraints_pass=True,
            all_D1_repeats_crossing_blocks=int(repeats.n_repeats_crossing_block.sum()),
            n_D1_repeats=int(repeats.n_immediate_repeats_D1.sum()),
            representative_timeline_tiles_400s=True,representative_task_slots_320s=True,
            group_level_only_outputs=True))
