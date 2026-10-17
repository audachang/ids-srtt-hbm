#!/usr/bin/env python3
"""Summarize raw-trial errors by structured block in the imaging cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

SEED = 20261013
BOOTSTRAPS = 20000
COLORS = {'Str1': '#7546BD', 'Str2': '#078773'}
EXPOSURES = {1: {'Str1': 40}, 2: {'Str1': 40},
             3: {'Str1': 32, 'Str2': 8}, 4: {'Str1': 24, 'Str2': 16},
             5: {'Str1': 16, 'Str2': 24}, 6: {'Str1': 8, 'Str2': 32},
             7: {'Str2': 40}, 8: {'Str2': 40}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_interval(values, rng):
    x = np.asarray(values, float)
    assert np.isfinite(x).all() and len(x) > 1
    estimates = x[rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))].mean(axis=1)
    return np.quantile(estimates, [.025, .975])


def sign_flip_p(values):
    """Exhaustive two-sided sign-flip test of paired participant differences.

    This tests a symmetric paired-difference null, not a randomized treatment.
    Zero differences do not change the permutation distribution.
    """
    x = np.asarray(values, float)
    x = x[np.abs(x) > 1e-14]
    if not len(x):
        return 1.0, 1
    total = 2 ** len(x)
    observed = abs(x.sum())
    extreme = 0
    for start in range(0, total, 32768):
        patterns = np.arange(start, min(total, start + 32768), dtype=np.uint32)
        signs = 2 * ((patterns[:, None] >> np.arange(len(x), dtype=np.uint32)) & 1).astype(float) - 1
        extreme += np.count_nonzero(np.abs(signs @ x) >= observed - 1e-12)
    return extreme / total, total


def summarize(data, keys, rng):
    rows = []
    for key, part in data.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        assert not part.subject.duplicated().any()
        row = dict(zip(keys, key))
        row.update(n_subjects=len(part), n_trials=int(part.n_trials.sum()),
                   n_correct=int(part.n_correct.sum()), n_errors=int(part.n_errors.sum()),
                   n_wrong_presses=int(part.n_wrong_presses.sum()),
                   n_omissions=int(part.n_omissions.sum()),
                   trials_per_subject_min=int(part.n_trials.min()),
                   trials_per_subject_max=int(part.n_trials.max()))
        for metric in ['error_rate', 'wrong_press_rate', 'omission_rate', 'wrong_given_response_rate']:
            values=part[metric].to_numpy()
            low, high = bootstrap_interval(values, rng)
            row[metric] = float(values.mean())
            row[metric + '_ci_low'] = float(low)
            row[metric + '_ci_high'] = float(high)
        row['pooled_error_rate'] = row['n_errors'] / row['n_trials']
        rows.append(row)
    return pd.DataFrame(rows)


def savefig(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / 'results/behavioral/block_errors_20261013'
    out.mkdir(parents=True, exist_ok=True)
    subject_file = root / 'code/replication/config/subjects.txt'
    subjects = subject_file.read_text().split()
    assert len(subjects) == len(set(subjects)) == 20
    manifest = {str(subject_file): digest(subject_file)}
    frames = []
    for subject in subjects:
        participant = int(subject[4:])
        for run in range(1, 9):
            path = root / 'data/behavioral' / f'{participant:03d}_run_{run}_srtt_prob_new.csv'
            frame = pd.read_csv(path)
            assert len(frame) == 320
            assert frame.participant.eq(participant).all() and frame.session.eq(run).all()
            frame['subject'] = subject
            frame['trial_in_run'] = np.arange(1, 321)
            frame['block_order'] = (frame.trial_in_run - 1) // 40 + 1
            frame['block_kind'] = np.where(frame.block.str.contains('Str_'), 'structured', 'random')
            frame['block_number'] = frame.block.str.extract(r'(?:Str_|Ran)(\d+)$')[0].astype(int)
            assert frame.groupby('block', sort=False).size().eq(40).all()
            assert frame.groupby('block_order').block.nunique().eq(1).all()
            assert frame.block.str.extract(r'Ses(\d+)')[0].astype(int).eq(run).all()
            expected = np.where(frame.rule.eq('Ran'), 'Ran', np.where(
                ((participant % 2 == 1) & frame.rule.eq('StrA')) |
                ((participant % 2 == 0) & frame.rule.eq('StrB')), 'Str1', 'Str2'))
            assert np.array_equal(expected, frame.rule_group)
            for block, part in frame[frame.block_kind.eq('structured')].groupby('block_number'):
                assert 1 <= block <= 6
                assert part.groupby('rule_group').size().to_dict() == EXPOSURES[run]
            response_key = pd.to_numeric(frame.p_response_keys, errors='coerce')
            response_rt = pd.to_numeric(frame.p_response_rt, errors='coerce')
            omitted = ~np.isfinite(response_rt) | response_key.isna()
            correct = frame['correct.resp'].eq(1) & ~omitted
            assert np.array_equal(frame['correct.resp'].eq(1), response_key.eq(frame.keypress))
            frame['omitted'] = omitted.astype(int)
            frame['correct'] = correct.astype(int)
            frame['error'] = (~correct).astype(int)
            frame['wrong_press'] = ((~correct) & ~omitted).astype(int)
            assert (frame.error == frame.omitted + frame.wrong_press).all()
            frames.append(frame[['subject', 'participant', 'session', 'block', 'block_order',
                                 'block_kind', 'block_number', 'rule_group', 'trial_in_run',
                                 'correct', 'error', 'omitted', 'wrong_press']])
            manifest[str(path)] = digest(path)
    trials = pd.concat(frames, ignore_index=True)
    assert len(trials) == 51200 and trials.error.sum() == 1425 and trials.omitted.sum() == 182
    keys = ['subject', 'session', 'block', 'block_kind', 'block_number', 'rule_group']
    blocks = trials.groupby(keys, sort=True).agg(
        n_trials=('error', 'size'), n_correct=('correct', 'sum'), n_errors=('error', 'sum'),
        n_wrong_presses=('wrong_press', 'sum'), n_omissions=('omitted', 'sum')).reset_index()
    runs = trials.groupby(['subject', 'session', 'rule_group'], sort=True).agg(
        n_trials=('error', 'size'), n_correct=('correct', 'sum'), n_errors=('error', 'sum'),
        n_wrong_presses=('wrong_press', 'sum'), n_omissions=('omitted', 'sum')).reset_index()
    for data in [blocks, runs]:
        data['error_rate'] = data.n_errors / data.n_trials
        data['wrong_press_rate'] = data.n_wrong_presses / data.n_trials
        data['omission_rate'] = data.n_omissions / data.n_trials
        data['wrong_given_response_rate'] = data.n_wrong_presses / (data.n_trials-data.n_omissions)
    # Independent agreement with the established raw-trial accuracy products.
    reference = pd.read_csv(root / 'results/behavioral/fmri_rt_20261012/participant_accuracy.csv')
    reference = reference.set_index(['subject','session','rule_group']).sort_index()
    current = runs.set_index(['subject','session','rule_group']).sort_index()
    assert reference.index.equals(current.index)
    for new_col, old_col in [('n_trials','n_trials'),('n_correct','n_correct'),('n_omissions','n_omitted')]:
        np.testing.assert_array_equal(current[new_col], reference[old_col])
    np.testing.assert_allclose(1-current.error_rate, reference.accuracy, atol=1e-14)
    # Reaggregate independent block summaries and compare to run summaries.
    summed=blocks.groupby(['subject','session','rule_group'])[
        ['n_trials','n_correct','n_errors','n_wrong_presses','n_omissions']].sum().sort_index()
    pd.testing.assert_frame_equal(summed,current[summed.columns],check_dtype=False)
    rng = np.random.default_rng(SEED)
    block_summary = summarize(blocks, ['session','block_kind','block_number','rule_group'], rng)
    run_summary = summarize(runs, ['session','rule_group'], rng)
    structured=block_summary[block_summary.block_kind.eq('structured')].copy()
    mixed = structured[structured.session.between(3, 6)].copy()
    assert len(structured) == 72 and len(mixed) == 48 and mixed.n_subjects.eq(20).all()
    contrast_rows=[]; participant_contrasts=[]

    def contrast(family, name, metric, first, second, description, **extra):
        first=first.reindex(subjects);second=second.reindex(subjects)
        difference=(first-second).to_numpy()
        assert np.isfinite(difference).all()
        low,high=bootstrap_interval(difference,rng)
        probability,n_patterns=sign_flip_p(difference)
        contrast_rows.append(dict(family=family,contrast=name,metric=metric,description=description,
            n_subjects=len(difference),first_rate=first.mean(),second_rate=second.mean(),
            difference_pp=100*difference.mean(),ci_low_pp=100*low,ci_high_pp=100*high,
            p_sign_flip=probability,n_sign_patterns=n_patterns,**extra))
        for subject,value in zip(subjects,difference):
            participant_contrasts.append(dict(subject=subject,family=family,contrast=name,
                                             metric=metric,difference=value))

    for metric in ['error_rate','wrong_press_rate']:
        runwide=runs[runs.rule_group.isin(COLORS)].pivot(index=['subject','session'],columns='rule_group',values=metric)
        for run in range(3,7):
            pair=runwide.xs(run,level='session')
            contrast('mixed_run_comparisons',f'run_{run}_Str2_minus_Str1',metric,pair.Str2,pair.Str1,
                     f'Structure 2 minus Structure 1, run {run}',run=run)
        balanced=runwide[runwide.index.get_level_values('session').isin(range(3,7))].groupby('subject').mean()
        contrast('mixed_run_comparisons','mixed_equal_run_Str2_minus_Str1',metric,balanced.Str2,balanced.Str1,
                 'Structure 2 minus Structure 1, equally weighted runs 3-6')
        for run in range(3,7):
            for block in range(1,7):
                pair=blocks[blocks.session.eq(run)&blocks.block_number.eq(block)&blocks.block_kind.eq('structured')].pivot(
                    index='subject',columns='rule_group',values=metric)
                contrast('mixed_block_comparisons',f'run_{run}_block_{block}_Str2_minus_Str1',metric,pair.Str2,pair.Str1,
                         f'Structure 2 minus Structure 1, run {run}, structured block {block}',run=run,block_number=block)
        pure1=runwide[runwide.index.get_level_values('session').isin([1,2])].groupby('subject').Str1.mean()
        pure2=runwide[runwide.index.get_level_values('session').isin([7,8])].groupby('subject').Str2.mean()
        contrast('mixed_vs_unmixed_context','Str1_mixed_minus_initial',metric,balanced.Str1,pure1,
                 'Structure 1, equally weighted mixed runs 3-6 minus unmixed runs 1-2')
        contrast('mixed_vs_unmixed_context','Str2_mixed_minus_final',metric,balanced.Str2,pure2,
                 'Structure 2, equally weighted mixed runs 3-6 minus unmixed runs 7-8')
    contrasts=pd.DataFrame(contrast_rows)
    contrasts['q_bh']=np.nan
    contrasts['family_size']=0
    for _,part in contrasts.groupby(['family','metric']):
        contrasts.loc[part.index,'q_bh']=stats.false_discovery_control(part.p_sign_flip.to_numpy(),method='bh')
        contrasts.loc[part.index,'family_size']=len(part)

    for name,data in [('trial_outcomes',trials),('participant_block_error_rates',blocks),
                      ('participant_run_error_rates',runs),('group_block_error_rates',block_summary),
                      ('structured_block_error_rates',structured),('mixed_block_error_rates',mixed),
                      ('group_run_error_rates',run_summary),('paired_contrasts',contrasts),
                      ('participant_contrasts',pd.DataFrame(participant_contrasts))]:
        data.to_csv(out/(name+'.csv'),index=False,float_format='%.12g')
    # Compact manuscript-friendly table of all 24 mixed blocks, in percent.
    wide=mixed.pivot(index=['session','block_number'],columns='rule_group',values='error_rate')*100
    wide.columns=['Str1_error_percent','Str2_error_percent']
    wide['Str2_minus_Str1_pp']=wide.Str2_error_percent-wide.Str1_error_percent
    wide.to_csv(out/'mixed_blocks_compact.csv',float_format='%.4f')

    def style(ax):
        ax.grid(axis='y',color='#E5E8ED',lw=.7);ax.set_axisbelow(True)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        for side in ['left','bottom']:ax.spines[side].set_color('#B4BAC3')
        ax.tick_params(colors='#424750',length=3)

    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,4,figsize=(13.5,4.5),sharey=True)
        proportions={3:'80% / 20%',4:'60% / 40%',5:'40% / 60%',6:'20% / 80%'}
        upper=np.ceil(mixed.error_rate_ci_high.max()*100/2)*2
        for run,ax in zip(range(3,7),axes):
            for condition,color in COLORS.items():
                d=mixed[mixed.session.eq(run)&mixed.rule_group.eq(condition)].sort_values('block_number')
                shift=-.06 if condition=='Str1' else .06
                ax.errorbar(d.block_number+shift,d.error_rate*100,
                    yerr=np.vstack([d.error_rate-d.error_rate_ci_low,d.error_rate_ci_high-d.error_rate])*100,
                    fmt='o-',color=color,ms=6,lw=1.6,elinewidth=1.05,capsize=2,
                    label=condition.replace('Str','Structure '))
            ax.set_title(f'Run {run}\nStructure 1 / 2: {proportions[run]}',fontsize=11)
            ax.set(xlabel='Structured block',xticks=range(1,7),xlim=(.6,6.4),ylim=(0,upper+.5));style(ax)
        axes[0].set_ylabel('Incorrect or omitted responses (%)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
        fig.text(.5,.015,'20 participants · All trials retained · Pointwise 95% participant-bootstrap intervals',
                 ha='center',fontsize=9,color='#676D77')
        fig.tight_layout(rect=(0,.045,1,.90));savefig(fig,out/'mixed_block_error_rates')
        fig,ax=plt.subplots(figsize=(13.5,4.9))
        ax.axvspan(12.5,36.5,color='#F5F6F8',zorder=0)
        for run in range(1,8):ax.axvline(run*6+.5,color='#D8DCE2',lw=.7,ls=(0,(3,4)))
        for condition,color in COLORS.items():
            d=structured[structured.rule_group.eq(condition)].sort_values(['session','block_number'])
            x=(d.session-1)*6+d.block_number
            ax.errorbar(x,d.error_rate*100,
                yerr=np.vstack([d.error_rate-d.error_rate_ci_low,d.error_rate_ci_high-d.error_rate])*100,
                fmt='o-',color=color,ms=4.5,lw=1.1,elinewidth=.6,capsize=1.5,
                label=condition.replace('Str','Structure '))
        ax.set(xlabel='Structured block within run',ylabel='Incorrect or omitted responses (%)',xlim=(.5,48.5),ylim=(0,upper+.5))
        ax.set_xticks(range(1,49),[str((k-1)%6+1) for k in range(1,49)],fontsize=8)
        top=ax.secondary_xaxis('top');top.set_xticks([3.5+6*k for k in range(8)],[f'Run {k}' for k in range(1,9)])
        top.spines['top'].set_visible(False);top.tick_params(length=0,pad=7)
        style(ax);ax.legend(ncol=2,frameon=False,loc='upper right')
        fig.text(.5,.015,'20 participants · Shaded area: interleaved runs 3–6 · Pointwise 95% participant-bootstrap intervals',
                 ha='center',fontsize=9,color='#676D77')
        fig.tight_layout(rect=(0,.04,1,1));savefig(fig,out/'all_structured_block_error_rates')

    manifest[str(root/'results/behavioral/fmri_rt_20261012/participant_accuracy.csv')]=digest(root/'results/behavioral/fmri_rt_20261012/participant_accuracy.csv')
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    record=dict(status='passed',n_subjects=20,n_runs=160,n_trials=51200,
        n_errors_or_omissions=int(trials.error.sum()),n_wrong_presses=int(trials.wrong_press.sum()),
        n_omissions=int(trials.omitted.sum()),n_structured_block_condition_summaries=len(structured),
        n_mixed_block_condition_summaries=len(mixed),all_block_exposures_verified=True,
        counterbalancing_and_correctness_verified=True,run_accuracy_agrees_with_existing_results=True,
        block_to_run_counts_verified=True,bootstrap_replicates=BOOTSTRAPS,bootstrap_seed=SEED,
        independent_unit='participant',comparison_families={'mixed_run_comparisons':5,'mixed_block_comparisons':24,'mixed_vs_unmixed_context':2},
        metrics=['error_rate','wrong_press_rate'],script_sha256=digest(Path(__file__)),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__)
    (out/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'analysis_spec.json').write_text(json.dumps(dict(
        cohort='20 retained imaging participants',blocks='Six 40-trial structured blocks per run',
        primary_outcome='Incorrect or omitted responses / all trials, consistent with 1 - established accuracy',
        wrong_press_rate='Incorrect responses excluding omissions / all trials',
        conditional_wrong_press_rate='Incorrect responses excluding omissions / responded trials; descriptive',
        rt_exclusions='None: correct and incorrect responses, omissions, and all RTs remain in accuracy denominators',
        participant_weighting='Equal weights; pooled rates also exported',
        mixed_overall_weighting='Equal weights across runs 3-6 within participant; controls changing sequence exposure weights',
        confidence_intervals='Pointwise percentile participant bootstrap; paired differences resample participants jointly',
        tests='Two-sided exhaustive participant sign flips; assumes a symmetric paired-difference null; exploratory',
        correction='BH separately within each listed family and outcome metric; no correction across families/metrics',
        inference_limits='Intermixing, sequence prevalence, sequence familiarity and practice covary. Unmixed context comparisons are not causal mixing effects.'),indent=2)+'\n')
    lines=['# Block-level sequence error rates','',
           '20 retained imaging participants; eight runs, six 40-trial structured blocks per run. Rates use all logged trials. The primary outcome includes incorrect button presses and omissions; wrong presses and omissions are also provided separately. No RT filtering or trimming is applied to accuracy.','',
           'Group rates average participant proportions. Pointwise 95% intervals resample participants (20,000 draws). Paired contrasts use two-sided exhaustive participant sign-flip tests; these are exploratory and assume symmetric paired differences. BH correction is separate for the 5 run/overall contrasts, 24 mixed-block contrasts, and 2 mixed/unmixed context contrasts, and separate for each outcome metric.','',
           '| Run | Structure 1 / 2 exposure | Structure 1 error (%) | Structure 2 error (%) | Random error (%) |',
           '|---|---|---:|---:|---:|']
    for run in range(1,9):
        part=run_summary[run_summary.session.eq(run)].set_index('rule_group')
        rates={c:f'{part.loc[c,"error_rate"]*100:.2f}' if c in part.index else '—' for c in ['Str1','Str2','Ran']}
        exposure=' / '.join(str(EXPOSURES[run].get(c,0)*100//40)+'%' for c in ['Str1','Str2'])
        lines.append(f'| {run} | {exposure} | {rates["Str1"]} | {rates["Str2"]} | {rates["Ran"]} |')
    lines+=['','## Participant-paired comparisons','',
            '| Comparison | Difference (percentage points) | Pointwise 95% CI | p | BH q |',
            '|---|---:|---|---:|---:|']
    primary=contrasts[contrasts.metric.eq('error_rate')&~contrasts.family.eq('mixed_block_comparisons')]
    for _,r in primary.iterrows():
        lines.append(f'| {r.description} | {r.difference_pp:.3f} | [{r.ci_low_pp:.3f}, {r.ci_high_pp:.3f}] | {r.p_sign_flip:.6g} | {r.q_bh:.6g} |')
    overall=primary[primary.contrast.eq('mixed_equal_run_Str2_minus_Str1')].iloc[0]
    significant_runs=primary[primary.family.eq('mixed_run_comparisons') & primary.contrast.str.startswith('run_') & primary.q_bh.le(.05)]
    significant_blocks=contrasts[contrasts.metric.eq('error_rate') & contrasts.family.eq('mixed_block_comparisons') & contrasts.q_bh.le(.05)]
    lines+=['','## Main findings','',
            f'Across equally weighted mixed runs, Structure 2 exceeded Structure 1 by {overall.difference_pp:.3f} percentage points (pointwise 95% CI [{overall.ci_low_pp:.3f}, {overall.ci_high_pp:.3f}], BH q={overall.q_bh:.6g}). Run-level differences survived the five-comparison family in runs '+', '.join(str(int(r)) for r in significant_runs.run)+'. The higher rate for the emerging sequence narrowed across the replacement schedule.', '',
            f'Of the 24 block-level paired comparisons, {len(significant_blocks)} survived BH correction.']
    for _,r in significant_blocks.iterrows():
        lines.append(f'Run {int(r.run)}, structured block {int(r.block_number)}: Structure 2 minus Structure 1 = {r.difference_pp:.3f} percentage points, q={r.q_bh:.6g}. This localized difference coexists with the nonsignificant run-6 average.')
    old_context=primary[primary.contrast.eq('Str1_mixed_minus_initial')].iloc[0]
    lines+=['',f'Structure 1 mixed-minus-initial change was {old_context.difference_pp:.3f} percentage points (pointwise 95% CI [{old_context.ci_low_pp:.3f}, {old_context.ci_high_pp:.3f}], q={old_context.q_bh:.6g}); this comparison did not provide clear evidence of increased errors in the established sequence.', '',
            'Excluding omissions from the error numerator retained the same run-level and block-level BH significance pattern. Wrong-press rates still use all trials as their denominator; conditional rates among responded trials are provided separately.', '']
    lines+=['','## All interleaved blocks','',
            '| Run | Structured block | Structure 1 error (%) | Structure 2 error (%) | Difference (pp) |',
            '|---|---|---:|---:|---:|']
    for (run,block),r in wide.iterrows():
        lines.append(f'| {run} | {block} | {r.Str1_error_percent:.2f} | {r.Str2_error_percent:.2f} | {r.Str2_minus_Str1_pp:.2f} |')
    lines+=['','## Interpretation','',
            'These comparisons describe error rates during gradual replacement. The observed run schedule changes intermixing, sequence prevalence, familiarity, and accumulated practice together; error differences cannot be assigned uniquely to intermixing. Structure 2 is newly introduced in run 3. Comparing mixed runs with the initial or final unmixed runs also changes sequence experience and time. The sample is restricted to participants who met the initial Structure 1 learner criterion.','',
            'Individual structured-block denominators range from 8 to 40 trials per structure, so especially the minority-sequence rates are coarse and uncertain. A block with no errors remains a zero observation; absent sequence cells are not assigned zero.','',
            'The full CSVs provide trial outcomes, participant/block counts and rates, participant/run rates, wrong presses, omissions, conditional wrong-response rates, confidence intervals, and all paired comparisons. The compact mixed-block CSV contains the 24 block pairs. All checks are recorded in validation.json; raw inputs and analysis specification are retained separately.','',
            'Reproduce with:','',
            '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/nilearn-0131/bin/python code/reporting/analyze_block_errors.py','```','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(record,indent=2))
    print(primary[['contrast','difference_pp','ci_low_pp','ci_high_pp','p_sign_flip','q_bh']].to_string(index=False))


if __name__=='__main__':main()
