#!/usr/bin/env python3
"""Plot participant error rates in the RT style and pair both in Figure 1."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'Ran': '#616873', 'Str1': '#7546BD', 'Str2': '#078773'}
LABELS = {'Ran': 'Random', 'Str1': 'Structure 1', 'Str2': 'Structure 2'}
OFFSETS = {'Ran': -.22, 'Str1': 0., 'Str2': .22}
PHASES = [(1.5, 'Initial sequence'), (4.5, 'Interleaved sequences'), (7.5, 'New sequence')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def savefig(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_panel(ax, group, participants, ylabel, limits, ticks):
    jitter = dict(zip(sorted(participants.subject.unique()), np.linspace(-.065, .065, 20)))
    ax.axvspan(2.5, 6.5, color='#F5F6F8', zorder=0)
    for boundary in [2.5, 6.5]:
        ax.axvline(boundary, color='#D8DCE2', lw=.9, ls=(0, (3, 4)), zorder=1)
    handles = []
    for condition, color in COLORS.items():
        points = participants[participants.condition.eq(condition)]
        x = points.run + OFFSETS[condition] + points.subject.map(jitter)
        ax.scatter(x, points.value, s=25, color=color, alpha=.23, edgecolors='none', zorder=2)
        estimates = group[group.condition.eq(condition)].sort_values('run')
        y = estimates.value
        handle = ax.errorbar(estimates.run + OFFSETS[condition], y,
            yerr=np.vstack([y - estimates.ci_low, estimates.ci_high - y]),
            fmt='o-', color=color, ms=9, mec='white', mew=1.1, lw=1.8,
            elinewidth=1.45, capsize=3.5, capthick=1.25,
            label=LABELS[condition], zorder=4)
        handles.append(handle)
    ax.set(xlabel='Run', ylabel=ylabel, xlim=(.5, 8.5), ylim=limits)
    ax.set_xticks(range(1, 9)); ax.set_yticks(ticks)
    ax.grid(axis='y', color='#E5E8ED', lw=.7, zorder=0); ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#AAB0B9'); ax.spines[side].set_linewidth(.8)
    ax.tick_params(axis='both', length=3, color='#AAB0B9', labelcolor='#424750')
    for center, label in PHASES:
        ax.text(center, 1.025, label, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', color='#676D77', fontsize=10.5)
    return handles


def make_figures(root):
    root = Path(root).resolve()
    out = root / 'results/behavioral/block_errors_20261013'
    rtout = root / 'results/behavioral/fmri_rt_20261012'
    figures = root / 'manuscript/figures'
    groups = pd.read_csv(out / 'group_run_error_rates.csv')
    participants = pd.read_csv(out / 'participant_run_error_rates.csv')
    trials = pd.read_csv(out / 'trial_outcomes.csv')
    keys = ['subject', 'session', 'rule_group']
    expected = trials.groupby(keys).error.agg(['sum', 'size'])
    observed = participants.set_index(keys).sort_index()
    assert observed.index.equals(expected.index)
    np.testing.assert_array_equal(observed.n_errors, expected['sum'])
    np.testing.assert_array_equal(observed.n_trials, expected['size'])
    np.testing.assert_allclose(observed.error_rate, expected['sum'] / expected['size'], atol=1e-12)
    assert len(groups) == 20 and len(participants) == 400
    assert participants.groupby(['session', 'rule_group']).subject.nunique().eq(20).all()
    for _, row in groups.iterrows():
        x = participants[participants.session.eq(row.session) & participants.rule_group.eq(row.rule_group)].error_rate
        np.testing.assert_allclose(row.error_rate, x.mean(), atol=1e-12)
    error_group = groups.rename(columns={'session':'run', 'rule_group':'condition',
        'error_rate':'value', 'error_rate_ci_low':'ci_low', 'error_rate_ci_high':'ci_high'})
    error_points = participants.rename(columns={'session':'run', 'rule_group':'condition', 'error_rate':'value'})
    error_model = out / 'mixed_model/binomial_cell_estimates.csv'
    model_inputs = []
    model_based = error_model.is_file()
    if model_based:
        model_validation = json.loads((out/'mixed_model/validation.json').read_text())
        assert model_validation['status']=='passed'
        for source, sha in model_validation['input_sha256'].items():
            assert digest(source)==sha, source
        fitted = pd.read_csv(error_model)
        assert len(fitted)==20
        assert set(zip(fitted.session,fitted.rule_group))==set(zip(groups.session,groups.rule_group))
        error_group = fitted[['session','rule_group','marginal_error_probability',
                              'marginal_ci_low','marginal_ci_high']].rename(columns={'session':'run','rule_group':'condition',
            'marginal_error_probability':'value','marginal_ci_low':'ci_low','marginal_ci_high':'ci_high'})
        assert error_group.columns.is_unique
        model_inputs = [error_model, out/'mixed_model/validation.json', out/'mixed_model/analysis_spec.json']
        large_point_label = 'GLMM estimates'
        error_summary = 'Binomial GLMM population-average probabilities integrated over normal random effects'
        error_intervals = ('Pointwise 95% whole-participant-refit bootstrap percentile intervals; '
                           f"{model_validation['bootstrap_binomial_valid']} valid refits")
    else:
        large_point_label = 'mean error rates'
        error_summary = 'Equal-weight mean of participant run/condition error proportions'
        error_intervals = 'Pointwise 95% participant-bootstrap percentile intervals, 20000 resamples'
    for col in ['value', 'ci_low', 'ci_high']:
        error_group[col] *= 100
    error_points['value'] *= 100
    upper = float(np.ceil(max(error_points.value.max(), error_group.ci_high.max()) / 5) * 5 + 2)
    lower = -1.0  # Keeps zero-error participant markers fully visible.

    cells = pd.read_csv(rtout / 'lme/min100_cell_estimates.csv')
    cells['run'] = cells.cell.str.extract(r'r(\d+)')[0].astype(int)
    cells['condition'] = cells.cell.str.extract(r'_(.+)')[0]
    rt_group = cells.rename(columns={'geometric_rt_ms':'value',
        'geometric_ci_low_ms':'ci_low', 'geometric_ci_high_ms':'ci_high'})
    rt_points = pd.read_csv(rtout / 'participant_run_rt_min100.csv').rename(
        columns={'session':'run', 'rule_group':'condition', 'rt_ms':'value'})
    assert set(zip(error_group.run, error_group.condition)) == set(zip(rt_group.run, rt_group.condition))
    assert set(error_points.subject) == set(rt_points.subject)
    assert len(rt_points) == 400

    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.labelsize':12, 'pdf.fonttype':42, 'ps.fonttype':42}):
        fig, ax = plt.subplots(figsize=(10, 6.2))
        fig.subplots_adjust(left=.10, right=.985, bottom=.17, top=.77)
        handles = draw_panel(ax, error_group, error_points, 'Error rate (%)',
                             (lower, upper), np.arange(0, upper + 1, 5))
        fig.suptitle('Error rates across sequence replacement',
                     y=.975, fontsize=16, fontweight='bold', color='#262B33')
        fig.text(.5, .923, '20 participants  ·  Incorrect or omitted responses / all trials',
                 ha='center', fontsize=11, color='#676D77')
        fig.legend(handles=handles, labels=list(LABELS.values()), loc='upper center',
                   bbox_to_anchor=(.5, .892), ncol=3, frameon=False,
                   handlelength=2.3, columnspacing=2.5)
        fig.text(.5, .04, f'Large points: {large_point_label} (95% CI)     '
                 'Faint points: participant error rates', ha='center', fontsize=10, color='#676D77')
        savefig(fig, figures / 'behavioral_error_rates')

        # Full-width stacked panels preserve the RT graph's readable scale.
        fig, axes = plt.subplots(2, 1, figsize=(10, 10.2))
        fig.subplots_adjust(left=.10, right=.985, bottom=.075, top=.82, hspace=.60)
        handles = draw_panel(axes[0], rt_group, rt_points, 'Reaction time (ms)',
                             (170, 570), range(200, 551, 50))
        draw_panel(axes[1], error_group, error_points, 'Error rate (%)',
                   (lower, upper), np.arange(0, upper + 1, 5))
        fig.suptitle('Behavior across sequence replacement',
                     y=.99, fontsize=16, fontweight='bold', color='#262B33')
        fig.text(.5, .945, '20 participants', ha='center', fontsize=11, color='#676D77')
        fig.legend(handles=handles, labels=list(LABELS.values()), loc='upper center',
                   bbox_to_anchor=(.5, .925), ncol=3, frameon=False,
                   handlelength=2.3, columnspacing=2.5)
        for ax, title in zip(axes, ['A  Reaction times: correct RT ≥ 100 ms',
                                    ('B  Error rates: binomial mixed model' if model_based else
                                     'B  Error rates: incorrect or omitted responses / all trials')]):
            ax.text(0, 1.14, title, transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=12, fontweight='bold', color='#262B33')
        axes[0].text(.5, -.24, 'Large points: LME estimates (95% CI)     Faint points: participant mean RTs',
                     transform=axes[0].transAxes, ha='center', fontsize=9.5, color='#676D77')
        axes[1].text(.5, -.24, f'Large points: {large_point_label} (95% CI)     Faint points: participant error rates',
                     transform=axes[1].transAxes, ha='center', fontsize=9.5, color='#676D77')
        savefig(fig, figures / 'behavioral_rt_and_errors')

    inputs = [out/'group_run_error_rates.csv', out/'participant_run_error_rates.csv',
              out/'trial_outcomes.csv', rtout/'lme/min100_cell_estimates.csv',
              rtout/'participant_run_rt_min100.csv'] + model_inputs
    metadata = dict(n_subjects=20, n_error_group_points=20, n_error_participant_points=400,
        n_rt_group_points=20, n_rt_participant_points=400, condition_offsets=OFFSETS,
        error_definition='Incorrect button press or omission / all trials; no RT filtering',
        error_summary=error_summary,
        error_intervals=error_intervals, error_mixed_model=model_based,
        rt_summary='100-ms sensitivity LME fixed-effect geometric RT; random effects set to zero',
        rt_intervals='Pointwise 95% intervals with df capped at 19',
        participant_jitter='Same fixed participant offsets from -0.065 to 0.065 runs as the RT graph',
        error_y_limits=[lower,upper], error_participant_values_all_visible=True,
        input_sha256={str(path):digest(path) for path in inputs}, script_sha256=digest(Path(__file__)),
        artifacts={name:{ext:digest(figures/(name+'.'+ext)) for ext in ['png','pdf']}
                   for name in ['behavioral_error_rates','behavioral_rt_and_errors']})
    (figures/'behavioral_rt_and_errors_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Created matching all-run error-rate graph and paired RT/error Figure 1.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args=p.parse_args()
    make_figures(args.root)


if __name__=='__main__':
    main()
