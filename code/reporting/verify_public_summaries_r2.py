#!/usr/bin/env python3
"""Check saved continuing-sequence summaries and all inherited release aggregates.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
Read only: no participant analysis, fitting or new inference.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import verify_public_summaries_r1_1 as inherited

A8 = 'results/glm_volume/review_r2_benchmark_20261020'
STATISTICS = inherited.STATISTICS
PREFIXES = ('new_sequence_', 'continuing_', 'paired_difference_')


def summary_blocks(rel, frame):
    yield from inherited.summary_blocks(rel, frame)
    # Wide benchmark exports use prefixes, unlike the older suffixed comparisons.
    # All contrasts share the fixed participant support, so their disclosure
    # identity is the same whole model/ROI/sample block as inherited summaries.
    for prefix in PREFIXES:
        if prefix+'n' not in frame:
            continue
        columns = {c: c[len(prefix):] for c in frame if c.startswith(prefix)}
        selected = frame[[c for c in frame if not any(c.startswith(p) for p in PREFIXES)] + list(columns)]
        selected = selected.rename(columns=columns)
        for block in inherited.summary_blocks(rel, selected):
            assert block['suffix'] == ''
            block.update(suffix=prefix, count=prefix+'n', fields=[prefix+c for c in block['fields']])
            yield block


def disclosure_plan(root):
    frames = {str(p.relative_to(root)): pd.read_csv(p) for p in sorted(root.rglob('*.csv'))}
    grouped = collections.defaultdict(list)
    for rel, frame in frames.items():
        for block in summary_blocks(rel, frame):
            grouped[block['slot']].append(block)
    suppressed = set()
    pairs = 0
    for blocks in grouped.values():
        nodes = {(b['cohort'], b['policy'], b['n']): b for b in blocks}
        unsafe = set()
        for key, a in nodes.items():
            if a['n'] < 2:
                unsafe.add(key)
            for b in nodes.values():
                if inherited.previous.nested(a, b) and b['n']-a['n'] == 1:
                    unsafe.add(key)
                    pairs += 1
        changed = True
        while changed:
            changed = False
            for key, a in nodes.items():
                if key not in unsafe and any(a['n'] == nodes[k]['n'] and
                        (inherited.previous.nested(a, nodes[k]) or inherited.previous.nested(nodes[k], a)) for k in unsafe):
                    unsafe.add(key)
                    changed = True
        for b in blocks:
            if (b['cohort'], b['policy'], b['n']) in unsafe:
                suppressed.add((b['path'], b['index'], b['suffix']))
    return frames, suppressed, pairs


blocked_fields = inherited.blocked_fields


def disclosure_check(root):
    frames, plan, pairs = disclosure_plan(root)
    checked = 0
    for rel, frame in frames.items():
        for block in summary_blocks(rel, frame):
            if (rel, block['index'], block['suffix']) in plan:
                fields = blocked_fields(frame, block)
                assert frame.loc[block['index'], fields].isna().all(), (
                    'Disclosure: nonblank nested/singleton statistic '+rel+' row '+str(block['index']+2)+' block '+block['suffix'])
                checked += len(fields)
    return dict(statistic_cells_checked=checked, nested_relations=pairs, blocked_csv_blocks=len(plan))


def main(root):
    disclosure = disclosure_check(root)
    inherited.main(root)
    spec = json.loads((root/'AGGREGATE_CHECK_SPEC_R2.json').read_text())
    arithmetic = 0
    duplicates = 0
    bounded = 0
    families = collections.Counter()
    keys = ['cohort', 'model', 'policy', 'phase', 'roi']

    def close(actual, expected, label):
        nonlocal arithmetic
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert np.allclose(a, b, rtol=2e-7, atol=2e-9, equal_nan=True), label
        arithmetic += int(np.isfinite(a).sum())

    def agrees(display, full, columns, label):
        nonlocal duplicates
        merged = display.merge(full, on=keys, suffixes=('_display', '_full'), validate='one_to_one')
        assert len(merged) == len(display), label+' missing keys'
        for col in columns:
            close(merged[col+'_display'], merged[col+'_full'], label+' '+col)
        duplicates += len(merged)

    def bh(frame, q, size, label):
        nonlocal bounded
        selected = frame[frame[q].notna()] if size == 4 else frame
        for _, part in selected.groupby(['cohort', 'model', 'policy', 'phase'], dropna=False):
            assert len(part) == size, label+' family size'
            assert not part.roi.duplicated().any(), label+' repeated ROI'
            if size == 4:
                assert set(part.roi) == set(inherited.previous.HEADLINE)
            probs = part.p.fillna(1)
            hidden = part.p.isna() & part.n.ge(2)
            if hidden.any():
                lo, hi = probs.copy(), probs.copy()
                lo.loc[hidden], hi.loc[hidden] = 0, 1
                lower = stats.false_discovery_control(lo, method='bh')
                upper = stats.false_discovery_control(hi, method='bh')
                assert (part[q].to_numpy() >= lower-2e-9).all() and (part[q].to_numpy() <= upper+2e-9).all(), label+' BH bounds'
                bounded += 1
            else:
                close(part[q], stats.false_discovery_control(probs, method='bh'), label+' BH')
            families[size] += 1

    for rel in spec['added_results']:
        if not rel.endswith('.csv'):
            continue
        frame = pd.read_csv(root/rel)
        assert not set(frame.columns.str.lower()) & {'subject','participant','subject_id','participant_id','sid','sub','path','file_path'}, rel
        for prefix in ('',)+PREFIXES:
            required = [prefix+c for c in ['n']+STATISTICS]
            if not set(required) <= set(frame):
                continue
            part = frame[frame[required].notna().all(axis=1)].rename(columns={prefix+c:c for c in ['n']+STATISTICS})
            assert part.se.gt(0).all(), rel+' SE'
            close(part.df, part.n-1, rel+' df')
            close(part.se, part.sd/np.sqrt(part.n), rel+' SE arithmetic')
            close(part.t, part['mean']/part.se, rel+' t')
            margin = stats.t.ppf(.975, part.df)*part.se
            close(part.ci_low, part['mean']-margin, rel+' CI low')
            close(part.ci_high, part['mean']+margin, rel+' CI high')
            close(part.p, 2*stats.t.sf(abs(part.t), part.df), rel+' p')
            close(part.cohen_dz, part['mean']/part.sd, rel+' standardized effect')

    continuing = pd.read_csv(root/A8/'continuing_sequence_results.csv')
    new = pd.read_csv(root/A8/'early_new_sequence_results.csv')
    paired = pd.read_csv(root/A8/'early_paired_difference_results.csv')
    assert len(continuing) == 192 and len(new) == len(paired) == 96
    assert set(continuing.phase) == {'early', 'middle'} and set(new.phase) == set(paired.phase) == {'early'}
    for frame in [continuing, new, paired]:
        assert set(frame.cohort) == {'N20', 'N22'}
        assert frame.model.eq('reference').all() and frame.policy.eq('observed80').all()
        bh(frame, 'q_BH48', 48, 'all parcels')
    bh(continuing, 'q_BH4', 4, 'headline')
    assert continuing.loc[~continuing.roi.isin(inherited.previous.HEADLINE), 'q_BH4'].isna().all()
    wide = pd.read_csv(root/A8/'early_comparison_all_rois.csv')
    for prefix, full in zip(PREFIXES, [new, continuing[continuing.phase.eq('early')], paired]):
        columns = ['n']+STATISTICS+['q_BH48']+(['q_BH4'] if prefix == 'continuing_' else [])
        selected = wide[keys+[prefix+c for c in columns]].rename(columns={prefix+c:c for c in columns})
        agrees(selected, full, columns, prefix+'wide')
    headline = pd.read_csv(root/A8/'headline_early_comparison.csv')
    assert len(headline) == 8 and set(headline.roi) == set(inherited.previous.HEADLINE)
    agrees(headline, wide, [c for c in wide if c not in keys+['roi_label']], 'headline wide')
    headline_cont = pd.read_csv(root/A8/'headline_continuing_sequence_results.csv')
    assert len(headline_cont) == 16 and set(headline_cont.roi) == set(inherited.previous.HEADLINE)
    agrees(headline_cont, continuing, ['n']+STATISTICS+['q_BH48', 'q_BH4'], 'headline continuing')
    close(wide.paired_difference_mean, wide.new_sequence_mean-wide.continuing_mean, 'paired mean identity')
    base = root/'results/glm_volume/reanalysis_20261018_updated_n22/roi_reporting'
    stage = pd.read_csv(base/'stage_results.csv').rename(columns={'stage':'phase', 'q':'q_BH48'})
    agrees(new, stage, ['n']+STATISTICS+['q_BH48'], 'existing early new sequence')
    within = pd.read_csv(base/'within_run_results.csv')
    within = within[within.contrast.eq('early_r34')].copy()
    within['phase'] = 'early'
    within = within.rename(columns={'q':'q_BH48'})
    scaled = ['mean', 'sd', 'se', 'ci_low', 'ci_high']
    within[scaled] *= 2
    agrees(paired, within, ['n']+STATISTICS+['q_BH48'], 'twice saved early within-run')
    counts = pd.read_csv(root/A8/'family_summary.csv')
    for row in counts.itertuples():
        part = continuing[continuing.cohort.eq(row.cohort) & continuing.phase.eq(row.phase)]
        assert row.slots_BH48 == 48 and row.slots_BH4 == 4
        assert row.testable_slots == int(part.n.ge(2).sum())
        assert row.discoveries_BH48 == int(part.q_BH48.le(.05).sum())
        assert row.headline_discoveries_BH4 == int(part.q_BH4.le(.05).sum())
    status = json.loads((root/'COVERAGE_STATUS.json').read_text())
    assert status['complete'], 'Article inventory is not final'
    print(json.dumps(dict(status='PASS', benchmark_arithmetic_values=arithmetic,
        BH_families_checked=dict(families), BH_families_with_suppressed_p_bounds=bounded,
        duplicate_rows_checked=duplicates, disclosure=disclosure,
        limits='Saved aggregate consistency and disclosure only; no model fitting or article rendering.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disclosure-only', action='store_true')
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.disclosure_only:
        print(json.dumps(disclosure_check(root), indent=2))
    elif args.legacy:
        disclosure_check(root)
        inherited.previous.legacy_check(root)
    else:
        main(root)
