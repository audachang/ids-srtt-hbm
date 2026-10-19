#!/usr/bin/env python3
"""Verify saved slice-timing sensitivity summaries and inherited public aggregates.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
No fitting, new inference, network operations or writes.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import verify_public_summaries_r1 as previous
sys.dont_write_bytecode = True
A7 = 'results/glm_volume/review_r1_slicetime_20261019'
STATISTICS = ['mean', 'sd', 'se', 'ci_low', 'ci_high', 'cohen_dz', 't', 'df', 'p']


def summary_blocks(rel, frame):
    """Use inherited slot identities, adding the aligned comparison side."""
    yield from previous.summary_blocks(rel, frame)
    if 'n_aligned' in frame:
        renamed = frame.rename(columns={c: c[:-8]+'_variant' for c in frame if c.endswith('_aligned')}).copy()
        renamed['model'] = 'slicetime_aligned'
        for block in previous.summary_blocks(rel, renamed):
            if block['suffix'] != '_variant':
                continue
            block.update(suffix='_aligned', count='n_aligned',
                         fields=[c[:-8]+'_aligned' for c in block['fields']])
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
                if previous.nested(a, b) and b['n']-a['n'] == 1:
                    unsafe.add(key)
                    pairs += 1
        changed = True
        while changed:
            changed = False
            for key, a in nodes.items():
                if key not in unsafe and any(a['n'] == nodes[k]['n'] and
                        (previous.nested(a, nodes[k]) or previous.nested(nodes[k], a)) for k in unsafe):
                    unsafe.add(key)
                    changed = True
        for b in blocks:
            if (b['cohort'], b['policy'], b['n']) in unsafe:
                suppressed.add((b['path'], b['index'], b['suffix']))
    return frames, suppressed, pairs


def blocked_fields(frame, block):
    fields = list(block['fields']) + [c for c in frame if c.startswith('mean_shift')]
    if block['suffix'] == '':
        fields += [c for c in frame if c.startswith(('structured_', 'random_', 'condition_mean_')) and '_q' not in c]
        fields += [c for c in ['component_wording', 'direction'] if c in frame]
    return sorted(set(fields))


def disclosure_check(root):
    frames, plan, pairs = disclosure_plan(root)
    checked = 0
    for rel, frame in frames.items():
        for block in summary_blocks(rel, frame):
            if (rel, block['index'], block['suffix']) not in plan:
                continue
            fields = blocked_fields(frame, block)
            assert frame.loc[block['index'], fields].isna().all(), (
                'Disclosure: nonblank nested/singleton statistic '+rel+' row '+str(block['index']+2)+' block '+block['suffix'])
            checked += len(fields)
    return dict(statistic_cells_checked=checked, nested_relations=pairs, blocked_csv_blocks=len(plan))


def main(root):
    disclosure = disclosure_check(root)
    previous.main(root)
    spec = json.loads((root/'AGGREGATE_CHECK_SPEC_R1_1.json').read_text())
    arithmetic = 0
    families = collections.Counter()
    bounded = 0
    duplicates = 0

    def close(actual, expected, label):
        nonlocal arithmetic
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        assert np.allclose(actual, expected, rtol=2e-7, atol=2e-9, equal_nan=True), label
        arithmetic += int(np.isfinite(actual).sum())

    def bh(frame, keys, p, q, slots, label):
        nonlocal bounded
        selected = frame[frame[q].notna()] if q == 'q_headline_early_4' else frame
        for _, part in selected.groupby(keys, dropna=False):
            assert len(part) == slots, (label, q, len(part), slots)
            count = 'n'+p[1:]
            hidden = part[p].isna() & part[count].ge(2)
            probs = part[p].fillna(1)
            if hidden.any():
                lo = probs.copy(); hi = probs.copy()
                lo.loc[hidden] = 0; hi.loc[hidden] = 1
                lower = stats.false_discovery_control(lo, method='bh')
                upper = stats.false_discovery_control(hi, method='bh')
                assert (part[q].to_numpy() >= lower-2e-9).all() and (part[q].to_numpy() <= upper+2e-9).all(), label+' BH bounds '+q
                bounded += 1
            else:
                close(part[q], stats.false_discovery_control(probs, method='bh'), label+' '+q)
            families[slots] += 1

    for rel in spec['added_results']:
        if not rel.endswith('.csv'):
            continue
        frame = pd.read_csv(root/rel)
        assert not set(frame.columns.str.lower()) & {'subject', 'participant', 'subject_id', 'participant_id', 'sid', 'sub', 'path', 'file_path'}, rel
        for suffix in ['', '_aligned', '_reference']:
            fields = [c+suffix for c in ['n', 'mean', 'sd', 'se', 'df', 't', 'ci_low', 'ci_high', 'p']]
            if not set(fields) <= set(frame):
                continue
            part = frame[frame[fields].notna().all(axis=1)]
            assert part['se'+suffix].gt(0).all(), rel+' SE'
            close(part['df'+suffix], part['n'+suffix]-1, rel+' df')
            close(part['se'+suffix], part['sd'+suffix]/np.sqrt(part['n'+suffix]), rel+' SE arithmetic')
            close(part['t'+suffix], part['mean'+suffix]/part['se'+suffix], rel+' t')
            margin = stats.t.ppf(.975, part['df'+suffix])*part['se'+suffix]
            close(part['ci_low'+suffix], part['mean'+suffix]-margin, rel+' CI low')
            close(part['ci_high'+suffix], part['mean'+suffix]+margin, rel+' CI high')
            close(part['p'+suffix], 2*stats.t.sf(abs(part['t'+suffix]), part['df'+suffix]), rel+' p')
            close(part['cohen_dz'+suffix], part['mean'+suffix]/part['sd'+suffix], rel+' standardized effect')
        if '/roi/' not in rel:
            continue
        name = Path(rel).name
        keys = ['cohort', 'model', 'policy']
        if name in ['stage_results.csv', 'component_balance_results.csv', 'phase_component_results.csv']:
            bh(frame, keys+['stage']+(['component'] if 'component' in frame else []), 'p', 'q', 48, rel)
        if name == 'stage_results.csv':
            bh(frame, keys, 'p', 'q_joint_stages_192', 192, rel)
        if name in ['component_balance_results.csv', 'headline_early_component_balance.csv']:
            bh(frame, keys+['component'], 'p', 'q_headline_early_4', 4, rel)
        if name == 'direct_phase_results.csv':
            bh(frame, keys, 'p', 'q_three_pairs_144', 144, rel)
            bh(frame, keys+['pair'], 'p', 'q_within_pair_48', 48, rel)
        if name == 'slicetime_aligned_vs_reference.csv':
            for suffix in ['_aligned', '_reference']:
                bh(frame, ['cohort', 'policy', 'stage'], 'p'+suffix, 'q'+suffix, 48, rel)
                bh(frame, ['cohort', 'policy'], 'p'+suffix, 'q_joint_stages_192'+suffix, 192, rel)
        if 'mean_shift' in frame:
            close(frame.mean_shift, frame.mean_aligned-frame.mean_reference, rel+' mean shift')

    def agrees(subset, full, keys, columns, label):
        nonlocal duplicates
        assert not full.duplicated(keys).any(), label+' duplicate keys'
        merged = subset.merge(full, on=keys, suffixes=('_display', '_full'), validate='one_to_one')
        assert len(merged) == len(subset), label+' missing keys'
        for col in columns:
            close(merged[col+'_display'], merged[col+'_full'], label+' '+col)
        duplicates += len(merged)

    keys = ['cohort', 'model', 'policy', 'stage', 'roi']
    stage = pd.read_csv(root/A7/'roi/stage_results.csv')
    components = pd.read_csv(root/A7/'roi/component_balance_results.csv')
    tidy = pd.read_csv(root/A7/'roi/headline_tidy_all_specifications.csv')
    agrees(tidy, stage, keys, STATISTICS+['q', 'q_joint_stages_192'], 'headline stage')
    early = pd.read_csv(root/A7/'roi/headline_early_component_balance.csv')
    agrees(early, components, keys+['component'], STATISTICS+['q', 'q_headline_early_4'], 'headline components')
    for comp in ['structured', 'random', 'condition_mean']:
        cols = ['mean', 'ci_low', 'ci_high', 'p', 'q', 'q_headline_early_4']
        selected = tidy[tidy.stage.ne('initial')][keys+[comp+'_'+c for c in cols]]
        selected = selected.rename(columns={comp+'_'+c:c for c in cols})
        agrees(selected, components[components.component.eq(comp)], keys, cols, comp+' display')
    for name in ['slicetime_aligned_vs_reference.csv', 'headline_and_accumbens_comparison.csv']:
        frame = pd.read_csv(root/A7/'roi'/name)
        for suffix, model in [('_aligned', 'slicetime_aligned'), ('_reference', 'reference')]:
            selected = frame.copy(); selected['model'] = model
            cols = ['n']+STATISTICS+['q', 'q_joint_stages_192']
            selected = selected[keys+[c+suffix for c in cols]].rename(columns={c+suffix:c for c in cols})
            agrees(selected, stage, keys, cols, name+suffix)
    ref = stage[stage.model.eq('reference')]
    agrees(ref, pd.read_csv(root/previous.IMAGING/'roi/stage_results.csv'), keys,
           ['n']+STATISTICS+['q', 'q_joint_stages_192'], 'inherited reference')
    # Saved phase/contrast algebra is checked only where all required public cells remain.
    wide = components.pivot(index=keys, columns='component', values='mean')
    close(wide.condition_mean, (wide.structured+wide.random)/2, 'component balance')
    close(wide.did_half, wide.structured-wide.random, 'component difference')
    joined = stage.set_index(keys).join(wide[['did_half']], how='inner')
    close(joined['mean'], 2*joined.did_half, 'stage/component scale')
    perm = pd.read_csv(root/A7/'voxelwise/permutation_summary.csv')
    comparison = pd.read_csv(root/A7/'voxelwise/aligned_vs_reference_counts.csv')
    cols = ['cohort', 'contrast', 'n_subjects', 'alpha', 'n_perm', 'n_mask_voxels']
    for suffix, model, full in [('_aligned', 'slicetime_aligned', perm), ('_reference', 'reference',
            pd.read_csv(root/'results/glm_volume/reanalysis_20261018_updated_n22/voxelwise/permutation_summary.csv'))]:
        selected = comparison[cols+['positive_voxels'+suffix, 'negative_voxels'+suffix]].rename(
            columns={c+suffix:c for c in ['positive_voxels', 'negative_voxels']})
        agrees(selected, full[full.model.eq(model)], cols, ['positive_voxels', 'negative_voxels'], 'voxel counts '+suffix)
    assert len(perm) == 10 and perm.n_perm.eq(10000).all() and perm.seed.eq(20261018).all()
    stage_counts = stage.assign(stage_BH48=stage.q.le(.05), joint_BH192=stage.q_joint_stages_192.le(.05))
    stage_counts = stage_counts.groupby(['cohort', 'model', 'stage'], as_index=False)[['stage_BH48', 'joint_BH192']].sum()
    agrees(pd.read_csv(root/A7/'stage_discovery_counts.csv'), stage_counts,
           ['cohort', 'model', 'stage'], ['stage_BH48', 'joint_BH192'], 'saved stage discovery counts')
    direct = pd.read_csv(root/A7/'roi/direct_phase_results.csv')
    direct_counts = direct.assign(direct_BH144=direct.q_three_pairs_144.le(.05))
    direct_counts = direct_counts.groupby(['cohort', 'model', 'pair'], as_index=False).direct_BH144.sum()
    agrees(pd.read_csv(root/A7/'direct_phase_discovery_counts.csv'), direct_counts,
           ['cohort', 'model', 'pair'], ['direct_BH144'], 'saved direct discovery counts')
    assert set(families) == {4, 48, 144, 192}
    assert json.loads((root/'COVERAGE_STATUS.json').read_text())['complete'], 'Article inventory is not final'
    print(json.dumps(dict(status='PASS', slice_timing_arithmetic_values=arithmetic,
        BH_families_checked=dict(sorted(families.items())), BH_families_with_suppressed_p_bounds=bounded,
        duplicate_rows_checked=duplicates, disclosure=disclosure,
        limits='Saved aggregate consistency and disclosure only; no model fitting or article rendering.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disclosure-only', action='store_true')
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('root', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.disclosure_only:
        print(json.dumps(disclosure_check(root), indent=2))
    elif args.legacy:
        disclosure_check(root); previous.legacy_check(root)
    else:
        main(root)
