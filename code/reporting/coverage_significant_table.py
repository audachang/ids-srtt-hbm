#!/usr/bin/env python3
"""Render significant coverage-aware estimates using only the Python standard library.

Input is the public group-summary CSV; no participant data or imaging is needed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

STAGES = ('initial', 'early', 'middle', 'late')


def roi_name(roi):
    side = 'Left' if roi.startswith('Left') else 'Right'
    name = roi[len(side):]
    names = {'SMA': 'BA6', 'DLPFC9': 'BA9', 'DLPFC46': 'BA46',
             'Cerebellum': 'cerebellar cortex', 'PrimaryMotor': 'BA4',
             'AccumbensArea': 'accumbens', 'ThalamusProper': 'thalamus'}
    return side, names.get(name, re.sub(r'(?<!^)([A-Z])', r' \1', name).lower())


def render(source, output):
    with source.open(newline='') as handle:
        data = list(csv.DictReader(handle))
    assert len(data) == 192 and len({(r['stage'], r['roi']) for r in data}) == 192
    assert {r['policy'] for r in data} == {'observed80'}
    assert {r['model'] for r in data} == {'censor_hp128'}
    selected = [r for r in data if float(r['q_within_contrast']) <= .05]
    for r in selected:
        assert int(r['df']) == int(r['n']) - 1
        assert r['significant_within_contrast'].lower() == 'true'
    header = r'ROI & Laterality & $N$ & Mean & 95\% CI & $t(df)$ & $q_{48}$ \\'
    text = [r'\begingroup', r'\singlespacing', r'\footnotesize',
            r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
            r'\begin{longtable}{@{}llrrrrr@{}}',
            r'\caption{Significant coverage-aware ROI effects from the censored high-pass GLM.}\label{tab:observedsignificant}\\',
            r'\toprule', header, r'\midrule', r'\endfirsthead',
            r'\multicolumn{7}{l}{\textit{Table \thetable\ (continued)}}\\',
            r'\toprule', header, r'\midrule', r'\endhead',
            r'\bottomrule', r'\endfoot']
    counts = {}; entries_count = 0
    for stage in STAGES:
        rows = [r for r in selected if r['stage'] == stage]
        counts[stage] = len(rows)
        if not rows:
            continue
        groups = {}
        for row in rows:
            side, name = roi_name(row['roi'])
            groups.setdefault(name, {})[side] = row
        def order(item):
            name, sides = item
            category = 0 if len(sides) == 2 else (1 if 'Left' in sides else 2)
            return category, -max(abs(float(r['t'])) for r in sides.values()), name
        groups = sorted(groups.items(), key=order)
        entries_count += len(groups)
        text += [r'\addlinespace', r'\multicolumn{7}{l}{\textbf{' + stage.title() + r'}} \\*']
        for index, (name, sides) in enumerate(groups):
            def cells(row):
                f = lambda key: float(row[key])
                q = '$<.001$' if f('q_within_contrast') < .001 else f"{f('q_within_contrast'):.3f}"
                return [row['n'], f"{f('mean'):.3f}",
                        f"[{f('ci_low'):.3f}, {f('ci_high'):.3f}]",
                        f"{f('t'):.2f} ({row['df']})", q]
            if len(sides) == 2:
                laterality = 'Bilateral'
                left, right = cells(sides['Left']), cells(sides['Right'])
                values = [r'\shortstack[r]{L: ' + a + r'\\R: ' + b + '}'
                          for a, b in zip(left, right)]
            else:
                laterality, row = next(iter(sides.items()))
                values = cells(row)
            ending = r'\\*' if index < len(groups) - 1 else r'\\'
            text.append(' & '.join([name, laterality] + values) + ' ' + ending)
    text += [r'\end{longtable}',
             r'\noindent\textit{Note.} Only within-stage BH $q_{48}\leq.05$ effects are shown; no late effect survived. Each participant contributes the observed ROI portion shared by all eight fitting masks, requiring at least 80\% parcel coverage. Bilateral entries contain separate left (L) and right (R) estimates, not a pooled test. $N$ and $df=N-1$ refer to each hemisphere. The 48 planned slots include untestable frontal-pole slots as $p=1$. Means sum two run coefficients. Confidence intervals are pointwise, unadjusted, and displayed only for selected effects. Neither intervals nor $q$ values correct for post hoc model selection. All 192 slots, including nonsignificant effects, appear in Supplementary Table~\ref{Supp-tab:allcoverage}. BA6 denotes the full atlas parcel.',
             r'\endgroup', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(text))
    csv_output = output.with_suffix('.csv')
    with csv_output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(selected)
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    table_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    counts=counts, estimates=len(selected), roi_entries=entries_count,
                    ordering='Stage; bilateral, left-only, right-only; descending absolute t within category',
                    fields=['n', 'mean', 'ci_low', 'ci_high', 't', 'df', 'q_within_contrast'])
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output), indent=2))
