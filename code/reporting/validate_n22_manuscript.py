#!/usr/bin/env python3
"""Read-only T9 checks against saved facts; never run analyses or Make targets.

Examples:
  python code/reporting/validate_n22_manuscript.py --phase 2 --build
  python code/reporting/validate_n22_manuscript.py --strict --build
  python code/reporting/validate_n22_manuscript.py --self-test

Errors return 1, including expected unfinished-phase errors. --strict also fails
on warnings. TODOs are informational. --phase 2 changes classification, never
suppresses a check. A default run assumes final integration. Builds are fresh
copies beneath /tmp/t9-t9-check/, without manuscript data/scripts symlinks or
old auxiliary files. --in-place requires --build and explicit coordinator approval
under the T9 brief. Only in-place builds render into docs/t9/render/.

Numeric matching establishes membership, not semantic identity of prose facts.
Table matching is restricted to that table's manifest fact IDs when available;
manifest byte hashes additionally protect its row/column identities. A human
must still review prose fact IDs and scientific interpretation. No scientific
statistics, intervals, corrections, or cross-row summaries are computed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRATCH = Path('/tmp/t9-t9-check')
FRAME = {
    'abstract', 'key_points', 'introduction', 'discussion', 'conclusion',
    'code_availability', 'data_availability', 'title_page_details',
}
IN_PROGRESS = {'results', 'supplement', 'figure_caption_text', 'figure_legends'}
NUM = re.compile(r'(?<![\w.])[+\-−]?(?:(?:\d{1,3}(?:(?:,|\{,\})\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?(?!\w)')
ASSET = re.compile(r'\\(input|include|includegraphics)\s*(?:\[([^\]]*)\]\s*)?\{([^{}]+)\}')
CORRECTION = re.compile(r'(?:\bBH\b|Benjamini|FDR|correction|family|families|multiplicity|descriptive|FWE)', re.I)
ROI = re.compile(r'\b(?:ROI|BA6|BA9|BA46|accumbens|pallidum|thalamus|fusiform|cerebell|frontal pole|middle temporal)', re.I)
RESULT = re.compile(r'(?:[+\-]\d*\.\d|\bq\s*(?:=|<|≤)|surviv|significant|estimate was|estimates were)', re.I)

# Allowlist is contextual, not a global bag of small integers. Sources for the
# retained protocol/acquisition/software constants: accepted t9-methods report,
# Methods design specification, and t9-tables report. Rounded results never use it.
DESIGN_RULES = [
    ('calendar years', r'\b(?:19|20)\d{2}\b', 'calendar/citation years'),
    ('software versions', r'\b(?:trimr|nlme|lme4|fMRIPrep|Nipype|FreeSurfer|ANTs|Nilearn)\}?\s*[~ ]*\d+(?:[.\-]\d+)+', 'retained software identifiers; t9-methods report'),
    ('atlas/sequence/subject identifiers', r'\b(?:BA\s*\d+|[NT][12]|AR\(1\)|Str[12]|Structure[~ ]+[12]|sub-\d+|MNI152NLin2009cAsym|202404EM024)\b', 'atlas, condition, acquisition and subject identifiers'),
    ('figure/table/section numbers', r'\b(?:Figure|Table|Section|Panel|Fig\.)s?[~ ]+[S]?\d+(?:[.\-–,~ ]+\d+)*', 'cross-reference numbering'),
    ('run/block/sequence indices', r'\b(?:runs?|blocks?|structures?|sequence|phase|Stage)[~ ]+\d+(?:\s*(?:--|[–,]|and|to)\s*\d+)*', 'task design indices'),
    ('confidence/coverage/family constants', r'\b(?:48|144|24|80|95|100)(?=\\?%|[~ -]+(?:ROI|planned|slot|contrast|test|behavioral))', 'fixed BH48/BH144/BH24, 80/95/100% support and 95% CIs'),
    ('acquisition/design parameters', r'\b(?:TR|TE)\s*=\s*(?:2000|2530|529|30|3\.3)\b|\b(?:4\.92|7\.38|2\.909)\b|\b(?:64|66|192|256)(?=\\times|\$|[~-]+(?:channel|matrix))|\b(?:50|200)(?=[~ ]+(?:slices|volumes))|\b(?:90|7|60)(?=\^\{\\circ\})', 'retained scanner parameters, t9-methods report'),
    ('physical design constants', r'(?<![\d.])(?:1|2|3|4|6|8|10|20|30|40|100|128|400)(?=[~ -]*(?:s\b|ms\b|mm\b|Hz\b|trials?\b|repetitions?\b))', 'retained task/acquisition timing, smoothing/grid and 100-ms RT sensitivity'),
    ('permutation seed', r'\b20261018\b', 'saved sign-flipping seed, t9-methods report'),
    ('administrative identifiers', r'\bNSTC\s+\d{3}-\d{4}-[A-Z]-\d{3}-\d{3}(?:-MY\d)?|\b\d{4}-\d{4}-\d{4}-\d{4}\b|\b10\.5281/zenodo\.\d+|\bversion[~ ]+\d+(?:\.\d+)+|\bCC[~ ]+BY[~ ]+4\.0', 'retained grant, ORCID, DOI and release/license identifiers, not numerical findings'),
    ('sample and df labels', r'\bN\s*=\s*(?:20|22)\b|\bdf\s*=\s*(?:21|19)\b|\b21\s*\(\$N=22\$\)\s*and\s*19\b', 'cohort.N22.n/cohort.N20.n and approximate N-1 df; t9-methods report'),
    ('correction families', r'\b(?:BH[- ](?:48|144|24)|(?:48|144|24)[- ](?:slot|ROI|contrast)|144\s+direct\s+phase|p\s*=\s*1)\b', 'fixed correction-family sizes and p=1 padding for untestable slots'),
    ('nuisance specification', r'\b12(?=[~ ]+motion)|\b(?:Motion|motion)/aCompCor\s+12\b', 'fixed expanded nuisance-regressor specification; t9-methods report'),
    ('motion thresholds', r'\b0\.[25](?=\s*mm)|(?<=>)\s*20(?=\\%)', 'saved motion-summary threshold definitions, t9-tables generator/source column names'),
    ('reimbursement', r'\bNTD[~ ]+1000\b', 'retained author-confirmed reimbursement; t9-methods report'),
]


def blank(match: re.Match) -> str:
    return ''.join('\n' if c == '\n' else ' ' for c in match.group())


def comments_removed(text: str) -> str:
    # A percent after an even number of backslashes starts a comment; \% does not.
    return re.sub(r'(?m)(?<!\\)((?:\\\\)*)%[^\n]*',
                  lambda m: m[1] + blank_span(m.group(), 0, len(m.group()))[len(m[1]):], text)


def balanced_end(text: str, start: int) -> int:
    """Return end of a braced argument, including nested braces; -1 if unclosed."""
    depth = 0
    for i in range(start, len(text)):
        if i and text[i - 1] == '\\':
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def prose_view(text: str) -> str:
    """Mask non-prose while preserving offsets and line numbers."""
    text = comments_removed(text)
    patterns = [
        ASSET.pattern,
        r'\\(?:cite\w*|label|ref|eqref|autoref|pageref|url|addbibresource|externaldocument)\*?(?:\[[^\]]*\])*\{[^{}]*\}',
        r'\\href\{[^{}]*\}',  # preserve the displayed second argument
        r'\\(?:documentclass|usepackage)(?:\[[^\]]*\])?\{[^{}]*\}',
        r'\\(?:setlength|setcounter)\{[^{}]*\}\{[^{}]*\}',
        r'\\renewcommand\{\\(?:arraystretch|baselinestretch)\}\{[^{}]*\}',
        r'\\def\\(?:arraystretch|baselinestretch)\{[^{}]*\}',
        r'\\(?:begin|end)\{(?:tabular|longtable|array)\}\{[^{}]*\}',
        r'\\(?:multirow|multicolumn)\{\d+\}',
        r'\\(?:vspace|hspace)\*?\{[^{}]*\}|\\\\\[[^\]]*\]',
        r'\\(?:newcommand|renewcommand)\{\\todo\}\[1\]\{[^\n]*',
        r'#\d|\[\d+\](?=\{)|\{[lcrpmb@|\d.\\]+\}(?=\s*\n)',
    ]
    for pattern in patterns:
        text = re.sub(pattern, blank, text)
    return text.replace('--', '  ').replace('−', '-')


def number_key(token: str) -> tuple[Decimal, int]:
    value = Decimal(token.replace('{,}', '').replace(',', '').replace('−', '-'))
    return value, max(0, -value.as_tuple().exponent)


class FactIndex:
    """Exact integers and rounding at the displayed precision, with traceable IDs."""
    def __init__(self, facts: list[dict]):
        self.facts = facts
        self.cache: dict[int, dict[Decimal, list[str]]] = {}

    def matching(self, token: str) -> list[str]:
        value, places = number_key(token)
        if places not in self.cache:
            index: dict[Decimal, list[str]] = defaultdict(list)
            for fact in self.facts:
                raw = fact.get('value')
                if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
                    d = Decimal(str(raw))
                    # A fractional result cannot validate an arbitrary rounded count.
                    if places or d == d.to_integral_value():
                        try:
                            index[Decimal(f'{raw:.{places}f}')].append(fact['id'])
                        except InvalidOperation:
                            pass
                formatted = str(fact.get('formatted', ''))
                for match in NUM.finditer(formatted):
                    try:
                        d, _ = number_key(match.group())
                        # "q < .001" is an inequality, not an exact .001 result.
                        if '<' not in formatted and (places or d == d.to_integral_value()):
                            index[d].append(fact['id'])
                    except InvalidOperation:
                        pass
            self.cache[places] = index
        return self.cache[places].get(value, [])


def allowed_number(text: str, match: re.Match) -> str | None:
    # Search a bounded local window; scanning the entire table for every token
    # made large complete-ROI tables needlessly quadratic.
    left, right = max(0, match.start() - 100), min(len(text), match.end() + 100)
    local = text[left:right]
    for name, pattern, _ in DESIGN_RULES:
        for region in re.finditer(pattern, local, re.I):
            if region.start() + left <= match.start() and region.end() + left >= match.end():
                return name
    before = text[max(0, match.start() - 45):match.start()]
    token = match.group().lstrip('+')
    if token in {'.05', '0.05', '.050', '0.050', '.0125', '0.0125', '.001', '0.001'} and re.search(r'(?:[pq]|FWE|FDR|threshold|TFCE|BH|alpha)[^.;!?]*$', before, re.I):
        return 'explicit inferential threshold'
    # Equation subscripts and declared linear-trend weights are design definitions.
    if re.search(r'(?:_|\^)\{?\s*$', before) or (re.search(r'weights?\s*\[', before, re.I) and token in {'-.3', '-.1', '.1', '.3'}):
        return 'equation index/design weight'
    if token == '100' and re.search(r'RT.*<\s*$', before) and re.match(r'\$?[~ ]*ms', text[match.end():]):
        return '100-ms anticipation definition'
    if re.search(r'\bdf\s*=\s*21\s*\(\$N=22\$\)\s*and\s*$', before) and token == '19':
        return 'approximate N-1 df'
    return None


def sentences(text: str):
    """Keep decimal points inside sentences and preserve original offsets."""
    return re.finditer(r'.+?(?:[.!?](?=\s|$)|$)', text, re.S)


@dataclass
class Finding:
    severity: str
    check: str
    file: str
    line: int
    message: str
    expected: str = ''


class Validator:
    def __init__(self, root: Path, phase: str = 'final'):
        self.root = root.resolve()
        self.manuscript = self.root / 'manuscript'
        self.phase = phase
        self.findings: list[Finding] = []
        self.sources: dict[Path, str] = {}
        self.hashes: dict[str, str] = {}
        self.assets: set[Path] = set()
        self.results_assets: set[Path] = set()
        self.audit: list[dict] = []
        self.counts: Counter = Counter()
        self.build_dir: Path | None = None
        self.changed_files: set[str] = set()

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def expected(self, path: Path) -> str:
        if self.phase != '2':
            return ''
        if path.stem in IN_PROGRESS:
            return 'Phase 2 owner is editing this file'
        if path.stem in FRAME:
            return 'Phase 3 framing revision is pending'
        return ''

    def add(self, check: str, path: Path, line: int, message: str, severity: str = 'ERROR', expected: str | None = None):
        self.findings.append(Finding(severity, check, self.relative(path), line, message,
                                     self.expected(path) if expected is None else expected))

    def read(self, path: Path) -> str:
        if path not in self.sources:
            data = path.read_bytes()
            self.hashes[self.relative(path)] = hashlib.sha256(data).hexdigest()
            self.sources[path] = data.decode('utf-8')
        return self.sources[path]

    def resolve_asset(self, kind: str, target: str) -> Path | None:
        base = self.manuscript / target
        candidates = [base]
        if not base.suffix:
            candidates = [base.with_suffix(ext) for ext in (['.tex', ''] if kind != 'includegraphics' else ['.pdf', '.png', '.jpg', '.jpeg', '.eps'])]
        return next((p for p in candidates if p.is_file()), None)

    def visit(self, path: Path, trail: tuple[Path, ...] = (), result_context: bool = False, supplement_context: bool = False):
        if path in trail:
            self.add('assets', path, 1, 'Recursive input cycle')
            return
        result_context = result_context or path.name == 'results.tex'
        supplement_context = supplement_context or path.name == 'supplement.tex'
        if path in self.sources and not result_context and not supplement_context:
            return
        text = comments_removed(self.read(path))
        for match in ASSET.finditer(text):
            kind, options, target = match.groups()
            line = text.count('\n', 0, match.start()) + 1
            self.counts['asset_references'] += 1
            asset = self.resolve_asset(kind, target)
            historical = target.startswith(('tables/', 'figures/')) and not Path(target).name.startswith('n22_')
            if result_context and historical:
                self.add('assets', path, line, f'Historical asset used in main Results: {target}')
            if supplement_context and historical:
                context = text[max(0, match.start() - 180):match.end() + 120]
                self.add('assets', path, line, f'Historical supplement asset {target}; context: {context.strip()}', 'WARN')
            if asset is None:
                self.add('assets', path, line, f'Missing {kind} target: {target}')
                continue
            if result_context:
                self.results_assets.add(asset)
            if kind == 'includegraphics':
                self.assets.add(asset)
                self.hashes[self.relative(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
                if asset.name == 'n22_fig3_wholebrain.pdf' and not re.search(r'\bpage\s*=', options or ''):
                    self.add('assets', path, line, 'Two-page Figure 3 base PDF needs explicit page selection or both page-specific files')
            else:
                self.visit(asset, trail + (path,), result_context, supplement_context)

    def load_facts(self):
        path = self.root / 'docs/t9/facts_n22.json'
        facts = json.loads(self.read(path))
        if not isinstance(facts, list) or not facts:
            raise ValueError('Ledger must be a nonempty list of scalar fact records')
        self.facts = {f['id']: f for f in facts}
        if len(self.facts) != len(facts):
            self.add('numbers', path, 1, 'Duplicate ledger IDs')
        self.fact_index = FactIndex(facts)
        self.counts['ledger_facts'] = len(facts)
        manifest_path = self.manuscript / 'tables/n22_assets_manifest.json'
        self.manifest = json.loads(self.read(manifest_path)) if manifest_path.is_file() else {}
        self.table_indexes: dict[str, FactIndex] = {}
        for name, ids in self.manifest.get('table_fact_ids', {}).items():
            missing = set(ids) - self.facts.keys()
            if missing:
                self.add('numbers', manifest_path, 1, f'{name}: unknown manifest fact IDs: {sorted(missing)}')
            self.table_indexes[name] = FactIndex([self.facts[i] for i in ids if i in self.facts])

    def check_numbers(self, path: Path, text: str):
        name = self.relative(path)
        if path.parent.name == 'tables' and not path.name.startswith('n22_'):
            self.counts['historical_tables_not_numerically_validated'] += 1
            return
        if path.name in {'manuscript.tex', 'supplement.tex', 'apa-bibliography.tex'}:
            # Only body prose is numeric content; layout in preambles is not.
            start = text.find('\\begin{document}')
            if start >= 0:
                text = blank_span(text, 0, start)
            elif path.name == 'apa-bibliography.tex':
                return
        view = prose_view(text)
        index = self.table_indexes.get(name, self.fact_index)
        if path.name.startswith('n22_') and path.parent.name == 'tables' and name not in self.table_indexes:
            self.add('numbers', path, 1, 'Referenced n22 table lacks manifest fact IDs')
        for match in NUM.finditer(view):
            token = match.group()
            line = view.count('\n', 0, match.start()) + 1
            self.counts['numeric_tokens'] += 1
            reason = allowed_number(text, match)
            # This table's third column is the saved run index (1--8), not a
            # scientific estimate. Manifest hashes protect the full row.
            if path.name == 'n22_design_diagnostics.tex' and re.fullmatch(r'[1-8]', token):
                row = text.splitlines()[line - 1]
                prefix = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
                if row.startswith(('N22 &', 'N20 &')) and prefix.count('&') == 2:
                    reason = 'design-table run index (1--8)'
            ids = [] if reason else index.matching(token)
            # Explicit inequality only matches facts genuinely below the bound.
            before = view[max(0, match.start() - 20):match.start()]
            if re.search(r'(?:q|p)\s*(?:\\leq|<|<=)\s*$|\$<\s*$', before, re.I) and Decimal(token) == Decimal('.001'):
                ids = [f['id'] for f in index.facts if isinstance(f.get('value'), (int, float)) and not isinstance(f['value'], bool) and 0 <= f['value'] < .001 and re.search(r'(?:^|[._])(?:q|p)(?:[._]|$)', f['id'])]
                reason = None
            status = 'allowlist' if reason else 'ledger' if ids else 'unmatched'
            self.counts['numbers_' + status] += 1
            self.audit.append({'file': name, 'line': line, 'token': token, 'status': status,
                               'allowlist': reason, 'fact_id': ids[0] if ids else None,
                               'candidate_count': len(ids)})
            if status == 'unmatched':
                self.add('numbers', path, line, f'Unmatched numeric token {token!r}; context: {text.splitlines()[line - 1].strip()}')

    def check_text(self, path: Path, text: str):
        clean = comments_removed(text)
        for todo in re.finditer(r'\\todo\s*\{', clean):
            end = balanced_end(clean, clean.index('{', todo.start()))
            line = clean.count('\n', 0, todo.start()) + 1
            self.add('TODO', path, line, clean[todo.start():end] if end != -1 else 'Unclosed TODO argument', 'INFO' if end != -1 else 'ERROR', expected='')
            self.counts['todos'] += 1
        # Do not flag the macro definition as an author TODO or prose.
        clean = prose_view(clean)
        rules = [
            (r'\b1(?:,|\{,\})?833\b', 'Forbidden 1,833: no validated N=22 source'),
            (r'\b29\s*[/,]\s*15\s*[/,]\s*5\s*[/,]\s*0\b', 'Obsolete 29/15/5/0 primary result'),
            (r'conservative\s+(?:participant|[^.\n]{0,35}degrees of freedom)', 'Behavioral participant df are approximate'),
            (r'rule[- ]switching\s+flexibility', 'Use adaptation to changing sequence prevalence'),
            (r'coverage[- ]aware', 'Internal pipeline wording'),
            (r'\b(?:band[- ]pass|FIR|temporal[- ]filter(?:ing)?)\b', 'Dropped historical FIR/filter material (D4)'),
        ]
        for pattern, message in rules:
            for match in re.finditer(pattern, clean, re.I):
                line = clean.count('\n', 0, match.start()) + 1
                # Author disclosure TODO is explicitly allowed by Gate G1.
                prefix = clean[clean.rfind('\n', 0, match.start()) + 1:match.start()]
                if 'D4' in message and '\\todo{' in prefix and re.search(r'drop|remov|disclos', prefix, re.I):
                    continue
                self.add('stale', path, line, message)
        for match in re.finditer(r'\bprespecified\b', clean, re.I):
            context = clean[max(0, match.start() - 180):match.end() + 240]
            if re.search(r'2\s*(?:\\times|×|x)\s*2|filter|censor', context, re.I):
                self.add('stale', path, clean.count('\n', 0, match.start()) + 1, 'Prespecified 2×2 contradicts model-selection chronology')
        # Sentence boundaries exclude decimals; line/offset mapping stays exact.
        semantic = clean
        for pattern in [r'\\(?:begin|end)\{[^{}]*\}(?:\[[^\]]*\])?',
                        r'\\(?:section|subsection|subsubsection|paragraph)\*?\{[^{}]*\}',
                        r'\\caption\{[^{}]*\}',
                        r'\\(?:centering|FloatBarrier|clearpage)\b']:
            semantic = re.sub(pattern, blank, semantic)
        for paragraph in re.finditer(r'\S(?:(?!\n[ \t]*\n).)*', semantic, re.S):
            for match in sentences(paragraph.group()):
                sentence = match.group()
                if not sentence.strip():
                    continue
                line = clean.count('\n', 0, paragraph.start() + match.start() + len(sentence) - len(sentence.lstrip())) + 1
                n20 = re.search(r'\bN\s*=\s*20\b|\b20[- ]participant', sentence, re.I)
                if n20 and re.search(r'\bprimary\b|main[- ]text (?:reports|analysis)', sentence, re.I) and not re.search(r'sensitiv|historical|earlier|overlap|rather than|instead', sentence, re.I):
                    self.add('consistency', path, line, 'N=20 in a primary-result sentence: ' + sentence.strip())
                if path.stem in {'results', 'abstract', 'supplement'} and ROI.search(sentence) and RESULT.search(sentence) and not CORRECTION.search(paragraph.group()) and not re.search(r'diagnostics were available', sentence, re.I):
                    self.add('consistency', path, line, 'ROI-result sentence lacks a correction family in its paragraph (or descriptive-component qualification): ' + sentence.strip(), 'WARN')
                selection = re.search(r'(?:select\w*|cho\w*|prefer\w*|adopt\w*)[^.!?]{0,100}(?:significan|most findings|largest number|strongest results)', sentence, re.I)
                if selection and not re.search(r'\brather than\s+(?:by\s+)?(?:statistical\s+)?significan', selection.group(), re.I) and not re.search(r'not |no |never |post hoc|earlier|historical|previous|selection|does not|did not|without', sentence, re.I):
                    self.add('hierarchy', path, line, 'Possible selection by significance: ' + sentence.strip())
                if re.search(r'\bRT\b|reaction.time', sentence, re.I) and re.search(r'caus\w*|mediat\w*|explain\w*|account\w* for|remov\w*[^.!?]*confound', sentence, re.I) and re.search(r'adjust|attenuat|modulat', sentence, re.I) and not re.search(r'\bnot\b|\bno\b|cannot|doesn.t|isn.t|without|neither', sentence, re.I):
                    self.add('hierarchy', path, line, 'Possible causal interpretation of RT adjustment: ' + sentence.strip())

    def check_stale_inventory(self):
        md = self.root / 'docs/t9/facts_n22.md'
        archive = self.root / 'archive/t9_preflight_20261019T065220Z/t9_inputs.tar'
        if not md.exists() or not archive.exists():
            self.add('stale', md, 1, 'Stale inventory or frozen snapshot unavailable; contextual stale-number check incomplete', 'WARN', expected='')
            return
        locations: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        in_stale = False
        # Stream only headings and the small stale inventory, never load 29 MB whole.
        with md.open() as source:
            for row in source:
                if row.startswith('## Stale numbers:'):
                    in_stale = True
                if in_stale:
                    m = re.match(r'\| `([^`]+):(\d+)` \| `([^`]+)` \|', row)
                    if m:
                        locations[m[1]][int(m[2])].add(m[3])
                        self.counts['stale_inventory_entries'] += 1
        with tarfile.open(archive, 'r') as snapshot:
            for path, text in list(self.sources.items()):
                rel = self.relative(path)
                if rel not in locations or path.parent.name == 'tables':
                    continue
                try:
                    entry = snapshot.extractfile(rel)
                except KeyError:
                    self.add('stale', path, 1, 'Frozen stale source unavailable', 'WARN')
                    continue
                if entry is None:
                    continue
                old_lines = entry.read().decode().splitlines()
                anchors = set()
                for n, stale_tokens in locations[rel].items():
                    if n > len(old_lines):
                        continue
                    for m in sentences(prose_view(old_lines[n - 1])):
                        tokens = {t.group() for t in NUM.finditer(m.group())}
                        # A frozen paragraph can mix stale results with unchanged
                        # acquisition/design sentences. Anchor only stale tokens.
                        spelled = any(stale in m.group() for stale in stale_tokens if not NUM.fullmatch(stale))
                        if ((tokens & stale_tokens) or spelled) and len(normalize(m.group())) > 35:
                            anchors.add(normalize(m.group()))
                for match in sentences(prose_view(text)):
                    if normalize(match.group()) in anchors:
                        n = text.count('\n', 0, match.start() + len(match.group()) - len(match.group().lstrip())) + 1
                        self.add('stale', path, n, 'Unchanged numerical claim from frozen stale inventory: ' + normalize(match.group()))

    def check_manifest(self):
        manifest_path = self.manuscript / 'tables/n22_assets_manifest.json'
        if not self.manifest:
            self.add('assets', manifest_path, 1, 'Asset manifest missing', expected='')
            return
        for name, expected_hash in self.manifest.get('outputs', {}).items():
            asset = self.root / name
            if asset in self.sources or asset in self.assets:
                actual = self.hashes.get(name)
                if isinstance(expected_hash, dict):
                    expected_hash = expected_hash.get('sha256')
                if actual != expected_hash:
                    self.add('assets', asset, 1, 'Consumed asset differs from manifest SHA-256', expected='')
        facts_record = self.manifest.get('fact_ledger', {})
        if isinstance(facts_record, dict) and facts_record.get('sha256') != self.hashes.get('docs/t9/facts_n22.json'):
            self.add('numbers', manifest_path, 1, 'Manifest ledger SHA-256 differs from current ledger', expected='')

    def run(self):
        self.load_facts()
        for path in [self.manuscript / 'manuscript.tex', self.manuscript / 'supplement.tex']:
            if path.is_file():
                self.visit(path)
            else:
                self.add('assets', path, 1, 'Required manuscript source missing', expected='')
        for path in sorted((self.manuscript / 'sections').glob('*.tex')):
            if path not in self.sources:
                self.visit(path)
        for path in sorted((self.manuscript / 'tables').glob('n22_*.tex')):
            if path not in self.sources:
                self.visit(path)
        for path, text in list(self.sources.items()):
            if path.suffix == '.tex':
                self.check_numbers(path, text)
                if path.parent.name != 'tables':
                    self.check_text(path, text)
        methods = self.manuscript / 'sections/methods.tex'
        text = self.sources.get(methods, '')
        if not re.search(r'primary[^.]{0,120}(?:22|N\s*=\s*22)|(?:22|N\s*=\s*22)[^.]{0,120}primary', text, re.I):
            self.add('consistency', methods, 1, 'Missing explicit N=22 primary description')
        # REV-015 user decision: the learner-selected N=20 sample is reported only in the supplement.
        sections = sorted(p for p in self.sources if p.parent == self.manuscript / 'sections')
        main_tables = set()
        for section in sections:
            for name in re.findall(r'\\input\{tables/([^}]+)\}', self.sources[section]):
                main_tables.add(self.manuscript / 'tables' / (name if name.endswith('.tex') else name + '.tex'))
        n20 = re.compile(r'\bN\s*=\s*20\b|\b20[- ]participant|learner-selected|\bnon-?learner', re.I)
        for path in sections + sorted(t for t in main_tables if t in self.sources):
            body = re.sub(r'(?<!\\)%[^\n]*', '', self.sources[path])
            for match in n20.finditer(body):
                self.add('consistency', path, body.count('\n', 0, match.start()) + 1,
                         'Learner-selected N=20 sample mentioned in the main text (supplement only)')
        supplement = self.manuscript / 'supplement.tex'
        if not re.search(r'(?:20|N\s*=\s*20)[^.]{0,180}sensitiv|sensitiv[^.]{0,180}(?:20|N\s*=\s*20)', self.sources.get(supplement, ''), re.I):
            self.add('consistency', supplement, 1, 'Missing explicit N=20 sensitivity description in the supplement')
        self.check_stale_inventory()
        self.check_manifest()
        self.check_concurrent_changes()

    def check_concurrent_changes(self):
        for name, old_hash in list(self.hashes.items()):
            path = self.root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != old_hash:
                if name not in self.changed_files:
                    self.add('snapshot', path, 1, 'File changed during validation; rerun after owner finishes', 'WARN', expected=self.expected(path) or 'Concurrent owner edit; snapshot is not final')
                    self.changed_files.add(name)

    def build(self, in_place: bool = False):
        if in_place:
            target = self.manuscript
        else:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            target = Path(tempfile.mkdtemp(prefix='build-', dir=SCRATCH)) / 'manuscript'
            target.mkdir()
            # Copy only captured TeX, bibliography/configuration and consumed
            # images. Never traverse manuscript/data or manuscript/scripts.
            for path, text in self.sources.items():
                if path.is_relative_to(self.manuscript) and path.suffix == '.tex':
                    dest = target / path.relative_to(self.manuscript)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(text)
            for path in self.manuscript.iterdir():
                if path.is_file() and not path.is_symlink() and path.suffix in {'.bib', '.sty', '.cls', '.bst'}:
                    shutil.copyfile(path, target / path.name)
            for path in self.assets:
                dest = target / path.relative_to(self.manuscript)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        self.build_dir = target
        command = [sys.executable, str(self.root / 'code/reporting/build_pdfs.py'), '--manuscript-dir', str(target), '--force']
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=240)
            self.counts['build_driver_exit'] = done.returncode
            (target / 't9_build_driver.txt').write_text(done.stdout + done.stderr)
            if done.returncode:
                self.add('build', target / 't9_build_driver.txt', 1, f'build_pdfs.py exited {done.returncode}; inspect driver/build logs', expected='Phase 2/3 sources are unfinished' if self.phase == '2' else '')
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.add('build', target / 't9_build_driver.txt', 1, str(exc), expected='')
        for name in ('supplement', 'manuscript'):
            log = target / (name + '.log')
            if not log.is_file():
                self.add('build', log, 1, 'Final TeX log unavailable; document not verified', expected='Phase 2 build stopped before this document' if self.phase == '2' else '')
                continue
            for line, message in parse_log(log.read_text(errors='replace')):
                self.counts['build_log_diagnostics'] += 1
                if 'Overfull' in message:
                    self.counts['build_overfull_gt10pt'] += 1
                self.add('build', log, line, message, expected='Phase 2/3 source integration is unfinished' if self.phase == '2' else '')
            pdf = target / (name + '.pdf')
            if not pdf.is_file():
                self.add('build', log, 1, 'Fresh PDF missing', expected='Phase 2 build incomplete' if self.phase == '2' else '')
            elif 'done' in locals() and done.returncode == 0:
                self.counts['pdfs_built'] += 1
            if pdf.is_file() and in_place and 'done' in locals() and done.returncode == 0:
                render = self.root / 'docs/t9/render'
                render.mkdir(parents=True, exist_ok=True)
                try:
                    result = subprocess.run(['pdftoppm', '-r', '60', '-png', str(pdf), str(render / name)], capture_output=True, text=True, timeout=180)
                    if result.returncode:
                        self.add('build', pdf, 1, 'pdftoppm failed: ' + result.stderr, expected='')
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.add('build', pdf, 1, str(exc), expected='')
        self.check_concurrent_changes()

    def exit_code(self, strict: bool = False) -> int:
        return int(any(f.severity == 'ERROR' or (strict and f.severity == 'WARN') for f in self.findings))

    def write_report(self, destination: Path, strict: bool = False):
        groups = [('Defects / review findings', [f for f in self.findings if not f.expected and f.severity != 'INFO']),
                  ('Expected unfinished Phase 2 / 3 findings', [f for f in self.findings if f.expected]),
                  ('Author TODOs', [f for f in self.findings if f.check == 'TODO'])]
        counts = Counter(f.severity for f in self.findings)
        self.counts['tex_sources'] = sum(p.suffix == '.tex' for p in self.sources)
        self.counts['n22_tables'] = sum(p.parent.name == 'tables' and p.name.startswith('n22_') and p.suffix == '.tex' for p in self.sources)
        self.counts['graphics'] = len(self.assets)
        lines = ['# T9 manuscript validation', '', f'Generated: {datetime.now(timezone.utc).isoformat()}. Phase: {self.phase}. Strict: {strict}.', '',
                 f'**Exit status: {self.exit_code(strict)}.** {counts["ERROR"]} errors, {counts["WARN"]} warnings; {self.counts["todos"]} author TODOs. Expected findings remain failures; this is not a final manuscript acceptance.', '',
                 '## Scope and result counts', '',
                 'Read-only checks of main source, all section sources, supplement, recursively referenced input files, and consumed graphics. Historical table numbers are inventoried as historical assets, not certified as N=22. Null source slots remain unavailable, not numeric zero.', '',
                 f'Build directory: `{self.build_dir}`.' if self.build_dir else 'Build not requested; no PDF/build/render verification claimed.', '',
                 'Numeric checks compare exact integers and values rounded at displayed precision using Python formatting (including its floating-point boundary behavior), with ledger `formatted` fallback. Table candidates are restricted to manifest IDs. Prose membership can match unrelated facts and cannot certify the correct cohort, contrast, units, or interpretation; review fact IDs in owner reports. Paragraph-level family and sentence-level wording checks are heuristics and require review. TODOs are informational; --strict also fails warnings.', '',
                 '| Check counter | Count |', '| --- | ---: |']
        lines += [f'| {key} | {value} |' for key, value in sorted(self.counts.items())]
        lines += ['', 'Checks performed: numbers; stale claims and forbidden 1,833; asset existence and provenance hashes; N hierarchy and ROI correction-family wording; significance-selection/RT causal wording; TODO listing; final TeX logs when --build is used.', '']
        for title, findings in groups:
            lines += ['## ' + title, '']
            if not findings:
                lines += ['None.', '']
            for finding in findings:
                message = re.sub(r'\s+', ' ', finding.message).replace('`', "'")
                suffix = f' **Expected:** {finding.expected}.' if finding.expected else ''
                lines.append(f'- **{finding.severity} / {finding.check}** `{finding.file}:{finding.line}`: {message}.{suffix}')
            lines.append('')
        lines += ['## Documented numeric allowlist', '', 'Sources: `docs/t9/reports/t9-methods.md`, retained task/acquisition specification, and `docs/t9/reports/t9-tables.md`. These constants are contextual exceptions, not permission to accept arbitrary result values.', '']
        lines += [f'- {name}: {source}.' for name, _, source in DESIGN_RULES]
        lines += ['- Explicit p/q/FWE thresholds (.05, .0125, .001); .001 inequalities must have a qualifying ledger p/q below the bound.', '- Equation subscripts, declared trend weights, citation/asset identifiers, LaTeX layout parameters and macro argument indices are not empirical results.', '', '## Input snapshot SHA-256', '', 'Concurrent writers may change these after the run. Rerun the full validator and build after the owners finish; Phase 2 findings do not establish final page layout or scientific validity.', '', '| Input | SHA-256 |', '| --- | --- |']
        lines += [f'| `{name}` | `{digest}` |' for name, digest in sorted(self.hashes.items())]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('\n'.join(lines) + '\n')


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def blank_span(text: str, start: int, end: int) -> str:
    return text[:start] + ''.join('\n' if c == '\n' else ' ' for c in text[start:end]) + text[end:]


def parse_log(text: str) -> list[tuple[int, str]]:
    findings = []
    lines = text.splitlines()
    for n, line in enumerate(lines, 1):
        context = ' '.join(lines[n - 1:n + 2])
        if re.search(r'(?:Reference|Citation) .+ undefined|There were undefined (?:references|citations)|undefined citations', context, re.I) and ('Warning' in line or 'undefined' in line):
            findings.append((n, context))
        elif re.search(r'not found|I can.t find file|Undefined control sequence|^! (?:LaTeX|Package).*Error', line, re.I) and not re.search(r'\bInfo:', line):
            findings.append((n, context))
        overfull = re.search(r'Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)', line)
        if overfull and float(overfull[1]) > 10:
            findings.append((n, context))
    return findings


def self_test() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    class Checks(unittest.TestCase):
        def fixture(self, root):
            for folder in ['manuscript/sections', 'manuscript/tables', 'docs/t9', 'archive/t9_preflight_20261019T065220Z']:
                (root / folder).mkdir(parents=True, exist_ok=True)
            facts = [
                {'id': 'roi.mean', 'value': .1234, 'formatted': '+0.123'},
                {'id': 'roi.q', 'value': .0002, 'formatted': 'q < .001'},
                {'id': 'different.mean', 'value': .1244, 'formatted': '+0.124'},
                {'id': 'cohort.N22.n', 'value': 22, 'formatted': '22'},
                {'id': 'cohort.N20.n', 'value': 20, 'formatted': '20'},
            ]
            ledger = root / 'docs/t9/facts_n22.json'
            ledger.write_text(json.dumps(facts))
            (root / 'docs/t9/facts_n22.md').write_text('## Stale numbers: pre-launch manuscript inventory\n')
            with tarfile.open(root / 'archive/t9_preflight_20261019T065220Z/t9_inputs.tar', 'w'):
                pass
            (root / 'manuscript/manuscript.tex').write_text('\\begin{document}\n\\input{sections/results}\n\\end{document}')
            (root / 'manuscript/sections/methods.tex').write_text('The primary analysis uses N=22.')
            (root / 'manuscript/sections/results.tex').write_text('ROI estimate was +0.123, with BH correction over 48 ROI slots.\n\\input{tables/n22_test}')
            (root / 'manuscript/supplement.tex').write_text('\\begin{document}\nSupplement. The learner-selected N=20 sample is a sensitivity analysis.\n\\end{document}')
            table = root / 'manuscript/tables/n22_test.tex'
            table.write_text('Estimate & $+0.123$ & $<.001$ \\\\ \n')
            manifest = {'fact_ledger': {'sha256': hashlib.sha256(ledger.read_bytes()).hexdigest()},
                        'table_fact_ids': {'manuscript/tables/n22_test.tex': ['roi.mean', 'roi.q']},
                        'outputs': {'manuscript/tables/n22_test.tex': hashlib.sha256(table.read_bytes()).hexdigest()}}
            (root / 'manuscript/tables/n22_assets_manifest.json').write_text(json.dumps(manifest))

        def test_valid_fixture_and_read_only(self):
            SCRATCH.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='test-', dir=SCRATCH) as directory:
                root = Path(directory)
                self.fixture(root)
                before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
                v = Validator(root)
                v.run()
                self.assertEqual(v.exit_code(True), 0, v.findings)
                self.assertEqual(v.counts['numbers_unmatched'], 0)
                self.assertEqual(before, {p: p.read_bytes() for p in before})

        def test_table_scope_and_manifest_hash(self):
            with tempfile.TemporaryDirectory(prefix='test-', dir=SCRATCH) as directory:
                root = Path(directory)
                self.fixture(root)
                (root / 'manuscript/tables/n22_test.tex').write_text('$+0.124$')
                v = Validator(root)
                v.run()
                self.assertTrue(any('Unmatched numeric' in f.message for f in v.findings))
                self.assertTrue(any('differs from manifest' in f.message for f in v.findings))
                # +.124 is a real ledger value, but belongs to another asset.
                self.assertTrue(v.fact_index.matching('+0.124'))

        def test_stale_sentence_not_acquisition(self):
            import io
            with tempfile.TemporaryDirectory(prefix='test-', dir=SCRATCH) as directory:
                root = Path(directory)
                self.fixture(root)
                text = 'The historical study reported 160 analyzed runs. Each analyzed run contained 200 volumes.\n'
                path = root / 'manuscript/sections/results.tex'
                path.write_text(text)
                (root / 'docs/t9/facts_n22.md').write_text('## Stale numbers: pre-launch manuscript inventory\n| `manuscript/sections/results.tex:1` | `160` | `cohort.N22.n` | obsolete |\n')
                with tarfile.open(root / 'archive/t9_preflight_20261019T065220Z/t9_inputs.tar', 'w') as archive:
                    member = tarfile.TarInfo('manuscript/sections/results.tex')
                    member.size = len(text.encode())
                    archive.addfile(member, io.BytesIO(text.encode()))
                v = Validator(root)
                v.run()
                stale = [f for f in v.findings if f.check == 'stale']
                self.assertEqual(len(stale), 1, stale)
                self.assertNotIn('200 volumes', stale[0].message)

        def test_build_copy_excludes_data_scripts_aux(self):
            with tempfile.TemporaryDirectory(prefix='test-', dir=SCRATCH) as directory:
                root = Path(directory)
                self.fixture(root)
                (root / 'private').mkdir()
                (root / 'private/scientific.dat').write_text('untouched')
                for name in ['data', 'scripts']:
                    (root / 'manuscript' / name).symlink_to(root / 'private', target_is_directory=True)
                (root / 'manuscript/manuscript.aux').write_text('canonical aux untouched')
                builder = root / 'code/reporting/build_pdfs.py'
                builder.parent.mkdir(parents=True)
                builder.write_text("import sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index('--manuscript-dir')+1])\nfor n in ['supplement','manuscript']:\n (p/(n+'.log')).write_text('No warnings.\\n')\n (p/(n+'.pdf')).write_bytes(b'test PDF placeholder')\n")
                v = Validator(root)
                v.run()
                v.build()
                self.assertEqual(v.exit_code(True), 0, v.findings)
                self.assertTrue(v.build_dir.is_relative_to(SCRATCH))
                for name in ['data', 'scripts', 'manuscript.aux']:
                    self.assertFalse((v.build_dir / name).exists())
                self.assertEqual((root / 'private/scientific.dat').read_text(), 'untouched')
                self.assertEqual((root / 'manuscript/manuscript.aux').read_text(), 'canonical aux untouched')
                self.assertFalse((root / 'manuscript/manuscript.pdf').exists())

        def test_asset_cycle(self):
            with tempfile.TemporaryDirectory(prefix='test-', dir=SCRATCH) as directory:
                root = Path(directory)
                self.fixture(root)
                (root / 'manuscript/sections/results.tex').write_text('\\input{sections/results}')
                v = Validator(root)
                v.run()
                self.assertTrue(any('Recursive input cycle' in f.message for f in v.findings))

        def test_phase2_in_place_guard(self):
            done = subprocess.run([sys.executable, __file__, '--phase', '2', '--build', '--in-place'], capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn('final phase authorization', done.stderr)

        def test_rounding_sign_count_null(self):
            idx = FactIndex([{'id': 'roi.mean', 'value': -.195984916025, 'formatted': '-0.196'}, {'id': 'count', 'value': 1837, 'formatted': '1837'}, {'id': 'null', 'value': None, 'formatted': '—'}, {'id': 'tiny.q', 'value': .0004, 'formatted': 'q < .001'}])
            self.assertEqual(idx.matching('-0.196'), ['roi.mean', 'roi.mean'])
            self.assertFalse(idx.matching('+0.196'))
            self.assertTrue(idx.matching('1,837'))
            self.assertFalse(idx.matching('1,833'))
            self.assertFalse(idx.matching('0'))
            self.assertFalse(idx.matching('.001'))
            boundary = FactIndex([{'id': 'boundary', 'value': .0005, 'formatted': 'not numeric'}])
            self.assertTrue(boundary.matching(f'{.0005:.3f}'))
        def test_masks_and_offsets(self):
            text = '% 999999\n\\citep{x2025} \\input{tables/n22_a.tex}\nEstimate $-0.196$; 1,837 and 95\\% CI.\n\\setlength{\\tabcolsep}{3pt}\n'
            view = prose_view(text)
            self.assertEqual(len(text), len(view))
            self.assertEqual([m.group() for m in NUM.finditer(view)], ['-0.196', '1,837', '95'])
            self.assertEqual(view.count('\n', 0, view.index('-0.196')), 2)
            self.assertIn('95\\%', comments_removed('95\\% CI % comment'))
            self.assertNotIn('comment', comments_removed('line\\\\% comment'))
        def test_todo_nested(self):
            text = r'\todo{Check \textbf{Figure 1} now}'
            self.assertEqual(balanced_end(text, text.index('{')), len(text))
        def test_log_final_diagnostics(self):
            log = "LaTeX Warning: Citation 'foo' on page 1 undefined on input line 4.\nOverfull \\hbox (10.01pt too wide) in paragraph at lines 5--6\nOverfull \\hbox (9.99pt too wide)\n! LaTeX Error: File `lost.pdf' not found.\nPackage biblatex Info: ... file 'biblatex-dm.cfg' not found.\n"
            found = parse_log(log)
            self.assertEqual(len(found), 3)
            self.assertEqual([x[0] for x in found], [1, 2, 4])
        def test_integration_defects_and_phase(self):
            SCRATCH.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='test-', dir=SCRATCH) as directory:
                root = Path(directory)
                for folder in ['manuscript/sections', 'manuscript/tables', 'docs/t9']:
                    (root / folder).mkdir(parents=True, exist_ok=True)
                (root / 'docs/t9/facts_n22.json').write_text(json.dumps([{'id':'roi.mean', 'value': .1234, 'formatted': '+0.123'}]))
                (root / 'manuscript/manuscript.tex').write_text('\\begin{document}\n\\input{sections/results}\n\\end{document}')
                (root / 'manuscript/sections/methods.tex').write_text('The primary sample is N=22. N=20 is a sensitivity.')
                result = root / 'manuscript/sections/results.tex'
                result.write_text('Primary N=20 result. ROI estimate was +0.123.\nRT adjustment explains the effect.\n1,833 anticipations.\n\\input{tables/old}\n\\includegraphics{figures/missing}\n\\todo{Review {nested} item}\n')
                (root / 'manuscript/tables/old.tex').write_text('historical')
                (root / 'manuscript/supplement.tex').write_text('Band-pass results. A prespecified $2\\times2$ sensitivity.')
                v = Validator(root, '2')
                v.run()
                messages = ' '.join(f.message for f in v.findings)
                for expected in ['Historical asset', 'Missing includegraphics', 'Forbidden 1,833', 'Dropped historical', 'N=20 in a primary', 'Possible causal', 'ROI-result sentence',
                                 'mentioned in the main text', 'Missing explicit N=20 sensitivity description in the supplement']:
                    self.assertIn(expected, messages)
                self.assertTrue(all(f.expected for f in v.findings if f.file.endswith('results.tex') and f.check != 'TODO'))
                self.assertEqual(v.counts['todos'], 1)
                self.assertEqual(v.exit_code(), 1)
                v.write_report(root / 'docs/t9/check_report.md')
                self.assertIn('Expected unfinished', (root / 'docs/t9/check_report.md').read_text())
                result.write_text('modified')
                v.check_concurrent_changes()
                self.assertTrue(any(f.check == 'snapshot' for f in v.findings))
        def test_paragraph_family_scope(self):
            for family in ['BH correction over 48 ROI slots.', 'Joint BH-144 correction.',
                           'Separate 24-contrast BH family.', 'TFCE/FWE .0125 family.',
                           'FWE .05 family.', 'These changes are descriptive.',
                           'Pointwise intervals have no multiplicity adjustment.']:
                with self.subTest(family=family):
                    v = Validator(SCRATCH)
                    v.check_text(SCRATCH / 'results.tex',
                                 'Left BA6 estimate was +0.123.\n' + family + '\n\n'
                                 'Right BA6 estimate was +0.123.\n')
                    warnings = [f for f in v.findings if f.check == 'consistency']
                    self.assertEqual(len(warnings), 1, v.findings)
                    self.assertEqual(warnings[0].line, 4)
                    self.assertIn('Right BA6', warnings[0].message)

        def test_selection_negations_and_positive_control(self):
            cases = [
                ('These parcels were selected by definition rather than by statistical significance.', False),
                ('These parcels were not selected by statistical significance.', False),
                ('These parcels were selected without using statistical significance.', False),
                ('These parcels were selected by statistical significance.', True),
                ('The model was adopted because it yielded the largest number of significant results.', True),
            ]
            for text, flagged in cases:
                with self.subTest(text=text):
                    v = Validator(SCRATCH)
                    v.check_text(SCRATCH / 'supplement.tex', text)
                    self.assertEqual(any(f.check == 'hierarchy' for f in v.findings), flagged, v.findings)

        def test_sentence_offsets_after_float(self):
            text = ('\\begin{figure}[p]\n\\centering\n'
                    '\\includegraphics[width=\\linewidth]{figures/n22_test.pdf}\n'
                    '\\caption{Example.}\n\\label{fig:test}\n\\end{figure}\n'
                    '\\FloatBarrier\n\n\\subsection{Example}\n\n'
                    'Left BA6 estimate was +0.123.\n'
                    'Right BA6 estimate was +0.123.\n')
            v = Validator(SCRATCH)
            v.check_text(SCRATCH / 'results.tex', text)
            warnings = [f for f in v.findings if f.check == 'consistency']
            self.assertEqual([f.line for f in warnings], [11, 12], warnings)
            self.assertNotIn('[p]', ' '.join(f.message for f in warnings))

        def test_strict_warning(self):
            v = Validator(Path('/tmp'))
            v.add('consistency', Path('/tmp/test.tex'), 1, 'review', 'WARN')
            self.assertEqual(v.exit_code(), 0)
            self.assertEqual(v.exit_code(True), 1)
    return int(not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks)).wasSuccessful())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--strict', action='store_true', help='fail on warnings as well as errors; TODOs remain informational')
    parser.add_argument('--phase', choices=['2', 'final'], default='final', help='classify unfinished owners explicitly, without suppressing failures')
    parser.add_argument('--build', action='store_true', help='build fresh PDFs in /tmp/t9-t9-check/')
    parser.add_argument('--in-place', action='store_true', help='explicitly authorized Phase 4 only: build manuscript/ and render both PDFs')
    parser.add_argument('--self-test', action='store_true', help='run validator regression tests in /tmp/t9-t9-check/')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--report', type=Path, help='default: ROOT/docs/t9/check_report.md')
    parser.add_argument('--audit-json', type=Path, help='optional per-token membership evidence (prefer a /tmp/t9-t9-check/ path)')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.in_place and (not args.build or args.phase == '2'):
        parser.error('--in-place requires --build and final phase authorization')
    validator = Validator(args.root, args.phase)
    try:
        validator.run()
        if args.build:
            validator.build(args.in_place)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        validator.add('validator', args.root / 'docs/t9/check_report.md', 1, f'Input/check failure: {exc}', expected='')
    report = args.report or args.root / 'docs/t9/check_report.md'
    validator.write_report(report, args.strict)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps({'counts': dict(validator.counts), 'tokens': validator.audit, 'findings': [vars(f) for f in validator.findings]}, indent=2) + '\n')
    print(f'T9 validation exit={validator.exit_code(args.strict)}; report={report}; errors={sum(f.severity == "ERROR" for f in validator.findings)}; warnings={sum(f.severity == "WARN" for f in validator.findings)}; TODOs={validator.counts["todos"]}')
    return validator.exit_code(args.strict)


if __name__ == '__main__':
    raise SystemExit(main())
