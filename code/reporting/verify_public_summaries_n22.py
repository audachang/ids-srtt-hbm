#!/usr/bin/env python3
"""Verify released aggregates only; never fit participant models.

MIT; copyright (c) 2026 Chiao-En Chan, Erik Chihhung Chang, Denise Hsien Wu.
Usage: python code/reporting/verify_public_summaries_n22.py RELEASE_ROOT
Requires NumPy, pandas, SciPy. Does not write any files.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats


def main(root):
    manifest=json.loads((root/'SHA256_manifest.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    expected={x['path'] for x in manifest}
    assert len(expected)==len(manifest),'Duplicate manifest entries'
    assert actual==expected|{'SHA256_manifest.json'},'Manifest completeness mismatch'
    for x in manifest:
        p=root/x['path']
        assert p.resolve().is_relative_to(root.resolve()),'Unsafe manifest path'
        assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
        assert p.stat().st_size<100_000_000,x['path']
    if (root/'CODE_REVIEW.csv').exists():
        review=pd.read_csv(root/'CODE_REVIEW.csv')
        for _,row in review.iterrows():
            p=root/row['path']
            assert row.credentials_or_tokens=='none found',row['path']+' credential review'
            if row.decision=='include verbatim':
                assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==row.source_sha256,row['path']+' code preservation'
                assert row.embedded_participant_data=='none found',row['path']+' embedded data review'
            else:assert not p.exists(),row['path']+' withheld code present'
    if (root/'FIGURE_REVIEW.json').exists():
        for rel,record in json.loads((root/'FIGURE_REVIEW.json').read_text()).items():
            assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==record['release_sha256'],rel+' reviewed figure hash'
            assert record['release_pages']==(47 if 'all_N22_specifications' in rel else record['original_pages']),rel+' page review'
    checks=0;families=0;blanks=0
    def close(a,b,label):
        nonlocal checks
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
        assert np.allclose(a,b,rtol=2e-7,atol=2e-9,equal_nan=True),label+' max difference '+str(np.nanmax(abs(a-b)))
        checks+=int(np.isfinite(a).sum())
    spec=json.loads((root/'AGGREGATE_CHECK_SPEC.json').read_text())
    for contract in spec['contracts']:
        name=contract['path'];f=pd.read_csv(root/name)
        v=f[[contract[k] for k in ['estimate','se','t','df','low','high'] if k in contract]].notna().all(axis=1)
        g=f[v];est=g[contract['estimate']];se=g[contract['se']];df=g[contract['df']]
        if 't' in contract:close(est/se,g[contract['t']],name+' t')
        for transform in contract.get('transforms',[]):
            close(np.exp(f[transform['source']]),f[transform['target']],name+' exponential transform')
        if contract.get('ci_method','t')=='t':
            margin=stats.t.ppf(.975,df)*se
            close(est-margin,g[contract['low']],name+' CI lower')
            close(est+margin,g[contract['high']],name+' CI upper')
        else:
            assert (g[contract['low']]<=g[contract['high']]).all(),name+' interval order'
        if 'suppression_n' in contract:
            low=f[contract['suppression_n']]<2
            fields=[x for x in ['mean','sd','se','ci_low','ci_high','cohen_dz','t','df','p'] if x in f]
            assert f.loc[low,fields].isna().all().all(),name+' unsuppressed singleton'
            blanks+=int(low.sum())*len(fields)
        for family in contract['families']:
            groups=f.groupby(family['group'],dropna=False) if family['group'] else [(None,f)]
            for _,g in groups:
                assert len(g)==family['slots'],(name,family,len(g))
                p=g[family['p']].fillna(1).to_numpy()
                q=stats.false_discovery_control(p,method='bh')
                close(q,g[family['q']],name+' BH '+family['q']);families+=1
    coverage=pd.read_csv(root/'RELEASE_COVERAGE.csv')
    assert len(coverage)==56 and coverage.source.nunique()==56,'Ledger source coverage'
    assert (coverage.ledger_entries==coverage.verifiable_entries+coverage.withheld_entries).all()
    assert coverage.ledger_entries.sum()==61572
    print(json.dumps(dict(status='PASS',manifest_files=len(manifest),contracts=len(spec['contracts']),arithmetic_values_checked=checks,BH_families_checked=families,suppressed_cells_checked=blanks,ledger_sources=56,ledger_entries=61572,verifiable_entries=int(coverage.verifiable_entries.sum()),limits='Aggregate arithmetic only; no participant models, new tests, or manuscript build verified.'),indent=2))


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: verify_public_summaries_n22.py RELEASE_ROOT')
    main(Path(sys.argv[1]).resolve())
