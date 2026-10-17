#!/usr/bin/env python3
"""Read-only validation of alternative-model and FIR manuscript products."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR as M, recorded_path, source_matches

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def bh(p):
    p=np.asarray(p);order=np.argsort(p);ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    out=np.empty(len(p));out[order]=np.minimum(np.minimum.accumulate(ranked[::-1])[::-1],1);return out

def test_results(frame,values):
    for (model,contrast),group in frame.groupby(['model','contrast'],sort=False):
        raw=values[(values.model==model)&(values.contrast==contrast)]
        matrix=raw.pivot(index='subject',columns='roi',values='value').loc[:,group.roi].to_numpy()
        result=stats.ttest_1samp(matrix,0,axis=0)
        assert matrix.shape==(20,48)
        assert np.allclose(group['mean'],matrix.mean(axis=0))
        assert np.allclose(group.t,result.statistic)
        assert np.allclose(group.p_two_sided,result.pvalue)
        assert np.allclose(group.q_bh_within_contrast,bh(result.pvalue))
        assert np.array_equal(group.significant_fdr_0_05.to_numpy(),bh(result.pvalue)<=.05)

def validate():
    alt=M/'data/model_alternatives_20260921';fir=M/'data/fir_20260921'
    ap=json.loads((alt/'summary_provenance.json').read_text());fp=json.loads((fir/'provenance.json').read_text())
    assert ap['n_fits']==800 and ap['n_subject_maps']==480
    assert source_matches(ap['script_sha256'], '99_summarize_model_alternatives.py')
    assert source_matches(ap['source_fit_script_sha256'], '98_run_model_alternatives.py')
    assert fp['n_fits']==160 and fp['family_size']==3072 and fp['n_permutations']==10000
    assert source_matches(fp['script_sha256'], '101_summarize_fir_analysis.py')
    assert source_matches(fp['fit_script_sha256'], '100_run_fir_analysis.py')
    assert fp['figure_sha256']==sha(M/'figures/fir_roi_lag_heatmap.png')
    assert fp['profiles_pdf_sha256']==sha(fir/'fir_all_roi_timecourses.pdf')
    assert fp['specification_sha256']==sha(fir/'ANALYSIS_SPEC.md')
    for folder,n in [(alt,800),(fir,160)]:
        qc=pd.read_csv(folder/'run_qc.csv');assert len(qc)==n
        assert (qc['rank']==qc.n_columns).all() and (qc.residual_df>=10).all()
        manifest=pd.read_csv(folder/'subject_map_manifest.csv')
        assert len(manifest)==(480 if folder==alt else 80)
        for row in manifest.to_dict('records'):
            path=recorded_path(row['path']);assert row['sha256']==sha(path)
            image=nib.load(path);assert image.header.get_data_dtype()==np.dtype('float32')
            assert np.isfinite(np.asarray(image.dataobj)).all()
    av=pd.read_csv(alt/'subject_roi_values.csv');ar=pd.read_csv(alt/'roi_results.csv')
    assert len(ar)==6*4*48 and len(av)==6*4*48*20
    test_results(ar,av)
    z=np.load(fir/'subject_roi_kernels.npz');x=z['values'];assert x.shape==(20,4,48,16)
    assert np.isfinite(x).all()
    integrated=pd.read_csv(fir/'integrated_roi_results.csv')
    lags=pd.read_csv(fir/'lag_roi_results.csv');omni=pd.read_csv(fir/'temporal_omnibus_roi_results.csv')
    assert len(lags)==3072 and len(omni)==192
    dist=np.load(fir/'permutation_max_abs_t.npy');assert dist.shape==(10000,)
    for ci,c in enumerate(z['contrasts']):
        f=integrated[integrated.contrast==c].set_index('roi').loc[z['rois']]
        r=stats.ttest_1samp(x[:,ci].sum(axis=-1),0,axis=0)
        assert np.allclose(f.t,r.statistic,rtol=2e-5,atol=2e-5)
        assert np.allclose(f.q_bh_within_contrast,bh(r.pvalue),rtol=2e-5,atol=2e-5)
        of=omni[omni.contrast==c];assert np.allclose(of.q_bh_within_contrast,bh(of.p_temporal_max_t))
        for ri,name in enumerate(z['rois']):
            f=lags[(lags.contrast==c)&(lags.roi==name)].sort_values('delay')
            r=stats.ttest_1samp(x[:,ci,ri],0,axis=0)
            assert np.allclose(f.t,r.statistic)
            expected=(1+(dist[:,None]>=np.abs(r.statistic)[None,:]).sum(axis=0))/10001
            assert np.allclose(f.p_fwe_all_rois_lags_contrasts,expected)
    assert (M/'tables/model_alternatives.tex').is_file() and (M/'tables/fir_results.tex').is_file()
    print('PASS: 800 alternative fits, 160 FIR fits, 560 checksummed subject maps, ROI t/FDR recomputation, FIR temporal FWE probabilities, and figure provenance.')

if __name__=='__main__':validate()
