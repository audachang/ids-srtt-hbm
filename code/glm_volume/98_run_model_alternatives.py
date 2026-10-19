#!/usr/bin/env python3
"""Exploratory FirstLevelModel alternatives; see frozen ANALYSIS_SPEC.md."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _layout import MANUSCRIPT_DIR as MANUSCRIPT, PROJECT_ROOT as ROOT, FIRST_LEVEL_DIR, fit_source_matches
OUT = FIRST_LEVEL_DIR / 'FLA_nilearn_model_alternatives_20260921'
REPORT = MANUSCRIPT / 'data/model_alternatives_20260921'
PRIMARY = FIRST_LEVEL_DIR / 'FLA_nilearn_float32_20260814'
PREVIOUS_FIT_SHA = '778751ea73388839919a414d53663e262e744588cd4c17d96e12bf888b38db49'

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

primary = load_module('primary_alternatives', '90_run_float32_eventrelated.py')
MOTION = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
MOTION12 = MOTION + [v + '_derivative1' for v in MOTION]
MODELS = {
    'primary_refit': ('glover', 'basic'),
    'glover_derivative': ('glover + derivative', 'basic'),
    'spm': ('spm', 'basic'),
    'motion12_tissue': ('glover', 'motion12_tissue'),
    'motion12_acompcor8': ('glover', 'motion12_acompcor8'),
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def confound_selection(path, mode):
    frame = pd.read_csv(path, sep='\t')
    if mode == 'basic':
        columns = list(primary.CONFOUND_COLS)
    elif mode == 'motion12_tissue':
        columns = MOTION12 + ['csf', 'white_matter']
    else:
        metadata = json.loads(path.with_suffix('.json').read_text())
        components = sorted(k for k, v in metadata.items()
                            if k.startswith('a_comp_cor_') and v.get('Mask') == 'combined'
                            and v.get('Retained') is True and k in frame)
        assert len(components) >= 1, (path, components)
        columns = MOTION12 + components[:8]
    missing = frame[columns].isna()
    # Only the expected first-frame motion derivatives may be undefined.
    for col in columns:
        inds = list(np.flatnonzero(missing[col]))
        assert not inds or (col.endswith('_derivative1') and inds == [0]), (path, col, inds)
    confounds = frame[columns].fillna(0).astype(float)
    assert np.isfinite(confounds.to_numpy()).all()
    return confounds

def fit_run(job):
    subject, run = job
    started = time.monotonic()
    markers=[OUT/v/'level1/eventrelated'/subject/f'run-{run:02d}'/'model_qc.json' for v in MODELS]
    if all(p.exists() for p in markers):
        for marker in markers:
            qc=json.loads(marker.read_text())
            assert fit_source_matches(qc['script_sha256'], Path(__file__).name) or qc['script_sha256'] == PREVIOUS_FIT_SHA
            assert all((marker.parent/f).exists() for f in qc['output_files'])
        return dict(subject=subject,run=run,elapsed_seconds=time.monotonic()-started,fits=len(MODELS),skipped=True)
    bold, mask, cpath, epath = primary.input_paths(subject, run)
    events = primary.load_events(epath, primary.CONDITIONS[run])
    # Reuse the loaded input across models; each fit uses FirstLevelModel directly.
    image = nib.load(bold)
    image = nib.Nifti1Image(np.asarray(image.dataobj), image.affine, image.header)
    records = []
    with threadpool_limits(limits=1):
        for variant, (hrf, confmode) in MODELS.items():
            target = OUT / variant / 'level1/eventrelated' / subject / f'run-{run:02d}'
            marker = target / 'model_qc.json'
            if marker.exists():
                qc = json.loads(marker.read_text())
                assert fit_source_matches(qc['script_sha256'], Path(__file__).name) or qc['script_sha256'] == PREVIOUS_FIT_SHA, 'Script changed: use a fresh output root'
                assert all((target / f).exists() for f in qc['output_files'])
                records.append(qc)
                continue
            confounds = confound_selection(cpath, confmode)
            assert len(confounds) == image.shape[-1]
            model = FirstLevelModel(t_r=2.0, hrf_model=hrf, drift_model='cosine',
                high_pass=1/128, smoothing_fwhm=4.0, signal_scaling=0,
                noise_model='ar1', mask_img=str(mask), standardize=False,
                minimize_memory=True, n_jobs=1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(image, events=events, confounds=confounds)
            design = model.design_matrices_[0]
            x = design.to_numpy()
            rank = int(np.linalg.matrix_rank(x))
            assert np.isfinite(x).all() and rank == x.shape[1]
            assert x.shape[0] - rank >= 10
            target.mkdir(parents=True, exist_ok=True)
            design.to_csv(target / 'design_matrix.tsv', sep='\t', index=False)
            output_files = []
            max_diff = 0.0
            for condition in primary.CONDITIONS[run]:
                effect = model.compute_contrast(condition, output_type='effect_size')
                effect = primary.as_float32_image(effect)
                array = np.asarray(effect.dataobj)
                assert np.isfinite(array).all()
                name = f'{condition}_effect_size.nii.gz'
                effect.to_filename(target / name)
                output_files.append(name)
                if variant == 'primary_refit':
                    old = nib.load(PRIMARY / 'level1/eventrelated' / subject / f'run-{run:02d}' / name)
                    assert np.allclose(array, np.asarray(old.dataobj), rtol=2e-5, atol=2e-5)
                    max_diff = max(max_diff, float(np.max(np.abs(array-np.asarray(old.dataobj)))))
            if variant == 'primary_refit':
                condition = 'Str1' if run <= 2 else 'Str2'
                result = model.compute_contrast(f'{condition} - Ran', output_type='all')
                for key in ['effect_size', 'effect_variance']:
                    output = primary.as_float32_image(result[key])
                    arr = np.asarray(output.dataobj)
                    assert np.isfinite(arr).all()
                    if key == 'effect_variance':
                        assert (arr >= 0).all()
                    name = f'structured_minus_random_{key}.nii.gz'
                    output.to_filename(target / name)
                    output_files.append(name)
            norms = np.linalg.norm(x, axis=0)
            qc = dict(subject=subject, run=run, model=variant, hrf=hrf,
                confounds=list(confounds), n_rows=x.shape[0], n_columns=x.shape[1],
                rank=rank, residual_df=x.shape[0]-rank,
                normalized_condition_number=float(np.linalg.cond(x/norms)),
                primary_reproduction_max_abs_difference=max_diff if variant=='primary_refit' else None,
                warnings=[str(w.message) for w in caught], output_files=output_files,
                input_paths=[str(p) for p in [bold, mask, cpath, epath]],
                confounds_sha256=digest(cpath), events_sha256=digest(epath),
                script_sha256=digest(__file__))
            write_json(marker, qc)
            records.append(qc)
    return dict(subject=subject, run=run, elapsed_seconds=time.monotonic()-started, fits=len(records))

def weighted_pair(effects, variances):
    valid = np.isfinite(variances) & (variances > 0)
    precision = np.divide(1.0, variances, out=np.zeros_like(variances), where=valid)
    total = precision.sum(axis=0)
    weights = np.divide(precision, total, out=np.zeros_like(precision), where=total>0)
    return 2 * np.sum(weights * effects, axis=0), weights, valid

def build_weighted(subject):
    reference = primary.canonical_reference()
    phases = []
    weight_records = []
    for pair in [(1,2), (3,4), (5,6), (7,8)]:
        effects, variances = [], []
        for run in pair:
            folder = OUT / 'primary_refit/level1/eventrelated' / subject / f'run-{run:02d}'
            for key, collection in [('effect_size', effects), ('effect_variance', variances)]:
                img = nib.load(folder / f'structured_minus_random_{key}.nii.gz')
                if primary.grid_signature(img) != primary.grid_signature(reference):
                    img = resample_to_img(img, reference, interpolation='linear',
                                          force_resample=True, copy_header=True)
                collection.append(np.asarray(img.dataobj, dtype=float))
        weighted, weights, valid = weighted_pair(np.array(effects), np.array(variances))
        phases.append(weighted)
        outdir = OUT / 'precision_weighted/phase_weights' / subject
        outdir.mkdir(parents=True, exist_ok=True)
        primary.as_float32_image(reference, weights[0]).to_filename(outdir / f'r{pair[0]}{pair[1]}_first_run_weight.nii.gz')
        both = valid.all(axis=0)
        weight_records.append(dict(subject=subject, runs=list(pair), both_valid=int(both.sum()),
            one_valid=int((valid.sum(axis=0)==1).sum()), neither_valid=int((~valid.any(axis=0)).sum()),
            first_run_weight_quantiles=np.quantile(weights[0][both], [0,.05,.5,.95,1]).tolist()))
    folder = OUT / 'precision_weighted/subject_contrasts/eventrelated' / subject
    folder.mkdir(parents=True, exist_ok=True)
    for i, contrast in enumerate(primary.PRIMARY_CONTRASTS):
        value = phases[0] if i==0 else phases[i]-phases[0]
        assert np.isfinite(value).all()
        primary.as_float32_image(reference, value).to_filename(folder / f'{contrast}.nii.gz')
    write_json(OUT / 'precision_weighted/provenance' / f'{subject}.json', weight_records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--runs', nargs='+', type=int, default=list(range(1,9)))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert nilearn.__version__ == '0.13.1'
    subjects = args.subjects or primary.read_subjects()
    assert set(subjects) <= set(primary.read_subjects())
    assert set(args.runs) <= set(range(1,9))
    # Verify weighting and scaling algebra, including covariance-derived variances.
    e=np.array([[2.,4.],[6.,8.]])
    assert np.allclose(weighted_pair(e,np.ones_like(e))[0],e.sum(axis=0))
    assert np.allclose(weighted_pair(e,np.array([[1.,1.],[3.,3.]]))[0],[6.,10.])
    write_json(REPORT / 'execution.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        software=dict(python=platform.python_version(), nilearn=nilearn.__version__,
                      numpy=np.__version__, nibabel=nib.__version__),
        models=MODELS, script_sha256=digest(__file__), specification_sha256=digest(REPORT/'ANALYSIS_SPEC.md'),
        subjects=subjects, runs=args.runs, workers=args.workers))
    jobs=[(s,r) for s in subjects for r in args.runs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fit_run,job) for job in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures),1):
            print(json.dumps(dict(completed=count,total=len(jobs),**future.result())),flush=True)
    if sorted(args.runs)==list(range(1,9)):
        for subject in subjects:
            for variant in MODELS:
                primary.build_subject_contrasts(subject, OUT/variant)
            build_weighted(subject)
            print('Built contrasts: '+subject,flush=True)
    print('Completed requested fits.',flush=True)

if __name__=='__main__':
    main()
