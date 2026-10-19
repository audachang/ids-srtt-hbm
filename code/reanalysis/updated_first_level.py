#!/usr/bin/env python3
"""Updated timing/sample fits; preserve full maps for new voxelwise checks."""
from updated_common import *
import concurrent.futures
import argparse
import time
import warnings
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.first_level.first_level import mean_scaling


def plan():
    inventory = pd.read_csv(OLD_BASE / 'T3_timing_repair/event_inventory.csv')
    changed = set(zip(inventory[inventory.changed].subject, inventory[inventory.changed].run))
    all_models = {(s, r) for s in N22 for r in range(1, 9) if s not in N20 or (s, r) in changed}
    jobs = {}
    for s in N22:
        for r in range(1, 9):
            if (s, r) in all_models:
                jobs[s, r] = list(VARIANTS)
            elif r in [3, 4, 5, 6]:
                jobs[s, r] = ['errors_omissions', 'rt_shared_modulator']
    jobs['sub-001', 1] = list(VARIANTS)
    jobs['sub-020', 6] = list(VARIANTS)
    return [(s, r, models, (s, r) not in changed and s in N20) for (s, r), models in sorted(jobs.items())]


def fit(job):
    subject, run, variants, unchanged = job
    out = BASE / 'first_level/runs' / subject / f'run-{run:02d}'
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'qc.json'
    producer = digest(__file__)
    shared = digest(Path(__file__).with_name('updated_common.py'))
    if marker.exists():
        q = json.loads(marker.read_text())
        assert q['script_sha256'] == producer and q['updated_common_sha256'] == shared
        for p, sha in q['output_sha256'].items():
            assert digest(out / p) == sha
        return q
    started = time.monotonic()
    bold, mask, conf, _ = paths(subject, run)
    events_path = event_path(subject, run)
    raw_path = ROOT / 'data/behavioral' / f'{subject[-3:]}_run_{run}_srtt_prob_new.csv'
    input_paths = [bold, mask, conf, events_path, raw_path, mask_path(subject)]
    input_paths += [paths(subject, r)[1] for r in range(1, 9)]
    input_paths += [ROOT / 'resources/roi_masks' / f'{roi}.nii.gz' for roi in ROIS]
    inputs = {str(p): dict(sha256=digest(p), bytes=p.stat().st_size) for p in input_paths}
    events = pd.read_csv(events_path, sep='\t')
    events['trial_type'] = events.trial_type.str.extract('(Ran|Str[12])$')[0]
    raw = pd.read_csv(raw_path)
    raw['omitted'] = ~np.isfinite(raw.p_response_rt) | raw.p_response_keys.astype(str).isin(['None', 'nan', ''])
    raw['wrong'] = (raw['correct.resp'] != 1) & ~raw.omitted
    confounds = pd.read_csv(conf, sep='\t')[CONFOUNDS].fillna(0).astype(float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        masker = NiftiMasker(mask_img=str(mask), smoothing_fwhm=4., standardize=False, detrend=False, t_r=2.)
        original_y = masker.fit_transform(str(bold))
    assert original_y.shape[0] == 200 and np.isfinite(original_y).all()
    reference = nib.load(ROOT / 'resources/roi_masks' / f'{ROIS[0]}.nii.gz')
    support = np.logical_and.reduce([image(paths(subject, r)[1], reference, True) > 0 for r in range(1, 9)])
    parcels = {roi: image(ROOT / 'resources/roi_masks' / f'{roi}.nii.gz', reference, True) > 0 for roi in ROIS}
    native = nib.load(mask)
    native_mask = np.asarray(native.dataobj) > 0
    values, diagnostics, reproductions = [], [], []
    for variant in variants:
        e = events.copy()
        hrf = 'spm + derivative' if variant == 'spm_derivative' else 'glover'
        if variant == 'errors_omissions':
            extras = []
            for flag, label in [('wrong', 'WrongPress'), ('omitted', 'Omission')]:
                g = raw[raw[flag]]
                if len(g):
                    extras.append(pd.DataFrame(dict(onset=g.stim_onset + (g.p_response_rt if flag == 'wrong' else 1.), duration=0., trial_type=label)))
            if extras:
                e = pd.concat([e, *extras], ignore_index=True)
        if variant in ['rt_modulator', 'rt_shared_modulator']:
            e['modulation'] = 1.
            valid = raw[~raw.omitted & raw.p_response_rt.gt(0) & raw.p_response_rt.lt(1)]
            groups = [('shared_run_centered', valid)] if variant == 'rt_shared_modulator' else valid.groupby('rule_group')
            extras = [pd.DataFrame(dict(onset=g.stim_onset, duration=0., trial_type='RT_' + c,
                                        modulation=g.p_response_rt - g.p_response_rt.mean())) for c, g in groups]
            e = pd.concat([e, *extras], ignore_index=True)
        design = make_first_level_design_matrix(np.arange(200) * 2., e, hrf_model=hrf, drift_model='cosine',
                                               high_pass=1/128, add_regs=confounds.to_numpy(), add_reg_names=list(confounds))
        keep = np.flatnonzero(np.loadtxt(mask_path(subject)).reshape(8, 200)[run - 1]) if variant == 'censored' else np.arange(200)
        design = design.iloc[keep]
        y, _ = mean_scaling(original_y[keep].copy(), axis=0)
        x = design.to_numpy()
        rank = int(np.linalg.matrix_rank(x))
        assert rank == x.shape[1] and len(keep) - rank >= 20
        folder = out / variant
        folder.mkdir(exist_ok=True)
        design.to_csv(folder / 'design_matrix.tsv', sep='\t', index=False)
        np.savetxt(folder / 'retained_indices.1D', keep, fmt='%d')
        labels, fits = run_glm(y, x, noise_model='ar1', bins=100, n_jobs=1)
        for condition in conditions(run):
            beta = np.zeros(y.shape[1])
            index = design.columns.get_loc(condition)
            for label, result in fits.items():
                beta[labels == label] = result.theta[index]
            array = np.zeros(native.shape, dtype=np.float32)
            array[native_mask] = beta
            im = nib.Nifti1Image(array, native.affine)
            if native.shape != reference.shape or not np.allclose(native.affine, reference.affine, atol=1e-5):
                im = resample_to_img(im, reference, interpolation='continuous', force_resample=True, copy_header=True)
            array = np.asarray(im.dataobj, dtype=np.float32)
            save32(array, reference, folder / f'{condition}_effect_size.nii.gz')
            if unchanged and variant in MODELS:
                path = FIRST / MODELS[variant] / 'level1/eventrelated' / subject / f'run-{run:02d}' / f'{condition}_effect_size.nii.gz'
                old = image(path, reference)
                np.testing.assert_allclose(array, old, atol=3e-5, rtol=3e-5)
                inputs[str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
                reproductions.append(dict(model=variant, condition=condition, max_difference=float(np.max(abs(array - old)))))
            for roi, parcel in parcels.items():
                observed = parcel & support
                value = float(array[observed].mean(dtype=np.float64)) if observed.any() and observed.sum() / parcel.sum() + 1e-12 >= .8 else np.nan
                values.append(dict(model=variant, subject=subject, run=run, condition=condition, roi=roi, value=value))
        task = conditions(run)
        nuisance = design[[c for c in design if c not in task]].to_numpy()
        adjusted = design[task].to_numpy() - nuisance @ np.linalg.lstsq(nuisance, design[task], rcond=None)[0]
        for index, condition in enumerate(task):
            target = adjusted[:, index]
            other = np.delete(adjusted, index, axis=1)
            residual = target - other @ np.linalg.lstsq(other, target, rcond=None)[0]
            diagnostics.append(dict(model=variant, subject=subject, run=run, condition=condition, n_rows=len(x),
                                    n_columns=x.shape[1], rank=rank, residual_df=len(x)-rank,
                                    nuisance_adjusted_vif=(target @ target)/(residual @ residual),
                                    n_wrong=int(raw.wrong.sum()), n_omissions=int(raw.omitted.sum())))
    csv(values, out / 'condition_values.csv')
    csv(diagnostics, out / 'design_qc.csv')
    q = dict(subject=subject, run=run, variants=variants, unchanged_timing=unchanged, inputs=inputs,
             script_sha256=producer, updated_common_sha256=shared,
             reference_reproduction=reproductions, elapsed_seconds=time.monotonic()-started,
             warnings=[str(w.message) for w in caught],
             output_sha256={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p != marker})
    writej(marker, q)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    assert nilearn.__version__ == '0.13.1'
    jobs = plan()
    if args.controls_only:
        jobs = [j for j in jobs if (j[0], j[1]) in [('sub-001', 1), ('sub-020', 6)]]
    t = Task('first_level')
    csv([dict(subject=s, run=r, models=' '.join(m), unchanged_timing=u) for s, r, m, u in jobs], t.path / ('control_plan.csv' if args.controls_only else 'job_plan.csv'))
    qcs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit, job) for job in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            q = future.result()
            qcs.append(q)
            print(f'{i}/{len(jobs)} {q["subject"]} r{q["run"]} {q["elapsed_seconds"]:.1f}s', flush=True)
    for q in qcs:
        t.inputs.update(q['inputs'])
        folder = t.path / 'runs' / q['subject'] / f'run-{q["run"]:02d}'
        for path in folder.rglob('*'):
            if path.is_file():
                t.record(path)
    if not args.controls_only:
        for name in ['condition_values', 'design_qc']:
            csv(pd.concat([pd.read_csv(t.path/'runs'/s/f'run-{r:02d}'/f'{name}.csv') for s,r,_,_ in jobs], ignore_index=True), t.path/f'{name}.csv')
        t.finish('# Updated first-level fits\n\nCorrected event timing for N20/N22, unchanged Glover/reference settings and calibrated v8-rule censor sensitivity. Six variants are fitted for added/changed runs and unchanged controls; error/shared-RT voxel maps are additionally fitted for every retained participant in runs 3–6. Archived reference/censored maps are reused only when their timing is byte-identical. Per-run inputs, full maps, designs, ranks, VIF and control reproduction checks are preserved. No manuscript changes.\n',
                 dict(passed=True, n_jobs=len(jobs), n_model_fits=sum(len(j[2]) for j in jobs),
                      n_control_reproduction_checks=sum(len(q['reference_reproduction']) for q in qcs),
                      max_control_voxel_difference=max([r['max_difference'] for q in qcs for r in q['reference_reproduction']] or [0])))


if __name__ == '__main__':
    main()
