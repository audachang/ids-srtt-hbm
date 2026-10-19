#!/usr/bin/env python3
"""Validate accelerated TFCE against complete, saved native permutation runs."""
from updated_common import *
import time
import importlib
from scipy.ndimage import generate_binary_structure, gaussian_filter
from nilearn.glm.second_level import non_parametric_inference
from tfce_sparse import sparse_tfce, sparse_chunk, NATIVE_TFCE, LIBRARY


def main():
    t = Task('permutation_backend_validation')
    for path in [Path(__file__).with_name('tfce_sparse.cpp'), Path(__file__).with_name('tfce_sparse.py'), LIBRARY]:
        t.record(path)
    mask = t.record(ROOT / 'results/glm_volume/whole_brain/common_intersection_mask.nii.gz')
    os.environ['SRTT_TFCE_MASK'] = str(mask)
    os.environ['SRTT_TFCE_STRICT'] = '1'
    im = nib.load(mask); m = np.asarray(im.dataobj) > 0
    rng = np.random.default_rng(20261018)
    rows = []
    for index in range(16):
        if index == 0:
            field = np.asarray(nib.load(t.record(OLD_BASE/'T2_within_run/voxelwise_float32/t.nii.gz')).dataobj, dtype=float)
        elif index == 1:
            field = np.zeros(m.shape, dtype=float)
        elif index == 2:
            field = m.astype(float) * 2.
        else:
            field = gaussian_filter(rng.normal(size=m.shape), .7) if index % 2 else rng.normal(size=m.shape)
            field[~m] = 0
            if index % 3 == 0:
                field = np.round(field, 1)
        field = np.ascontiguousarray(field[..., None])
        started = time.monotonic(); expected = NATIVE_TFCE(field, generate_binary_structure(3,1)); native_time = time.monotonic()-started
        started = time.monotonic(); actual = sparse_tfce(field, generate_binary_structure(3,1)); sparse_time = time.monotonic()-started
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-12)
        rows.append(dict(case=index, max_difference=float(np.max(abs(actual-expected))), native_seconds=native_time, sparse_seconds=sparse_time))
    csv(rows, t.path/'field_checks.csv')
    pl = importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
    sl = importlib.import_module('nilearn.glm.second_level.second_level')
    checks = []
    for model, folder, maps in [('reference','voxelwise_float32','pooled_subject_maps'),('censored','voxelwise_censored_float32','censored_pooled_maps')]:
        paths20 = [t.record(OLD_BASE/'T2_within_run'/maps/f'{s}_Str2_minus_Str1_r36_mean.nii.gz') for s in N20]
        prior = np.load(t.record(OLD_BASE/'T2_within_run'/folder/'raw_permutation_arrays.npz'))
        original_chunk, original_tfce, original_call = pl._permuted_ols_on_chunk, pl.calculate_tfce, sl.permuted_ols
        captured = {}
        def capture(*args, **kwargs):
            result = original_call(*args, **kwargs)
            captured.update(result)
            return result
        pl._permuted_ols_on_chunk = sparse_chunk
        pl.calculate_tfce = sparse_tfce
        sl.permuted_ols = capture
        started = time.monotonic()
        try:
            non_parametric_inference([str(p) for p in paths20], design_matrix=pd.DataFrame({'intercept':np.ones(20)}),
                second_level_contrast='intercept', mask=str(mask), model_intercept=True, n_perm=10000,
                two_sided_test=True, random_state=20261018, n_jobs=8, tfce=True, verbose=0)
        finally:
            pl._permuted_ols_on_chunk, pl.calculate_tfce, sl.permuted_ols = original_chunk, original_tfce, original_call
        differences = {}
        for key in prior.files:
            assert key in captured
            np.testing.assert_allclose(captured[key], prior[key], atol=1e-9, rtol=1e-12)
            differences[key] = float(np.max(abs(captured[key]-prior[key])))
        np.testing.assert_array_equal(captured['logp_max_tfce'], prior['logp_max_tfce'])
        np.savez_compressed(t.path/f'{model}_replayed_permutation_arrays.npz', **captured)
        checks.append(dict(model=model, n_perm=10000, seed=20261018, chunks=8, max_differences=differences,
                           corrected_logp_exact=True, elapsed_seconds=time.monotonic()-started))
        writej(t.path/'replay_checks.json', checks)
        print(model, '10000-permutation native replay passed', checks[-1]['elapsed_seconds'], flush=True)
    t.finish('# Exact sparse TFCE backend validation\n\nThe new compiled implementation runs the same 100 ascending Nilearn thresholds, six-neighbor connected components, E=.5/H=2, sign handling, and per-voxel accumulation. Thresholds come directly from Nilearn. It does not approximate TFCE or reduce permutations. Strict mode rejects unsupported inputs instead of silently using another method. Sixteen observed, zero, tied, signed and synthetic fields are checked against native Nilearn. Two full saved native analyses (reference and censored) are then replayed with all 10,000 permutations, unchanged seed and eight chunks. All observed statistics, all 20,000 null maxima, and corrected log-p maps are compared; corrected log-p values must match exactly. Compiled source and library hashes are recorded.\n',
                 dict(passed=True, n_field_checks=len(rows), n_full_permutation_replays=len(checks),
                      n_native_null_maxima_checked=20000, all_corrected_logp_exact=True, replay_checks=checks))


if __name__ == '__main__':
    main()
