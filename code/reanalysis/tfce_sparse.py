"""Validated sparse implementation of Nilearn's unchanged TFCE definition."""
from pathlib import Path
import os
import ctypes
import importlib
import numpy as np
import nibabel as nib
from scipy.ndimage import generate_binary_structure
from nilearn.mass_univariate._utils import calculate_tfce as NATIVE_TFCE, _return_score_threshs

pl = importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
NATIVE_CHUNK = pl._permuted_ols_on_chunk
LIBRARY = Path(__file__).with_name('libtfce_sparse.so')
_state = None


def graph():
    global _state
    path = os.environ['SRTT_TFCE_MASK']
    if _state is not None and _state[0] == path:
        return _state[1:]
    mask = np.asarray(nib.load(path).dataobj) > 0
    indices = np.flatnonzero(mask.ravel())
    coords = np.argwhere(mask)
    lookup = np.full(mask.shape, -1, dtype=np.int32)
    lookup[mask] = np.arange(len(indices), dtype=np.int32)
    neighbors = np.full((len(indices), 6), -1, dtype=np.int32)
    for d, offset in enumerate([(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]):
        points = coords + offset
        valid = ((points >= 0) & (points < np.asarray(mask.shape))).all(axis=1)
        neighbors[valid, d] = lookup[tuple(points[valid].T)]
    library = ctypes.CDLL(str(LIBRARY))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    library.srtt_tfce.argtypes = [ctypes.c_int, array, ints, ctypes.c_int, array, array]
    library.srtt_tfce.restype = None
    _state = (path, mask, indices, neighbors, library)
    return _state[1:]


def sparse_tfce(arr4d, bin_struct, E=.5, H=2, dh='auto', two_sided_test=True):
    if E != .5 or H != 2 or not two_sided_test or arr4d.dtype != np.float64 or not np.array_equal(bin_struct, generate_binary_structure(3, 1)):
        if os.environ.get('SRTT_TFCE_STRICT') == '1':
            raise ValueError('Unsupported parameters for strict sparse TFCE')
        return NATIVE_TFCE(arr4d, bin_struct, E, H, dh, two_sided_test)
    mask, indices, neighbors, library = graph()
    assert arr4d.shape[:3] == mask.shape and np.isfinite(arr4d).all()
    if np.any(arr4d[~mask] != 0):
        if os.environ.get('SRTT_TFCE_STRICT') == '1':
            raise ValueError('Nonzero statistic outside the declared TFCE search mask')
        return NATIVE_TFCE(arr4d, bin_struct, E, H, dh, two_sided_test)
    output = np.zeros_like(arr4d)
    for i in range(arr4d.shape[3]):
        field = arr4d[..., i]
        thresholds = np.ascontiguousarray(_return_score_threshs(field, dh, two_sided_test))
        values = np.ascontiguousarray(field.ravel()[indices])
        result = np.zeros(len(indices), dtype=np.float64)
        library.srtt_tfce(len(indices), values, neighbors, len(thresholds), thresholds, result)
        full = np.zeros(mask.size, dtype=np.float64)
        full[indices] = result
        output[..., i] = full.reshape(mask.shape)
    return output


def sparse_chunk(*args, **kwargs):
    old = pl.calculate_tfce
    pl.calculate_tfce = sparse_tfce
    try:
        return NATIVE_CHUNK(*args, **kwargs)
    finally:
        pl.calculate_tfce = old
