"""Component-tree TFCE with unchanged native thresholds and connectivity."""
from tfce_sparse import graph, NATIVE_TFCE, NATIVE_CHUNK
from pathlib import Path
import numpy as np
import ctypes
import importlib
from scipy.ndimage import generate_binary_structure
from nilearn.mass_univariate._utils import _return_score_threshs

LIBRARY=Path(__file__).with_name('libtfce_union.so')
pl=importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
_library=None


def union_tfce(arr4d,bin_struct,E=.5,H=2,dh='auto',two_sided_test=True):
    global _library
    assert E==.5 and H==2 and two_sided_test and arr4d.dtype==np.float64
    assert np.array_equal(bin_struct,generate_binary_structure(3,1))
    mask,indices,neighbors,_=graph()
    assert arr4d.shape[:3]==mask.shape and np.isfinite(arr4d).all() and not np.any(arr4d[~mask]!=0)
    if _library is None:
        _library=ctypes.CDLL(str(LIBRARY))
        doubles=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        _library.srtt_tfce.argtypes=[ctypes.c_int,doubles,ints,ctypes.c_int,doubles,doubles]
        _library.srtt_tfce.restype=None
    output=np.zeros_like(arr4d)
    for i in range(arr4d.shape[3]):
        field=arr4d[...,i]
        thresholds=np.ascontiguousarray(_return_score_threshs(field,dh,two_sided_test))
        values=np.ascontiguousarray(field.ravel()[indices]);result=np.zeros(len(indices))
        _library.srtt_tfce(len(indices),values,neighbors,len(thresholds),thresholds,result)
        full=np.zeros(mask.size);full[indices]=result;output[...,i]=full.reshape(mask.shape)
    return output


def union_chunk(*args,**kwargs):
    old=pl.calculate_tfce;pl.calculate_tfce=union_tfce
    try:return NATIVE_CHUNK(*args,**kwargs)
    finally:pl.calculate_tfce=old
