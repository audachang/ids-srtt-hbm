"""Exact TFCE computation on contiguous arrays; preserves values and iteration order.

NiftiMasker unmasking returns Fortran-layout volumes. The original TFCE routine
is called unchanged after a C-contiguous copy. Chunk seeds/counts are unchanged.
"""
import importlib
import numpy as np
pl=importlib.import_module('nilearn.mass_univariate.permuted_least_squares')
NATIVE_TFCE=pl.calculate_tfce
NATIVE_CHUNK=pl._permuted_ols_on_chunk

def contiguous_tfce(arr,*args,**kwargs):
 return NATIVE_TFCE(np.ascontiguousarray(arr),*args,**kwargs)

def contiguous_chunk(*args,**kwargs):
 old=pl.calculate_tfce;pl.calculate_tfce=contiguous_tfce
 try:return NATIVE_CHUNK(*args,**kwargs)
 finally:pl.calculate_tfce=old
