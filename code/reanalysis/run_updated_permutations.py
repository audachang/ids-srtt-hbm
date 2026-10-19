#!/usr/bin/env python3
"""Run or resume full permutations and complete native-precision exports."""
from updated_common import *
from updated_tfce import jobs,run
from postprocess_updated_tfce import one,main as finish
import concurrent.futures
import multiprocessing


def execute(job):
    cohort,model,contrast,_=job;folder=BASE/'voxelwise'/cohort/model/contrast
    if (folder/'raw_permutation_arrays.npz').exists():return one(job)
    try:return run(job)
    except AssertionError:
        # The launcher may reject float32 roundoff after saving all nulls.
        # Nothing is accepted unless the separate full raw/native audit passes.
        if not (folder/'raw_permutation_arrays.npz').exists():raise
        return one(job)
    finally:
        # Close the finished inner pool before its outer worker exits.
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True,kill_workers=True)


if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(execute,jobs()):print(result,flush=True)
    finish()
