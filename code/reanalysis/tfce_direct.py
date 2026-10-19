"""Native one-sample permutation logic without per-permutation image objects."""
from tfce_union import union_tfce, LIBRARY
from tfce_sparse import graph, NATIVE_TFCE
import tfce_union
import numpy as np
from sklearn.utils import check_random_state
from nilearn.mass_univariate._utils import _return_score_threshs
from nilearn.mass_univariate.permuted_least_squares import t_score_with_covars_and_normalized_design


def direct_values(values):
    # The observed image call initializes the validated C ABI in each process.
    mask,indices,neighbors,_=graph()
    if tfce_union._library is None:
        field=np.zeros(mask.shape+(1,));field[...,0][mask]=values
        union_tfce(field,bin_struct=np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]))
    thresholds=np.ascontiguousarray(_return_score_threshs(values,'auto',True))
    scores=np.ascontiguousarray(values,dtype=np.float64);result=np.zeros(len(scores),dtype=np.float64)
    tfce_union._library.srtt_tfce(len(scores),scores,neighbors,len(thresholds),thresholds,result)
    return result


def direct_chunk(scores_original_data,tested_vars,target_vars,thread_id,threshold=None,
                 confounding_vars=None,masker=None,n_perm=10000,n_perm_chunk=10000,
                 intercept_test=True,two_sided_test=True,tfce=False,tfce_original_data=None,
                 random_state=None,verbose=0):
    assert intercept_test and two_sided_test and tfce and threshold is None and confounding_vars is None
    assert tested_vars.shape[1]==1
    mask,indices,_,_=graph()
    assert np.array_equal(np.asarray(masker.mask_img_.dataobj)>0,mask)
    assert target_vars.shape[1]==len(indices)
    rng=check_random_state(random_state)
    h0=np.empty((1,n_perm_chunk));h0_tfce=np.empty((1,n_perm_chunk))
    ranks=np.zeros((1,target_vars.shape[1]));tfce_ranks=np.zeros_like(ranks)
    for i in range(n_perm_chunk):
        # Same cumulative sign swaps and RNG calls as native Nilearn.
        target_vars=target_vars*(rng.randint(2,size=(len(target_vars),1))*2-1)
        scores=np.asfortranarray(t_score_with_covars_and_normalized_design(tested_vars,target_vars,None))
        h0[0,i]=np.nanmax(np.abs(scores));ranks+=h0[:,i,None]<np.abs(scores_original_data).T
        h0_tfce[0,i]=np.max(np.abs(direct_values(scores[:,0])))
        tfce_ranks+=h0_tfce[:,i,None]<np.abs(tfce_original_data.T)
    return ranks,h0,None,None,tfce_ranks,h0_tfce
