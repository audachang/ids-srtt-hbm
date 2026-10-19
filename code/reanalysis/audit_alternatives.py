#!/usr/bin/env python3
"""Independently re-extract preserved alternative maps for T6."""
from common import *
ALTERNATIVES={
 'nocensor_bandpass':'FLA_nilearn_sensitivity_20260921/nocensor_bandpass',
 'censor_bandpass':'FLA_nilearn_sensitivity_20260921/censor_bandpass',
 'glover_derivative':'FLA_nilearn_model_alternatives_20260921/glover_derivative',
 'spm':'FLA_nilearn_model_alternatives_20260921/spm',
 'motion12_tissue':'FLA_nilearn_model_alternatives_20260921/motion12_tissue',
 'motion12_acompcor8':'FLA_nilearn_model_alternatives_20260921/motion12_acompcor8',
 'precision_weighted':'FLA_nilearn_model_alternatives_20260921/precision_weighted'}
if __name__=='__main__':
 t=Task('T6_robustness/saved_alternatives_audit');ref,masks,supports,coverage=atlas(t);rows=[]
 for model,rel in ALTERNATIVES.items():
  for stage,contrast in zip(STAGES,CONTRASTS):
   for s in SUBJECTS:
    data=image(t.record(FIRST/rel/'subject_contrasts/eventrelated'/s/f'{contrast}.nii.gz'),ref)
    assert np.isfinite(data).all()
    for roi,v in extract(data,s,masks,supports).items():rows.append(dict(model=model,policy='observed80',stage=stage,subject=s,roi=roi,value=v))
  print(model,flush=True)
 d=pd.DataFrame(rows);old=pd.read_csv(t.record(ROOT/'results/glm_volume/reporting_20261012/participant_roi_values.csv')).query('policy=="observed80"');check=d.merge(old,on=['model','policy','stage','subject','roi'],suffixes=('_new','_old'),validate='one_to_one')
 assert len(check)==7*20*48*4;np.testing.assert_allclose(check.value_new,check.value_old,atol=1e-10,rtol=1e-10,equal_nan=True)
 csv(d,t.path/'participant_roi_values.csv')
 t.finish('# T6 preserved alternatives audit\n\nRe-extracted all 560 preserved subject contrast maps for seven alternative models, using fixed participant eight-run observed80 support. Independently reproduced all saved participant values. Band-pass models remain flagged and are not tallied as robust specifications. These are new extractions from existing full-voxel fits, not newly fitted alternatives.',dict(passed=True,n_maps=560,n_participant_values=len(d),max_saved_difference=float(np.nanmax(abs(check.value_new-check.value_old)))))
