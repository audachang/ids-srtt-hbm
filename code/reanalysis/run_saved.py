# The original script is withheld because it embeds individual observations.
# This file provides only the generic direct phase-comparison function so
# that released scripts importing it can run with approved restricted inputs.
import numpy as np
import pandas as pd
from common import ROIS, summarize, bh

def direct(values):
 rows=[]
 for (model,policy),d in values.groupby(['model','policy']):
  p=d.pivot(index=['subject','roi'],columns='stage',values='value')
  for a,b in [('early','middle'),('early','late'),('middle','late')]:
   for roi in ROIS:rows.append(dict(model=model,policy=policy,pair=f'{a}_minus_{b}',roi=roi,**summarize(p.xs(roi,level='roi')[a]-p.xs(roi,level='roi')[b])))
 d=pd.DataFrame(rows);d['q_three_pairs_144']=np.nan;d['q_within_pair_48']=np.nan
 for _,g in d.groupby(['model','policy']):
  assert len(g)==144;d.loc[g.index,'q_three_pairs_144']=bh(g.p)
 for _,g in d.groupby(['model','policy','pair']):d.loc[g.index,'q_within_pair_48']=bh(g.p)
 return d
