"""Shared, versioned reanalysis IO; historical source data are read only."""
from pathlib import Path
import os,sys,json,hashlib,platform
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,nibabel as nib,nilearn,scipy
from scipy import stats
from nilearn.image import resample_to_img
ROOT=Path(os.environ.get('SRTT_SOURCE_ROOT','/media/data3/chiaoen_srtt'))
DEST=Path(__file__).resolve().parents[2]
BASE=DEST/'results/glm_volume/reanalysis_20261018'
BEHAV=DEST/'results/behavioral/reanalysis_20261018'
TOP=DEST/'results/reanalysis_20261018'
SUBJECTS=(ROOT/'code/replication/config/subjects.txt').read_text().split()
ROIS=(ROOT/'code/replication/config/rois.txt').read_text().split()
STAGES=['initial','early','middle','late']
CONTRASTS=['Str1_minus_Ran_r12','Switch_r34_vs_r12_did','Switch_r56_vs_r12_did','Switch_r78_vs_r12_did']
MODELS={'reference':'FLA_nilearn_float32_20260814','censored':'FLA_nilearn_sensitivity_20260921/censor_hp128'}
FIRST=ROOT/'results/glm_volume/first_level'
POLICIES={'observed80':.8,'observed95':.95,'complete100':1.}
CONFOUNDS=['csf','white_matter','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def writej(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(d,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def csv(d,p):pd.DataFrame(d).to_csv(p,index=False,float_format='%.12g')
class Task:
 def __init__(self,name,behavior=False):
  self.path=(BEHAV if behavior else BASE)/name;self.path.mkdir(parents=True,exist_ok=True);self.inputs={}
 def record(self,p):
  p=Path(p)
  if str(p) not in self.inputs:self.inputs[str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}
  return p
 def finish(self,report,validation,status='complete'):
  sources=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.R'))
  for p in [DEST/'docs/REANALYSIS_PLAN_2026-10-18.md',ROOT/'code/replication/config/subjects.txt',ROOT/'code/replication/config/rois.txt']:self.record(p)
  writej(self.path/'input_manifest.json',{'inputs':self.inputs,'sources':{str(p):digest(p) for p in sources},'argv':sys.argv,'source_root':str(ROOT),'software':dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,nibabel=nib.__version__,nilearn=nilearn.__version__),'thread_limits':{'OPENBLAS_NUM_THREADS':os.environ['OPENBLAS_NUM_THREADS'],'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS']}})
  writej(self.path/'validation.json',{'status':status,**validation,'output_sha256':{p.name:digest(p) for p in self.path.iterdir() if p.is_file() and p.name not in ['input_manifest.json','validation.json','REPORT.md']}})
  (self.path/'REPORT.md').write_text(report+'\n')
  print(self.path.name,status,flush=True)
def summarize(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
 z=dict(n=n,mean=float(v.mean()) if n else np.nan,sd=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,cohen_dz=np.nan,t=np.nan,df=n-1,p=np.nan)
 if n<2:return z
 sd=v.std(ddof=1);se=sd/np.sqrt(n)
 if sd==0:return z|dict(sd=sd,se=se)
 half=stats.t.ppf(.975,n-1)*se;t=v.mean()/se
 return z|dict(sd=sd,se=se,ci_low=v.mean()-half,ci_high=v.mean()+half,cohen_dz=v.mean()/sd,t=t,p=2*stats.t.sf(abs(t),n-1))
def bh(v):return stats.false_discovery_control(np.where(np.isfinite(v),v,1.),method='bh')
def infer(d,keys,family):
 rows=[]
 for k,g in d.groupby(keys,sort=False,dropna=False):
  if not isinstance(k,tuple):k=(k,)
  rows.append(dict(zip(keys,k))|summarize(g.value))
 out=pd.DataFrame(rows);out['q']=np.nan
 for _,g in out.groupby(family,sort=False):
  assert len(g)==48,(family,len(g));out.loc[g.index,'q']=bh(g.p)
 return out

def paths(s,r):
 pre=f'{s}_task-srttprob_run-{r:02d}'
 f=ROOT/'data/derivatives/fmriprep'/s/'func'
 return (f/'MNI_space'/f'{pre}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',f/'Masks'/f'{pre}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',f/'Confounds'/f'{pre}_desc-confounds_timeseries.tsv',ROOT/'data/bids'/s/'func'/f'{pre}_eventrelated.tsv')
def conditions(r):return ['Ran','Str1'] if r<=2 else ['Ran','Str2'] if r>=7 else ['Ran','Str1','Str2']
def image(p,ref,nearest=False):
 im=nib.load(p)
 if im.shape[:3]!=ref.shape[:3] or not np.allclose(im.affine,ref.affine,atol=1e-5):
  im=resample_to_img(im,ref,interpolation='nearest' if nearest else 'continuous',force_resample=True,copy_header=True)
 return np.asarray(im.dataobj,dtype=float)
def atlas(task):
 ref=nib.load(task.record(ROOT/'resources/roi_masks'/f'{ROIS[0]}.nii.gz'))
 masks={r:image(task.record(ROOT/'resources/roi_masks'/f'{r}.nii.gz'),ref,True)>0 for r in ROIS}
 supports={};coverage=[]
 for s in SUBJECTS:
  support=np.ones(ref.shape,bool)
  for run in range(1,9):support&=image(task.record(paths(s,run)[1]),ref,True)>0
  supports[s]=support
  for r,m in masks.items():coverage.append(dict(subject=s,roi=r,fraction=float(support[m].mean()),n_roi=int(m.sum()),n_observed=int((support&m).sum())))
 return ref,masks,supports,pd.DataFrame(coverage)
def extract(data,s,masks,supports,cut=.8):
 out={}
 for r,m in masks.items():
  valid=m&supports[s]&np.isfinite(data)
  out[r]=float(data[valid].mean()) if valid.sum()/m.sum()+1e-12>=cut and valid.any() else np.nan
 return out

def savefig(fig,p):
 fig.savefig(str(p)+'.pdf',bbox_inches='tight');fig.savefig(str(p)+'.png',dpi=180,bbox_inches='tight')
 import matplotlib.pyplot as plt
 plt.close(fig)
def plot_intervals(ax,d,x,label,color):
 d=d.set_index(x).reindex(label)
 ax.errorbar(range(len(label)),d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o-',ms=3,lw=1,color=color)
