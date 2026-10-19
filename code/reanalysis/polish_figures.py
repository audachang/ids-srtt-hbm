"""Anatomically accurate labels for the reanalysis figures; no numerical refit."""
from common import *
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def label(roi):
 side='Left ' if roi.startswith('Left') else 'Right '
 name=roi.removeprefix('Left').removeprefix('Right')
 mapping={'SMA':'BA6','PrimaryMotor':'BA4','DLPFC9':'BA9','DLPFC46':'BA46','Cerebellum':'cerebellar cortex','AccumbensArea':'accumbens','ThalamusProper':'thalamus'}
 return side+mapping.get(name,re.sub(r'(?<!^)([A-Z])',r' \1',name).lower())

def main():
 out=BASE/'T5_components';cs=pd.read_csv(out/'phase_component_results.csv');primary=pd.read_csv(BASE/'T1_reference/reported_roi_results.csv')
 focus=['LeftSMA','RightAccumbensArea','LeftPutamen','LeftCaudate','LeftHippocampus','LeftCerebellum']
 def make(rois):
  fig,axes=plt.subplots(3,4,figsize=(14,10))
  for i,roi in enumerate(rois):
   ax=axes[i//2,(i%2)*2];other=axes[i//2,(i%2)*2+1]
   for c,color in [('structured','#4866a4'),('random','gray')]:
    d=cs[(cs.model=='reference')&(cs.roi==roi)&(cs.component==c)].set_index('stage').reindex(STAGES)
    ax.errorbar(range(4),d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o-',ms=3,label=c.title(),color=color)
   d=primary[primary.roi==roi].set_index('stage').reindex(STAGES)
   other.errorbar(range(4),d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=3,color='#16846d')
   other.plot(range(1,4),d['mean'].iloc[1:],c='#16846d');other.axvline(.5,c='.7',ls=':')
   for a in [ax,other]:a.axhline(0,c='.7',lw=.6);a.set_xticks(range(4),STAGES,fontsize=8);a.set_title(label(roi),fontsize=10)
   ax.set_ylabel('Two-run mean coefficient');other.set_ylabel('Initial / summed DiD')
  axes[0,0].legend(fontsize=8);fig.suptitle('Reference condition components and contrasts; pointwise 95% CI');fig.tight_layout();return fig
 savefig(make(focus),out/'condition_did_components')
 with PdfPages(out/'all_roi_condition_profiles.pdf') as pdf:
  for offset in range(0,48,6):fig=make(ROIS[offset:offset+6]);pdf.savefig(fig);plt.close(fig)
 trajectory=pd.read_csv(BASE/'T2_within_run/trajectory_results.csv');wi=pd.read_csv(BASE/'T2_within_run/within_run_roi_results.csv')
 focus=set(wi[(wi.model=='reference')&(wi.q<=.05)].roi)|{r for r in ROIS if any(k in r for k in ['SMA','Precentral','Caudate','Putamen','Pallidum','Accumbens','Hippocampus','Cerebellum'])}
 with PdfPages(BASE/'T2_within_run/trajectories.pdf') as pdf:
  for roi in sorted(focus):
   fig,axes=plt.subplots(1,2,figsize=(10,3.3),sharey=True)
   for ax,model in zip(axes,MODELS):
    for c,color in [('Ran','gray'),('Str1','#4866a4'),('Str2','#a44d79')]:
     d=trajectory[(trajectory.model==model)&(trajectory.roi==roi)&(trajectory.condition==c)].sort_values('run')
     ax.errorbar(d.run,d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o-',ms=3,label=c,color=color)
    ax.axhline(0,c='.7',lw=.6);ax.set_title(model);ax.set_xlabel('Run');ax.set_xticks(range(1,9))
   axes[0].set_ylabel('Coefficient vs implicit fitted baseline');axes[0].legend();fig.suptitle(label(roi)+' (pointwise 95% CI)');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
if __name__=='__main__':main()
