#!/usr/bin/env python3
"""Set printed figure sizes so condition and model labels remain legible."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from augment_roi_reporting import label, savefig, STAGES, MODELS, MODEL_LABELS, EXAMPLES

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
root=a.root;out=root/'results/glm_volume/reporting_20261012';figdir=root/'manuscript/figures'
components=pd.read_csv(out/'phase_component_results.csv');results=pd.read_csv(out/'reported_roi_results.csv')
fig,axes=plt.subplots(3,2,figsize=(7.1,8.4))
for row,roi in enumerate(['LeftSMA','RightDLPFC9','LeftHippocampus']):
    ax=axes[row,0]
    for component,c,name in [('structured','#8050a3','Structured'),('random','.4','Random')]:
        d=components[(components.roi==roi)&(components.component==component)].set_index('stage').reindex(STAGES)
        ax.errorbar(range(4),d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o-',
                    ms=4,lw=1,color=c,label=name)
    ax.set_ylabel('Mean run coefficient',fontsize=10)
    ax=axes[row,1];d=results[results.roi==roi].set_index('stage').reindex(STAGES)
    ax.errorbar(range(4),d['mean'],yerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=4,color='#218c72')
    ax.plot(range(1,4),d['mean'].to_numpy()[1:],color='#218c72',lw=1)
    ax.axvline(.5,ls=':',lw=.8,color='.7');ax.set_ylabel('Initial / summed DiD',fontsize=10)
    for ax in axes[row]:
        ax.set_xticks(range(4),['Initial','Early','Middle','Late'],fontsize=9)
        ax.tick_params(axis='y',labelsize=9);ax.axhline(0,lw=.7,color='.65');ax.grid(alpha=.15)
        ax.set_title(label(roi),fontsize=11)
axes[0,0].legend(fontsize=9,frameon=False)
fig.suptitle('Conditions and contrasts (pointwise 95% CI)',fontsize=13)
fig.tight_layout(rect=(0,0,1,.97));savefig(fig,figdir/'condition_did_components')
allresults=pd.read_csv(out/'all_model_roi_results.csv');allresults=allresults[allresults.policy=='observed80']
colors=['#607284','#8050a3','#218c72','#d37736']
for page in range(2):
    fig,axes=plt.subplots(3,1,figsize=(7.6,9.3),sharex=True)
    for ax,roi in zip(axes,EXAMPLES[page*3:page*3+3]):
        for j,stage in enumerate(STAGES):
            d=allresults[(allresults.roi==roi)&(allresults.stage==stage)].set_index('model').reindex(MODELS)
            y=np.arange(9)+(j-1.5)*.17
            ax.errorbar(d['mean'],y,xerr=np.vstack([d['mean']-d.ci_low,d.ci_high-d['mean']]),fmt='o',ms=3,
                        color=colors[j],lw=.8,label=stage.title())
        ax.set_yticks(range(9),MODEL_LABELS,fontsize=9);ax.invert_yaxis();ax.axvline(0,color='.65',lw=.7)
        ax.set_title(label(roi),fontsize=12);ax.grid(axis='x',alpha=.15)
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,ncol=4,fontsize=9,loc='upper center',bbox_to_anchor=(.65,1),frameon=False)
    axes[-1].set_xlabel('Summed contrast estimate (pointwise 95% CI)',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.965));savefig(fig,figdir/f'model_effect_intervals_page{page+1}')
coverage=pd.read_csv(out/'participant_roi_coverage.csv').pivot(index='roi',columns='subject',values='fraction')
fig,ax=plt.subplots(figsize=(7.8,10))
im=ax.imshow(coverage,cmap='viridis',vmin=0,vmax=1,aspect='auto')
ax.set_yticks(range(48),[label(x) for x in coverage.index],fontsize=9)
ax.set_xticks(range(20),coverage.columns,rotation=90,fontsize=8)
ax.set_title('Eight-run ROI coverage',fontsize=12)
cb=fig.colorbar(im,ax=ax,pad=.01,fraction=.03);cb.set_label('Fraction observed',fontsize=10)
fig.tight_layout();savefig(fig,figdir/'roi_coverage')
