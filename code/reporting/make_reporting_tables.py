#!/usr/bin/env python3
"""Typeset current coverage-aware estimates and regenerate their anatomy display."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd
from augment_roi_reporting import label, fmt, make_table, STAGES, MODELS, MODEL_LABELS, digest
from coverage_significant_table import render as render_significant_table


def probability(value):
    if not np.isfinite(value): return '--'
    return '$<.001$' if value<.001 else fmt(value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[2])
    args=p.parse_args();dest=args.output_root.resolve();root=args.source_root.resolve()
    out=dest/'results/glm_volume/reporting_20261012';tables=dest/'manuscript/tables';figdir=dest/'manuscript/figures'
    d=pd.read_csv(out/'reported_roi_results.csv')
    render_significant_table(out/'reported_roi_results.csv', tables/'observed_significant_rois.tex')
    rows=[]
    for stage in STAGES:
        a=d[d.stage==stage];sig=a[a.significant_within_contrast]
        top=a.loc[a.t.abs().idxmax()]
        rows.append([stage.title(),str(len(sig)),str(sum(sig['mean']>0)),str(sum(sig['mean']<0)),
            str(int(a.n.min()))+'--'+str(int(a.n.max())),
            str(sum(a.p.notna())),probability(a.q_within_contrast.min()),
            label(top.roi)+', $t('+str(int(top.df))+')='+fmt(top.t,2)+'$'])
    text=[r'\begin{table}[t]',r'\centering',r'\begin{threeparttable}',
          r'\caption{Coverage-aware ROI results from the censored high-pass GLM}',
          r'\label{tab:reported}',r'\small',r'\setlength{\tabcolsep}{3pt}',
          r'\begin{tabular}{lrrrrrrp{0.28\linewidth}}',r'\toprule',
          r'Stage & $N_{q\leq.05}$ & Pos. & Neg. & $N$ range & Tested & Min. $q$ & Largest $|t|$ \\',r'\midrule']
    text += [' & '.join(r)+r' \\' for r in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\begin{tablenotes}[flushleft]',r'\footnotesize',
             r'\item Note. Effects are averaged over the participant-specific ROI portion observed in all eight fitting masks. At least 80\% of the atlas parcel must be observed. Nominal cohort size is 20; actual $N$ varies by ROI. Both frontal-pole slots are untestable and retained as $p=1$ in BH adjustment across 48 slots within each stage. There are 46 testable ROIs per stage. Intervals in the complete table are pointwise and unadjusted. BA6 is the complete atlas parcel rather than an SMA-specific mask.',
             r'\end{tablenotes}',r'\end{threeparttable}',r'\end{table}','']
    (tables/'observed_nilearn.tex').write_text('\n'.join(text))
    summary=pd.read_csv(out/'model_stage_summary.csv');rows=[]
    for model,model_label in zip(MODELS,MODEL_LABELS):
        a=summary[(summary.model==model)&(summary.policy=='observed80')].set_index('stage')
        rows.append([model_label]+[str(int(a.loc[s,'n_within48'])) for s in STAGES])
    text=[r'\begin{table}[t]',r'\centering',r'\small',
          r'\caption{Model sensitivity on fixed observed ROI support (80\% coverage).}',
          r'\label{tab:observedmodels}',r'\begin{tabular}{lrrrr}',r'\toprule',
          r'Model & Initial & Early & Middle & Late \\',r'\midrule']
    text += [' & '.join(r)+r' \\' for r in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',
             r'\begin{minipage}{0.95\linewidth}\footnotesize Counts use BH across 48 planned slots within model and stage. All models use the same participant-specific spatial support and eligibility. Precision weighting changes the within-phase estimand. Counts describe threshold crossings, not tests of differences between models.\end{minipage}',r'\end{table}','']
    (tables/'observed_model_counts.tex').write_text('\n'.join(text))
    rows=[]
    for policy,policy_label in [('legacy_full_parcel','Full-parcel mean ($N=20$)'),
             ('observed80',r'Observed support, 80\%'),('observed95',r'Observed support, 95\%'),
             ('complete100',r'Complete parcel, 100\%')]:
        a=summary[(summary.model=='censor_hp128')&(summary.policy==policy)].set_index('stage')
        rows.append([policy_label]+[str(int(a.loc[s,'n_within48'])) for s in STAGES])
    text=[r'\begin{table}[t]',r'\centering',r'\small',
          r'\caption{Sensitivity to the coverage requirement in the censored high-pass model.}',
          r'\label{tab:coveragepolicies}',r'\begin{tabular}{lrrrr}',r'\toprule',
          r'Extraction & Initial & Early & Middle & Late \\',r'\midrule']
    text += [' & '.join(r)+r' \\' for r in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',
             r'\begin{minipage}{0.95\linewidth}\footnotesize All corrections retain 48 planned slots. Stricter requirements change participant composition and the observed tissue being averaged, as well as precision. Full-parcel means include zeros outside fitting masks.\end{minipage}',r'\end{table}','']
    (tables/'coverage_policy_counts.tex').write_text('\n'.join(text))
    rows=[]
    for stage in STAGES:
        for _,r in d[d.stage==stage].iterrows():
            rows.append([stage.title(),label(r.roi),str(int(r.n)),fmt(r['mean']) if r.n>=2 else '--',fmt(r.sd),fmt(r.se),
                         '['+fmt(r.ci_low)+', '+fmt(r.ci_high)+']',fmt(r.cohen_dz),fmt(r.t),
                         probability(r.p),probability(r.q_within_contrast),probability(r.q_joint_stages)])
    make_table(tables/'coverage_all_roi_results.tex',
       'All ROI results on observed support (80\\% coverage requirement). Pointwise intervals are unadjusted.',
       ['Stage','ROI','$N$','Mean','SD','SE','95\\% CI','$d_z$','$t$','$p$','$q_{48}$','$q_{192}$'],rows,
       'tab:allcoverage',note='The 48 planned ROI slots are retained in BH adjustment; untestable slots enter correction as $p=1$. Means based on fewer than two participants are suppressed. Contrast estimates sum two run coefficients. Intervals and standardized effects remain conditional on the selected model and sample.')
    contrasts=pd.read_csv(out/'direct_stage_comparisons.csv');rows=[]
    for _,r in contrasts[contrasts.policy=='observed80'].iterrows():
        rows.append([r.pair.replace('_minus_',' -- ').title(),label(r.roi),str(int(r.n)),fmt(r['mean']),
            '['+fmt(r.ci_low)+', '+fmt(r.ci_high)+']',fmt(r.cohen_dz),fmt(r.t),probability(r.p),probability(r.q_three_pairs_144)])
    make_table(tables/'direct_stage_results.tex','Direct within-participant comparisons of sequence-change phases.',
        ['Comparison','ROI','$N$','Mean difference','95\\% CI','$d_z$','$t$','$p$','$q_{144}$'],rows,'tab:directstage')
    ref=nib.load(root/'resources/roi_masks/LeftSMA.nii.gz')
    anatomy=root/'data/templates/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
    cuts=[-36,-6,18,36,48,60]
    fig,axes=plt.subplots(4,6,figsize=(13,8.3),facecolor='black')
    fig.subplots_adjust(left=.12,right=.91,bottom=.04,top=.98,wspace=.01,hspace=.10)
    for ax in axes.flat:
        ax.set_facecolor('black')
    count={}
    for k,stage in enumerate(STAGES):
        a=d[(d.stage==stage)&d.significant_within_contrast]
        data=np.zeros(ref.shape,np.float32)
        for _,row in a.iterrows():
            m=np.asarray(nib.load(root/'resources/roi_masks'/f'{row.roi}.nii.gz').dataobj)>0
            take=m&(np.abs(row.t)>np.abs(data));data[take]=row.t
        im=nib.Nifti1Image(data,ref.affine);count[stage]=len(a)
        for j,z in enumerate(cuts):
            if len(a):
                display=plotting.plot_stat_map(im,bg_img=str(anatomy),display_mode='z',cut_coords=[z],
                    axes=axes[k,j],figure=fig,colorbar=False,threshold=1e-6,vmax=8,symmetric_cbar=True,
                    cmap='RdBu_r',annotate=False,draw_cross=False,black_bg=True,
                    resampling_interpolation='nearest')
            else:
                display=plotting.plot_anat(str(anatomy),display_mode='z',cut_coords=[z],axes=axes[k,j],
                    figure=fig,annotate=False,draw_cross=False,black_bg=True,colorbar=False)
            display.annotate(left_right=(k==0 and j==0),
                             positions=(k==0 or j==0),size=10,color='white')
        bounds=axes[k,0].get_position()
        fig.text(.012,(bounds.y0+bounds.y1)/2,stage.title()+f'\n({len(a)} ROIs)',fontsize=11,va='center',color='white')
        if len(a)==0: fig.text(.50,.006,'No within-stage FDR effects',ha='center',fontsize=9,color='white')
    import matplotlib.cm as cm
    cb=fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(-8,8),cmap='RdBu_r'),
                   cax=fig.add_axes([.93,.33,.012,.35]))
    cb.set_label('ROI t statistic (df varies)',fontsize=9,color='white')
    cb.ax.tick_params(colors='white',labelsize=9)
    cb.outline.set_edgecolor('white')
    fig.savefig(figdir/'observed_roi_axial.png',dpi=240,bbox_inches='tight',facecolor='black',edgecolor='black')
    fig.savefig(figdir/'observed_roi_axial.pdf',bbox_inches='tight',facecolor='black',edgecolor='black');plt.close(fig)
    metadata=dict(counts=count,source_csv_sha256=digest(out/'reported_roi_results.csv'),
                  anatomy_sha256=digest(anatomy),cuts_mm=cuts,orientation='neurological',
                  policy='observed80',color_scale=[-8,8],symmetric_cbar=True,
                  background='black',left_right_labels='Upper-left panel only',
                  coordinate_labels='First row and first column only',
                  script_sha256=digest(Path(__file__)),
                  png_sha256=digest(figdir/'observed_roi_axial.png'),
                  pdf_sha256=digest(figdir/'observed_roi_axial.pdf'),display='Reference parcels colored by observed-subset ROI-average t; constituent voxels are not separately tested.')
    (figdir/'observed_roi_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
