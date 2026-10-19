#!/usr/bin/env python3
"""Descriptive atlas labels and complete axial displays of updated TFCE maps."""
from updated_common import *
from visualize_within_run import atlas_values
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_stat_map


def main():
    t=Task('anatomical_display');summary=pd.read_csv(t.record(BASE/'voxelwise/permutation_summary.csv'));assert len(summary)==24
    ref=nib.load(t.record(BASE/'voxelwise_inputs/common_N22_mask.nii.gz'))
    atlases={name:atlas_values(t,name,ref) for name in ['FS.afni.MNI2009c_asym','Brodmann_pijn_afni']}
    rois={roi:image(t.record(ROOT/'resources/roi_masks'/f'{roi}.nii.gz'),ref,True)>0 for roi in ROIS};union=np.logical_or.reduce(list(rois.values()))
    annotations=[];overlap=[];roi_overlap=[];totals=[];maps={};cuts=set()
    for _,row in summary.iterrows():
        key={k:row[k] for k in ['cohort','model','contrast']};folder=BASE/'voxelwise'/row.cohort/row.model/row.contrast
        im=nib.load(t.record(folder/'signed_tfce_corrected_t.nii.gz'));data=np.asarray(im.dataobj);sig=data!=0;n=int(sig.sum())
        counts={**key,'n_corrected_voxels':n,'n_positive':int((data>0).sum()),'n_negative':int((data<0).sum()),'n_in_original_roi_union':int((sig&union).sum())};totals.append(counts)
        for name,(values,lut) in atlases.items():
            for sign in [-1,1]:
                v,c=np.unique(values[data*sign>0],return_counts=True)
                for val,count in zip(v,c):overlap.append({**key,'atlas':name,'sign':sign,'label_id':int(val),'label':lut.get(int(val),'Unassigned'),'n_voxels':int(count),'fraction_of_all_corrected':count/n if n else np.nan})
        for roi,m in rois.items():roi_overlap.append({**key,'roi':roi,'n_positive':int(((data>0)&m).sum()),'n_negative':int(((data<0)&m).sum()),'n_total':int((sig&m).sum())})
        components=pd.read_csv(t.record(folder/'components.csv'))
        for _,c in components.iterrows():
            idx=np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine),[c.peak_x,c.peak_y,c.peak_z])).astype(int)
            annotations.append({**key,**c.to_dict(),**{name+'_peak_label':lut.get(int(v[tuple(idx)]),'Unassigned') for name,(v,lut) in atlases.items()}})
        if row.contrast=='pooled_r36_Str2_minus_Str1' and (row.cohort=='N22' or row.model in ['reference','censored']):
            maps[(row.cohort,row.model)]=im
            if n:cuts.update(np.unique(nib.affines.apply_affine(ref.affine,np.argwhere(sig))[:,2]).tolist())
    csv(totals,t.path/'anatomical_summary.csv');csv(overlap,t.path/'atlas_overlap.csv');csv(roi_overlap,t.path/'original_roi_overlap.csv');csv(annotations,t.path/'annotated_components.csv')
    template=ROOT/'data/templates/tpl-MNI152NLin2009cAsym'
    a=nib.load(t.record(template/'tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'));am=nib.load(t.record(template/'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'))
    bg=nib.Nifti1Image(np.asarray(a.dataobj)*np.asarray(am.dataobj),a.affine)
    cuts=sorted(cuts) or [39.];vmax=max(1.,max(float(abs(np.asarray(im.dataobj)).max()) for im in maps.values()))
    with PdfPages(t.path/'pooled_TFCE_all_occupied_axial.pdf') as pdf:
        for (cohort,model),im in maps.items():
            for page in range(math.ceil(len(cuts)/12)):
                levels=cuts[page*12:(page+1)*12];fig,axes=plt.subplots(math.ceil(len(levels)/4),4,figsize=(14,3*math.ceil(len(levels)/4)),squeeze=False)
                for j,(ax,z) in enumerate(zip(axes.ravel(),levels)):
                    plot_stat_map(im,bg_img=bg,display_mode='z',cut_coords=[z],threshold=1e-12,cmap='RdBu_r',vmin=-vmax,vmax=vmax,symmetric_cbar=True,
                        colorbar=j==len(levels)-1,annotate=True,draw_cross=False,black_bg=False,radiological=False,resampling_interpolation='nearest',axes=ax,figure=fig,title=f'z={z:g} mm')
                for ax in axes.ravel()[len(levels):]:ax.set_visible(False)
                fig.suptitle(f'{cohort}, {model}: pooled Str2 − Str1; two-sided TFCE/FWE .05\nNeurological; native 3-mm statistics; nearest-neighbor display',fontsize=12)
                fig.subplots_adjust(top=.86 if len(levels)<=4 else .92,hspace=.2,wspace=.08)
                pdf.savefig(fig,bbox_inches='tight');fig.savefig(t.path/f'{cohort}_{model}_axial_page{page+1}.png',dpi=140,bbox_inches='tight');plt.close(fig)
    report='# Updated descriptive anatomy and displays\n\nAll 24 corrected maps receive atlas and original-48-ROI overlap summaries and component peak labels. Atlas labels are nearest-neighbor assignments at native 3-mm analysis voxel centers. Unassigned voxels remain unassigned and are not assigned tissue by proximity. Components are descriptive 26-connected sets, not cluster-level inference; label overlap does not establish a mechanism or introduce a new ROI test.\n\nThe pooled-contrast PDF shows every occupied axial level across six maps: N20 reference/censored and N22 reference/censored/errors/shared-RT. Each uses its separate single-contrast voxelwise FWE .05 family, the same N22 search mask and sign convention Str2>Str1. Native statistics are displayed with nearest-neighbor resampling onto stored MNI152NLin2009cAsym 1-mm anatomy; neurological orientation. Differences in threshold patterns are not tests of model or cohort differences.\n\n| Sample | Model | Positive | Negative | In original ROI union |\n|---|---|---:|---:|---:|\n'
    for row in totals:
        if row['contrast']=='pooled_r36_Str2_minus_Str1':report+=f'| {row["cohort"]} | {row["model"]} | {row["n_positive"]} | {row["n_negative"]} | {row["n_in_original_roi_union"]}/{row["n_corrected_voxels"]} |\n'
    t.finish(report,dict(passed=True,n_maps_annotated=24,n_maps_displayed=6,all_occupied_axial_levels=cuts,descriptive_only=True,interpolation='nearest'))


if __name__=='__main__':main()
