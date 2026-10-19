#!/usr/bin/env python3
"""Descriptive anatomical labeling and nearest-neighbor displays of T2 maps."""
from common import *
import re,html,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_stat_map

def atlas_values(task,name,reference):
 p=task.record(Path('/usr/local/abin')/f'{name}.nii.gz');im=nib.load(p)
 text=html.unescape(re.sub(r'"\s*\n\s*"','',im.header.extensions[0].get_content().decode()))
 labels={int(value):label for label,value in re.findall(r'STRUCT="([^"]+)"\s+VAL="(\d+)"',text)};labels[0]='Unassigned'
 raw=nib.Nifti1Image(np.asarray(im.dataobj).squeeze().astype('int16'),im.affine)
 aligned=resample_to_img(raw,reference,interpolation='nearest',force_resample=True,copy_header=True)
 return np.asarray(aligned.dataobj).astype(int),labels

def main():
 t=Task('T2_within_run/anatomical_display');roots={'reference':'voxelwise_float32','censored':'voxelwise_censored_float32'}
 maps={model:nib.load(t.record(BASE/'T2_within_run'/folder/'signed_tfce_fwe05_t.nii.gz')) for model,folder in roots.items()}
 ref=maps['reference'];atlases={name:atlas_values(t,name,ref) for name in ['FS.afni.MNI2009c_asym','Brodmann_pijn_afni']}
 roi_masks={roi:image(t.record(ROOT/'resources/roi_masks'/f'{roi}.nii.gz'),ref,True)>0 for roi in ROIS};union=np.logical_or.reduce(list(roi_masks.values()))
 labels=[];overlap=[];components=[];summary={};cuts=set();maximum=0.
 for model,im in maps.items():
  data=np.asarray(im.dataobj);sig=data!=0;maximum=max(maximum,float(abs(data).max()));n=int(sig.sum())
  z=nib.affines.apply_affine(ref.affine,np.argwhere(sig))[:,2] if n else []
  cuts.update(np.unique(z).tolist());summary[model]=dict(n_corrected_voxels=n,n_in_original_roi_union=int((sig&union).sum()),n_positive=int((data>0).sum()),n_negative=int((data<0).sum()))
  for name,(values,lut) in atlases.items():
   for sign in [-1,1]:
    keep=data*sign>0;v,c=np.unique(values[keep],return_counts=True)
    for value,count in zip(v,c):labels.append(dict(model=model,atlas=name,sign=sign,label_id=value,label=lut.get(int(value),str(value)),n_voxels=count,fraction_of_all_corrected=count/n if n else np.nan))
  for roi,m in roi_masks.items():overlap.append(dict(model=model,roi=roi,n_positive=int(((data>0)&m).sum()),n_negative=int(((data<0)&m).sum()),n_total=int((sig&m).sum()),fraction_of_all_corrected=int((sig&m).sum())/n if n else np.nan))
  c=pd.read_csv(t.record(BASE/'T2_within_run'/roots[model]/'components.csv'))
  for _,row in c.iterrows():
   idx=np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine),[row.peak_x,row.peak_y,row.peak_z])).astype(int)
   components.append(dict(model=model,**row.to_dict(),**{name+'_peak_label':lut.get(int(values[tuple(idx)]),'Unassigned') for name,(values,lut) in atlases.items()}))
 csv(pd.DataFrame(labels,columns=['model','atlas','sign','label_id','label','n_voxels','fraction_of_all_corrected']),t.path/'atlas_overlap.csv');csv(overlap,t.path/'original_roi_overlap.csv');csv(components,t.path/'annotated_components.csv');writej(t.path/'summary.json',summary)
 template=ROOT/'data/templates/tpl-MNI152NLin2009cAsym';anat=nib.load(t.record(template/'tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'));am=nib.load(t.record(template/'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'))
 background=nib.Nifti1Image(np.asarray(anat.dataobj)*np.asarray(am.dataobj),anat.affine)
 cuts=sorted(cuts) or [39];columns=4;rows=math.ceil(len(cuts)/columns);maximum=max(maximum,1.)
 with PdfPages(t.path/'within_run_tfce_axial.pdf') as pdf:
  for model,im in maps.items():
   fig,axes=plt.subplots(rows,columns,figsize=(14,3.2*rows));flat=np.asarray(axes).ravel()
   for j,(ax,z) in enumerate(zip(flat,cuts)):
    display=plot_stat_map(im,bg_img=background,display_mode='z',cut_coords=[z],threshold=1e-12,cmap='RdBu_r',vmin=-maximum,vmax=maximum,symmetric_cbar=True,colorbar=(j==len(cuts)-1),annotate=True,draw_cross=False,black_bg=False,radiological=False,resampling_interpolation='nearest',axes=ax,figure=fig,title=f'z = {z:g} mm')
   for ax in flat[len(cuts):]:ax.set_visible(False)
   fig.suptitle(f'{model}: pooled Str2 − Str1; two-sided TFCE FWE ≤ .05\nPositive = Str2 > Str1; native 3-mm statistics; neurological display',fontsize=13)
   fig.subplots_adjust(top=.91,hspace=.18,wspace=.08);pdf.savefig(fig,bbox_inches='tight');fig.savefig(t.path/f'{model}_within_run_tfce.png',dpi=180,bbox_inches='tight');plt.close(fig)
 report='# T2 anatomical display and descriptive overlap\n\nNative 3-mm corrected t maps are displayed on the stored 1-mm MNI152NLin2009cAsym anatomy with nearest-neighbor resampling only. Every occupied axial level in either model is displayed for both models. Neurological orientation; red = Str2 > Str1, blue = Str2 < Str1. Each model uses its separate single-contrast voxelwise TFCE/FWE .05 threshold. No model-difference test or cluster-level p is inferred from threshold patterns.\n\n'
 for model,s in summary.items():report+=f'- {model}: {s["n_positive"]} positive, {s["n_negative"]} negative corrected voxels; {s["n_in_original_roi_union"]}/{s["n_corrected_voxels"]} overlap the union of the original 48 ROI masks.\n'
 report+='\nAtlas labels are **descriptive**, nearest-neighbor assignments at the analysis voxel centers. Unassigned voxels remain unassigned; they are not relabeled by proximity or assumed to be cortex/white matter. The full-brain search and original ROI family cover different tissue. These overlaps do not add ROI tests or validate a mechanism. Component sizes are descriptive 26-connected sets, not cluster-level inference.\n'
 for model in roots:
  a=pd.DataFrame(labels);a=a[(a.model==model)&(a.atlas=='FS.afni.MNI2009c_asym')].sort_values('n_voxels',ascending=False)
  report+='\n'+model+' atlas overlap: '+', '.join(f'{int(r["sign"]):+d} {r.label}: {int(r.n_voxels)}' for _,r in a.iterrows())+'.\n'
 t.finish(report,dict(passed=True,descriptive_only=True,all_occupied_z_levels=cuts,n_models=2,summary=summary,spatial_interpolation='nearest only',anatomical_labels_inference=False))
if __name__=='__main__':main()
