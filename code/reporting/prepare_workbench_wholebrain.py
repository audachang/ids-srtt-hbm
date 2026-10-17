#!/usr/bin/env python3
"""Project corrected volume results onto coordinate-matched cohort surfaces."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import nibabel as nib
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=args.root.resolve()
    out=root/'results/glm_volume/whole_brain/workbench_20261013'
    out.mkdir(parents=True,exist_ok=True)
    template=root/'resources/fs_LR_32';fmriprep=root/'data/derivatives/fmriprep'
    subjects=(root/'code/replication/config/subjects.txt').read_text().split()
    assert len(subjects)==20 and len(set(subjects))==20
    manifest={};commands=[];lock=threading.Lock()
    env=os.environ.copy();env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1';env['OMP_NUM_THREADS']='1'
    def source(path):
        with lock:
            manifest[str(path)]=digest(path)
        return path
    def run(command):
        command=list(map(str,command))
        result=subprocess.run(command,env=env,capture_output=True,text=True)
        with lock: commands.append({'argv':command,'returncode':result.returncode,'stderr':result.stderr})
        if result.returncode: raise RuntimeError(result.stderr or result.stdout)
        return result.stdout
    def participant(subject):
        folder=out/'surfaces'/subject;folder.mkdir(parents=True,exist_ok=True)
        anat=fmriprep/subject/'anat';arrays=[];images=[];paths=[]
        for hemi in ['L','R']:
            sphere=source(anat/f'{subject}_hemi-{hemi}_space-fsLR_desc-msmsulc_sphere.surf.gii')
            target=source(template/f'fs_LR.32k.{hemi}.sphere.surf.gii')
            for kind in ['white','midthickness','pial']:
                native=source(anat/f'{subject}_hemi-{hemi}_{kind}.surf.gii')
                resampled=folder/f'{hemi}.{kind}.T1w.32k.surf.gii'
                run(['wb_command','-surface-resample',native,sphere,target,'BARYCENTRIC',resampled])
                image=nib.load(resampled);coords=image.darrays[0].data.astype(float)
                arrays.append(coords);images.append(image)
                paths.append(folder/f'{hemi}.{kind}.MNI152NLin2009cAsym.32k.surf.gii')
        # ANTs points use LPS and the opposite direction from image resampling.
        points=np.concatenate(arrays);lps=points*np.array([-1,-1,1])
        csv=folder/'native_points.csv';pd.DataFrame(lps,columns=['x','y','z']).assign(t=0).to_csv(csv,index=False)
        transformed=folder/'MNI_points.csv'
        inverse_image=source(anat/f'{subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')
        forward_image=source(anat/f'{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')
        run(['/usr/local/ANTs/bin/antsApplyTransformsToPoints','-d','3','-p','1','-i',csv,'-o',transformed,'-t',inverse_image])
        mni_lps=pd.read_csv(transformed)[['x','y','z']].to_numpy()
        assert np.isfinite(mni_lps).all()
        check_indices=np.linspace(0,len(points)-1,1000,dtype=int)
        check=folder/'roundtrip_input.csv';back=folder/'roundtrip_output.csv'
        pd.DataFrame(mni_lps[check_indices],columns=['x','y','z']).assign(t=0).to_csv(check,index=False)
        run(['/usr/local/ANTs/bin/antsApplyTransformsToPoints','-d','3','-p','1','-i',check,'-o',back,'-t',forward_image])
        errors=np.linalg.norm(pd.read_csv(back)[['x','y','z']].to_numpy()-lps[check_indices],axis=1)
        assert np.quantile(errors,.99)<.5,(subject,np.quantile(errors,.99))
        coords=mni_lps*np.array([-1,-1,1]);offset=0
        for image,path,array in zip(images,paths,arrays):
            image.darrays[0].data=coords[offset:offset+len(array)].astype(np.float32);offset+=len(array)
            image.darrays[0].coordsys=nib.gifti.GiftiCoordSystem(4,4,np.eye(4))
            image.meta['ProjectionSpace']='MNI152NLin2009cAsym'
            nib.save(image,path)
        print(subject,'complete; roundtrip p99',round(float(np.quantile(errors,.99)),4),'mm',flush=True)
        return {'subject':subject,'roundtrip_median_mm':float(np.median(errors)),
                'roundtrip_p99_mm':float(np.quantile(errors,.99)), 'roundtrip_max_mm':float(errors.max())}
    qc=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(participant,s) for s in subjects]): qc.append(future.result())
    pd.DataFrame(qc).sort_values('subject').to_csv(out/'surface_transform_qc.csv',index=False)
    for hemi in ['L','R']:
        for kind in ['white','midthickness','pial']:
            surfaces=[out/'surfaces'/s/f'{hemi}.{kind}.MNI152NLin2009cAsym.32k.surf.gii' for s in subjects]
            command=['wb_command','-surface-average',out/f'cohort.{hemi}.{kind}.MNI152NLin2009cAsym.32k.surf.gii']
            for surface in surfaces: command+=['-surf',surface]
            run(command)
    contrasts=['Str1_minus_Ran_r12','Switch_r34_vs_r12_did','Switch_r56_vs_r12_did','Switch_r78_vs_r12_did']
    labels=['Initial','Early','Middle','Late'];counts=[];metrics={h:[] for h in ['L','R']}
    for contrast,label,expected in zip(contrasts,labels,[7356,7,0,0]):
        group=root/'results/glm_volume/whole_brain/group'/contrast
        signed=source(group/'signed_t_tfce_fwe_bonferroni4.nii.gz')
        logp=source(group/'logp_max_tfce.nii.gz');tmap=source(group/'one_sample_t_map.nii.gz')
        image=nib.load(signed);values=np.asarray(image.dataobj)
        np.testing.assert_allclose(values,np.where(np.asarray(nib.load(logp).dataobj)>=-np.log10(.0125),np.asarray(nib.load(tmap).dataobj),0),atol=0,rtol=0)
        assert np.count_nonzero(values)==expected
        roi=out/f'{label}.significant_volume_roi.nii.gz'
        nib.save(nib.Nifti1Image((values!=0).astype(np.uint8),image.affine),roi)
        row={'contrast':contrast,'label':label,'corrected_volume_voxels':expected}
        for hemi in ['L','R']:
            metric=out/f'{label}.{hemi}.corrected_t.func.gii'
            if expected:
                run(['wb_command','-volume-to-surface-mapping',signed,out/f'cohort.{hemi}.midthickness.MNI152NLin2009cAsym.32k.surf.gii',metric,
                     '-ribbon-constrained',out/f'cohort.{hemi}.white.MNI152NLin2009cAsym.32k.surf.gii',out/f'cohort.{hemi}.pial.MNI152NLin2009cAsym.32k.surf.gii',
                     '-volume-roi',roi,'-voxel-subdiv','5'])
            else:
                data=nib.gifti.GiftiDataArray(np.zeros(32492,dtype=np.float32),intent='NIFTI_INTENT_SHAPE')
                data.meta['AnatomicalStructurePrimary']='CortexLeft' if hemi=='L' else 'CortexRight'
                nib.save(nib.gifti.GiftiImage(darrays=[data]),metric)
            mapped=nib.load(metric).darrays[0].data
            run(['wb_command','-set-structure',metric,'CORTEX_LEFT' if hemi=='L' else 'CORTEX_RIGHT'])
            assert np.isfinite(mapped).all()
            if expected:
                selected=values[values!=0]
                assert ((mapped[mapped!=0]>=selected.min()-1e-4)&(mapped[mapped!=0]<=selected.max()+1e-4)).all()
            row[f'{hemi}_nonzero_vertices']=int(np.count_nonzero(mapped))
            metrics[hemi].append(metric)
        counts.append(row)
    for hemi in ['L','R']:
        merged=out/f'wholebrain.{hemi}.corrected_t.func.gii'
        command=['wb_command','-metric-merge',merged]
        for metric in metrics[hemi]:command+=['-metric',metric]
        run(command)
        for i,label in enumerate(labels,1):run(['wb_command','-set-map-names',merged,'-map',str(i),label])
        run(['wb_command','-metric-palette',merged,'MODE_USER_SCALE','-pos-user','0','9','-neg-user','0','-9',
             '-palette-name','cool-warm','-disp-pos','true','-disp-neg','true','-disp-zero','false'])
    pd.DataFrame(counts).to_csv(out/'projection_summary.csv',index=False)
    (out/'commands.json').write_text(json.dumps(commands,indent=2))
    record={'status':'passed','software':run(['wb_command','-version']).strip(),'n_subjects':20,
            'volume_space':'MNI152NLin2009cAsym','surface_mesh':'fsLR32k',
            'coordinate_method':'Resample native T1w white/mid/pial via saved MSMSulc spheres; transform vertices with inverse image H5 (LPS); average 20 normalized surfaces.',
            'mapping':'Workbench ribbon-constrained, volume ROI limited to corrected voxels, subdivision 5; no smoothing or dilation.',
            'inference':'Original uncensored volume TFCE/FWE p<=.0125 across four contrasts; projection is descriptive, not surface inference.',
            'counts':counts,'input_sha256':manifest,'script_sha256':digest(Path(__file__)),
            'output_sha256':{str(p):digest(p) for p in out.glob('*.gii')}}
    (out/'projection_metadata.json').write_text(json.dumps(record,indent=2))
    print('Validated projection:',counts,flush=True)


if __name__=='__main__':main()
