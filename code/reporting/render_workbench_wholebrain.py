#!/usr/bin/env python3
"""Render Workbench scenes and assemble the cortical projection figure."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import nibabel as nib
import numpy as np
from PIL import Image


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/glm_volume/whole_brain/workbench_20261013'
FIG=ROOT/'manuscript/figures'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    metadata=json.loads((OUT/'projection_metadata.json').read_text())
    assert metadata['status']=='passed'
    for path,sha in metadata['input_sha256'].items():assert digest(path)==sha,path
    for path,sha in metadata['output_sha256'].items():assert digest(path)==sha,path
    commands=[]
    def run(command):
        command=list(map(str,command));env=os.environ.copy()
        env['LIBGL_ALWAYS_SOFTWARE']='1';env['QT_QPA_PLATFORM']='xcb'
        result=subprocess.run(command,env=env,text=True,capture_output=True)
        commands.append({'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stderr or result.stdout)
    cifti=OUT/'wholebrain.corrected_t.dscalar.nii'
    run(['wb_command','-cifti-create-dense-scalar',cifti,'-left-metric',OUT/'wholebrain.L.corrected_t.func.gii','-right-metric',OUT/'wholebrain.R.corrected_t.func.gii'])
    run(['wb_command','-cifti-palette',cifti,'MODE_USER_SCALE',cifti,'-pos-user','0','9','-neg-user','0','-9','-palette-name','cool-warm','-disp-pos','true','-disp-neg','true','-disp-zero','false'])
    template=OUT/'wholebrain_montage_template.scene'
    if not template.exists():template.write_bytes((OUT/'wholebrain.scene').read_bytes())
    tree=ET.parse(template);root=tree.getroot();base_scene=root.find('Scene')
    directory=root.find('SceneInfoDirectory');base_info=directory.find('SceneInfo')
    assert base_scene is not None and base_info is not None
    root.remove(base_scene);directory.remove(base_info)
    labels=['Initial','Early','Middle','Late']
    for index,label in enumerate(labels):
        scene=copy.deepcopy(base_scene);scene.set('Index',str(index));scene.find('Name').text=label
        scene.find('Description').text='Corrected volume TFCE/FWE p <= .0125 projected onto the cohort cortical ribbon; descriptive surface display.'
        info=copy.deepcopy(base_info);info.set('Index',str(index));info.find('Name').text=label
        info.find('Description').text=scene.find('Description').text
        for child in list(info):
            if child.tag=='Image':info.remove(child)
        for overlay in scene.iter('Object'):
            if overlay.get('Class')=='Overlay':
                objects={child.get('Name'):child for child in overlay if child.tag=='Object'}
                if objects.get('selectedMapFile') is not None and objects['selectedMapFile'].text==cifti.name:
                    objects['selectedMapIndex'].text=str(index)
                    objects['selectedMapName'].text=label
            if overlay.get('Name')=='m_colorBar':
                for child in overlay:
                    if child.get('Name')=='m_displayedFlag':child.text='false'
        root.append(scene);directory.append(info)
    scenes=OUT/'wholebrain.scene';ET.indent(tree,space='  ');tree.write(scenes,encoding='utf-8',xml_declaration=True)
    for index,label in enumerate(labels,1):
        run(['xvfb-run','-a','wb_command','-scene-capture-image',scenes,str(index),OUT/f'{label}.workbench.png','-size-width-height','2400','480'])
    images=[np.asarray(Image.open(OUT/f'{label}.workbench.png').convert('RGB')) for label in labels]
    assert np.array_equal(images[2],images[3]),'Null maps must render identically.'
    signed_pixels=images[0].astype(int)
    assert np.any((signed_pixels[:,:,2]>signed_pixels[:,:,0]+15)&(signed_pixels[:,:,2]>signed_pixels[:,:,1]+5)), 'Initial negative effects missing.'
    assert np.any((images[1][:,:,0].astype(int)>images[1][:,:,2].astype(int)+15)), 'Early positive effects missing.'
    palette_path=OUT/'workbench_cool_warm.json'
    palette=json.loads(palette_path.read_text())
    rgb=np.asarray(palette['rgb_anchors'])/255
    assert rgb.shape==(33,3)
    cmap=LinearSegmentedColormap.from_list('workbench_cool_warm',list(zip(np.linspace(0,1,33),rgb)),N=1025)
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'white','font.size':11})
    fig=plt.figure(figsize=(13,12.6),facecolor='black')
    fig.text(.5,.969,'Whole-brain results: cortical surface views',ha='center',fontsize=19,weight='bold')
    fig.text(.5,.936,'Uncensored reference GLM • volume TFCE/FWE p ≤ .0125',ha='center',fontsize=11,color='#CFD4DD')
    x0=.125;width=.855
    headers=['Left lateral','Left medial','Right lateral','Right medial']
    for i,header in enumerate(headers):fig.text(x0+width*(i+.5)/4,.900,header,ha='center',fontsize=11)
    bottom=[.700,.510,.320,.130]
    for i,(label,image,row) in enumerate(zip(labels,images,bottom)):
        ax=fig.add_axes([x0,row,width,width*13/5/12.6]);ax.imshow(image);ax.set_axis_off()
        fig.text(.018,row+.095,label,fontsize=13,weight='bold',va='center')
        count=metadata['counts'][i]['corrected_volume_voxels']
        fig.text(.018,row+.060,f'{count:,} voxels',fontsize=10,color='#CAD0DA',va='center')
        if count==0:fig.text(.54,row-.006,'No suprathreshold voxels',ha='center',fontsize=10,color='#D8DCE3')
    ax=fig.add_axes([.32,.080,.36,.013])
    cbar=fig.colorbar(ScalarMappable(norm=Normalize(-9,9),cmap=cmap),cax=ax,orientation='horizontal',ticks=[-9,-6,-3,0,3,6,9])
    cbar.ax.tick_params(colors='white',labelsize=10);cbar.outline.set_edgecolor('#C2C6D0')
    fig.text(.5,.049,'Projected t statistic (df = 19)',ha='center',fontsize=10)
    fig.text(.5,.022,'Counts refer to corrected volume voxels. Three of seven early voxels intersect the display ribbon; see axial views for all effects.',ha='center',fontsize=9,color='#CAD0DA')
    for extension in ['png','pdf']:fig.savefig(FIG/f'whole_brain_workbench_surface.{extension}',dpi=320,facecolor='black')
    plt.close(fig)
    # Independently check that only corrected early voxels contributed.
    weights=OUT/'Early.R.ribbon_weights.txt';sampled=set()
    run(['wb_command','-volume-to-surface-mapping',ROOT/'results/glm_volume/whole_brain/group/Switch_r34_vs_r12_did/signed_t_tfce_fwe_bonferroni4.nii.gz',
         OUT/'cohort.R.midthickness.MNI152NLin2009cAsym.32k.surf.gii',OUT/'Early.R.weight_check.func.gii',
         '-ribbon-constrained',OUT/'cohort.R.white.MNI152NLin2009cAsym.32k.surf.gii',OUT/'cohort.R.pial.MNI152NLin2009cAsym.32k.surf.gii',
         '-volume-roi',OUT/'Early.significant_volume_roi.nii.gz','-voxel-subdiv','5','-output-weights-text',weights])
    if weights.exists():
        for line in weights.read_text().splitlines():
            fields=[float(x) for x in line.split(',')]
            for i in range(int(fields[1])):sampled.add(tuple(map(int,fields[2+4*i:5+4*i])))
        source=ROOT/'results/glm_volume/whole_brain/group/Switch_r34_vs_r12_did/signed_t_tfce_fwe_bonferroni4.nii.gz'
        significant=set(map(tuple,np.argwhere(np.asarray(nib.load(source).dataobj)!=0)))
        assert sampled<=significant and len(sampled)==3
    (OUT/'render_commands.json').write_text(json.dumps(commands,indent=2))
    record={'status':'passed','projection_metadata_sha256':digest(OUT/'projection_metadata.json'),
            'script_sha256':digest(Path(__file__)),'template_scene_sha256':digest(template),
            'palette_definition_sha256':digest(palette_path),
            'scene_sha256':digest(scenes),'cifti_sha256':digest(cifti),
            'rendering':'Connectome Workbench 2.1.0, native scene-capture-image, 2400 x 480 per contrast',
            'view_order':headers,'color_scale_t':[-9,9],'background':'black',
            'early_corrected_volume_voxels':7,'early_volume_voxels_intersecting_average_cortical_ribbon':len(sampled),
            'null_maps_identical':True,'output_sha256':{str(FIG/f'whole_brain_workbench_surface.{ext}'):digest(FIG/f'whole_brain_workbench_surface.{ext}') for ext in ['png','pdf']}}
    (OUT/'render_validation.json').write_text(json.dumps(record,indent=2))
    print('Rendered and validated Workbench scenes and surface figure.',flush=True)


if __name__=='__main__':main()
