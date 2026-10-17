#!/usr/bin/env python3
"""Render and tabulate the reported censored GLM from existing, verified ROI inputs.

No fitting or new inferential model is performed. Saved ROI statistics are
independently checked against the 80 existing participant contrast maps.
"""
from pathlib import Path
import argparse, importlib.util, json
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from _layout import MANUSCRIPT_DIR, PACKAGE_ROOT, ROI_MASK_DIR, recorded_path

SOURCE = MANUSCRIPT_DIR / 'data/sensitivity'
MODEL = 'censor_hp128'
STAGES = ('initial', 'early', 'middle', 'late')
COUNTS = (31, 15, 5, 0)
helper_path = Path(__file__).with_name('92_make_brain_figures.py')
spec = importlib.util.spec_from_file_location('srtt_axial_display', helper_path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def label(roi):
    import re
    side = 'Left' if roi.startswith('Left') else 'Right'
    name = roi[len(side):]
    replacements = {'SMA': 'BA6', 'DLPFC9': 'BA9', 'DLPFC46': 'BA46',
                    'PrimaryMotor': 'primary motor', 'Cerebellum': 'cerebellar cortex',
                    'ThalamusProper': 'thalamus', 'AccumbensArea': 'accumbens'}
    return side + ' ' + replacements.get(name, re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower())


def bh(p):
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p)+1)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def validate_inputs():
    table = pd.read_csv(SOURCE / 'roi_results.csv')
    table = table[table.model == MODEL].copy()
    assert len(table) == 192 and not table.duplicated(['contrast', 'roi']).any()
    assert set(table.n) == {20} and set(table.df) == {19}
    manifest = pd.read_csv(SOURCE / 'subject_map_manifest.csv')
    manifest = manifest[manifest.model == MODEL]
    assert len(manifest) == 80 and manifest.subject.nunique() == 20
    records = [h.file_record(SOURCE / f) for f in ('roi_results.csv', 'subject_map_manifest.csv')]
    rois = sorted(table.roi.unique())
    assert len(rois) == 48
    reference = nib.load(recorded_path(manifest.iloc[0].path))
    mask_vectors = []
    for roi in rois:
        mask = nib.load(ROI_MASK_DIR / (roi+'.nii.gz'))
        assert h.same_grid(mask, reference), roi
        values = np.asarray(mask.dataobj) > 0
        assert values.any(), roi
        mask_vectors.append(values)
    selected = {}
    for index, contrast in enumerate(h.CONTRASTS):
        group = table[table.contrast == contrast].set_index('roi').loc[rois]
        rows = manifest[manifest.contrast == contrast].sort_values('subject')
        assert len(rows) == 20
        samples = []
        for row in rows.itertuples():
            path = recorded_path(row.path)
            assert h.sha256_file(path) == row.sha256, path
            image = nib.load(path)
            assert h.same_grid(image, reference)
            data = h.finite_3d(image, str(path))
            samples.append([float(np.mean(data[mask])) for mask in mask_vectors])
            records.append(h.file_record(path))
        samples = np.asarray(samples)
        tests = stats.ttest_1samp(samples, 0, axis=0)
        q = bh(tests.pvalue)
        assert np.allclose(group['mean'], samples.mean(axis=0), rtol=1e-8, atol=1e-10)
        assert np.allclose(group.t, tests.statistic, rtol=1e-8, atol=1e-10)
        assert np.allclose(group.p_two_sided, tests.pvalue, rtol=1e-8, atol=1e-12)
        assert np.allclose(group.q_bh_within_contrast, q, rtol=1e-8, atol=1e-12)
        assert int(np.count_nonzero(q <= .05)) == COUNTS[index]
        sig = group[group.q_bh_within_contrast <= .05].reset_index().sort_values(['t','roi'])
        selected[contrast] = [{'roi': r.roi, 't': float(r.t),
                              'q_bh_within_contrast': float(r.q_bh_within_contrast),
                              'cortical': not any(token in r.roi for token in h.NONCORTICAL_ROI_TOKENS)}
                             for r in sig.itertuples()]
    return table, selected, reference, records


def write_tables(table):
    summary = [r'\begin{table}[t]', r'\centering', r'\begin{threeparttable}',
               r'\caption{ROI results from the reported censored high-pass GLM}',
               r'\label{tab:reported}', r'\small', r'\setlength{\tabcolsep}{4pt}',
               r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}rrrr>{\raggedright\arraybackslash}p{0.28\linewidth}@{}}',
               r'\toprule', r'Contrast & $N_{q\leq.05}$ & Positive & Negative & Lowest $q$ & Largest $|t|$ effect \\', r'\midrule']
    names = ('Initial structure','Early sequence change','Middle sequence change','Late sequence change')
    for stage, name in zip(STAGES, names):
        g = table[table.stage == stage]
        sig = g[g.q_bh_within_contrast <= .05]
        peak = g.loc[g.t.abs().idxmax()]
        lowest_q = '$<.001$' if g.q_bh_within_contrast.min() < .001 else f'{g.q_bh_within_contrast.min():.3f}'
        summary.append(f'{name} & {len(sig)} & {(sig.t>0).sum()} & {(sig.t<0).sum()} & {lowest_q} & {label(peak.roi)}, $t(19)={peak.t:.2f}$ '+r'\\')
    summary += [r'\bottomrule',r'\end{tabular}',r'\begin{tablenotes}[flushleft]',r'\footnotesize',
                r'\item Note. $N=20$; BH FDR covers 48 ROIs separately within each contrast. Lowest $q$ and largest $|t|$ are drawn from all 48 ROIs, including the late contrast with no significant ROI. The model uses volume censoring and a 128-s high-pass. BA6 denotes the complete atlas parcel, including but not restricted to SMA.',
                r'\end{tablenotes}',r'\end{threeparttable}',r'\end{table}']
    (MANUSCRIPT_DIR/'tables/censored_nilearn.tex').write_text('\n'.join(summary)+'\n')
    full = [r'\small',r'\begin{longtable}{llrrr}',
            r'\caption{FDR-significant ROI effects using full-parcel means from the censored high-pass GLM. $N=20$; $df=19$; BH FDR covers 48 ROIs within each contrast. Bilateral entries report left (L) and right (R) values on separate lines.}\label{tab:censoredfull}\\',
            r'\toprule',r'ROI & Laterality & Mean contrast & $t(19)$ & $q$ \\',r'\midrule',r'\endfirsthead',
            r'\toprule',r'ROI & Laterality & Mean contrast & $t(19)$ & $q$ \\',r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot']
    for stage in STAGES:
        sig = table[(table.stage==stage)&(table.q_bh_within_contrast<=.05)]
        if sig.empty:
            continue
        grouped = {}
        for row in sig.itertuples():
            side, name = label(row.roi).split(' ', 1)
            grouped.setdefault(name, {})[side] = row
        # Bilateral results first, followed by left-only and right-only results.
        # Within each category retain descending peak absolute t ordering.
        def order(item):
            name, sides = item
            category = 0 if len(sides) == 2 else (1 if 'Left' in sides else 2)
            return category, -max(abs(row.t) for row in sides.values()), name
        entries = sorted(grouped.items(), key=order)
        if stage != STAGES[0]:
            full.append(r'\addlinespace')
        full.append(r'\multicolumn{5}{l}{\textbf{' + stage.capitalize() + r'}} \\*')
        for index, (name, sides) in enumerate(entries):
            def statistic(row, field):
                if field == 'q':
                    return '$<.001$' if row.q_bh_within_contrast < .001 else f'{row.q_bh_within_contrast:.3f}'
                return f'{row.mean:.3f}' if field == 'mean' else f'{row.t:.2f}'
            if len(sides) == 2:
                laterality = 'Bilateral'
                cells = [r'\shortstack[r]{L: ' + statistic(sides['Left'], field)
                         + r'\\R: ' + statistic(sides['Right'], field) + '}'
                         for field in ('mean', 't', 'q')]
            else:
                laterality, row = next(iter(sides.items()))
                cells = [statistic(row, field) for field in ('mean', 't', 'q')]
            # Keep a stage together while keeping each bilateral estimate pair intact.
            row_end = r'\\*' if index < len(entries) - 1 else r'\\'
            full.append(f'{name} & {laterality} & ' + ' & '.join(cells) + ' ' + row_end)
    full += [r'\end{longtable}',r'\normalsize']
    (MANUSCRIPT_DIR/'tables/censored_significant_rois.tex').write_text('\n'.join(full)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dpi', type=int, default=360)
    args=parser.parse_args()
    assert args.dpi >= 300
    table, selected, reference, records=validate_inputs()
    print('PASS: censored ROI tests independently reproduced from 80 saved subject maps (31/15/5/0).',flush=True)
    h.ROI_FIGURE_CONTRASTS=h.CONTRASTS[:3]
    volumes, mask_records=h.build_roi_display_volumes(selected,reference,ROI_MASK_DIR)
    template=PACKAGE_ROOT/'data/templates/tpl-MNI152NLin2009cAsym'
    anatomy_path=template/'tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
    mask_path=template/'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'
    anatomy=h.load_axial_anatomy(anatomy_path,mask_path)
    vmax=h.rounded_symmetric_limit([volumes[c]['all_values'] for c in h.ROI_FIGURE_CONTRASTS])
    fig=h.plt.figure(figsize=(14,11.4),facecolor='black')
    grid=fig.add_gridspec(9,6,left=.035,right=.98,top=.89,bottom=.145,
                          height_ratios=(.22,1,1)*3,hspace=.32,wspace=.02)
    fig.suptitle('Censored high-pass GLM: horizontal slices',color='white',fontsize=22,fontweight='bold',y=.985)
    fig.text(.5,.943,'ROI means surviving BH FDR; neurological orientation (L on the left)',color='white',fontsize=14,ha='center')
    for i,c in enumerate(h.ROI_FIGURE_CONTRASTS):
        sig=table[(table.contrast==c)&(table.q_bh_within_contrast<=.05)]
        text=f'{chr(65+i)}  {STAGES[i].capitalize()} contrast  |  {len(sig)} ROIs: +{(sig.t>0).sum()} / -{(sig.t<0).sum()}'
        h.strip_label(fig,grid,3*i,text)
        for offset in range(2):
            h.axial_strip(fig,grid,3*i+1+offset,volumes[c]['all_image'],anatomy,h.AXIAL_CUTS_MM[offset*6:(offset+1)*6],vmax)
    fig.text(.5,.103,'Colored masks represent ROI-average tests, not voxelwise significance.\nLate contrast: no ROIs survived FDR. Overlapping masks show the larger absolute t.',color='white',fontsize=13,ha='center',va='center')
    h.axial_colorbar(fig,vmax,'ROI one-sample t(19): negative (blue), positive (red)',.047)
    output=h.save_png(fig,MANUSCRIPT_DIR/'figures/censored_roi_axial.png',args.dpi,'Censored high-pass ROI results')
    write_tables(table)
    metadata={'model':MODEL,'analysis_role':'visualization of ROI-average inference; no voxelwise inference',
              'script':h.file_record(Path(__file__)), 'display_helpers':h.file_record(helper_path),
              'n_subjects':20,'df':19,'roi_counts':list(COUNTS),'output':output,'dpi':args.dpi,
              'expected_pixel_dimensions':[14*args.dpi,round(11.4*args.dpi)],
              'plane':'axial','orientation':'neurological','axial_cut_coords_mm':list(h.AXIAL_CUTS_MM),
              'statistical_interpolation':'nearest','display_smoothing':None,'spatial_dilation':False,
              'anatomy':h.file_record(anatomy_path),'anatomical_brain_mask':h.file_record(mask_path),
              'contrasts':{c:{'n_significant_rois':len(selected[c]),'rois':selected[c]} for c in h.CONTRASTS},
              'inputs':records+mask_records,
              'tables':{name:h.file_record(MANUSCRIPT_DIR/'tables'/name) for name in ('censored_nilearn.tex','censored_significant_rois.tex')}}
    (MANUSCRIPT_DIR/'figures/censored_roi_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    print('Rendered censored axial figure and both tables.',flush=True)

if __name__=='__main__':main()
