#!/usr/bin/env python3
"""Audit atlas -> native AFNI -> resampled AFNI -> manuscript NIfTI masks.

Reads source data only; writes audit tables/JSON to --out-dir. Optional
--reconstruct-dir writes new reconstructed masks, never the analysis inputs.
Atlas identities are established by exact voxel/grid comparison, not filenames.
Demographic agreement across copied tables is not primary-record verification.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nibabel.processing import resample_from_to
from scipy.ndimage import label

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import MANUSCRIPT_DIR as MANUSCRIPT, PROJECT_ROOT as PROJECT, SUBJECT_LIST, ROI_LIST, BIDS_DIR, ROI_MASK_DIR
ORIGINAL = Path('/home/fmri2404/chiaoen/SrttNewV2')
ATLAS_NAMES = ('FS.afni.MNI2009c_asym', 'Brodmann_pijn_afni')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def same_grid(a, b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-5, rtol=0)


def atlas_labels(image):
    text = image.header.extensions[0].get_content().decode()
    text = html.unescape(re.sub(r'"\s*\n\s*"', '', text))
    labels = {int(v): name for name, v in re.findall(r'STRUCT="([^"]+)"\s+VAL="(\d+)"', text)}
    if not labels:
        raise ValueError('Missing embedded AFNI atlas label table')
    return labels


def demo_summary(frame):
    return {'n': len(frame), 'age_mean': float(frame.age.mean()),
            'age_sd_sample_ddof1': float(frame.age.std(ddof=1)),
            'age_min': int(frame.age.min()), 'age_max': int(frame.age.max()),
            'sex_counts': {k: int(v) for k, v in frame.sex.value_counts().items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, default=ORIGINAL)
    parser.add_argument('--atlas-dir', type=Path, default=Path('/usr/local/abin'))
    parser.add_argument('--out-dir', type=Path, default=MANUSCRIPT / 'data/provenance_audit')
    parser.add_argument('--reconstruct-dir', type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    native_dir = args.original_root / 'Nifti/code/FirstLevelAnalysis_ROI/ROI_masks'
    target_dir = ROI_MASK_DIR
    if args.reconstruct_dir:
        destination = args.reconstruct_dir.resolve()
        if destination == target_dir.resolve() or destination == native_dir.resolve():
            raise ValueError('Reconstruction must not overwrite original inputs')
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError('Reconstruction directory must be empty')
    rois_path = ROI_LIST
    rois = rois_path.read_text().split()
    assert len(rois) == len(set(rois)) == 48
    files = {}

    def record(path, role):
        path = Path(path)
        files[str(path)] = {'path': str(path), 'role': role,
                            'bytes': path.stat().st_size, 'sha256': sha(path)}
        return files[str(path)]['sha256']

    record(Path(__file__), 'audit_script')
    record(rois_path, 'roi_list')
    atlases = {}
    labels_rows = []
    for name in ATLAS_NAMES:
        path = args.atlas_dir / (name + '.nii.gz')
        image = nib.load(path)
        values = np.asarray(image.dataobj).squeeze()
        labels = atlas_labels(image)
        record(path, 'atlas')
        atlases[name] = (image, values, labels)
        labels_rows.extend({'atlas': name, 'value': v, 'label': text} for v, text in labels.items())
    record(args.atlas_dir / 'AFNI_atlas_spaces.niml', 'atlas_registration_and_description')
    pd.DataFrame(labels_rows).to_csv(out / 'atlas_labels.csv', index=False)
    prior_qc_path = MANUSCRIPT / 'data/whole_brain/roi_mask_qc.csv'
    record(prior_qc_path, 'original_group_input_manifest')
    prior_qc = pd.read_csv(prior_qc_path).set_index('roi')
    common_path = MANUSCRIPT / 'data/whole_brain/common_mask.nii.gz'
    if not common_path.exists():
        # Obtain the exact name from the existing saved product.
        candidates = list((MANUSCRIPT / 'data/whole_brain').glob('*mask*.nii.gz'))
        assert len(candidates) == 1, candidates
        common_path = candidates[0]
    record(common_path, 'whole_brain_common_mask')
    common_image = nib.load(common_path)
    common = np.asarray(common_image.dataobj) > 0
    rows, histories, target_masks, atlas_comparisons = [], {}, {}, []
    for roi in rois:
        native_path = native_dir / (roi + '+tlrc.HEAD')
        rs_path = native_dir / (roi + '_rs+tlrc.HEAD')
        target_path = target_dir / (roi + '.nii.gz')
        native, rs, target = (nib.load(p) for p in (native_path, rs_path, target_path))
        native_values, rs_values, target_values = (np.asarray(im.dataobj).squeeze() for im in (native, rs, target))
        assert all(np.isfinite(v).all() and set(np.unique(v)) <= {0, 1}
                   for v in (native_values, rs_values, target_values)), roi
        native_mask, target_mask = native_values > 0, target_values > 0
        assert native_mask.any() and target_mask.any(), roi
        matches = []
        for atlas, (image, values, labels) in atlases.items():
            assert same_grid(native, image), (roi, atlas, 'native grid differs')
            unique, counts = np.unique(values[native_mask], return_counts=True)
            # Record every intersected label, including background; no winner-only inference.
            for v, count in zip(unique, counts):
                atlas_n = int(np.count_nonzero(values == v))
                differences = int(native_mask.sum()) + atlas_n - 2 * int(count)
                atlas_comparisons.append({'roi': roi, 'atlas': atlas, 'atlas_value': int(v),
                    'atlas_label': labels.get(int(v), 'background/unlisted'),
                    'intersection_voxels': int(count), 'atlas_label_voxels': atlas_n,
                    'native_roi_voxels': int(native_mask.sum()),
                    'symmetric_difference_voxels': differences})
                if v != 0 and differences == 0:
                    matches.append((atlas, int(v), labels[int(v)]))
        assert len(matches) == 1, (roi, matches)
        atlas, value, atlas_label = matches[0]
        atlas_image, atlas_values, _ = atlases[atlas]
        rebuilt = resample_from_to(nib.Nifti1Image((atlas_values == value).astype('uint8'), atlas_image.affine),
                                   (target.shape[:3], target.affine), order=0)
        rebuilt_values = np.asarray(rebuilt.dataobj)
        difference = int(np.count_nonzero(rebuilt_values != target_values))
        assert difference == 0, (roi, 'atlas reconstruction differs', difference)
        assert same_grid(rs, target) and np.array_equal(rs_values, target_values), (roi, 'AFNI/NIfTI mismatch')
        assert same_grid(target, common_image), roi
        target_hash = record(target_path, 'analysis_mask')
        assert target_hash == prior_qc.loc[roi, 'sha256'], (roi, 'group manifest checksum differs')
        assert int(target_mask.sum()) == int(prior_qc.loc[roi, 'mask_voxels_gt_zero'])
        for head in (native_path, rs_path):
            record(head, 'original_afni_header')
            brick = head.with_suffix('.BRIK')
            if not brick.exists():
                brick = head.with_suffix('.BRIK.gz')
            record(brick, 'original_afni_voxels')
        histories[roi] = {stage: {key: im.header.info.get(key) for key in
                         ('HISTORY_NOTE', 'LABEL_1', 'IDCODE_STRING', 'IDCODE_DATE', 'TEMPLATE_SPACE')}
                         for stage, im in [('native', native), ('resampled', rs)]}
        xyz = nib.affines.apply_affine(target.affine, np.argwhere(target_mask))
        wrong_side = int(np.count_nonzero(xyz[:, 0] > 0 if roi.startswith('Left') else xyz[:, 0] < 0))
        assert wrong_side == 0, (roi, 'opposite hemisphere voxels', wrong_side)
        count = int(target_mask.sum())
        note = 'Atlas label confirmed'
        if roi.endswith('SMA'):
            note = 'Use BA6; complete atlas parcel, not an SMA-specific subdivision'
        elif roi.endswith('Cerebellum'):
            note = 'Use cerebellar cortex; does not include cerebellar white matter'
        elif 'DLPFC' in roi:
            note = 'Use BA' + roi.split('DLPFC')[1] + '; complete atlas parcel, not a separately restricted DLPFC mask'
        rows.append({'roi': roi, 'atlas': atlas, 'atlas_value': value, 'atlas_label': atlas_label,
                     'native_voxels_1mm': int(native_mask.sum()), 'native_dice': 1.0,
                     'native_symmetric_difference_voxels': 0,
                     'analysis_voxels_3mm': count, 'analysis_volume_mm3': count * 27,
                     'atlas_to_analysis_difference_voxels': difference,
                     'afni_rs_equals_nifti': True, 'analysis_manifest_sha256_matches': True,
                     'analysis_sha256': target_hash, 'opposite_hemisphere_voxels': wrong_side,
                     'midline_voxels': int(np.count_nonzero(xyz[:, 0] == 0)),
                     'centroid_x': float(xyz[:, 0].mean()), 'centroid_y': float(xyz[:, 1].mean()),
                     'centroid_z': float(xyz[:, 2].mean()),
                     'components_26': int(label(target_mask, structure=np.ones((3, 3, 3)))[1]),
                     'voxels_in_common_whole_brain_mask': int(np.count_nonzero(target_mask & common)),
                     'label_note': note})
        target_masks[roi] = target_mask
        if args.reconstruct_dir:
            nib.save(rebuilt, destination / (roi + '.nii.gz'))
    pd.DataFrame(rows).to_csv(out / 'roi_provenance.csv', index=False)
    pd.DataFrame(atlas_comparisons).to_csv(out / 'native_atlas_overlaps.csv', index=False)
    save_json(out / 'afni_history.json', histories)
    pair_rows = []
    for left, right in combinations(rois, 2):
        a, b = target_masks[left], target_masks[right]
        n = int(np.count_nonzero(a & b))
        pair_rows.append({'roi_a': left, 'roi_b': right, 'intersection_voxels': n,
                         'dice': float(2*n / (a.sum()+b.sum())), 'identical': bool(np.array_equal(a,b))})
    pd.DataFrame(pair_rows).to_csv(out / 'roi_pairwise_overlap.csv', index=False)
    assert not any(row['identical'] for row in pair_rows), 'Duplicate analysis masks'

    subjects_path = SUBJECT_LIST
    record(subjects_path, 'retained_subject_list')
    subjects = subjects_path.read_text().split()
    assert len(subjects) == len(set(subjects)) == 20
    canonical_path = BIDS_DIR / 'participants.tsv'
    canonical = pd.read_csv(canonical_path, sep='\t').set_index('participant_id')
    retained = canonical.loc[subjects]
    demo_sources = [canonical_path, args.original_root/'Nifti/participants.tsv',
                    PROJECT/'archive/legacy_bids_root/participants.tsv', PROJECT/'archive/fMRIPrep2/participants.tsv',
                    PROJECT/'archive/fMRIPrep2/participants.tsv.old']
    demographics = []
    for path in demo_sources:
        if not path.exists():
            continue
        record(path, 'demographic_table')
        frame = pd.read_csv(path, sep='\t')
        assert not frame.participant_id.duplicated().any(), path
        indexed = frame.set_index('participant_id')
        covered = [s for s in subjects if s in indexed.index]
        shared = sorted(set(indexed.index) & set(canonical.index))
        disagreements = int((indexed.loc[shared, ['age','sex']] != canonical.loc[shared, ['age','sex']]).any(axis=1).sum())
        demographics.append({'path': str(path), 'all_rows': demo_summary(frame),
                             'retained_coverage': len(covered), 'missing_retained_ids': sorted(set(subjects)-set(covered)),
                             'age_sex_disagreement_rows_vs_package': disagreements,
                             'retained_summary': demo_summary(indexed.loc[covered])})
    metadata = []
    for path in (BIDS_DIR/'participants.json', args.original_root/'Nifti/participants.json', PROJECT/'archive/legacy_bids_root/participants.json', PROJECT/'archive/fMRIPrep2/participants.json'):
        if path.exists():
            record(path, 'demographic_metadata')
            content = json.loads(path.read_text())
            metadata.append({'path': str(path), 'age': content.get('age'), 'sex': content.get('sex')})
    save_json(out/'demographics.json', {'status': 'thesis-reported recruitment; internally consistent retained metadata; primary age/sex records unverified',
              'retained_summary': demo_summary(retained), 'sources': demographics, 'metadata': metadata,
              'thesis_recruited': {'n':25, 'male':12, 'age_mean':21.8, 'age_sd':2.7,
                                    'source':'doc/Chan_2025_MAthesis_IDS-SRTT.md p.74 lines 14-19'},
              'retained_demographics_are_not_reported_in_thesis': True,
              'note':'Matching copies do not constitute independent demographic verification. No participant-level ages/sex are exported by this audit.'})
    for ext in ('md', 'pdf'):
        record(PROJECT / ('docs/Chan_2025_MAthesis_IDS-SRTT.' + ext), 'thesis_source')
    pd.DataFrame(files.values()).to_csv(out / 'input_manifest.csv', index=False)
    save_json(out / 'summary.json', {'audit_version': 1, 'n_rois':48,
             'exact_native_atlas_matches':48, 'exact_atlas_to_analysis_reconstructions':48,
             'exact_resampled_afni_to_nifti_matches':48, 'historical_group_mask_checksums_match':48,
             'atlas_roi_counts':pd.Series([r['atlas'] for r in rows]).value_counts().to_dict(),
             'overlapping_roi_pairs':sum(r['intersection_voxels']>0 for r in pair_rows),
             'duplicate_masks':0, 'hemisphere_mismatch_voxels':0,
             'roi_status':'PASS; display labels require BA6, BA9/46 and cerebellar cortex wording',
             'demographic_status':'Thesis claims checked; retained totals reproduced; primary records unverified',
             'ethics_status':'Thesis reports written informed consent; approving committee/protocol not found',
             'software': {'numpy':np.__version__, 'nibabel':nib.__version__, 'scipy':scipy.__version__, 'pandas':pd.__version__},
             'resampling':'nibabel.processing.resample_from_to, order=0, constant zero outside source',
             'limitations':['Exact spatial identity recovers a reproducible construction; it does not establish the historical GUI selection rationale or preregistration.',
                            'No first-level fitting, ROI statistical recomputation, or mask replacement is performed.',
                            'No ethics approval, protocol coverage or data-sharing permission is inferred from a consent statement.']})
    print('PASS: 48 native atlas identities, 48 exact resampled reconstructions, 48 AFNI/NIfTI matches; demographics reconciled with stated source limitations.')
    print(out)


if __name__ == '__main__':
    main()
