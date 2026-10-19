"""Seal A7 provenance, report and owned artifact inventory."""
from slicetime_common import *
from datetime import datetime
from zoneinfo import ZoneInfo


def seal_sources():
    """Record final assembled sources alongside initial execution snapshots."""
    manifests=sorted(OUT.rglob('input_manifest.json'),key=lambda p:len(p.parts),reverse=True)
    for manifest in manifests:
        q=json.loads(manifest.read_text());original=q['sources'];current={p:digest(p) for p in original}
        for p in (ROOT/'code/review_r1').glob('slicetime_*.py'):current[str(p)]=digest(p)
        if current!=original:q['execution_source_snapshot']=original;q['sources']=current
        owned_original={p:r for p,r in q['inputs'].items() if Path(p).is_relative_to(OUT)}
        for p,r in owned_original.items():
            q['inputs'][p]=dict(sha256=digest(p),bytes=Path(p).stat().st_size)
        if any(q['inputs'][p]!=r for p,r in owned_original.items()):q['execution_owned_input_snapshot']=owned_original
        writej(manifest,q)
        validation=manifest.with_name('validation.json');v=json.loads(validation.read_text())
        v['output_sha256']={str(p.relative_to(manifest.parent)):digest(p) for p in manifest.parent.rglob('*') if p.is_file() and p!=validation}
        writej(validation,v)


def main():
    t=ImagingTask('');v=json.loads(t.record(OUT/'validation/validation.json').read_text());assert v['passed']
    meta=pd.read_csv(t.record(OUT/'metadata/participant_run_slice_timing.csv'))
    controls=pd.read_csv(t.record(OUT/'controls/reproduction_checks.csv'))
    first=json.loads(t.record(OUT/'first_level/validation.json').read_text())
    focus=pd.read_csv(t.record(OUT/'roi/headline_and_accumbens_comparison.csv'))
    balance=pd.read_csv(t.record(OUT/'roi/headline_early_component_balance.csv'))
    counts=pd.read_csv(t.record(OUT/'voxelwise/aligned_vs_reference_counts.csv'))
    stages=pd.read_csv(t.record(OUT/'roi/stage_results.csv'));direct=pd.read_csv(t.record(OUT/'roi/direct_phase_results.csv'))
    n22counts=counts[counts.cohort.eq('N22')].set_index('contrast')
    wholebrain_note=(f"For N22, initial negative corrected voxels change from {int(n22counts.loc['initial','negative_voxels_reference'])} to {int(n22counts.loc['initial','negative_voxels_aligned'])}; early positive voxels change from {int(n22counts.loc['early','positive_voxels_reference'])} to {int(n22counts.loc['early','positive_voxels_aligned'])}; pooled positive voxels change from {int(n22counts.loc['pooled_r36_Str2_minus_Str1','positive_voxels_reference'])} to {int(n22counts.loc['pooled_r36_Str2_minus_Str1','positive_voxels_aligned'])}. These are descriptive thresholded counts.")
    group_counts=stages.assign(stage_BH48=stages.q.le(.05),joint_BH192=stages.q_joint_stages_192.le(.05)).groupby(['cohort','model','stage'],as_index=False)[['stage_BH48','joint_BH192']].sum()
    csv(group_counts,OUT/'stage_discovery_counts.csv')
    direct_counts=direct.assign(direct_BH144=direct.q_three_pairs_144.le(.05)).groupby(['cohort','model','pair'],as_index=False).direct_BH144.sum()
    csv(direct_counts,OUT/'direct_phase_discovery_counts.csv')
    for name in ['metadata','controls','first_level','roi','voxelwise_inputs','voxelwise','validation']:
        t.record(OUT/name/'input_manifest.json');t.record(OUT/name/'validation.json')
    now=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')
    report=f'''# A7 slice-timing-aligned sensitivity — {now}

Status: **DONE**. All 176 aligned first-level fits, full ROI outputs for both cohorts, and ten voxelwise analyses completed and passed the checks below. N22 uncensored **unaligned reference remains primary**; aligned fits are a sensitivity, and N20 remains supplement-only. No model, cohort, ROI, phase or threshold was selected from results.

## Specification and outputs

- `metadata/participant_run_slice_timing.csv`: all 176 StartTime/RepetitionTime values, sidecar paths and hashes. Included StartTime spans {meta.StartTime.min():.3f}–{meta.StartTime.max():.3f} s; TR=2 s; 200 volumes. The plan's 0.950–0.962-s range describes all 194 preprocessed runs, not only these 176 included runs. No timing field was missing.
- `controls/`: sub-001 run 1 and sub-020 run 6 fitted at zero frame shift before any aligned fit; all 240 ROI-condition slots (including NaNs) matched saved reference values exactly. The aligned versions of these controls were then fit before the full run.
- `first_level/`: 176 aligned fits; participant condition values, per-run condition effect maps, complete design matrices with frame-time index, and QC. Frame times are `np.arange(200)*RepetitionTime+StartTime` using each run's JSON. Glover HRF, cosine 1/128-s high-pass, AR(1), 100 bins, 4-mm smoothing, eight reference nuisance columns (six motion + CSF/WM), percent mean scaling and all volumes retained. No censor spikes or repeat regressors are added. Condition maps are native float32 before continuous reference-space resampling. Observed80 extraction uses each participant's eight-run mask intersection, fixed independently of coefficient values.
- `roi/`: reference and aligned, both cohorts, every 48-ROI slot. Stage BH48, joint-stage BH192, direct-phase BH144 (plus labelled within-pair BH48), full phase components and A2 paired changes; headline early BH4 separately for each component. Untestable slots enter BH as p=1. Stage contrasts use two-run sums; ΔS/ΔR use two-run means, M=(ΔS+ΔR)/2 and DiD/2=ΔS−ΔR. One-sample t, df=n−1, pointwise 95% CIs. `slicetime_aligned_vs_reference.csv` compares all stages; `headline_and_accumbens_comparison.csv` includes the four headline early ROIs and both accumbens initial, for both cohorts. Historical LeftSMA/RightSMA keys denote full BA6 parcels.
- `voxelwise_inputs/`: all 110 participant maps, four summed historical stage contrasts and equal-run-mean Str2−Str1 across runs 3–6. N20 is the existing subset of N22.
- `voxelwise/`: ten matched reference reruns. Same unchanged 58,550-voxel N22 mask and validated direct component-tree TFCE backend, 10,000 two-sided sign-flip permutations, seed **20261018**, eight-chunk schedule. Historical stages have FWE .05/4 per cohort/model; pooled within-run has a separate FWE .05 family. TFCE E=.5/H=2, native integration/sign rules and six-neighbor connectivity. Descriptive components use 26 neighbors. Raw null maxima, observed t/TFCE, FWE log-p, signed corrected t, effect/SE/pointwise CI maps and component tables are retained.

## Neutral comparisons

All four N22 early headline ROIs retain stage BH48 and joint BH192 detections under alignment. Their aligned summed estimates are 0.1602 (left BA6), 0.1636 (right BA6), 0.1205 (right pallidum), and 0.1384 (right thalamus), compared with 0.1658, 0.1768, 0.1317 and 0.1526 in the reference. Right accumbens initial stage q changes from .0371 to .0508 (joint q .0932 to .1245); left accumbens remains detected in both families (aligned stage q .0090; joint q .0299). Only right BA6 meets the A2 declining-random pointwise-CI wording rule in the four early ROIs under either specification; its aligned headline BH4 q is .1264. These threshold statements describe this fixed sensitivity and do not change the primary hierarchy.

N22 stage estimates and pointwise CIs on the historical two-run-sum scale; q and joint q use 48/192 slots respectively:

```text
'''+focus[focus.cohort.eq('N22')][['stage','roi','n_aligned','mean_reference','ci_low_reference','ci_high_reference','q_reference','q_joint_stages_192_reference','mean_aligned','ci_low_aligned','ci_high_aligned','q_aligned','q_joint_stages_192_aligned','mean_shift']].to_string(index=False)+'''
```

N22 early M on the phase-mean scale, with pointwise 95% CIs and headline BH4 q:

```text
'''+balance[balance.cohort.eq('N22')&balance.component.eq('condition_mean')][['model','roi','n','mean','ci_low','ci_high','p','q_headline_early_4']].to_string(index=False)+'''
```

The fixed A2 wording rule applies per ROI: declining-random attribution is permitted only when M's pointwise CI is entirely below zero; otherwise relative contributions remain unresolved. That wording rule does not establish a BH-corrected discovery or an absent structured increase. Full ΔS/ΔR/M CIs are in `roi/component_balance_results.csv`; per-row wording is in `roi/headline_tidy_all_specifications.csv`.

TFCE corrected voxel counts, aligned and reference side by side (both cohorts):

```text
'''+counts[['cohort','contrast','alpha','positive_voxels_reference','negative_voxels_reference','positive_voxels_aligned','negative_voxels_aligned']].to_string(index=False)+'''
```

'''+wholebrain_note+'''

Discovery counts for all stages/ROIs and all direct phase pairs are exported at the root. Differences in estimates, intervals, q values or thresholded voxel counts are descriptive, not tests of differences between models. The overlapping cohorts are not compared as independent samples. No primary model is reselected and no independent confirmation, causal interpretation or circuit localization follows from this sensitivity.

## Validation, preservation and runtime

```json
'''+json.dumps({k:x for k,x in v.items() if k!='output_sha256'},indent=2)+'''
```

All 176 shifted designs were independently reconstructed from original sidecars/events/nuisance inputs; all observed80 ROI map slots and 110 participant voxelwise maps were recomputed. Saved participant values independently reproduced stage, component, phase-component and direct-phase estimates, intervals and p values; all required BH families and paired algebraic identities passed. All unique first-level inputs were rehashed; output hashes and signed voxel counts passed. Per-analysis checks independently recomputed observed t, native TFCE, exact empirical max-t/TFCE FWE, and float32 map casts. Software: explicit project Nilearn **0.13.1** environment. No more than four first-level workers or eight permutation workers ran, in sequential phases; BLAS/OMP threads=1. JOBLIB_TEMP_FOLDER and MPLCONFIGDIR were explicitly inside `/tmp/r2-r2-slicetime/` for every analytical command.

The preservation baseline verifies existing code, docs, manuscript, resources, updated N22 results and all prior review analyses, without following directory symlinks. Its exact path/hash inventory is in owned scratch; validation records the count and any mismatch. All recorded files were unchanged. Historical raw/replication trees are not claimed as covered by this traversal; actual first-level input hashes cover their read-only analysis inputs.

First-level orchestration took '''+str(first['elapsed_seconds'])+''' s (cached two aligned controls included); total permutation computation took '''+str(float(counts.permutation_seconds.sum()))+''' s. Per-run and per-analysis timings are retained. Validation elapsed seconds are in the JSON above.

## Deviations and implementation notes

**No deviation from A7's scientific specification or owned-path rule.** A7 explicitly fixes permutation seed 20261018, overriding the general R1 default as a matched reference rerun. The preprocessed included-run range is narrower than the all-run range in the plan, as recorded above; no values were substituted.

An initial preservation scan traversed the large historical replication tree and was terminated before writing its baseline. The replacement scan explicitly covers the paths stated above and completed before full fitting finished. Metadata and control gates were already complete; the scan change did not change any analysis, input, fit or result. The initial and replacement logs are retained. Source manifests keep initial execution snapshots when final code assembly added or updated an unrelated source, alongside the final source inventory; no fitted producer or common-module source changed after a fit was produced.

## Commands and handoff

All commands ran from `/media/data3/chiaoen_srtt`, with `PYTHONDONTWRITEBYTECODE=1`, `JOBLIB_TEMP_FOLDER=/tmp/r2-r2-slicetime/joblib`, `MPLCONFIGDIR=/tmp/r2-r2-slicetime/matplotlib`, and interpreter `/media/data3/chiaoen_srtt/envs/nilearn-0131/bin/python -B`:

```sh
python -B code/review_r1/slicetime_preflight.py
python -B code/review_r1/slicetime_preflight.py --baseline-only
python -B code/review_r1/slicetime_fits.py controls --workers 4
python -B code/review_r1/slicetime_fits.py full --workers 4
python -B code/review_r1/slicetime_summaries.py
python -B code/review_r1/slicetime_voxelwise.py prepare
python -B code/review_r1/slicetime_voxelwise.py run
python -B code/review_r1/slicetime_finalize.py --seal-sources
python -B code/review_r1/slicetime_validate.py
python -B code/review_r1/slicetime_finalize.py
python -B code/review_r1/slicetime_check_artifacts.py
```

Read-only preparation used pwd, rg, cat, sed, tail, ls, git status, ps, du, AST parsing and Python inspections. Complete logs remain under owned scratch. `files_created.csv` lists SHA-256 and bytes for every owned code/result/report and retained scratch file, excluding the inventory itself and root validation.json to avoid a hash cycle. Root validation hashes the inventory; the report hash is listed there. Each analysis task has REPORT.md, validation.json and input_manifest.json.

Shared MANUSCRIPT_STATUS.md and REVISION_LOG.md were left unchanged because the user explicitly restricted this role to its owned paths. The coordinator should record A7 completion, validation and the next step in those shared handoff files. Independent audit/coordinator review and manuscript integration remain separate; no manuscript/UI/build verification, publication, external messages, commits or PRs were performed.
'''
    report_path=ROOT/'docs/review_round1/reports/r2-slicetime.md';report_path.write_text(report)
    t.finish(report,dict(passed=True,completed=['A7'],n_first_level_fits=176,n_zero_shift_controls=2,n_voxelwise_analyses=10,n_permutations_each=10000,deviations=[],manuscript_modified=False,independent_audit_pending=True))
    files=list((ROOT/'code/review_r1').glob('slicetime_*.py'))+[report_path]
    files += [p for p in OUT.rglob('*') if p.is_file() and p not in [OUT/'files_created.csv',OUT/'validation.json']]
    files += [p for p in Path('/tmp/r2-r2-slicetime').rglob('*') if p.is_file() and p.name!='finalize.log']
    csv([dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))],OUT/'files_created.csv')
    q=json.loads((OUT/'validation.json').read_text());q['output_sha256']['files_created.csv']=digest(OUT/'files_created.csv');writej(OUT/'validation.json',q)
    print('Inventory files',len(set(files)),flush=True)

if __name__=='__main__':
    if '--seal-sources' in sys.argv:seal_sources()
    else:main()
