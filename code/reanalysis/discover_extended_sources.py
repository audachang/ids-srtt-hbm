#!/usr/bin/env python3
"""Read-only source provenance audit across home and media-drive archives."""
from common import *
import re
import shutil

EXP3 = Path('/home/fmri2404/chiaoen/Chiaoen_AllExpData/Exp3')
DRIVE = Path('/media/data2/pinwei/Joanne_SRT_2')
SHARED = DRIVE / 'For_teamNCU_Joanne'
N22 = SUBJECTS + ['sub-005', 'sub-023']


def main():
    t = Task('source_discovery')
    scope = json.loads(Path('/tmp/srtt_extended_root_discovery.json').read_text())
    writej(t.path / 'search_scope.json', scope)
    locations = [
        (DRIVE / 'data', 'normalized trial CSVs', '192 files match the current source bytes'),
        (SHARED / 'Import_data', 'native PsychoPy CSVs', '22 participants, eight runs each'),
        (SHARED / 'My_modifiedRecursive.R', 'original custom RT-cleaning helper', 'recovered source; not executed by this audit'),
        (SHARED / 'fmri_srtt_beh_new2_analysis.R', 'original behavioral analysis', 'learner-test implementation found'),
        (SHARED / 'fmri_behavioral_data.csv', 'participant/cognitive metadata', '25-row table with keep/delete annotations; no exclusion reasons'),
        (EXP3 / '實驗程式', 'condition lists and stimulus-generation code', '1,408 block lists match all N22 logged targets/rules'),
        (EXP3 / 'fmri_new_V2/實驗程式/myproject_new_prob_v2/data', 'native PsychoPy CSV/log/psydat archive', 'partial native archive; complement with Import_data'),
        (Path('/media/data3/Joanne_SRT_pw/data/fmriprep_dropped'), 'additional excluded-participant imaging inputs', '005/023 BOLD/confounds/motion copies compared'),
        (Path('/media/data2/pinwei/Proj_CSR/data/SRT/Joanne'), 'other SRT study candidate', 'explicit/implicit datasets; not established as this fMRI cohort'),
        (Path('/media/data3/fmriworkshop2025/Joanne'), 'other project', 'FaceRecognition; not merged into SRTT'),
    ]
    csv([dict(path=str(p), role=role, finding=finding, exists=p.exists()) for p, role, finding in locations],
        t.path / 'source_locations.csv')
    duplicates = []
    for source in sorted((DRIVE / 'data').glob('*srtt_prob_new.csv')):
        canonical = ROOT / 'data/behavioral' / source.name
        a, b = digest(t.record(source)), digest(t.record(canonical))
        assert a == b
        duplicates.append(dict(drive_path=str(source), canonical_path=str(canonical), sha256=a, byte_identical=True))
    assert len(duplicates) == 192
    csv(duplicates, t.path / 'normalized_trial_copy_comparison.csv')
    native = []
    for source in sorted((SHARED / 'Import_data').glob('*.csv')):
        canonical = EXP3 / 'fmri_beh_V2/SRTT/Import_data' / source.name
        a, b = digest(t.record(source)), digest(t.record(canonical))
        assert a == b
        native.append(dict(drive_path=str(source), home_path=str(canonical), sha256=a, byte_identical=True))
    assert len(native) == 176
    csv(native, t.path / 'native_csv_copy_comparison.csv')
    blocks = []
    for subject in N22:
        for run in range(1, 9):
            raw_path = t.record(ROOT / 'data/behavioral' / f'{subject[-3:]}_run_{run}_srtt_prob_new.csv')
            raw = pd.read_csv(raw_path)
            for block, group in raw.groupby('block', sort=False):
                path = t.record(EXP3 / '實驗程式' / f'condlist_sub{int(subject[-3:])}_{block}.csv')
                actual = pd.read_csv(path)
                assert len(actual) == len(group) == 40
                np.testing.assert_array_equal(actual.keypress, group.keypress)
                np.testing.assert_array_equal(actual.rule, group.rule)
                blocks.append(dict(subject=subject, run=run, block=block, path=str(path),
                                   sha256=digest(path), n_matching_trials=len(group), targets_and_rules_match=True))
    assert len(blocks) == 1408
    csv(blocks, t.path / 'condition_list_trial_comparison.csv')
    metadata = pd.read_csv(t.record(SHARED / 'fmri_behavioral_data.csv'))
    selection = metadata[['participant', 'Filter1']].copy()
    selection['subject'] = selection.participant.str.replace('CE', 'sub-', regex=False)
    assert set(selection[selection.Filter1 == 'keep'].subject) == set(SUBJECTS)
    csv(selection, t.path / 'participant_selection_annotations.csv')
    t.record(SHARED / 'ids-srtt_fmri_behavioral_data.csv')
    code = [
        ('custom_cleaner', SHARED / 'My_modifiedRecursive.R'),
        ('drive_analysis', SHARED / 'fmri_srtt_beh_new2_analysis.R'),
        ('home_analysis', EXP3 / 'fmri_beh_V2/SRTT/fmri_srtt_beh_new2_analysis.R'),
        ('home_shared_analysis', EXP3 / 'fmri_beh_V2/SRTT/For_teamNCU_Joanne/fmri_srtt_beh_new2_analysis.R'),
        ('stimulus_generator', EXP3 / '實驗程式/Create_stimuli.py'),
        ('time_conversion', EXP3 / 'fmri_new_V2/create_log_files_new_V2.R'),
        ('program_archive_executable', EXP3 / '實驗程式/SRTT_fMRI_prob_new_exe_v2.py'),
        ('program_archive_lastrun', EXP3 / '實驗程式/SRTT_fMRI_prob_new_v2_lastrun.py'),
        ('existing_drive_cleaner', Path('/media/data3/Joanne_SRT_pw/scripts/proc_behav_data.R')),
    ]
    versions = []
    for label, path in code:
        source = t.record(path)
        target = t.path / 'source_snapshots' / (label + path.suffix)
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        versions.append(dict(role=label, original=str(source), snapshot=str(target), sha256=digest(source)))
    csv(versions, t.path / 'code_versions.csv')
    comparisons = []
    alternate = Path('/media/data3/Joanne_SRT_pw/data/fmriprep_dropped')
    for subject in ['sub-005', 'sub-023']:
        for run in range(1, 9):
            bold, _, confounds, _ = paths(subject, run)
            for kind, canonical in [('BOLD_MNI', bold), ('confounds', confounds)]:
                source = alternate / subject / 'func' / canonical.name
                a, b = digest(t.record(source)), digest(t.record(canonical))
                comparisons.append(dict(subject=subject, run=run, kind=kind, additional_copy=str(source),
                                        canonical=str(canonical), additional_sha256=a, canonical_sha256=b,
                                        byte_identical=a == b))
        source = alternate / subject / 'func/head-motion_confound.1D'
        canonical = Path('/home/fmri2404/chiaoen/SrttNewV2/Nifti/derivatives/fMRIPrep') / subject / 'func/Confounds/head-motion_confound.1D'
        a, b = digest(t.record(source)), digest(t.record(canonical))
        comparisons.append(dict(subject=subject, run='all', kind='AFNI_motion', additional_copy=str(source),
                                canonical=str(canonical), additional_sha256=a, canonical_sha256=b, byte_identical=a == b))
    csv(comparisons, t.path / 'additional_imaging_copy_comparison.csv')
    n_same = sum(x['byte_identical'] for x in comparisons)
    settings = dict(minRT=0, trimming_condition='block', omitErrors_default=True,
                    learner_run=2, structured_blocks=[f'Ses2Str_{i}' for i in range(1, 7)],
                    random_block='Ses2Ran2', variance_test='R var.test, two-sided default',
                    equal_variances_if_F_test_p_gt=.05, t_test_alternative='greater',
                    t_test_group_order=['Ran', 'Str'], original_cleaner_replayed=False,
                    original_learner_output_recovered=False)
    writej(t.path / 'recovered_learner_test_settings.json', settings)
    report = f'''# Extended source search: home and /media

This follow-up updates source availability; it does not refit imaging or behavioral models. Original files, previous reports and manuscript sources are unchanged. Research-relevant recovered code is copied verbatim into source_snapshots/ with source hashes.

## Confirmed sources

- `/media/data2/pinwei/Joanne_SRT_2/data/`: 192 trial CSVs, all byte-identical to the current source. They are duplicate source data, not additional participants.
- `/media/data2/pinwei/Joanne_SRT_2/For_teamNCU_Joanne/Import_data/`: 176 native CSVs for N=22, eight runs each. All match the home Exp3/SRTT/Import_data versions byte-for-byte.
- `For_teamNCU_Joanne/My_modifiedRecursive.R`: recovered custom cleaner. The original analysis calls it with minRT=0, condVar=block, errors omitted by default. This is distinct from the currently saved rule-wise trimr cleaning and needs its own reproduction/comparison.
- `fmri_srtt_beh_new2_analysis.R`: the recovered learner test uses ALL SIX run-2 structured blocks versus the final random block, an F-test for equal variances (p>.05 chooses pooled variance; otherwise Welch), then a one-sided Ran>Str t-test. This conflicts with the previously inferred final-structured-block comparison. The three analysis versions are hashed separately; original cleaning/test execution and historical learner-test results are not certified by merely finding these files.
- `fmri_behavioral_data.csv`: 25 participant metadata rows, including cognitive measures and a Filter1 keep/delete flag. Its 20 keep IDs exactly match the retained imaging list. Delete flags include 003/005/007/011/023, but do not specify the reason; this is not a complete exclusion log. Only IDs and flags are exported, not demographic/cognitive values.
- `/home/fmri2404/chiaoen/Chiaoen_AllExpData/Exp3/實驗程式/`: 1,408 condition-list blocks match all 56,320 logged N=22 targets and physical rule labels. The recovered generator documents random targets sampled with immediate repeats forbidden; structured A/B order is shuffled under specified frequency and ordering constraints. Actual saved condition lists are available, but the original random seed/state is not established.
- `/media/data3/Joanne_SRT_pw/data/fmriprep_dropped/`: additional 005/023 imaging inputs. {n_same}/{len(comparisons)} compared BOLD, confound and original AFNI-motion copies match the currently used source bytes. See the per-file table for differences if any.

## Other roots and search limits

`/media/data2/pinwei/Proj_CSR/data/SRT/Joanne/` contains explicit/implicit SRT datasets whose relation to this fMRI cohort is unverified. They were not merged. `/media/data3/fmriworkshop2025/Joanne/` is a FaceRecognition project. The large `/media/data3/14456124.zip` index identifies BOLD5000 data, not this SRTT experiment; its contents were not extracted.

The initial directory-name scan covered /home/aclexp, /home/fmri2404 and /media to depth six, stopping at named candidate roots; selected research roots were then indexed and checked in detail. See search_scope.json for exact pruning and unreadable directories. Git internals, credential directories, caches, environments and agent-chat histories were not read. Symlink targets and paths beyond that discovery depth were not exhaustively searched. This is an evidence-backed source inventory, not a claim that no other copies exist anywhere on the machine.

## Analysis implications

Use the verified files as provenance for the planned rerun. Replay the recovered custom cleaner/test in a new output version and compare it with saved cleaning before changing the Methods description or interpreting 005/023 as non-learners. Reconcile all-six-block versus final-block wording using the recovered code and author history. Document stimulus generation using both the actual lists and the recovered code. Timing repairs and newly reconstructed censor masks remain separate versioned inputs from the preceding follow-up; this search does not replace or invalidate their direct timestamp/mask checks. No new cognitive covariates or brain-behavior associations are automatically added.
'''
    t.finish(report, dict(passed=True, normalized_duplicate_files=192, native_duplicate_files=176,
                          matching_condition_lists=1408, matching_logged_trials=56320,
                          explicit_keep_ids_match=True, custom_cleaner_recovered=True,
                          additional_imaging_comparisons=len(comparisons), additional_imaging_byte_matches=n_same,
                          original_learner_analysis_replayed=False, n22_imaging_refitted=False),
             status='source_inventory_complete_with_provenance_questions')


if __name__ == '__main__':
    main()
