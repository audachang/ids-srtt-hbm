Analysis code and group summaries for Neural responses during motor statistical learning and gradual sequence replacement
IDS-SRTT HBM companion — v1.2.2 (2026-10-20)

Analysis source code and group summaries only.
Public repository: https://github.com/audachang/ids-srtt-hbm
GitHub release: https://github.com/audachang/ids-srtt-hbm/releases/tag/v1.2.2
Version 1.2.2 published on 2026-10-20.
Version DOI: https://doi.org/10.5281/zenodo.23485389
Frozen commit: 80147786fb839dbd6b5e6bb43cade0542180aad3
Version 1.2.1: doi:10.5281/zenodo.23484681.
Version 1.2.0: doi:10.5281/zenodo.23474855.
Version 1.1.0: doi:10.5281/zenodo.23471923.
Version 1.0.0: doi:10.5281/zenodo.23438462.

What's new in 1.2.2

This version adds the continuing-sequence benchmark, early new-sequence estimates,
and participant-paired new-minus-continuing comparisons with their source code.
The continuing contrast compares the old sequence with Random from the initial
phase to early and middle replacement. All contrasts retain the historical two-run
summed scale and the same eligible participants within each parcel. The continuing
benchmark reports separate four-parcel headline and 48-parcel families for each
phase and cohort; the early new and paired comparisons retain their 48-parcel
families. The article title now uses "gradual sequence replacement".

These secondary analyses are exploratory and do not change the primary uncensored
reference analysis. Confidence intervals are pointwise. Overlapping cohorts are
not independent groups; intervals containing zero do not establish equivalence.
The changing prevalence schedule cannot identify a causal effect of replacement.
Brodmann area 6 parcels retain the storage names LeftSMA and RightSMA; they cover
the full Brodmann parcels. The slice-timing-aligned sensitivity is retained.

Complementary suppression applies jointly to the added and inherited group summaries,
including duplicates and related components. Numerical ROI reports that could restore
suppressed values are withheld. This version supersedes version 1.2.1 for citation.
Earlier published versions remain available and unaltered; this release cannot
remove disclosures already present in earlier archives.

The 22-participant uncensored reference analysis
with at least 80% observed regional support remains primary. The overlapping
learner-selected 20-participant subsample appears only in the supplement.
Intervals are pointwise. Analysis choices followed earlier observations;
see the article Methods for the disclosed model-selection history.

Verify the supplied summaries (NumPy, pandas and SciPy required):
  python -B code/reporting/verify_public_summaries_r2.py .
  python -B code/reporting/verify_public_summaries_r2.py --legacy .
The first command checks new aggregate arithmetic, t intervals, back-transforms,
48/192/144/4-slot imaging and 24-slot behavioral BH families, suppressed cells,
file inventory and coverage totals. It does not fit models or generate new
scientific results. Bootstrap intervals are checked for ordering and their
back-transforms; their estimation requires withheld observations.
Families containing withheld p values are checked against admissible BH bounds;
exact q recomputation is limited to complete families. The saved q values and
planned slots are retained. All CSVs are checked for nested-sample disclosure.
The second runs the unchanged v1.1.0 verifier using its preserved original
coverage table, RELEASE_COVERAGE_V1_1.csv. Its original direct command assumes
the former coverage size and does not support the expanded coverage table.
Its checksum, available arithmetic, code, retained-figure and suppression checks run
against this release. Its family specification assigns incomplete BH families
to the new verifier's bounds check. Neither command verifies the manuscript or model fits.

FILE_DICTIONARY.csv describes every file. COLUMN_DICTIONARY.json explains
statistical units; AGGREGATE_CHECK_SPEC_R1.json, AGGREGATE_CHECK_SPEC_R1_1.json and AGGREGATE_CHECK_SPEC_R2.json specify the added checks.
RELEASE_COVERAGE.csv accounts for 301,453 article
values from 122 sources, of which
282,321 values can be checked against public files.
COVERAGE_STATUS.json records completeness and the source inventory checksum.
Restricted source filenames use opaque references in the coverage table;
their recorded values remain withheld and their source checksums are retained.
Source hashes in saved validation reports describe original outputs;
SHA256_manifest.json gives current release hashes for all files except itself.
AGGREGATION_NOTES.json records redactions and suppression. For N<2, estimates
and test values are blank; N, planned slots and BH q are retained. Blank means
missing. Smaller nested sets differing by one observation, equivalent displays,
and their related ROI components and contrasts have all statistics blanked;
N and BH q remain. Affected ROI figures and derived tables/reports are withheld.
Singleton design-frequency durations and exact individual timing
precision are withheld. Complete realized order tables with singleton patterns
are excluded. The single-run timeline CSV is withheld. The retained timeline
figure is an example run schedule showing stimulus design only, with no responses.

Restricted inputs include raw/preprocessed MRI, participant maps and masks,
confounds, censor vectors, trial logs, individual summaries and random effects,
fitted models, bootstrap identities, raw nulls and participant manifests.
Leave-one-out values are withheld because individual observations are recoverable.
Seven complete original scripts embedding individual observations remain
withheld. The released code/reanalysis/run_saved.py contains only the generic
direct comparison function copied verbatim, with necessary imports. Its source
checksum and line range are in CODE_REVIEW.csv; no other imported function
from that original is missing. Subject labels and paths in code refer to
restricted inputs that are not distributed. Re-estimation requires approved
input access, subject-list configuration, the appropriate SRTT_SOURCE_ROOT,
separate software and atlas/template resources, and any additional restricted
helpers needed for the chosen analysis. Public summaries alone cannot rerun
participant models or the original article asset-generation scripts.

Original software under code/ is MIT. Group summaries, figures, documentation,
environment records and descriptive metadata are CC BY 4.0. See LICENSE.txt
and LICENSES/; third-party resources retain their own terms. Requests for
participant data go to the corresponding authors and depend on participant
consent, ethics approval and institutional requirements. Access is not granted
by this release and is not guaranteed.
