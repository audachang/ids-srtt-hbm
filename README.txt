IDS-SRTT HBM companion — v1.1.0 (2026-10-19)

Scope: analysis source code and group summaries only.
Public repository: https://github.com/audachang/ids-srtt-hbm
GitHub release: https://github.com/audachang/ids-srtt-hbm/releases/tag/v1.1.0
Archived version: v1.1.0.
Version DOI: https://doi.org/10.5281/zenodo.23471923
Frozen commit: c05796046172e26daa25b5ca35f7ee1b5cd9527d
Version 1.0.0 remains available at doi:10.5281/zenodo.23438462.
Licenses: original software under code/ is MIT; group summaries, figures,
documentation and environment records are CC BY 4.0. See LICENSE.txt and LICENSES/.

Version 1.1.0 adds imaging and behavioral summaries for 22 participants,
group figures, source descriptions and verification instructions. The earlier
20-participant products from version 1.0.0 are retained unchanged.
The 22-participant analysis with the uncensored reference model and at least
80% observed regional support is primary. The learner-selected 20-participant
subsample and motion-censored models are sensitivity analyses. The subsample
appears only in the article supplement. The samples overlap and are not
independent groups. The analyses were not preregistered. The reporting model
was chosen after earlier analyses; see the article Methods.

Quick start (NumPy, pandas and SciPy required):
  python code/reporting/verify_public_summaries_n22.py .
Checks file checksums and inventory, t statistics, pointwise t intervals,
RT back-transformations, planned BH correction families and blank suppressed
cells. Bootstrap percentile intervals are checked for ordering. These checks
verify supplied aggregate outputs, not the underlying participant models.
Software-environment records from version 1.0.0 are retained for reference.

FILE_DICTIONARY.csv describes file units and contents. COLUMN_DICTIONARY.json
explains statistical columns. RELEASE_COVERAGE.csv lists each source file
behind the article's reported values and whether it is included, included
with suppression, or withheld. Of 61,572 recorded values, 53,742 can be
checked against released files. Some values remain unavailable because their
sources contain individual observations or their text references changed
when identifying details were removed.

Estimates from fewer than two participants are blank; blank means missing,
not zero. Sample sizes, planned comparison slots and BH-adjusted probabilities
are retained. AGGREGATION_NOTES.json describes suppression and redaction.
Original source hashes in saved validation reports describe their source files;
SHA256_manifest.json gives release hashes for all files except itself.
FIGURE_REVIEW.json gives source and release hashes for all ten added figures.
AGGREGATION_NOTES.json records the specifications PDF, whose original page 8
was omitted because its estimates came from one participant.

Restricted inputs include raw/preprocessed MRI, participant maps, masks,
confounds and censor vectors, trial logs, individual summaries, random effects,
fitted models, bootstrap identities and participant input manifests.
Leave-one-out and influence tables are withheld because individual values
can be recovered from them. Seven scripts containing individual observations
are also withheld; RELEASE_EXCLUSIONS.json lists them and explains why.
Subject labels and filesystem paths in code refer to restricted inputs that
are not distributed. Subject-list configuration and data paths must be set
for an approved copy of the inputs. Separate software dependencies and
atlas/template resources are required under their own terms.

The article's table-generation script, make_n22_manuscript_assets.py, is
included for reference. It requires restricted source records, original inputs
with matching checksums and individual influence values, so it cannot run
unchanged using this public subset. Released group estimates can be used to
format tables. Full model re-estimation requires approved restricted inputs
and any excluded helper scripts needed by the selected analysis.

Requests for participant data go to the corresponding authors and depend on
participant consent, ethics approval and institutional requirements; access
is not guaranteed. No participant-level data access is granted by this release.
