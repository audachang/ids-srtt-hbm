# Matched-sample design diagnostics

All 352 reference/censored designs for N22 are checked for full rank, nuisance-adjusted task VIF, within-run sequence correlation and IID unit-noise contrast variance. Separate N20/N22 medians are exported. For reference designs, full-voxel fitted AR-bin histograms are regenerated for added/changed runs using the exact native OLS residual/Yule-Walker/binning rule; two unchanged controls must exactly match the archived full-fit histograms. Unchanged-run histograms are reused. Averaging whitened-design unit-innovation-noise contrast variances over the actual run-mask AR bins gives the reference AR diagnostic, not an ROI sampling variance or SE. DiD unit variances add independent-run contributions with the original sum weights. Censored diagnostics use their actual retained rows; full-mask AR-bin diagnostics are explicitly limited to reference fits.

Observed event lists and recovered source generation rules are documented in source_discovery; the original RNG seed/state remains unavailable. These diagnostics do not isolate prevalence from time/practice/familiarity.

