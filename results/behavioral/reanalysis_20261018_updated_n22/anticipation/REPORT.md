# Updated anticipation analysis

N20 and N22 rates, accuracy and original eight-trial positions are reported before RT/error exclusions. Any responded 0<RT<100ms trial is anticipatory, including incorrect presses. Twenty thousand whole-participant bootstrap draws per cohort provide descriptive pointwise probability-difference intervals for the fixed 24 contrast vectors. They are not GLMM odds ratios and carry no corrected-significance claim.

N22 retains a zero-event run-3 random cell, so the requested saturated 20-cell unregularized logit GLMM still has no finite MLE. The earlier N20 optimizer diagnostics are preserved; no unstable N22 odds ratios or p values are reported. No regularized/reduced model is silently substituted. The source logs/code now exist, but complete keyboard-event response assignment/cue-offset reconstruction remains a separate audit, and RT is not actual measured cue duration.

