# N20 reaction-time mixed model

The source-replayed custom block cleaner supplies the correct responded trials. Log RT uses the unchanged 20 observed run-condition cells, independent participant intercept/sequence slopes and nested run/block intercepts; REML, within-block AR(1) indexed by original trial order, condition-specific residual variance. The primary policy retains RT>0; the sensitivity requires RT≥100ms. The same 24 contrast vectors are used. Tests use approximate t degrees of freedom capped at 19, with BH across 24 per policy; pointwise model-based 95% intervals and geometric ratios are exported. The df cap is not an exact or guaranteed conservative calibration.

- primary: 49069 trials, AR1=0.2816, 16/24 BH findings.
- min100: 47236 trials, AR1=0.2929, 18/24 BH findings.

N20 saved models were reused only after an exact saved-data check; its trial keys and RT values also reproduce the historical cleaned dataset. N22 was refitted. Primary and min100 estimates answer different response-selection questions.

