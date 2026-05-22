# Stage 5 Code Cleanup Plan

This plan classifies the experimental code added during Stage 4A and Stage 5. No code is removed in this stage.

## Primary Method

### `dynamic_tau0`

- Classification: keep.
- Reason: strongest reliable method from the formal baseline.
- Recommendation: keep as the primary dynamic replay baseline and the default base method for any future redesign.

## Failed / Experimental Method Variants

### `dynamic_tau0_contact_gated`

- Classification: remove or mark experimental.
- Reason: freeze-style gate did not reduce no-contact / `attempt_limit` failures and damaged the strong 4 cm/s case.
- Recommendation: do not leave this visible as a recommended baseline. If retained, label it clearly as a failed Stage 5A diagnostic condition.

### `dynamic_tau0_preclose_gated`

- Classification: remove or mark experimental.
- Reason: pre-close alignment avoided some timeout behavior but still did not preserve baseline performance or reliably reduce no-contact failures.
- Recommendation: do not use in formal comparisons. Keep only if needed to reproduce the negative Stage 5A.1 result.

### `dynamic_tau0_close_retimed`

- Classification: remove or mark experimental.
- Reason: offline oracle feasibility was positive, but online smoke did not preserve 4 cm/s and did not reliably reduce no-contact / `attempt_limit` failures.
- Recommendation: do not present as a method improvement. Keep only as a reproducibility hook for the Stage 5C negative result, or move to a separate experimental script/module.

## Latency Validation Modes

### `--latency-validation` in `simulation/run_experiment.py`

- Classification: keep but mark experimental.
- Reason: useful for reproducing the Stage 4A negative PyBullet latency result, but not part of the primary method.
- Recommendation: retain only if documented as a diagnostic mode. Do not mix latency grids into baseline method claims.

### `simulation/run_delay_tau_audit.py`

- Classification: keep.
- Reason: lightweight synthetic validation of `predict_ahead(tau)` semantics. It supports the claim that the math is valid under clean CV assumptions.
- Recommendation: keep as a diagnostic script.

### `simulation/run_latency_tracking_only.py`

- Classification: keep but mark experimental.
- Reason: useful for tracking-only tau diagnostics, but the result was negative for PyBullet tracking.
- Recommendation: keep as a diagnostic script and avoid presenting it as a method.

## Diagnostics

### Frame-level diagnostics

- Classification: keep.
- Reason: useful for failure analysis and metric validation.
- Recommendation: keep behind `--record-diagnostics` so routine experiments do not produce excessive files by default.

### `attempt_diagnostics.csv`

- Classification: keep.
- Reason: Stage 5B showed attempt-level close timing diagnostics are useful for understanding no-contact / `attempt_limit` failures.
- Recommendation: keep as a diagnostic output when `--record-diagnostics` is enabled.

## Runner / Output Schema

### Additional raw and summary fields for failed variants

- Classification: keep but mark experimental.
- Reason: fields such as `contact_gate_*` and `close_retime_*` are useful for reproducing negative results, but they make the main output schema larger.
- Recommendation: either keep them with clear documentation or move method-specific fields into separate diagnostic CSVs in a later cleanup.

## Overall Recommendation

Keep diagnostics. Keep `dynamic_tau0` as primary. Mark or remove failed method variants before presenting the repository as a clean baseline study. Do not leave `dynamic_tau0_contact_gated`, `dynamic_tau0_preclose_gated`, or `dynamic_tau0_close_retimed` looking like recommended baselines.

The safest next cleanup step, after explicit approval, is to move failed method variants behind an `experimental` namespace or remove them from default CLI choices while preserving the Stage 5 reports.
