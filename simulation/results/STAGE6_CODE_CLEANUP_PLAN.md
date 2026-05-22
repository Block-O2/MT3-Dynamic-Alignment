# Stage 6 Code Cleanup Plan

This plan classifies the current code paths after Stage 6A. No code is deleted in this stage.

## Keep

### `static_replay`

- Classification: keep.
- Reason: required moving-object failure baseline.

### `dynamic_tau0`

- Classification: keep.
- Reason: strongest reliable method from the formal baseline and later smoke tests.
- Recommendation: keep as the primary dynamic baseline.

### `simulation/run_experiment.py` logging and output structure

- Classification: keep.
- Reason: reproducible command/config/raw/summary/log/analysis output is central to the project.

### Attempt diagnostics

- Classification: keep.
- Reason: `attempt_diagnostics.csv` helped diagnose no-contact / `attempt_limit` failures.
- Recommendation: keep behind `--record-diagnostics`.

## Keep As Experimental Or Remove

### `dynamic_tau0_contact_gated`

- Classification: keep as experimental or remove.
- Reason: freeze-style contact gate was not promising and damaged the 4 cm/s case.
- Recommendation: do not present as a recommended method.

### `dynamic_tau0_preclose_gated`

- Classification: keep as experimental or remove.
- Reason: pre-close gating avoided some timeout behavior but did not preserve baseline performance.
- Recommendation: do not present as a recommended method.

### `dynamic_tau0_close_retimed`

- Classification: keep as experimental or remove.
- Reason: offline feasibility was positive, but online smoke was not promising.
- Recommendation: keep only for reproducibility of the negative Stage 5C result, or remove after archiving the summary.

### `dynamic_phase_servo`

- Classification: keep as experimental or remove.
- Reason: Stage 6A architecture smoke failed at all tested speeds and did not preserve the baseline.
- Recommendation: do not present as a recommended method. If retained, label clearly as a failed prototype.

## Keep As Diagnostic Scripts

### `simulation/run_delay_tau_audit.py`

- Classification: keep as diagnostic.
- Reason: useful synthetic validation of `predict_ahead(tau)` under clean CV assumptions.

### `simulation/run_latency_tracking_only.py`

- Classification: keep as diagnostic if useful.
- Reason: useful for tracking-only latency checks, but results were negative for the current PyBullet setup.

## Do Not Present Failed Variants As Recommended Methods

The repository should not leave failed exploratory variants looking like formal baselines or successful improvements. The current recommendation is:

- primary method: `dynamic_tau0`
- documented failure baseline: `static_replay`
- diagnostic or experimental only: tau/latency modes, CT, contact gates, close retiming, phase servo

## Suggested Cleanup After Approval

After explicit approval, one of these cleanup paths would be appropriate:

- Move failed variants into an experimental namespace or separate script.
- Hide failed variants from default CLI examples while preserving them for reproducibility.
- Remove failed method variants after summaries are committed, keeping diagnostics and canonical baseline results.

Do not delete raw/canonical results.
