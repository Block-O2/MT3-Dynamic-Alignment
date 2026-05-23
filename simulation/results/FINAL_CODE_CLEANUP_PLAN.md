# Final Code Cleanup Plan

This plan classifies the current experiment conditions and support utilities after the full exploration. No method code is deleted in this cleanup stage.

## Keep As Primary

### `dynamic_tau0`

Keep as the primary current method.

Reason:

- It was the strongest reliable condition in the formal baseline.
- It clearly outperformed `static_replay`.
- Later experimental variants did not reliably improve it.

## Keep As Baselines

### `static_replay`

Keep as the moving-object failure baseline.

Reason:

- It demonstrates that static world-frame replay fails on moving tabletop objects.

### `dynamic_cv`

Keep as a baseline if prediction/tau comparisons are discussed.

Reason:

- It represents CV Kalman prediction with `tau=0.1`.
- It did not outperform `dynamic_tau0` in latency-free PyBullet.

### `dynamic_ct`

Keep as an optional baseline.

Reason:

- It exercises the CT/EKF path.
- It did not meaningfully differ from CV under the current planar motion setup.
- It should not be presented as a successful improvement.

## Keep As Diagnostic Only

### Latency validation utilities

Keep as diagnostic-only utilities:

- synthetic delay/tau audit
- PyBullet latency tracking-only script or mode
- observation-delay and control-delay diagnostics

Reason:

- They document that tau is synthetic-valid but not validated as a PyBullet tracking/grasp improvement.
- They should not be presented as formal grasp experiments.

### Attempt diagnostics

Keep.

Reason:

- Attempt-level diagnostics were useful for identifying `attempt_limit` / no-contact failures.
- They support honest failure analysis and reproducibility.

### Frame diagnostics

Keep optional.

Reason:

- Frame diagnostics are useful for debugging but should not be produced by default in formal runs.

## Remove Or Mark Experimental / Not Recommended

The following conditions should either be removed in a future approved cleanup or clearly hidden behind an experimental flag. They should not appear as recommended baselines.

### `dynamic_tau0_contact_gated`

Classification: experimental / not recommended.

Reason:

- Freeze-style gate did not reduce no-contact / attempt-limit failures.
- It damaged the strong 4 cm/s baseline case.

### `dynamic_tau0_preclose_gated`

Classification: experimental / not recommended.

Reason:

- Pre-close freezing did not preserve baseline performance.
- It remained a brittle timing patch.

### `dynamic_tau0_close_retimed`

Classification: experimental / not recommended.

Reason:

- Offline close opportunities existed, but online smoke was not promising.
- It did not reliably preserve or improve `dynamic_tau0`.

### `dynamic_phase_servo`

Classification: experimental / not recommended.

Reason:

- Smoke test failed at all tested speeds.
- Close triggers occurred, but attach/lift failed after trigger.
- It likely disrupted the original demo's continuous contact/lift timing.

### `feasibility_aware_replay`

Classification: experimental / not recommended.

Reason:

- It preserved the MT3-style formulation but froze too much.
- It improved 8 cm/s in smoke but hurt 2/4 cm/s.

### `feasibility_aware_replay_v2`

Classification: experimental / not recommended.

Reason:

- It reduced freeze frames slightly.
- It did not recover the 2/4 cm/s baseline.
- Final demo progress barely improved.

## CLI Cleanup

The runner should keep official/default conditions focused on:

- `static_replay`
- `dynamic_tau0`
- `dynamic_cv`
- `dynamic_ct`

Failed method variants should require an explicit experimental flag and should be labeled as experimental / not recommended. Diagnostics options should remain available because they support reproducibility and failure analysis.

Do not delete canonical formal baseline data.

## Current Recommended Method

```text
dynamic_tau0
```

The repository is ready to support a clean simulation baseline / negative-results report. Further method development should be done on a new branch if it introduces a principled optimizer rather than more threshold patches.
