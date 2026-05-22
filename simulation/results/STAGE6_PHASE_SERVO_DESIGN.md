# Stage 6 Phase Servo Design

## Why Previous Patch Methods Failed

The formal baseline showed that dynamic object-frame replay is useful, but Stage 5 method patches did not produce a reliable improvement:

- `tau` is mathematically valid under clean constant-velocity synthetic delay, but it did not improve current PyBullet tracking or grasp behavior.
- CT did not meaningfully differ from CV under the current low-speed, short-horizon setup.
- Freeze-style contact gating disrupted otherwise successful replay timing.
- Pre-close gating avoided some timeout behavior but still failed to preserve baseline performance.
- Simple close-phase retiming had positive offline oracle opportunities, but online smoke did not preserve the strong 4 cm/s baseline and did not reliably reduce no-contact / `attempt_limit`.

The common issue is that these methods patched demo-clock replay timing. They did not replace the underlying assumption that a recorded static demo clock can drive dynamic contact timing.

## Conceptual Change

`dynamic_phase_servo` is an experimental architecture change. It keeps the same object tracker and low-level robot controller, but replaces blind demo-clock closing with explicit phase-aware object-relative servoing:

```text
APPROACH -> PREGRASP_SERVO -> CLOSE -> VERIFY_ATTACH -> LIFT -> DONE
                                                      -> FAILED
```

The controller uses object-relative targets generated from the current tracked object transform and selected static-demo key poses. The gripper close is event-triggered by geometry instead of being driven directly by the original demo clock.

## Phase Definitions

### APPROACH

- Purpose: move toward an object-relative approach pose while keeping the gripper open.
- Target pose: `T_delta(current) @ T_demo(approach_pose_time)`.
- Exit: EE-to-target XY error below `approach_ee_to_target_threshold_mm`.
- Timeout: `approach_timeout_s`.
- Failure: `approach_timeout`.

### PREGRASP_SERVO

- Purpose: servo to the object-relative pregrasp/contact pose while keeping the gripper open.
- Target pose: `T_delta(current) @ T_demo(pregrasp_pose_time)`.
- Exit: EE-to-target XY error below `pregrasp_ee_to_target_threshold_mm` and EE-to-object XY distance below `close_ee_to_object_threshold_mm`.
- Timeout: `pregrasp_timeout_s`.
- Failure: `pregrasp_timeout`.

### CLOSE

- Purpose: maintain the object-relative pregrasp/contact pose while closing the gripper.
- Target pose: `T_delta(current) @ T_demo(pregrasp_pose_time)`.
- Exit: fixed-constraint attachment is created by the existing attach logic.
- Timeout: `close_timeout_s`.
- Failure: `close_timeout`.

### VERIFY_ATTACH

- Purpose: allow the existing fixed-constraint attachment state to be verified without changing attach logic.
- Target pose: current close/pregrasp target.
- Exit: attachment exists.
- Timeout: `verify_attach_timeout_s`.
- Failure: `attach_failed`.

### LIFT

- Purpose: lift after attachment using a target close to the existing demo lift behavior.
- Target pose: an object-relative lift demo pose when tracking remains available; otherwise the existing world/demo lift pose.
- Exit: demo lift duration elapsed or success criteria satisfied at trial end.
- Failure: existing success criteria decide `lift_failed` / other trial failures.

### DONE

- Purpose: terminal completed phase after lift sequence.

### FAILED

- Purpose: terminal failure phase with explicit `phase_failure_reason`.

## Logged Metrics

Raw trial fields:

- `phase_controller_enabled`
- `final_phase`
- `phase_failure_reason`
- `approach_duration_s`
- `pregrasp_duration_s`
- `close_duration_s`
- `verify_attach_duration_s`
- `lift_duration_s`
- `n_phase_transitions`
- `close_triggered`
- `close_trigger_sim_t`
- `close_trigger_ee_to_target_mm`
- `close_trigger_ee_to_object_mm`
- `attach_verified`
- `phase_timeout`

Frame diagnostics, when enabled, include:

- `sim_t`
- `phase`
- `t_demo`
- EE position
- object position
- target position
- EE-to-target XY error
- EE-to-object XY error
- gripper width
- `close_triggered`
- `attach_verified`

## Intentionally Not Changed

The phase-servo prototype does not change:

- PyBullet physics.
- Object trajectory.
- Low-level Cartesian velocity controller.
- Success criteria.
- Retry and fixed-constraint attachment logic, except that phase timeouts can terminate the phase-servo path explicitly.
- The canonical formal baseline results.
- Existing baseline conditions such as `dynamic_tau0`.

## Risks and Expected Failure Modes

- The chosen phase targets may be too crude because they use a few static-demo key poses instead of a full manipulation policy.
- Phase timeouts may expose that the low-level controller cannot reach some object-relative targets quickly enough.
- Event-triggered close may still be poorly synchronized with PyBullet's simplified attachment condition.
- The fixed-constraint grasp simplification can create misleading successes or failures.
- The architecture may preserve interpretability while reducing success if the phase definitions are wrong.
- A robust version likely needs explicit contact verification and a true object-relative servo controller, not only phase labels around existing pose commands.
