# Stage 4A Latency Validation Summary

## Motivation

`tau` is intended to represent measured total system latency compensation: perception + computation + actuation. It is not a free simulation tuning knob and should not be presented as the main method contribution.

The formal PyBullet baseline showed that `dynamic_cv` with `tau=0.1` did not outperform `dynamic_tau0` in the latency-free PyBullet setup. That result did not automatically invalidate `tau`, because the formal baseline did not contain a controlled latency source. Stage 4A tested whether `predict_ahead(tau)` becomes useful only when a known delay is injected.

## Stage 4A: Grasp-Level Artificial Observation Delay

Stage 4A injected artificial observation delay into the grasp-level replay path. The object continued moving at current simulation time, while the tracker received delayed object point-cloud content. The smoke comparison included delay-free and 100 ms delayed observations with `tau=0` and `tau=0.1`.

Matched `tau` did not improve success or failure behavior under injected observation delay. In the delayed cells, `tau=0.1` was worse than `tau=0` in the smoke run.

Conclusion:

```text
TAU_NOT_VALIDATED
```

## Stage 4A.1: Synthetic CV Delay Audit

Stage 4A.1 isolated the Kalman time semantics without PyBullet grasping. It used a known constant-velocity trajectory at 4 cm/s and 8 cm/s with delayed observations.

Under clean CV assumptions, matched `tau` reduced prediction error from approximately `speed x delay` to near zero. For example, at 8 cm/s with 100 ms delay, `tau=0` produced about 8 mm current-state error, while `tau=0.1` reduced the synthetic prediction error to near zero.

This validates the mathematical behavior of `predict_ahead(tau)` under clean constant-velocity assumptions.

Conclusion:

```text
TAU_SYNTHETIC_VALID
```

## Stage 4A.2: PyBullet Observation-Delay Tracking-Only

Stage 4A.2 removed gripper close, fixed constraints, grasp success, retry logic, and lift logic. The robot only tracked a moving-object-relative target generated through `DynamicAlignmentTracker.get_target_pose()`.

Observation-delay conditions compared delayed point-cloud content with both arrival timestamp and capture timestamp semantics. Matched `tau` did not improve tracking-only metrics. It increased target-to-object relative error in the tested PyBullet tracking cells, and `tau=0.1` also hurt when delay was 0 ms.

Conclusion:

```text
TAU_SYNTHETIC_ONLY_NOT_TRACKING
```

## Stage 4A.3: PyBullet Control-Delay Tracking-Only

Stage 4A.3 tested whether `tau` better matches command/control/actuation delay than pure observation delay. The tracker received current object observations with no artificial observation delay. The target generator computed current targets with `get_target_pose(..., tau)`, but the robot executed targets from a command buffer corresponding to `sim_t - control_delay`.

Matched `tau` did not improve EE-to-target tracking error under command delay. It worsened EE-to-target error for all tested 100 ms and 150 ms control-delay cells, and `tau=0.1` also hurt when delay was 0 ms.

Conclusion:

```text
STOP_TAU_PROCEED_TO_CONTACT_GATING
```

## Final Interpretation

`predict_ahead(tau)` is mathematically valid under clean constant-velocity assumptions. The Stage 4A.1 synthetic audit demonstrated the expected behavior directly.

However, the current PyBullet perception/control/replay setup does not show a `tau` benefit. The likely bottlenecks are point-cloud estimation noise, velocity-estimation instability, camera segmentation artifacts, robot servo dynamics, target buffering semantics, and interaction with replay timing. Grasp-level success is too coarse to rescue this result, and tracking-only diagnostics already failed to validate `tau` in PyBullet.

Therefore, `tau` should remain as a future real-system latency interface for measured perception + computation + actuation delay. It should not be presented as a current simulation method contribution, and it should not be tuned for PyBullet success.

## Updated Project Direction

- Use `dynamic_tau0` as the base method for the next improvement stage.
- Do not tune `tau` for success.
- Do not use CT as the main improvement path.
- Do not run a larger grasp-level latency grid unless PyBullet tracking-only behavior is fixed and revalidated.
- Proceed next, after approval, to contact-aware temporal gating targeting `attempt_limit` / no-contact failures.

This remains a simulation-only PyBullet conclusion. It does not validate real robot latency compensation, RealSense perception, or real contact behavior.
