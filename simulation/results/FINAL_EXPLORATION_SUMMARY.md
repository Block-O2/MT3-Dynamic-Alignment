# Final Exploration Summary

## 1. Project Status

This repository is a simulation-only PyBullet robotics prototype. It does not validate real robot behavior, RealSense perception, real contact dynamics, or robust deployment.

The current experiments use a deterministic fixed-constraint grasp attachment once the simulated gripper closes near the object. This is a useful PyBullet simplification for controlled comparison, but it is not real contact validation.

The honest supported claim is limited to controlled PyBullet tabletop simulation:

> Dynamic object-frame replay can be compared against static replay and related baselines for slowly moving planar objects under fixed simulation assumptions.

## 2. Formal Baseline Conclusion

The canonical formal baseline is stored in:

```text
simulation/results/canonical/formal_baseline/
```

Formal baseline conditions:

- `static_replay`
- `dynamic_tau0`
- `dynamic_cv`
- `dynamic_ct`

Speeds:

- 2.0, 4.0, 6.0, 8.0 cm/s

Trials:

- 10 per condition per speed

Key results:

| condition | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| `static_replay` | 0.000 | 0.000 | 0.000 | 0.000 |
| `dynamic_tau0` | 0.600 | 0.900 | 0.900 | 0.200 |
| `dynamic_cv` | 0.300 | 0.800 | 0.800 | 0.300 |
| `dynamic_ct` | 0.300 | 0.800 | 0.800 | 0.300 |

Conclusions:

- `static_replay` failed at all speeds.
- Dynamic object-frame replay clearly outperformed static replay.
- `dynamic_tau0` was the strongest reliable baseline in this latency-free PyBullet setup.
- `dynamic_cv` with `tau=0.1` did not outperform `dynamic_tau0`.
- `dynamic_ct` did not differ meaningfully from `dynamic_cv`.
- Dominant failures were `attempt_limit` / no-contact.

## 3. Tau / Latency Exploration

The tau line was tested because tau is intended to represent measured real-system latency: perception, computation, command transmission, and actuation.

Findings:

- Synthetic constant-velocity delay validation passed. Matched tau reduced prediction error as expected under clean CV assumptions.
- PyBullet observation-delay tracking did not validate tau.
- PyBullet control-delay tracking did not validate tau.
- Grasp-level artificial observation delay did not show success improvement with matched tau.

Conclusion:

`predict_ahead(tau)` is mathematically valid under clean CV assumptions, but tau is not validated as a current PyBullet method contribution. It should remain a future real-system latency interface, not a tuned simulation improvement.

## 4. Contact / Gating Exploration

Several timing patches were tested to address `attempt_limit` / no-contact failures.

Findings:

- Freeze-style contact gate was not promising.
- Pre-close gate was not promising.
- Close retiming was not promising.

Interpretation:

Simple timing patches tended to disrupt demo replay rather than improve it. They did not reliably reduce no-contact failures and could damage speeds where `dynamic_tau0` already worked well.

## 5. Phase-Servo Exploration

`dynamic_phase_servo` was tested as a larger architecture prototype with explicit phases:

- approach
- pregrasp servo
- close
- verify attach
- lift

Smoke result:

- `dynamic_phase_servo` failed at all tested speeds.
- Close triggers occurred at plausible EE-object and EE-target distances.
- Failures happened after trigger, primarily in attach/lift behavior.

Conclusion:

The simple phase controller likely broke the demo's implicit continuous contact/lift timing. It should not proceed to formal comparison.

## 6. Feasibility-Aware Replay

Feasibility-aware replay was introduced to stay aligned with MT3-style general replay:

```text
T_target(t) = T_delta(t) @ T_demo(s(t))
```

The only controlled variable was `s_dot(t)`, the demo phase progression rate. The gripper command remained tied to demo phase `s(t)`. No grasp-specific close trigger or manual phase annotation was added.

Stage 7B v1:

- `dynamic_tau0`: 2 cm/s 2/2, 4 cm/s 2/2, 8 cm/s 1/2
- `feasibility_aware_replay`: 2 cm/s 1/2, 4 cm/s 1/2, 8 cm/s 2/2

Diagnosis:

- v1 improved 8 cm/s in the smoke test.
- v1 hurt 2/4 cm/s.
- Failures were associated with excessive freezing and incomplete demo progress.

Stage 7B.1 v2:

- `feasibility_aware_replay_v2`: 2 cm/s 1/2, 4 cm/s 1/2, 8 cm/s 2/2

Diagnosis:

- v2 reduced freeze frames slightly.
- v2 did not recover the 2/4 cm/s `dynamic_tau0` baseline.
- Final demo progress barely improved.

Conclusion:

Simple heuristic control of `s_dot(t)` is not reliable enough. It can help high-speed cases, but it destabilizes originally strong low/mid-speed cases and may disrupt gripper/contact/lift timing embedded in the original demonstration.

## 7. Final Current Conclusion

Current strongest reliable method:

```text
dynamic_tau0
```

Do not claim the following as successful improvements:

- tau / latency compensation
- CT prediction
- freeze-style contact gating
- pre-close gating
- close retiming
- phase servo
- feasibility-aware replay
- feasibility-aware replay v2

If continuing method development, future work needs a principled method rather than more threshold patches. A plausible direction would be a new branch with a receding-horizon phase optimizer or MPC-style formulation that explicitly optimizes tracking error, target velocity feasibility, phase progress, smoothness, and task completion.

For this branch, the clean scientific position is to present a rigorous simulation baseline and negative-results exploration, with `dynamic_tau0` as the strongest current baseline.
