# Stage 6 Phase Servo Summary

## Why Stage 6A Was Attempted

Stage 5 showed that small timing patches were not promising. `tau` was synthetic-valid but not useful in the current PyBullet tracking/grasp setup, CT did not improve over CV, contact gating harmed strong baseline cases, pre-close gating did not recover failures, and simple close-phase retiming did not reliably reduce no-contact / `attempt_limit`.

Stage 6A was therefore attempted as a larger architecture prototype: replace demo-clock replay patches with an explicit object-relative phase controller.

## What `dynamic_phase_servo` Changed

`dynamic_phase_servo` kept dynamic_tau0 object tracking as the base, but changed the replay architecture:

- It introduced explicit phases: `APPROACH`, `PREGRASP_SERVO`, `CLOSE`, `VERIFY_ATTACH`, `LIFT`, `DONE`, and `FAILED`.
- It used object-relative target poses from selected static-demo key poses.
- It made close event-triggered by EE-to-target and EE-to-object geometry.
- It logged phase durations, phase transitions, close trigger geometry, attach verification, and phase failure reasons.

It intentionally did not change physics, object trajectory, low-level controller, success criteria, retry logic, or fixed-constraint behavior.

## Smoke Test Results

Smoke test grid:

- conditions: `dynamic_tau0`, `dynamic_phase_servo`
- speeds: 2, 4, 8 cm/s
- trials: 2 per condition-speed

Results:

| Method | 2 cm/s | 4 cm/s | 8 cm/s |
|---|---:|---:|---:|
| `dynamic_tau0` | 2/2 | 2/2 | 1/2 |
| `dynamic_phase_servo` | 0/2 | 0/2 | 0/2 |

## Key Diagnostic

The close trigger did occur in every `dynamic_phase_servo` trial. Trigger geometry was plausible:

- mean close-trigger EE-to-object distance: about 25 mm
- mean close-trigger EE-to-target distance: about 13 mm

The failures happened after the trigger, in attach/lift behavior. Most phase-servo trials ended with no finite contact error / `attempt_limit`, and the phase diagnostics reported mostly `lift_failed` plus one `approach_failed`.

## Interpretation

The failure is not simply a threshold or close-trigger failure. The trigger fired at reasonable geometry, but the phase-servo prototype did not preserve the continuous contact/lift timing that the original demo replay implicitly provided.

The likely issue is phase design / target definition: switching from pregrasp servoing into close/lift using a few key object-relative poses was too crude for the existing fixed-constraint grasp setup.

## Conclusion

`dynamic_phase_servo` is not promising in its current form.

Do not run a formal comparison for this condition. The current strongest reliable method remains:

```text
dynamic_tau0
```

## Future Work

If continuing method development, redesign around demo-derived object-relative grasp primitives and explicit contact verification, not more phase/gate/timing patches. A useful future architecture would need a real closed-loop object-relative manipulation controller rather than a small wrapper around static-demo replay.
