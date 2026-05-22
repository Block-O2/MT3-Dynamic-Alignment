# Stage 5 Method Exploration Summary

## Starting Point

The formal PyBullet baseline showed that `static_replay` failed across all tested moving-object speeds, while dynamic object-frame replay clearly outperformed static replay. Among the tested baselines, `dynamic_tau0` was the strongest reliable method.

The dominant remaining failures were `attempt_limit` / no-contact failures. These are still simulation-only failures under the current deterministic fixed-constraint grasp simplification, not evidence of real contact behavior.

## Tau Line Summary

The latency-compensation investigation found a split result:

- Synthetic constant-velocity delay validation passed: matched `tau` reduced prediction error as expected under clean CV assumptions.
- PyBullet observation-delay tracking-only validation failed to show a benefit.
- PyBullet control-delay tracking-only validation also failed to show a benefit.

Conclusion: `tau` should remain as a future real-system interface for measured perception + computation + actuation delay. It is not validated as a current PyBullet method contribution, and it should not be tuned for simulation success.

## Contact Gate Summary

`dynamic_tau0_contact_gated` attempted a freeze-style pre-close gate. It did not reduce no-contact / `attempt_limit` failures and damaged the 4 cm/s case, where the ungated `dynamic_tau0` baseline was strong.

Conclusion: freeze-style contact gating is not promising in this setup.

## Pre-Close Gate Summary

`dynamic_tau0_preclose_gated` changed the gate into a pre-close alignment state. It avoided some timeout behavior from the first gate, but it still did not preserve baseline performance and did not reliably reduce no-contact / `attempt_limit` failures.

Conclusion: pre-close freezing is not promising.

## Close-Retiming Summary

Stage 5B failure anatomy showed that failed `dynamic_tau0` trials were usually not caused by large EE-to-target error. The EE often tracked the generated target, but EE-to-object distance during close/attach approached or exceeded the retry threshold. That motivated an offline close-retiming feasibility check.

Offline oracle feasibility found candidate close opportunities in failed 6 cm/s and 8 cm/s trials. However, the online `dynamic_tau0_close_retimed` smoke test did not preserve the strong 4 cm/s baseline and did not reliably reduce no-contact / `attempt_limit` failures overall.

Conclusion: simple close-phase retiming is not promising.

## Final Method Conclusion

The current strongest reliable method remains:

```text
dynamic_tau0
```

Do not claim `tau`, CT, freeze-style contact gating, pre-close gating, or close retiming as successful improvements. The evidence from these PyBullet smoke tests is negative or mixed, and the fixed-constraint grasp simplification remains a major limitation.

Future improvement likely requires a more fundamental closed-loop manipulation controller, not small timing patches around the static demo replay.

## Recommended Next Project Direction

Stop incremental PyBullet timing tweaks for now. The useful next project step is one of:

- write a simulation study / negative-results report around the formal baseline and failed improvement attempts; or
- redesign the method around a full phase-aware controller with explicit object-relative servoing and contact verification.

Any future method should still keep the simulation-only positioning clear and should not claim real robot validation without hardware experiments.
