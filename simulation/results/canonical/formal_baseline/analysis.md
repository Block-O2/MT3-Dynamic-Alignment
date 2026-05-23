# formal_baseline

Stage 3 baseline smoke test. The run keeps the same robot, object, trajectory, controller, thresholds, physics, and success criteria as the existing grasping experiment.
tau is treated as the intended measured system-delay interface parameter, not tuned for success.

- status: succeeded
- runtime_s: 1001.809
- speeds_cm_s: [2.0, 4.0, 6.0, 8.0]
- trials: 10
- requested_conditions: ['static_replay', 'dynamic_tau0', 'dynamic_cv', 'dynamic_ct']
- executed_conditions: ['dynamic_ct', 'dynamic_cv', 'dynamic_tau0', 'static_replay']
- deferred_conditions: []

Important limitation: grasp attachment remains the existing deterministic fixed PyBullet constraint when the gripper closes near the box. This is not real contact or hardware validation.

## Primary Metrics

- success_rate
- n_no_contact
- n_finite_contact_error
- main_failure_counts
- mean_lift_mm
- mean_progress_pct
- mean_contact_position_error_finite_only, interpreted with n_finite_contact_error
- mean_ee_to_target_error_contact_window_finite_only
- mean_contact_window_ee_to_object_xy_finite_only

## Secondary Diagnostics

- orientation and approach finite-only errors
- raw per-trial EE-to-target and target-to-object fields
- diagnostics/*.csv frame-level traces when --record-diagnostics is enabled
- object-estimation and phase-mixed pre-contact tracking metrics in raw_results.csv only; these are not ranking metrics

## Known Simulation Artifacts

- Deterministic fixed PyBullet constraint is used for grasp attachment once the gripper closes near the object.
- This is not real contact validation.
- Contact, orientation, and approach error means in summary.csv are finite-only means. n_no_contact and n_finite_contact_error must be read with those means.
- object_estimation_error_theta_deg is not emitted as a meaningful value in this stage because the current box trajectory and estimator configuration do not provide a useful yaw diagnostic.

Results:
- dynamic_ct, 2.0 cm/s: n=10, success_rate=0.300, failures={'attempt_limit': 7, 'none': 3}
- dynamic_ct, 4.0 cm/s: n=10, success_rate=0.800, failures={'none': 8, 'attempt_limit': 1, 'orientation': 1}
- dynamic_ct, 6.0 cm/s: n=10, success_rate=0.800, failures={'none': 8, 'orientation': 1, 'attempt_limit': 1}
- dynamic_ct, 8.0 cm/s: n=10, success_rate=0.300, failures={'attempt_limit': 5, 'approach': 1, 'none': 3, 'orientation': 1}
- dynamic_cv, 2.0 cm/s: n=10, success_rate=0.300, failures={'attempt_limit': 7, 'none': 3}
- dynamic_cv, 4.0 cm/s: n=10, success_rate=0.800, failures={'none': 8, 'attempt_limit': 1, 'orientation': 1}
- dynamic_cv, 6.0 cm/s: n=10, success_rate=0.800, failures={'none': 8, 'orientation': 1, 'attempt_limit': 1}
- dynamic_cv, 8.0 cm/s: n=10, success_rate=0.300, failures={'attempt_limit': 5, 'approach': 1, 'none': 3, 'orientation': 1}
- dynamic_tau0, 2.0 cm/s: n=10, success_rate=0.600, failures={'none': 6, 'attempt_limit': 4}
- dynamic_tau0, 4.0 cm/s: n=10, success_rate=0.900, failures={'none': 9, 'attempt_limit': 1}
- dynamic_tau0, 6.0 cm/s: n=10, success_rate=0.900, failures={'none': 9, 'attempt_limit': 1}
- dynamic_tau0, 8.0 cm/s: n=10, success_rate=0.200, failures={'none': 2, 'attempt_limit': 8}
- static_replay, 2.0 cm/s: n=10, success_rate=0.000, failures={'attempt_limit': 10}
- static_replay, 4.0 cm/s: n=10, success_rate=0.000, failures={'attempt_limit': 10}
- static_replay, 6.0 cm/s: n=10, success_rate=0.000, failures={'attempt_limit': 10}
- static_replay, 8.0 cm/s: n=10, success_rate=0.000, failures={'attempt_limit': 10}

## Formal Baseline Questions

1. Does static_replay consistently fail as a moving-object baseline? Yes; overall success_rate=0.000.
2. Does dynamic replay outperform static replay? Yes; dynamic mean success_rate=0.583.
3. Does dynamic_cv outperform dynamic_tau0? No; dynamic_cv=0.550, dynamic_tau0=0.650.
4. Does dynamic_ct meaningfully differ from dynamic_cv? No; absolute success-rate difference=0.000.
5. At which speeds does each method fail most often? dynamic_ct: 2.0 cm/s (0.300); dynamic_cv: 2.0 cm/s (0.300); dynamic_tau0: 8.0 cm/s (0.200); static_replay: 2.0 cm/s (0.000).
6. What are the dominant failure modes? {'attempt_limit': 82, 'orientation': 6, 'approach': 2}.
7. Are no-contact failures reduced by dynamic methods? static_replay no-contact=40; dynamic no-contact={'dynamic_tau0': 14, 'dynamic_cv': 14, 'dynamic_ct': 14}.
8. Do contact-window EE-to-target and EE-to-object metrics align with success? Success finite contact-window EE-target mean=7.277 mm; failure finite mean=8.364 mm. Interpret only with finite contact counts.
9. Are any results suspicious or likely artifacts of fixed-constraint grasping? No obvious large-contact-error successes; fixed-constraint attachment remains a known artifact.
10. Based on this formal baseline, is it justified to proceed to Stage 4 stress tests or Stage 5 method improvement? Recommendation: PROCEED_TO_METHOD_IMPROVEMENT.

PROCEED_TO_METHOD_IMPROVEMENT
