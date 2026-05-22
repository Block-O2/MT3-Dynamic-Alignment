# Experiment Log

## Overview

| Experiment | What it tests | Key finding | Script |
|------------|---------------|-------------|--------|
| 02 | Single-object tracking from PyBullet depth images | Overhead RGB-D tracking reaches millimeter-level steady-state error | `simulation/02_tracker_sim.py` |
| 03 | Closed-loop Panda tracking of a moving object | The end-effector can maintain a constant relative pose to the tracked object | `simulation/03_closed_loop.py` |
| 04 | Latency compensation baseline | `tau=0.1s` prediction reduces relative error compared with pure reactive control | `simulation/04_baseline_comparison.py` |
| 05 | Static demo replay on a moving object | A static-object demo can be replayed in the moving object's frame without retraining | `simulation/05_mt3_integration.py` |
| 06 | Speed sensitivity | Performance degrades gradually near the designed speed limit | `simulation/06_speed_sensitivity.py` |
| 07 | Motion type sensitivity | Circular and random smooth motions are easier than sharp linear reversals | `simulation/07_motion_type_comparison.py` |
| 08 | Method comparison | Kalman + prediction gives the most reliable steady-state behavior under latency and noise | `simulation/08_method_comparison.py` |
| 09 | Robustness analysis | Kalman filtering rejects outlier-driven degradation better than velocity feedforward | `simulation/09_robustness_analysis.py` |
| 10 | End-to-end moving-object grasping | Final strict grasping result shows an inverted-U speed response with best performance at 4-6 cm/s | `simulation/10_grasping_experiment.py` |

## Experiment Evolution: Grasping (`10_grasping_experiment`)

### v1: Position Control, Simple Lift Criterion

Configuration changes made:
- Used PyBullet IK with `POSITION_CONTROL`.
- Success was defined mainly as whether the box was lifted above the table.
- The gripper attachment was deterministic once close enough to the object.

Results obtained:
- Moving-object grasping reported near 100% success.

Why it failed / what was learned:
- The criterion was too lenient. A grasp could be counted as successful even if the end-effector was not accurately following the demonstrated grasp pose.
- Lift-only success did not reveal lateral tracking errors, orientation mismatch, or last-moment corrections.

What changed next:
- Added strict geometric criteria: position error < 15 mm, orientation error < 20 degrees, and approach consistency within 20 mm during the final 0.5 s before contact.

### v2: Strict Criteria + Binary Adaptive Gate

Configuration changes made:
- Added strict success checks: 15 mm position, 20 degree orientation, 20 mm approach consistency.
- Added a binary adaptive gate: if lateral error exceeded the threshold, demo time froze; otherwise it advanced normally.

Results obtained:
- Success dropped to 0%.

Why it failed / what was learned:
- The binary gate was too brittle. Noise or transient lateral error could freeze the demo for long periods.
- Once frozen, the grasp phase often never progressed far enough to close and lift correctly.

What changed next:
- Replaced binary freeze/advance with a continuous progress rate based on lateral error.

### v3: Continuous Adaptive Gate

Configuration changes made:
- Replaced binary gating with:

```text
progress_rate = clip(1 - (lateral_error_mm - threshold_mm) / 20, 0, 1)
```

- Demo time advanced by `dt * progress_rate`.
- Added per-trial progress-rate reporting.

Results obtained:
- Moving-object success improved to roughly 30-40%.

Why it failed / what was learned:
- The gate itself was more stable, but the success metric was still partly wrong.
- Position and approach checks were comparing the end-effector to the box center, while the demonstrated grasp target is intentionally offset from the box center.

What changed next:
- Fixed success metrics to compare the end-effector against the demo target in the object frame, not the box center.

### v4: Correct Metric + Cartesian Velocity Control

Configuration changes made:
- Position success became `||EE_xy - demo_target_xy|| < 15 mm`.
- Approach consistency used the same demo-target metric.
- Replaced PyBullet IK position control with Cartesian velocity control through the Jacobian.
- Tracking-floor diagnostic was changed to measure error to `T_delta · T_WE_demo(0)`, the actual intended grasp approach point.

Results obtained:
- Tracking floor improved from roughly 20 mm under IK position control to roughly 2-5 mm under Cartesian velocity control at 2-6 cm/s.
- Grasping success improved, but approach and contact-window failures remained.

Why it failed / what was learned:
- Direct Cartesian velocity control reduced steady-state tracking error.
- Remaining failures came from coupling horizontal object tracking with vertical demo descent in a single Cartesian error vector.

What changed next:
- Decoupled horizontal tracking from vertical descent using a prioritized Jacobian solve.

### v5: Decoupled Jacobian Horizontal/Vertical Control

Configuration changes made:
- Solved horizontal XY tracking first with higher gain.
- Projected vertical descent into the null space of the horizontal solution.
- Added a pre-close check: if the box was more than 40 mm from the end-effector before gripper closing, abort the attempt and restart the demo.
- Added a 5-frame moving average for lateral error before passing it to the adaptive gate.
- Ran final canonical experiment at speeds `[2, 4, 6, 8]` cm/s, 10 trials each, seed 42.

Results obtained:
- Final moving-object success rates:

| Speed | Success rate |
|-------|--------------|
| 2 cm/s | 30% |
| 4 cm/s | 80% |
| 6 cm/s | 80% |
| 8 cm/s | 30% |

Why it failed / what was learned:
- The final curve is an inverted-U.
- At 2 cm/s, motion is too slow to provide a strong velocity signal for Kalman prediction and adaptive replay.
- At 8 cm/s, the tracking mean can still be acceptable, but variance and controller bandwidth limits cause contact-window failures.

What changed next:
- This became the final canonical grasping configuration for the current simulation study.

## Key Findings

- Optimal speed range: 4-6 cm/s.
- 2 cm/s failure mode: weak Kalman velocity signal.
- 8 cm/s failure mode: control bandwidth limit and high tracking variance.
- Type B failure: the gripper closes in empty space because the object has moved away before contact.
- Velocity control vs position control: tracking floor improved from roughly 20 mm to 2-5 mm.

## Raw Data

All final experiment data is stored in:

```text
simulation/results/raw/
```

Final canonical grasping run:

```text
simulation/results/raw/10_grasping_final_seed42.csv
```

Seed: `42` for reproducibility.

## Notes on Simulation Limitations

- PyBullet position-control IK produced roughly 10 mm or larger tracking error, motivating the switch to Cartesian velocity control.
- The simulation does not model real motor inertia and controller bandwidth in full detail, so real hardware is expected to be harder.
- Grasping success is defined by strict geometric criteria rather than physical contact force or tactile feedback.
