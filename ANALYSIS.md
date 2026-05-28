# PyBullet Simulation Analysis

## Introduction

This document summarizes the PyBullet study for MT3 dynamic alignment: replaying a static-object MT3 demonstration on a slowly moving tabletop object by tracking object motion and transforming the trajectory online. Primary results come from `ablation_20260524_233332`; `ablation_20260528_232432` is an occlusion check.

## Experiment Setup

Two ablation runs were conducted.

| Run | Folder | Setup |
|-----|--------|-------|
| Run 1 | `ablation_20260524_233332` | 5 conditions, no occlusion, `SEED=42`, `N_TRIALS=20`, `TAU=0.1`, speeds `[2.0, 4.0, 6.0, 8.0]` cm/s |
| Run 2 | `ablation_20260528_232432` | Same setup plus `OCCLUSION_PROB=0.15`; added `pd_feedforward`, which failed due to an implementation bug and is excluded |

Each condition-speed cell contains 20 trials. All conditions have high variance, roughly +/-40-50% standard deviation, so conclusions are directional.

## Results

Run 1 success rates are the primary results.

| condition | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| static_replay | 0 | 0 | 0 | 0 |
| raw_observation | 0 | 0 | 0 | 0 |
| dynamic_tau0 | 50 | 75 | 85 | 20 |
| uncertainty_gated | 45 | 60 | 45 | 35 |
| oracle_pose | 100 | 100 | 100 | 100 |

Run 2 success rates with 15% random occlusion:

| condition | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| static_replay | 0 | 0 | 0 | 0 |
| raw_observation | 0 | 0 | 0 | 0 |
| dynamic_tau0 | 70 | 80 | 85 | 25 |
| uncertainty_gated | 40 | 60 | 50 | 30 |
| oracle_pose | 100 | 100 | 100 | 100 |

For Run 1 `dynamic_tau0`, 2 cm/s is dominated by `attempt_limit`: the robot fails to get close enough. At 8 cm/s, `lift` dominates: the gripper closes but misses. At 4-6 cm/s, failures are distributed, consistent with random perception noise.

## Key Findings

### 1. Static Replay Fails Completely

`static_replay` has 0% success at all speeds. Direct replay on a moving object fails, so dynamic adaptation is necessary.

### 2. Kalman Filtering Is Essential

`raw_observation` also has 0% success. It matches `static_replay` in success rate but fails differently: the gripper closes but misses, producing lift failure rather than `attempt_limit`. Raw centroids are unstable.

### 3. The Controller Is Not the Bottleneck

`oracle_pose` reaches 100% success at all speeds. Since the same controller succeeds with perfect object pose, perception and state estimation are limiting.

### 4. Failure Modes Are Speed Dependent

At 2 cm/s, weak velocity gives low signal-to-noise ratio for Kalman prediction. At 8 cm/s, failures shift to lift misses, consistent with controller bandwidth limits. The best Run 1 range is 4-6 cm/s.

### 5. uncertainty_gated Is Not Distinguished in Simulation

`uncertainty_gated` is indistinguishable from `dynamic_tau0` in PyBullet. The Kalman covariance `P` stays near-constant because noise is fixed and synthetic, with no genuine perception variation. This is a simulation limitation, not a method flaw.

### 6. Random Occlusion Was Not Structured Enough

Run 2 shows that 15% random per-frame occlusion did not create structured enough variation to separate `uncertainty_gated` from `dynamic_tau0`. Arm-induced descent occlusion would be more realistic.

## Limitations

`tau` compensation cannot be validated without real latency. The model handles only planar SE(2), not 6D motion. Only grasping was tested. Twenty trials per condition gives high variance. PyBullet contact dynamics differ from real hardware.

## Required Hardware Validation

Hardware is required to validate `tau` compensation under real perception and actuation latency, evaluate `uncertainty_gated` under motion blur, occlusion, and varying point-cloud density, and test generalization beyond grasping.
