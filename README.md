# MT3 Dynamic Alignment

Continuous object tracking and latency-compensated demo replay for applying a static MT3 manipulation demonstration to moving objects.

> **Status:** Core algorithm implemented and validated in PyBullet simulation. Hardware validation on Sawyer + RealSense D415 is pending.

## Demo

![Franka Panda grasping a moving object in PyBullet](simulation/results/plot/grasping_demo.gif)

Franka Panda replays a grasp demonstrated on a static object while the object moves on a tabletop. No retraining is used.

## Core Formula

```text
T_WE_target(t) = T_delta(t + tau) · T_WE_demo(t)
```

`T_WE_demo(t)` is the end-effector pose sequence recorded by MT3 on a static object. `T_delta(t + tau)` is the tracked object motion predicted ahead by the system latency `tau`, so the robot executes the demo in the moving object's frame instead of chasing stale observations.

## Key Results

### End-to-End Grasping

![Grasping Success Rate](simulation/results/plot/grasping_success_rate.png)

The final grasping experiment shows an inverted-U speed response: 4-6 cm/s is the best operating range, while very slow motion gives a weak velocity signal and fast motion reaches the controller bandwidth limit.

### Latency Compensation

![Baseline Comparison](simulation/results/plot/baseline_comparison.png)

Predicting 100 ms ahead reduces relative tracking error compared with pure reactive control.

### Robustness

![Robustness Analysis](simulation/results/plot/robustness_analysis.png)

Kalman filtering remains stable under noisy observations, especially when velocity feedforward is degraded by outliers.

### Method Comparison

![Method Comparison](simulation/results/plot/method_comparison_bar.png)

The Kalman + prediction method matches or exceeds the alternative controllers while providing the most reliable steady-state behavior.

<details>
<summary>Full Experiment Results</summary>

### Tracker Simulation

![XY Trajectory](simulation/results/plot/trajectory_xy.png)
![Position Error](simulation/results/plot/position_error.png)

The overhead RGB-D simulation tracks a box moving in a circle with millimeter-level steady-state error.

### Closed-Loop Panda Tracking

![Closed-loop 3D trajectories](simulation/results/plot/closed_loop_3d_trajectories.png)
![Closed-loop relative error](simulation/results/plot/closed_loop_relative_error.png)

The Panda end-effector maintains a constant relative pose to the moving object using tracker output.

### Static Demo to Moving Object

![MT3 Integration Error](simulation/results/plot/mt3_integration_error.png)
![MT3 Integration Trajectory](simulation/results/plot/mt3_integration_trajectory.png)

A trajectory demonstrated on a static object is replayed on a moving object in the object frame.

### Speed and Motion Sensitivity

![Speed Sensitivity](simulation/results/plot/speed_sensitivity.png)
![Motion Type Comparison](simulation/results/plot/motion_type_comparison.png)

The system remains effective across multiple object speeds and motion patterns, with degradation near reversal points and high-speed limits.

### Additional Test Figures

![Convergence](simulation/results/plot/convergence.png)
![Latency Compensation Test](simulation/results/plot/latency_compensation.png)
![Tracking Comparison](simulation/results/plot/tracking_comparison.png)
![Method Error Over Time](simulation/results/plot/method_comparison_time.png)

Raw final grasping data is stored at [`simulation/results/raw/10_grasping_final_seed42.csv`](simulation/results/raw/10_grasping_final_seed42.csv).

</details>

## How It Works

The tracker estimates how the object has moved since the static demonstration was recorded. A Kalman filter smooths this motion estimate and predicts slightly into the future to compensate for camera, computation, and actuation latency. The robot then transforms every demo pose by the predicted object motion, so the same recorded action follows the moving object. For grasping, adaptive replay can slow the demo when alignment is poor and continue once the arm is back on track.

## Module Structure

```text
MT3_dynamic_alignment/
├── dynamic_alignment/          Core implementation
│   ├── types.py                Shared data structures
│   ├── motion_models.py        Constant-velocity and coordinated-turn models
│   ├── kalman.py               Kalman / EKF filter
│   ├── pose_estimator.py       Point cloud to object motion observation
│   └── tracker.py              Main tracking and target-pose interface
│
├── examples/
│   └── simulate_and_plot.py    Minimal synthetic simulation example
│
├── simulation/                 PyBullet experiments
│   ├── 02_tracker_sim.py
│   ├── 03_closed_loop.py
│   ├── 04_baseline_comparison.py
│   ├── 05_mt3_integration.py
│   ├── 06_speed_sensitivity.py
│   ├── 07_motion_type_comparison.py
│   ├── 08_method_comparison.py
│   ├── 09_robustness_analysis.py
│   ├── 10_grasping_experiment.py
│   └── results/
│       ├── plot/               Figures and demo GIFs
│       └── raw/                Raw experiment CSV files
│
├── tests/                      Hardware-free unit tests
├── MT3_dynamic_alignment_notes.md
└── 2305.05926v1.pdf            MT3 paper
```

## Quick Start

Create the environment:

```bash
conda create -n dynamic_mt3 python=3.11 numpy matplotlib pytest -y
conda activate dynamic_mt3
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run the minimal simulation example:

```bash
python examples/simulate_and_plot.py
```

PyBullet experiments require `pybullet`, `pillow`, and `imageio` in the same environment.

## Integration with MT3

Real deployment requires replacing only two perception stubs and swapping the target-pose source in the controller.

```python
# 1. Replace hardware stubs in pose_estimator.py
PoseEstimator.get_point_cloud_from_realsense()   # RealSense SDK
PoseEstimator.segment_object_by_bbox()           # MT3 point-cloud segmentation

# 2. Initialize once after MT3/GICP alignment completes
tracker.init(current_cloud, initial_theta=gicp_theta, timestamp=t0)

# 3. Use dynamic target poses in the control loop
state = tracker.update(current_cloud, timestamp=t)
T_target = tracker.get_target_pose(demo_data, t_demo=phase_time, tau=0.1)
```

The rest of the MT3 demo data format can remain unchanged: `DemoData` stores the original end-effector pose sequence and timestamps.

## Design Notes

See [`MT3_dynamic_alignment_notes.md`](MT3_dynamic_alignment_notes.md) for the derivation, latency compensation analysis, motion-model assumptions, and two-phase replay design.

## Acknowledgments

This project extends the ideas in MT3 (*Multi-Task Trajectory Transfer*, Science Robotics 2025) from static one-time alignment to continuous object-relative replay.
