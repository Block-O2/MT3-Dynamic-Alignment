# MT3 Dynamic Alignment

Continuous object tracking and latency-compensated demo replay for applying a static MT3 manipulation demonstration to moving objects.

> **Status:** PyBullet simulation study complete. Five-condition ablation (static_replay, raw_observation, dynamic_tau0, uncertainty_gated, oracle_pose) validated across 4 speeds × 20 trials. Hardware validation on Sawyer + RealSense D415 is the identified next step.

## Demo

![Franka Panda grasping a moving object in PyBullet](simulation/results/plot/grasping_demo.gif)

Franka Panda replays a grasp demonstrated on a static object while the object moves on a tabletop. No retraining is used.

## Core Formula

```text
T_WE_target(t) = T_delta(t + tau) · T_WE_demo(t)
```

`T_WE_demo(t)` is the end-effector pose sequence recorded by MT3 on a static object. `T_delta(t + tau)` is the tracked object motion predicted ahead by the system latency `tau`, so the robot executes the demo in the moving object's frame instead of chasing stale observations.

## Key Results

![Grasping Success Rate](simulation/results/plot/grasping_success_rate.png)

The current PyBullet ablation uses five conditions across four object speeds and 20 trials per speed. `static_replay` achieves 0% success at all speeds, confirming that replaying a static-object demonstration directly on a moving object fails. `dynamic_tau0` reaches 70-85% success at 2-6 cm/s and drops to 25% at 8 cm/s. `oracle_pose` achieves 100% success at all speeds, showing that the controller is not the bottleneck. `raw_observation` achieves 0%, indicating that Kalman filtering is essential. `uncertainty_gated` is comparable to `dynamic_tau0` in simulation; the covariance matrix remains near-constant in PyBullet, so real hardware is needed to distinguish the two.

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
│   ├── dev/                    Development and diagnostic scripts, 02-09_*.py
│   ├── 10_grasping_experiment.py
│   └── results/
│       ├── canonical/          Timestamped ablation runs, each containing command.txt, config.json, raw_trials.csv, and plots
│       └── plot/               Figures and demo GIFs
│
├── tests/                      Hardware-free unit tests
├── MT3_dynamic_alignment_notes.md
└── 2511.10110v1.pdf            MT3 paper
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

This project studies the extension of MT3 (Multi-Task Trajectory Transfer, Science Robotics 2025) demo replay from static to slowly moving tabletop objects, using a single demonstration with no retraining.
