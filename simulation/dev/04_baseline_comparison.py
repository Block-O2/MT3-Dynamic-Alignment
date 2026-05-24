"""
Compare closed-loop tracking with and without latency compensation.

This script reuses the scene and helpers from simulation/03_closed_loop.py:
- Franka Panda arm
- moving box on the front-workspace semicircle
- overhead RGB-D camera point-cloud tracking

It runs two conditions back-to-back:
A. tau = 0.0, pure reactive tracking
B. tau = 0.1, latency-compensated tracking

The output plot is saved to:
simulation/results/baseline_comparison.png
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynamic_mt3_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dynamic_mt3_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_alignment.pose_estimator import EstimatorConfig
from dynamic_alignment.tracker import DynamicAlignmentTracker
from dynamic_alignment.types import make_static_demo


def load_closed_loop_module():
    """Load simulation/03_closed_loop.py despite its digit-prefixed filename."""
    module_path = Path(__file__).resolve().with_name("03_closed_loop.py")
    spec = importlib.util.spec_from_file_location("closed_loop_sim", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sim03 = load_closed_loop_module()


def setup_scene() -> tuple[int, int, tuple[list[float], list[float]], tuple, object]:
    """Create the PyBullet scene and return handles needed by the loop."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.8)
    p.setTimeStep(sim03.DT)
    p.loadURDF("plane.urdf")
    sim03.create_table()
    box_id = sim03.create_box()
    panda_id = sim03.load_panda()

    camera_matrices = sim03.make_camera_matrices()
    target_orientation = sim03.DOWNWARD_EE_ORIENTATION
    ik_params = sim03.get_joint_limits(panda_id)

    initial_box_pos = np.array(sim03.box_position(0.0), dtype=float)
    initial_ee_target = initial_box_pos + sim03.DESIRED_EE_OFFSET
    demo_data = make_static_demo(
        sim03.transform_from_position_quaternion(initial_ee_target, target_orientation),
        timestamp=0.0,
    )

    for _ in range(90):
        sim03.command_ee_pose(panda_id, initial_ee_target, target_orientation, ik_params)
        p.stepSimulation()

    return box_id, panda_id, camera_matrices, ik_params, demo_data


def run_condition(tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Run one closed-loop condition and return timestamps and relative errors."""
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet DIRECT")

    timestamps: list[float] = []
    relative_errors: list[float] = []

    try:
        box_id, panda_id, camera_matrices, ik_params, demo_data = setup_scene()
        view_matrix, projection_matrix = camera_matrices
        target_orientation = sim03.DOWNWARD_EE_ORIENTATION

        estimator_config = EstimatorConfig(
            min_points=20,
            use_pca_angle=False,
            z_plane_threshold=0.03,
        )
        tracker = DynamicAlignmentTracker(
            tau=tau,
            estimator_config=estimator_config,
            kalman_R_diag=np.array([0.004, 0.004, math.radians(30.0)]),
            init_vel_cov=0.10,
        )

        n_steps = int(sim03.DURATION * sim03.FPS)
        for frame_idx in range(n_steps + 1):
            sim_t = frame_idx * sim03.DT
            box_pos = np.array(sim03.box_position(sim_t), dtype=float)
            p.resetBasePositionAndOrientation(
                box_id,
                box_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
            )

            cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
            if not tracker.is_initialized:
                if cloud.shape[0] < estimator_config.min_points:
                    raise RuntimeError(
                        "Initial depth segmentation produced too few box points: "
                        f"{cloud.shape[0]} < {estimator_config.min_points}"
                    )
                tracker.init(cloud, initial_theta=0.0, timestamp=sim_t)
                target_pose = demo_data.T0
            else:
                tracker.update(cloud, timestamp=sim_t)
                target_pose = tracker.get_target_pose(demo_data, t_demo=0.0, tau=tau)

            sim03.command_ee_pose(
                panda_id,
                np.asarray(target_pose[:3, 3], dtype=float),
                target_orientation,
                ik_params,
            )
            p.stepSimulation()

            ee_pos = sim03.get_ee_position(panda_id)
            relative_error = float(
                np.linalg.norm((ee_pos - box_pos) - sim03.DESIRED_EE_OFFSET)
            )
            timestamps.append(sim_t)
            relative_errors.append(relative_error)

    finally:
        p.disconnect()

    return np.asarray(timestamps, dtype=float), np.asarray(relative_errors, dtype=float)


def plot_comparison(
    times: np.ndarray,
    errors_no_prediction: np.ndarray,
    errors_with_prediction: np.ndarray,
    out_path: Path,
) -> None:
    """Plot both relative error curves on shared axes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    err0_mm = errors_no_prediction * 1000.0
    err1_mm = errors_with_prediction * 1000.0
    mean0 = float(np.mean(err0_mm))
    mean1 = float(np.mean(err1_mm))

    plt.figure(figsize=(9, 4.8))
    plt.plot(times, err0_mm, color="tab:blue", linewidth=2,
             label=f"No prediction (τ=0), mean={mean0:.1f} mm")
    plt.plot(times, err1_mm, color="tab:orange", linewidth=2,
             label=f"With prediction (τ=0.1s), mean={mean1:.1f} mm")
    plt.fill_between(times, err0_mm, err1_mm, color="0.5", alpha=0.18)
    plt.xlabel("time (s)")
    plt.ylabel("relative position error (mm)")
    plt.title("Effect of Latency Compensation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    times0, errors0 = run_condition(tau=0.0)
    times1, errors1 = run_condition(tau=0.1)

    if times0.shape != times1.shape or not np.allclose(times0, times1):
        raise RuntimeError("Condition timestamps do not match")

    out_path = PROJECT_ROOT / "simulation" / "results" / "baseline_comparison.png"
    plot_comparison(times0, errors0, errors1, out_path)
    print("Saved simulation/results/baseline_comparison.png")


if __name__ == "__main__":
    main()
