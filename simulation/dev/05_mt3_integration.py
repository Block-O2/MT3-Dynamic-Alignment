"""
MT3 integration experiment: replay a static-object demo on a moving object.

Demo phase:
- The box is static at the MT3 reference pose.
- Franka Panda records a simple approach trajectory for 5 seconds.

Replay phase:
- The same box moves along the front-workspace semicircle from 03_closed_loop.py.
- DynamicAlignmentTracker estimates T_delta from the static reference cloud.
- The recorded demo is replayed as T_WE_target(t) = T_delta(t + tau) @ T_WE_demo(t).

Outputs:
- simulation/results/mt3_integration_error.png
- simulation/results/mt3_integration_trajectory.png
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
from dynamic_alignment.types import DemoData


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

DEMO_DURATION = 5.0
REPLAY_DURATION = 15.0
TAU = 0.1
STATIC_BOX_POS = np.array(
    [0.5, 0.0, sim03.TABLE_TOP_Z + sim03.BOX_HALF_EXTENT],
    dtype=float,
)
# Keep a small lateral offset so the overhead RGB-D camera can still see the box
# during the approach; the demonstrated motion is the vertical descent component.
DEMO_START_OFFSET = np.array([-0.12, 0.0, 0.20], dtype=float)
DEMO_END_OFFSET = np.array([-0.12, 0.0, 0.02], dtype=float)
SUCCESS_THRESHOLD_M = 0.030
CONTROL_SUBSTEPS = 8


def setup_scene() -> tuple[int, int, tuple[list[float], list[float]], tuple]:
    """Create the static scene shared by demo and replay phases."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.8)
    p.setTimeStep(sim03.DT / CONTROL_SUBSTEPS)
    p.loadURDF("plane.urdf")
    sim03.create_table()
    box_id = sim03.create_box()
    panda_id = sim03.load_panda()
    p.resetBasePositionAndOrientation(box_id, STATIC_BOX_POS.tolist(), [0.0, 0.0, 0.0, 1.0])

    camera_matrices = sim03.make_camera_matrices()
    ik_params = sim03.get_joint_limits(panda_id)
    return box_id, panda_id, camera_matrices, ik_params


def make_pose(position: np.ndarray) -> np.ndarray:
    """Build a downward-facing end-effector pose."""
    return sim03.transform_from_position_quaternion(
        position,
        sim03.DOWNWARD_EE_ORIENTATION,
    )


def command_and_step(panda_id: int, position: np.ndarray, ik_params: tuple) -> None:
    """Command Panda to a pose and advance one simulation step."""
    sim03.command_ee_pose(
        panda_id,
        position,
        sim03.DOWNWARD_EE_ORIENTATION,
        ik_params,
    )
    for _ in range(CONTROL_SUBSTEPS):
        p.stepSimulation()


def record_static_demo(panda_id: int, ik_params: tuple) -> DemoData:
    """Record the actual Franka end-effector trajectory on a static object."""
    poses: list[np.ndarray] = []
    timestamps: list[float] = []
    n_steps = int(DEMO_DURATION * sim03.FPS)

    for frame_idx in range(n_steps + 1):
        t_demo = frame_idx * sim03.DT
        alpha = min(1.0, t_demo / DEMO_DURATION)
        offset = (1.0 - alpha) * DEMO_START_OFFSET + alpha * DEMO_END_OFFSET
        target_pos = STATIC_BOX_POS + offset
        command_and_step(panda_id, target_pos, ik_params)

        ee_pos = sim03.get_ee_position(panda_id)
        poses.append(make_pose(ee_pos))
        timestamps.append(t_demo)

    return DemoData(poses=poses, timestamps=timestamps)


def run_replay(
    box_id: int,
    panda_id: int,
    camera_matrices: tuple[list[float], list[float]],
    ik_params: tuple,
    reference_cloud: np.ndarray,
    demo_data: DemoData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay recorded demo on moving object using DynamicAlignmentTracker."""
    view_matrix, projection_matrix = camera_matrices
    estimator_config = EstimatorConfig(
        min_points=20,
        use_pca_angle=False,
        z_plane_threshold=0.03,
    )
    tracker = DynamicAlignmentTracker(
        tau=TAU,
        estimator_config=estimator_config,
        kalman_R_diag=np.array([0.004, 0.004, math.radians(30.0)]),
        init_vel_cov=0.10,
    )
    tracker.init(reference_cloud, initial_theta=0.0, timestamp=0.0)

    times: list[float] = []
    relative_errors: list[float] = []
    demo_fidelity_errors: list[float] = []
    box_positions: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []

    n_steps = int(REPLAY_DURATION * sim03.FPS)
    for frame_idx in range(1, n_steps + 1):
        replay_t = frame_idx * sim03.DT
        t_demo = replay_t % demo_data.duration
        box_pos = np.array(sim03.box_position(replay_t), dtype=float)

        p.resetBasePositionAndOrientation(
            box_id,
            box_pos.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )

        cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
        tracker.update(cloud, timestamp=replay_t)
        target_pose = tracker.get_target_pose(demo_data, t_demo=t_demo, tau=TAU)
        command_and_step(panda_id, np.asarray(target_pose[:3, 3], dtype=float), ik_params)

        ee_pos = sim03.get_ee_position(panda_id)
        demo_pose = demo_data.get_pose_at(t_demo)
        desired_relative = demo_pose[:3, 3] - STATIC_BOX_POS
        actual_relative = ee_pos - box_pos

        relative_error = float(np.linalg.norm(actual_relative[:2]))
        demo_fidelity = float(np.linalg.norm(actual_relative - desired_relative))

        times.append(replay_t)
        relative_errors.append(relative_error)
        demo_fidelity_errors.append(demo_fidelity)
        box_positions.append(box_pos)
        ee_positions.append(ee_pos)

    return (
        np.asarray(times, dtype=float),
        np.asarray(relative_errors, dtype=float),
        np.asarray(demo_fidelity_errors, dtype=float),
        np.asarray(box_positions, dtype=float),
        np.asarray(ee_positions, dtype=float),
    )


def plot_error(
    times: np.ndarray,
    relative_errors: np.ndarray,
    demo_fidelity_errors: np.ndarray,
    out_path: Path,
) -> None:
    """Plot relative replay error and demo-frame fidelity error."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rel_mm = relative_errors * 1000.0
    fidelity_mm = demo_fidelity_errors * 1000.0
    demo_offset_mm = float(np.mean(rel_mm))

    plt.figure(figsize=(9, 4.8))
    plt.axhspan(0.0, SUCCESS_THRESHOLD_M * 1000.0, color="tab:green", alpha=0.10,
                label="demo fidelity success band (<30mm)")
    plt.plot(times, rel_mm, color="tab:blue", linewidth=2,
             label="EE-to-box distance (should be constant)")
    plt.axhline(demo_offset_mm, color="tab:blue", linewidth=1.5, linestyle="--",
                alpha=0.8, label=f"demo offset ({demo_offset_mm:.1f}mm)")
    plt.plot(times, fidelity_mm, color="tab:orange", linewidth=1.8,
             label="deviation from demo trajectory in object frame (success < 30mm)")
    for cycle_start in np.arange(0.0, REPLAY_DURATION + 1e-9, DEMO_DURATION):
        plt.axvspan(cycle_start, min(cycle_start + DEMO_DURATION, REPLAY_DURATION),
                    color="0.5", alpha=0.04)
    plt.xlabel("time (s)")
    plt.ylabel("error (mm)")
    plt.title("MT3 Static Demo Replay on Moving Object")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_trajectory(
    times: np.ndarray,
    box_positions: np.ndarray,
    ee_positions: np.ndarray,
    out_path: Path,
) -> None:
    """Plot end-effector XY position relative to the moving box."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    relative_positions = ee_positions - box_positions
    mask = times >= 2.0
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)

    relative_xy = relative_positions[mask, :2]
    plot_times = times[mask]
    mean_xy = relative_xy.mean(axis=0)

    plt.figure(figsize=(6, 6))
    plt.plot(relative_xy[:, 0], relative_xy[:, 1], color="tab:blue", linewidth=1.5,
             alpha=0.45, label="relative EE path after convergence")
    scatter = plt.scatter(
        relative_xy[:, 0],
        relative_xy[:, 1],
        c=plot_times,
        cmap="viridis",
        s=14,
        alpha=0.9,
        label="frames after t=2s",
    )
    plt.scatter([mean_xy[0]], [mean_xy[1]], color="black", s=40, marker="x",
                label="mean relative position")

    success_circle = plt.Circle(
        mean_xy,
        SUCCESS_THRESHOLD_M,
        color="tab:green",
        fill=False,
        linewidth=2,
        linestyle="--",
        label="30mm success band",
    )
    ax = plt.gca()
    ax.add_patch(success_circle)

    plt.xlabel("relative x = EE_x - box_x (m)")
    plt.ylabel("relative y = EE_y - box_y (m)")
    plt.title("End-Effector Position in Object Frame (should cluster near center)")
    max_dev = float(np.max(np.linalg.norm(relative_xy - mean_xy, axis=1)))
    half_width = max(0.05, max_dev + 0.02)
    plt.xlim(mean_xy[0] - half_width, mean_xy[0] + half_width)
    plt.ylim(mean_xy[1] - half_width, mean_xy[1] + half_width)
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    cbar = plt.colorbar(scatter)
    cbar.set_label("time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet DIRECT")

    try:
        box_id, panda_id, camera_matrices, ik_params = setup_scene()
        view_matrix, projection_matrix = camera_matrices

        reference_cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
        if reference_cloud.shape[0] < 20:
            raise RuntimeError(
                "Static reference depth segmentation produced too few box points: "
                f"{reference_cloud.shape[0]} < 20"
            )

        for _ in range(60):
            command_and_step(panda_id, STATIC_BOX_POS + DEMO_START_OFFSET, ik_params)

        demo_data = record_static_demo(panda_id, ik_params)
        replay = run_replay(
            box_id,
            panda_id,
            camera_matrices,
            ik_params,
            reference_cloud,
            demo_data,
        )
    finally:
        p.disconnect()

    times, relative_errors, demo_fidelity_errors, box_positions, ee_positions = replay
    out_dir = PROJECT_ROOT / "simulation" / "results"
    plot_error(
        times,
        relative_errors,
        demo_fidelity_errors,
        out_dir / "mt3_integration_error.png",
    )
    plot_trajectory(
        times,
        box_positions,
        ee_positions,
        out_dir / "mt3_integration_trajectory.png",
    )

    success_rate = float(np.mean(demo_fidelity_errors < SUCCESS_THRESHOLD_M))
    mean_error_mm = float(np.mean(demo_fidelity_errors) * 1000.0)
    print(f"MT3 replay success rate: {success_rate * 100:.1f}%")
    print(f"Mean demo fidelity error: {mean_error_mm:.1f} mm")
    if success_rate >= 0.80:
        print("SUCCESS: relative error < 30mm for >80% of replay frames")
    else:
        print("FAILURE: relative error criterion not met")
    print("Saved simulation/results/mt3_integration_error.png")
    print("Saved simulation/results/mt3_integration_trajectory.png")


if __name__ == "__main__":
    main()
