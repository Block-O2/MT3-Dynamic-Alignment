"""
PyBullet depth-camera simulation for DynamicAlignmentTracker.

Scene:
- A cube moves in a circle on a tabletop/plane.
- A top-down PyBullet camera captures a depth image every frame.
- Depth pixels closer than the tabletop are unprojected into a world point cloud.
- The segmented cube cloud is fed into DynamicAlignmentTracker.
- Ground-truth and estimated XY trajectories are saved after 10 seconds.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynamic_mt3_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dynamic_mt3_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_alignment.pose_estimator import EstimatorConfig
from dynamic_alignment.tracker import DynamicAlignmentTracker


RADIUS = 0.30
OMEGA = 0.30
DURATION = 10.0
FPS = 30
DT = 1.0 / FPS

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CAMERA_Z = 1.20
NEAR_PLANE = 0.01
FAR_PLANE = 2.0
DEPTH_MARGIN = 0.015

CUBE_HALF_EXTENT = 0.04


def cube_position(t: float) -> tuple[float, float, float]:
    """Ground-truth cube center for circular motion."""
    x = RADIUS * math.cos(OMEGA * t)
    y = RADIUS * math.sin(OMEGA * t)
    z = CUBE_HALF_EXTENT
    return x, y, z


def make_camera_matrices() -> tuple[list[float], list[float]]:
    """Create a fixed top-down camera over the origin."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.0, 0.0, CAMERA_Z],
        cameraTargetPosition=[0.0, 0.0, 0.0],
        cameraUpVector=[0.0, 1.0, 0.0],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        nearVal=NEAR_PLANE,
        farVal=FAR_PLANE,
    )
    return view_matrix, projection_matrix


def depth_buffer_to_metric(depth_buffer: np.ndarray) -> np.ndarray:
    """Convert OpenGL depth buffer values in [0, 1] to camera depth in meters."""
    return (FAR_PLANE * NEAR_PLANE) / (
        FAR_PLANE - (FAR_PLANE - NEAR_PLANE) * depth_buffer
    )


def depth_image_to_world_cloud(
    depth_buffer: np.ndarray,
    view_matrix: list[float],
    projection_matrix: list[float],
) -> np.ndarray:
    """
    Segment the cube by depth threshold and unproject selected depth pixels.

    The tabletop/plane is the far dominant surface in this top-down view.
    Pixels closer than that plane by DEPTH_MARGIN are treated as cube pixels.
    """
    depth_metric = depth_buffer_to_metric(depth_buffer)
    plane_depth = float(np.percentile(depth_metric, 90.0))
    mask = depth_metric < (plane_depth - DEPTH_MARGIN)

    vs, us = np.nonzero(mask)
    if us.size == 0:
        return np.empty((0, 3), dtype=float)

    x_ndc = 2.0 * us.astype(float) / (IMAGE_WIDTH - 1) - 1.0
    y_ndc = 1.0 - 2.0 * vs.astype(float) / (IMAGE_HEIGHT - 1)
    z_ndc = 2.0 * depth_buffer[vs, us].astype(float) - 1.0

    clip_points = np.column_stack(
        [x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc, dtype=float)]
    )

    view = np.array(view_matrix, dtype=float).reshape(4, 4).T
    projection = np.array(projection_matrix, dtype=float).reshape(4, 4).T
    inv_projection_view = np.linalg.inv(projection @ view)

    world_h = (inv_projection_view @ clip_points.T).T
    world_h /= world_h[:, 3:4]
    return world_h[:, :3]


def capture_object_cloud(
    view_matrix: list[float],
    projection_matrix: list[float],
) -> np.ndarray:
    """Capture a depth image and return the segmented cube point cloud."""
    _, _, _, depth, _ = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    depth_buffer = np.asarray(depth, dtype=float).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    return depth_image_to_world_cloud(depth_buffer, view_matrix, projection_matrix)


def create_cube() -> int:
    """Create a small cube with known geometry instead of relying on URDF scale."""
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[CUBE_HALF_EXTENT] * 3,
        rgbaColor=[0.9, 0.2, 0.1, 1.0],
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[CUBE_HALF_EXTENT] * 3,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=cube_position(0.0),
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def plot_results(
    times: np.ndarray,
    true_xy: np.ndarray,
    estimated_xy: np.ndarray,
    out_dir: Path,
) -> None:
    """Save trajectory and position-error plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(true_xy[:, 0], true_xy[:, 1], label="true", linewidth=2)
    plt.plot(estimated_xy[:, 0], estimated_xy[:, 1], label="estimated", linewidth=2)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("XY trajectory")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_xy.png", dpi=200)
    plt.close()

    errors = np.linalg.norm(estimated_xy - true_xy, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(times, errors * 1000.0, linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("position error (mm)")
    plt.title("Position error over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "position_error.png", dpi=200)
    plt.close()


def main() -> None:
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet GUI")

    true_positions: list[tuple[float, float]] = []
    estimated_positions: list[tuple[float, float]] = []
    timestamps: list[float] = []

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.8)
        p.setTimeStep(DT)
        p.loadURDF("plane.urdf")

        cube_id = create_cube()
        view_matrix, projection_matrix = make_camera_matrices()

        estimator_config = EstimatorConfig(
            min_points=20,
            use_pca_angle=False,
            z_plane_threshold=0.03,
        )
        tracker = DynamicAlignmentTracker(
            tau=0.0,
            estimator_config=estimator_config,
            kalman_R_diag=np.array([0.003, 0.003, math.radians(30.0)]),
            init_vel_cov=0.10,
        )

        initial_xy: tuple[float, float] | None = None
        n_steps = int(DURATION * FPS)

        for frame_idx in range(n_steps + 1):
            sim_t = frame_idx * DT
            x, y, z = cube_position(sim_t)
            p.resetBasePositionAndOrientation(
                cube_id,
                [x, y, z],
                [0.0, 0.0, 0.0, 1.0],
            )
            p.stepSimulation()

            cloud = capture_object_cloud(view_matrix, projection_matrix)

            if not tracker.is_initialized:
                if cloud.shape[0] < estimator_config.min_points:
                    raise RuntimeError(
                        "Initial depth segmentation produced too few cube points: "
                        f"{cloud.shape[0]} < {estimator_config.min_points}"
                    )
                tracker.init(cloud, initial_theta=0.0, timestamp=sim_t)
                initial_xy = (x, y)
                est_x, est_y = x, y
            else:
                state = tracker.update(cloud, timestamp=sim_t)
                if initial_xy is None:
                    raise RuntimeError("Tracker initialized without initial_xy")
                est_x = initial_xy[0] + state.delta_x
                est_y = initial_xy[1] + state.delta_y

            timestamps.append(sim_t)
            true_positions.append((x, y))
            estimated_positions.append((est_x, est_y))

            time.sleep(DT)

    finally:
        p.disconnect()

    plot_results(
        times=np.asarray(timestamps, dtype=float),
        true_xy=np.asarray(true_positions, dtype=float),
        estimated_xy=np.asarray(estimated_positions, dtype=float),
        out_dir=PROJECT_ROOT / "simulation" / "results",
    )

    print("Saved simulation/results/trajectory_xy.png")
    print("Saved simulation/results/position_error.png")


if __name__ == "__main__":
    main()
