"""
Closed-loop PyBullet simulation with a Franka Panda and DynamicAlignmentTracker.

Scene:
- Franka Panda arm loaded from pybullet_data/franka_panda/panda.urdf.
- A small box moves on a table in a circle.
- A fixed overhead RGB-D camera provides depth images for box tracking.

Each frame:
1. Segment the box point cloud from the depth image.
2. Feed it to DynamicAlignmentTracker.
3. Convert tracker output to an end-effector target pose.
4. Solve IK and command Panda arm joints with POSITION_CONTROL.
5. Record end-effector position, box position, and relative-position error.
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
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_alignment.pose_estimator import EstimatorConfig
from dynamic_alignment.tracker import DynamicAlignmentTracker
from dynamic_alignment.types import make_static_demo


RADIUS = 0.15
OMEGA = 0.30
DURATION = 15.0
FPS = 30
DT = 1.0 / FPS

TABLE_CENTER = np.array([0.45, 0.0, -0.025], dtype=float)
TABLE_HALF_EXTENTS = np.array([0.55, 0.45, 0.025], dtype=float)
TABLE_TOP_Z = float(TABLE_CENTER[2] + TABLE_HALF_EXTENTS[2])

BOX_HALF_EXTENT = 0.035
BOX_CENTER = np.array([0.50, 0.0], dtype=float)

DESIRED_EE_OFFSET = np.array([-0.12, 0.0, 0.22], dtype=float)
DOWNWARD_EE_ORIENTATION = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
EE_LINK_INDEX = 11
ARM_JOINT_INDICES = list(range(7))
FINGER_JOINT_INDICES = [9, 10]

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
GIF_SAMPLE_STRIDE = 2
GIF_PLAYBACK_SPEEDUP = 3.0
CAMERA_Z = 1.25
NEAR_PLANE = 0.01
FAR_PLANE = 2.0
DEPTH_MARGIN = 0.015

BOX_Z_MIN = TABLE_TOP_Z + 0.005
BOX_Z_MAX = TABLE_TOP_Z + 2.4 * BOX_HALF_EXTENT
BOX_X_MIN = BOX_CENTER[0] - BOX_HALF_EXTENT - 0.05
BOX_X_MAX = BOX_CENTER[0] + RADIUS + BOX_HALF_EXTENT + 0.05
BOX_Y_MIN = BOX_CENTER[1] - RADIUS - BOX_HALF_EXTENT - 0.05
BOX_Y_MAX = BOX_CENTER[1] + RADIUS + BOX_HALF_EXTENT + 0.05


def box_position(t: float) -> tuple[float, float, float]:
    """Ground-truth box center for front-workspace semicircle motion."""
    sweep = math.pi
    phase = (OMEGA * t) % (2.0 * sweep)
    if phase <= sweep:
        angle = -0.5 * math.pi + phase
    else:
        angle = -0.5 * math.pi + (2.0 * sweep - phase)

    x = BOX_CENTER[0] + RADIUS * math.cos(angle)
    y = BOX_CENTER[1] + RADIUS * math.sin(angle)
    z = TABLE_TOP_Z + BOX_HALF_EXTENT
    return float(x), float(y), float(z)


def transform_from_position_quaternion(position: np.ndarray, quat: tuple[float, ...]) -> np.ndarray:
    """Build a 4x4 transform from world position and quaternion."""
    transform = np.eye(4)
    transform[:3, :3] = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def make_camera_matrices() -> tuple[list[float], list[float]]:
    """Create a fixed overhead camera looking at the table workspace."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[BOX_CENTER[0], BOX_CENTER[1], CAMERA_Z],
        cameraTargetPosition=[BOX_CENTER[0], BOX_CENTER[1], TABLE_TOP_Z],
        cameraUpVector=[0.0, 1.0, 0.0],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=65.0,
        aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
        nearVal=NEAR_PLANE,
        farVal=FAR_PLANE,
    )
    return view_matrix, projection_matrix


def make_demo_camera_matrices() -> tuple[list[float], list[float]]:
    """Create a camera view suitable for recorded demo playback."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.95, -0.65, 0.55],
        cameraTargetPosition=[0.42, 0.0, 0.12],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
        nearVal=0.01,
        farVal=2.0,
    )
    return view_matrix, projection_matrix


def capture_demo_frame(
    view_matrix: list[float],
    projection_matrix: list[float],
) -> Image.Image:
    """Render one RGB frame for the fallback GIF recorder."""
    _, _, rgba, _, _ = p.getCameraImage(
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    frame = np.asarray(rgba, dtype=np.uint8).reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 4)
    return Image.fromarray(frame[:, :, :3], mode="RGB")


def depth_buffer_to_metric(depth_buffer: np.ndarray) -> np.ndarray:
    """Convert OpenGL depth buffer values in [0, 1] to metric camera depth."""
    return (FAR_PLANE * NEAR_PLANE) / (
        FAR_PLANE - (FAR_PLANE - NEAR_PLANE) * depth_buffer
    )


def depth_image_to_world_cloud(
    depth_buffer: np.ndarray,
    view_matrix: list[float],
    projection_matrix: list[float],
) -> np.ndarray:
    """
    Segment the box from a rendered depth image and unproject to world points.

    Depth first removes the table plane. The world-z band then rejects Panda links,
    which are also closer than the table but not at the box height.
    """
    depth_metric = depth_buffer_to_metric(depth_buffer)
    plane_depth = float(np.percentile(depth_metric, 90.0))
    closer_than_table = depth_metric < (plane_depth - DEPTH_MARGIN)

    vs, us = np.nonzero(closer_than_table)
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
    world = world_h[:, :3]

    mask = (
        (world[:, 0] >= BOX_X_MIN)
        & (world[:, 0] <= BOX_X_MAX)
        & (world[:, 1] >= BOX_Y_MIN)
        & (world[:, 1] <= BOX_Y_MAX)
        & (world[:, 2] >= BOX_Z_MIN)
        & (world[:, 2] <= BOX_Z_MAX)
    )
    return world[mask]


def capture_box_cloud(
    view_matrix: list[float],
    projection_matrix: list[float],
) -> np.ndarray:
    """Capture a depth image and return a segmented box point cloud."""
    _, _, _, depth, _ = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    depth_buffer = np.asarray(depth, dtype=float).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    return depth_image_to_world_cloud(depth_buffer, view_matrix, projection_matrix)


def create_table() -> int:
    """Create a simple collision table with top surface at z=0."""
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=TABLE_HALF_EXTENTS.tolist(),
        rgbaColor=[0.55, 0.55, 0.55, 1.0],
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=TABLE_HALF_EXTENTS.tolist(),
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=TABLE_CENTER.tolist(),
    )


def create_box() -> int:
    """Create the moving tabletop box."""
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[BOX_HALF_EXTENT] * 3,
        rgbaColor=[0.9, 0.15, 0.1, 1.0],
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[BOX_HALF_EXTENT] * 3,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=box_position(0.0),
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def load_panda() -> int:
    """Load Panda and open the gripper to keep the box visible from overhead."""
    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, TABLE_TOP_Z],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=True,
    )

    home_positions = [0.0, -0.45, 0.0, -2.2, 0.0, 1.75, 0.78]
    for joint_idx, joint_pos in zip(ARM_JOINT_INDICES, home_positions):
        p.resetJointState(panda_id, joint_idx, joint_pos)
    for joint_idx in FINGER_JOINT_INDICES:
        p.resetJointState(panda_id, joint_idx, 0.04)
    return panda_id


def get_joint_limits(panda_id: int) -> tuple[list[float], list[float], list[float], list[float]]:
    """Read arm joint limits and return IK parameters."""
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    joint_ranges: list[float] = []
    rest_poses: list[float] = []

    for joint_idx in ARM_JOINT_INDICES:
        info = p.getJointInfo(panda_id, joint_idx)
        lower = float(info[8])
        upper = float(info[9])
        if lower >= upper:
            lower, upper = -math.pi, math.pi
        lower_limits.append(lower)
        upper_limits.append(upper)
        joint_ranges.append(upper - lower)
        rest_poses.append(float(p.getJointState(panda_id, joint_idx)[0]))

    return lower_limits, upper_limits, joint_ranges, rest_poses


def command_ee_pose(
    panda_id: int,
    target_position: np.ndarray,
    target_orientation: tuple[float, ...],
    ik_params: tuple[list[float], list[float], list[float], list[float]],
) -> None:
    """Solve IK and command Panda arm joints."""
    lower_limits, upper_limits, joint_ranges, rest_poses = ik_params
    ik_solution = p.calculateInverseKinematics(
        bodyUniqueId=panda_id,
        endEffectorLinkIndex=EE_LINK_INDEX,
        targetPosition=target_position.tolist(),
        targetOrientation=target_orientation,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=80,
        residualThreshold=1e-4,
    )
    p.setJointMotorControlArray(
        bodyUniqueId=panda_id,
        jointIndices=ARM_JOINT_INDICES,
        controlMode=p.POSITION_CONTROL,
        targetPositions=list(ik_solution[:7]),
        forces=[87.0] * 7,
        positionGains=[0.06] * 7,
        velocityGains=[1.0] * 7,
    )
    p.setJointMotorControlArray(
        bodyUniqueId=panda_id,
        jointIndices=FINGER_JOINT_INDICES,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[0.04, 0.04],
        forces=[20.0, 20.0],
    )


def get_ee_position(panda_id: int) -> np.ndarray:
    """Return current Panda end-effector world position."""
    return np.array(p.getLinkState(panda_id, EE_LINK_INDEX, computeForwardKinematics=True)[0])


def plot_results(
    times: np.ndarray,
    box_positions: np.ndarray,
    ee_positions: np.ndarray,
    relative_errors: np.ndarray,
    out_dir: Path,
) -> None:
    """Save closed-loop trajectory and relative-error plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 5))
    ax_box = fig.add_subplot(1, 2, 1, projection="3d")
    ax_ee = fig.add_subplot(1, 2, 2, projection="3d")

    ax_box.plot(box_positions[:, 0], box_positions[:, 1], box_positions[:, 2], linewidth=2)
    ax_box.set_title("Box trajectory")
    ax_box.set_xlabel("x (m)")
    ax_box.set_ylabel("y (m)")
    ax_box.set_zlabel("z (m)")

    ax_ee.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], linewidth=2)
    ax_ee.set_title("End-effector trajectory")
    ax_ee.set_xlabel("x (m)")
    ax_ee.set_ylabel("y (m)")
    ax_ee.set_zlabel("z (m)")

    all_positions = np.vstack([box_positions, ee_positions])
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    centers = 0.5 * (mins + maxs)
    spans = np.maximum(maxs - mins, 0.1)
    max_span = float(spans.max())
    for ax in (ax_box, ax_ee):
        ax.set_xlim(centers[0] - max_span / 2, centers[0] + max_span / 2)
        ax.set_ylim(centers[1] - max_span / 2, centers[1] + max_span / 2)
        ax.set_zlim(centers[2] - max_span / 2, centers[2] + max_span / 2)

    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_3d_trajectories.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(8, 4))
    plt.plot(times, relative_errors * 1000.0, linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("relative position error (mm)")
    plt.title("End-effector relative error to moving box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "closed_loop_relative_error.png", dpi=200)
    plt.close()


def _prepare_gif_frame(frame: Image.Image) -> Image.Image:
    """Resize frame to 640px width and add the 3x playback overlay."""
    if frame.width != VIDEO_WIDTH:
        height = int(round(frame.height * VIDEO_WIDTH / frame.width))
        frame = frame.resize((VIDEO_WIDTH, height), Image.Resampling.LANCZOS)
    else:
        frame = frame.copy()

    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    label = "3x"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 8
    margin = 12
    x0 = frame.width - text_w - 2 * pad - margin
    y0 = frame.height - text_h - 2 * pad - margin
    x1 = frame.width - margin
    y1 = frame.height - margin

    draw.rounded_rectangle((x0, y0, x1, y1), radius=5, fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad), label, fill=(255, 255, 255), font=font)
    return frame


def save_demo_gif(frames: list[Image.Image], out_path: Path) -> None:
    """Save sampled rendered frames as a compact 3x animated GIF."""
    if not frames:
        return
    frame_duration_ms = max(
        20,
        int(round((1000.0 * GIF_SAMPLE_STRIDE / FPS) / GIF_PLAYBACK_SPEEDUP)),
    )
    prepared_frames = [_prepare_gif_frame(frame) for frame in frames]
    prepared_frames[0].save(
        out_path,
        save_all=True,
        append_images=prepared_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet GUI")

    out_dir = PROJECT_ROOT / "simulation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "demo.gif"
    gif_frames: list[Image.Image] = []

    timestamps: list[float] = []
    box_positions: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []
    relative_errors: list[float] = []

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.8)
        p.setTimeStep(DT)
        p.loadURDF("plane.urdf")
        create_table()
        box_id = create_box()
        panda_id = load_panda()

        view_matrix, projection_matrix = make_camera_matrices()
        demo_view_matrix, demo_projection_matrix = make_demo_camera_matrices()
        target_orientation = DOWNWARD_EE_ORIENTATION
        ik_params = get_joint_limits(panda_id)

        initial_box_pos = np.array(box_position(0.0), dtype=float)
        initial_ee_target = initial_box_pos + DESIRED_EE_OFFSET
        demo_data = make_static_demo(
            transform_from_position_quaternion(initial_ee_target, target_orientation),
            timestamp=0.0,
        )

        for _ in range(90):
            command_ee_pose(panda_id, initial_ee_target, target_orientation, ik_params)
            p.stepSimulation()

        estimator_config = EstimatorConfig(
            min_points=20,
            use_pca_angle=False,
            z_plane_threshold=0.03,
        )
        tracker = DynamicAlignmentTracker(
            tau=0.1,
            estimator_config=estimator_config,
            kalman_R_diag=np.array([0.004, 0.004, math.radians(30.0)]),
            init_vel_cov=0.10,
        )

        n_steps = int(DURATION * FPS)
        for frame_idx in range(n_steps + 1):
            sim_t = frame_idx * DT
            box_pos = np.array(box_position(sim_t), dtype=float)
            p.resetBasePositionAndOrientation(
                box_id,
                box_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
            )

            cloud = capture_box_cloud(view_matrix, projection_matrix)
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
                target_pose = tracker.get_target_pose(demo_data, t_demo=0.0, tau=0.1)

            command_ee_pose(
                panda_id,
                np.asarray(target_pose[:3, 3], dtype=float),
                target_orientation,
                ik_params,
            )
            p.stepSimulation()

            ee_pos = get_ee_position(panda_id)
            relative_error = float(np.linalg.norm((ee_pos - box_pos) - DESIRED_EE_OFFSET))

            if frame_idx % GIF_SAMPLE_STRIDE == 0:
                gif_frames.append(capture_demo_frame(demo_view_matrix, demo_projection_matrix))

            timestamps.append(sim_t)
            box_positions.append(box_pos)
            ee_positions.append(ee_pos)
            relative_errors.append(relative_error)

            time.sleep(DT)

    finally:
        p.disconnect()

    plot_results(
        times=np.asarray(timestamps, dtype=float),
        box_positions=np.asarray(box_positions, dtype=float),
        ee_positions=np.asarray(ee_positions, dtype=float),
        relative_errors=np.asarray(relative_errors, dtype=float),
        out_dir=out_dir,
    )
    save_demo_gif(gif_frames, gif_path)

    print("Saved simulation/results/demo.gif")
    print("Saved simulation/results/closed_loop_3d_trajectories.png")
    print("Saved simulation/results/closed_loop_relative_error.png")


if __name__ == "__main__":
    main()
