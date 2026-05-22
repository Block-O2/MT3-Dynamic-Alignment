"""
MT3 grasping experiment in PyBullet.

Scene:
- Franka Panda arm.
- Small box moves along the same front-workspace semicircle used in earlier
  MT3 replay experiments.
- A static-object grasp demo is replayed on a moving object through
  DynamicAlignmentTracker:

      T_WE_target(t) = T_delta(t + tau) @ T_WE_demo(t)

The grasp is made deterministic by creating a fixed PyBullet constraint when
the gripper closes near the box. This keeps the experiment focused on the MT3
alignment/replay question instead of PyBullet contact tuning.

Outputs:
- simulation/results/plot/grasping_demo.gif
- simulation/results/plot/grasping_success_rate.png
- simulation/results/raw/10_grasping_final_seed42.csv
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import random
import sys
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynamic_mt3_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dynamic_mt3_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

USE_GUI = False
TEST_SPEEDS_CM_S = (2.0, 4.0, 6.0, 8.0)
N_TRIALS = 10
TAU = 0.1
RADIUS = 0.15
STATIC_BOX_POS = np.array(
    [0.5, 0.0, sim03.TABLE_TOP_Z + sim03.BOX_HALF_EXTENT],
    dtype=float,
)
CONTROL_SUBSTEPS = 8
TRACKING_DIAGNOSTIC_DURATION = 5.0
TRACKING_DIAGNOSTIC_STEADY_STATE_START = 2.0
CARTESIAN_VELOCITY_GAIN = 10.0
CARTESIAN_DAMPING = 0.05
MAX_JOINT_VELOCITY = 1.0

DEMO_APPROACH_DURATION = 2.5
DEMO_CLOSE_DURATION = 0.6
DEMO_LIFT_DURATION = 1.9
DEMO_DURATION = DEMO_APPROACH_DURATION + DEMO_CLOSE_DURATION + DEMO_LIFT_DURATION
REPLAY_DURATION = 25.0
CONTACT_OFFSET = np.array([0.0, 0.0, 0.045], dtype=float)
START_OFFSET = np.array([0.0, 0.0, 0.20], dtype=float)
LIFT_OFFSET = CONTACT_OFFSET + np.array([0.0, 0.0, 0.10], dtype=float)
OPEN_GRIPPER = 0.04
CLOSED_GRIPPER = 0.0

GRASP_ATTACH_DISTANCE = 0.085
GRASP_CLOSE_THRESHOLD = 0.012
GRASP_CLOSE_BOX_EE_DISTANCE_M = 0.040
MAX_GRASP_ATTEMPTS = 3
LATERAL_ERROR_SMOOTHING_FRAMES = 5
LIFT_SUCCESS_MARGIN = 0.05
BOX_MASS = 0.05
CONTACT_POSITION_TOLERANCE_M = 0.015
DESCENT_GATE_THRESHOLD_M = 0.008
APPROACH_WINDOW_S = 0.5
APPROACH_HORIZONTAL_TOLERANCE_M = 0.020
ORIENTATION_TOLERANCE_DEG = 20.0

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
GIF_SAMPLE_STRIDE = 2
GIF_PLAYBACK_SPEEDUP = 3.0
RECORD_REPRESENTATIVE_SPEED_CM_S = 4.0
RECORD_REPRESENTATIVE_TRIAL = 0
DEBUG_ADAPTIVE_GATE = False
DEBUG_CONTACT_WINDOW = False


class GraspDemo(NamedTuple):
    poses: DemoData
    gripper_widths: np.ndarray


class TrialResult(NamedTuple):
    success: bool
    final_lift_m: float
    max_lift_m: float
    progress_rate: float
    lift_success: bool
    position_success: bool
    orientation_success: bool
    approach_success: bool
    attempt_limit_failure: bool
    n_attempts: int
    contact_position_error_mm: float
    orientation_error_deg: float
    approach_max_error_mm: float


def moving_box_position(t: float, speed_cm_s: float) -> np.ndarray:
    """Front-workspace semicircle with speed-controlled angular velocity."""
    radius = RADIUS
    omega = (speed_cm_s / 100.0) / radius
    sweep = math.pi
    phase = (omega * t) % (2.0 * sweep)
    if phase <= sweep:
        angle = -0.5 * math.pi + phase
    else:
        angle = -0.5 * math.pi + (2.0 * sweep - phase)
    return np.array(
        [
            STATIC_BOX_POS[0] + radius * math.cos(angle),
            STATIC_BOX_POS[1] + radius * math.sin(angle),
            STATIC_BOX_POS[2],
        ],
        dtype=float,
    )


def create_graspable_box(position: np.ndarray) -> int:
    """Create a dynamic tabletop box that can be attached to the gripper."""
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[sim03.BOX_HALF_EXTENT] * 3,
        rgbaColor=[0.9, 0.15, 0.1, 1.0],
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[sim03.BOX_HALF_EXTENT] * 3,
    )
    box_id = p.createMultiBody(
        baseMass=BOX_MASS,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    p.changeDynamics(box_id, -1, lateralFriction=1.0, spinningFriction=0.02, rollingFriction=0.02)
    return box_id


def setup_scene() -> tuple[int, int, tuple[list[float], list[float]], tuple]:
    """Create table, dynamic box, Panda, camera, and IK parameters."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.8)
    p.setTimeStep(sim03.DT / CONTROL_SUBSTEPS)
    p.loadURDF("plane.urdf")
    sim03.create_table()
    box_id = create_graspable_box(STATIC_BOX_POS)
    panda_id = sim03.load_panda()
    camera_matrices = sim03.make_camera_matrices()
    ik_params = sim03.get_joint_limits(panda_id)
    return box_id, panda_id, camera_matrices, ik_params


def make_pose(position: np.ndarray) -> np.ndarray:
    """Build a downward-facing end-effector pose."""
    return sim03.transform_from_position_quaternion(
        position,
        sim03.DOWNWARD_EE_ORIENTATION,
    )


def set_gripper(panda_id: int, width: float) -> None:
    """Command Panda finger joints."""
    width = float(np.clip(width, CLOSED_GRIPPER, OPEN_GRIPPER))
    p.setJointMotorControlArray(
        bodyUniqueId=panda_id,
        jointIndices=sim03.FINGER_JOINT_INDICES,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[width, width],
        forces=[45.0, 45.0],
    )


def command_ee_velocity(
    panda_id: int,
    target_position: np.ndarray,
) -> None:
    """Command arm joint velocities with XY tracking prioritized over Z descent."""
    ee_state = p.getLinkState(panda_id, sim03.EE_LINK_INDEX, computeForwardKinematics=True)
    ee_pos = np.asarray(ee_state[4], dtype=float)
    error = np.asarray(target_position, dtype=float) - ee_pos
    error_xy = error[:2]
    error_z = float(error[2])

    movable_joint_indices = [
        joint_idx
        for joint_idx in range(p.getNumJoints(panda_id))
        if p.getJointInfo(panda_id, joint_idx)[2] != p.JOINT_FIXED
    ]
    arm_column_indices = [
        movable_joint_indices.index(joint_idx)
        for joint_idx in sim03.ARM_JOINT_INDICES
    ]
    joint_states = p.getJointStates(panda_id, movable_joint_indices)
    joint_positions = [float(state[0]) for state in joint_states]
    zeros = [0.0] * len(movable_joint_indices)
    jac_t, _ = p.calculateJacobian(
        panda_id,
        sim03.EE_LINK_INDEX,
        localPosition=[0.0, 0.0, 0.0],
        objPositions=joint_positions,
        objVelocities=zeros,
        objAccelerations=zeros,
    )
    jac = np.asarray(jac_t, dtype=float)[:, arm_column_indices]
    lambda_sq = CARTESIAN_DAMPING ** 2

    jac_xy = jac[:2, :]
    j_inv_xy = jac_xy.T @ np.linalg.inv(jac_xy @ jac_xy.T + lambda_sq * np.eye(2))
    q_dot_xy = j_inv_xy @ (error_xy * 15.0)

    jac_z = jac[2:3, :]
    null_xy = np.eye(len(sim03.ARM_JOINT_INDICES)) - j_inv_xy @ jac_xy
    z_denom = float((jac_z @ null_xy @ jac_z.T)[0, 0] + lambda_sq)
    q_dot_z = (null_xy @ jac_z.T).reshape(-1) * ((error_z * CARTESIAN_VELOCITY_GAIN) / z_denom)

    joint_velocities = q_dot_xy + q_dot_z
    joint_velocities = np.clip(joint_velocities, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)

    p.setJointMotorControlArray(
        bodyUniqueId=panda_id,
        jointIndices=sim03.ARM_JOINT_INDICES,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=joint_velocities.tolist(),
        forces=[87.0] * len(sim03.ARM_JOINT_INDICES),
    )


def command_and_step(
    panda_id: int,
    position: np.ndarray,
    ik_params: tuple,
    gripper_width: float,
) -> None:
    """Command EE pose and gripper, then advance one control frame."""
    set_gripper(panda_id, gripper_width)
    for _ in range(CONTROL_SUBSTEPS):
        command_ee_velocity(panda_id, np.asarray(position, dtype=float))
        p.stepSimulation()


def gripper_width_at(t_demo: float) -> float:
    """Open during approach, close, then stay closed during lift."""
    if t_demo < DEMO_APPROACH_DURATION:
        return OPEN_GRIPPER
    if t_demo < DEMO_APPROACH_DURATION + DEMO_CLOSE_DURATION:
        alpha = (t_demo - DEMO_APPROACH_DURATION) / DEMO_CLOSE_DURATION
        return float((1.0 - alpha) * OPEN_GRIPPER + alpha * CLOSED_GRIPPER)
    return CLOSED_GRIPPER


def demo_offset_at(t_demo: float) -> np.ndarray:
    """Recorded static grasp demo offset relative to the box reference pose."""
    if t_demo < DEMO_APPROACH_DURATION:
        alpha = t_demo / DEMO_APPROACH_DURATION
        return (1.0 - alpha) * START_OFFSET + alpha * CONTACT_OFFSET
    if t_demo < DEMO_APPROACH_DURATION + DEMO_CLOSE_DURATION:
        return CONTACT_OFFSET.copy()
    lift_t = t_demo - DEMO_APPROACH_DURATION - DEMO_CLOSE_DURATION
    alpha = min(1.0, lift_t / DEMO_LIFT_DURATION)
    return (1.0 - alpha) * CONTACT_OFFSET + alpha * LIFT_OFFSET


def record_static_grasp_demo(panda_id: int, ik_params: tuple) -> GraspDemo:
    """Record a static grasp approach-close-lift trajectory as DemoData."""
    poses: list[np.ndarray] = []
    timestamps: list[float] = []
    gripper_widths: list[float] = []

    n_steps = int(DEMO_DURATION * sim03.FPS)
    for frame_idx in range(n_steps + 1):
        t_demo = frame_idx * sim03.DT
        target_pos = STATIC_BOX_POS + demo_offset_at(t_demo)
        gripper_width = gripper_width_at(t_demo)
        command_and_step(panda_id, target_pos, ik_params, gripper_width)

        ee_pos = sim03.get_ee_position(panda_id)
        poses.append(make_pose(ee_pos))
        timestamps.append(t_demo)
        gripper_widths.append(gripper_width)

    return GraspDemo(
        poses=DemoData(poses=poses, timestamps=timestamps),
        gripper_widths=np.asarray(gripper_widths, dtype=float),
    )


def gripper_width_from_demo(demo: GraspDemo, t_demo: float) -> float:
    """Interpolate recorded gripper width."""
    timestamps = np.asarray(demo.poses.timestamps, dtype=float)
    return float(np.interp(t_demo, timestamps, demo.gripper_widths))


def gripper_phase(t_demo: float) -> str:
    """Return coarse gripper phase label for debug logging."""
    if t_demo < DEMO_APPROACH_DURATION:
        return "open"
    if t_demo < DEMO_APPROACH_DURATION + DEMO_CLOSE_DURATION:
        return "closing"
    return "closed"


def should_log_contact_window(t_demo: float) -> bool:
    """Log final 1s before close starts through 1s after close completes."""
    return (
        DEMO_APPROACH_DURATION - 1.0
        <= t_demo
        <= DEMO_APPROACH_DURATION + DEMO_CLOSE_DURATION + 1.0
    )


def is_descent_phase(demo_data: DemoData, t_demo: float) -> bool:
    """Return True while the demo end-effector z-position is decreasing."""
    t_now = float(np.clip(t_demo, 0.0, demo_data.duration))
    t_prev = max(0.0, t_now - sim03.DT)
    z_prev = float(demo_data.get_pose_at(t_prev)[2, 3])
    z_now = float(demo_data.get_pose_at(t_now)[2, 3])
    return z_now < z_prev - 1e-6


def make_tracker(reference_cloud: np.ndarray) -> DynamicAlignmentTracker:
    """Create DynamicAlignmentTracker with the same settings as MT3 replay scripts."""
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
    return tracker


def maybe_attach_box(
    panda_id: int,
    box_id: int,
    gripper_width: float,
    existing_constraint: int | None,
) -> int | None:
    """Attach box to gripper if the closing gripper is near the object."""
    if existing_constraint is not None or gripper_width > GRASP_CLOSE_THRESHOLD:
        return existing_constraint

    ee_pos = sim03.get_ee_position(panda_id)
    box_pos = np.array(p.getBasePositionAndOrientation(box_id)[0], dtype=float)
    if float(np.linalg.norm(ee_pos - box_pos)) > GRASP_ATTACH_DISTANCE:
        return None

    ee_quat = p.getLinkState(panda_id, sim03.EE_LINK_INDEX, computeForwardKinematics=True)[1]
    inv_pos, inv_quat = p.invertTransform(ee_pos.tolist(), ee_quat)
    child_frame_pos, child_frame_orn = p.multiplyTransforms(
        inv_pos,
        inv_quat,
        box_pos.tolist(),
        [0.0, 0.0, 0.0, 1.0],
    )
    constraint_id = p.createConstraint(
        parentBodyUniqueId=panda_id,
        parentLinkIndex=sim03.EE_LINK_INDEX,
        childBodyUniqueId=box_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0.0, 0.0, 0.0],
        parentFramePosition=child_frame_pos,
        parentFrameOrientation=child_frame_orn,
        childFramePosition=[0.0, 0.0, 0.0],
        childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    p.changeConstraint(constraint_id, maxForce=250.0)
    return constraint_id


def gripper_axis_alignment_error_deg(panda_id: int) -> float:
    """Return angle between gripper closing axis and the box long axis in XY."""
    ee_quat = p.getLinkState(panda_id, sim03.EE_LINK_INDEX, computeForwardKinematics=True)[1]
    ee_rotation = np.array(p.getMatrixFromQuaternion(ee_quat), dtype=float).reshape(3, 3)
    gripper_axis_xy = ee_rotation[:2, 0]
    norm = float(np.linalg.norm(gripper_axis_xy))
    if norm < 1e-9:
        return 90.0
    gripper_axis_xy /= norm

    box_long_axis_xy = np.array([1.0, 0.0], dtype=float)
    # Axis alignment is sign-invariant: 0 deg and 180 deg are both aligned.
    cos_angle = abs(float(np.dot(gripper_axis_xy, box_long_axis_xy)))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def approach_is_consistent(
    approach_errors_xy: list[tuple[float, float]],
    contact_time: float,
) -> tuple[bool, float]:
    """Check horizontal gripper-box alignment during the final pre-contact window."""
    window_start = contact_time - APPROACH_WINDOW_S
    window_errors = [
        error
        for timestamp, error in approach_errors_xy
        if window_start <= timestamp <= contact_time
    ]
    if not window_errors:
        return False, float("inf")

    max_error = float(np.max(window_errors))
    return max_error <= APPROACH_HORIZONTAL_TOLERANCE_M, max_error * 1000.0


def smooth_append(values: list[float], value: float, window_size: int) -> float:
    """Append a scalar value and return the moving-average over the latest window."""
    values.append(float(value))
    if len(values) > window_size:
        del values[:-window_size]
    return float(np.mean(values))


def reset_adaptive_replay_state(tracker: DynamicAlignmentTracker, demo_clock: list[float]) -> None:
    """Reset adaptive demo replay state after an aborted grasp attempt."""
    demo_clock[0] = 0.0
    tracker._adaptive_last_t_demo = None


def make_video_camera_matrices() -> tuple[list[float], list[float]]:
    """Create a side camera for the representative grasping GIF."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.95, -0.72, 0.55],
        cameraTargetPosition=[0.45, 0.0, 0.12],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
        nearVal=0.01,
        farVal=2.0,
    )
    return view_matrix, projection_matrix


def capture_video_frame(view_matrix: list[float], projection_matrix: list[float]) -> Image.Image:
    """Render one RGB frame for GIF output."""
    _, _, rgba, _, _ = p.getCameraImage(
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    frame = np.asarray(rgba, dtype=np.uint8).reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 4)
    return Image.fromarray(frame[:, :, :3], mode="RGB")


def _prepare_gif_frame(frame: Image.Image) -> Image.Image:
    """Resize to 640px wide and add 3x playback label."""
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
    """Save sampled rendered frames as a compact 3x GIF."""
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(
        20,
        int(round((1000.0 * GIF_SAMPLE_STRIDE / sim03.FPS) / GIF_PLAYBACK_SPEEDUP)),
    )
    prepared = [_prepare_gif_frame(frame) for frame in frames]
    prepared[0].save(
        out_path,
        save_all=True,
        append_images=prepared[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def run_trial(
    speed_cm_s: float,
    moving_object: bool,
    trial_idx: int,
    record_gif: bool = False,
) -> TrialResult:
    """Run one static-baseline or moving-object grasp trial."""
    client_id = p.connect(p.GUI if USE_GUI else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    gif_frames: list[Image.Image] = []
    plot_dir = PROJECT_ROOT / "simulation" / "results" / "plot"

    try:
        box_id, panda_id, camera_matrices, ik_params = setup_scene()
        view_matrix, projection_matrix = camera_matrices
        video_view, video_projection = make_video_camera_matrices()

        # Capture the reference cloud before moving the arm over the object;
        # otherwise the overhead depth camera sees the gripper instead of the box.
        reference_cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
        if reference_cloud.shape[0] < 20:
            raise RuntimeError(
                "Static reference depth segmentation produced too few box points: "
                f"{reference_cloud.shape[0]} < 20"
            )

        for _ in range(60):
            command_and_step(panda_id, STATIC_BOX_POS + START_OFFSET, ik_params, OPEN_GRIPPER)

        demo = record_static_grasp_demo(panda_id, ik_params)

        # Reset scene state before replay so the recorded lift does not carry over.
        p.resetBasePositionAndOrientation(box_id, STATIC_BOX_POS.tolist(), [0.0, 0.0, 0.0, 1.0])
        p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for _ in range(30):
            command_and_step(panda_id, STATIC_BOX_POS + START_OFFSET, ik_params, OPEN_GRIPPER)

        tracker = make_tracker(reference_cloud)
        constraint_id: int | None = None
        max_box_z = float(STATIC_BOX_POS[2])
        final_box_z = float(STATIC_BOX_POS[2])
        approach_errors_xy: list[tuple[float, float]] = []
        contact_position_error_mm = float("inf")
        orientation_error_deg = float("inf")
        approach_max_error_mm = float("inf")
        position_success = False
        orientation_success = False
        approach_success = False
        demo_clock = [0.0]
        attempts_used = 1
        lateral_error_window: list[float] = []
        demo_target_error_window: list[float] = []
        aborted_by_attempt_limit = False

        n_steps = int(REPLAY_DURATION * sim03.FPS)
        for frame_idx in range(n_steps + 1):
            replay_t = frame_idx * sim03.DT

            if moving_object and constraint_id is None:
                box_pos = moving_box_position(replay_t + trial_idx * 0.17, speed_cm_s)
                p.resetBasePositionAndOrientation(box_id, box_pos.tolist(), [0.0, 0.0, 0.0, 1.0])
                p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            elif not moving_object and constraint_id is None:
                p.resetBasePositionAndOrientation(box_id, STATIC_BOX_POS.tolist(), [0.0, 0.0, 0.0, 1.0])
                p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

            if moving_object:
                ee_pos_before_command = sim03.get_ee_position(panda_id)
                cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
                if frame_idx == 0:
                    target_pose = demo.poses.get_pose_at(demo_clock[0])
                    lateral_error_mm = 0.0
                else:
                    tracker.update(cloud, timestamp=replay_t)
                    alignment_pose = tracker.get_target_pose(
                        demo.poses,
                        t_demo=0.0,
                        tau=TAU,
                    )
                    alignment_target = np.asarray(alignment_pose[:3, 3], dtype=float)
                    lateral_error_mm = float(
                        np.linalg.norm((ee_pos_before_command - alignment_target)[:2]) * 1000.0
                    )
                    smoothed_lateral_error_mm = smooth_append(
                        lateral_error_window,
                        lateral_error_mm,
                        LATERAL_ERROR_SMOOTHING_FRAMES,
                    )
                    demo_clock[0] = min(demo_clock[0] + sim03.DT, demo.poses.duration)
                    previous_progress_sum = tracker._adaptive_progress_sum
                    adaptive_threshold_mm = (
                        DESCENT_GATE_THRESHOLD_M * 1000.0
                        if is_descent_phase(demo.poses, demo_clock[0])
                        else CONTACT_POSITION_TOLERANCE_M * 1000.0
                    )
                    target_pose, _ = tracker.get_target_pose_adaptive(
                        demo.poses,
                        t_demo=demo_clock,
                        lateral_error_mm=smoothed_lateral_error_mm,
                        threshold_mm=adaptive_threshold_mm,
                        tau=TAU,
                    )
                    frame_progress_rate = tracker._adaptive_progress_sum - previous_progress_sum
                    if DEBUG_ADAPTIVE_GATE and frame_idx % 30 == 0:
                        print(
                            f"adaptive debug frame={frame_idx:03d} "
                            f"lateral_error={lateral_error_mm:.1f}mm "
                            f"smoothed={smoothed_lateral_error_mm:.1f}mm "
                            f"threshold={adaptive_threshold_mm:.1f}mm "
                            f"progress_rate={frame_progress_rate:.2f} "
                            f"t_demo={demo_clock[0]:.3f}s "
                            f"duration={demo.poses.duration:.3f}s"
                        )
                t_demo = demo_clock[0]
            else:
                t_demo = min(replay_t, demo.poses.duration)
                target_pose = demo.poses.get_pose_at(t_demo)

            gripper_width = gripper_width_from_demo(demo, t_demo)
            if moving_object and constraint_id is None and gripper_width < OPEN_GRIPPER:
                ee_pos_before_close = sim03.get_ee_position(panda_id)
                box_pos_before_close = np.array(p.getBasePositionAndOrientation(box_id)[0], dtype=float)
                close_distance = float(np.linalg.norm((ee_pos_before_close - box_pos_before_close)[:2]))
                if close_distance > GRASP_CLOSE_BOX_EE_DISTANCE_M:
                    if DEBUG_CONTACT_WINDOW:
                        print(
                            f"abort attempt={attempts_used} frame={frame_idx:03d} "
                            f"close_distance={close_distance * 1000.0:.1f}mm "
                            f"t_demo={t_demo:.3f}s"
                        )
                    attempts_used += 1
                    if attempts_used > MAX_GRASP_ATTEMPTS:
                        aborted_by_attempt_limit = True
                        break

                    reset_adaptive_replay_state(tracker, demo_clock)
                    approach_errors_xy.clear()
                    lateral_error_window.clear()
                    demo_target_error_window.clear()
                    target_pose = tracker.get_target_pose(
                        demo.poses,
                        t_demo=0.0,
                        tau=TAU,
                    )
                    command_and_step(
                        panda_id,
                        np.asarray(target_pose[:3, 3], dtype=float),
                        ik_params,
                        OPEN_GRIPPER,
                    )
                    continue

            command_and_step(
                panda_id,
                np.asarray(target_pose[:3, 3], dtype=float),
                ik_params,
                gripper_width,
            )

            ee_pos = sim03.get_ee_position(panda_id)
            box_pos_after_step = np.array(p.getBasePositionAndOrientation(box_id)[0], dtype=float)
            target_pos_after_step = np.asarray(target_pose[:3, 3], dtype=float)
            ee_to_target_mm = float(np.linalg.norm((ee_pos - target_pos_after_step)[:2]) * 1000.0)
            horizontal_error = float(np.linalg.norm((ee_pos - box_pos_after_step)[:2]))
            demo_target_error = ee_to_target_mm / 1000.0
            smoothed_demo_target_error = smooth_append(
                demo_target_error_window,
                demo_target_error,
                LATERAL_ERROR_SMOOTHING_FRAMES,
            )
            approach_errors_xy.append((replay_t, smoothed_demo_target_error))

            if moving_object and DEBUG_CONTACT_WINDOW and should_log_contact_window(t_demo):
                print(
                    f"contact debug frame={frame_idx:03d} "
                    f"sim_t={replay_t:.3f}s t_demo={t_demo:.3f}s "
                    f"phase={gripper_phase(t_demo):7s} "
                    f"ee_pos=({ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}) "
                    f"target_pos=({target_pos_after_step[0]:.4f},{target_pos_after_step[1]:.4f},{target_pos_after_step[2]:.4f}) "
                    f"box_pos=({box_pos_after_step[0]:.4f},{box_pos_after_step[1]:.4f},{box_pos_after_step[2]:.4f}) "
                    f"ee_target_lat={ee_to_target_mm:.1f}mm "
                    f"ee_box_lat={horizontal_error * 1000.0:.1f}mm "
                    f"gripper_width={gripper_width:.3f}"
                )

            previous_constraint_id = constraint_id
            constraint_id = maybe_attach_box(panda_id, box_id, gripper_width, constraint_id)
            if previous_constraint_id is None and constraint_id is not None:
                contact_position_error_mm = ee_to_target_mm
                orientation_error_deg = gripper_axis_alignment_error_deg(panda_id)
                approach_success, approach_max_error_mm = approach_is_consistent(
                    approach_errors_xy,
                    contact_time=replay_t,
                )
                position_success = demo_target_error <= CONTACT_POSITION_TOLERANCE_M
                orientation_success = orientation_error_deg <= ORIENTATION_TOLERANCE_DEG

            box_z = float(p.getBasePositionAndOrientation(box_id)[0][2])
            max_box_z = max(max_box_z, box_z)
            final_box_z = box_z

            if record_gif and frame_idx % GIF_SAMPLE_STRIDE == 0:
                gif_frames.append(capture_video_frame(video_view, video_projection))

    finally:
        p.disconnect()

    if record_gif:
        save_demo_gif(gif_frames, plot_dir / "grasping_demo.gif")

    final_lift_m = final_box_z - float(STATIC_BOX_POS[2])
    max_lift_m = max_box_z - float(STATIC_BOX_POS[2])
    lift_success = final_lift_m > LIFT_SUCCESS_MARGIN
    success = lift_success and position_success and orientation_success and approach_success
    return TrialResult(
        success=success,
        final_lift_m=final_lift_m,
        max_lift_m=max_lift_m,
        progress_rate=tracker.progress_rate if moving_object else 1.0,
        lift_success=lift_success,
        position_success=position_success,
        orientation_success=orientation_success,
        approach_success=approach_success,
        attempt_limit_failure=aborted_by_attempt_limit,
        n_attempts=min(attempts_used, MAX_GRASP_ATTEMPTS),
        contact_position_error_mm=contact_position_error_mm,
        orientation_error_deg=orientation_error_deg,
        approach_max_error_mm=approach_max_error_mm,
    )


def trial_main_failure(result: TrialResult) -> str:
    """Return the primary failed success condition for one trial."""
    if result.success:
        return "none"
    if result.attempt_limit_failure:
        return "attempt_limit"
    if not result.lift_success:
        return "lift"
    if not result.position_success:
        return "position"
    if not result.orientation_success:
        return "orientation"
    if not result.approach_success:
        return "approach"
    return "unknown"


def initialize_raw_csv(out_path: Path) -> None:
    """Create the canonical raw-results CSV with header."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "condition",
                "speed_cm_s",
                "trial",
                "success",
                "lift_mm",
                "pos_err_mm",
                "ori_err_deg",
                "approach_mm",
                "progress_pct",
                "main_failure",
                "n_attempts",
            ]
        )


def append_raw_csv_row(
    out_path: Path,
    condition: str,
    speed_cm_s: float,
    trial_idx: int,
    result: TrialResult,
) -> None:
    """Append one trial's metrics to the canonical raw-results CSV."""
    with out_path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                condition,
                f"{speed_cm_s:.1f}",
                trial_idx + 1,
                int(result.success),
                f"{result.final_lift_m * 1000.0:.3f}",
                f"{result.contact_position_error_mm:.3f}",
                f"{result.orientation_error_deg:.3f}",
                f"{result.approach_max_error_mm:.3f}",
                f"{result.progress_rate * 100.0:.3f}",
                trial_main_failure(result),
                result.n_attempts,
            ]
        )


def run_all_trials() -> dict[str, dict[float, list[TrialResult]]]:
    """Run static baseline and moving-object grasp trials."""
    results: dict[str, dict[float, list[TrialResult]]] = {
        "Static baseline": {},
        "Moving object": {},
    }
    raw_csv_path = PROJECT_ROOT / "simulation" / "results" / "raw" / "10_grasping_final_seed42.csv"
    initialize_raw_csv(raw_csv_path)
    for speed in TEST_SPEEDS_CM_S:
        for label, moving in (("Static baseline", False), ("Moving object", True)):
            trial_results = []
            for trial_idx in range(N_TRIALS):
                record_gif = (
                    moving
                    and speed == RECORD_REPRESENTATIVE_SPEED_CM_S
                    and trial_idx == RECORD_REPRESENTATIVE_TRIAL
                )
                result = run_trial(speed, moving, trial_idx, record_gif=record_gif)
                trial_results.append(result)
                append_raw_csv_row(raw_csv_path, label, speed, trial_idx, result)
                print(
                    f"{label:15s} speed={speed:4.1f}cm/s "
                    f"trial={trial_idx + 1}/{N_TRIALS}: "
                    f"success={result.success}, "
                    f"lift={result.final_lift_m * 1000.0:.1f}mm, "
                    f"pos={result.contact_position_error_mm:.1f}mm, "
                    f"ori={result.orientation_error_deg:.1f}deg, "
                    f"approach={result.approach_max_error_mm:.1f}mm, "
                    f"progress={result.progress_rate * 100.0:.1f}%"
                )
            results[label][speed] = trial_results
    return results


def run_tracking_diagnostic(speed_cm_s: float) -> tuple[float, float]:
    """Track the moving box without demo replay and report steady-state lateral error."""
    client_id = p.connect(p.GUI if USE_GUI else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    try:
        box_id, panda_id, camera_matrices, ik_params = setup_scene()
        view_matrix, projection_matrix = camera_matrices
        reference_cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
        if reference_cloud.shape[0] < 20:
            raise RuntimeError(
                "Static reference depth segmentation produced too few box points: "
                f"{reference_cloud.shape[0]} < 20"
            )

        tracker = make_tracker(reference_cloud)
        static_target_pose = make_pose(STATIC_BOX_POS + START_OFFSET)
        lateral_errors_mm: list[float] = []

        n_steps = int(TRACKING_DIAGNOSTIC_DURATION * sim03.FPS)
        for frame_idx in range(n_steps + 1):
            sim_t = frame_idx * sim03.DT
            box_pos = moving_box_position(sim_t, speed_cm_s)
            p.resetBasePositionAndOrientation(box_id, box_pos.tolist(), [0.0, 0.0, 0.0, 1.0])
            p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

            cloud = sim03.capture_box_cloud(view_matrix, projection_matrix)
            if frame_idx == 0:
                target_pose = static_target_pose
            else:
                tracker.update(cloud, timestamp=sim_t)
                target_pose = tracker.get_target_pose(
                    make_static_tracking_demo(static_target_pose),
                    t_demo=0.0,
                    tau=TAU,
                )

            command_and_step(
                panda_id,
                np.asarray(target_pose[:3, 3], dtype=float),
                ik_params,
                OPEN_GRIPPER,
            )

            if sim_t > TRACKING_DIAGNOSTIC_STEADY_STATE_START:
                ee_pos = sim03.get_ee_position(panda_id)
                target_pos = np.asarray(target_pose[:3, 3], dtype=float)
                lateral_error_mm = float(np.linalg.norm((ee_pos - target_pos)[:2]) * 1000.0)
                lateral_errors_mm.append(lateral_error_mm)

    finally:
        p.disconnect()

    errors = np.asarray(lateral_errors_mm, dtype=float)
    return float(errors.mean()), float(errors.std(ddof=0))


def make_static_tracking_demo(target_pose: np.ndarray) -> DemoData:
    """Build a one-frame DemoData object for tracking-only diagnostics."""
    return DemoData(poses=[target_pose], timestamps=[0.0])


def run_tracking_diagnostics() -> dict[float, tuple[float, float]]:
    """Run tracking floor diagnostics for all configured speeds."""
    diagnostics: dict[float, tuple[float, float]] = {}
    print("Tracking floor diagnostic (no demo replay, t > 2s)")
    print("speed_cm_s | mean_lateral_error_mm | std_mm")
    print("-----------|-----------------------|-------")
    for speed in TEST_SPEEDS_CM_S:
        mean_mm, std_mm = run_tracking_diagnostic(speed)
        diagnostics[speed] = (mean_mm, std_mm)
        print(f"{speed:10.1f} | {mean_mm:21.1f} | {std_mm:5.1f}")
    return diagnostics


def failure_counts(trial_results: list[TrialResult]) -> dict[str, int]:
    """Count failed success subconditions."""
    return {
        "lift": sum(not result.lift_success for result in trial_results),
        "position": sum(not result.position_success for result in trial_results),
        "orientation": sum(not result.orientation_success for result in trial_results),
        "approach": sum(not result.approach_success for result in trial_results),
        "attempt_limit": sum(result.attempt_limit_failure for result in trial_results),
    }


def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return mean/std over finite entries only; no-contact trials use inf sentinels."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=0))


def print_summary_table(results: dict[str, dict[float, list[TrialResult]]]) -> None:
    """Print full success-rate table."""
    print()
    print("Grasping experiment summary")
    print("condition       | speed_cm_s | success_rate_% | progress_% | mean_lift_mm | pos_err_mm | ori_err_deg | approach_mm | main_failure")
    print("----------------|------------|----------------|------------|--------------|------------|-------------|-------------|-------------")
    for label in ("Static baseline", "Moving object"):
        for speed in TEST_SPEEDS_CM_S:
            trial_results = results[label][speed]
            successes = np.array([r.success for r in trial_results], dtype=float)
            progress_rates = np.array([r.progress_rate for r in trial_results], dtype=float) * 100.0
            lifts = np.array([r.final_lift_m for r in trial_results], dtype=float) * 1000.0
            pos_errors = np.array([r.contact_position_error_mm for r in trial_results], dtype=float)
            ori_errors = np.array([r.orientation_error_deg for r in trial_results], dtype=float)
            approach_errors = np.array([r.approach_max_error_mm for r in trial_results], dtype=float)
            pos_mean, pos_std = finite_mean_std(pos_errors)
            ori_mean, ori_std = finite_mean_std(ori_errors)
            approach_mean, approach_std = finite_mean_std(approach_errors)
            counts = failure_counts(trial_results)
            main_failure = max(counts.items(), key=lambda item: item[1])
            failure_label = main_failure[0] if main_failure[1] > 0 else "none"
            print(
                f"{label:15s} | {speed:10.1f} | "
                f"{successes.mean() * 100.0:6.1f} ± {successes.std(ddof=0) * 100.0:4.1f} | "
                f"{progress_rates.mean():6.1f} ± {progress_rates.std(ddof=0):4.1f} | "
                f"{lifts.mean():7.1f} ± {lifts.std(ddof=0):4.1f} | "
                f"{pos_mean:6.1f} ± {pos_std:4.1f} | "
                f"{ori_mean:6.1f} ± {ori_std:4.1f} | "
                f"{approach_mean:6.1f} ± {approach_std:4.1f} | "
                f"{failure_label}"
            )

    print()
    print("Failure diagnosis")
    print("condition       | speed_cm_s | lift | position | orientation | approach | attempt_limit")
    print("----------------|------------|------|----------|-------------|----------|--------------")
    for label in ("Static baseline", "Moving object"):
        for speed in TEST_SPEEDS_CM_S:
            counts = failure_counts(results[label][speed])
            print(
                f"{label:15s} | {speed:10.1f} | "
                f"{counts['lift']:4d} | {counts['position']:8d} | "
                f"{counts['orientation']:11d} | {counts['approach']:8d} | "
                f"{counts['attempt_limit']:13d}"
            )


def plot_success_rates(results: dict[str, dict[float, list[TrialResult]]], out_path: Path) -> None:
    """Plot static-baseline and moving-object grasp success rate by speed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    speeds = np.asarray(TEST_SPEEDS_CM_S, dtype=float)

    plt.figure(figsize=(8, 4.8))
    styles = {
        "Static baseline": ("tab:blue", "o"),
        "Moving object": ("tab:orange", "s"),
    }
    for label, (color, marker) in styles.items():
        success_rates = np.array(
            [
                np.mean([result.success for result in results[label][speed]]) * 100.0
                for speed in TEST_SPEEDS_CM_S
            ],
            dtype=float,
        )
        plt.plot(
            speeds,
            success_rates,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=label,
        )
        for x_value, y_value in zip(speeds, success_rates):
            plt.text(x_value, min(103.0, y_value + 2.0), f"{y_value:.0f}%",
                     ha="center", va="bottom", color=color)

    failure_notes = {
        2.0: "weak velocity\nsignal",
        4.0: "optimal\nrange",
        6.0: "optimal\nrange",
        8.0: "control bandwidth\nlimit",
    }
    for speed in speeds:
        note = failure_notes.get(float(speed))
        if note is not None:
            plt.text(
                speed,
                -8.0,
                note,
                ha="center",
                va="top",
                fontsize=8,
                color="0.25",
            )

    plt.xlabel("object speed (cm/s)")
    plt.ylabel("grasp success rate (%)")
    plt.title("Grasping Success Rate vs Object Speed")
    plt.xticks(speeds)
    plt.ylim(-18.0, 105.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    diagnostics = run_tracking_diagnostics()
    high_floor = {
        speed: mean_mm
        for speed, (mean_mm, _) in diagnostics.items()
        if mean_mm >= CONTACT_POSITION_TOLERANCE_M * 1000.0
    }
    if high_floor:
        for speed, mean_mm in high_floor.items():
            print(
                f"tracking floor too high for current gains: "
                f"{speed:.1f}cm/s mean={mean_mm:.1f}mm"
            )
        return

    results = run_all_trials()
    print_summary_table(results)
    plot_dir = PROJECT_ROOT / "simulation" / "results" / "plot"
    plot_success_rates(results, plot_dir / "grasping_success_rate.png")
    print("Saved simulation/results/raw/10_grasping_final_seed42.csv")
    print("Saved simulation/results/plot/grasping_success_rate.png")
    print("Saved simulation/results/plot/grasping_demo.gif")


if __name__ == "__main__":
    main()
