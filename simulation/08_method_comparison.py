"""
Method comparison for MT3 static-demo replay.

Same setup as simulation/05_mt3_integration.py:
- circular motion, R=0.15m
- object speed 5 cm/s
- 10 trials per method

Methods:
A. Ours: DynamicAlignmentTracker with tau=0.1
B. Velocity feedforward: raw centroid + last observed velocity * tau, no Kalman
C. Pure reactive: DynamicAlignmentTracker with tau=0.0

Outputs:
- simulation/results/method_comparison_bar.png
- simulation/results/method_comparison_time.png
"""

from __future__ import annotations

import importlib.util
import math
import os
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_alignment.pose_estimator import EstimatorConfig
from dynamic_alignment.tracker import DynamicAlignmentTracker
from dynamic_alignment.types import DemoData


def load_mt3_module():
    """Load simulation/05_mt3_integration.py despite its digit-prefixed filename."""
    module_path = Path(__file__).resolve().with_name("05_mt3_integration.py")
    spec = importlib.util.spec_from_file_location("mt3_integration_sim", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mt3 = load_mt3_module()

USE_GUI = False
N_TRIALS = 10
OBJECT_SPEED_M_S = 0.05
LATENCY_FRAMES = 3
OBSERVATION_POSITION_NOISE_STD = 0.004
WARMUP_SECONDS = 1.0
METHODS = (
    "Ours (Kalman + prediction)",
    "Velocity feedforward",
    "Pure reactive (no prediction)",
)
METHOD_COLORS = {
    "Ours (Kalman + prediction)": "tab:green",
    "Velocity feedforward": "tab:blue",
    "Pure reactive (no prediction)": "tab:red",
}


class TrialResult(NamedTuple):
    success_rate_pct: float
    mean_tracking_error_mm: float
    mean_demo_fidelity_error_mm: float
    steady_state_rms_error_mm: float
    demo_fidelity_errors: np.ndarray
    times: np.ndarray


def circular_motion(t: float) -> tuple[float, float, float]:
    """Front-workspace semicircle with R=0.15m and speed 5 cm/s."""
    center_x, center_y = 0.50, 0.0
    radius = 0.15
    omega = OBJECT_SPEED_M_S / radius
    sweep = math.pi
    phase = (omega * t) % (2.0 * sweep)
    if phase <= sweep:
        angle = -0.5 * math.pi + phase
    else:
        angle = -0.5 * math.pi + (2.0 * sweep - phase)
    return (
        center_x + radius * math.cos(angle),
        center_y + radius * math.sin(angle),
        float(mt3.sim03.TABLE_TOP_Z + mt3.sim03.BOX_HALF_EXTENT),
    )


def cloud_centroid_xy(cloud: np.ndarray) -> np.ndarray | None:
    """Return segmented point-cloud centroid in XY, or None if invalid."""
    if cloud.ndim != 2 or cloud.shape[0] < 20 or cloud.shape[1] != 3:
        return None
    z_med = float(np.median(cloud[:, 2]))
    mask = np.abs(cloud[:, 2] - z_med) <= 0.03
    filtered = cloud[mask]
    if filtered.shape[0] < 20:
        return None
    return filtered[:, :2].mean(axis=0)


def make_delta_transform(delta_xy: np.ndarray) -> np.ndarray:
    """Create an SE(3) transform with XY translation only."""
    transform = np.eye(4)
    transform[0, 3] = float(delta_xy[0])
    transform[1, 3] = float(delta_xy[1])
    return transform


def add_observation_noise(cloud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add D415-scale per-observation position noise before any method sees it."""
    if cloud.size == 0:
        return cloud
    noisy_cloud = cloud.copy()
    noisy_cloud[:, :2] += rng.normal(0.0, OBSERVATION_POSITION_NOISE_STD, size=(1, 2))
    return noisy_cloud


def setup_trial_scene() -> tuple[int, int, tuple[list[float], list[float]], tuple, np.ndarray, DemoData]:
    """Set up scene, capture reference cloud, and record static demo."""
    box_id, panda_id, camera_matrices, ik_params = mt3.setup_scene()
    view_matrix, projection_matrix = camera_matrices

    reference_cloud = mt3.sim03.capture_box_cloud(view_matrix, projection_matrix)
    if reference_cloud.shape[0] < 20:
        raise RuntimeError(
            "Static reference depth segmentation produced too few box points: "
            f"{reference_cloud.shape[0]} < 20"
        )

    for _ in range(60):
        mt3.command_and_step(
            panda_id,
            mt3.STATIC_BOX_POS + mt3.DEMO_START_OFFSET,
            ik_params,
        )
    demo_data = mt3.record_static_demo(panda_id, ik_params)
    return box_id, panda_id, camera_matrices, ik_params, reference_cloud, demo_data


def make_tracker(tau: float, reference_cloud: np.ndarray) -> DynamicAlignmentTracker:
    """Create and initialize a DynamicAlignmentTracker."""
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
    tracker.init(reference_cloud, initial_theta=0.0, timestamp=0.0)
    return tracker


def run_single_trial(method: str, trial_idx: int) -> TrialResult:
    """Run one method trial and return aggregate metrics plus time-series error."""
    original_box_position = mt3.sim03.box_position
    mt3.sim03.box_position = circular_motion
    rng = np.random.default_rng(2026 + trial_idx)
    connection_mode = p.GUI if USE_GUI else p.DIRECT
    client_id = p.connect(connection_mode)
    if client_id < 0:
        mt3.sim03.box_position = original_box_position
        raise RuntimeError("Failed to connect to PyBullet")

    try:
        box_id, panda_id, camera_matrices, ik_params, reference_cloud, demo_data = setup_trial_scene()
        view_matrix, projection_matrix = camera_matrices
        reference_centroid = cloud_centroid_xy(reference_cloud)
        if reference_centroid is None:
            raise RuntimeError("Reference centroid unavailable for feedforward baseline")

        tracker: DynamicAlignmentTracker | None = None
        if method == "Ours (Kalman + prediction)":
            tracker = make_tracker(tau=0.1, reference_cloud=reference_cloud)
        elif method == "Pure reactive (no prediction)":
            tracker = make_tracker(tau=0.0, reference_cloud=reference_cloud)
        elif method != "Velocity feedforward":
            raise ValueError(f"Unknown method: {method}")

        previous_raw_xy: np.ndarray | None = None
        previous_time: float | None = None
        previous_velocity = np.zeros(2, dtype=float)
        observation_buffer: list[tuple[float, np.ndarray]] = []

        times: list[float] = []
        tracking_errors: list[float] = []
        demo_fidelity_errors: list[float] = []

        n_steps = int(mt3.REPLAY_DURATION * mt3.sim03.FPS)
        for frame_idx in range(1, n_steps + 1):
            replay_t = frame_idx * mt3.sim03.DT
            t_demo = replay_t % demo_data.duration
            box_pos = np.array(mt3.sim03.box_position(replay_t), dtype=float)
            p.resetBasePositionAndOrientation(
                box_id,
                box_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
            )

            cloud = mt3.sim03.capture_box_cloud(view_matrix, projection_matrix)
            observation_buffer.append((replay_t, cloud))
            if len(observation_buffer) <= LATENCY_FRAMES:
                target_pose = demo_data.get_pose_at(t_demo)
                mt3.command_and_step(
                    panda_id,
                    np.asarray(target_pose[:3, 3], dtype=float),
                    ik_params,
                )
                continue

            delayed_t, delayed_cloud = observation_buffer.pop(0)
            delayed_cloud = add_observation_noise(delayed_cloud, rng)

            if method == "Velocity feedforward":
                raw_xy = cloud_centroid_xy(delayed_cloud)
                if raw_xy is None:
                    raw_xy = previous_raw_xy if previous_raw_xy is not None else reference_centroid
                if previous_raw_xy is not None and previous_time is not None:
                    dt = max(delayed_t - previous_time, 1e-9)
                    previous_velocity = (raw_xy - previous_raw_xy) / dt
                predicted_xy = raw_xy + previous_velocity * mt3.TAU
                estimated_delta_xy = predicted_xy - reference_centroid
                target_pose = make_delta_transform(estimated_delta_xy) @ demo_data.get_pose_at(t_demo)
                previous_raw_xy = raw_xy.copy()
                previous_time = delayed_t
            else:
                assert tracker is not None
                state = tracker.update(delayed_cloud, timestamp=delayed_t)
                tau = 0.1 if method == "Ours (Kalman + prediction)" else 0.0
                target_pose = tracker.get_target_pose(demo_data, t_demo=t_demo, tau=tau)
                estimated_delta_xy = np.array([state.delta_x, state.delta_y], dtype=float)

            mt3.command_and_step(
                panda_id,
                np.asarray(target_pose[:3, 3], dtype=float),
                ik_params,
            )

            ee_pos = mt3.sim03.get_ee_position(panda_id)
            demo_pose = demo_data.get_pose_at(t_demo)
            desired_relative = demo_pose[:3, 3] - mt3.STATIC_BOX_POS
            actual_relative = ee_pos - box_pos
            demo_fidelity = float(np.linalg.norm(actual_relative - desired_relative))

            true_delta_xy = box_pos[:2] - mt3.STATIC_BOX_POS[:2]
            tracking_error = float(np.linalg.norm(estimated_delta_xy - true_delta_xy))

            times.append(replay_t)
            tracking_errors.append(tracking_error)
            demo_fidelity_errors.append(demo_fidelity)

    finally:
        p.disconnect()
        mt3.sim03.box_position = original_box_position

    tracking_arr = np.asarray(tracking_errors, dtype=float)
    fidelity_arr = np.asarray(demo_fidelity_errors, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    steady_mask = times_arr > WARMUP_SECONDS
    if not np.any(steady_mask):
        steady_mask = np.ones_like(times_arr, dtype=bool)
    steady_fidelity = fidelity_arr[steady_mask]
    return TrialResult(
        success_rate_pct=float(np.mean(steady_fidelity < mt3.SUCCESS_THRESHOLD_M) * 100.0),
        mean_tracking_error_mm=float(np.mean(tracking_arr) * 1000.0),
        mean_demo_fidelity_error_mm=float(np.mean(fidelity_arr) * 1000.0),
        steady_state_rms_error_mm=float(np.sqrt(np.mean(steady_fidelity ** 2)) * 1000.0),
        demo_fidelity_errors=fidelity_arr,
        times=times_arr,
    )


def run_all_trials() -> dict[str, list[TrialResult]]:
    """Run all methods and trials."""
    results: dict[str, list[TrialResult]] = {}
    for method in METHODS:
        method_results = []
        for trial_idx in range(N_TRIALS):
            result = run_single_trial(method, trial_idx)
            method_results.append(result)
            print(
                f"{method:30s} trial={trial_idx + 1}/{N_TRIALS}: "
                f"success={result.success_rate_pct:.1f}%, "
                f"tracking={result.mean_tracking_error_mm:.1f}mm, "
                f"fidelity={result.mean_demo_fidelity_error_mm:.1f}mm, "
                f"steady_rms={result.steady_state_rms_error_mm:.1f}mm"
            )
        results[method] = method_results
    return results


def print_summary_table(results: dict[str, list[TrialResult]]) -> None:
    """Print mean ± std summary table."""
    print()
    print("Method comparison summary")
    print("method                         | success_rate_% | tracking_error_mm | demo_fidelity_mm | steady_rms_mm")
    print("-------------------------------|----------------|-------------------|------------------|--------------")
    for method in METHODS:
        trial_results = results[method]
        success = np.array([r.success_rate_pct for r in trial_results], dtype=float)
        tracking = np.array([r.mean_tracking_error_mm for r in trial_results], dtype=float)
        fidelity = np.array([r.mean_demo_fidelity_error_mm for r in trial_results], dtype=float)
        steady_rms = np.array([r.steady_state_rms_error_mm for r in trial_results], dtype=float)
        print(
            f"{method:30s} | "
            f"{success.mean():6.1f} ± {success.std(ddof=0):4.1f} | "
            f"{tracking.mean():7.1f} ± {tracking.std(ddof=0):4.1f} | "
            f"{fidelity.mean():7.1f} ± {fidelity.std(ddof=0):4.1f} | "
            f"{steady_rms.mean():7.1f} ± {steady_rms.std(ddof=0):4.1f}"
        )


def plot_bar(results: dict[str, list[TrialResult]], out_path: Path) -> None:
    """Plot steady-state success rate and RMS error with trial error bars."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success_means = np.array(
        [np.mean([r.success_rate_pct for r in results[method]]) for method in METHODS],
        dtype=float,
    )
    success_stds = np.array(
        [np.std([r.success_rate_pct for r in results[method]], ddof=0) for method in METHODS],
        dtype=float,
    )
    rms_means = np.array(
        [np.mean([r.steady_state_rms_error_mm for r in results[method]]) for method in METHODS],
        dtype=float,
    )
    rms_stds = np.array(
        [np.std([r.steady_state_rms_error_mm for r in results[method]], ddof=0) for method in METHODS],
        dtype=float,
    )
    colors = [METHOD_COLORS[method] for method in METHODS]
    x = np.arange(len(METHODS))

    fig, (ax_success, ax_rms) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    ax_success.bar(
        x,
        success_means,
        yerr=success_stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        alpha=0.88,
    )
    ax_success.axhline(80.0, color="black", linestyle="--", linewidth=1.5,
                       label="success threshold (80%)")
    for idx, mean in enumerate(success_means):
        ax_success.text(idx, min(102.0, mean + 2.0), f"{mean:.1f}%",
                        ha="center", va="bottom")
    ax_success.set_ylabel("success rate after 1s (%)")
    ax_success.set_title("Method Comparison (10 trials each)")
    ax_success.set_ylim(0.0, 105.0)
    ax_success.grid(True, axis="y", alpha=0.3)
    ax_success.legend(loc="lower right", framealpha=0.95)

    ax_rms.bar(
        x,
        rms_means,
        yerr=rms_stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        alpha=0.88,
    )
    for idx, mean in enumerate(rms_means):
        ax_rms.text(idx, mean + max(1.0, 0.04 * float(np.max(rms_means))),
                    f"{mean:.1f}mm", ha="center", va="bottom")
    ax_rms.set_ylabel("steady-state RMS error (mm)")
    ax_rms.set_xticks(x)
    ax_rms.set_xticklabels(METHODS, rotation=12, ha="right")
    ax_rms.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_time_series(results: dict[str, list[TrialResult]], out_path: Path) -> None:
    """Plot representative trial demo-fidelity error over time for all methods."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    for method in METHODS:
        result = results[method][0]
        plt.plot(
            result.times,
            result.demo_fidelity_errors * 1000.0,
            color=METHOD_COLORS[method],
            linewidth=2,
            label=method,
        )
    plt.axhline(30.0, color="black", linestyle="--", linewidth=1.5,
                label="success threshold (30mm)")
    plt.xlabel("time (s)")
    plt.ylabel("demo fidelity error (mm)")
    plt.title("Method Comparison: Representative Trial Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    results = run_all_trials()
    print_summary_table(results)
    out_dir = PROJECT_ROOT / "simulation" / "results"
    plot_bar(results, out_dir / "method_comparison_bar.png")
    plot_time_series(results, out_dir / "method_comparison_time.png")
    print("Saved simulation/results/method_comparison_bar.png")
    print("Saved simulation/results/method_comparison_time.png")


if __name__ == "__main__":
    main()
