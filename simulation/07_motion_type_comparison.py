"""
Motion-type comparison for MT3 static-demo replay.

Same MT3 integration setup as simulation/05_mt3_integration.py.
Object speed is fixed at 5 cm/s, and three moving-object trajectories are tested:
1. Linear back-and-forth along x
2. Front-workspace semicircle
3. Smooth random walk from low-frequency sinusoids

For each motion type, run 3 trials and report:
- success rate: % frames with demo fidelity error < 30mm
- mean demo fidelity error
- standard deviation of demo fidelity error

Output:
- simulation/results/motion_type_comparison.png
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

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
VISUALIZE_REAL_TIME = False
OBJECT_SPEED_M_S = 0.05
N_TRIALS = 3
MOTION_TYPES = ("Linear", "Circular", "Random")
MotionFn = Callable[[float], tuple[float, float, float]]


class TrialResult(NamedTuple):
    success_rate_pct: float
    mean_demo_fidelity_error_mm: float
    std_demo_fidelity_error_mm: float


def _z() -> float:
    return float(mt3.sim03.TABLE_TOP_Z + mt3.sim03.BOX_HALF_EXTENT)


def make_linear_motion() -> MotionFn:
    """Back-and-forth along x with max speed 5 cm/s."""
    center_x, center_y = 0.50, 0.0
    amplitude = 0.15
    omega = OBJECT_SPEED_M_S / amplitude

    def motion(t: float) -> tuple[float, float, float]:
        return (
            center_x + amplitude * math.sin(omega * t),
            center_y,
            _z(),
        )

    return motion


def make_circular_motion() -> MotionFn:
    """Front-workspace semicircle with R=0.15m and speed 5 cm/s."""
    center_x, center_y = 0.50, 0.0
    radius = 0.15
    omega = OBJECT_SPEED_M_S / radius

    def motion(t: float) -> tuple[float, float, float]:
        sweep = math.pi
        phase = (omega * t) % (2.0 * sweep)
        if phase <= sweep:
            angle = -0.5 * math.pi + phase
        else:
            angle = -0.5 * math.pi + (2.0 * sweep - phase)
        return (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
            _z(),
        )

    return motion


def make_random_motion(trial_idx: int) -> MotionFn:
    """
    Smooth random walk in a 0.3m x 0.3m front workspace.

    The trajectory is a sum of low-frequency sinusoids, scaled so peak speed is
    at most 5 cm/s and the workspace remains within center +/- 0.15m.
    """
    rng = np.random.default_rng(1000 + trial_idx)
    center = np.array([0.50, 0.0], dtype=float)
    freqs = rng.uniform(0.025, 0.075, size=4)
    phases_x = rng.uniform(0.0, 2.0 * math.pi, size=4)
    phases_y = rng.uniform(0.0, 2.0 * math.pi, size=4)
    amps_x = rng.normal(0.0, 1.0, size=4)
    amps_y = rng.normal(0.0, 1.0, size=4)

    def raw_xy(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        x = sum(a * np.sin(2.0 * math.pi * f * t + ph)
                for a, f, ph in zip(amps_x, freqs, phases_x))
        y = sum(a * np.sin(2.0 * math.pi * f * t + ph)
                for a, f, ph in zip(amps_y, freqs, phases_y))
        return np.column_stack([x, y])

    def raw_dxy(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        dx = sum(a * 2.0 * math.pi * f * np.cos(2.0 * math.pi * f * t + ph)
                 for a, f, ph in zip(amps_x, freqs, phases_x))
        dy = sum(a * 2.0 * math.pi * f * np.cos(2.0 * math.pi * f * t + ph)
                 for a, f, ph in zip(amps_y, freqs, phases_y))
        return np.column_stack([dx, dy])

    sample_t = np.linspace(0.0, mt3.REPLAY_DURATION, 1000)
    raw = raw_xy(sample_t)
    raw -= raw.mean(axis=0, keepdims=True)
    max_abs = float(np.max(np.abs(raw)))
    max_speed = float(np.max(np.linalg.norm(raw_dxy(sample_t), axis=1)))
    scale = min(0.15 / max(max_abs, 1e-9), OBJECT_SPEED_M_S / max(max_speed, 1e-9))

    def motion(t: float) -> tuple[float, float, float]:
        xy = center + scale * (raw_xy(np.array([t]))[0] - raw.mean(axis=0))
        xy = np.clip(xy, center - 0.15, center + 0.15)
        return float(xy[0]), float(xy[1]), _z()

    return motion


def make_motion(motion_type: str, trial_idx: int) -> MotionFn:
    if motion_type == "Linear":
        return make_linear_motion()
    if motion_type == "Circular":
        return make_circular_motion()
    if motion_type == "Random":
        return make_random_motion(trial_idx)
    raise ValueError(f"Unknown motion type: {motion_type}")


def run_single_trial(motion_type: str, trial_idx: int) -> TrialResult:
    """Run one MT3 replay trial for a motion type."""
    original_box_position = mt3.sim03.box_position
    mt3.sim03.box_position = make_motion(motion_type, trial_idx)

    connection_mode = p.GUI if USE_GUI else p.DIRECT
    client_id = p.connect(connection_mode)
    if client_id < 0:
        mt3.sim03.box_position = original_box_position
        raise RuntimeError("Failed to connect to PyBullet")

    try:
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

        estimator_config = EstimatorConfig(
            min_points=20,
            use_pca_angle=False,
            z_plane_threshold=0.03,
        )
        tracker = DynamicAlignmentTracker(
            tau=mt3.TAU,
            estimator_config=estimator_config,
            kalman_R_diag=np.array([0.004, 0.004, math.radians(30.0)]),
            init_vel_cov=0.10,
        )
        tracker.init(reference_cloud, initial_theta=0.0, timestamp=0.0)

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
            tracker.update(cloud, timestamp=replay_t)
            target_pose = tracker.get_target_pose(demo_data, t_demo=t_demo, tau=mt3.TAU)
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
            demo_fidelity_errors.append(demo_fidelity)

            if USE_GUI and VISUALIZE_REAL_TIME:
                time.sleep(mt3.sim03.DT)

    finally:
        p.disconnect()
        mt3.sim03.box_position = original_box_position

    fidelity_arr = np.asarray(demo_fidelity_errors, dtype=float)
    return TrialResult(
        success_rate_pct=float(np.mean(fidelity_arr < mt3.SUCCESS_THRESHOLD_M) * 100.0),
        mean_demo_fidelity_error_mm=float(np.mean(fidelity_arr) * 1000.0),
        std_demo_fidelity_error_mm=float(np.std(fidelity_arr, ddof=0) * 1000.0),
    )


def run_all_trials() -> dict[str, list[TrialResult]]:
    """Run all motion types and trials."""
    results: dict[str, list[TrialResult]] = {}
    for motion_type in MOTION_TYPES:
        trial_results = []
        for trial_idx in range(N_TRIALS):
            result = run_single_trial(motion_type, trial_idx)
            trial_results.append(result)
            print(
                f"{motion_type:8s} trial={trial_idx + 1}/{N_TRIALS}: "
                f"success={result.success_rate_pct:.1f}%, "
                f"fidelity={result.mean_demo_fidelity_error_mm:.1f}±"
                f"{result.std_demo_fidelity_error_mm:.1f}mm"
            )
        results[motion_type] = trial_results
    return results


def print_summary_table(results: dict[str, list[TrialResult]]) -> None:
    """Print aggregate metrics to console."""
    print()
    print("Motion type comparison summary")
    print("motion_type | success_rate_% | demo_fidelity_mm | within_trial_std_mm")
    print("------------|----------------|------------------|--------------------")
    for motion_type, trial_results in results.items():
        success = np.array([r.success_rate_pct for r in trial_results], dtype=float)
        mean_fidelity = np.array([r.mean_demo_fidelity_error_mm for r in trial_results], dtype=float)
        std_fidelity = np.array([r.std_demo_fidelity_error_mm for r in trial_results], dtype=float)
        print(
            f"{motion_type:11s} | "
            f"{success.mean():6.1f} ± {success.std(ddof=0):4.1f} | "
            f"{mean_fidelity.mean():7.1f} ± {mean_fidelity.std(ddof=0):4.1f} | "
            f"{std_fidelity.mean():7.1f} ± {std_fidelity.std(ddof=0):4.1f}"
        )


def _bar_color(success_rate: float) -> str:
    if success_rate > 90.0:
        return "tab:green"
    if success_rate >= 80.0:
        return "gold"
    return "tab:red"


def plot_motion_type_comparison(
    results: dict[str, list[TrialResult]],
    out_path: Path,
) -> None:
    """Plot success rate by motion type."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(MOTION_TYPES)
    means = np.array(
        [np.mean([r.success_rate_pct for r in results[label]]) for label in labels],
        dtype=float,
    )
    stds = np.array(
        [np.std([r.success_rate_pct for r in results[label]], ddof=0) for label in labels],
        dtype=float,
    )
    colors = [_bar_color(float(mean)) for mean in means]

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.88)
    plt.axhline(
        80.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="acceptable performance threshold (80%)",
    )
    for idx, mean in enumerate(means):
        plt.text(idx, min(102.0, mean + 2.0), f"{mean:.1f}%", ha="center", va="bottom")
    plt.xlabel("motion type")
    plt.ylabel("success rate (%)")
    plt.title("Success Rate by Motion Type (v=5 cm/s)")
    plt.ylim(0.0, 105.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    results = run_all_trials()
    print_summary_table(results)
    out_path = PROJECT_ROOT / "simulation" / "results" / "motion_type_comparison.png"
    plot_motion_type_comparison(results, out_path)
    print("Saved simulation/results/motion_type_comparison.png")


if __name__ == "__main__":
    main()
