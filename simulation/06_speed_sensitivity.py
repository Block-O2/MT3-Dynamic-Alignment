"""
Speed sensitivity experiment for MT3 dynamic replay.

Same setup as simulation/05_mt3_integration.py, but sweeps object speed:
v = [2, 4, 6, 8, 10] cm/s with R = 0.15m and omega = v / R.

For each speed, run 3 trials and report:
- Success rate: % replay frames with demo fidelity error < 30mm
- Mean tracking error: tracker-estimated object XY vs ground-truth object XY
- Mean demo fidelity error: EE deviation from T_WE_demo in object frame

Output:
- simulation/results/speed_sensitivity.png
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

SPEEDS_CM_S = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
N_TRIALS = 3


class TrialResult(NamedTuple):
    success_rate_pct: float
    mean_tracking_error_mm: float
    mean_demo_fidelity_error_mm: float


def run_single_trial(speed_cm_s: float) -> TrialResult:
    """Run one MT3 replay trial at the requested object speed."""
    original_omega = float(mt3.sim03.OMEGA)
    mt3.sim03.OMEGA = (speed_cm_s / 100.0) / mt3.sim03.RADIUS

    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        mt3.sim03.OMEGA = original_omega
        raise RuntimeError("Failed to connect to PyBullet DIRECT")

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
            state = tracker.update(cloud, timestamp=replay_t)
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

            estimated_delta_xy = np.array([state.delta_x, state.delta_y], dtype=float)
            true_delta_xy = box_pos[:2] - mt3.STATIC_BOX_POS[:2]
            tracking_error = float(np.linalg.norm(estimated_delta_xy - true_delta_xy))

            tracking_errors.append(tracking_error)
            demo_fidelity_errors.append(demo_fidelity)

    finally:
        p.disconnect()
        mt3.sim03.OMEGA = original_omega

    tracking_arr = np.asarray(tracking_errors, dtype=float)
    fidelity_arr = np.asarray(demo_fidelity_errors, dtype=float)
    return TrialResult(
        success_rate_pct=float(np.mean(fidelity_arr < mt3.SUCCESS_THRESHOLD_M) * 100.0),
        mean_tracking_error_mm=float(np.mean(tracking_arr) * 1000.0),
        mean_demo_fidelity_error_mm=float(np.mean(fidelity_arr) * 1000.0),
    )


def run_sweep() -> dict[float, list[TrialResult]]:
    """Run all speeds and trials."""
    results: dict[float, list[TrialResult]] = {}
    for speed_cm_s in SPEEDS_CM_S:
        trial_results = []
        for trial_idx in range(N_TRIALS):
            result = run_single_trial(float(speed_cm_s))
            trial_results.append(result)
            print(
                f"speed={speed_cm_s:.0f} cm/s trial={trial_idx + 1}/{N_TRIALS}: "
                f"success={result.success_rate_pct:.1f}%, "
                f"tracking={result.mean_tracking_error_mm:.1f}mm, "
                f"fidelity={result.mean_demo_fidelity_error_mm:.1f}mm"
            )
        results[float(speed_cm_s)] = trial_results
    return results


def print_summary_table(results: dict[float, list[TrialResult]]) -> None:
    """Print aggregate metrics to console."""
    print()
    print("Speed sensitivity summary")
    print("speed_cm_s | success_rate_% | tracking_error_mm | demo_fidelity_mm")
    print("-----------|----------------|-------------------|-----------------")
    for speed_cm_s, trial_results in results.items():
        success = np.array([r.success_rate_pct for r in trial_results], dtype=float)
        tracking = np.array([r.mean_tracking_error_mm for r in trial_results], dtype=float)
        fidelity = np.array([r.mean_demo_fidelity_error_mm for r in trial_results], dtype=float)
        print(
            f"{speed_cm_s:10.0f} | "
            f"{success.mean():6.1f} ± {success.std(ddof=0):4.1f} | "
            f"{tracking.mean():7.1f} ± {tracking.std(ddof=0):4.1f} | "
            f"{fidelity.mean():7.1f} ± {fidelity.std(ddof=0):4.1f}"
        )


def plot_speed_sensitivity(results: dict[float, list[TrialResult]], out_path: Path) -> None:
    """Plot success rate against object speed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speeds = np.array(sorted(results.keys()), dtype=float)
    success_means = np.array(
        [np.mean([r.success_rate_pct for r in results[speed]]) for speed in speeds],
        dtype=float,
    )
    success_stds = np.array(
        [np.std([r.success_rate_pct for r in results[speed]], ddof=0) for speed in speeds],
        dtype=float,
    )

    plt.figure(figsize=(8, 4.8))
    plt.errorbar(
        speeds,
        success_means,
        yerr=success_stds,
        color="tab:blue",
        marker="o",
        linewidth=2,
        capsize=4,
        label="success rate across 3 trials",
    )
    plt.axvline(
        10.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="designed operating range (≤10 cm/s)",
    )
    plt.xlabel("object speed (cm/s)")
    plt.ylabel("success rate (%)")
    plt.title("System Performance vs Object Speed")
    plt.ylim(0.0, 105.0)
    plt.xticks(speeds)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    results = run_sweep()
    print_summary_table(results)
    out_path = PROJECT_ROOT / "simulation" / "results" / "speed_sensitivity.png"
    plot_speed_sensitivity(results, out_path)
    print("Saved simulation/results/speed_sensitivity.png")


if __name__ == "__main__":
    main()
