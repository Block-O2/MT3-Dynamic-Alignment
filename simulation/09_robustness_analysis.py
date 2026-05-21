"""
Robustness analysis for MT3 static-demo replay.

Base setup matches simulation/08_method_comparison.py:
- circular motion, R=0.15m, v=5 cm/s
- 100ms latency
- 4mm observation position noise
- 10 trials per method

Progressive conditions:
1. Baseline: fixed 100ms latency, 4mm noise
2. +Jitter: latency ~ N(100ms, 20ms)
3. +Dropout: plus 8% random frame dropout
4. +Outliers: plus 2% chance of 80mm position spike

Metric:
- steady-state RMS demo-fidelity error in mm, t > 1s

Output:
- simulation/results/robustness_analysis.png
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

from dynamic_alignment.tracker import DynamicAlignmentTracker


def load_method_comparison_module():
    """Load simulation/08_method_comparison.py despite its digit-prefixed filename."""
    module_path = Path(__file__).resolve().with_name("08_method_comparison.py")
    spec = importlib.util.spec_from_file_location("method_comparison_sim", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sim08 = load_method_comparison_module()
mt3 = sim08.mt3

USE_GUI = False
N_TRIALS = 10
WARMUP_SECONDS = 1.0
BASE_LATENCY_S = 0.100
JITTER_STD_S = 0.020
DROPOUT_PROBABILITY = 0.08
OUTLIER_PROBABILITY = 0.02
OUTLIER_MAGNITUDE_M = 0.080
TIME_EPSILON = 1e-9
PRINT_LATENCY_DIAGNOSTICS = True

METHODS = sim08.METHODS
METHOD_COLORS = sim08.METHOD_COLORS


class RobustnessCondition(NamedTuple):
    name: str
    use_jitter: bool
    dropout_probability: float
    outlier_probability: float


CONDITIONS = (
    RobustnessCondition("Baseline", False, 0.0, 0.0),
    RobustnessCondition("+Jitter", True, 0.0, 0.0),
    RobustnessCondition("+Dropout", True, DROPOUT_PROBABILITY, 0.0),
    RobustnessCondition("+Outliers", True, DROPOUT_PROBABILITY, OUTLIER_PROBABILITY),
)


class TrialResult(NamedTuple):
    steady_state_rms_error_mm: float
    mean_demo_fidelity_error_mm: float
    success_rate_pct: float


def sample_latency(condition: RobustnessCondition, rng: np.random.Generator) -> float:
    """Sample nonnegative observation latency for the current frame."""
    if not condition.use_jitter:
        return BASE_LATENCY_S
    return float(np.clip(rng.normal(BASE_LATENCY_S, JITTER_STD_S), 0.0, 0.200))


def choose_delayed_observation(
    observation_buffer: list[tuple[float, np.ndarray]],
    target_time: float,
) -> tuple[float, np.ndarray] | None:
    """Return newest observation with timestamp <= target_time."""
    chosen_idx = None
    for idx, (obs_time, _) in enumerate(observation_buffer):
        if obs_time <= target_time + TIME_EPSILON:
            chosen_idx = idx
        else:
            break

    if chosen_idx is None:
        return None

    delayed = observation_buffer[chosen_idx]
    del observation_buffer[:chosen_idx + 1]
    return delayed


def corrupt_observation_cloud(
    cloud: np.ndarray,
    condition: RobustnessCondition,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply sensor noise and optional outlier spike before any method sees the observation."""
    noisy_cloud = sim08.add_observation_noise(cloud, rng)
    if noisy_cloud.size == 0 or rng.random() >= condition.outlier_probability:
        return noisy_cloud

    angle = rng.uniform(0.0, 2.0 * math.pi)
    spike = OUTLIER_MAGNITUDE_M * np.array([math.cos(angle), math.sin(angle)])
    outlier_cloud = noisy_cloud.copy()
    outlier_cloud[:, :2] += spike.reshape(1, 2)
    return outlier_cloud


def make_tracker(method: str, reference_cloud: np.ndarray) -> DynamicAlignmentTracker | None:
    """Create tracker for Kalman-based methods."""
    if method == "Ours (Kalman + prediction)":
        return sim08.make_tracker(tau=0.1, reference_cloud=reference_cloud)
    if method == "Pure reactive (no prediction)":
        return sim08.make_tracker(tau=0.0, reference_cloud=reference_cloud)
    if method == "Velocity feedforward":
        return None
    raise ValueError(f"Unknown method: {method}")


def run_single_trial(
    condition: RobustnessCondition,
    method: str,
    trial_idx: int,
    latency_samples: list[float] | None = None,
) -> TrialResult:
    """Run one robustness trial and return steady-state metrics."""
    original_box_position = mt3.sim03.box_position
    mt3.sim03.box_position = sim08.circular_motion
    rng = np.random.default_rng(9000 + 1000 * CONDITIONS.index(condition) + 100 * METHODS.index(method) + trial_idx)

    client_id = p.connect(p.GUI if USE_GUI else p.DIRECT)
    if client_id < 0:
        mt3.sim03.box_position = original_box_position
        raise RuntimeError("Failed to connect to PyBullet")

    try:
        box_id, panda_id, camera_matrices, ik_params, reference_cloud, demo_data = sim08.setup_trial_scene()
        view_matrix, projection_matrix = camera_matrices
        reference_centroid = sim08.cloud_centroid_xy(reference_cloud)
        if reference_centroid is None:
            raise RuntimeError("Reference centroid unavailable for feedforward baseline")

        tracker = make_tracker(method, reference_cloud)
        observation_buffer: list[tuple[float, np.ndarray]] = []

        previous_raw_xy: np.ndarray | None = None
        previous_time: float | None = None
        previous_velocity = np.zeros(2, dtype=float)
        frozen_delta_xy = np.zeros(2, dtype=float)

        times: list[float] = []
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

            latency = sample_latency(condition, rng)
            if latency_samples is not None:
                latency_samples.append(latency)
            delayed = choose_delayed_observation(observation_buffer, replay_t - latency)
            if delayed is None:
                if method == "Ours (Kalman + prediction)":
                    assert tracker is not None
                    last_timestamp = tracker._last_timestamp if tracker._last_timestamp is not None else 0.0
                    effective_tau = mt3.TAU + max(0.0, replay_t - float(last_timestamp))
                    target_pose = tracker.get_target_pose(
                        demo_data,
                        t_demo=t_demo,
                        tau=effective_tau,
                    )
                else:
                    target_pose = sim08.make_delta_transform(frozen_delta_xy) @ demo_data.get_pose_at(t_demo)
                mt3.command_and_step(
                    panda_id,
                    np.asarray(target_pose[:3, 3], dtype=float),
                    ik_params,
                )
                continue

            delayed_t, delayed_cloud = delayed
            is_dropout = rng.random() < condition.dropout_probability

            if method == "Ours (Kalman + prediction)":
                assert tracker is not None
                cloud_for_method = (
                    np.empty((0, 3), dtype=float)
                    if is_dropout
                    else corrupt_observation_cloud(delayed_cloud, condition, rng)
                )
                tracker.update(cloud_for_method, timestamp=delayed_t)
                target_pose = tracker.get_target_pose(demo_data, t_demo=t_demo, tau=mt3.TAU)
            elif method == "Pure reactive (no prediction)":
                if not is_dropout:
                    assert tracker is not None
                    cloud_for_method = corrupt_observation_cloud(delayed_cloud, condition, rng)
                    state = tracker.update(cloud_for_method, timestamp=delayed_t)
                    frozen_delta_xy = np.array([state.delta_x, state.delta_y], dtype=float)
                target_pose = sim08.make_delta_transform(frozen_delta_xy) @ demo_data.get_pose_at(t_demo)
            else:
                if not is_dropout:
                    cloud_for_method = corrupt_observation_cloud(delayed_cloud, condition, rng)
                    raw_xy = sim08.cloud_centroid_xy(cloud_for_method)
                    if raw_xy is None:
                        raw_xy = previous_raw_xy if previous_raw_xy is not None else reference_centroid
                    if previous_raw_xy is not None and previous_time is not None:
                        dt = max(delayed_t - previous_time, 1e-9)
                        previous_velocity = (raw_xy - previous_raw_xy) / dt
                    predicted_xy = raw_xy + previous_velocity * mt3.TAU
                    frozen_delta_xy = predicted_xy - reference_centroid
                    previous_raw_xy = raw_xy.copy()
                    previous_time = delayed_t
                target_pose = sim08.make_delta_transform(frozen_delta_xy) @ demo_data.get_pose_at(t_demo)

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

            times.append(replay_t)
            demo_fidelity_errors.append(demo_fidelity)

    finally:
        p.disconnect()
        mt3.sim03.box_position = original_box_position

    fidelity_arr = np.asarray(demo_fidelity_errors, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    steady_mask = times_arr > WARMUP_SECONDS
    if not np.any(steady_mask):
        steady_mask = np.ones_like(times_arr, dtype=bool)
    steady_fidelity = fidelity_arr[steady_mask]

    return TrialResult(
        steady_state_rms_error_mm=float(np.sqrt(np.mean(steady_fidelity ** 2)) * 1000.0),
        mean_demo_fidelity_error_mm=float(np.mean(steady_fidelity) * 1000.0),
        success_rate_pct=float(np.mean(steady_fidelity < mt3.SUCCESS_THRESHOLD_M) * 100.0),
    )


def run_all_trials() -> dict[str, dict[str, list[TrialResult]]]:
    """Run all conditions and methods."""
    results: dict[str, dict[str, list[TrialResult]]] = {}
    for condition in CONDITIONS:
        condition_results: dict[str, list[TrialResult]] = {}
        condition_latency_samples: list[float] = []
        for method in METHODS:
            trial_results = []
            for trial_idx in range(N_TRIALS):
                latency_samples = (
                    condition_latency_samples
                    if PRINT_LATENCY_DIAGNOSTICS and method == METHODS[0] and trial_idx == 0
                    else None
                )
                result = run_single_trial(condition, method, trial_idx, latency_samples=latency_samples)
                trial_results.append(result)
                print(
                    f"{condition.name:10s} | {method:30s} "
                    f"trial={trial_idx + 1}/{N_TRIALS}: "
                    f"steady_rms={result.steady_state_rms_error_mm:.1f}mm, "
                    f"success={result.success_rate_pct:.1f}%"
                )
            condition_results[method] = trial_results
        if PRINT_LATENCY_DIAGNOSTICS and condition_latency_samples:
            lat_ms = np.asarray(condition_latency_samples, dtype=float) * 1000.0
            preview = ", ".join(f"{value:.1f}" for value in lat_ms[:20])
            print(
                f"{condition.name:10s} latency samples (ms): "
                f"mean={lat_ms.mean():.1f}, std={lat_ms.std(ddof=0):.1f}, "
                f"min={lat_ms.min():.1f}, max={lat_ms.max():.1f}, "
                f"first20=[{preview}]"
            )
        results[condition.name] = condition_results
    return results


def print_summary_table(results: dict[str, dict[str, list[TrialResult]]]) -> None:
    """Print full robustness table."""
    print()
    print("Robustness analysis summary")
    print("condition  | method                         | steady_rms_mm | mean_fidelity_mm | success_rate_%")
    print("-----------|--------------------------------|---------------|------------------|---------------")
    for condition in CONDITIONS:
        for method in METHODS:
            trial_results = results[condition.name][method]
            rms = np.array([r.steady_state_rms_error_mm for r in trial_results], dtype=float)
            fidelity = np.array([r.mean_demo_fidelity_error_mm for r in trial_results], dtype=float)
            success = np.array([r.success_rate_pct for r in trial_results], dtype=float)
            print(
                f"{condition.name:10s} | {method:30s} | "
                f"{rms.mean():7.1f} ± {rms.std(ddof=0):4.1f} | "
                f"{fidelity.mean():7.1f} ± {fidelity.std(ddof=0):4.1f} | "
                f"{success.mean():6.1f} ± {success.std(ddof=0):4.1f}"
            )


def plot_robustness(results: dict[str, dict[str, list[TrialResult]]], out_path: Path) -> None:
    """Plot grouped bar chart of steady-state RMS error."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    condition_names = [condition.name for condition in CONDITIONS]
    x = np.arange(len(condition_names), dtype=float)
    width = 0.23
    offsets = np.linspace(-width, width, len(METHODS))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, method in zip(offsets, METHODS):
        means = np.array(
            [
                np.mean([r.steady_state_rms_error_mm for r in results[condition.name][method]])
                for condition in CONDITIONS
            ],
            dtype=float,
        )
        stds = np.array(
            [
                np.std([r.steady_state_rms_error_mm for r in results[condition.name][method]], ddof=0)
                for condition in CONDITIONS
            ],
            dtype=float,
        )
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            alpha=0.88,
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in means], padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_names)
    ax.set_ylabel("steady-state RMS error (mm)")
    ax.set_title("Robustness Analysis Under Latency, Noise, Dropout, and Outliers")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    results = run_all_trials()
    print_summary_table(results)
    out_path = PROJECT_ROOT / "simulation" / "results" / "robustness_analysis.png"
    plot_robustness(results, out_path)
    print("Saved simulation/results/robustness_analysis.png")


if __name__ == "__main__":
    main()
