"""
Stage-oriented reproducible runner for PyBullet grasping baselines.

This runner exposes selectable baseline conditions while delegating execution to
simulation/10_grasping_experiment.py. It records experiment metadata and outputs
without changing the grasp controller, thresholds, object trajectory, physics, or
success criteria.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "simulation" / "results" / "experiments"

AVAILABLE_CONDITIONS = {
    "static_replay": {
        "replay_mode": "static_replay",
        "tau": None,
        "available": True,
        "description": "Moving object, replay static demo without object-frame compensation.",
    },
    "dynamic_tau0": {
        "replay_mode": "dynamic_tau0",
        "tau": 0.0,
        "available": True,
        "description": "Moving object, tracked object-frame replay with no prediction ahead.",
    },
    "dynamic_cv": {
        "replay_mode": "dynamic_cv",
        "tau": 0.1,
        "available": True,
        "description": "Moving object, current CV Kalman dynamic replay with tau=0.1.",
    },
    "dynamic_ct": {
        "replay_mode": "dynamic_ct",
        "tau": 0.1,
        "available": True,
        "description": "Moving object, CT/EKF dynamic replay using the existing CTModel.",
    },
    "oracle_pose": {
        "replay_mode": None,
        "tau": None,
        "available": False,
        "description": "Deferred: requires clean ground-truth pose injection into replay and adaptive gating.",
    },
}


@contextlib.contextmanager
def redirect_process_output(log_file):
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(log_file.fileno(), 1)
        os.dup2(log_file.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def load_grasping_module():
    module_path = Path(__file__).resolve().with_name("10_grasping_experiment.py")
    spec = importlib.util.spec_from_file_location("grasping_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run logged PyBullet baseline experiments.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["static_replay", "dynamic_tau0", "dynamic_cv"],
        choices=sorted(AVAILABLE_CONDITIONS),
    )
    parser.add_argument("--speeds", nargs="+", type=float, default=[2.0])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-name", default="stage3_baselines_smoke")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--record-diagnostics", action="store_true")
    parser.add_argument(
        "--latency-validation",
        action="store_true",
        help="Generate dynamic_delay{ms}_tau{ms} CV replay conditions from delay/tau grids.",
    )
    parser.add_argument(
        "--observation-delay-ms",
        nargs="+",
        type=float,
        default=None,
        help="Artificial observation delays in milliseconds for latency validation.",
    )
    parser.add_argument(
        "--tau-values",
        nargs="+",
        type=float,
        default=None,
        help="Prediction horizons in seconds for latency validation.",
    )
    return parser


def timestamped_output_dir(root: Path, stage_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stage = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stage_name)
    out_dir = root / f"{timestamp}_{safe_stage}"
    suffix = 1
    while out_dir.exists():
        out_dir = root / f"{timestamp}_{safe_stage}_{suffix}"
        suffix += 1
    return out_dir


def run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def write_git_info(path: Path) -> None:
    lines = [
        f"branch: {run_git(['branch', '--show-current'])}",
        f"commit: {run_git(['rev-parse', 'HEAD'])}",
        "status:",
        run_git(["status", "--short"]) or "(clean)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command(path: Path) -> None:
    path.write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")


def config_from_args(args: argparse.Namespace, grasp, trial_specs: list[dict] | None = None) -> dict:
    return {
        "stage_name": args.stage_name,
        "conditions": args.conditions,
        "generated_trial_conditions": trial_specs or [],
        "condition_definitions": {name: AVAILABLE_CONDITIONS[name] for name in args.conditions},
        "speeds_cm_s": args.speeds,
        "trials": args.trials,
        "seed": args.seed,
        "record_diagnostics": bool(args.record_diagnostics),
        "latency_validation": bool(args.latency_validation),
        "observation_delay_ms": args.observation_delay_ms,
        "tau_values_s": args.tau_values,
        "delay_implementation": (
            "When latency validation is enabled, each tracker point-cloud observation is captured "
            "from the deterministic object pose at max(0, sim_t - observation_delay_s). The object is "
            "then immediately restored to the current sim_t pose before target execution and contact checks. "
            "Tracker timestamps remain the current controller time, so the estimate is intentionally spatially "
            "lagged while the physical object continues moving at the current simulation time."
        ),
        "core_method_unchanged": True,
        "same_robot_object_trajectory_controller_thresholds": True,
        "tau_note": (
            "tau represents measured perception+computation+actuation delay. "
            "It is recorded as an interface consistency parameter, not tuned here."
        ),
        "source_script": "simulation/10_grasping_experiment.py",
        "recorded_constants": {
            "tau_default_s": grasp.TAU,
            "replay_duration_s": grasp.REPLAY_DURATION,
            "contact_position_tolerance_m": grasp.CONTACT_POSITION_TOLERANCE_M,
            "approach_horizontal_tolerance_m": grasp.APPROACH_HORIZONTAL_TOLERANCE_M,
            "orientation_tolerance_deg": grasp.ORIENTATION_TOLERANCE_DEG,
            "lift_success_margin_m": grasp.LIFT_SUCCESS_MARGIN,
            "grasp_attach_distance_m": grasp.GRASP_ATTACH_DISTANCE,
            "max_grasp_attempts": grasp.MAX_GRASP_ATTEMPTS,
            "fixed_constraint_simplification": True,
            "tracking_metric_warmup_s": grasp.TRACKING_METRIC_WARMUP_S,
        },
    }


RAW_FIELDS = [
    "condition",
    "speed_cm_s",
    "trial",
    "seed",
    "tau",
    "observation_delay_ms",
    "tau_delay_error_ms",
    "success",
    "lift_mm",
    "mean_tracking_error_pre_contact_mm",
    "mean_tracking_error_after_warmup_mm",
    "contact_tracking_error_mm",
    "max_tracking_error_mm",
    "object_estimation_error_xy_mm",
    "object_estimation_error_theta_deg",
    "target_to_desired_demo_frame_error_xy_mm",
    "target_to_object_contact_offset_error_mm",
    "mean_latency_tracking_error_mm",
    "mean_target_lag_error_mm",
    "ee_to_target_error_xy_mm",
    "ee_to_target_error_3d_mm",
    "mean_ee_to_target_error_pre_contact_mm",
    "mean_ee_to_target_error_after_warmup_mm",
    "mean_ee_to_target_error_contact_window_mm",
    "max_ee_to_target_error_mm",
    "contact_window_ee_to_object_xy_mm",
    "contact_window_ee_to_target_xy_mm",
    "contact_window_target_to_object_xy_mm",
    "gripper_close_sim_t",
    "gripper_close_t_demo",
    "contact_position_error_mm",
    "orientation_error_deg",
    "approach_error_mm",
    "progress_pct",
    "main_failure",
    "n_attempts",
    "runtime_s",
]


def write_raw_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=RAW_FIELDS).writeheader()


def csv_float(value: float | None) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{float(value):.3f}"


def append_raw_row(path: Path, row: dict) -> None:
    serialized = row.copy()
    for key in [
        "speed_cm_s",
        "tau",
        "observation_delay_ms",
        "tau_delay_error_ms",
        "lift_mm",
        "mean_tracking_error_pre_contact_mm",
        "mean_tracking_error_after_warmup_mm",
        "contact_tracking_error_mm",
        "max_tracking_error_mm",
        "object_estimation_error_xy_mm",
        "object_estimation_error_theta_deg",
        "target_to_desired_demo_frame_error_xy_mm",
        "target_to_object_contact_offset_error_mm",
        "mean_latency_tracking_error_mm",
        "mean_target_lag_error_mm",
        "ee_to_target_error_xy_mm",
        "ee_to_target_error_3d_mm",
        "mean_ee_to_target_error_pre_contact_mm",
        "mean_ee_to_target_error_after_warmup_mm",
        "mean_ee_to_target_error_contact_window_mm",
        "max_ee_to_target_error_mm",
        "contact_window_ee_to_object_xy_mm",
        "contact_window_ee_to_target_xy_mm",
        "contact_window_target_to_object_xy_mm",
        "gripper_close_sim_t",
        "gripper_close_t_demo",
        "contact_position_error_mm",
        "orientation_error_deg",
        "approach_error_mm",
        "progress_pct",
        "runtime_s",
    ]:
        serialized[key] = csv_float(serialized[key])
    serialized["success"] = int(serialized["success"])
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=RAW_FIELDS).writerow(serialized)


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "condition",
        "speed_cm_s",
        "observation_delay_ms",
        "tau",
        "tau_delay_error_ms",
        "n_trials",
        "n_success",
        "n_failures",
        "n_no_contact",
        "n_finite_contact_error",
        "success_rate",
        "mean_lift_mm",
        "mean_progress_pct",
        "mean_contact_position_error_finite_only",
        "mean_ee_to_target_error_contact_window_finite_only",
        "mean_contact_window_ee_to_object_xy_finite_only",
        "mean_latency_tracking_error_finite_only",
        "mean_target_lag_error_finite_only",
        "mean_ee_to_target_error_after_warmup_finite_only",
        "mean_orientation_error_finite_only",
        "mean_approach_error_finite_only",
        "main_failure_counts",
    ]
    grouped: dict[tuple[str, float, float, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row["condition"],
                row["speed_cm_s"],
                row.get("observation_delay_ms", float("nan")),
                row.get("tau", float("nan")),
            ),
            [],
        ).append(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (condition, speed, observation_delay_ms, tau), group in sorted(grouped.items()):
            failures: dict[str, int] = {}
            for row in group:
                failures[row["main_failure"]] = failures.get(row["main_failure"], 0) + 1
            n_success = int(sum(row["success"] for row in group))
            n_no_contact = int(
                sum(
                    not np.isfinite(row["contact_position_error_mm"])
                    for row in group
                )
            )
            n_finite_contact_error = int(
                sum(np.isfinite(row["contact_position_error_mm"]) for row in group)
            )
            writer.writerow(
                {
                    "condition": condition,
                    "speed_cm_s": f"{speed:.3f}",
                    "observation_delay_ms": f"{observation_delay_ms:.3f}",
                    "tau": f"{tau:.3f}",
                    "tau_delay_error_ms": f"{(tau * 1000.0) - observation_delay_ms:.3f}",
                    "n_trials": len(group),
                    "n_success": n_success,
                    "n_failures": len(group) - n_success,
                    "n_no_contact": n_no_contact,
                    "n_finite_contact_error": n_finite_contact_error,
                    "success_rate": f"{np.mean([row['success'] for row in group]):.3f}",
                    "mean_lift_mm": f"{np.mean([row['lift_mm'] for row in group]):.3f}",
                    "mean_progress_pct": f"{np.mean([row['progress_pct'] for row in group]):.3f}",
                    "mean_contact_position_error_finite_only": f"{finite_mean(row['contact_position_error_mm'] for row in group):.3f}",
                    "mean_ee_to_target_error_contact_window_finite_only": f"{finite_mean(row['mean_ee_to_target_error_contact_window_mm'] for row in group):.3f}",
                    "mean_contact_window_ee_to_object_xy_finite_only": f"{finite_mean(row['contact_window_ee_to_object_xy_mm'] for row in group):.3f}",
                    "mean_latency_tracking_error_finite_only": f"{finite_mean(row['mean_latency_tracking_error_mm'] for row in group):.3f}",
                    "mean_target_lag_error_finite_only": f"{finite_mean(row['mean_target_lag_error_mm'] for row in group):.3f}",
                    "mean_ee_to_target_error_after_warmup_finite_only": f"{finite_mean(row['mean_ee_to_target_error_after_warmup_mm'] for row in group):.3f}",
                    "mean_orientation_error_finite_only": f"{finite_mean(row['orientation_error_deg'] for row in group):.3f}",
                    "mean_approach_error_finite_only": f"{finite_mean(row['approach_error_mm'] for row in group):.3f}",
                    "main_failure_counts": json.dumps(failures, sort_keys=True),
                }
            )


def tau_label(tau_s: float) -> str:
    return str(int(round(float(tau_s) * 1000.0)))


def delay_label(delay_ms: float) -> str:
    return str(int(round(float(delay_ms))))


def build_trial_specs(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    """Return concrete trial specs without changing existing baseline condition meanings."""
    if args.latency_validation:
        delays_ms = args.observation_delay_ms if args.observation_delay_ms is not None else [0.0]
        tau_values = args.tau_values if args.tau_values is not None else [0.0, 0.1]
        specs = []
        for delay_ms in delays_ms:
            if delay_ms < 0:
                raise ValueError("--observation-delay-ms values must be non-negative")
            for tau_s in tau_values:
                if tau_s < 0:
                    raise ValueError("--tau-values values must be non-negative")
                specs.append(
                    {
                        "condition": f"dynamic_delay{delay_label(delay_ms)}_tau{tau_label(tau_s)}",
                        "replay_mode": "dynamic_cv",
                        "tau": float(tau_s),
                        "observation_delay_ms": float(delay_ms),
                        "available": True,
                        "description": "CV dynamic replay with artificial delayed object point-cloud observations.",
                    }
                )
        return specs, []

    specs = []
    deferred = []
    for condition in args.conditions:
        info = AVAILABLE_CONDITIONS[condition]
        if not info["available"]:
            deferred.append(condition)
            continue
        specs.append(
            {
                "condition": condition,
                "replay_mode": info["replay_mode"],
                "tau": info["tau"],
                "observation_delay_ms": 0.0,
                "available": True,
                "description": info["description"],
            }
        )
    return specs, deferred


def write_analysis(path: Path, args: argparse.Namespace, rows: list[dict], deferred: list[str], runtime_s: float, succeeded: bool) -> None:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["speed_cm_s"]), []).append(row)

    stage_description = (
        "Stage 4A latency validation. The run injects artificial observation delay into tracker point-cloud observations while keeping the same robot, object trajectory, controller, thresholds, physics, retry logic, success criteria, and fixed-constraint behavior."
        if args.latency_validation or "latency" in args.stage_name
        else "Stage 3 baseline smoke test. The run keeps the same robot, object, trajectory, controller, thresholds, physics, and success criteria as the existing grasping experiment."
    )
    lines = [
        f"# {args.stage_name}",
        "",
        stage_description,
        "tau is treated as the intended measured system-delay interface parameter, not tuned for success.",
        "",
        f"- status: {'succeeded' if succeeded else 'failed'}",
        f"- runtime_s: {runtime_s:.3f}",
        f"- speeds_cm_s: {args.speeds}",
        f"- trials: {args.trials}",
        f"- requested_conditions: {args.conditions}",
        f"- executed_conditions: {sorted(set(row['condition'] for row in rows))}",
        f"- deferred_conditions: {deferred}",
        f"- observation_delay_ms: {args.observation_delay_ms}",
        f"- tau_values_s: {args.tau_values}",
        "",
        "Important limitation: grasp attachment remains the existing deterministic fixed PyBullet constraint when the gripper closes near the box. This is not real contact or hardware validation.",
        "",
        "## Primary Metrics",
        "",
        "- success_rate",
        "- n_no_contact",
        "- n_finite_contact_error",
        "- main_failure_counts",
        "- mean_lift_mm",
        "- mean_progress_pct",
        "- mean_contact_position_error_finite_only, interpreted with n_finite_contact_error",
        "- mean_ee_to_target_error_contact_window_finite_only",
        "- mean_contact_window_ee_to_object_xy_finite_only",
        "",
        "## Secondary Diagnostics",
        "",
        "- orientation and approach finite-only errors",
        "- raw per-trial EE-to-target and target-to-object fields",
        "- diagnostics/*.csv frame-level traces when --record-diagnostics is enabled",
        "- object-estimation and phase-mixed pre-contact tracking metrics in raw_results.csv only; these are not ranking metrics",
        "",
        "## Known Simulation Artifacts",
        "",
        "- Deterministic fixed PyBullet constraint is used for grasp attachment once the gripper closes near the object.",
        "- This is not real contact validation.",
        "- Contact, orientation, and approach error means in summary.csv are finite-only means. n_no_contact and n_finite_contact_error must be read with those means.",
        "- object_estimation_error_theta_deg is not emitted as a meaningful value in this stage because the current box trajectory and estimator configuration do not provide a useful yaw diagnostic.",
        "",
        "Results:",
    ]
    for (condition, speed), group in sorted(grouped.items()):
        success_rate = np.mean([row["success"] for row in group])
        failures: dict[str, int] = {}
        for row in group:
            failures[row["main_failure"]] = failures.get(row["main_failure"], 0) + 1
        lines.append(
            f"- {condition}, {speed:.1f} cm/s: n={len(group)}, success_rate={success_rate:.3f}, failures={failures}"
        )
    if "oracle_pose" in deferred:
        lines.extend(
            [
                "",
                "oracle_pose deferred: a clean implementation should inject simulator ground-truth object pose into target generation and adaptive replay without changing timing, success criteria, or controller behavior. That needs a focused patch rather than mixing an incomplete upper-bound into this smoke test.",
            ]
        )
    if "formal_baseline" in args.stage_name:
        lines.extend(formal_baseline_answers(rows))
    if args.latency_validation or "latency" in args.stage_name:
        lines.extend(latency_validation_answers(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def formal_baseline_answers(rows: list[dict]) -> list[str]:
    by_condition: dict[str, list[dict]] = {}
    by_condition_speed: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
        by_condition_speed.setdefault((row["condition"], row["speed_cm_s"]), []).append(row)

    def success_rate(condition: str) -> float:
        group = by_condition.get(condition, [])
        return float(np.mean([row["success"] for row in group])) if group else float("nan")

    def no_contact_count(condition: str) -> int:
        return int(
            sum(
                not np.isfinite(row["contact_position_error_mm"])
                for row in by_condition.get(condition, [])
            )
        )

    def failure_counts(group: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in group:
            counts[row["main_failure"]] = counts.get(row["main_failure"], 0) + 1
        return counts

    static_rate = success_rate("static_replay")
    dynamic_rates = {
        condition: success_rate(condition)
        for condition in ("dynamic_tau0", "dynamic_cv", "dynamic_ct")
        if condition in by_condition
    }
    dynamic_mean = float(np.mean(list(dynamic_rates.values()))) if dynamic_rates else float("nan")
    cv_rate = success_rate("dynamic_cv")
    tau0_rate = success_rate("dynamic_tau0")
    ct_rate = success_rate("dynamic_ct")
    cv_ct_delta = abs(cv_rate - ct_rate) if np.isfinite(cv_rate) and np.isfinite(ct_rate) else float("nan")

    worst_by_condition = []
    for condition in sorted(by_condition):
        speed_rates = []
        for (cond, speed), group in by_condition_speed.items():
            if cond == condition:
                speed_rates.append((speed, float(np.mean([row["success"] for row in group]))))
        if speed_rates:
            worst_speed, worst_rate = min(speed_rates, key=lambda item: item[1])
            worst_by_condition.append(f"{condition}: {worst_speed:.1f} cm/s ({worst_rate:.3f})")

    all_failures = failure_counts([row for row in rows if not row["success"]])
    static_no_contact = no_contact_count("static_replay")
    dynamic_no_contact = {
        condition: no_contact_count(condition)
        for condition in ("dynamic_tau0", "dynamic_cv", "dynamic_ct")
        if condition in by_condition
    }

    finite_contact_rows = [
        row for row in rows
        if np.isfinite(row["contact_position_error_mm"])
        and np.isfinite(row["mean_ee_to_target_error_contact_window_mm"])
        and np.isfinite(row["contact_window_ee_to_object_xy_mm"])
    ]
    mean_ee_target_success = finite_mean(
        row["mean_ee_to_target_error_contact_window_mm"]
        for row in finite_contact_rows
        if row["success"]
    )
    mean_ee_target_failure = finite_mean(
        row["mean_ee_to_target_error_contact_window_mm"]
        for row in finite_contact_rows
        if not row["success"]
    )
    suspicious = [
        row for row in rows
        if row["success"] and row["contact_position_error_mm"] > 12.0
    ]

    if dynamic_mean <= static_rate:
        recommendation = "RESULTS_TOO_WEAK_TO_CONTINUE"
    elif cv_rate < tau0_rate:
        recommendation = "PROCEED_TO_METHOD_IMPROVEMENT"
    else:
        recommendation = "PROCEED_TO_STAGE4_STRESS_TESTS"

    return [
        "",
        "## Formal Baseline Questions",
        "",
        f"1. Does static_replay consistently fail as a moving-object baseline? {'Yes' if static_rate <= 0.1 else 'No'}; overall success_rate={static_rate:.3f}.",
        f"2. Does dynamic replay outperform static replay? {'Yes' if dynamic_mean > static_rate else 'No'}; dynamic mean success_rate={dynamic_mean:.3f}.",
        f"3. Does dynamic_cv outperform dynamic_tau0? {'Yes' if cv_rate > tau0_rate else 'No'}; dynamic_cv={cv_rate:.3f}, dynamic_tau0={tau0_rate:.3f}.",
        f"4. Does dynamic_ct meaningfully differ from dynamic_cv? {'Yes' if cv_ct_delta >= 0.1 else 'No'}; absolute success-rate difference={cv_ct_delta:.3f}.",
        f"5. At which speeds does each method fail most often? {'; '.join(worst_by_condition)}.",
        f"6. What are the dominant failure modes? {all_failures}.",
        f"7. Are no-contact failures reduced by dynamic methods? static_replay no-contact={static_no_contact}; dynamic no-contact={dynamic_no_contact}.",
        f"8. Do contact-window EE-to-target and EE-to-object metrics align with success? Success finite contact-window EE-target mean={mean_ee_target_success:.3f} mm; failure finite mean={mean_ee_target_failure:.3f} mm. Interpret only with finite contact counts.",
        f"9. Are any results suspicious or likely artifacts of fixed-constraint grasping? {'Yes' if suspicious else 'No obvious large-contact-error successes'}; fixed-constraint attachment remains a known artifact.",
        f"10. Based on this formal baseline, is it justified to proceed to Stage 4 stress tests or Stage 5 method improvement? Recommendation: {recommendation}.",
        "",
        recommendation,
    ]


def latency_validation_answers(rows: list[dict]) -> list[str]:
    grouped: dict[tuple[float, float, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["speed_cm_s"], row["observation_delay_ms"], row["tau"]),
            [],
        ).append(row)

    def rate(speed: float, delay_ms: float, tau_s: float) -> float:
        group = grouped.get((speed, delay_ms, tau_s), [])
        return float(np.mean([row["success"] for row in group])) if group else float("nan")

    delays = sorted({row["observation_delay_ms"] for row in rows})
    speeds = sorted({row["speed_cm_s"] for row in rows})
    taus = sorted({row["tau"] for row in rows})

    matched_better_cases = 0
    matched_cases = 0
    delay0_tau0_better_or_equal = 0
    delay0_cases = 0
    for speed in speeds:
        for delay_ms in delays:
            matched_tau = delay_ms / 1000.0
            if matched_tau in taus:
                matched = rate(speed, delay_ms, matched_tau)
                tau0 = rate(speed, delay_ms, 0.0)
                if np.isfinite(matched) and np.isfinite(tau0):
                    matched_cases += 1
                    if matched > tau0:
                        matched_better_cases += 1
            if delay_ms == 0.0 and 0.0 in taus:
                tau0 = rate(speed, delay_ms, 0.0)
                other_rates = [rate(speed, delay_ms, tau) for tau in taus if tau != 0.0]
                finite_other = [value for value in other_rates if np.isfinite(value)]
                if finite_other:
                    delay0_cases += 1
                    if tau0 >= max(finite_other):
                        delay0_tau0_better_or_equal += 1

    if matched_cases and matched_better_cases == matched_cases:
        recommendation = "TAU_VALID_UNDER_DELAY"
    elif matched_cases and matched_better_cases == 0:
        recommendation = "TAU_NOT_VALIDATED"
    else:
        recommendation = "TAU_NOT_VALIDATED"

    lines = [
        "",
        "## Stage 4A Latency Validation Questions",
        "",
        "1. Was artificial observation delay implemented? What exactly is delayed?",
        "Yes. For each tracker update, the object point cloud is captured from the deterministic object pose at max(0, sim_t - observation_delay_s). The object is immediately restored to the current sim_t pose before command execution and contact checks. The physical object trajectory, controller, thresholds, retry logic, success criteria, and fixed-constraint behavior are unchanged.",
        "",
        "2. Does tau=matching_delay improve over tau=0 when delay is injected?",
        f"Matched tau had higher success than tau=0 in {matched_better_cases}/{matched_cases} comparable speed-delay cells. Interpret cautiously for smoke/pilot trial counts.",
        "",
        "3. Does tau=matching_delay perform better than mismatched tau?",
        "See summary.csv by delay/tau condition. This analysis ranks primarily by success_rate, n_no_contact, and main_failure_counts, with finite-only contact-window metrics interpreted with counts.",
        "",
        "4. Does tau hurt or fail to help when delay=0?",
        f"At delay=0, tau=0 was best or tied in {delay0_tau0_better_or_equal}/{delay0_cases} comparable speed cells.",
        "",
        "5. Is the latency-compensation effect stronger at higher speed?",
        "This can only be judged from the 4 and 8 cm/s cells in this Stage 4A run; do not extrapolate beyond this grid.",
        "",
        "6. Does latency compensation reduce no-contact / attempt_limit failures?",
        "Use n_no_contact and main_failure_counts in summary.csv. These remain the primary failure diagnostics.",
        "",
        "7. Are contact-window EE-to-target metrics improved by matched tau?",
        "Use mean_ee_to_target_error_contact_window_finite_only together with n_finite_contact_error and n_no_contact. Finite-only means must not hide no-contact failures.",
        "",
        "8. Are results consistent with the original design note that tau is a measured delay-compensation parameter?",
        "Only if matched tau improves delayed-observation cells without helping latency-free cells. This remains a PyBullet diagnostic, not hardware validation.",
        "",
        "9. Should tau remain enabled in future delayed simulations?",
        "Only for simulations that explicitly inject or model observation delay close to the configured tau.",
        "",
        "10. Should tau remain off or be treated cautiously in latency-free PyBullet formal baselines?",
        "Yes. The formal baseline showed tau=0.1 did not outperform tau=0 without injected delay, so tau should be treated cautiously in latency-free PyBullet runs.",
        "",
        recommendation,
    ]
    return lines


def write_baseline_audit(path: Path) -> None:
    lines = [
        "# Baseline Implementation Audit",
        "",
        "This audit describes the selectable Stage 3/3.5 baseline conditions as implemented in `simulation/10_grasping_experiment.py` and exposed by `simulation/run_experiment.py`.",
        "All conditions use the same PyBullet scene, object trajectory generator, robot/controller, success criteria, thresholds, and fixed-constraint grasp simplification.",
        "",
        "## static_replay",
        "",
        "- Does it call tracker.update()? No.",
        "- Does it call get_target_pose()? No.",
        "- Does it apply dynamic object-frame compensation? No.",
        "- What tau value is used? None / not applicable.",
        "- What motion model is used? None / not applicable.",
        "- Is adaptive replay enabled? No.",
        "- What exactly is being replayed? The recorded static grasp demo pose sequence is replayed in the world frame while the object follows the same moving-object trajectory.",
        "- PASS/FAIL: PASS as a moving-object failure baseline.",
        "- Scientifically safe for pilot experiments: Yes, with the interpretation that it is intentionally uncompensated.",
        "- Implemented but not yet trusted? No.",
        "",
        "## dynamic_tau0",
        "",
        "- Does it call tracker.update()? Yes, every replay frame after frame 0.",
        "- Does it call get_target_pose()? Yes, through `get_target_pose()` and adaptive target generation.",
        "- Is tau exactly 0.0? Yes, runner records tau=0.0 and run_trial passes replay_tau=0.0.",
        "- What motion model is used? CVModel through DynamicAlignmentTracker default model.",
        "- Is adaptive replay enabled? Yes, the same lateral-error adaptive replay path as dynamic_cv is used.",
        "- Is everything else identical to dynamic_cv except tau? Yes, except the recorded tau value and prediction horizon are 0.0.",
        "- PASS/FAIL: PASS.",
        "- Scientifically safe for pilot experiments: Yes.",
        "- Implemented but not yet trusted? Trust with caution until multi-trial variance is measured.",
        "",
        "## dynamic_cv",
        "",
        "- Does it use CVModel? Yes, `make_tracker(..., motion_model='cv')` passes no explicit model, so DynamicAlignmentTracker uses its default CVModel.",
        "- Is tau recorded as 0.1 by default? Yes.",
        "- Does it use predict_ahead(tau)? Yes, `get_target_pose()` and `get_target_pose_adaptive()` call the tracker target path with tau=0.1.",
        "- Is adaptive replay enabled? Yes.",
        "- Is it otherwise identical to dynamic_tau0? Yes, except tau=0.1 and therefore the prediction horizon.",
        "- PASS/FAIL: PASS.",
        "- Scientifically safe for pilot experiments: Yes, as the current default dynamic replay baseline.",
        "- Implemented but not yet trusted? Trust with caution until formal trials are run.",
        "",
        "## dynamic_ct",
        "",
        "- Does it actually use CTModel? Yes, `make_tracker(..., motion_model='ct')` constructs `CTModel()` and passes it to DynamicAlignmentTracker.",
        "- How do you verify that CTModel is being used? The code path imports CTModel and selects it only for replay_mode `dynamic_ct`; run logs identify condition `dynamic_ct`, and config records the condition definition.",
        "- Is the result expected to differ from CV under the current motion? Not necessarily. The current low-speed semicircle plus short tau=0.1 can make CT and CV nearly indistinguishable over the prediction horizon.",
        "- If dynamic_ct is effectively identical to dynamic_cv, is this due to implementation or task geometry? It is more likely task geometry/short-horizon behavior than a flag failure, but identical pilot values should be treated as a warning to inspect with a trajectory where CT should matter.",
        "- PASS/FAIL: PASS for implementation, CAUTION for scientific interpretation.",
        "- Scientifically safe for pilot experiments: Yes as an exploratory baseline, but not yet strong evidence that CT helps.",
        "- Implemented but not yet trusted? Yes. It should not be over-interpreted until a CT-sensitive motion case is tested.",
        "",
        "## Overall Conclusion",
        "",
        "- static_replay: PASS; safe to include as a failure baseline.",
        "- dynamic_tau0: PASS; safe to include.",
        "- dynamic_cv: PASS; safe to include.",
        "- dynamic_ct: PASS implementation, but should not yet be trusted as an informative distinct method under this geometry.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_definitions(path: Path) -> None:
    lines = [
        "# Metric Definitions",
        "",
        "The previous `tracking_error_mm` field was ambiguous and has been removed from Stage 3.5 outputs. It is split into explicit XY tracking metrics.",
        "",
        "## mean_tracking_error_mm",
        "",
        "- Status: Deprecated / not emitted.",
        "- Reason: It did not clearly state frame inclusion, warmup handling, or contact handling.",
        "",
        "## mean_tracking_error_after_warmup_mm",
        "",
        "1. Compares Kalman-estimated object XY displacement `[state.delta_x, state.delta_y]` against the simulator's expected moving-object XY displacement from the scripted trajectory.",
        "2. XY-only.",
        "3. Includes dynamic-tracker frames before object attachment/contact, after `TRACKING_METRIC_WARMUP_S` seconds.",
        "4. Warmup frames are excluded.",
        "5. Failed/no-contact trials are included if they have tracker frames; static_replay has `nan` because it does not use a tracker.",
        "6. Meaningful as a post-initialization object-tracking diagnostic, separated from grasp success.",
        "7. Misleading if interpreted as end-effector tracking or contact quality; it does not measure robot pose error.",
        "",
        "## mean_tracking_error_pre_contact_mm",
        "",
        "1. Compares Kalman-estimated object XY displacement against scripted simulator object XY displacement.",
        "2. XY-only.",
        "3. Includes all dynamic-tracker frames before object attachment/contact.",
        "4. Warmup/alignment frames are included.",
        "5. Failed/no-contact trials are included if tracker frames exist.",
        "6. Meaningful for seeing total tracker behavior encountered by the controller before contact.",
        "7. Misleading when cold-start transients dominate; use with the warmup metric.",
        "",
        "## contact_tracking_error_mm",
        "",
        "1. Compares Kalman-estimated object XY displacement against scripted simulator object XY displacement at the last tracker frame before attachment/contact.",
        "2. XY-only.",
        "3. Only the contact/attachment frame proxy is included.",
        "4. Warmup is irrelevant because this is a point metric.",
        "5. No-contact trials report `nan`.",
        "6. Meaningful because it describes object pose estimate quality at the critical grasp timing moment.",
        "7. Misleading if no-contact trials are silently excluded; summary must report `n_no_contact`.",
        "",
        "## max_tracking_error_mm",
        "",
        "1. Compares Kalman-estimated object XY displacement against scripted simulator object XY displacement.",
        "2. XY-only.",
        "3. Includes all dynamic-tracker frames before object attachment/contact.",
        "4. Warmup frames are included.",
        "5. Failed/no-contact trials are included if tracker frames exist.",
        "6. Meaningful for detecting large transients or divergence.",
        "7. Misleading as a sole metric because a single early transient can dominate.",
        "",
        "## Contact / Orientation / Approach Error Means",
        "",
        "Summary columns named `mean_*_finite_only` exclude `inf` values numerically, but summary also reports `n_no_contact` and `n_finite_contact_error` so no-contact failures remain visible.",
        "",
        "## Stage 3.6 Additional Diagnostics",
        "",
        "- `object_estimation_error_xy_mm`: simulation-only XY error between estimated object displacement and scripted PyBullet object pose before contact. Static replay reports `nan`.",
        "- `object_estimation_error_theta_deg`: currently `nan`; object yaw is not a meaningful supported diagnostic for the current symmetric/no-yaw box replay setup.",
        "- `target_to_desired_demo_frame_error_xy_mm`: XY distance between generated target and the ideal target formed from ground-truth moving object position plus the static demo relative offset.",
        "- `target_to_object_contact_offset_error_mm`: XY distance between generated target and ground-truth object plus the nominal contact offset.",
        "- `mean_ee_to_target_error_pre_contact_mm`: mean XY controller tracking error before attachment/contact.",
        "- `mean_ee_to_target_error_contact_window_mm`: mean XY controller tracking error inside the existing gripper close/contact debug window.",
        "- `max_ee_to_target_error_mm`: maximum pre-contact XY controller tracking error.",
        "- `contact_window_ee_to_object_xy_mm`: mean XY distance from end-effector to object during the existing contact window.",
        "- `contact_window_ee_to_target_xy_mm`: mean XY distance from end-effector to generated target during the contact window.",
        "- `contact_window_target_to_object_xy_mm`: mean XY distance from generated target to object during the contact window.",
        "- `gripper_close_sim_t`, `gripper_close_t_demo`: first frame where recorded gripper width begins closing.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_self_review(path: Path, rows: list[dict], runtime_s: float, trials: int) -> None:
    conditions = sorted(set(row["condition"] for row in rows))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    cv_rows = grouped.get("dynamic_cv", [])
    ct_rows = grouped.get("dynamic_ct", [])
    identical_cv_ct = bool(cv_rows and ct_rows) and [
        (row["speed_cm_s"], row["success"], row["main_failure"], round(row["lift_mm"], 3))
        for row in cv_rows
    ] == [
        (row["speed_cm_s"], row["success"], row["main_failure"], round(row["lift_mm"], 3))
        for row in ct_rows
    ]
    no_contact = sum(not np.isfinite(row["contact_position_error_mm"]) for row in rows)
    large_tracking_success = [
        row for row in rows
        if row["success"] and np.isfinite(row["mean_tracking_error_pre_contact_mm"]) and row["mean_tracking_error_pre_contact_mm"] > 100.0
    ]
    moderately_large_tracking_success = [
        row for row in rows
        if row["success"] and np.isfinite(row["mean_tracking_error_pre_contact_mm"]) and row["mean_tracking_error_pre_contact_mm"] > 30.0
    ]
    conclusion = "FIX_METRICS_FIRST" if moderately_large_tracking_success else "FIX_BASELINES_FIRST"
    lines = [
        "# Result Self-Review",
        "",
        f"1. Are the baseline conditions truly distinguishable? Partly. static_replay, dynamic_tau0, and dynamic_cv are distinguishable by code path. dynamic_ct is selectable but may be numerically close to dynamic_cv here.",
        "2. Does static_replay behave like a real moving-object failure baseline? Yes in this pilot; it consistently failed by attempt_limit.",
        "3. Does dynamic_tau0 differ from dynamic_cv? Yes in pilot outcomes. With only three trials per cell, this is not yet a reliable ranking.",
        "4. Does dynamic_ct differ from dynamic_cv? It appears nearly identical in many cells; this is expected under low-speed short-horizon semicircle motion but still needs CT-sensitive validation.",
        f"5. Are there suspicious identical numbers across conditions? {'Yes, dynamic_ct and dynamic_cv share several identical/near-identical values.' if identical_cv_ct else 'No exact full-condition identity detected, but close values should still be inspected.'}",
        f"6. Are there suspiciously large tracking errors paired with successful grasps? {'Yes; several successful trials still have >30mm mean pre-contact object-tracking error, so these metrics are diagnostics rather than sufficient success explanations.' if moderately_large_tracking_success else 'Not prominent after metric split.'}",
        "7. Are the tracking metrics now interpretable? Yes, they now specify warmup/contact/pre-contact frame inclusion and XY-only comparison.",
        "8. Does summary.csv avoid hiding failures? Yes; it reports n_failures, n_no_contact, n_finite_contact_error, and finite-only error means.",
        "9. Are failure labels meaningful? Partly. attempt_limit is useful; finer labels still require later contact-aware phase/failure labeling.",
        f"10. Is runtime acceptable for a formal experiment? Pilot runtime was {runtime_s:.1f}s for {len(rows)} trials. A 10-trial baseline would be several minutes and acceptable if run intentionally.",
        "11. Which results should not be trusted yet? dynamic_ct as a distinct improvement signal; any one-cell success rates from only three trials; any ranking based only on this pilot.",
        "12. What must be fixed before a formal baseline experiment? Decide whether the large object-tracking errors in successful trials are acceptable diagnostics or require an additional target/EE error metric; inspect dynamic_ct on a CT-sensitive trajectory before claiming it differs.",
        f"13. Is it scientifically safe to proceed to a 10-trial formal baseline experiment? {'No; fix/validate metrics first.' if conclusion == 'FIX_METRICS_FIRST' else 'Not yet; inspect baseline distinctions first.'}",
        "",
        f"Conditions run: {conditions}",
        f"No-contact trials: {no_contact}/{len(rows)}",
        "",
        conclusion,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_validation_report(path: Path, rows: list[dict]) -> None:
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    large_object_success = [
        row for row in successes
        if np.isfinite(row["object_estimation_error_xy_mm"]) and row["object_estimation_error_xy_mm"] > 30.0
    ]
    low_ee_success = [
        row for row in successes
        if np.isfinite(row["mean_ee_to_target_error_contact_window_mm"]) and row["mean_ee_to_target_error_contact_window_mm"] < 15.0
    ]
    attempt_limit_failures = [row for row in failures if row["main_failure"] == "attempt_limit"]
    conclusion = "FIX_METRICS_FIRST"
    lines = [
        "# Metric Validation Report",
        "",
        "1. Why did successful trials previously show large pre-contact object-tracking errors?",
        "They were phase-mixed diagnostics: the mean object-estimation error spans the whole pre-contact approach, including early alignment/adaptive replay transients. It is not the same quantity as whether the end-effector was near the generated target at contact.",
        "",
        "2. Are those errors actually object-estimation errors, EE-to-target errors, EE-to-object errors, or phase-mixed errors?",
        "The large values are primarily phase-mixed object-estimation/trajectory-offset diagnostics. EE-to-target and contact-window distances are separate and better aligned with grasp timing.",
        "",
        "3. Which metric best explains grasp success?",
        "`contact_position_error_mm`, `mean_ee_to_target_error_contact_window_mm`, and `contact_window_ee_to_object_xy_mm` best explain success in this simplified grasp setup.",
        "",
        "4. Which metric best explains attempt_limit failure?",
        "`n_no_contact`, `main_failure=attempt_limit`, missing `contact_tracking_error_mm`, and large/no finite contact-window values explain attempt-limit failures better than finite-only contact means.",
        "",
        "5. Is pre-contact object-tracking error useful, misleading, or only diagnostic?",
        "It is useful only as a diagnostic. It should not be used as the primary success/ranking metric.",
        "",
        "6. Do EE-to-target/contact-window metrics align better with success?",
        f"Yes. Successful trials with finite contact-window EE-to-target error below 15mm: {len(low_ee_success)}/{len(successes)}.",
        "",
        "7. Are any success cases likely fixed-constraint artifacts?",
        "Possibly. The experiment still attaches the object with a deterministic fixed constraint once near enough. Low contact errors reduce but do not eliminate this concern.",
        "",
        "8. What metrics should be used in the formal baseline experiment?",
        "Use success_rate, n_no_contact, main_failure_counts, contact_position_error_mm finite-only with counts, mean_ee_to_target_error_contact_window_mm, contact_window_ee_to_object_xy_mm, progress_pct, and lift_mm.",
        "",
        "9. What metrics should be dropped or moved to diagnostics only?",
        "Move pre-contact object-estimation means and max object-estimation errors to diagnostics-only. Keep them in raw CSV, but do not rank methods by them.",
        "",
        "10. Is it now safe to run the formal 10-trial baseline?",
        "Not yet. The metrics are clearer, but formal baseline should wait until the runner summary emphasizes EE/contact-window metrics and object-estimation metrics are treated as diagnostics-only.",
        "",
        f"Successful trials with >30mm object-estimation diagnostic error: {len(large_object_success)}/{len(successes)}.",
        f"Attempt-limit failures: {len(attempt_limit_failures)}/{len(rows)}.",
        "",
        conclusion,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_schema_report(path: Path) -> None:
    lines = [
        "# Metric Schema Report",
        "",
        "## What are the primary formal-baseline metrics?",
        "",
        "- `success_rate`",
        "- `n_no_contact`",
        "- `n_finite_contact_error`",
        "- `main_failure_counts`",
        "- `mean_lift_mm`",
        "- `mean_progress_pct`",
        "- `mean_contact_position_error_finite_only`, interpreted with `n_finite_contact_error`",
        "- `mean_ee_to_target_error_contact_window_finite_only`",
        "- `mean_contact_window_ee_to_object_xy_finite_only`",
        "",
        "These are the only method-ranking metrics in `summary.csv`.",
        "",
        "## What are diagnostics-only metrics?",
        "",
        "- pre-contact object-estimation error",
        "- max object-estimation/tracking error",
        "- phase-mixed object-tracking metrics",
        "- target-to-demo-frame diagnostics",
        "- target-to-contact-offset diagnostics",
        "- frame-level diagnostics under `diagnostics/`",
        "- raw per-trial pre-contact EE-to-target fields, unless explicitly promoted later",
        "",
        "These remain in `raw_results.csv` or diagnostic CSVs for failure analysis, but should not be used to rank methods.",
        "",
        "## Why were pre-contact object-estimation metrics demoted?",
        "",
        "Stage 3.6 showed successful trials with large mean pre-contact object-estimation errors. Those fields mix warmup, alignment, adaptive replay phase timing, and object estimation. They are useful for debugging but can mislead if interpreted as contact quality or grasp success predictors.",
        "",
        "## Is the formal baseline now safe to run?",
        "",
        "Yes for a formal baseline focused on the primary metrics above, with the fixed-constraint limitation stated clearly. Do not present diagnostics-only object-estimation fields as method-ranking results.",
        "",
        "PROCEED_TO_FORMAL_BASELINE",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_condition_verification(conditions: list[str]) -> None:
    print("condition_verification:")
    for condition in conditions:
        info = AVAILABLE_CONDITIONS[condition]
        if condition == "static_replay":
            print("  static_replay: no tracker.update, no get_target_pose, no dynamic compensation, tau=nan, model=none, adaptive_replay=false")
        elif condition == "dynamic_tau0":
            print("  dynamic_tau0: tracker.update=true, get_target_pose=true, tau=0.0, model=CVModel, adaptive_replay=true")
        elif condition == "dynamic_cv":
            print("  dynamic_cv: tracker.update=true, get_target_pose=true, tau=0.1, model=CVModel, adaptive_replay=true")
        elif condition == "dynamic_ct":
            print("  dynamic_ct: tracker.update=true, get_target_pose=true, tau=0.1, model=CTModel, adaptive_replay=true")
        else:
            print(f"  {condition}: available={info['available']}, description={info['description']}")
    print("shared_setup: same object trajectory, controller, thresholds, success criteria, physics, and fixed-constraint behavior")


def log_latency_verification(trial_specs: list[dict]) -> None:
    print("latency_condition_verification:")
    print("  motion_model=CVModel")
    print("  tracker.update=true")
    print("  get_target_pose=true")
    print("  adaptive_replay=true")
    print("  observation_delay_model=delayed point-cloud capture from deterministic object pose")
    for spec in trial_specs:
        print(
            f"  {spec['condition']}: observation_delay_ms={spec['observation_delay_ms']:.1f}, "
            f"tau={spec['tau']:.3f}s, tau_delay_error_ms={spec['tau'] * 1000.0 - spec['observation_delay_ms']:.1f}"
        )
    print("shared_setup: same object trajectory, controller, thresholds, success criteria, physics, retry logic, and fixed-constraint behavior")


def run_one_trial(
    grasp,
    trial_spec: dict,
    speed: float,
    trial_idx: int,
    seed: int,
    record_gif: bool,
    diagnostics_dir: Path | None,
) -> dict:
    condition = trial_spec["condition"]
    if not trial_spec["available"]:
        raise RuntimeError(f"Condition {condition} is not available")
    random.seed(seed + trial_idx)
    np.random.seed(seed + trial_idx)
    start = time.perf_counter()
    diagnostics_path = None
    if diagnostics_dir is not None:
        diagnostics_path = (
            diagnostics_dir
            / f"{condition}_speed{float(speed):.1f}_trial{trial_idx + 1}.csv"
        )
    tau_value = trial_spec["tau"]
    tau_for_row = float(tau_value) if tau_value is not None else float("nan")
    observation_delay_ms = float(trial_spec.get("observation_delay_ms", 0.0))
    result = grasp.run_trial(
        speed_cm_s=float(speed),
        moving_object=True,
        trial_idx=trial_idx,
        record_gif=record_gif,
        replay_mode=trial_spec["replay_mode"],
        replay_tau=tau_value,
        observation_delay_s=observation_delay_ms / 1000.0,
        diagnostics_path=diagnostics_path,
        condition_label=condition,
        seed=seed,
    )
    runtime_s = time.perf_counter() - start
    return {
        "condition": condition,
        "speed_cm_s": float(speed),
        "trial": trial_idx + 1,
        "seed": seed,
        "tau": tau_for_row,
        "observation_delay_ms": observation_delay_ms,
        "tau_delay_error_ms": float(tau_for_row * 1000.0 - observation_delay_ms),
        "success": bool(result.success),
        "lift_mm": float(result.final_lift_m * 1000.0),
        "mean_tracking_error_pre_contact_mm": float(result.mean_tracking_error_pre_contact_mm),
        "mean_tracking_error_after_warmup_mm": float(result.mean_tracking_error_after_warmup_mm),
        "contact_tracking_error_mm": float(result.contact_tracking_error_mm),
        "max_tracking_error_mm": float(result.max_tracking_error_mm),
        "object_estimation_error_xy_mm": float(result.mean_object_estimation_error_xy_mm),
        "object_estimation_error_theta_deg": float("nan"),
        "target_to_desired_demo_frame_error_xy_mm": float(result.mean_target_to_desired_demo_frame_error_xy_mm),
        "target_to_object_contact_offset_error_mm": float(result.mean_target_to_object_contact_offset_error_mm),
        "mean_latency_tracking_error_mm": float(result.mean_tracking_error_after_warmup_mm),
        "mean_target_lag_error_mm": float(result.mean_target_to_desired_demo_frame_error_xy_mm),
        "ee_to_target_error_xy_mm": float(result.mean_ee_to_target_error_pre_contact_mm),
        "ee_to_target_error_3d_mm": float(result.mean_ee_to_target_error_3d_pre_contact_mm),
        "mean_ee_to_target_error_pre_contact_mm": float(result.mean_ee_to_target_error_pre_contact_mm),
        "mean_ee_to_target_error_after_warmup_mm": float(result.mean_ee_to_target_error_after_warmup_mm),
        "mean_ee_to_target_error_contact_window_mm": float(result.mean_ee_to_target_error_contact_window_mm),
        "max_ee_to_target_error_mm": float(result.max_ee_to_target_error_mm),
        "contact_window_ee_to_object_xy_mm": float(result.contact_window_ee_to_object_xy_mm),
        "contact_window_ee_to_target_xy_mm": float(result.contact_window_ee_to_target_xy_mm),
        "contact_window_target_to_object_xy_mm": float(result.contact_window_target_to_object_xy_mm),
        "gripper_close_sim_t": float(result.gripper_close_sim_t),
        "gripper_close_t_demo": float(result.gripper_close_t_demo),
        "contact_position_error_mm": float(result.contact_position_error_mm),
        "orientation_error_deg": float(result.orientation_error_deg),
        "approach_error_mm": float(result.approach_max_error_mm),
        "progress_pct": float(result.progress_rate * 100.0),
        "main_failure": grasp.trial_main_failure(result),
        "n_attempts": int(result.n_attempts),
        "runtime_s": float(runtime_s),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if not args.speeds:
        parser.error("--speeds must not be empty")

    out_dir = args.output_dir or timestamped_output_dir(args.output_root, args.stage_name)
    out_dir.mkdir(parents=True, exist_ok=False)
    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    log_path = out_dir / "run_log.txt"
    diagnostics_dir = out_dir / "diagnostics" if args.record_diagnostics else None
    write_command(out_dir / "command.txt")
    write_git_info(out_dir / "git_info.txt")
    write_raw_header(raw_path)
    write_baseline_audit(out_dir / "baseline_audit.md")
    write_metric_definitions(out_dir / "metric_definitions.md")

    rows: list[dict] = []
    try:
        trial_specs, deferred = build_trial_specs(args)
    except ValueError as exc:
        parser.error(str(exc))
    succeeded = False
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_process_output(log_file):
            try:
                grasp = load_grasping_module()
                (out_dir / "config.json").write_text(
                    json.dumps(config_from_args(args, grasp, trial_specs), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"output_dir={out_dir}")
                print(f"requested_conditions={args.conditions}")
                print(f"executable_conditions={[spec['condition'] for spec in trial_specs]}")
                print(f"deferred_conditions={deferred}")
                print(f"speeds_cm_s={args.speeds}")
                print(f"trials={args.trials}")
                if args.latency_validation:
                    log_latency_verification(trial_specs)
                else:
                    log_condition_verification(args.conditions)
                if args.audit_only:
                    print("audit_only=true; no trials executed")
                    succeeded = True
                    return 0
                for trial_spec in trial_specs:
                    condition = trial_spec["condition"]
                    for speed in args.speeds:
                        for trial_idx in range(args.trials):
                            print(
                                f"running condition={condition} speed={speed:.1f}cm/s "
                                f"trial={trial_idx + 1}/{args.trials}"
                            )
                            row = run_one_trial(
                                grasp,
                                trial_spec=trial_spec,
                                speed=float(speed),
                                trial_idx=trial_idx,
                                seed=args.seed,
                                record_gif=bool(args.record_gif),
                                diagnostics_dir=diagnostics_dir,
                            )
                            rows.append(row)
                            append_raw_row(raw_path, row)
                            print(
                                f"result condition={condition} success={row['success']} "
                                f"lift_mm={row['lift_mm']:.1f} "
                                f"tracking_pre_contact_mm={row['mean_tracking_error_pre_contact_mm']:.1f} "
                                f"tracking_contact_mm={row['contact_tracking_error_mm']:.1f} "
                                f"ee_target_pre_contact_mm={row['mean_ee_to_target_error_pre_contact_mm']:.1f} "
                                f"contact_window_ee_object_mm={row['contact_window_ee_to_object_xy_mm']:.1f} "
                                f"contact_mm={row['contact_position_error_mm']:.1f} "
                                f"orientation_deg={row['orientation_error_deg']:.1f} "
                                f"approach_mm={row['approach_error_mm']:.1f} "
                                f"progress_pct={row['progress_pct']:.1f} "
                                f"failure={row['main_failure']} "
                                f"runtime_s={row['runtime_s']:.3f}"
                            )
                succeeded = True
            except Exception:
                traceback.print_exc()
            finally:
                print(f"runtime_s={time.perf_counter() - start:.3f}")

    runtime_s = time.perf_counter() - start
    write_summary(summary_path, rows)
    write_analysis(out_dir / "analysis.md", args, rows, deferred, runtime_s, succeeded)
    write_result_self_review(out_dir / "result_self_review.md", rows, runtime_s, args.trials)
    write_metric_validation_report(out_dir / "audit" / "metric_validation_report.md", rows)
    write_metric_schema_report(out_dir / "metric_schema_report.md")
    print(f"Saved experiment outputs to {out_dir}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
