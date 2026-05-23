"""
Stage 4A.2 PyBullet tracking-only latency validation.

This runner bridges synthetic delay/tau validation and grasp-level replay. It
does not close the gripper, create fixed constraints, evaluate grasp success, or
reuse retry/lift logic. The robot tracks a static object-relative target through
DynamicAlignmentTracker.get_target_pose().
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
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pybullet as p

from dynamic_alignment.types import make_static_demo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "simulation" / "results" / "experiments"
DEFAULT_STAGE_NAME = "stage4a2_latency_tracking_only"


def load_grasping_module():
    module_path = Path(__file__).resolve().with_name("10_grasping_experiment.py")
    spec = importlib.util.spec_from_file_location("grasping_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PyBullet tracking-only latency validation.")
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--speeds", nargs="+", type=float, default=[4.0, 8.0])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--delay-type",
        choices=["observation", "control"],
        default="observation",
        help="observation delays tracker point-cloud content; control delays target command execution.",
    )
    parser.add_argument(
        "--control-delay-ms",
        nargs="+",
        type=float,
        default=[0.0, 100.0, 150.0],
        help="Control/command delays for --delay-type control.",
    )
    parser.add_argument(
        "--timestamp-modes",
        nargs="+",
        choices=["arrival", "capture"],
        default=["arrival", "capture"],
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


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def finite_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max())


def condition_specs(delay_type: str, control_delays_ms: list[float] | None = None) -> list[dict]:
    if delay_type == "control":
        specs = []
        for delay_ms in control_delays_ms or [0.0, 100.0, 150.0]:
            if delay_ms < 0:
                raise ValueError("control delays must be non-negative")
            matching_tau = delay_ms / 1000.0
            specs.append(
                {
                    "condition": f"control_delay{int(round(delay_ms))}_tau0",
                    "observation_delay_ms": 0.0,
                    "command_delay_ms": float(delay_ms),
                    "tau": 0.0,
                }
            )
            if matching_tau > 0.0:
                specs.append(
                    {
                        "condition": f"control_delay{int(round(delay_ms))}_tau{int(round(delay_ms))}",
                        "observation_delay_ms": 0.0,
                        "command_delay_ms": float(delay_ms),
                        "tau": matching_tau,
                    }
                )
            else:
                specs.append(
                    {
                        "condition": "control_delay0_tau100",
                        "observation_delay_ms": 0.0,
                        "command_delay_ms": 0.0,
                        "tau": 0.1,
                    }
                )
        return specs

    return [
        {"condition": "delay0_tau0", "observation_delay_ms": 0.0, "command_delay_ms": 0.0, "tau": 0.0},
        {"condition": "delay0_tau100", "observation_delay_ms": 0.0, "command_delay_ms": 0.0, "tau": 0.1},
        {"condition": "delay100_tau0", "observation_delay_ms": 100.0, "command_delay_ms": 0.0, "tau": 0.0},
        {"condition": "delay100_tau100", "observation_delay_ms": 100.0, "command_delay_ms": 0.0, "tau": 0.1},
        {"condition": "delay150_tau0", "observation_delay_ms": 150.0, "command_delay_ms": 0.0, "tau": 0.0},
        {"condition": "delay150_tau150", "observation_delay_ms": 150.0, "command_delay_ms": 0.0, "tau": 0.15},
    ]


def capture_delayed_cloud(grasp, box_id: int, view_matrix, projection_matrix, current_t: float, delay_s: float, speed_cm_s: float, trial_idx: int):
    cloud_t = max(0.0, current_t - delay_s)
    current_pos = grasp.moving_box_position(current_t + trial_idx * 0.17, speed_cm_s)
    if delay_s > 0.0:
        delayed_pos = grasp.moving_box_position(cloud_t + trial_idx * 0.17, speed_cm_s)
        p.resetBasePositionAndOrientation(box_id, delayed_pos.tolist(), [0.0, 0.0, 0.0, 1.0])
        p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        cloud = grasp.sim03.capture_box_cloud(view_matrix, projection_matrix)
        p.resetBasePositionAndOrientation(box_id, current_pos.tolist(), [0.0, 0.0, 0.0, 1.0])
        p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        return cloud, cloud_t
    return grasp.sim03.capture_box_cloud(view_matrix, projection_matrix), current_t


def run_tracking_trial(
    grasp,
    spec: dict,
    timestamp_mode: str,
    delay_type: str,
    speed_cm_s: float,
    trial_idx: int,
    seed: int,
    duration_s: float,
    warmup_s: float,
) -> dict:
    random.seed(seed + trial_idx)
    np.random.seed(seed + trial_idx)
    delay_s = float(spec["observation_delay_ms"]) / 1000.0
    command_delay_s = float(spec.get("command_delay_ms", 0.0)) / 1000.0
    tau_s = float(spec["tau"])
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    ee_target_errors_xy: list[float] = []
    ee_target_errors_3d: list[float] = []
    target_object_relative_errors_xy: list[float] = []
    object_estimation_errors_xy: list[float] = []
    lag_errors_mm: list[float] = []
    command_buffer: list[tuple[float, np.ndarray]] = []
    n_updates = 0
    start = time.perf_counter()
    try:
        box_id, panda_id, camera_matrices, ik_params = grasp.setup_scene()
        view_matrix, projection_matrix = camera_matrices
        reference_cloud = grasp.sim03.capture_box_cloud(view_matrix, projection_matrix)
        if reference_cloud.shape[0] < 20:
            raise RuntimeError(f"reference cloud too small: {reference_cloud.shape[0]}")

        for _ in range(60):
            grasp.command_and_step(
                panda_id,
                grasp.STATIC_BOX_POS + grasp.START_OFFSET,
                ik_params,
                grasp.OPEN_GRIPPER,
            )

        tracker = grasp.make_tracker(reference_cloud, motion_model="cv")
        demo_pose = grasp.make_pose(grasp.STATIC_BOX_POS + grasp.START_OFFSET)
        demo = make_static_demo(demo_pose)
        target_pose = demo_pose

        n_steps = int(duration_s * grasp.sim03.FPS)
        for frame_idx in range(n_steps + 1):
            sim_t = frame_idx * grasp.sim03.DT
            current_box_pos = grasp.moving_box_position(sim_t + trial_idx * 0.17, speed_cm_s)
            p.resetBasePositionAndOrientation(box_id, current_box_pos.tolist(), [0.0, 0.0, 0.0, 1.0])
            p.resetBaseVelocity(box_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

            if frame_idx > 0:
                if delay_type == "control":
                    cloud = grasp.sim03.capture_box_cloud(view_matrix, projection_matrix)
                    capture_t = sim_t
                    update_t = sim_t
                else:
                    cloud, capture_t = capture_delayed_cloud(
                        grasp,
                        box_id,
                        view_matrix,
                        projection_matrix,
                        current_t=sim_t,
                        delay_s=delay_s,
                        speed_cm_s=speed_cm_s,
                        trial_idx=trial_idx,
                    )
                    update_t = sim_t if timestamp_mode == "arrival" else capture_t
                state = tracker.current_state
                if state is not None and update_t > state.timestamp:
                    tracker.update(cloud, timestamp=update_t)
                    n_updates += 1
                target_pose = tracker.get_target_pose(demo, t_demo=0.0, tau=tau_s)

            generated_target_pos = np.asarray(target_pose[:3, 3], dtype=float)
            command_buffer.append((sim_t, generated_target_pos.copy()))
            execution_cutoff_t = max(0.0, sim_t - command_delay_s)
            executed_target_pos = command_buffer[0][1]
            for command_t, command_pos in command_buffer:
                if command_t <= execution_cutoff_t + 1e-12:
                    executed_target_pos = command_pos
                else:
                    break
            target_pos = executed_target_pos if delay_type == "control" else generated_target_pos
            grasp.command_and_step(panda_id, target_pos, ik_params, grasp.OPEN_GRIPPER)
            ee_pos = grasp.sim03.get_ee_position(panda_id)

            if sim_t >= warmup_s:
                desired_target = current_box_pos + grasp.START_OFFSET
                ee_target_errors_xy.append(float(np.linalg.norm((ee_pos - target_pos)[:2]) * 1000.0))
                ee_target_errors_3d.append(float(np.linalg.norm(ee_pos - target_pos) * 1000.0))
                target_object_relative_errors_xy.append(
                    float(np.linalg.norm((target_pos - desired_target)[:2]) * 1000.0)
                )
                current_state = tracker.current_state
                if current_state is not None:
                    estimated_object_xy = grasp.STATIC_BOX_POS[:2] + np.array(
                        [current_state.delta_x, current_state.delta_y],
                        dtype=float,
                    )
                    object_estimation_errors_xy.append(
                        float(np.linalg.norm(estimated_object_xy - current_box_pos[:2]) * 1000.0)
                    )
                    lag_errors_mm.append(
                        float(np.linalg.norm(estimated_object_xy - current_box_pos[:2]) * 1000.0)
                    )
    finally:
        p.disconnect()

    runtime_s = time.perf_counter() - start
    return {
        "timestamp_mode": timestamp_mode,
        "condition": spec["condition"],
        "speed_cm_s": speed_cm_s,
        "trial": trial_idx + 1,
        "seed": seed,
        "observation_delay_ms": float(spec["observation_delay_ms"]),
        "command_delay_ms": float(spec.get("command_delay_ms", 0.0)),
        "tau": tau_s,
        "tau_delay_error_ms": tau_s * 1000.0 - (
            float(spec.get("command_delay_ms", 0.0))
            if delay_type == "control"
            else float(spec["observation_delay_ms"])
        ),
        "n_updates": n_updates,
        "mean_ee_to_target_error_xy_mm": finite_mean(ee_target_errors_xy),
        "mean_ee_to_target_error_3d_mm": finite_mean(ee_target_errors_3d),
        "max_ee_to_target_error_xy_mm": finite_max(ee_target_errors_xy),
        "mean_target_to_object_relative_error_xy_mm": finite_mean(target_object_relative_errors_xy),
        "mean_object_estimation_error_xy_mm": finite_mean(object_estimation_errors_xy),
        "mean_lag_error_mm": finite_mean(lag_errors_mm),
        "runtime_s": runtime_s,
    }


def csv_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            serialized = {
                key: csv_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def write_summary(path: Path, rows: list[dict], delay_type: str) -> list[dict]:
    grouped: dict[tuple[str, str, float, float, float, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row["timestamp_mode"],
                row["condition"],
                row["speed_cm_s"],
                row["observation_delay_ms"],
                row["command_delay_ms"],
                row["tau"],
            ),
            [],
        ).append(row)

    summary_rows = []
    for (timestamp_mode, condition, speed, observation_delay_ms, command_delay_ms, tau), group in sorted(grouped.items()):
        active_delay_ms = command_delay_ms if delay_type == "control" else observation_delay_ms
        summary_rows.append(
            {
                "timestamp_mode": timestamp_mode,
                "condition": condition,
                "speed_cm_s": speed,
                "delay_type": delay_type,
                "observation_delay_ms": observation_delay_ms,
                "command_delay_ms": command_delay_ms,
                "tau": tau,
                "tau_delay_error_ms": tau * 1000.0 - active_delay_ms,
                "n_trials": len(group),
                "mean_ee_to_target_error_xy_mm": finite_mean(row["mean_ee_to_target_error_xy_mm"] for row in group),
                "mean_ee_to_target_error_3d_mm": finite_mean(row["mean_ee_to_target_error_3d_mm"] for row in group),
                "max_ee_to_target_error_xy_mm": finite_max(row["max_ee_to_target_error_xy_mm"] for row in group),
                "mean_target_to_object_relative_error_xy_mm": finite_mean(row["mean_target_to_object_relative_error_xy_mm"] for row in group),
                "mean_object_estimation_error_xy_mm": finite_mean(row["mean_object_estimation_error_xy_mm"] for row in group),
                "mean_lag_error_mm": finite_mean(row["mean_lag_error_mm"] for row in group),
                "mean_runtime_s": finite_mean(row["runtime_s"] for row in group),
                "target_error_reduction_pct": float("nan"),
            }
        )

    index = {
        (
            row["timestamp_mode"],
            row["speed_cm_s"],
            row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"],
            row["tau"],
        ): row
        for row in summary_rows
    }
    for row in summary_rows:
        active_delay_ms = row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"]
        if active_delay_ms <= 0.0 or row["tau"] <= 0.0:
            continue
        baseline = index.get(
            (
                row["timestamp_mode"],
                row["speed_cm_s"],
                active_delay_ms,
                0.0,
            )
        )
        if baseline is None:
            continue
        base_error = baseline["mean_target_to_object_relative_error_xy_mm"]
        matched_error = row["mean_target_to_object_relative_error_xy_mm"]
        if np.isfinite(base_error) and base_error > 0 and np.isfinite(matched_error):
            row["target_error_reduction_pct"] = 100.0 * (base_error - matched_error) / base_error

    write_csv(path, summary_rows)
    return summary_rows


def conclusion_from_summary(summary_rows: list[dict], delay_type: str) -> str:
    comparable = [
        row
        for row in summary_rows
        if (row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"]) > 0.0
        and abs(
            row["tau"] * 1000.0
            - (row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"])
        ) < 1e-6
        and np.isfinite(row["target_error_reduction_pct"])
    ]
    if not comparable:
        return "CONTROL_DELAY_IMPLEMENTATION_UNTRUSTED" if delay_type == "control" else "TRACKING_ONLY_IMPLEMENTATION_UNTRUSTED"
    improved = [row for row in comparable if row["target_error_reduction_pct"] > 0.0]
    if len(improved) == len(comparable):
        return "TAU_VALID_UNDER_CONTROL_DELAY" if delay_type == "control" else "TAU_VALID_IN_TRACKING_ONLY"
    if delay_type == "control":
        if len(improved) == 0:
            return "STOP_TAU_PROCEED_TO_CONTACT_GATING"
        return "TAU_NOT_VALIDATED_IN_PYBULLET_TRACKING"
    if any(row["timestamp_mode"] == "capture" for row in comparable) and any(
        row["target_error_reduction_pct"] > 0.0 for row in comparable
    ):
        return "TIMESTAMP_SEMANTICS_NEEDS_FIX"
    return "TAU_SYNTHETIC_ONLY_NOT_TRACKING"


def write_analysis(path: Path, summary_rows: list[dict], runtime_s: float, trials: int, delay_type: str) -> None:
    conclusion = conclusion_from_summary(summary_rows, delay_type)
    matched_rows = [
        row
        for row in summary_rows
        if (row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"]) > 0.0
        and abs(
            row["tau"] * 1000.0
            - (row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"])
        ) < 1e-6
    ]
    delay0_rows = [
        row for row in summary_rows
        if (row["command_delay_ms"] if delay_type == "control" else row["observation_delay_ms"]) == 0.0
    ]
    reductions = {
        f"{row['timestamp_mode']} {row['speed_cm_s']:.1f}cm/s "
        f"{(row['command_delay_ms'] if delay_type == 'control' else row['observation_delay_ms']):.0f}ms":
        row["target_error_reduction_pct"]
        for row in matched_rows
    }
    if delay_type == "control":
        write_control_delay_analysis(path, summary_rows, runtime_s, trials, reductions, delay0_rows, conclusion)
        return
    lines = [
        "# Stage 4A.2 PyBullet Tracking-Only Latency Validation",
        "",
        f"- status: succeeded",
        f"- runtime_s: {runtime_s:.3f}",
        f"- trials_per_condition: {trials}",
        "- scope: tracking-only PyBullet validation; no gripper close, no fixed constraint, no grasp success, no retry/lift logic.",
        "- target: static object-relative pose generated through DynamicAlignmentTracker.get_target_pose().",
        "",
        "## Questions",
        "",
        "1. Does matched tau reduce EE-to-target tracking error under injected observation delay?",
        "The primary latency metric here is target-to-object relative error, because EE-to-target also includes robot servo tracking limits. Matched-tau target-error reductions are: "
        + json.dumps(reductions, sort_keys=True),
        "",
        "2. Does tau hurt or fail to help when delay=0?",
        f"Delay=0 rows are present for direct inspection: {[(row['timestamp_mode'], row['speed_cm_s'], row['tau'], row['mean_target_to_object_relative_error_xy_mm']) for row in delay0_rows]}.",
        "",
        "3. Is the effect stronger at 8 cm/s than 4 cm/s?",
        "Compare target_error_reduction_pct by speed in summary.csv. The expected absolute uncompensated lag grows with speed.",
        "",
        "4. Does timestamp semantics appear consistent?",
        "Both arrival and capture timestamp modes are recorded. Capture timestamp is physically cleaner for delayed observations; arrival timestamp matches the Stage 4A grasp smoke implementation.",
        "",
        "5. If synthetic tau works but PyBullet tracking-only does not, what is the likely bottleneck?",
        "Likely bottlenecks are point-cloud measurement noise, estimator/PCA behavior, robot servo lag, camera segmentation, or timestamp/content mismatch. Grasp success is not involved in this run.",
        "",
        "6. Is it justified to retry grasp-level Stage 4A after this?",
        "Only if matched tau reduces target-to-object relative error in this tracking-only run. Otherwise fix timestamp semantics or tracking implementation first.",
        "",
        "7. Should tau be kept as a validated latency module, or only as a theoretical/synthetic module for now?",
        "This depends on the conclusion below. Synthetic validation alone was positive; tracking-only validation determines whether it survives PyBullet perception/control.",
        "",
        conclusion,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_control_delay_analysis(
    path: Path,
    summary_rows: list[dict],
    runtime_s: float,
    trials: int,
    reductions: dict[str, float],
    delay0_rows: list[dict],
    conclusion: str,
) -> None:
    matched_rows = [
        row
        for row in summary_rows
        if row["command_delay_ms"] > 0.0
        and abs(row["tau"] * 1000.0 - row["command_delay_ms"]) < 1e-6
    ]
    ee_reductions = {}
    index = {
        (row["timestamp_mode"], row["speed_cm_s"], row["command_delay_ms"], row["tau"]): row
        for row in summary_rows
    }
    for row in matched_rows:
        baseline = index.get((row["timestamp_mode"], row["speed_cm_s"], row["command_delay_ms"], 0.0))
        if baseline is None:
            continue
        base_error = baseline["mean_ee_to_target_error_xy_mm"]
        matched_error = row["mean_ee_to_target_error_xy_mm"]
        if np.isfinite(base_error) and base_error > 0.0 and np.isfinite(matched_error):
            ee_reductions[
                f"{row['speed_cm_s']:.1f}cm/s {row['command_delay_ms']:.0f}ms"
            ] = 100.0 * (base_error - matched_error) / base_error

    lines = [
        "# Stage 4A.3 Control-Delay Tracking Diagnostic",
        "",
        "- status: succeeded",
        f"- runtime_s: {runtime_s:.3f}",
        f"- trials_per_condition: {trials}",
        "- scope: tracking-only PyBullet diagnostic; no gripper close, no fixed constraint, no grasp success, no retry/lift logic.",
        "",
        "## Questions",
        "",
        "1. How was control delay implemented?",
        "The tracker receives current object point clouds with no artificial observation delay. Each frame generates a current target through `DynamicAlignmentTracker.get_target_pose(..., tau)`. The robot command path stores generated targets in a time-indexed buffer and executes the newest target whose generation time is <= `sim_t - command_delay`. No `time.sleep()` is used and the PyBullet timestep is unchanged.",
        "",
        "2. Does matched tau reduce EE-to-target tracking error under command/control delay?",
        "Matched-tau EE-to-target XY error reductions are: " + json.dumps(ee_reductions, sort_keys=True),
        "",
        "3. Does tau hurt or fail to help when delay=0?",
        f"Delay=0 rows are present for direct inspection: {[(row['speed_cm_s'], row['tau'], row['mean_ee_to_target_error_xy_mm'], row['mean_target_to_object_relative_error_xy_mm']) for row in delay0_rows]}.",
        "",
        "4. Is the effect stronger at 8 cm/s than 4 cm/s?",
        "Compare target_error_reduction_pct and EE reduction by speed in summary.csv. With only one trial per cell, treat speed trends as smoke diagnostics only.",
        "",
        "5. Does this support the interpretation that tau corresponds to control/actuation delay?",
        "Only if matched tau reduces tracking errors for command-delay cells without hurting delay=0 cells. This run does not model real hardware control latency.",
        "",
        "6. If tau still fails, what is the likely bottleneck?",
        "Likely bottlenecks are target prediction quality under PyBullet point-cloud tracking, robot servo lag, target buffering semantics, or the fact that `tau` shifts target generation while the robot controller still has its own tracking dynamics.",
        "",
        "7. Should we continue investigating tau, or stop tau work and proceed to contact-aware temporal gating?",
        "If matched tau does not consistently reduce tracking-only errors, stop treating tau as a PyBullet-validated improvement path and prioritize contact-aware temporal gating after approval.",
        "",
        conclusion,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")

    out_dir = args.output_dir or timestamped_output_dir(args.output_root, args.stage_name)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_command(out_dir / "command.txt")
    write_git_info(out_dir / "git_info.txt")
    specs = condition_specs(args.delay_type, args.control_delay_ms)
    timestamp_modes = ["control"] if args.delay_type == "control" else args.timestamp_modes
    config = {
        "stage_name": args.stage_name,
        "delay_type": args.delay_type,
        "speeds_cm_s": args.speeds,
        "trials": args.trials,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "seed": args.seed,
        "timestamp_modes": timestamp_modes,
        "conditions": specs,
        "no_gripper_close": True,
        "no_fixed_constraint": True,
        "no_grasp_success": True,
        "physics_controller_thresholds_success_criteria_unchanged": True,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict] = []
    start = time.perf_counter()
    with (out_dir / "run_log.txt").open("w", encoding="utf-8") as log_file:
        with redirect_process_output(log_file):
            print(f"output_dir={out_dir}")
            print("running PyBullet tracking-only latency validation")
            grasp = load_grasping_module()
            for timestamp_mode in timestamp_modes:
                for spec in specs:
                    for speed in args.speeds:
                        for trial_idx in range(args.trials):
                            print(
                                f"running mode={timestamp_mode} condition={spec['condition']} "
                                f"speed={speed:.1f} trial={trial_idx + 1}/{args.trials}"
                            )
                            row = run_tracking_trial(
                                grasp=grasp,
                                spec=spec,
                                timestamp_mode=timestamp_mode,
                                delay_type=args.delay_type,
                                speed_cm_s=float(speed),
                                trial_idx=trial_idx,
                                seed=args.seed,
                                duration_s=float(args.duration_s),
                                warmup_s=float(args.warmup_s),
                            )
                            rows.append(row)
                            print(
                                f"result mean_target_object_mm={row['mean_target_to_object_relative_error_xy_mm']:.3f} "
                                f"mean_ee_target_xy_mm={row['mean_ee_to_target_error_xy_mm']:.3f} "
                                f"mean_object_est_mm={row['mean_object_estimation_error_xy_mm']:.3f} "
                                f"runtime_s={row['runtime_s']:.3f}"
                            )
            print(f"runtime_s={time.perf_counter() - start:.3f}")

    runtime_s = time.perf_counter() - start
    write_csv(out_dir / "raw_results.csv", rows)
    summary_rows = write_summary(out_dir / "summary.csv", rows, args.delay_type)
    write_analysis(out_dir / "analysis.md", summary_rows, runtime_s, args.trials, args.delay_type)
    print(f"Saved tracking-only latency outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
