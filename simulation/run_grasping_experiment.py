"""
Reproducible runner for the existing PyBullet grasping experiment.

This wrapper intentionally calls simulation/10_grasping_experiment.py without
changing its control, physics, trajectory, thresholds, or success criteria. Its
only purpose is to run a configurable subset and save a complete experiment
record under simulation/results/experiments/.
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
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "simulation" / "results" / "experiments"


def load_grasping_module():
    module_path = Path(__file__).resolve().with_name("10_grasping_experiment.py")
    spec = importlib.util.spec_from_file_location("grasping_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def redirect_process_output(log_file):
    """Redirect Python and native-extension stdout/stderr to log_file."""
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


def parse_speeds(value: str) -> list[float]:
    speeds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not speeds:
        raise argparse.ArgumentTypeError("speed list cannot be empty")
    return speeds


def parse_conditions(value: str) -> list[tuple[str, bool]]:
    normalized = value.lower().strip()
    if normalized == "both":
        return [("Static baseline", False), ("Moving object", True)]
    if normalized == "static":
        return [("Static baseline", False)]
    if normalized == "moving":
        return [("Moving object", True)]
    raise argparse.ArgumentTypeError("condition must be one of: both, static, moving")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a logged subset of the existing PyBullet grasping experiment."
    )
    parser.add_argument("--stage-name", default="stage2_smoke")
    parser.add_argument("--speeds", type=parse_speeds, default=parse_speeds("2.0"))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--conditions", type=parse_conditions, default=parse_conditions("both"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
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


def run_git_command(args: list[str]) -> str:
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
        f"branch: {run_git_command(['branch', '--show-current'])}",
        f"commit: {run_git_command(['rev-parse', 'HEAD'])}",
        "status:",
        run_git_command(["status", "--short"]) or "(clean)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command(path: Path) -> None:
    command = " ".join([sys.executable, *sys.argv])
    path.write_text(command + "\n", encoding="utf-8")


def config_from_args(args: argparse.Namespace, grasp) -> dict:
    return {
        "stage_name": args.stage_name,
        "speeds_cm_s": args.speeds,
        "trials_per_condition": args.trials,
        "conditions": [label for label, _ in args.conditions],
        "seed": args.seed,
        "record_gif": bool(args.record_gif),
        "plots": not bool(args.no_plots),
        "core_method_unchanged": True,
        "source_script": "simulation/10_grasping_experiment.py",
        "notes": [
            "This runner calls the existing run_trial function.",
            "It does not change controller logic, object trajectory, thresholds, or success criteria.",
            "The grasp experiment uses a deterministic fixed PyBullet constraint when the gripper closes near the box.",
        ],
        "method_constants_recorded": {
            "tau_s": grasp.TAU,
            "test_speeds_default_cm_s": list(grasp.TEST_SPEEDS_CM_S),
            "default_trials": grasp.N_TRIALS,
            "replay_duration_s": grasp.REPLAY_DURATION,
            "contact_position_tolerance_m": grasp.CONTACT_POSITION_TOLERANCE_M,
            "approach_horizontal_tolerance_m": grasp.APPROACH_HORIZONTAL_TOLERANCE_M,
            "orientation_tolerance_deg": grasp.ORIENTATION_TOLERANCE_DEG,
            "lift_success_margin_m": grasp.LIFT_SUCCESS_MARGIN,
            "grasp_attach_distance_m": grasp.GRASP_ATTACH_DISTANCE,
            "max_grasp_attempts": grasp.MAX_GRASP_ATTEMPTS,
            "fixed_constraint_simplification": True,
        },
    }


def initialize_raw_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "condition",
                "speed_cm_s",
                "trial",
                "success",
                "lift_mm",
                "max_lift_mm",
                "pos_err_mm",
                "ori_err_deg",
                "approach_mm",
                "progress_pct",
                "main_failure",
                "n_attempts",
            ]
        )


def write_raw_row(path: Path, grasp, condition: str, speed: float, trial_idx: int, result) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                condition,
                f"{speed:.1f}",
                trial_idx + 1,
                int(result.success),
                f"{result.final_lift_m * 1000.0:.3f}",
                f"{result.max_lift_m * 1000.0:.3f}",
                f"{result.contact_position_error_mm:.3f}",
                f"{result.orientation_error_deg:.3f}",
                f"{result.approach_max_error_mm:.3f}",
                f"{result.progress_rate * 100.0:.3f}",
                grasp.trial_main_failure(result),
                result.n_attempts,
            ]
        )


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
        "n_trials",
        "success_rate_pct",
        "mean_lift_mm",
        "mean_max_lift_mm",
        "mean_pos_err_mm_finite",
        "mean_ori_err_deg_finite",
        "mean_approach_mm_finite",
        "mean_progress_pct",
        "failure_none",
        "failure_lift",
        "failure_position",
        "failure_orientation",
        "failure_approach",
        "failure_attempt_limit",
        "failure_unknown",
    ]
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["speed_cm_s"]), []).append(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (condition, speed), group in sorted(grouped.items()):
            failures = {name: 0 for name in ["none", "lift", "position", "orientation", "approach", "attempt_limit", "unknown"]}
            for row in group:
                failures[row["main_failure"]] = failures.get(row["main_failure"], 0) + 1
            writer.writerow(
                {
                    "condition": condition,
                    "speed_cm_s": f"{speed:.1f}",
                    "n_trials": len(group),
                    "success_rate_pct": f"{np.mean([row['success'] for row in group]) * 100.0:.3f}",
                    "mean_lift_mm": f"{np.mean([row['lift_mm'] for row in group]):.3f}",
                    "mean_max_lift_mm": f"{np.mean([row['max_lift_mm'] for row in group]):.3f}",
                    "mean_pos_err_mm_finite": f"{finite_mean(row['pos_err_mm'] for row in group):.3f}",
                    "mean_ori_err_deg_finite": f"{finite_mean(row['ori_err_deg'] for row in group):.3f}",
                    "mean_approach_mm_finite": f"{finite_mean(row['approach_mm'] for row in group):.3f}",
                    "mean_progress_pct": f"{np.mean([row['progress_pct'] for row in group]):.3f}",
                    "failure_none": failures.get("none", 0),
                    "failure_lift": failures.get("lift", 0),
                    "failure_position": failures.get("position", 0),
                    "failure_orientation": failures.get("orientation", 0),
                    "failure_approach": failures.get("approach", 0),
                    "failure_attempt_limit": failures.get("attempt_limit", 0),
                    "failure_unknown": failures.get("unknown", 0),
                }
            )


def write_analysis(path: Path, args: argparse.Namespace, rows: list[dict], runtime_s: float, succeeded: bool) -> None:
    lines = [
        f"# {args.stage_name}",
        "",
        "This is a reproducibility/logging run for the existing PyBullet grasping experiment.",
        "No controller logic, physics assumptions, object trajectory, thresholds, or success criteria were changed by this runner.",
        "",
        f"- status: {'succeeded' if succeeded else 'failed'}",
        f"- runtime_s: {runtime_s:.3f}",
        f"- speeds_cm_s: {args.speeds}",
        f"- trials_per_condition: {args.trials}",
        f"- conditions: {[label for label, _ in args.conditions]}",
        f"- seed: {args.seed}",
        "",
        "Important limitation: grasp attachment is simplified with a deterministic fixed PyBullet constraint when the gripper closes near the box.",
        "This run should not be interpreted as real contact or hardware validation.",
        "",
        "Per-condition results:",
    ]
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["speed_cm_s"]), []).append(row)
    for (condition, speed), group in sorted(grouped.items()):
        success_rate = np.mean([row["success"] for row in group]) * 100.0
        failures: dict[str, int] = {}
        for row in group:
            failures[row["main_failure"]] = failures.get(row["main_failure"], 0) + 1
        lines.append(
            f"- {condition}, {speed:.1f} cm/s: n={len(group)}, success_rate={success_rate:.1f}%, failures={failures}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["speed_cm_s"]), []).append(row)

    labels = []
    rates = []
    for (condition, speed), group in sorted(grouped.items()):
        labels.append(f"{condition}\n{speed:.1f} cm/s")
        rates.append(np.mean([row["success"] for row in group]) * 100.0)

    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(labels)), 4.5))
    ax.bar(np.arange(len(labels)), rates, color="tab:blue", edgecolor="black", alpha=0.85)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0.0, 105.0)
    ax.set_title("Logged Grasping Experiment Summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / "success_rate.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")

    out_dir = args.output_dir or timestamped_output_dir(args.output_root, args.stage_name)
    out_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = out_dir / "plots"
    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    log_path = out_dir / "run_log.txt"

    write_command(out_dir / "command.txt")
    write_git_info(out_dir / "git_info.txt")
    initialize_raw_csv(raw_path)

    rows: list[dict] = []
    start = time.perf_counter()
    succeeded = False
    grasp = None
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_process_output(log_file):
            try:
                grasp = load_grasping_module()
                random.seed(args.seed)
                np.random.seed(args.seed)
                (out_dir / "config.json").write_text(
                    json.dumps(config_from_args(args, grasp), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"output_dir={out_dir}")
                print(f"speeds_cm_s={args.speeds}")
                print(f"conditions={[label for label, _ in args.conditions]}")
                print(f"trials={args.trials}")
                for speed in args.speeds:
                    for condition, moving in args.conditions:
                        for trial_idx in range(args.trials):
                            print(
                                f"running condition={condition} speed={speed:.1f}cm/s "
                                f"trial={trial_idx + 1}/{args.trials}"
                            )
                            result = grasp.run_trial(
                                float(speed),
                                bool(moving),
                                trial_idx,
                                record_gif=bool(args.record_gif),
                            )
                            main_failure = grasp.trial_main_failure(result)
                            write_raw_row(raw_path, grasp, condition, float(speed), trial_idx, result)
                            row = {
                                "condition": condition,
                                "speed_cm_s": float(speed),
                                "trial": trial_idx + 1,
                                "success": bool(result.success),
                                "lift_mm": float(result.final_lift_m * 1000.0),
                                "max_lift_mm": float(result.max_lift_m * 1000.0),
                                "pos_err_mm": float(result.contact_position_error_mm),
                                "ori_err_deg": float(result.orientation_error_deg),
                                "approach_mm": float(result.approach_max_error_mm),
                                "progress_pct": float(result.progress_rate * 100.0),
                                "main_failure": main_failure,
                                "n_attempts": int(result.n_attempts),
                            }
                            rows.append(row)
                            print(
                                f"result success={row['success']} lift_mm={row['lift_mm']:.1f} "
                                f"pos_err_mm={row['pos_err_mm']:.1f} "
                                f"ori_err_deg={row['ori_err_deg']:.1f} "
                                f"approach_mm={row['approach_mm']:.1f} "
                                f"progress_pct={row['progress_pct']:.1f} "
                                f"main_failure={main_failure}"
                            )
                succeeded = True
            except Exception:
                traceback.print_exc()
            finally:
                runtime_s = time.perf_counter() - start
                print(f"runtime_s={runtime_s:.3f}")

    runtime_s = time.perf_counter() - start
    write_summary(summary_path, rows)
    if not args.no_plots:
        maybe_plot_summary(plots_dir, rows)
    write_analysis(out_dir / "analysis.md", args, rows, runtime_s, succeeded)
    print(f"Saved experiment outputs to {out_dir}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
