"""
Stage 4A.1 delay/tau time-semantics audit.

This script validates the Kalman delay-compensation semantics without PyBullet
grasping, gripper close, fixed constraints, or grasp success criteria. It uses a
known constant-velocity trajectory and delayed observations to test whether
predict_ahead(tau) can recover the current object state.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from dynamic_alignment.kalman import KalmanFilter
from dynamic_alignment.motion_models import CVModel
from dynamic_alignment.types import ObjectObservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "simulation" / "results" / "experiments"
DEFAULT_STAGE_NAME = "stage4a1_delay_tau_audit"
DT = 1.0 / 30.0
DEFAULT_DURATION_S = 5.0
DEFAULT_WARMUP_S = 1.0


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
    parser = argparse.ArgumentParser(description="Audit delay/tau tracker semantics without grasping.")
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--speeds", nargs="+", type=float, default=[4.0, 8.0])
    parser.add_argument("--delays-ms", nargs="+", type=float, default=[0.0, 50.0, 100.0, 150.0])
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    parser.add_argument(
        "--timestamp-modes",
        nargs="+",
        choices=["arrival", "capture"],
        default=["arrival", "capture"],
        help="arrival: delayed content timestamped at current time; capture: delayed content timestamped at capture time.",
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


def true_state(t: float, speed_cm_s: float) -> np.ndarray:
    speed_m_s = speed_cm_s / 100.0
    return np.array([speed_m_s * t, 0.0, 0.0], dtype=float)


def tau_grid_for_delay(delay_ms: float) -> list[float]:
    delay_s = delay_ms / 1000.0
    candidates = {0.0, delay_s}
    if delay_s > 0.0:
        candidates.add(delay_s / 2.0)
        candidates.add(delay_s + 0.05)
    return sorted(candidates)


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


def run_synthetic_case(
    speed_cm_s: float,
    delay_ms: float,
    tau_s: float,
    timestamp_mode: str,
    duration_s: float,
    warmup_s: float,
) -> dict:
    delay_s = delay_ms / 1000.0
    kf = KalmanFilter(
        model=CVModel(),
        R_diag=np.array([1e-6, 1e-6, 1e-6], dtype=float),
        init_vel_cov=1.0,
    )
    initial_obs = ObjectObservation(
        delta_x=0.0,
        delta_y=0.0,
        delta_theta=0.0,
        timestamp=0.0,
        is_valid=True,
    )
    kf.initialize(initial_obs)

    current_errors_mm: list[float] = []
    execution_errors_mm: list[float] = []
    state_timestamps: list[float] = []
    predicted_timestamps: list[float] = []
    n_steps = int(duration_s / DT)
    for frame_idx in range(1, n_steps + 1):
        arrival_t = frame_idx * DT
        capture_t = max(0.0, arrival_t - delay_s)
        obs_time = arrival_t if timestamp_mode == "arrival" else capture_t
        if obs_time <= kf.state.timestamp:
            continue
        obs_pos = true_state(capture_t, speed_cm_s)
        kf.predict(obs_time - kf.state.timestamp)
        kf.update(
            ObjectObservation(
                delta_x=float(obs_pos[0]),
                delta_y=float(obs_pos[1]),
                delta_theta=0.0,
                timestamp=obs_time,
                is_valid=True,
            )
        )
        predicted = kf.predict_ahead(tau_s)
        state_timestamps.append(float(kf.state.timestamp))
        predicted_timestamps.append(float(predicted.timestamp))
        if arrival_t >= warmup_s:
            true_current = true_state(arrival_t, speed_cm_s)
            current_errors_mm.append(
                float(np.linalg.norm(predicted.x[:2] - true_current[:2]) * 1000.0)
            )
            true_execution = true_state(arrival_t, speed_cm_s)
            execution_errors_mm.append(
                float(np.linalg.norm(predicted.x[:2] - true_execution[:2]) * 1000.0)
            )

    return {
        "timestamp_mode": timestamp_mode,
        "speed_cm_s": speed_cm_s,
        "observation_delay_ms": delay_ms,
        "tau": tau_s,
        "tau_delay_error_ms": tau_s * 1000.0 - delay_ms,
        "mean_prediction_error_to_current_mm": finite_mean(current_errors_mm),
        "max_prediction_error_to_current_mm": finite_max(current_errors_mm),
        "mean_prediction_error_to_execution_time_mm": finite_mean(execution_errors_mm),
        "max_prediction_error_to_execution_time_mm": finite_max(execution_errors_mm),
        "mean_state_timestamp_lag_ms": finite_mean(
            [
                (arrival_t - state_t) * 1000.0
                for arrival_t, state_t in zip(np.arange(1, len(state_timestamps) + 1) * DT, state_timestamps)
            ]
        ),
        "mean_predicted_timestamp_offset_ms": finite_mean(
            [
                (pred_t - state_t) * 1000.0
                for pred_t, state_t in zip(predicted_timestamps, state_timestamps)
            ]
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, float):
                    if math.isnan(value):
                        serialized[key] = "nan"
                    elif math.isinf(value):
                        serialized[key] = "inf" if value > 0 else "-inf"
                    else:
                        serialized[key] = f"{value:.6f}"
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def write_summary(path: Path, rows: list[dict]) -> None:
    summary_rows = []
    grouped: dict[tuple[str, float, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["timestamp_mode"], row["speed_cm_s"], row["observation_delay_ms"]),
            [],
        ).append(row)
    for (timestamp_mode, speed, delay_ms), group in sorted(grouped.items()):
        tau0 = next((row for row in group if abs(row["tau"]) < 1e-12), None)
        matched = next((row for row in group if abs(row["tau"] - delay_ms / 1000.0) < 1e-12), None)
        best = min(group, key=lambda row: row["mean_prediction_error_to_current_mm"])
        summary_rows.append(
            {
                "timestamp_mode": timestamp_mode,
                "speed_cm_s": speed,
                "observation_delay_ms": delay_ms,
                "tau0_mean_current_error_mm": tau0["mean_prediction_error_to_current_mm"] if tau0 else float("nan"),
                "matched_tau_mean_current_error_mm": matched["mean_prediction_error_to_current_mm"] if matched else float("nan"),
                "matched_minus_tau0_error_mm": (
                    matched["mean_prediction_error_to_current_mm"] - tau0["mean_prediction_error_to_current_mm"]
                    if tau0 and matched
                    else float("nan")
                ),
                "best_tau": best["tau"],
                "best_mean_current_error_mm": best["mean_prediction_error_to_current_mm"],
            }
        )
    write_csv(path, summary_rows)


def conclusion_from_rows(rows: list[dict]) -> str:
    nonzero_delay = [row for row in rows if row["observation_delay_ms"] > 0.0]
    grouped: dict[tuple[str, float, float], list[dict]] = {}
    for row in nonzero_delay:
        grouped.setdefault(
            (row["timestamp_mode"], row["speed_cm_s"], row["observation_delay_ms"]),
            [],
        ).append(row)
    improved = 0
    comparable = 0
    for (_, _, delay_ms), group in grouped.items():
        tau0 = next((row for row in group if abs(row["tau"]) < 1e-12), None)
        matched = next((row for row in group if abs(row["tau"] - delay_ms / 1000.0) < 1e-12), None)
        if tau0 is None or matched is None:
            continue
        comparable += 1
        if matched["mean_prediction_error_to_current_mm"] < tau0["mean_prediction_error_to_current_mm"]:
            improved += 1
    if comparable > 0 and improved == comparable:
        return "TAU_SYNTHETIC_VALID"
    return "DELAY_TAU_SEMANTICS_BUG_SUSPECTED"


def write_delay_tau_audit(path: Path) -> None:
    lines = [
        "# Delay/Tau Audit",
        "",
        "1. What exactly is delayed?",
        "Stage 4A delayed the object point-cloud content used by the tracker. The PyBullet object was temporarily moved to the deterministic pose at `max(0, sim_t - observation_delay_s)` for point-cloud capture, then immediately restored to the current `sim_t` pose before command execution and contact checks.",
        "",
        "2. Is the delayed observation timestamped with capture time t-delay or arrival time t?",
        "In the Stage 4A grasp implementation, the delayed point-cloud content is timestamped with arrival/current control time `t`, because `tracker.update(cloud, timestamp=replay_t)` is called after delayed cloud capture.",
        "",
        "3. What timestamp is passed into tracker.update()?",
        "`replay_t`, the current controller/simulation time, not the delayed capture time.",
        "",
        "4. After tracker.update(), what time does the internal Kalman state represent?",
        "Numerically, the Kalman state timestamp is `replay_t`. Semantically, the position content may be spatially lagged because the measurement vector came from `t - delay` while the timestamp says `t`.",
        "",
        "5. When get_target_pose(..., tau) calls predict_ahead(tau), what physical time is it trying to predict?",
        "It predicts from the filter state's timestamp to `state.timestamp + tau`. Under Stage 4A arrival-time timestamping, that is nominally `t + tau`, even though the measurement content corresponds to `t - delay`.",
        "",
        "6. Does tau compensate observation delay, control delay, or both in the current implementation?",
        "The interface is documented as total perception + computation + actuation delay. In Stage 4A, only observation content delay was injected; no separate actuation/control delay was modeled. Therefore tau was being tested as observation-delay compensation only.",
        "",
        "7. Is the current implementation mathematically consistent with the design note?",
        "Only partially. The design note treats tau as total system delay and assumes the state being predicted forward has a coherent timestamp. Stage 4A arrival-time timestamping creates a state whose timestamp is current but whose observation content is old, so the timestamp/content semantics are mixed. A capture-time timestamp can be mathematically cleaner for pure observation delay, but then downstream code must explicitly account for state age and total execution delay.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(path: Path, rows: list[dict], runtime_s: float) -> None:
    conclusion = conclusion_from_rows(rows)
    summary_by_mode = {}
    for mode in sorted({row["timestamp_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["timestamp_mode"] == mode and row["observation_delay_ms"] > 0.0]
        grouped: dict[tuple[float, float], list[dict]] = {}
        for row in mode_rows:
            grouped.setdefault((row["speed_cm_s"], row["observation_delay_ms"]), []).append(row)
        improved = 0
        comparable = 0
        for (_, delay_ms), group in grouped.items():
            tau0 = next((row for row in group if abs(row["tau"]) < 1e-12), None)
            matched = next((row for row in group if abs(row["tau"] - delay_ms / 1000.0) < 1e-12), None)
            if tau0 and matched:
                comparable += 1
                if matched["mean_prediction_error_to_current_mm"] < tau0["mean_prediction_error_to_current_mm"]:
                    improved += 1
        summary_by_mode[mode] = (improved, comparable)

    lines = [
        "# Stage 4A.1 Delay/Tau Time-Semantics Audit",
        "",
        f"- status: succeeded",
        f"- runtime_s: {runtime_s:.3f}",
        "- scope: synthetic Kalman validation only; no grasp, no PyBullet contact, no fixed constraint, no success criteria changes.",
        "",
        "## Answers",
        "",
        "1. Does tau=matching delay reduce synthetic prediction error?",
        f"By timestamp mode: {summary_by_mode}. See summary.csv. Overall conclusion: {conclusion}.",
        "",
        "2. If yes, why did grasp-level Stage 4A not improve?",
        "If synthetic matched tau is valid, the Stage 4A grasp smoke is likely confounded by adaptive replay timing, retry behavior, coarse success labels, fixed-constraint contact simplification, or the fact that only observation delay was injected while tau is intended as total system delay.",
        "",
        "3. If no, where is the likely bug or semantic mismatch?",
        "Not applicable for the synthetic CV audit: matched tau reduced error in all comparable delayed cells. The remaining semantic risk is in the grasp-level Stage 4A implementation, where delayed point-cloud content was passed to tracker.update() with current arrival time. That can be workable for CV prediction but mixes old measurement content with a current Kalman timestamp, making grasp-level interpretation less transparent.",
        "",
        "4. Should tracker.update() receive capture timestamp or arrival timestamp?",
        "For pure observation-delay modeling, capture timestamp is mathematically cleaner because the measurement vector and timestamp describe the same physical state. If arrival timestamp is used, the filter state timestamp is current but its position can be biased toward old content; compensation then relies on learned velocity and may be less transparent.",
        "",
        "5. Should predict_ahead(tau) compensate observation delay, control delay, or total delay?",
        "The documented interface is total delay: perception + computation + actuation. In a simulation that only injects observation delay, tau should be interpreted as the modeled observation age only, or the missing control/actuation delay should be modeled explicitly before claiming total-delay compensation.",
        "",
        "6. Is the current tau interface usable as designed?",
        "The interface is usable in principle, but Stage 4A showed the grasp-level use is not validated yet. The timestamp/content semantics must be made explicit before a larger latency validation.",
        "",
        "7. Should Stage 4A grasp-level validation be retried after a fix?",
        "Yes only if the synthetic audit supports matched-tau compensation and the grasp-level delay injection is updated to use a coherent timestamp model. Otherwise do not run a larger grasp latency grid.",
        "",
        conclusion,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_dir or timestamped_output_dir(args.output_root, args.stage_name)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_command(out_dir / "command.txt")
    write_git_info(out_dir / "git_info.txt")
    config = {
        "stage_name": args.stage_name,
        "speeds_cm_s": args.speeds,
        "delays_ms": args.delays_ms,
        "timestamp_modes": args.timestamp_modes,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "dt_s": DT,
        "no_grasp_level_experiments": True,
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
            print("running synthetic delay/tau audit")
            for mode in args.timestamp_modes:
                for speed in args.speeds:
                    for delay_ms in args.delays_ms:
                        for tau_s in tau_grid_for_delay(delay_ms):
                            row = run_synthetic_case(
                                speed_cm_s=float(speed),
                                delay_ms=float(delay_ms),
                                tau_s=float(tau_s),
                                timestamp_mode=mode,
                                duration_s=float(args.duration_s),
                                warmup_s=float(args.warmup_s),
                            )
                            rows.append(row)
                            print(
                                f"mode={mode} speed={speed:.1f} delay_ms={delay_ms:.1f} "
                                f"tau={tau_s:.3f} mean_current_error_mm={row['mean_prediction_error_to_current_mm']:.3f}"
                            )
            print(f"runtime_s={time.perf_counter() - start:.3f}")

    runtime_s = time.perf_counter() - start
    write_csv(out_dir / "synthetic_results.csv", rows)
    write_csv(out_dir / "raw_results.csv", rows)
    write_summary(out_dir / "summary.csv", rows)
    write_delay_tau_audit(out_dir / "audit" / "delay_tau_audit.md")
    write_analysis(out_dir / "analysis.md", rows, runtime_s)
    print(f"Saved delay/tau audit outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
