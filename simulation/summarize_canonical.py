"""Validate archived trial coverage and print counts without modifying any files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize(run: Path) -> None:
    config = json.loads((run / "config.json").read_text())
    expected_trials = set(range(1, config["N_TRIALS"] + 1))
    expected_cells = {
        (condition, float(speed))
        for condition in config["CONDITIONS"]
        for speed in config["TEST_SPEEDS_CM_S"]
    }
    cells: dict[tuple[str, float], dict[int, int]] = defaultdict(dict)
    with (run / "raw_trials.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["condition"], float(row["speed_cm_s"]))
            trial = int(row["trial"])
            success = int(row["success"])
            if key not in expected_cells or trial not in expected_trials:
                raise ValueError(f"Unexpected cell or trial: {key}, {trial}")
            if success not in (0, 1) or trial in cells[key]:
                raise ValueError(f"Invalid success or duplicate trial: {key}, {trial}")
            cells[key][trial] = success
    if set(cells) != expected_cells:
        raise ValueError(f"Missing cells: {expected_cells - set(cells)}")
    print(f"\n{run.name}: {sum(map(len, cells.values()))} raw rows")
    print("condition | speed_cm_s | successes/trials | percent")
    for condition in config["CONDITIONS"]:
        for speed in config["TEST_SPEEDS_CM_S"]:
            trials = cells[(condition, float(speed))]
            if set(trials) != expected_trials:
                raise ValueError(f"Incomplete trials: {condition}, {speed}")
            successes = sum(trials.values())
            note = " [excluded historical condition]" if condition == "pd_feedforward" else ""
            print(f"{condition} | {speed:g} | {successes}/{len(trials)} | "
                  f"{100 * successes / len(trials):g}%{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path,
                        help="Run directories; default: both committed canonical archives")
    args = parser.parse_args()
    canonical = Path(__file__).resolve().parent / "results" / "canonical"
    runs = args.runs or [canonical / "ablation_20260524_233332",
                         canonical / "ablation_20260528_232432"]
    for run in runs:
        summarize(run)


if __name__ == "__main__":
    main()
