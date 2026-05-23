# MT3 Dynamic Alignment

Simulation-only research prototype for object-frame replay of a static manipulation demonstration on slowly moving tabletop objects.

## Current Status

This repository currently supports a controlled PyBullet simulation study. It does not validate real robot behavior, RealSense perception, real contact dynamics, or robust hardware deployment.

The current strongest reliable method is:

```text
dynamic_tau0
```

The canonical formal baseline and consolidated exploration summary are stored under:

```text
simulation/results/
├── canonical/formal_baseline/
├── FINAL_EXPLORATION_SUMMARY.md
├── FINAL_CODE_CLEANUP_PLAN.md
└── README_RESULTS.md
```

The formal baseline uses a PyBullet fixed-constraint grasp attachment once the gripper closes near the object. This is a simulation artifact and should not be presented as real contact validation.

## Supported Claim

The honest claim supported by the current repository is limited to:

> Under controlled PyBullet tabletop simulations, dynamic object-frame replay can be compared against static replay and prediction baselines for slowly moving planar objects.

Do not interpret the current results as real robot validation, full dynamic manipulation, or a complete MT3 extension.

## Formal Baseline Summary

The formal baseline compared:

- `static_replay`
- `dynamic_tau0`
- `dynamic_cv`
- `dynamic_ct`

Speeds:

- 2.0, 4.0, 6.0, 8.0 cm/s

Trials:

- 10 trials per condition per speed

Key conclusions:

- `static_replay` failed at all tested speeds on moving objects.
- Dynamic object-frame replay clearly outperformed static replay.
- `dynamic_tau0` was the strongest reliable baseline.
- `dynamic_cv` / tau prediction did not outperform `dynamic_tau0` in the latency-free PyBullet setup.
- `dynamic_ct` did not meaningfully differ from `dynamic_cv`.
- Dominant failures were `attempt_limit` / no-contact.

See [`simulation/results/FINAL_EXPLORATION_SUMMARY.md`](simulation/results/FINAL_EXPLORATION_SUMMARY.md) for the full interpretation.

## Exploratory Methods

The following variants are retained only for transparency and diagnostics. They should not be presented as successful methods:

- tau / latency compensation in the current PyBullet grasp setup
- `dynamic_tau0_contact_gated`
- `dynamic_tau0_preclose_gated`
- `dynamic_tau0_close_retimed`
- `dynamic_phase_servo`
- `feasibility_aware_replay`
- `feasibility_aware_replay_v2`

Further method work should use a principled optimizer or MPC-style phase formulation on a new branch, rather than more threshold patches.

## Module Structure

```text
dynamic_alignment/             Core tracking and replay modules
simulation/                    PyBullet experiments and logged runners
simulation/run_experiment.py   Reproducible experiment runner
simulation/results/            Canonical baseline and consolidated summaries
tests/                         Hardware-free unit tests
```

## Quick Start

Create a CPU-only environment:

```bash
conda create -n dynamic_mt3 python=3.11 numpy matplotlib pytest -y
conda activate dynamic_mt3
```

Install PyBullet-related packages if running simulations:

```bash
conda install -c conda-forge pybullet pillow imageio -y
```

Run tests:

```bash
python -m pytest tests/ -q
```

Run the official/default smoke conditions:

```bash
python -m simulation.run_experiment \
  --conditions static_replay dynamic_tau0 dynamic_cv dynamic_ct \
  --speeds 2.0 \
  --trials 1 \
  --seed 42
```

Failed exploratory methods require an explicit flag:

```bash
python -m simulation.run_experiment \
  --include-experimental-methods \
  --conditions feasibility_aware_replay_v2 \
  --speeds 4.0 \
  --trials 1
```

## Integration Notes

The code contains interfaces inspired by an eventual real-system setup, including point-cloud pose estimation and latency prediction. Those interfaces are not validated here with a real robot or RealSense camera.

Any future hardware claim would require separate real-world experiments with explicitly logged perception, control, latency, contact, and failure metrics.

## Acknowledgments

This project explores object-relative replay ideas related to MT3-style demonstration transfer. The current repository should be read as a simulation baseline and negative-results exploration, not as a validated robot manipulation system.
