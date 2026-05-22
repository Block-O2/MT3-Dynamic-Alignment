# Stage 3 Formal Baseline Summary

## Purpose

This repository is currently a simulation-only robotics research prototype. The Stage 3 goal was to evaluate, under controlled PyBullet tabletop simulations, whether dynamic object-frame replay is a meaningful baseline improvement over static replay for slowly moving planar objects.

The claim supported here is limited to simulation: dynamic object-frame replay can be compared against static replay and no-prediction baselines under this PyBullet setup. These results do not validate real robot performance, real contact behavior, RealSense perception, or a full dynamic MT3 extension.

## Stages Run

- Stage 2 smoke / output structure: verified a minimal DIRECT PyBullet path and created reproducible experiment folders with command, git info, config, logs, raw CSV, summary CSV, and analysis.
- Stage 3 baseline smoke: exposed `static_replay`, `dynamic_tau0`, `dynamic_cv`, and `dynamic_ct` through the runner and verified small runs.
- Stage 3.5 baseline audit and pilot: audited baseline code paths and found that the original tracking metric was phase-mixed and could mislead.
- Stage 3.6 metric validation: added EE-to-target and contact-window diagnostics to separate object-estimation diagnostics from grasp-relevant metrics.
- Stage 3.7 metric schema cleanup: demoted pre-contact object-estimation and phase-mixed tracking metrics to diagnostics-only; promoted formal baseline primary metrics.
- Formal baseline: ran 4 conditions x 4 speeds x 10 trials = 160 trials.

Earlier smoke, audit, pilot, metric-diagnostic, and schema runs were useful during development, but they are superseded by the canonical formal baseline result and this consolidated summary.

## Canonical Result

The canonical formal baseline result is stored at:

```text
simulation/results/canonical/formal_baseline/
```

It contains:

- `command.txt`
- `git_info.txt`
- `config.json`
- `raw_results.csv`
- `summary.csv`
- `analysis.md`

## Primary Metrics

The formal baseline uses these primary metrics:

- `success_rate`
- `n_no_contact`
- `n_finite_contact_error`
- `main_failure_counts`
- `mean_lift_mm`
- `mean_progress_pct`
- `mean_contact_position_error_finite_only`
- `mean_ee_to_target_error_contact_window_finite_only`
- `mean_contact_window_ee_to_object_xy_finite_only`

Finite-only contact metrics must always be interpreted together with `n_finite_contact_error` and `n_no_contact`.

## Diagnostics-Only Metrics

These metrics are useful for debugging but should not be used to rank methods:

- pre-contact object-estimation error
- phase-mixed object-tracking metrics
- max object-estimation/tracking error

They can be retained in raw or diagnostic outputs, but method ranking should use the primary metrics above.

## Formal Baseline Key Results

Success rates:

| Method | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| `static_replay` | 0.000 | 0.000 | 0.000 | 0.000 |
| `dynamic_tau0` | 0.600 | 0.900 | 0.900 | 0.200 |
| `dynamic_cv` | 0.300 | 0.800 | 0.800 | 0.300 |
| `dynamic_ct` | 0.300 | 0.800 | 0.800 | 0.300 |

No-contact failures:

- `static_replay`: 40
- `dynamic_tau0`: 14
- `dynamic_cv`: 14
- `dynamic_ct`: 14

Dominant failure modes among failed trials:

- `attempt_limit`: 82
- `orientation`: 6
- `approach`: 2

## Main Interpretation

- Dynamic object-frame replay clearly outperforms static replay in this simulation.
- `tau=0.1` CV prediction does not outperform `tau=0` in this latency-free PyBullet setup.
- CT does not differ meaningfully from CV under this low-speed, short-horizon setup.
- The dominant failure mode is attempt-limit / no-contact.

## Limitations

- PyBullet only.
- Deterministic fixed-constraint grasp attachment is used.
- This is not real contact validation.
- This is not hardware validation.
- This is not evidence for general dynamic manipulation.
- This is not a full MT3 dynamic extension.

## Recommended Next Method Direction

The next method direction should target attempt-limit / no-contact failures. Contact-aware temporal gating is a more plausible next step than tuning `tau` or relying on CT as the main improvement path.

Do not present `tau` tuning as the scientific contribution. In this project, `tau` represents the intended measured real-system delay: perception + computation + actuation.
