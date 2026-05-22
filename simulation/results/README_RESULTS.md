# Results Directory

This directory keeps reproducible simulation outputs separate from intermediate development runs.

## Canonical Results

`canonical/formal_baseline/` contains the current main formal baseline result. It preserves the command, git metadata, config, raw trial CSV, summary CSV, and analysis for the 160-trial Stage 3 formal baseline.

Use this folder as the source of record for the current baseline conclusion.

## Consolidated Summary

`STAGE3_FORMAL_BASELINE_SUMMARY.md` contains the consolidated Stage 2 through formal-baseline summary. It explains which earlier smoke, audit, pilot, metric-diagnostic, and schema runs were superseded, which metrics are primary, which metrics are diagnostics-only, and what the formal baseline supports.

`STAGE4A_LATENCY_VALIDATION_SUMMARY.md` contains the consolidated tau/latency validation summary. It records that `predict_ahead(tau)` is valid under synthetic constant-velocity assumptions, but was not validated in the current PyBullet observation-delay or control-delay tracking-only diagnostics. The current project direction is to stop tau tuning and proceed next to contact-aware temporal gating after approval.

`STAGE5_METHOD_EXPLORATION_SUMMARY.md` contains the consolidated post-baseline method exploration summary. It records that freeze-style contact gating, pre-close gating, and simple close-phase retiming were not promising in smoke tests. The current strongest reliable method remains `dynamic_tau0`.

`STAGE5_CODE_CLEANUP_PLAN.md` lists the experimental method additions and recommends keeping diagnostics while marking or removing failed method variants before presenting the repository as a clean baseline study.

Key files:

- `canonical/formal_baseline/`
- `STAGE3_FORMAL_BASELINE_SUMMARY.md`
- `STAGE4A_LATENCY_VALIDATION_SUMMARY.md`
- `STAGE5_METHOD_EXPLORATION_SUMMARY.md`
- `STAGE5_CODE_CLEANUP_PLAN.md`

## Intermediate Runs

`experiments/` is treated as scratch/intermediate output. Future routine runs should be written there during development, but they should not be committed unless explicitly promoted to `canonical/`.

`archive/` and `scratch/` are also reserved for non-canonical results and are ignored by git.

## Scientific Scope

These results are simulation-only PyBullet results. They do not validate real robot behavior, RealSense perception, real contact dynamics, or a full dynamic MT3 extension.
