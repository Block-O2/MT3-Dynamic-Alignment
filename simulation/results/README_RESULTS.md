# Results Directory

This directory keeps the repository's reproducible simulation result of record separate from scratch experiment outputs.

## Canonical Result

`canonical/formal_baseline/` contains the main reproducible formal baseline result.

It preserves:

- `command.txt`
- `git_info.txt`
- `config.json`
- `raw_results.csv`
- `summary.csv`
- `analysis.md`

Use this folder as the source of record for the formal baseline conclusion.

## Consolidated Conclusion

`FINAL_EXPLORATION_SUMMARY.md` contains the consolidated method exploration conclusion.

It incorporates the earlier stage-level conclusions:

- formal baseline
- tau / latency validation
- contact gating
- close retiming
- phase servo
- feasibility-aware replay

The current recommended method is:

```text
dynamic_tau0
```

## Code Cleanup Plan

`FINAL_CODE_CLEANUP_PLAN.md` classifies current conditions and utilities:

- keep as primary: `dynamic_tau0`
- keep as baselines: `static_replay`, `dynamic_cv`, `dynamic_ct`
- keep as diagnostics: latency utilities and attempt/frame diagnostics
- mark as experimental/not recommended: contact gates, close retiming, phase servo, and feasibility-aware replay variants

The default/recommended runner conditions are `static_replay`, `dynamic_tau0`, `dynamic_cv`, and `dynamic_ct`. Failed exploratory methods require an explicit experimental flag and should not be presented as successful methods.

## Intermediate Runs

`experiments/` is scratch/intermediate output and should not be committed.

`archive/` stores old intermediate outputs and superseded stage summaries when they need to be preserved locally without cluttering the top-level result directory.

`scratch/` is reserved for temporary result work.

These folders are ignored by git.

## Scientific Scope

These results are PyBullet simulation-only results. They do not validate:

- real robot behavior
- RealSense perception
- real contact dynamics
- robust real-world manipulation
- a full dynamic MT3 extension

The fixed-constraint grasp attachment remains a simulation artifact and should be stated clearly in any report.
