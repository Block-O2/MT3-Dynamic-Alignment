# Canonical experiment archives

Each timestamped folder is an immutable experimental record. Do not regenerate
plots or edit CSV/configuration files in these archived folders to update the
portfolio. New experiments belong in a new timestamped folder.

| Run | Setup | Contents | Dynamic replay at 2/4/6/8 cm/s |
|---|---|---|---|
| [ablation_20260524_233332](ablation_20260524_233332) | No artificial dropout; seed 42; 20 trials/cell | Command, config, 400 raw trials; no plots stored in this folder | 10/20, 15/20, 17/20, 4/20 |
| [ablation_20260528_232432](ablation_20260528_232432) | 15% random point-cloud dropout; seed 42; 20 trials/cell | Command, config, 480 raw trials, chart and GIF | 14/20, 16/20, 17/20, 5/20 |

**Featured source:** May 28 is used by both README languages and the personal
website. Its chart reports five conditions; the raw CSV and config also retain
80 `pd_feedforward` rows excluded by the earlier analysis due to a reported
implementation bug. The current script has no implementation of that condition.
Do not silently treat the archive as 400 raw trials.

Both runs use `TAU=0.1 s`, including the historically named `dynamic_tau0`.
The old [`plot/grasping_success_rate.png`](../plot/grasping_success_rate.png)
matches May 24 percentages and is retained as an older result. The separate
[`raw/10_grasping_final_seed42.csv`](../raw/10_grasping_final_seed42.csv) is a
10-trial development run, not either 20-trial ablation.

From the repository root, validate trial IDs and print all scores read-only:

```bash
python simulation/summarize_canonical.py
```

The recorded command files omit the Python interpreter. The recorded Git SHA
is the base HEAD, not a complete snapshot of the source used for the run: in
particular, May 28's recorded `80148c3` lacks the later committed dropout change
and the uncommitted historical sixth condition. Exact historical reruns cannot
be guaranteed from these metadata alone. See [reproduction details](../../../REPRODUCING.md)
and [analysis](../../../ANALYSIS.md). No original experiment files were changed
by the 2026-11-04 documentation review.
