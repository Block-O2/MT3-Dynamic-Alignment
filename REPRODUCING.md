# Reproduction and evidence review

Reviewed on **2026-11-04**, starting from `7bf8abc3221671303b4987a10f73b9650fb4ddea`.
That review preserved algorithm, experiment and test source files, raw trials,
configurations, figures and GIFs. Documentation and a read-only archive-summary
utility were its only behavioral additions. A **2026-11-05** follow-up removed
the reference paper PDF from the current tree; see the license review below.

## Environment and installation

The exercised environment is macOS arm64, Python **3.11.14**, NumPy **2.4.4**,
Matplotlib **3.10.9**, pytest **9.0.3**, PyBullet **3.2.5**, Pillow **12.2.0**.
The requirement files pin these tested package versions. They are not a complete
transitive dependency lock or a claim of support for every OS/Python version.

```bash
git clone https://github.com/Block-O2/MT3-Dynamic-Alignment.git
cd MT3-Dynamic-Alignment
conda create -n dynamic_mt3 python=3.11 pip -y
conda activate dynamic_mt3
python -m pip install -r requirements.txt
python -m pytest tests/ -v
python examples/simulate_and_plot.py
```

Use another environment name if `dynamic_mt3` already exists. The example uses
Matplotlib's noninteractive Agg backend and creates `results/tracking_comparison.png`,
`results/latency_compensation.png`, and `results/convergence.png`. It does not
write into `simulation/results/`. Set `MPLCONFIGDIR` to a writable temporary
directory if Matplotlib reports a cache-permission warning.

For the current grasping script and its imported `simulation/dev/03_closed_loop.py`:

```bash
python -m pip install -r requirements-simulation.txt
```

These scripts use NumPy, Matplotlib, PyBullet (including `pybullet_data`) and
Pillow. `imageio` and `seaborn` are not imported by these entry points and are
not required for them. Core tests require only NumPy and pytest. No ROS,
RealSense SDK, GPU training stack or original MT3 dataset is required.

PyBullet connects in DIRECT mode (`USE_GUI=False`). The code requests
`ER_BULLET_HARDWARE_OPENGL` when obtaining depth/RGB images. That path worked
in the tested local DIRECT environment; renderer behavior on another system
must be checked, and DIRECT mode is not a guarantee that graphics dependencies
are available everywhere. Some platforms may need a PyBullet source build.
No new graphics/system packages were installed in this review.

## Recompute archived scores without simulation

```bash
python simulation/summarize_canonical.py
# Or select a single archive:
python simulation/summarize_canonical.py simulation/results/canonical/ablation_20260528_232432
```

This standard-library-only utility checks the configuration's conditions,
speeds, binary success values and complete trial IDs before printing every
condition's numerator, denominator and percentage. It includes the excluded
historical condition explicitly and writes no files. It does not reclassify
success, rerun physics, or redraw any archived figure.

## Run the current PyBullet ablation

```bash
python simulation/10_grasping_experiment.py
```

The command first runs tracking diagnostics at all four speeds. If any mean
tracking error is at least 15 mm, `main()` returns before the ablation; check
that a run folder and complete CSV were actually produced rather than relying
only on the process exit code.

The current constants are seed 42, 20 trials per cell, 2/4/6/8 cm/s,
`TAU=0.1`, dropout probability 0.15, and five conditions (400 trials total).
There are no command-line flags to change these constants. A full run may take
hours. It writes a new `simulation/results/canonical/ablation_<timestamp>/`
folder containing command, configuration, raw CSV, success chart and demo GIF.
The GIF is recorded at 4 cm/s, trial index 0 for `dynamic_tau0`, with a 3x
playback label. Keep all new results separate from the two historical folders.

### Historical source provenance

The archived `command.txt` files contain only the script path; invoke them with
`python`, from the repository root. They do not specify interpreter/package
versions. The `git_commit_hash` field records the base HEAD, not a full dirty
working-tree snapshot:

- May 24 records `17c8aa47ade863d65b08e407c56b9346f50f2969`; the five-condition
  experiment code was committed later in `879db2b`.
- May 28 records `80148c3253726424ea84a4d2c161ba673dc5bd40`, whose script lacks
  dropout. The dropout changes were committed in `db4cdfd`; the historical
  `pd_feedforward` implementation is absent from the committed current script.
- May 28's config lists six conditions and its CSV contains 480 trials. The
  archived chart reports five; the earlier analysis says the sixth condition
  had an implementation bug. Those 80 rows are retained and clearly excluded.

Thus a checkout of the recorded SHA plus its command is **not enough to
reconstruct the archived run exactly**. The current command is runnable as a
new experiment but does not reproduce the historical six-condition random
number sequence. The review verifies archived counts independently; it does
not repair or rewrite old metadata or promise identical rerun percentages.

## Validation performed

| Check | Observed result | Scope |
|---|---|---|
| Existing unit suite: `python -m pytest tests/ -v` | 69 passed | Existing Python 3.11 environment; no hardware |
| Synthetic example: `python examples/simulate_and_plot.py` | Completed; all three PNGs generated | Synthetic data only |
| Archive-summary utility | 400 + 480 rows, all condition-speed cells complete and unique | Both original CSVs; no writes |
| Pinned dependency resolution | Passed with `pip install --no-index -r requirements-simulation.txt` in a temporary venv using system site packages | Reuses installed packages; **not** a clean network installation |
| Tracking diagnostics | Means 1.885, 2.942, 3.652, 12.017 mm at 2/4/6/8 cm/s | All below the script's 15 mm abort threshold |
| Five individual condition trials at 4 cm/s | All completed without exceptions; dynamic and oracle succeeded, others failed | Execution smoke check, not success-rate estimates |
| Recorded dynamic trial at 4 cm/s | Completed successfully and generated a GIF | Actual local DIRECT rendering and Pillow encoding |
| Reduced full `main()` pipeline | 20 trials completed; all five output files generated | One trial per condition/speed, temporary output root; not a full benchmark |

A reduced full-pipeline check also **passed**: the existing `main()` path ran
all five conditions at all four speeds, one trial per cell (20 trials total).
Only `N_TRIALS` and `PROJECT_ROOT` were changed **in memory**; source and
constants on disk remained unchanged. The tracking diagnostics completed,
and nonempty command, config, 20-row raw CSV, success chart and demo GIF were
verified in a temporary directory. Those temporary outputs were removed;
the smoke-check proportions are not reported as research results.

The default **400-trial experiment is not rerun in full** in this documentation
review. The 480-trial historical run is not reconstructed. A clean environment
installation from package indexes, Windows/Linux portability, real latency,
real robot integration, and real contact grasping are not validated.

## Paper and license review

### Follow-up decision: 2026-11-05

The former root paper `2511.10110v1.pdf` was **28,837,171 bytes**, 43 pages:
*Learning a Thousand Tasks in a Day*, by Kamil Dreczkowski, Pietro Vitiello,
Vitalis Vosylius and Edward Johns. The inspected copy's SHA-256 was:

```text
4666980d84163a011b91eb32ac7e7df14fb3cfec65d08291c706d005e34aed82
```

The review checked this specific copy, the [arXiv v1
record](https://arxiv.org/abs/2511.10110v1), the [authors' project
page](https://www.robot-learning.uk/learning-1000-tasks), and the linked
[Science Robotics article](https://www.science.org/doi/10.1126/scirobotics.adv7594):

- The PDF's first-page footer identifies it as the author's version, posted
  by permission of AAAS for personal use, with redistribution restricted.
  The arXiv v1 Comments repeat that restriction.
- The same PDF's internal `/License` field and arXiv's license link point to
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  This does not resolve the conflicting, copy-specific restriction.
- The authors' page links the paper and publisher, but no explicit permission
  resolving this copy's redistribution conflict was found there. The publisher
  page returned HTTP 403 in this check, so its full copyright/rights-holder
  statement could not be verified. The PDF identifies the authors and AAAS's
  posting permission; this review does not infer copyright ownership from that
  permission notice.

**Clear public redistribution permission for this specific copy could not be
confirmed. The PDF has therefore been removed from the current repository tree,
with the official citation and links retained below.** No entry point or test
reads it, and a search of all tracked text files found no relative PDF link
requiring repair. Installation, reproduction, code and experiment evidence are
unaffected.

This is an ordinary deletion commit, **not a Git history rewrite**. Earlier
commits and historical archives may still contain the old PDF; their copyright
risk has not been cleared by this change. No paper content, other data, license
file or historical archive was modified. No repository-wide license is inferred
from the paper's license metadata.

> Kamil Dreczkowski, Pietro Vitiello, Vitalis Vosylius, and Edward Johns.
> Learning a Thousand Tasks in a Day. Science Robotics 10, eadv7594 (2025).
> DOI: [10.1126/scirobotics.adv7594](https://doi.org/10.1126/scirobotics.adv7594).
> [arXiv:2511.10110v1](https://arxiv.org/abs/2511.10110v1) ·
> [Authors' project](https://www.robot-learning.uk/learning-1000-tasks).
