# Swan experiments

## Sequential workbook campaign

`bash slurm/run_date_plan.sh` reads `SAT_manuscript/Starfm dates.xlsx` without
editing it. No extra Excel library is needed. Sheets must be named
`path N - row N`, with A:C headers `Pair 1`, `Prediction date`, `Pair 2`.
It preserves sheet/row/pair order. The existing pipeline still internally
normalizes explicit pairs as before; this runner does not change that code.
No satellite-availability preflight or extra date selection is performed.

Preview (no downloads, computation, or job submission):

```bash
cd /path/to/starfm-workflow
bash slurm/run_date_plan.sh --campaign multiregion_v1 --remove-unnecessary
```

Submit the same plan:

```bash
bash slurm/run_date_plan.sh --campaign multiregion_v1 --remove-unnecessary --submit
```

Copy the updated workbook to the stated relative path on Swan first. Python 3
must be available for the submitter; use `module load anaconda` if needed.
Each compute job activates `thesis_paper_env` and uses the existing pipeline
wrapper. This does not build STARFM; use the already-built Linux executable.

For each region and date in workbook order, and each band B1/B2/B3/B4/B5/B7:

1. VIIRS coregistration: moving-window, AROSICS (B5 reference), none.
2. Sensor comparison: MODIS and VIIRS, excluding B1 from this comparison.
3. VIIRS search distances: 550, 650, 750, 850, 950, 1150 m (uncertainty 50).
4. VIIRS uncertainties: 35, 50, 65 (distance 750 m).

Each configuration is its own job with a fresh wall-time allocation. There is
no pair-situation test. Sensor and parameter tests use moving-window. Defaults
are cloud limit 5%, requested registration window 1024, QA/mask options off.
The runner does not change effective-window behavior of the existing pipeline.
Shared NBAR is reused. `--remove-unnecessary` discards per-run intermediates
after successful validation; downloads and reused NBAR remain. Omit it to keep
intermediates. `--cloud-percent` and `--starfm-executable` can be overridden.

For the current 55 cases: 4510 jobs (990 coregistration, 550 sensor,
1980 distance, 990 uncertainty). Repeated baseline configurations are retained
as separate test records, not independent scientific replicates.

Jobs are chained with `afterany`, not `afterok`: only one campaign job can run
at a time, in exact plan order, and failures/timeouts do not block later jobs.
This does NOT serialize other manually submitted jobs or other campaigns.
Do not run another pipeline against the same data root concurrently.
Canceling one job allows its successors to proceed; cancel the whole chain if
you intend to stop the campaign. Resources/time are set in `slurm/date_plan.sbatch`.

Slurm submission-count limits may prevent all 4510 jobs being queued at once.
Already accepted jobs remain in the chain. Repeat the identical submit command
after capacity is available; `submissions.json` prevents resubmitting accepted jobs.
Do not change a campaign's workbook/settings after planning; use another campaign
name for a changed plan. Do not run different campaigns concurrently with the
same data root. Submission interruption at an uncertain point is flagged rather
than silently submitting a duplicate. Keep the terminal open until submission
finishes; once accepted, the jobs do not require an open terminal.

Records live under `COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/multiregion_v1/`:
`plan.json`, `submissions.json`, `logs/`, and per-task `status/`.
Each pipeline output has its unique test ID under `OUTPUTS/TESTS/`.
Inspect scheduler state for tasks marked `running` without a summary: a hard
timeout/kill prevents Python from writing the final status. They are NOT counted
as completed. Failed jobs are not automatically retried or deleted.

Collect all available metrics (safe to repeat as jobs finish):

```bash
bash slurm/run_date_plan.sh --collect \
  COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/multiregion_v1/plan.json
```

This creates `results.csv` and `results.json` beside the plan, including test
family, region, target/pair dates, band, settings, status and all available metrics.
Skipped/missing results remain visible. The submission ledger includes every job
ID. Resubmitting the campaign only submits jobs not yet accepted by Slurm; it
does not retry failed accepted jobs. Review those logs before scheduling retries.

Run from the cloned repository on Swan. Each wrapper loads anaconda and activates
thesis_paper_env. Credentials come from the ignored config.yaml in the project root.
The defaults are path 29/row 32, target 2023-09-18, VIIRS, B1,
pair dates 2023-09-02 and 2023-09-26, cloud <=5%, moving-window,
registration using the prediction band, window 1024, STARFM distance 750 m and uncertainty 50.

The sensor comparison uses B3 by default because MODIS cannot support Landsat B1.
AROSICS defaults to B5. Moving-window always registers each prediction band using
that same band, including all-band runs. COREG_BAND overrides AROSICS only.
The comparison keeps window 1024 fixed; with output bands other than B5 it
compares method-and-matching-band configurations rather than methods alone.

## Setup and submission

```bash
cd /path/to/starfm-workflow
mkdir -p logs
make -B -C StarFM_original/source
sbatch --export=ALL,BAND=B1 slurm/test_parameters.sbatch
sbatch --export=ALL,BAND=B3 slurm/test_sensors.sbatch
sbatch --export=ALL,BAND=B1 slurm/test_coregistration.sbatch

# Run only the zero-shift/grid-alignment baseline (use BAND=all for all bands).
sbatch --export=ALL,BAND=all,REMOVE_UNNECESSARY=1,COREGISTRATION_METHODS=none \
  slurm/test_coregistration.sbatch
sbatch --export=ALL,BAND=B1 slurm/test_pair_dates.sbatch
```

For one configurable pipeline run rather than a multi-run experiment, submit
`run_nearest_pair_pipeline.sbatch`. Its environment variables map directly to
the pipeline arguments. Pair dates use `PAIR_DATE_1` and optional
`PAIR_DATE_2`, because comma-separated `sbatch --export` values cannot safely
contain a space-separated list.

```bash
sbatch \
  --export=ALL,PREDICTION_DATE=2023-09-18,WRS_PATH=29,WRS_ROW=32,BAND=B1,COARSE_SENSOR=VIIRS,PAIR_DATE_1=2023-09-02,PAIR_DATE_2=2023-09-26,CLOUD_PERCENT=5,VALIDATE_HELD_OUT=1,COREGISTRATION_METHOD=moving-window,WINDOW_SIZE=1024,SEARCH_DISTANCE=950,UNCERTAINTY=50,STARFM_EXECUTABLE=StarFM_original/source/StarFM.exe,TEST_ID=best_viirs_p029_r032_20230918_b1_d0950_u0050 \
  slurm/run_nearest_pair_pipeline.sbatch
```

Additional controls are `MAXIMUM_DAYS`, `COREGISTRATION_BAND`,
`MINIMUM_WINDOW_VALID_FRACTION`, `USE_QUALITY_MASK`,
`STARFM_USE_MASK_FILES`, `REUSE_EXISTING_NBAR`, `OUTPUT_DIR`, `PROJECT_DIR`,
and `INTERMEDIATE_OUTPUTS=default|keep|discard`. Use `BAND=all` for all bands,
or set `PAIR_DATE_2=` for a one-pair run.

These are individual submission examples, not a recommendation to start them
concurrently. Shared downloads and NBAR creation are not concurrency safe.
Wait for a job to finish or use --dependency=afterok:JOBID for the next submission.
Do not run other pipelines using the same OUTPUT_DIR at the same time.

The LOGS directory must exist BEFORE sbatch. Slurm cannot create missing parent
directories for its logs. Logs are thesis_JOBID.out and thesis_JOBID.err.
The scripts request 24 hours, 2 CPUs and 32000 MB. Check your partition/account's
allowed wall time; override using sbatch --time=... --cpus-per-task=... --mem=...
24 hours is a request, not a runtime estimate. The parameter job contains five
distance runs plus five uncertainty runs per band.
Use an account flag if your allocation requires one.

## Controls

Preview the exact commands locally without downloading or running STARFM:

```bash
DRY_RUN=1 BAND=B1 bash slurm/run_experiment.sh parameters
DRY_RUN=1 BAND=B3 bash slurm/run_experiment.sh sensors
```

Use sbatch --export=ALL,NAME=value,NAME=value (no spaces).

* BAND=B1: one band; BAND=all: all supported bands (VIIRS six, MODIS five,
  sensor comparison five common bands).
* REMOVE_UNNECESSARY=0 (default): retain intermediates for diagnosis.
  Set to 1 to discard intermediates after successful validation. Existing NBAR
  mosaics remain cached because the runners request reuse; downloaded inputs remain.
* PREDICTION_DATE, WRS_PATH, WRS_ROW, PAIR_BEFORE, PAIR_AFTER:
  override target/scene/bracketing dates.
* COARSE_SENSOR=VIIRS or MODIS: applies except to the sensor comparison, which
  runs both. Choose a MODIS-supported BAND if selecting MODIS.
* CLOUD_PERCENT=5, COREG_BAND (optional AROSICS override), WINDOW_SIZE=1024,
  SEARCH_DISTANCE=750, UNCERTAINTY=50: shared settings. The parameter experiment
  tests distance at 550, 650, 750, 850, and 950 m while uncertainty is 50;
  it then tests uncertainty at 20, 35, 50, 65, and 80 while distance is 750 m.
* STARFM_EXECUTABLE=StarFM_original/source/StarFM.exe: Linux build.
* PROJECT_DIR and OUTPUT_DIR: project and imagery/output roots.
* EXPERIMENT_ID: optional short unique label. The default includes the Slurm job
  ID, so submitting again creates new result folders without deleting old results.

The four pair scenarios are:

| Scenario | Pair dates |
| --- | --- |
| one | 2023-09-26 (nearer of the original bracketing pair) |
| bracket | 2023-09-02, 2023-09-26 |
| before | 2023-08-25, 2023-09-02 |
| after | 2023-09-26, 2023-10-04 |

The additional dates are candidates, not verified clear scenes. Each must pass
the same 5% threshold. Override SECOND_BEFORE, SECOND_AFTER, or ONE_PAIR_DATE.
Changing the target also requires setting appropriate pair dates.
Distances from the target differ between configurations, so interpret this as a
practical scenario comparison rather than an isolated causal test of pair count.
These scripts use the full scene; they do not yet implement the small agricultural
ROI or verify agricultural stability.

## Outputs and completion

Each run produces a separate OUTPUTS/TESTS/EXPERIMENT_ID_* folder.
All four experiments collect completed summaries into
OUTPUTS/EXPERIMENTS/EXPERIMENT_ID/results.csv and results.json.
The parameter and sensor runners also retain their SWEEPS, UNCERTAINTY_SWEEPS,
and SENSOR_COMPARISONS aggregate tables.
Metrics currently use each run's own validation mask; equal-pixel comparisons
require a subsequent common-mask analysis.

Runs stop on the first error. Already completed per-test summaries remain.
If interrupted, collect those partial results manually:

```bash
python slurm/collect_results.py --output-dir COREGISTRATION_FILES --experiment-id YOUR_ID
```

This collects completed runs only; it does not imply the whole experiment finished.
The pipeline does not resume STARFM mid-band or skip existing named tests.

The wrappers set OpenMP threads to the Slurm CPU allocation and OMP_STACKSIZE=64M.
The Python STARFM wrapper also writes only short cwd-relative names into
`starfm_parameters.txt` (for example `predicted_B1_20230918.bin`) and runs the
binary with `cwd` set to the run directory. That avoids overflowing the original
`writeHeader()` 200-byte `"<out>.hdr"` buffer on long Swan output paths; do not
pass absolute output paths into STARFM parameter files. Still test the Linux
executable successfully on one band before committing to the entire suite.
# Concurrent final predictions (four hours per task)

Preview on Swan (run with bash, not sbatch):

```bash
bash slurm/run_date_plan.sh \
  --workbook "SAT_manuscript/Starfm dates main.xlsx" \
  --mode final --campaign final_main_v1 \
  --max-concurrent 8 --remove-unnecessary
```

Repeat with `--submit` to submit the array. One task runs one band for one
workbook row. Each task has 2 CPUs, 32000 MB requested memory and a four-hour
wall-clock limit including download/lock wait time. `--max-concurrent 8`
allows at most eight running array tasks (not eight groups of six bands).
Slurm may schedule fewer. The limit applies to this array, not other campaigns.
Without `--max-concurrent`, the existing sequential 24-hour submission remains
unchanged. Array settings are in `slurm/date_plan_array.sbatch`; the submitter
also explicitly sets `--time=04:00:00`.

Logs: `COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/final_main_v1/logs/<index>_<array_job_id>.out`
and `.err`. Task status files store Slurm's task job IDs. Result folders remain
under `OUTPUTS/TESTS/<test_id>`. Collect after completion:

```bash
python slurm/date_plan.py --collect COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/final_main_v1/plan.json
```

To retry after ALL campaign jobs finish, repeat the original submission command
with BOTH `--submit --retry-failed`. This queries Slurm and refuses while any
earlier array is active/pending. It submits only tasks without a test summary.
Incomplete task output directories are moved to `.failed_<timestamp>` backups
under the same TESTS parent; these are retained for recovery and consume space.
Input and NBAR caches are never moved/deleted. A time-limit termination can
leave status marked running; retry selection uses the completion summary, not
that stale status. Retries restart an individual prediction, not a STARFM row.

Safety details:

- Every download has an atomic-directory lock; staged downloads keep
  their file extension, are signature/length checked and atomically published.
- Reused NBAR is generated under a per-key lock in a unique staging directory.
  The complete product directory is atomically published in
  `OUTPUTS/NBAR_CACHE_V2/<key>`. Inputs (path/size/mtime), tile set, sensor/band,
  date, Python code and GDAL/NumPy versions determine the key. Source files must
  be treated as immutable; in-place edits preserving size/mtime are unsupported.
- Legacy NBAR files have no completion/provenance marker and are not reused by
  the new cache; the first run may regenerate NBAR. Shared V2 cache is retained.
- Coregistration, STARFM and validation are task-private. Duplicate task workers
  are locked. Cleanup does not include shared download or V2 NBAR cache files.
- An interrupted submission leaves a pending marker; inspect squeue and
  reconcile that marker rather than blindly resubmitting and duplicating jobs.
- Old pipeline versions do NOT honor these locks. Do not run older jobs against
  the same data tree, edit source/config files, or manually clean caches while
  the campaign is active. Locks use atomic mkdir, because Swan's tested mount
  did not enforce advisory flock across nodes. A lock waits up to one hour,
  then fails safely instead of allowing concurrent writers.
- Slurm's MaxArraySize/job quotas and external service limits still apply. A
  rejected array reports Slurm's error and does not automatically split jobs.
- Hard-killed jobs can leave hidden `.download-*`/`.nbar-*` staging directories.
  They are never consumed as completed data. Clean those only when no jobs use
  the cache. A hard kill can also leave a `.lock.d` directory. Its `owner.json`
  records the host, process, UID, Slurm job ID and array identity. Waiting jobs
  check at most once per minute per lock and automatically recover it only
  after queue absence and an unambiguous terminal accounting record (including
  terminal job steps) confirm completion. The lock creation time must fit the
  recorded job lifetime; a 30-second post-completion grace period applies.
  NODE_FAIL/BOOT_FAIL, missing/ambiguous accounting, scheduler errors, legacy
  owners without UID, and non-Slurm owners are NOT automatically recovered.
  Recovery itself is serialized by an atomic `.recovery` directory; if a
  recovery process is forcibly killed, its guard can require manual inspection.
  No lock is stolen simply because it is old. Ordinary exceptions release locks.
  Old plain `.lock` files are unused by the directory protocol.
- Array tasks use `--no-requeue`: a node failure fails the task instead of
  requeueing it under the same job ID (which would block on its own stale task
  lock). Pick such tasks up with `--retry-failed`; their leftover locks still
  need manual inspection.

# Final selected-configuration campaign

Use `bash slurm/run_date_plan.sh --mode final --campaign final_950_u35 --remove-unnecessary`
to preview, then repeat with `--submit` to submit. The default workbook remains
`SAT_manuscript/Starfm dates.xlsx`; override it with `--workbook PATH`.
This mode schedules six jobs per workbook row (B1, B2, B3, B4, B5, B7),
sequentially via afterany dependencies. It preserves both supplied pair dates
and uses held-out validation, cloud limit 5%, moving-window registration on the
prediction band, search distance 950 m, and uncertainty 35. The existing
window and quality-mask settings remain unchanged. MODIS is used for B2/B3/B4/B5/B7
in 029/032 and 043/034; VIIRS is used for B1 everywhere and all bands in 143/042.
Unknown regions are rejected rather than assigned an untested sensor policy.

This is a selected configuration, not an independently proven joint optimum.
There are no availability prechecks or automatic date/threshold substitutions.
Previously missing inputs or ineligible held-out scenes can still fail.
Use a new campaign name to avoid conflicting with existing experiment plans.
Results and logs use the existing `OUTPUTS/TESTS/<test_id>` and
`OUTPUTS/DATE_PLANS/<campaign>/logs` layout. Cleanup retains validation products.
Collect with `python slurm/date_plan.py --collect COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/final_950_u35/plan.json`.

