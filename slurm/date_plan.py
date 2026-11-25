"""Sequential Slurm jobs from the three-column date workbook (standard library only).

No satellite availability queries, pair-date sorting, or workbook modifications.
Each job runs one pipeline configuration, with afterany chaining in workbook order.
"""
import argparse
from collections import Counter
import csv
from datetime import date, timedelta
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from shared_cache import file_lock
BANDS = ("B1", "B2", "B3", "B4", "B5", "B7")
DISTANCES = (550, 650, 750, 850, 950, 1150)
UNCERTAINTIES = (35, 50, 65)
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_dates(path):
    """Read sheet names and A:C values, supporting Excel dates and ISO date text."""
    cases = []
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        properties = workbook.find("m:workbookPr", NS)
        is_1904 = properties is not None and properties.get("date1904") in ("1", "true")
        epoch = date(1904, 1, 1) if is_1904 else date(1899, 12, 30)
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        relationships = {n.get("Id"): n.get("Target") for n in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        for sheet in workbook.find("m:sheets", NS):
            match = re.fullmatch(r"path\s+(\d+)\s*-\s*row\s+(\d+)", sheet.get("name"), re.I)
            if not match:
                raise ValueError(f"Expected sheet name 'path N - row N': {sheet.get('name')}")
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = relationships[rid]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            sheet_xml = ET.fromstring(archive.read(member))
            rows = []
            for row in sheet_xml.findall("m:sheetData/m:row", NS):
                values = {}
                for cell in row.findall("m:c", NS):
                    col = re.sub(r"\d", "", cell.get("r"))
                    if col not in ("A", "B", "C"):
                        continue
                    if cell.find("m:f", NS) is not None:
                        raise ValueError(f"Use explicit dates, not formulas: {sheet.get('name')}!{cell.get('r')}")
                    raw = cell.findtext("m:v", default="", namespaces=NS)
                    kind = cell.get("t", "n")
                    if kind == "s":
                        value = strings[int(raw)]
                    elif kind == "inlineStr":
                        value = "".join(cell.find("m:is", NS).itertext())
                    elif kind == "n" and raw:
                        number = float(raw)
                        if not number.is_integer():
                            raise ValueError("Date contains a time component")
                        value = (epoch + timedelta(days=int(number))).isoformat()
                    else:
                        value = raw
                    values[col] = value
                if any(values.values()):
                    rows.append((row.get("r"), [values.get(c, "") for c in "ABC"]))
            if not rows or rows[0][1] != ["Pair 1", "Prediction date", "Pair 2"]:
                raise ValueError(f"Unexpected headers in {sheet.get('name')}")
            for row_number, values in rows[1:]:
                a, prediction, b = [date.fromisoformat(v).isoformat() for v in values]
                if len({a, prediction, b}) != 3:
                    raise ValueError(f"Duplicate/target pair dates in {sheet.get('name')} row {row_number}")
                cases.append(dict(path=int(match[1]), row=int(match[2]), prediction_date=prediction,
                                  pair_dates=[a, b], source_sheet=sheet.get("name"), source_row=int(row_number)))
    if not cases:
        raise ValueError("Workbook contains no cases")
    return cases


def make_tasks(cases, campaign, mode="experiments"):
    if mode not in ("experiments", "final"):
        raise ValueError(f"Unknown campaign mode: {mode}")
    tasks = []
    # Preserve workbook region/row order. Each band finishes its tests before the next band.
    for case in cases:
        for band in BANDS:
            configurations = [("coregistration", "VIIRS", m, 750, 50) for m in ("moving-window", "arosics", "none")]
            if band != "B1":
                configurations += [("sensors", s, "moving-window", 750, 50) for s in ("MODIS", "VIIRS")]
            configurations += [("distance", "VIIRS", "moving-window", d, 50) for d in DISTANCES]
            configurations += [("uncertainty", "VIIRS", "moving-window", 750, u) for u in UNCERTAINTIES]
            if mode == "final":
                region = (case["path"], case["row"])
                if region not in ((29, 32), (43, 34), (143, 42)):
                    raise ValueError(f"No final-run sensor policy defined for WRS-2 {region}")
                sensor = "VIIRS" if band == "B1" or region == (143, 42) else "MODIS"
                configurations = [("final", sensor, "moving-window", 950, 35)]
            for family, sensor, method, distance, uncertainty in configurations:
                index = len(tasks)
                test_id = f"{campaign}_{index:05d}_p{case['path']:03d}r{case['row']:03d}_{case['prediction_date'].replace('-', '')}_{band.lower()}_{family}"
                tasks.append(dict(case, index=index, test_id=test_id, band=band, family=family,
                                  sensor=sensor, method=method, distance=distance, uncertainty=uncertainty))
    return tasks


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def task_environment(plan, task):
    return dict(os.environ, PROJECT_DIR=str(PROJECT), OUTPUT_DIR=plan["output_dir"],
                PREDICTION_DATE=task["prediction_date"], WRS_PATH=str(task["path"]), WRS_ROW=str(task["row"]),
                BAND=task["band"], COARSE_SENSOR=task["sensor"], PAIR_DATE_1=task["pair_dates"][0],
                PAIR_DATE_2=task["pair_dates"][1], CLOUD_PERCENT=str(plan["cloud_percent"]),
                COREGISTRATION_METHOD=task["method"], COREGISTRATION_BAND="B5" if task["method"] == "arosics" else task["band"],
                WINDOW_SIZE="1024", SEARCH_DISTANCE=str(task["distance"]), UNCERTAINTY=str(task["uncertainty"]),
                TEST_ID=task["test_id"], VALIDATE_HELD_OUT="1", USE_QUALITY_MASK="0", STARFM_USE_MASK_FILES="0",
                REUSE_EXISTING_NBAR="1", INTERMEDIATE_OUTPUTS="discard" if plan["remove_unnecessary"] else "keep",
                STARFM_EXECUTABLE=plan["starfm_executable"], DRY_RUN="0", MINIMUM_WINDOW_VALID_FRACTION="0.30")


def execute(plan_path, index):
    # Also protects against accidental duplicate submissions of a task.
    lock = plan_path.parent / 'task_locks' / f'{index:05d}.lock'
    lock.parent.mkdir(exist_ok=True)
    with file_lock(lock):
        return execute_locked(plan_path, index)


def execute_locked(plan_path, index):
    plan = json.loads(plan_path.read_text())
    task = plan["tasks"][index]
    summary = Path(plan["output_dir"]) / "OUTPUTS" / "TESTS" / task["test_id"] / "test_summary.json"
    if os.environ.get('RETRY_INCOMPLETE') == '1' and summary.parent.exists() and not summary.exists():
        # Recoverable move: preserve failed intermediates; never touch cache.
        archive = summary.parent.with_name(summary.parent.name + f'.failed_{time.time_ns()}')
        summary.parent.rename(archive)
    status_path = plan_path.parent / "status" / f"{index:05d}.json"
    status_path.parent.mkdir(exist_ok=True)
    status = dict(task, job_id=os.environ.get("SLURM_JOB_ID"), status="running")
    save_json(status_path, status)
    if summary.is_file():
        code = 0
    else:
        code = subprocess.run(["bash", str(PROJECT / "slurm/run_nearest_pair_pipeline.sbatch")],
                              env=task_environment(plan, task), cwd=PROJECT).returncode
    status.update(status="completed" if code == 0 and summary.is_file() else "failed", exit_code=code)
    save_json(status_path, status)
    return 0 if status["status"] == "completed" else 1


def collect_rows(plan_path):
    plan = json.loads(plan_path.read_text())
    rows = []
    for task in plan["tasks"]:
        row = dict(task, campaign=plan.get("campaign"), cloud_percent=plan["cloud_percent"])
        row["pair_dates"] = ";".join(row["pair_dates"])
        record = plan_path.parent / "status" / f"{task['index']:05d}.json"
        state = json.loads(record.read_text()) if record.exists() else {}
        row.update(status=state.get("status", "not_started"), job_id=state.get("job_id"))
        summary = Path(plan["output_dir"]) / "OUTPUTS" / "TESTS" / task["test_id"] / "test_summary.json"
        if summary.is_file():
            row.update(json.loads(summary.read_text()), status="completed")
        rows.append(row)
    return rows


def write_results(directory, rows, name="results"):
    save_json(directory / f"{name}.json", rows)
    with (directory / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def result_key(row):
    """Identify a prediction independently of campaign, workbook row and task index."""
    return (row["path"], row["row"], row["prediction_date"], tuple(sorted(row["pair_dates"].split(";"))),
            row["band"], row["family"], row["sensor"], row["method"], row["distance"], row["uncertainty"])


def collect(plan_path, overrides=()):
    rows = collect_rows(plan_path)
    write_results(plan_path.parent, rows)
    print(dict(Counter(r["status"] for r in rows)))
    if not overrides:
        return
    # Completed override results fill matching incomplete base predictions.
    # The workbook row and index stay those of the base campaign.
    position = {result_key(r): i for i, r in enumerate(rows)}
    filled = 0
    for override in overrides:
        for row in collect_rows(override):
            i = position.get(result_key(row))
            if i is None:
                raise ValueError(f"{override}: {row['test_id']} matches no prediction in {plan_path}")
            if row["status"] != "completed":
                continue
            base = rows[i]
            if base["status"] == "completed" and base["campaign"] != row["campaign"]:
                raise ValueError(f"{base['test_id']} and {row['test_id']} both completed; refusing to choose one")
            rows[i] = dict(row, index=base["index"], source_sheet=base["source_sheet"], source_row=base["source_row"])
            filled += 1
    write_results(plan_path.parent, rows, "results_merged")
    print(f"Merged {filled} override results into results_merged.csv:", dict(Counter(r["status"] for r in rows)))


def submit(plan_path):
    plan = json.loads(plan_path.read_text())
    ledger_path = plan_path.parent / "submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    pending_path = plan_path.parent / "submission_pending.json"
    if pending_path.exists():
        pending = json.loads(pending_path.read_text())
        if not any(entry["first"] == pending["index"] for entry in ledger):
            raise RuntimeError("A previous submission was interrupted before its job ID was saved. Inspect squeue and reconcile submission_pending.json before continuing; no duplicate was submitted.")
        pending_path.unlink()
    start = ledger[-1]["last"] + 1 if ledger else 0
    previous = ledger[-1]["job_id"] if ledger else None
    for first in range(start, len(plan["tasks"])):
        last = first
        command = ["sbatch", "--parsable", "--export=ALL",
                   f"--output={plan_path.parent}/logs/{first:05d}_%j.out", f"--error={plan_path.parent}/logs/{first:05d}_%j.err"]
        if previous:
            command.append(f"--dependency=afterany:{previous}")
        command.append(str(PROJECT / "slurm/date_plan.sbatch"))
        save_json(pending_path, dict(index=first, test_id=plan["tasks"][first]["test_id"]))
        try:
            result = subprocess.run(command, env=dict(os.environ, PROJECT_DIR=str(PROJECT), PLAN_FILE=str(plan_path), PLAN_INDEX=str(first)),
                                    text=True, capture_output=True, check=True)
        except subprocess.CalledProcessError as error:
            pending_path.unlink()
            raise RuntimeError(f"Slurm refused task {first}: {error.stderr.strip()}. Already submitted jobs remain queued. Repeat the same command later to submit remaining tasks.") from error
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unrecognized submission response: {result.stdout!r}; inspect squeue before retrying")
        ledger.append(dict(first=first, last=last, job_id=job_id))
        save_json(ledger_path, ledger)
        pending_path.unlink()
        print(f"Submitted {job_id}: task {first}, dependent on {previous or 'none'}", flush=True)
        previous = job_id


def submit_array(plan_path, concurrency, retry=False, only=None):
    plan = json.loads(plan_path.read_text())
    if only is not None and (unknown := sorted(set(only) - {t['index'] for t in plan['tasks']})):
        raise ValueError(f'Unknown task indices: {unknown}')
    ledger_path = plan_path.parent / 'array_submissions.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    pending = plan_path.parent / 'array_submission_pending.json'
    if pending.exists():
        raise RuntimeError('Unconfirmed array submission: reconcile array_submission_pending.json with squeue before retrying.')
    if (plan_path.parent / 'submissions.json').exists():
        raise RuntimeError('This campaign already used sequential submission. Use a new campaign name for arrays.')
    if ledger and not retry:
        print('Array already submitted. Use --retry-failed after it finishes.')
        return
    if retry and not ledger:
        raise RuntimeError('No previous array to retry')
    for entry in ledger:
        active = subprocess.run(['squeue', '-h', '-j', entry['job_id'], '-o', '%i'],
                                text=True, capture_output=True)
        # Slurm forgets finished jobs after MinJobAge; squeue then rejects the ID.
        if active.returncode and 'Invalid job id' not in active.stderr:
            raise RuntimeError(f'squeue failed for job {entry["job_id"]}: {active.stderr.strip()}')
        if active.returncode == 0 and active.stdout.strip():
            raise RuntimeError('Campaign still has active or pending jobs; wait before retrying.')
    tests = Path(plan['output_dir']) / 'OUTPUTS/TESTS'
    indices = [t['index'] for t in plan['tasks']
               if not (tests / t['test_id'] / 'test_summary.json').is_file()]
    if only is not None:
        indices = [i for i in indices if i in set(only)]
    if not indices:
        print('All tasks completed; nothing to submit.')
        return
    # Single configuration per array task. Use a compact expression for all tasks.
    selection = f'0-{len(indices)-1}' if indices == list(range(len(indices))) else ','.join(map(str, indices))
    command = ['sbatch', '--parsable', '--export=ALL', '--time=04:00:00',
               f'--array={selection}%{concurrency}',
               f'--output={plan_path.parent}/logs/%a_%A.out',
               f'--error={plan_path.parent}/logs/%a_%A.err',
               str(PROJECT / 'slurm/date_plan_array.sbatch')]
    save_json(pending, dict(indices=indices, retry=retry))
    try:
        result = subprocess.run(command, env=dict(os.environ, PROJECT_DIR=str(PROJECT),
                                PLAN_FILE=str(plan_path), RETRY_INCOMPLETE='1' if retry else '0'),
                                text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        pending.unlink()
        raise RuntimeError(f'Slurm rejected array: {exc.stderr}') from exc
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError('Unknown submission response; reconcile pending marker before resubmitting.')
    ledger.append(dict(job_id=job_id, indices=indices, concurrency=concurrency, time_limit='04:00:00'))
    save_json(ledger_path, ledger)
    pending.unlink()
    print(f'Submitted array {job_id}: {len(indices)} tasks, at most {concurrency} concurrent, 4 hours each.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=PROJECT / "SAT_manuscript/Starfm dates.xlsx")
    parser.add_argument("--campaign", default="multiregion_v1")
    parser.add_argument("--mode", choices=("experiments", "final"), default="experiments",
                        help="final: one 950 m / 35 moving-window run per band and workbook row, with region-specific sensor selection")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "COREGISTRATION_FILES")
    parser.add_argument("--cloud-percent", type=float, default=5)
    parser.add_argument("--remove-unnecessary", action="store_true")
    parser.add_argument("--starfm-executable", default="StarFM_original/source/StarFM.exe")
    parser.add_argument("--submit", action="store_true", help="Otherwise only prepare the plan")
    parser.add_argument('--max-concurrent', type=int, help='Use a band-level Slurm array with this concurrency limit and 4 hours per task')
    parser.add_argument('--retry-failed', action='store_true', help='Resubmit incomplete array tasks after the campaign finishes; preserve failed directories')
    parser.add_argument('--only-tasks', type=lambda v: [int(x) for x in v.split(',')],
                        help='With --retry-failed: resubmit only these comma-separated task indices')
    parser.add_argument("--collect", type=Path, help="Collect metrics from an existing plan; no submission")
    parser.add_argument("--merge", type=Path, action="append", default=[],
                        help="With --collect: plan whose completed results fill matching incomplete predictions (repeatable)")
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        return execute(args.execute, args.index)
    if args.collect:
        collect(args.collect, args.merge)
        return 0
    if args.max_concurrent is not None and args.max_concurrent < 1:
        parser.error('--max-concurrent must be positive')
    if args.retry_failed and (not args.submit or args.max_concurrent is None):
        parser.error('--retry-failed requires --submit and --max-concurrent')
    if args.only_tasks is not None and not args.retry_failed:
        parser.error('--only-tasks requires --retry-failed')
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,19}", args.campaign):
        parser.error("Campaign must be 1-20 letters, numbers, hyphens or underscores")
    if not 0 <= args.cloud_percent <= 100:
        parser.error("Cloud percent must be 0-100")
    cases = read_dates(args.workbook)
    plan = dict(campaign=args.campaign, output_dir=str(args.output_dir.resolve()),
                cloud_percent=args.cloud_percent, remove_unnecessary=args.remove_unnecessary,
                starfm_executable=args.starfm_executable, tasks=make_tasks(cases, args.campaign, args.mode))
    directory = args.output_dir.resolve() / "OUTPUTS" / "DATE_PLANS" / args.campaign
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    plan_path = directory / "plan.json"
    # Prevent two submitters from duplicating the same campaign.
    with file_lock(directory / "submit.lock", timeout=0):
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("This campaign already has a different plan. Use a new --campaign name.")
        save_json(plan_path, plan)
        print(f"{len(cases)} date/region cases; {len(plan['tasks'])} single-configuration jobs")
        print(dict(Counter(t["family"] for t in plan["tasks"])))
        print(f"Plan: {plan_path}")
        if args.max_concurrent is not None:
            print(f"Array mode: up to {args.max_concurrent} concurrent band-level tasks; 4 hours per task.")
        if args.submit:
            if args.max_concurrent is not None:
                submit_array(plan_path, args.max_concurrent, args.retry_failed, args.only_tasks)
            else:
                if (directory / 'array_submissions.json').exists():
                    raise RuntimeError('Campaign already submitted as an array')
                submit(plan_path)
        else:
            print("Preview only. Add --submit to send jobs to Slurm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
