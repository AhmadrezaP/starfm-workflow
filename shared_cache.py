"""Shared-cache primitives using atomic mkdir (works with Lustre localflock)."""
from contextlib import contextmanager
import os
import socket
import time
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import re
from datetime import datetime, timezone


def slurm_owner_finished(owner):
    """Require current-queue absence AND unambiguous terminal accounting.

    Never infer death from age, an empty query, or an unavailable scheduler.
    Start/end times tie the accounting record to this lock's creation time.
    """
    job = str(owner.get('job_id', ''))
    if not re.fullmatch(r'[0-9]+', job) or owner.get('uid') != os.getuid():
        return False
    try:
        created = float(owner['created'])
        env = dict(os.environ, TZ='UTC', SLURM_TIME_FORMAT='standard')
        queue = subprocess.run(['squeue', '--noheader', '--array', '--user', str(os.getuid()),
                                '--format=%i'], capture_output=True, text=True, check=True,
                               timeout=15, env=env)
        identities = {job}
        if owner.get('array_job_id') and owner.get('array_task_id'):
            identities.add(f"{owner['array_job_id']}_{owner['array_task_id']}")
        if identities.intersection(queue.stdout.split()):
            return False
        since = datetime.fromtimestamp(created - 86400, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        result = subprocess.run(['sacct', '-j', job, '--duplicates', '--starttime', since,
                                 '--noheader', '--parsable2', '--format=JobIDRaw,State,Start,End'],
                                capture_output=True, text=True, check=True, timeout=15, env=env)
        records = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
        parents = [row for row in records if row[0] == job]
        if len(parents) != 1:
            return False
        # NODE_FAIL can leave an unreachable node with live processes; do not
        # infer that its writers have stopped merely from scheduler state.
        terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'DEADLINE'}
        related = [row for row in records if row[0] == job or row[0].startswith(job + '.')]
        if any(len(row) != 4 or row[1].split()[0] not in terminal for row in related):
            return False
        parent = parents[0]
        start, end = [datetime.fromisoformat(v).replace(tzinfo=timezone.utc).timestamp() for v in parent[2:4]]
        return start <= created <= end and end <= time.time() - 30
    except (OSError, ValueError, KeyError, IndexError, OverflowError, subprocess.SubprocessError):
        return False


def recover_lock(directory):
    """Serialize reapers using mkdir; retain unknown/malformed owners untouched."""
    guard = directory.with_name(directory.name + '.recovery')
    try:
        guard.mkdir()
    except FileExistsError:
        return False
    try:
        record = directory / 'owner.json'
        try:
            original = record.read_bytes()
            owner = json.loads(original)
            if not isinstance(owner, dict) or not slurm_owner_finished(owner):
                return False
            # All recoverers use the guard. A confirmed dead owner cannot
            # normally release/reacquire; still check the exact record again.
            if record.read_bytes() != original or set(p.name for p in directory.iterdir()) != {'owner.json'}:
                return False
            record.unlink()
            directory.rmdir()
            print(f"Recovered abandoned lock from finished Slurm job {owner['job_id']}: {directory}", flush=True)
            return True
        except (OSError, ValueError, TypeError):
            return False
    finally:
        guard.rmdir()


@contextmanager
def file_lock(path, timeout=3600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.with_name(path.name + '.d')
    deadline = time.monotonic() + timeout
    next_check = 0.0
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + 60
                if recover_lock(directory):
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Lock still held: {directory}. Inspect owner.json; never remove it while the owning job is active.')
            time.sleep(.2)
    owner = directory / 'owner.json'
    try:
        owner.write_text(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(),
                                        uid=os.getuid(), job_id=os.environ.get('SLURM_JOB_ID'),
                                        array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                                        array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), created=time.time())))
        yield
    finally:
        owner.unlink(missing_ok=True)
        directory.rmdir()


def cached_download(destination, valid, download):
    destination = Path(destination)
    with file_lock(destination.with_name(destination.name + '.lock')):
        if destination.is_file() and valid(destination):
            return destination
        # Same filesystem, same suffix: validation sees the real file type.
        with tempfile.TemporaryDirectory(prefix='.download-', dir=destination.parent) as directory:
            staged = Path(directory) / destination.name
            download(staged)
            if not valid(staged):
                raise ValueError(f'Invalid downloaded asset: {destination.name}')
            staged.replace(destination)
    return destination


def cache_key(inputs, settings, code_files):
    identities = []
    for path in sorted(set(map(Path, inputs))):
        stat = path.stat()
        identities.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    code = [(str(p.name), hashlib.sha256(p.read_bytes()).hexdigest()) for p in map(Path, code_files)]
    payload = json.dumps([identities, settings, code], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def cached_product(root, key, build, validate):
    """Publish a complete directory once; readers never see partial products."""
    root = Path(root)
    destination = root / key
    with file_lock(root / (key + '.lock')):
        marker = destination / 'complete.json'
        if marker.is_file():
            name = json.loads(marker.read_text())['product']
            product = destination / name
            validate(product)
            return product
        if destination.exists():
            raise RuntimeError(f'Unexpected incomplete cache directory: {destination}')
        with tempfile.TemporaryDirectory(prefix='.nbar-', dir=root) as directory:
            staged = Path(directory) / 'product'
            staged.mkdir()
            product = build(staged)
            validate(product)
            relative = product.relative_to(staged)
            (staged / 'complete.json').write_text(json.dumps({'product': str(relative)}))
            staged.rename(destination)
        return destination / relative
