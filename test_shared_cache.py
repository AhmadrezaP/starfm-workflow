import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
import json
import os
import subprocess
from unittest.mock import patch
from datetime import datetime, timezone

from shared_cache import cached_download, cached_product, cache_key, file_lock, recover_lock, slurm_owner_finished


def valid_file(path):
    return path.read_bytes() == b'complete'


def download_worker(root, start):
    root = Path(root)
    start.wait()
    def download(staged):
        with (root / 'download_calls').open('a') as stream:
            stream.write('call\n')
        staged.write_bytes(b'partial')
        time.sleep(.1)
        staged.write_bytes(b'complete')
    cached_download(root / 'image.tif', valid_file, download)


def product_worker(root, start):
    root = Path(root)
    start.wait()
    def build(directory):
        with (root / 'product_calls').open('a') as stream:
            stream.write('call\n')
        path = directory / 'mosaic.tif'
        path.write_bytes(b'partial')
        time.sleep(.1)
        path.write_bytes(b'complete')
        return path
    def validate(path):
        if not valid_file(path):
            raise ValueError('bad product')
    cached_product(root / 'cache', 'key', build, validate)


class SharedCacheTests(unittest.TestCase):
    def test_slurm_confirmation_requires_terminal_matching_record(self):
        owner = dict(job_id='123', uid=os.getuid(), created=1700000050,
                     array_job_id='100', array_task_id='2')
        stamp = lambda n: datetime.fromtimestamp(n, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        good = f'123|TIMEOUT|{stamp(1700000000)}|{stamp(1700000100)}\n'
        for queue, accounting, expected in [
            ('', good, True), ('100_2\n', good, False), ('123\n', good, False),
            ('', '', False), ('', good + good, False),
            ('', good.replace('TIMEOUT', 'RUNNING'), False),
            ('', good + '123.batch|RUNNING|Unknown|Unknown\n', False),
            ('', good.replace(stamp(1700000100), stamp(1700000040)), False),
            ('', good.replace(stamp(1700000000), stamp(1700000060)), False),
        ]:
            with self.subTest(queue=queue, accounting=accounting):
                outputs = [subprocess.CompletedProcess([], 0, queue, ''), subprocess.CompletedProcess([], 0, accounting, '')]
                with patch('shared_cache.subprocess.run', side_effect=outputs), patch('shared_cache.time.time', return_value=1700000200):
                    self.assertEqual(slurm_owner_finished(owner), expected)
        with patch('shared_cache.subprocess.run', side_effect=OSError('scheduler unavailable')):
            self.assertFalse(slurm_owner_finished(owner))
        self.assertFalse(slurm_owner_finished(dict(job_id='123', created=1700000050)))

    def test_abandoned_lock_recovery_and_reacquisition(self):
        with tempfile.TemporaryDirectory() as root:
            lock = Path(root) / 'item.lock'
            directory = Path(str(lock) + '.d')
            directory.mkdir()
            (directory / 'owner.json').write_text(json.dumps({'job_id': '123'}))
            with patch('shared_cache.slurm_owner_finished', return_value=False):
                with self.assertRaises(TimeoutError):
                    with file_lock(lock, timeout=0):
                        pass
            self.assertTrue(directory.exists())
            with patch('shared_cache.slurm_owner_finished', return_value=True):
                with file_lock(lock, timeout=0):
                    self.assertEqual(json.loads((directory / 'owner.json').read_text())['pid'], os.getpid())
            self.assertFalse(directory.exists())

    def test_unknown_owner_or_busy_recovery_guard_is_retained(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'item.lock.d'
            directory.mkdir()
            self.assertFalse(recover_lock(directory))
            (directory / 'owner.json').write_text('broken json')
            self.assertFalse(recover_lock(directory))
            directory.with_name(directory.name + '.recovery').mkdir()
            self.assertFalse(recover_lock(directory))
            self.assertTrue(directory.exists())

    def test_concurrent_download_and_product_generate_once(self):
        context = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as root:
            for worker, log in [(download_worker, 'download_calls'), (product_worker, 'product_calls')]:
                start = context.Event()
                processes = [context.Process(target=worker, args=(root, start)) for _ in range(4)]
                for p in processes:
                    p.start()
                start.set()
                for p in processes:
                    p.join(20)
                    if p.is_alive():
                        p.terminate()
                        p.join()
                    self.assertEqual(p.exitcode, 0)
                self.assertEqual((Path(root) / log).read_text(), 'call\n')

    def test_interrupted_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'image.tif'
            def bad(staged):
                staged.write_bytes(b'partial')
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):
                cached_download(path, valid_file, bad)
            self.assertFalse(path.exists())
            cached_download(path, valid_file, lambda p: p.write_bytes(b'complete'))
            self.assertTrue(valid_file(path))

    def test_failed_product_is_not_published(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            def build(directory):
                (directory / 'bad').write_text('partial')
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):
                cached_product(root, 'key', build, lambda p: None)
            self.assertFalse((root / 'key').exists())

    def test_cache_identity_changes_with_inputs_settings_code(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'input'
            code = Path(root) / 'code.py'
            source.write_text('input')
            code.write_text('version1')
            original = cache_key([source], {'band': 'B1'}, [code])
            self.assertNotEqual(original, cache_key([source], {'band': 'B2'}, [code]))
            code.write_text('version2')
            self.assertNotEqual(original, cache_key([source], {'band': 'B1'}, [code]))
            code.write_text('version1')
            source.write_text('different')
            self.assertNotEqual(original, cache_key([source], {'band': 'B1'}, [code]))


if __name__ == '__main__':
    unittest.main()
