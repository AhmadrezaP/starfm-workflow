from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from run_nearest_pair_pipeline import _download


class DownloadCacheTests(unittest.TestCase):
    def response(self, body, length=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {'Content-Type': 'image/tiff', 'Content-Length': str(len(body) if length is None else length)}
        response.url = 'https://example.test/imagery'
        response.status_code = 200
        response.iter_content.side_effect = lambda **kwargs: iter([body])
        return response

    def setUp(self):
        sleep = patch('run_nearest_pair_pipeline.time.sleep')
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_valid_download_published_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'image.tif'
            data = bytes.fromhex('49492a00') + b'example'
            with patch('run_nearest_pair_pipeline.requests.get', return_value=self.response(data)) as request:
                self.assertEqual(_download('https://example.test/imagery', destination), destination)
                self.assertEqual(destination.read_bytes(), data)
                _download('https://example.test/imagery', destination)
                self.assertEqual(request.call_count, 1)

    def test_gateway_timeout_and_truncation_are_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'image.tif'
            data = bytes.fromhex('49492a00') + b'example'
            timeout = self.response(b'')
            timeout.status_code = 504
            responses = [timeout, self.response(data[:3], len(data)), self.response(data)]
            with patch('run_nearest_pair_pipeline.requests.get', side_effect=responses) as request:
                self.assertEqual(_download('https://example.test/imagery', destination), destination)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(destination.read_bytes(), data)

    def test_client_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            denied = self.response(b'')
            denied.status_code = 401
            denied.raise_for_status.side_effect = requests.HTTPError('401')
            with patch('run_nearest_pair_pipeline.requests.get', return_value=denied) as request:
                with self.assertRaises(requests.HTTPError):
                    _download('https://example.test/imagery', Path(directory) / 'image.tif')
            self.assertEqual(request.call_count, 1)

    def test_short_or_nonbinary_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'image.tif'
            for data, length in [(b'II*\x00short', 100), (b'<html>Error</html>', None)]:
                with self.subTest(data=data), patch('run_nearest_pair_pipeline.requests.get', return_value=self.response(data, length)):
                    with self.assertRaises(RuntimeError):
                        _download('https://example.test/imagery', destination)
                    self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
