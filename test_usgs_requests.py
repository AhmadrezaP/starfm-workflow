from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from run_nearest_pair_pipeline import USGS_STAC_SEARCH, REQUEST_RETRY_ATTEMPTS, _request_json, _usgs_api, _usgs_m2m_token


def reply(status, body=None):
    response = MagicMock(status_code=status)
    response.json.return_value = body if body is not None else {}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} error")
    return response


class UsgsRetryTests(unittest.TestCase):
    def setUp(self):
        sleep = patch("run_nearest_pair_pipeline.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def post(self, *responses):
        return patch("run_nearest_pair_pipeline.requests.post", side_effect=list(responses))

    def test_server_error_is_retried(self):
        with self.post(reply(500), reply(200, {"data": "key"})) as post:
            self.assertEqual(_usgs_m2m_token(None, "user", "token"), "key")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(self.sleep.call_count, 1)

    def test_dropped_connection_is_retried(self):
        with self.post(requests.ConnectionError("reset"), reply(200, {"data": []})) as post:
            _usgs_api("scene-search", token="key", body={})
        self.assertEqual(post.call_count, 2)

    def test_persistent_server_error_stops_after_limit(self):
        with self.post(*[reply(503)] * REQUEST_RETRY_ATTEMPTS) as post:
            with self.assertRaises(requests.HTTPError):
                _usgs_m2m_token(None, "user", "token")
        self.assertEqual(post.call_count, REQUEST_RETRY_ATTEMPTS)

    def test_client_errors_and_rejected_credentials_are_not_retried(self):
        with self.post(reply(401)) as post:
            with self.assertRaises(requests.HTTPError):
                _usgs_m2m_token(None, "user", "token")
        self.assertEqual(post.call_count, 1)
        rejected = reply(200, {"errorCode": "AUTH_INVALID", "errorMessage": "bad token"})
        with self.post(rejected) as post:
            with self.assertRaisesRegex(RuntimeError, "AUTH_INVALID"):
                _usgs_m2m_token(None, "user", "token")
        self.assertEqual(post.call_count, 1)
        self.sleep.assert_not_called()

    def test_login_holds_shared_lock(self):
        with tempfile.TemporaryDirectory() as root:
            lock = Path(root) / ".m2m_login.lock.d"
            seen = []

            def login(*args, **kwargs):
                seen.append(lock.is_dir())
                return reply(200, {"data": "key"})

            with patch("run_nearest_pair_pipeline.requests.post", side_effect=login):
                self.assertEqual(_usgs_m2m_token(None, "user", "token", lock_directory=Path(root)), "key")
            self.assertEqual(seen, [True])
            self.assertFalse(lock.exists())

    def test_stac_search_bad_gateway_is_retried(self):
        with self.post(reply(502), reply(200, {"features": []})) as post:
            self.assertEqual(_request_json(USGS_STAC_SEARCH, payload={"limit": 1}), {"features": []})
        self.assertEqual(post.call_count, 2)

    def test_cmr_search_retries_get_and_keeps_client_errors(self):
        with patch("run_nearest_pair_pipeline.requests.get", side_effect=[requests.Timeout("slow"), reply(200, {"feed": {}})]) as get:
            self.assertEqual(_request_json("https://cmr.example.test/search", params={"page_size": 1}), {"feed": {}})
        self.assertEqual(get.call_count, 2)
        with patch("run_nearest_pair_pipeline.requests.get", side_effect=[reply(404)]) as get:
            with self.assertRaises(requests.HTTPError):
                _request_json("https://cmr.example.test/search", params={})
        self.assertEqual(get.call_count, 1)

    def test_login_rate_limit_waits_and_retries(self):
        limited = reply(200, {"errorCode": "RATE_LIMIT_USER_AUTH", "errorMessage": "maximum authentication rate"})
        with self.post(limited, reply(200, {"data": "key"})) as post:
            self.assertEqual(_usgs_m2m_token(None, "user", "token"), "key")
        self.assertEqual(post.call_count, 2)
        self.assertGreaterEqual(self.sleep.call_args_list[0].args[0], 60)

    def test_persistent_login_rate_limit_is_reported(self):
        limited = reply(200, {"errorCode": "RATE_LIMIT_USER_AUTH", "errorMessage": "maximum authentication rate"})
        with self.post(*[limited] * REQUEST_RETRY_ATTEMPTS) as post:
            with self.assertRaisesRegex(RuntimeError, "RATE_LIMIT_USER_AUTH"):
                _usgs_m2m_token(None, "user", "token")
        self.assertEqual(post.call_count, REQUEST_RETRY_ATTEMPTS)

    def test_existing_api_key_skips_login(self):
        with self.post() as post:
            self.assertEqual(_usgs_m2m_token("existing", None, None), "existing")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
