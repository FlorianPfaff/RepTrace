from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_nodeeg_data.py"
SPEC = importlib.util.spec_from_file_location("download_nodeeg_data", SCRIPT_PATH)
download_nodeeg_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(download_nodeeg_data)

_TEST_ENV = {
    "NODEEG_WEBDAV_URL": "https://example.test/public.php/webdav/",
    "NODEEG_DATA_KEY": "key",
    "NODEEG_DATA_PASSWORD": "test-secret",  # nosec B105
}


def _completed(stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


class _FakeRclone:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args, **_kwargs):
        self.calls.append(args)
        if args[1] == "obscure":
            return _completed("obscured-password\n")
        if args[1] == "copy":
            data_root = Path(args[3])
            data_root.mkdir(parents=True, exist_ok=True)
            (data_root / "sub-01_epo.fif").write_bytes(b"epochs")
            (data_root / "sub-01_events.csv").write_text("trial,label\n", encoding="utf-8")
            return _completed()
        raise AssertionError(f"unexpected rclone command: {args}")


class DownloadNodeegDataTests(unittest.TestCase):
    def test_download_uses_webdav_options_and_validates_staged_subjects(self) -> None:
        fake_run = _FakeRclone()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, _TEST_ENV, clear=False), patch.object(download_nodeeg_data.subprocess, "run", side_effect=fake_run):
                result = download_nodeeg_data.main(
                    [
                        "--data-root",
                        str(Path(tmp_dir) / "nod"),
                        "--remote-path",
                        "staged/nod",
                        "--require-subject-count",
                        "1",
                    ]
                )

        self.assertEqual(result, 0)
        copy_call = next(call for call in fake_run.calls if call[1] == "copy")
        self.assertEqual(copy_call[2], ":webdav:staged/nod")
        self.assertIn("--webdav-vendor", copy_call)
        self.assertIn("owncloud", copy_call)
        self.assertIn("--include", copy_call)
        self.assertIn("sub-*_epo.fif", copy_call)
        self.assertIn("sub-*_events.csv", copy_call)

    def test_missing_webdav_secret_fails_before_rclone(self) -> None:
        fake_run = _FakeRclone()
        env_without_password = {key: value for key, value in _TEST_ENV.items() if key != "NODEEG_DATA_PASSWORD"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, env_without_password, clear=True), patch.object(download_nodeeg_data.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit):
                    download_nodeeg_data.main(["--data-root", str(Path(tmp_dir) / "nod"), "--require-subject-count", "1"])

        self.assertEqual(fake_run.calls, [])


if __name__ == "__main__":
    unittest.main()
