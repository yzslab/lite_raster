import errno
import multiprocessing
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lite_raster import _toolchain


# Controlled stand-in for NVCC. A partial output exists during compilation.
_COMPILER = """
import os, sys, time
from pathlib import Path
root, mode, _, output = sys.argv[1:]
root = Path(root)
output = Path(output)
assert output.parent == Path(os.environ["TMPDIR"])
with (root / "invocations").open("a") as log:
    print(mode, file=log)
output.write_bytes(b"partial")
if mode == "fail":
    sys.exit(17)
if mode == "wait" or mode.startswith("race-"):
    (root / ("compiling-" + mode)).write_text(str(output.parent))
    for _ in range(1000):
        if (root / ("finish-" + mode)).exists():
            break
        time.sleep(.02)
    else:
        sys.exit(18)
if mode == "race-fail":
    sys.exit(17)
output.write_bytes(mode.encode() if mode.startswith("race-") else b"ready")
"""


class JitCacheTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.target = self.root / "raster.so"

    def command(self, mode):
        return [sys.executable, "-c", _COMPILER, str(self.root), mode]

    def worker(self, mode):
        process = multiprocessing.get_context("fork").Process(
            target=_toolchain._publish, args=(self.target, self.command(mode)))
        process.start()
        def stop():
            if process.is_alive():
                process.kill()
            process.join(5)
        self.addCleanup(stop)
        return process

    def wait_for_compiler(self, mode="wait"):
        marker = self.root / ("compiling-" + mode)
        for _ in range(500):
            if marker.is_file() and marker.read_text():
                return Path(marker.read_text())
            time.sleep(.02)
        self.fail("compiler did not start")

    def test_concurrent_cold_cache_and_independent_keys(self):
        first = self.worker("wait")
        self.wait_for_compiler()
        waiters = [self.worker("retry") for _ in range(5)]
        other = self.root / "projection.so"
        _toolchain._publish(other, self.command("independent"))
        self.assertEqual(other.read_bytes(), b"ready")
        self.assertFalse(self.target.exists())
        (self.root / "finish-wait").touch()
        for p in [first, *waiters]:
            p.join(10)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual((self.root / "invocations").read_text().splitlines(), ["wait", "independent"])
        self.assertEqual(self.target.read_bytes(), b"ready")

    def test_compiler_failure_leaves_variant_retryable(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _toolchain._publish(self.target, self.command("fail"))
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.glob(".build-*")))
        _toolchain._publish(self.target, self.command("retry"))
        self.assertEqual(self.target.read_bytes(), b"ready")

    def test_killed_owner_can_be_replaced(self):
        first = self.worker("wait")
        abandoned = self.wait_for_compiler()
        first.kill()
        first.join(5)
        self.assertEqual(first.exitcode, -signal.SIGKILL)
        # A killed owner can leave its marker; the next process can bypass it.
        with patch.object(_toolchain, "_LOCK_WAIT_SECONDS", 0):
            waiter = self.worker("retry")
        (self.root / "finish-wait").touch()
        waiter.join(10)
        self.assertEqual(waiter.exitcode, 0)
        self.assertEqual((self.root / "invocations").read_text().splitlines(), ["wait", "retry"])
        self.assertEqual(self.target.read_bytes(), b"ready")
        self.assertTrue(abandoned.is_dir())

    def check_concurrent_publish(self):
        modes = ("race-first", "race-second", "race-fail")
        workers = [self.worker(mode) for mode in modes]
        directories = [self.wait_for_compiler(mode) for mode in modes]
        self.assertEqual(len(set(directories)), len(workers))
        for directory in directories:
            self.assertEqual((directory / self.target.name).read_bytes(), b"partial")
        self.assertFalse(self.target.exists())
        # Publish the first result, then let the second build finish and the
        # third compiler fail. All three callers should use the first result.
        for mode, worker, directory in zip(modes, workers, directories):
            (self.root / ("finish-" + mode)).touch()
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
            self.assertEqual(self.target.read_bytes(), b"race-first")
            self.assertFalse(directory.exists())

    def test_marker_creation_denied(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError(errno.EACCES, "cannot create marker")):
            self.check_concurrent_publish()

    def test_lock_reports_success_on_each_client(self):
        with patch.object(Path, "mkdir", return_value=None):
            self.check_concurrent_publish()

    def test_busy_lock_times_out(self):
        marker = self.target.with_suffix(".building")
        marker.mkdir()
        with patch.object(_toolchain, "_LOCK_WAIT_SECONDS", 0):
            _toolchain._publish(self.target, self.command("independent"))
        self.assertEqual(self.target.read_bytes(), b"ready")
        self.assertTrue(marker.is_dir())

    def test_marker_cleanup_failure_still_serves_published_result(self):
        with patch.object(Path, "rmdir", side_effect=PermissionError):
            _toolchain._publish(self.target, self.command("independent"))
        with patch.object(Path, "mkdir", side_effect=AssertionError("warm cache tried to lock")):
            _toolchain._publish(self.target, self.command("unexpected"))
        self.assertEqual((self.root / "invocations").read_text().splitlines(), ["independent"])

    def test_filesystem_without_hard_links(self):
        with patch.object(_toolchain.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "no hard links")):
            _toolchain._publish(self.target, self.command("independent"))
        self.assertEqual(self.target.read_bytes(), b"ready")
