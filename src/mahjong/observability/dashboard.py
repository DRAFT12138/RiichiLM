"""Start or reuse a detached local TensorBoard service for an output root."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

import psutil
from tensorboard import manager


def _ready(info, root):
    """Require a live process and its actual HTTP logdir, not a stale info file."""
    try:
        process = psutil.Process(info.pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        cwd = Path(process.cwd())
        if not info.logdir or (cwd / info.logdir).resolve() != root:
            return False
        url = f'http://127.0.0.1:{info.port}{info.path_prefix}'
        # Local monitoring must not be routed through HTTP_PROXY.
        with build_opener(ProxyHandler({})).open(url + '/data/logdir', timeout=0.5) as response:
            served = json.load(response)['logdir']
        return (cwd / served).resolve() == root
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        return False


def ensure_dashboard(root, *, timeout=15):
    """Start one detached server per output root; TensorBoard finds a free port."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.tensorboard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for info in manager.get_all():
            if _ready(info, root):
                return f'http://127.0.0.1:{info.port}{info.path_prefix}/#scalars', True
        with (root / '.tensorboard.log').open('w') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'tensorboard.main', '--logdir', str(root),
                 '--host', '127.0.0.1', '--port', 'default', '--reload_interval', '2',
                 '--load_fast=false'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'TensorBoard exited {process.returncode}; see {root / ".tensorboard.log"}')
                for info in manager.get_all():
                    if info.pid == process.pid and _ready(info, root):
                        return f'http://127.0.0.1:{info.port}{info.path_prefix}/#scalars', False
                time.sleep(0.1)
            raise RuntimeError(f'TensorBoard startup timed out; see {root / ".tensorboard.log"}')
        except BaseException:
            # Only clean up the child started here; never stop a reused server.
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
