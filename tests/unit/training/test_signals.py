import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest

from mahjong.training.signals import SignalStopToken, signal_handlers


def test_handler_context_restores_previous_handlers():
    original = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    token = SignalStopToken()
    with signal_handlers(token):
        assert all(signal.getsignal(number) != old for number, old in original.items())
    assert all(signal.getsignal(number) == old for number, old in original.items())


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_real_first_signal_sets_stop_and_second_forces_exit(number):
    program = '''
import signal
from mahjong.training.signals import SignalStopToken, signal_handlers
stop = SignalStopToken()
with signal_handlers(stop):
    print('READY', flush=True)
    while not stop.requested():
        signal.pause()
    print('STOPPED:' + str(stop.signal_number()), flush=True)
    while True:
        signal.pause()
'''
    process = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline().strip() == "READY"
        os.kill(process.pid, number)
        assert select.select([process.stdout], [], [], 10)[0]
        assert process.stdout.readline().strip() == f"STOPPED:{int(number)}"
        os.kill(process.pid, number)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + int(number)
        assert "KeyboardInterrupt" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
